#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/PathControl.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/multirobot/base/ProblemDefinition.h>
#include <ompl/multirobot/control/PlanControl.h>
#include <ompl/multirobot/control/SpaceInformation.h>
#include <ompl/multirobot/control/planners/kcbs/KCBS.h>
#include <ompl/util/RandomNumbers.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace ob = ompl::base;
namespace oc = ompl::control;
namespace omrb = ompl::multirobot::base;
namespace omrc = ompl::multirobot::control;

namespace
{
std::vector<double> vector_from_array(const py::array_t<double, py::array::c_style | py::array::forcecast> &array,
                                      const std::string &name)
{
    const auto info = array.request();
    if (info.ndim != 1)
        throw std::invalid_argument(name + " must be a 1D array.");
    const auto *data = static_cast<const double *>(info.ptr);
    return std::vector<double>(data, data + info.shape[0]);
}

std::vector<std::vector<double>>
matrix_from_array(const py::array_t<double, py::array::c_style | py::array::forcecast> &array,
                  const std::string &name)
{
    const auto info = array.request();
    if (info.ndim != 2)
        throw std::invalid_argument(name + " must be a 2D array.");
    const auto rows = static_cast<std::size_t>(info.shape[0]);
    const auto cols = static_cast<std::size_t>(info.shape[1]);
    const auto *data = static_cast<const double *>(info.ptr);
    std::vector<std::vector<double>> values(rows, std::vector<double>(cols));
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t col = 0; col < cols; ++col)
            values[row][col] = data[row * cols + col];
    return values;
}

double option_double(const py::dict &options, const char *key, double default_value)
{
    if (!options.contains(key) || options[key].is_none())
        return default_value;
    return py::cast<double>(options[key]);
}

int option_int(const py::dict &options, const char *key, int default_value)
{
    if (!options.contains(key) || options[key].is_none())
        return default_value;
    return py::cast<int>(options[key]);
}

bool option_bool(const py::dict &options, const char *key, bool default_value)
{
    if (!options.contains(key) || options[key].is_none())
        return default_value;
    return py::cast<bool>(options[key]);
}

std::vector<double> state_values(const ob::State *state, unsigned int dim)
{
    const auto *values = state->as<ob::RealVectorStateSpace::StateType>()->values;
    return std::vector<double>(values, values + dim);
}

void assign_state(ob::State *state, const std::vector<double> &pos)
{
    auto *values = state->as<ob::RealVectorStateSpace::StateType>()->values;
    for (std::size_t idx = 0; idx < pos.size(); ++idx)
        values[idx] = pos[idx];
}

struct Point2
{
    double x{0.0};
    double y{0.0};
};

struct Halfspace2
{
    double nx{0.0};
    double ny{0.0};
    double bound{0.0};
};

double cross(const Point2 &origin, const Point2 &a, const Point2 &b)
{
    return (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
}

std::vector<Point2> convex_hull(std::vector<Point2> points)
{
    constexpr double eps = 1e-12;
    std::sort(points.begin(), points.end(), [](const Point2 &lhs, const Point2 &rhs) {
        if (lhs.x == rhs.x)
            return lhs.y < rhs.y;
        return lhs.x < rhs.x;
    });
    points.erase(std::unique(points.begin(), points.end(), [](const Point2 &lhs, const Point2 &rhs) {
                     return std::abs(lhs.x - rhs.x) <= eps && std::abs(lhs.y - rhs.y) <= eps;
                 }),
                 points.end());
    if (points.size() <= 1)
        return points;

    std::vector<Point2> hull;
    hull.reserve(points.size() * 2);
    for (const auto &point : points)
    {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), point) <= eps)
            hull.pop_back();
        hull.push_back(point);
    }
    const auto lower_size = hull.size();
    for (auto iter = points.rbegin() + 1; iter != points.rend(); ++iter)
    {
        while (hull.size() > lower_size && cross(hull[hull.size() - 2], hull.back(), *iter) <= eps)
            hull.pop_back();
        hull.push_back(*iter);
    }
    if (!hull.empty())
        hull.pop_back();
    return hull;
}

double signed_area(const std::vector<Point2> &polygon)
{
    double area = 0.0;
    for (std::size_t idx = 0; idx < polygon.size(); ++idx)
    {
        const auto &p = polygon[idx];
        const auto &q = polygon[(idx + 1) % polygon.size()];
        area += p.x * q.y - q.x * p.y;
    }
    return 0.5 * area;
}

struct ConvexPolygon2
{
    std::vector<Point2> vertices;
    std::vector<Halfspace2> halfspaces;

    static ConvexPolygon2 fromVertices(const std::vector<std::vector<double>> &matrix, const std::string &name)
    {
        std::vector<Point2> points;
        points.reserve(matrix.size());
        for (const auto &row : matrix)
        {
            if (row.size() != 2)
                throw std::invalid_argument(name + " must contain 2D vertices.");
            points.push_back({row[0], row[1]});
        }
        auto hull = convex_hull(std::move(points));
        if (hull.size() < 3)
            throw std::invalid_argument(name + " must span a nondegenerate convex polygon.");
        if (signed_area(hull) < 0.0)
            std::reverse(hull.begin(), hull.end());

        ConvexPolygon2 polygon;
        polygon.vertices = std::move(hull);
        polygon.halfspaces.reserve(polygon.vertices.size());
        for (std::size_t idx = 0; idx < polygon.vertices.size(); ++idx)
        {
            const auto &p = polygon.vertices[idx];
            const auto &q = polygon.vertices[(idx + 1) % polygon.vertices.size()];
            const double dx = q.x - p.x;
            const double dy = q.y - p.y;
            polygon.halfspaces.push_back({dy, -dx, dy * p.x - dx * p.y});
        }
        return polygon;
    }

    bool contains(const Point2 &point, double tol = 1e-9) const
    {
        for (const auto &halfspace : halfspaces)
        {
            if (halfspace.nx * point.x + halfspace.ny * point.y > halfspace.bound + tol)
                return false;
        }
        return true;
    }
};

std::vector<std::vector<double>> matrix_from_python_object(const py::handle &item, const std::string &name)
{
    return matrix_from_array(py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(item), name);
}

std::vector<ConvexPolygon2> convex_polygons_from_payload(const py::dict &payload,
                                                         const char *key,
                                                         const std::string &name)
{
    std::vector<ConvexPolygon2> polygons;
    if (!payload.contains(key) || payload[key].is_none())
        return polygons;
    const py::list items = py::cast<py::list>(payload[key]);
    polygons.reserve(static_cast<std::size_t>(py::len(items)));
    for (py::ssize_t idx = 0; idx < py::len(items); ++idx)
    {
        polygons.push_back(ConvexPolygon2::fromVertices(
            matrix_from_python_object(items[idx], name + "[" + std::to_string(idx) + "]"),
            name + "[" + std::to_string(idx) + "]"));
    }
    return polygons;
}

std::pair<double, double> project_polygon(const ConvexPolygon2 &polygon, double axis_x, double axis_y)
{
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (const auto &point : polygon.vertices)
    {
        const double value = axis_x * point.x + axis_y * point.y;
        lo = std::min(lo, value);
        hi = std::max(hi, value);
    }
    return {lo, hi};
}

bool has_separating_axis(const ConvexPolygon2 &lhs, const ConvexPolygon2 &rhs)
{
    constexpr double eps = 1e-12;
    auto check_axes = [&](const ConvexPolygon2 &source) {
        for (std::size_t idx = 0; idx < source.vertices.size(); ++idx)
        {
            const auto &p = source.vertices[idx];
            const auto &q = source.vertices[(idx + 1) % source.vertices.size()];
            const double axis_x = -(q.y - p.y);
            const double axis_y = q.x - p.x;
            const auto [lhs_lo, lhs_hi] = project_polygon(lhs, axis_x, axis_y);
            const auto [rhs_lo, rhs_hi] = project_polygon(rhs, axis_x, axis_y);
            if (lhs_hi < rhs_lo - eps || rhs_hi < lhs_lo - eps)
                return true;
        }
        return false;
    };
    return check_axes(lhs) || check_axes(rhs);
}

bool intersects(const ConvexPolygon2 &lhs, const ConvexPolygon2 &rhs)
{
    return !has_separating_axis(lhs, rhs);
}

ConvexPolygon2 robot_box(const std::vector<double> &point, double radius)
{
    const double x = point[0];
    const double y = point[1];
    return ConvexPolygon2::fromVertices(
        {
            {x - radius, y - radius},
            {x + radius, y - radius},
            {x + radius, y + radius},
            {x - radius, y + radius},
        },
        "robot footprint");
}

class NativeEnvironment2D
{
public:
    NativeEnvironment2D(std::vector<ConvexPolygon2> static_polygons, std::vector<ConvexPolygon2> cspace_polygons)
      : static_polygons_(std::move(static_polygons)), cspace_polygons_(std::move(cspace_polygons))
    {
    }

    bool isStateValid(const std::vector<double> &point, double robot_radius) const
    {
        const Point2 point2{point[0], point[1]};
        if (!static_polygons_.empty())
        {
            const auto footprint = robot_box(point, robot_radius);
            return std::none_of(static_polygons_.begin(), static_polygons_.end(), [&](const ConvexPolygon2 &obstacle) {
                return intersects(obstacle, footprint);
            });
        }
        if (!cspace_polygons_.empty())
        {
            return std::any_of(cspace_polygons_.begin(), cspace_polygons_.end(), [&](const ConvexPolygon2 &cell) {
                return cell.contains(point2);
            });
        }
        return true;
    }

private:
    std::vector<ConvexPolygon2> static_polygons_;
    std::vector<ConvexPolygon2> cspace_polygons_;
};

NativeEnvironment2D native_environment_from_payload(const py::dict &payload)
{
    return NativeEnvironment2D(
        convex_polygons_from_payload(payload, "static_polygons", "static_polygons"),
        convex_polygons_from_payload(payload, "cspace_polygons", "cspace_polygons"));
}

class DiskStateValidityChecker : public ob::StateValidityChecker
{
public:
    DiskStateValidityChecker(const ob::SpaceInformationPtr &si,
                             std::shared_ptr<const NativeEnvironment2D> native_environment,
                             unsigned int dim,
                             double robot_radius)
      : ob::StateValidityChecker(si)
      , native_environment_(std::move(native_environment))
      , dim_(dim)
      , robot_radius_(robot_radius)
      , robot_width_(2.0 * robot_radius)
    {
        if (!(std::isfinite(robot_radius) && robot_radius >= 0.0))
            throw std::invalid_argument("robot_radius must be finite and nonnegative.");
        if (!native_environment_)
            throw std::invalid_argument("native_environment must not be null.");
    }

    bool isValid(const ob::State *state) const override
    {
        // Control propagation can leave the sampling bounds. OMPL delegates
        // rejection of those propagated states to this validity checker.
        if (!si_->satisfiesBounds(state))
            return false;
        const auto point = state_values(state, dim_);
        return native_environment_->isStateValid(point, robot_radius_);
    }

    bool areStatesValid(const ob::State *state1,
                        const std::pair<const ob::SpaceInformationPtr, const ob::State *> state2) const override
    {
        if (state1 == nullptr || state2.second == nullptr)
            return false;
        const auto p = state_values(state1, dim_);
        const auto q = state_values(state2.second, dim_);
        double squared = 0.0;
        for (unsigned int idx = 0; idx < dim_; ++idx)
        {
            const double delta = p[idx] - q[idx];
            squared += delta * delta;
        }
        return std::sqrt(squared) > robot_width_;
    }

private:
    std::shared_ptr<const NativeEnvironment2D> native_environment_;
    unsigned int dim_;
    double robot_radius_;
    double robot_width_;
};

class PointGoalRegion : public ob::GoalSampleableRegion
{
public:
    PointGoalRegion(const ob::SpaceInformationPtr &si, std::vector<double> goal, double threshold)
      : ob::GoalSampleableRegion(si), goal_(std::move(goal))
    {
        threshold_ = threshold;
    }

    double distanceGoal(const ob::State *state) const override
    {
        const auto point = state_values(state, static_cast<unsigned int>(goal_.size()));
        double value = 0.0;
        for (std::size_t idx = 0; idx < goal_.size(); ++idx)
            value = std::max(value, std::abs(point[idx] - goal_[idx]));
        return value;
    }

    void sampleGoal(ob::State *state) const override
    {
        assign_state(state, goal_);
    }

    unsigned int maxSampleCount() const override
    {
        return 1;
    }

private:
    std::vector<double> goal_;
};

struct RootSolutionStatus
{
    unsigned int exact_solutions{0};
    unsigned int solution_paths{0};
    unsigned int num_robots{0};

    bool allExact() const
    {
        return num_robots > 0 && exact_solutions == num_robots;
    }
};

RootSolutionStatus root_solution_status(const std::vector<ob::ProblemDefinitionPtr> &problem_definitions)
{
    RootSolutionStatus status;
    status.num_robots = static_cast<unsigned int>(problem_definitions.size());
    for (const auto &pdef : problem_definitions)
    {
        if (pdef->hasExactSolution())
            status.exact_solutions += 1;
        if (pdef->getSolutionPath())
            status.solution_paths += 1;
    }
    return status;
}

std::vector<std::vector<double>> trajectory_from_path(const oc::PathControlPtr &path, unsigned int dim)
{
    std::vector<std::vector<double>> trajectory;
    if (!path || path->getStateCount() < 2)
        return trajectory;
    path->interpolate();
    const auto &durations = path->getControlDurations();
    double t0 = 0.0;
    for (std::size_t idx = 0; idx + 1 < path->getStateCount(); ++idx)
    {
        const double duration = idx < durations.size() ? durations[idx] : 0.0;
        const double t1 = t0 + duration;
        const auto p = state_values(path->getState(static_cast<unsigned int>(idx)), dim);
        const auto q = state_values(path->getState(static_cast<unsigned int>(idx + 1)), dim);
        std::vector<double> segment;
        segment.reserve(2 * dim + 2);
        segment.insert(segment.end(), p.begin(), p.end());
        segment.push_back(t0);
        segment.insert(segment.end(), q.begin(), q.end());
        segment.push_back(t1);
        trajectory.push_back(std::move(segment));
        t0 = t1;
    }
    return trajectory;
}

py::dict failure_payload(double runtime,
                         double root_solve_time = -1.0,
                         unsigned int nodes_expanded = 0,
                         unsigned int approx_solutions = 0,
                         RootSolutionStatus root_status = {})
{
    py::dict payload;
    payload["is_success"] = false;
    payload["cost"] = std::numeric_limits<double>::infinity();
    payload["time"] = runtime;
    payload["trajectories"] = std::vector<std::vector<std::vector<double>>>{};
    payload["num_nodes_expanded"] = nodes_expanded;
    payload["num_approximate_solutions"] = approx_solutions;
    payload["root_solve_time"] = root_solve_time;
    payload["root_exact_solutions"] = root_status.exact_solutions;
    payload["root_solution_paths"] = root_status.solution_paths;
    payload["root_all_exact"] = root_status.allExact();
    return payload;
}

py::dict solve_kcbs(py::array_t<double, py::array::c_style | py::array::forcecast> starts_array,
                    py::array_t<double, py::array::c_style | py::array::forcecast> goals_array,
                    py::array_t<double, py::array::c_style | py::array::forcecast> vlimits_array,
                    py::array_t<double, py::array::c_style | py::array::forcecast> lb_array,
                    py::array_t<double, py::array::c_style | py::array::forcecast> ub_array,
                    double robot_radius,
                    int seed,
                    py::dict options,
                    py::dict environment_payload)
{
    const auto starts = matrix_from_array(starts_array, "starts");
    const auto goals = matrix_from_array(goals_array, "goals");
    const auto vlimits = vector_from_array(vlimits_array, "vlimits");
    const auto lb = vector_from_array(lb_array, "lb");
    const auto ub = vector_from_array(ub_array, "ub");
    if (starts.empty())
        throw std::invalid_argument("K-CBS requires at least one robot.");
    const auto num_robots = starts.size();
    const auto dim = starts.front().size();
    if (dim != 2)
        throw std::invalid_argument("K-CBS adapter currently supports 2D MRMP instances.");
    if (goals.size() != num_robots || vlimits.size() != num_robots)
        throw std::invalid_argument("starts, goals, and vlimits must have the same robot count.");
    if (lb.size() != dim || ub.size() != dim)
        throw std::invalid_argument("lb and ub must match the state dimension.");
    for (std::size_t robot = 0; robot < num_robots; ++robot)
    {
        if (starts[robot].size() != dim || goals[robot].size() != dim)
            throw std::invalid_argument("all starts and goals must have the same dimension.");
        if (!(std::isfinite(vlimits[robot]) && vlimits[robot] > 0.0))
            throw std::invalid_argument("all vlimits must be finite and positive.");
    }

    const double budget = option_double(options, "max_runtime_in_secs", std::numeric_limits<double>::infinity());
    const double low_level_solve_time = option_double(options, "low_level_solve_time", 1.0);
    const double propagation_step_size = option_double(options, "propagation_step_size", 0.1);
    const int min_control_duration = option_int(options, "min_control_duration", 1);
    const int max_control_duration = option_int(options, "max_control_duration", 10);
    const double goal_tolerance = option_double(options, "goal_tolerance", 1e-3);
    const double goal_bias = option_double(options, "goal_bias", 0.05);
    const bool intermediate_states = option_bool(options, "intermediate_states", false);
    const int num_threads = option_int(options, "num_threads", 4);
    if (!(std::isfinite(budget) && budget > 0.0))
        throw std::invalid_argument("max_runtime_in_secs must be finite and positive.");
    if (!(std::isfinite(low_level_solve_time) && low_level_solve_time > 0.0))
        throw std::invalid_argument("low_level_solve_time must be finite and positive.");
    if (!(std::isfinite(propagation_step_size) && propagation_step_size > 0.0))
        throw std::invalid_argument("propagation_step_size must be finite and positive.");
    if (min_control_duration < 1 || max_control_duration < min_control_duration)
        throw std::invalid_argument("control duration bounds must satisfy 1 <= min <= max.");
    if (!(std::isfinite(goal_tolerance) && goal_tolerance >= 0.0))
        throw std::invalid_argument("goal_tolerance must be finite and nonnegative.");

    ompl::RNG::setSeed(static_cast<std::uint_fast32_t>(seed == 0 ? 1 : seed));
    const auto native_environment = std::make_shared<const NativeEnvironment2D>(
        native_environment_from_payload(environment_payload));

    auto ma_si = std::make_shared<omrc::SpaceInformation>();
    auto ma_pdef = std::make_shared<omrb::ProblemDefinition>(ma_si);
    std::vector<ob::ProblemDefinitionPtr> individual_pdefs;
    individual_pdefs.reserve(num_robots);
    for (std::size_t robot = 0; robot < num_robots; ++robot)
    {
        auto space = std::make_shared<ob::RealVectorStateSpace>(static_cast<unsigned int>(dim));
        ob::RealVectorBounds bounds(static_cast<unsigned int>(dim));
        for (unsigned int axis = 0; axis < dim; ++axis)
        {
            bounds.setLow(axis, lb[axis]);
            bounds.setHigh(axis, ub[axis]);
        }
        space->setBounds(bounds);
        space->setName("Robot " + std::to_string(robot));

        auto cspace = std::make_shared<oc::RealVectorControlSpace>(space, static_cast<unsigned int>(dim));
        ob::RealVectorBounds cbounds(static_cast<unsigned int>(dim));
        for (unsigned int axis = 0; axis < dim; ++axis)
        {
            cbounds.setLow(axis, -vlimits[robot]);
            cbounds.setHigh(axis, vlimits[robot]);
        }
        cspace->setBounds(cbounds);

        auto si = std::make_shared<oc::SpaceInformation>(space, cspace);
        si->setStateValidityChecker(
            std::make_shared<DiskStateValidityChecker>(si, native_environment, static_cast<unsigned int>(dim), robot_radius));
        si->setStatePropagator([dim](const ob::State *start,
                                     const oc::Control *control,
                                     const double duration,
                                     ob::State *result) {
            const auto *x = start->as<ob::RealVectorStateSpace::StateType>()->values;
            const auto *u = control->as<oc::RealVectorControlSpace::ControlType>()->values;
            auto *y = result->as<ob::RealVectorStateSpace::StateType>()->values;
            for (std::size_t axis = 0; axis < dim; ++axis)
                y[axis] = x[axis] + u[axis] * duration;
        });
        si->setPropagationStepSize(propagation_step_size);
        si->setMinMaxControlDuration(
            static_cast<unsigned int>(min_control_duration),
            static_cast<unsigned int>(max_control_duration));

        ob::ScopedState<> start(space);
        assign_state(start.get(), starts[robot]);
        auto pdef = std::make_shared<ob::ProblemDefinition>(si);
        pdef->addStartState(start);
        pdef->setGoal(std::make_shared<PointGoalRegion>(si, goals[robot], goal_tolerance));

        ma_si->addIndividual(si);
        ma_pdef->addIndividual(pdef);
        individual_pdefs.push_back(pdef);
    }
    ma_si->lock();
    ma_pdef->lock();

    ma_si->setPlannerAllocator([goal_bias, intermediate_states](const ob::SpaceInformationPtr &si) -> ob::PlannerPtr {
        auto si_control = std::static_pointer_cast<oc::SpaceInformation>(si);
        auto planner = std::make_shared<oc::RRT>(si_control);
        planner->setGoalBias(goal_bias);
        planner->setIntermediateStates(intermediate_states);
        return planner;
    });

    auto planner = std::make_shared<omrc::KCBS>(ma_si);
    planner->setProblemDefinition(ma_pdef);
    planner->setLowLevelSolveTime(low_level_solve_time);
    planner->setNumThreads(static_cast<unsigned int>(std::max(1, num_threads)));
    if (options.contains("merge_bound") && !options["merge_bound"].is_none())
        planner->setMergeBound(static_cast<unsigned int>(py::cast<int>(options["merge_bound"])));

    const auto start_time = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    };

    ob::PlannerStatus status;
    {
        py::gil_scoped_release release;
        status = planner->solve(ob::timedPlannerTerminationCondition(budget));
    }
    const double runtime = elapsed();
    const auto root_status = root_solution_status(individual_pdefs);
    const double root_solve_time = planner->getRootSolveTime();
    const unsigned int nodes_expanded = planner->getNumberOfNodesExpanded();
    const unsigned int approx_solutions = planner->getNumberOfApproximateSolutions();
    if (!status)
        return failure_payload(runtime, root_solve_time, nodes_expanded, approx_solutions, root_status);
    if (!root_status.allExact())
        return failure_payload(runtime, root_solve_time, nodes_expanded, approx_solutions, root_status);

    const auto solution_plan = ma_pdef->getSolutionPlan();
    if (!solution_plan)
        return failure_payload(runtime, root_solve_time, nodes_expanded, approx_solutions, root_status);
    auto control_plan = solution_plan->as<omrc::PlanControl>();
    std::vector<std::vector<std::vector<double>>> trajectories;
    trajectories.reserve(num_robots);
    double total_cost = 0.0;
    for (std::size_t robot = 0; robot < num_robots; ++robot)
    {
        auto path = control_plan->getPath(static_cast<unsigned int>(robot));
        total_cost += path->length();
        auto trajectory = trajectory_from_path(path, static_cast<unsigned int>(dim));
        if (trajectory.empty())
            return failure_payload(runtime, root_solve_time, nodes_expanded, approx_solutions, root_status);
        trajectories.push_back(std::move(trajectory));
    }

    py::dict payload;
    payload["is_success"] = true;
    payload["cost"] = total_cost;
    payload["time"] = runtime;
    payload["trajectories"] = trajectories;
    payload["num_nodes_expanded"] = nodes_expanded;
    payload["num_approximate_solutions"] = approx_solutions;
    payload["root_solve_time"] = root_solve_time;
    payload["root_exact_solutions"] = root_status.exact_solutions;
    payload["root_solution_paths"] = root_status.solution_paths;
    payload["root_all_exact"] = root_status.allExact();
    return payload;
}
}  // namespace

PYBIND11_MODULE(_ompl_kcbs_native, m)
{
    m.doc() = "Official Multi-Robot-OMPL K-CBS adapter for stgcs Env-based MRMP instances.";
    m.def("solve_kcbs", &solve_kcbs, py::arg("starts"), py::arg("goals"), py::arg("vlimits"), py::arg("lb"),
          py::arg("ub"), py::arg("robot_radius"), py::arg("seed"), py::arg("options"),
          py::arg("environment_payload"));
}
