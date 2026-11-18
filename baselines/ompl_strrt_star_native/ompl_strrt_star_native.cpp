#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/objectives/MinimizeArrivalTime.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SpaceTimeStateSpace.h>
#include <ompl/base/spaces/TimeStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/rrt/STRRTstar.h>
#include <ompl/util/Exception.h>
#include <ompl/util/RandomNumbers.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace ob = ompl::base;
namespace og = ompl::geometric;

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

std::vector<std::vector<double>> matrix_from_array(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &array,
    const std::string &name)
{
    const auto info = array.request();
    if (info.ndim != 2)
        throw std::invalid_argument(name + " must be a 2D array.");
    const auto *data = static_cast<const double *>(info.ptr);
    std::vector<std::vector<double>> matrix(static_cast<std::size_t>(info.shape[0]),
                                            std::vector<double>(static_cast<std::size_t>(info.shape[1])));
    for (py::ssize_t row = 0; row < info.shape[0]; ++row)
    {
        for (py::ssize_t col = 0; col < info.shape[1]; ++col)
            matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                data[row * info.shape[1] + col];
    }
    return matrix;
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

ConvexPolygon2 robot_swept_box(const std::vector<double> &p, const std::vector<double> &q, double radius)
{
    const double px = p[0];
    const double py = p[1];
    const double qx = q[0];
    const double qy = q[1];
    return ConvexPolygon2::fromVertices(
        {
            {px - radius, py - radius},
            {px + radius, py - radius},
            {px + radius, py + radius},
            {px - radius, py + radius},
            {qx - radius, qy - radius},
            {qx + radius, qy - radius},
            {qx + radius, qy + radius},
            {qx - radius, qy + radius},
        },
        "robot swept footprint");
}

std::optional<std::pair<double, double>> segment_interval_in_polygon(const ConvexPolygon2 &polygon,
                                                                     const Point2 &p,
                                                                     const Point2 &q,
                                                                     double tol)
{
    double lo = 0.0;
    double hi = 1.0;
    const double dx = q.x - p.x;
    const double dy = q.y - p.y;
    for (const auto &halfspace : polygon.halfspaces)
    {
        const double slope = halfspace.nx * dx + halfspace.ny * dy;
        const double rhs = halfspace.bound + tol - halfspace.nx * p.x - halfspace.ny * p.y;
        if (std::abs(slope) <= tol)
        {
            if (rhs < 0.0)
                return std::nullopt;
            continue;
        }
        const double alpha = rhs / slope;
        if (slope > 0.0)
            hi = std::min(hi, alpha);
        else
            lo = std::max(lo, alpha);
        if (lo > hi + tol)
            return std::nullopt;
    }
    if (hi < -tol || lo > 1.0 + tol)
        return std::nullopt;
    lo = std::max(0.0, lo);
    hi = std::min(1.0, hi);
    if (lo > hi + tol)
        return std::nullopt;
    return std::make_pair(lo, hi);
}

bool intervals_cover_unit(std::vector<std::pair<double, double>> intervals, double tol)
{
    std::sort(intervals.begin(), intervals.end());
    double covered_until = 0.0;
    for (const auto &[lo, hi] : intervals)
    {
        if (hi < covered_until - tol)
            continue;
        if (lo > covered_until + tol)
            return false;
        covered_until = std::max(covered_until, hi);
        if (covered_until >= 1.0 - tol)
            return true;
    }
    return covered_until >= 1.0 - tol;
}

class NativeEnvironment2D
{
public:
    NativeEnvironment2D(std::vector<ConvexPolygon2> static_polygons, std::vector<ConvexPolygon2> cspace_polygons)
      : static_polygons_(std::move(static_polygons)), cspace_polygons_(std::move(cspace_polygons))
    {
    }

    bool hasGeometry() const
    {
        return !static_polygons_.empty() || !cspace_polygons_.empty();
    }

    bool isStateValid(const std::vector<double> &point, double robot_radius) const
    {
        if (point.size() != 2)
            return !hasGeometry();
        const Point2 point2{point[0], point[1]};
        if (!static_polygons_.empty())
        {
            if (robot_radius <= 0.0)
            {
                return std::none_of(static_polygons_.begin(), static_polygons_.end(), [&](const ConvexPolygon2 &obstacle) {
                    return obstacle.contains(point2);
                });
            }
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

    bool isMotionValid(const std::vector<double> &p, const std::vector<double> &q, double robot_radius) const
    {
        if (p.size() != 2 || q.size() != 2)
            return !hasGeometry();
        if (!static_polygons_.empty())
        {
            const Point2 point_p{p[0], p[1]};
            const Point2 point_q{q[0], q[1]};
            if (robot_radius <= 0.0)
            {
                return std::none_of(static_polygons_.begin(), static_polygons_.end(), [&](const ConvexPolygon2 &obstacle) {
                    return segment_interval_in_polygon(obstacle, point_p, point_q, 1e-9).has_value();
                });
            }
            const auto swept = robot_swept_box(p, q, robot_radius);
            return std::none_of(static_polygons_.begin(), static_polygons_.end(), [&](const ConvexPolygon2 &obstacle) {
                return intersects(obstacle, swept);
            });
        }
        if (!cspace_polygons_.empty())
        {
            std::vector<std::pair<double, double>> intervals;
            const Point2 point_p{p[0], p[1]};
            const Point2 point_q{q[0], q[1]};
            for (const auto &cell : cspace_polygons_)
            {
                auto interval = segment_interval_in_polygon(cell, point_p, point_q, 1e-9);
                if (interval.has_value())
                    intervals.push_back(*interval);
            }
            return intervals_cover_unit(std::move(intervals), 1e-9);
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

class ScaledLInfRealVectorStateSpace : public ob::RealVectorStateSpace
{
public:
    ScaledLInfRealVectorStateSpace(unsigned int dim, std::vector<double> vlimit)
      : ob::RealVectorStateSpace(dim), vlimit_(std::move(vlimit))
    {
        if (vlimit_.size() != dim)
            throw std::invalid_argument("vlimit dimension must match the spatial state dimension.");
        for (double value : vlimit_)
        {
            if (!(std::isfinite(value) && value > 0.0))
                throw std::invalid_argument("All vlimit entries must be finite and positive.");
        }
        setName("ScaledLInfRealVector" + getName());
    }

    double distance(const ob::State *state1, const ob::State *state2) const override
    {
        const auto *x = state1->as<ob::RealVectorStateSpace::StateType>();
        const auto *y = state2->as<ob::RealVectorStateSpace::StateType>();
        double value = 0.0;
        for (unsigned int idx = 0; idx < dimension_; ++idx)
            value = std::max(value, std::abs(x->values[idx] - y->values[idx]) / vlimit_[idx]);
        return value;
    }

    double getMaximumExtent() const override
    {
        double value = 0.0;
        const auto &bounds = getBounds();
        for (unsigned int idx = 0; idx < dimension_; ++idx)
            value = std::max(value, std::abs(bounds.high[idx] - bounds.low[idx]) / vlimit_[idx]);
        return value;
    }

private:
    std::vector<double> vlimit_;
};

class SeededSpaceTimeStateSampler : public ob::StateSampler
{
public:
    SeededSpaceTimeStateSampler(const ob::StateSpace *space, std::uint_fast32_t seed)
      : ob::StateSampler(space)
      , space_time_(const_cast<ob::SpaceTimeStateSpace *>(space->as<ob::SpaceTimeStateSpace>()))
      , bounds_(space_time_->getSpaceComponent()->as<ob::RealVectorStateSpace>()->getDimension())
    {
        rng_.setLocalSeed(seed == 0 ? 1 : seed);
        const auto *spatial = space_time_->getSpaceComponent()->as<ob::RealVectorStateSpace>();
        bounds_ = spatial->getBounds();
    }

    void sampleUniform(ob::State *state) override
    {
        auto *compound = state->as<ob::CompoundState>();
        auto *spatial = compound->as<ob::RealVectorStateSpace::StateType>(0);
        for (unsigned int idx = 0; idx < bounds_.low.size(); ++idx)
            spatial->values[idx] = rng_.uniformReal(bounds_.low[idx], bounds_.high[idx]);
        sampleTime(compound->as<ob::TimeStateSpace::StateType>(1));
    }

    void sampleUniformNear(ob::State *state, const ob::State *near, double distance) override
    {
        auto *compound = state->as<ob::CompoundState>();
        auto *spatial = compound->as<ob::RealVectorStateSpace::StateType>(0);
        const auto *near_spatial = near->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0);
        for (unsigned int idx = 0; idx < bounds_.low.size(); ++idx)
        {
            const double low = std::max(bounds_.low[idx], near_spatial->values[idx] - distance);
            const double high = std::min(bounds_.high[idx], near_spatial->values[idx] + distance);
            spatial->values[idx] = rng_.uniformReal(low, high);
        }
        sampleTimeNear(
            compound->as<ob::TimeStateSpace::StateType>(1),
            near->as<ob::CompoundState>()->as<ob::TimeStateSpace::StateType>(1),
            distance);
    }

    void sampleGaussian(ob::State *state, const ob::State *mean, double std_dev) override
    {
        auto *compound = state->as<ob::CompoundState>();
        auto *spatial = compound->as<ob::RealVectorStateSpace::StateType>(0);
        const auto *mean_spatial = mean->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0);
        for (unsigned int idx = 0; idx < bounds_.low.size(); ++idx)
        {
            const double value = rng_.gaussian(mean_spatial->values[idx], std_dev);
            spatial->values[idx] = std::min(std::max(value, bounds_.low[idx]), bounds_.high[idx]);
        }
        sampleTimeGaussian(
            compound->as<ob::TimeStateSpace::StateType>(1),
            mean->as<ob::CompoundState>()->as<ob::TimeStateSpace::StateType>(1),
            std_dev);
    }

private:
    void sampleTime(ob::TimeStateSpace::StateType *time_state)
    {
        auto *time_space = space_time_->getTimeComponent();
        if (!time_space->isBounded())
        {
            time_state->position = 0.0;
            return;
        }
        time_state->position = rng_.uniformReal(time_space->getMinTimeBound(), time_space->getMaxTimeBound());
    }

    void sampleTimeNear(ob::TimeStateSpace::StateType *time_state,
                        const ob::TimeStateSpace::StateType *near_time,
                        double distance)
    {
        auto *time_space = space_time_->getTimeComponent();
        if (!time_space->isBounded())
        {
            time_state->position = 0.0;
            return;
        }
        const double low = std::max(time_space->getMinTimeBound(), near_time->position - distance);
        const double high = std::min(time_space->getMaxTimeBound(), near_time->position + distance);
        time_state->position = rng_.uniformReal(low, high);
    }

    void sampleTimeGaussian(ob::TimeStateSpace::StateType *time_state,
                            const ob::TimeStateSpace::StateType *mean_time,
                            double std_dev)
    {
        auto *time_space = space_time_->getTimeComponent();
        if (!time_space->isBounded())
        {
            time_state->position = 0.0;
            return;
        }
        const double value = rng_.gaussian(mean_time->position, std_dev);
        time_state->position = std::min(
            std::max(value, time_space->getMinTimeBound()),
            time_space->getMaxTimeBound());
    }

    ob::SpaceTimeStateSpace *space_time_;
    ob::RealVectorBounds bounds_;
};

std::vector<double> spatial_values(const ob::State *state, unsigned int dim)
{
    const auto *compound = state->as<ob::CompoundState>();
    const auto *spatial = compound->as<ob::RealVectorStateSpace::StateType>(0);
    std::vector<double> values(dim);
    for (unsigned int idx = 0; idx < dim; ++idx)
        values[idx] = spatial->values[idx];
    return values;
}

double state_time(const ob::State *state)
{
    return state->as<ob::CompoundState>()->as<ob::TimeStateSpace::StateType>(1)->position;
}

double scaled_spatial_goal_error(const ob::State *state,
                                 const std::vector<double> &goal,
                                 const std::vector<double> &vlimit)
{
    const auto pos = spatial_values(state, static_cast<unsigned int>(goal.size()));
    double value = 0.0;
    for (std::size_t idx = 0; idx < goal.size(); ++idx)
        value = std::max(value, std::abs(pos[idx] - goal[idx]) / vlimit[idx]);
    return value;
}

py::array_t<double> py_array_from_vector(const std::vector<double> &values)
{
    py::array_t<double> array(values.size());
    auto out = array.mutable_unchecked<1>();
    for (py::ssize_t idx = 0; idx < out.shape(0); ++idx)
        out(idx) = values[static_cast<std::size_t>(idx)];
    return array;
}

class EnvMotionValidator : public ob::MotionValidator
{
public:
    EnvMotionValidator(const ob::SpaceInformationPtr &si,
                       std::shared_ptr<const NativeEnvironment2D> native_environment,
                       py::object extra_collision_checker,
                       unsigned int dim,
                       double robot_radius)
      : ob::MotionValidator(si)
      , native_environment_(std::move(native_environment))
      , extra_collision_checker_(std::move(extra_collision_checker))
      , state_space_(si_->getStateSpace().get())
      , dim_(dim)
      , robot_radius_(robot_radius)
    {
    }

    bool checkMotion(const ob::State *s1, const ob::State *s2) const override
    {
        if (!si_->isValid(s2))
        {
            invalid_++;
            return false;
        }

        auto *space = state_space_->as<ob::SpaceTimeStateSpace>();
        const double delta_space = space->distanceSpace(s1, s2);
        const double delta_time = state_time(s2) - state_time(s1);
        if (!(delta_time > 0.0 && delta_space <= delta_time + 1e-9))
        {
            invalid_++;
            return false;
        }

        const auto p = spatial_values(s1, dim_);
        const auto q = spatial_values(s2, dim_);
        if (!native_environment_->isMotionValid(p, q, robot_radius_))
        {
            invalid_++;
            return false;
        }
        if (!extra_collision_checker_.is_none())
        {
            py::gil_scoped_acquire gil;
            const bool collides = py::cast<bool>(extra_collision_checker_(
                py_array_from_vector(p),
                py_array_from_vector(q),
                state_time(s1),
                state_time(s2)));
            if (collides)
            {
                invalid_++;
                return false;
            }
        }
        return true;
    }

    bool checkMotion(const ob::State *, const ob::State *, std::pair<ob::State *, double> &) const override
    {
        throw ompl::Exception("EnvMotionValidator::checkMotion", "last-valid variant is not implemented");
    }

private:
    std::shared_ptr<const NativeEnvironment2D> native_environment_;
    py::object extra_collision_checker_;
    ob::StateSpace *state_space_;
    unsigned int dim_;
    double robot_radius_;
};

bool env_state_valid(const std::shared_ptr<const NativeEnvironment2D> &native_environment,
                     const py::object &extra_collision_checker,
                     const ob::State *state,
                     unsigned int dim,
                     double t0,
                     double robot_radius)
{
    const double t = state_time(state);
    if (!(std::isfinite(t) && t >= t0 - 1e-9))
        return false;
    const auto p = spatial_values(state, dim);
    if (!native_environment->isStateValid(p, robot_radius))
        return false;
    if (!extra_collision_checker.is_none())
    {
        py::gil_scoped_acquire gil;
        return !py::cast<bool>(extra_collision_checker(py_array_from_vector(p), py_array_from_vector(p), t, t));
    }
    return true;
}

void assign_compound_state(ob::State *state, const std::vector<double> &pos, double time)
{
    auto *compound = state->as<ob::CompoundState>();
    auto *spatial = compound->as<ob::RealVectorStateSpace::StateType>(0);
    for (std::size_t idx = 0; idx < pos.size(); ++idx)
        spatial->values[idx] = pos[idx];
    compound->as<ob::TimeStateSpace::StateType>(1)->position = time;
}

std::vector<std::vector<double>> trajectory_from_states(const std::vector<const ob::State *> &states, unsigned int dim)
{
    std::vector<std::vector<double>> trajectory;
    if (states.size() < 2)
        return trajectory;

    for (std::size_t idx = 1; idx < states.size(); ++idx)
    {
        std::vector<double> segment;
        segment.reserve(2 * (dim + 1));
        const auto from_pos = spatial_values(states[idx - 1], dim);
        const auto to_pos = spatial_values(states[idx], dim);
        segment.insert(segment.end(), from_pos.begin(), from_pos.end());
        segment.push_back(state_time(states[idx - 1]));
        segment.insert(segment.end(), to_pos.begin(), to_pos.end());
        segment.push_back(state_time(states[idx]));
        trajectory.push_back(std::move(segment));
    }
    return trajectory;
}

py::dict failure_payload(double runtime)
{
    py::dict payload;
    payload["is_success"] = false;
    payload["cost"] = std::numeric_limits<double>::infinity();
    payload["time"] = runtime;
    payload["trajectory"] = std::vector<std::vector<double>>{};
    return payload;
}

py::dict solution_payload(const std::vector<const ob::State *> &states,
                          unsigned int dim,
                          const std::vector<double> &goal,
                          const std::vector<double> &vlimit,
                          double goal_tolerance,
                          double runtime)
{
    if (states.size() < 2)
        return failure_payload(runtime);
    if (scaled_spatial_goal_error(states.back(), goal, vlimit) > goal_tolerance)
        return failure_payload(runtime);
    py::dict payload;
    payload["is_success"] = true;
    payload["cost"] = state_time(states.back()) - state_time(states.front());
    payload["time"] = runtime;
    payload["trajectory"] = trajectory_from_states(states, dim);
    return payload;
}

struct SolutionSnapshot
{
    bool is_success{false};
    double cost{std::numeric_limits<double>::infinity()};
    double runtime{-1.0};
    std::vector<std::vector<double>> trajectory{};
};

SolutionSnapshot solution_snapshot(const std::vector<const ob::State *> &states,
                                   unsigned int dim,
                                   const std::vector<double> &goal,
                                   const std::vector<double> &vlimit,
                                   double goal_tolerance,
                                   double runtime)
{
    SolutionSnapshot snapshot;
    snapshot.runtime = runtime;
    if (states.size() < 2)
        return snapshot;
    if (scaled_spatial_goal_error(states.back(), goal, vlimit) > goal_tolerance)
        return snapshot;
    snapshot.is_success = true;
    snapshot.cost = state_time(states.back()) - state_time(states.front());
    snapshot.trajectory = trajectory_from_states(states, dim);
    return snapshot;
}

py::dict solution_payload(const SolutionSnapshot &snapshot)
{
    if (!snapshot.is_success)
        return failure_payload(snapshot.runtime);
    py::dict payload;
    payload["is_success"] = true;
    payload["cost"] = snapshot.cost;
    payload["time"] = snapshot.runtime;
    payload["trajectory"] = snapshot.trajectory;
    return payload;
}

class PatchedSTRRTstar : public og::STRRTstar
{
public:
    explicit PatchedSTRRTstar(const ob::SpaceInformationPtr &si) : og::STRRTstar(si) {}

    void setLocalSeed(std::uint_fast32_t seed)
    {
        rng_.setLocalSeed(seed == 0 ? 1 : seed);
    }

    void setGoalValidityChecker(std::function<bool(const ob::State *)> checker)
    {
        goalValidityChecker_ = std::move(checker);
    }

    void setMaxTimeBoundFactor(double factor)
    {
        if (!(std::isfinite(factor) && factor > 0.0))
            throw std::invalid_argument("max_time_bound_factor must be finite and positive.");
        maxTimeBoundFactor_ = factor;
    }

    ob::PlannerStatus solve(const ob::PlannerTerminationCondition &ptc) override
    {
        checkValidity();
        auto *goal_base = pdef_->getGoal().get();
        if (goal_base == nullptr || !goal_base->hasType(ob::GOAL_SAMPLEABLE_REGION))
        {
            OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
            return ob::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
        }
        auto *goal = goal_base->as<ob::GoalSampleableRegion>();

        while (const ob::State *st = pis_.nextStart())
        {
            auto *motion = new Motion(si_);
            si_->copyState(motion->state, st);
            motion->root = motion->state;
            tStart_->add(motion);
            startMotion_ = motion;
        }

        if (tStart_->size() == 0)
        {
            OMPL_ERROR("%s: Motion planning start tree could not be initialized!", getName().c_str());
            return ob::PlannerStatus::INVALID_START;
        }
        if (!goal->couldSample())
        {
            OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
            return ob::PlannerStatus::INVALID_GOAL;
        }

        OMPL_INFORM("%s: Starting planning with %d states already in datastructure", getName().c_str(),
                    static_cast<int>(tStart_->size() + tGoal_->size()));

        TreeGrowingInfo tgi;
        tgi.xstate = si_->allocState();

        std::vector<Motion *> nbh;
        const ob::ReportIntermediateSolutionFn intermediate_solution_callback =
            pdef_->getIntermediateSolutionCallback();

        Motion *approxsol = nullptr;
        double approxdif = std::numeric_limits<double>::infinity();
        auto *rmotion = new Motion(si_);
        ob::State *rstate = rmotion->state;
        bool start_tree = true;
        bool solved = false;
        unsigned int batch_size = initialBatchSize_;
        int num_batch_samples = static_cast<int>(tStart_->size() + tGoal_->size());
        int new_batch_goal_samples = 0;
        bool first_batch = true;
        double old_batch_sample_prob = 1.0;
        double old_batch_time_bound_factor = initialTimeBoundFactor_;
        double new_batch_time_bound_factor = initialTimeBoundFactor_;
        bool force_goal_sample = true;

        OMPL_INFORM("%s: Starting planning with time bound factor %.2f", getName().c_str(),
                    new_batch_time_bound_factor);

        while (!ptc)
        {
            numIterations_++;
            TreeData &tree = start_tree ? tStart_ : tGoal_;
            tgi.start = start_tree;
            start_tree = !start_tree;
            TreeData &other_tree = start_tree ? tStart_ : tGoal_;

            if (!isTimeBounded_ && static_cast<unsigned int>(num_batch_samples) >= batch_size)
            {
                if (first_batch)
                {
                    first_batch = false;
                    old_batch_sample_prob = 0.5 * (1 / timeBoundFactorIncrease_);
                }
                if (!increaseTimeBoundWithinLimit(false, old_batch_time_bound_factor, new_batch_time_bound_factor,
                                                  start_tree, batch_size, num_batch_samples))
                    break;
                if (!newBatchGoalMotions_.empty())
                {
                    goalMotions_.insert(goalMotions_.end(), newBatchGoalMotions_.begin(),
                                        newBatchGoalMotions_.end());
                    newBatchGoalMotions_.clear();
                }
                continue;
            }

            sampleOldBatch_ = (first_batch || isTimeBounded_ || !sampleUniformForUnboundedTime_ ||
                               rng_.uniform01() <= old_batch_sample_prob);

            ob::State *goal_state{nullptr};
            if (sampleOldBatch_)
            {
                if (goalMotions_.empty() && isTimeBounded_)
                    goal_state = nextGoal(ptc, old_batch_time_bound_factor, new_batch_time_bound_factor);
                else if (goalMotions_.empty() && !isTimeBounded_)
                {
                    goal_state = nextGoal(static_cast<int>(batch_size), old_batch_time_bound_factor,
                                          new_batch_time_bound_factor);
                    if (goal_state == nullptr)
                    {
                        if (!increaseTimeBoundWithinLimit(true, old_batch_time_bound_factor, new_batch_time_bound_factor,
                                                          start_tree, batch_size, num_batch_samples))
                            break;
                        continue;
                    }
                }
                else if (force_goal_sample ||
                         goalMotions_.size() < (tGoal_->size() - new_batch_goal_samples) / goalStateSampleRatio_)
                {
                    goal_state = nextGoal(1, old_batch_time_bound_factor, new_batch_time_bound_factor);
                    force_goal_sample = false;
                }
            }
            else
            {
                if (newBatchGoalMotions_.empty())
                {
                    goal_state = nextGoal(static_cast<int>(batch_size), old_batch_time_bound_factor,
                                          new_batch_time_bound_factor);
                    if (goal_state == nullptr)
                    {
                        if (!increaseTimeBoundWithinLimit(false, old_batch_time_bound_factor, new_batch_time_bound_factor,
                                                          start_tree, batch_size, num_batch_samples))
                            break;
                        continue;
                    }
                }
                else if (force_goal_sample ||
                         newBatchGoalMotions_.size() <
                             static_cast<unsigned long>(new_batch_goal_samples / goalStateSampleRatio_))
                {
                    goal_state = nextGoal(1, old_batch_time_bound_factor, new_batch_time_bound_factor);
                    force_goal_sample = false;
                }
            }

            // Check terminal occupancy only for goal roots. Passing through the
            // goal and leaving again remains a valid intermediate motion.
            if (goal_state != nullptr && goalValidityChecker_ && !goalValidityChecker_(goal_state))
            {
                force_goal_sample = true;
                continue;
            }
            if (goal_state != nullptr)
            {
                auto *motion = new Motion(si_);
                si_->copyState(motion->state, goal_state);
                motion->root = motion->state;
                tGoal_->add(motion);
                if (sampleOldBatch_)
                    goalMotions_.push_back(motion);
                else
                {
                    newBatchGoalMotions_.push_back(motion);
                    new_batch_goal_samples++;
                }

                minimumTime_ = std::min(
                    minimumTime_,
                    si_->getStateSpace()->as<ob::SpaceTimeStateSpace>()->timeToCoverDistance(
                        startMotion_->state, goal_state));
                num_batch_samples++;
            }

            bool success = sampler_.sample(rstate);
            if (!success)
            {
                force_goal_sample = true;
                continue;
            }

            GrowState gs = growTree(tree, tgi, rmotion, nbh, false);
            if (gs == TRAPPED)
                continue;

            num_batch_samples++;
            Motion *added_motion = tgi.xmotion;
            Motion *start_motion = nullptr;
            Motion *goal_motion = nullptr;

            bool new_solution = false;
            if (!tgi.start && rewireState_ != OFF)
            {
                new_solution = rewireGoalTree(added_motion);
                if (new_solution)
                {
                    std::queue<Motion *> queue;
                    queue.push(added_motion);
                    while (!queue.empty())
                    {
                        Motion *candidate = queue.front();
                        queue.pop();
                        if (candidate->connectionPoint != nullptr)
                        {
                            goal_motion = candidate;
                            start_motion = candidate->connectionPoint;
                            break;
                        }
                        for (Motion *child : candidate->children)
                            queue.push(child);
                    }
                }
            }

            if (gs != REACHED)
                si_->copyState(rstate, tgi.xstate);

            tgi.start = start_tree;
            if (!new_solution)
            {
                int total_samples = static_cast<int>(tStart_->size() + tGoal_->size());
                GrowState gsc = growTree(other_tree, tgi, rmotion, nbh, true);
                if (gsc == REACHED)
                {
                    new_solution = true;
                    start_motion = start_tree ? tgi.xmotion : added_motion;
                    goal_motion = start_tree ? added_motion : tgi.xmotion;
                    if (start_motion->parent != nullptr)
                        start_motion = start_motion->parent;
                    else
                        goal_motion = goal_motion->parent;
                }
                num_batch_samples += static_cast<int>(tStart_->size() + tGoal_->size()) - total_samples;
            }

            const double new_dist = tree->getDistanceFunction()(added_motion, other_tree->nearest(added_motion));
            if (new_dist < distanceBetweenTrees_)
                distanceBetweenTrees_ = new_dist;

            if (new_solution && start_motion != nullptr && goal_motion != nullptr &&
                goal->isStartGoalPairValid(start_motion->root, goal_motion->root))
            {
                constructSolution(start_motion, goal_motion, intermediate_solution_callback);
                solved = true;
                if (ptc || upperTimeBound_ == minimumTime_)
                    break;
            }
            else if (!start_tree)
            {
                double dist = 0.0;
                goal->isSatisfied(tgi.xmotion->state, &dist);
                if (dist < approxdif)
                {
                    approxdif = dist;
                    approxsol = tgi.xmotion;
                }
            }
        }

        si_->freeState(tgi.xstate);
        si_->freeState(rstate);
        delete rmotion;

        OMPL_INFORM("%s: Created %u states (%u start + %u goal)", getName().c_str(),
                    tStart_->size() + tGoal_->size(), tStart_->size(), tGoal_->size());

        if (approxsol && !solved)
        {
            std::vector<Motion *> mpath;
            while (approxsol != nullptr)
            {
                mpath.push_back(approxsol);
                approxsol = approxsol->parent;
            }

            auto path(std::make_shared<og::PathGeometric>(si_));
            for (int idx = static_cast<int>(mpath.size()) - 1; idx >= 0; --idx)
                path->append(mpath[static_cast<std::size_t>(idx)]->state);
            pdef_->addSolutionPath(path, true, approxdif, getName());
            return ob::PlannerStatus::APPROXIMATE_SOLUTION;
        }
        if (solved)
        {
            ob::PlannerSolution psol(bestSolution_);
            psol.setPlannerName(getName());
            ob::OptimizationObjectivePtr optimization_objective = std::make_shared<ob::MinimizeArrivalTime>(si_);
            psol.setOptimized(optimization_objective, ob::Cost(bestTime_), false);
            pdef_->addSolutionPath(psol);
        }

        return solved ? ob::PlannerStatus::EXACT_SOLUTION : ob::PlannerStatus::TIMEOUT;
    }

protected:
    bool increaseTimeBoundWithinLimit(bool has_equal_bounds,
                                      double &old_batch_time_bound_factor,
                                      double &new_batch_time_bound_factor,
                                      bool &start_tree,
                                      unsigned int &batch_size,
                                      int &num_batch_samples)
    {
        if (!std::isfinite(maxTimeBoundFactor_))
        {
            increaseTimeBound(has_equal_bounds, old_batch_time_bound_factor, new_batch_time_bound_factor, start_tree,
                              batch_size, num_batch_samples);
            return true;
        }
        if (new_batch_time_bound_factor >= maxTimeBoundFactor_)
        {
            OMPL_INFORM("%s: Reached maximum time bound factor %.2f", getName().c_str(), maxTimeBoundFactor_);
            return false;
        }

        const double saved_increase = timeBoundFactorIncrease_;
        const double capped_increase = maxTimeBoundFactor_ / new_batch_time_bound_factor;
        if (std::isfinite(capped_increase) && capped_increase > 1.0 && capped_increase < timeBoundFactorIncrease_)
            timeBoundFactorIncrease_ = capped_increase;

        increaseTimeBound(has_equal_bounds, old_batch_time_bound_factor, new_batch_time_bound_factor, start_tree,
                          batch_size, num_batch_samples);
        timeBoundFactorIncrease_ = saved_increase;
        old_batch_time_bound_factor = std::min(old_batch_time_bound_factor, maxTimeBoundFactor_);
        new_batch_time_bound_factor = std::min(new_batch_time_bound_factor, maxTimeBoundFactor_);
        return true;
    }

    void constructSolution(Motion *start_motion,
                           Motion *goal_motion,
                           const ob::ReportIntermediateSolutionFn &intermediate_solution_callback)
    {
        if (goal_motion->connectionPoint == nullptr)
        {
            goal_motion->connectionPoint = start_motion;
            Motion *motion = goal_motion;
            while (motion != nullptr)
            {
                motion->numConnections++;
                motion = motion->parent;
            }
        }

        auto new_time = goal_motion->root->as<ob::CompoundState>()->as<ob::TimeStateSpace::StateType>(1)->position;
        if (new_time >= upperTimeBound_)
            return;

        numSolutions_++;
        isTimeBounded_ = true;
        if (!newBatchGoalMotions_.empty())
        {
            goalMotions_.insert(goalMotions_.end(), newBatchGoalMotions_.begin(), newBatchGoalMotions_.end());
            newBatchGoalMotions_.clear();
        }

        Motion *solution = start_motion;
        std::vector<Motion *> start_path;
        while (solution != nullptr)
        {
            start_path.push_back(solution);
            solution = solution->parent;
        }

        solution = goal_motion;
        std::vector<Motion *> goal_path;
        while (solution != nullptr)
        {
            goal_path.push_back(solution);
            solution = solution->parent;
        }

        std::vector<const ob::State *> const_path;
        auto path(std::make_shared<og::PathGeometric>(si_));
        path->getStates().reserve(start_path.size() + goal_path.size());
        for (int idx = static_cast<int>(start_path.size()) - 1; idx >= 0; --idx)
        {
            const_path.push_back(start_path[static_cast<std::size_t>(idx)]->state);
            path->append(start_path[static_cast<std::size_t>(idx)]->state);
        }
        for (Motion *motion : goal_path)
        {
            const_path.push_back(motion->state);
            path->append(motion->state);
        }

        bestSolution_ = path;
        auto *reached_goal = path->getState(path->getStateCount() - 1);
        bestTime_ = reached_goal->as<ob::CompoundState>()->as<ob::TimeStateSpace::StateType>(1)->position;

        if (intermediate_solution_callback)
            intermediate_solution_callback(this, const_path, ob::Cost(bestTime_));

        upperTimeBound_ = (bestTime_ - minimumTime_) * optimumApproxFactor_ + minimumTime_;
        // Keep OMPL's narrowed time bound, but skip its destructive prune path:
        // in OMPL 1.7 it can leave stale connectionPoint references when
        // planning continues after the first solution.
    }

private:
    double maxTimeBoundFactor_{std::numeric_limits<double>::infinity()};
    std::function<bool(const ob::State *)> goalValidityChecker_;
};

py::dict solve_linf_strrt(py::dict environment_payload,
                          py::object extra_collision_checker,
                          py::array_t<double, py::array::c_style | py::array::forcecast> start_array,
                          py::array_t<double, py::array::c_style | py::array::forcecast> goal_array,
                          double t0,
                          py::array_t<double, py::array::c_style | py::array::forcecast> vlimit_array,
                          py::array_t<double, py::array::c_style | py::array::forcecast> lb_array,
                          py::array_t<double, py::array::c_style | py::array::forcecast> ub_array,
                          int seed,
                          py::dict options)
{
    const auto start = vector_from_array(start_array, "start");
    const auto goal = vector_from_array(goal_array, "goal");
    auto vlimit = vector_from_array(vlimit_array, "vlimit");
    const auto lb = vector_from_array(lb_array, "lb");
    const auto ub = vector_from_array(ub_array, "ub");
    const unsigned int dim = static_cast<unsigned int>(start.size());
    if (goal.size() != dim || lb.size() != dim || ub.size() != dim)
        throw std::invalid_argument("start, goal, lb, and ub must have matching dimensions.");
    if (vlimit.size() == 1 && dim > 1)
        vlimit = std::vector<double>(dim, vlimit.front());
    if (vlimit.size() != dim)
        throw std::invalid_argument("vlimit must be scalar or match the spatial dimension.");
    const double robot_radius = py::cast<double>(environment_payload["robot_radius"]);
    if (!(std::isfinite(robot_radius) && robot_radius >= 0.0))
        throw std::invalid_argument("robot_radius must be finite and nonnegative.");
    const auto native_environment = std::make_shared<const NativeEnvironment2D>(
        native_environment_from_payload(environment_payload));
    if (dim != 2 && native_environment->hasGeometry())
        throw std::invalid_argument("Native STRRT* Env geometry checking currently supports 2D geometry only.");

    auto spatial = std::make_shared<ScaledLInfRealVectorStateSpace>(dim, vlimit);
    ob::RealVectorBounds bounds(dim);
    for (unsigned int idx = 0; idx < dim; ++idx)
    {
        bounds.setLow(idx, lb[idx]);
        bounds.setHigh(idx, ub[idx]);
    }
    spatial->setBounds(bounds);

    const double time_weight = option_double(options, "time_weight", 0.5);
    auto space = std::make_shared<ob::SpaceTimeStateSpace>(spatial, 1.0, time_weight);
    const auto local_seed = static_cast<std::uint_fast32_t>(seed == 0 ? 1 : seed);
    space->setStateSamplerAllocator([local_seed](const ob::StateSpace *state_space) -> ob::StateSamplerPtr {
        return std::make_shared<SeededSpaceTimeStateSampler>(state_space, local_seed);
    });
    const double time_upper_bound = option_double(options, "time_upper_bound", std::numeric_limits<double>::infinity());
    const bool stay_at_goal = option_bool(options, "stay_at_goal", false);
    if (stay_at_goal && !std::isfinite(time_upper_bound))
        throw std::invalid_argument("stay_at_goal requires a finite time_upper_bound.");
    if (std::isfinite(time_upper_bound))
    {
        if (time_upper_bound < t0)
            throw std::invalid_argument("time_upper_bound must be greater than or equal to t0.");
        space->setTimeBounds(t0, time_upper_bound);
    }
    space->updateEpsilon();

    auto si = std::make_shared<ob::SpaceInformation>(space);
    si->setStateValidityChecker([native_environment, extra_collision_checker, dim, t0, robot_radius, space](const ob::State *state) {
        return space->satisfiesBounds(state) &&
               env_state_valid(native_environment, extra_collision_checker, state, dim, t0, robot_radius);
    });
    si->setMotionValidator(
        std::make_shared<EnvMotionValidator>(si, native_environment, extra_collision_checker, dim, robot_radius));
    si->setup();

    auto pdef = std::make_shared<ob::ProblemDefinition>(si);
    const double goal_tolerance = option_double(options, "goal_tolerance", 1e-7);
    if (!(std::isfinite(goal_tolerance) && goal_tolerance >= 0.0))
        throw std::invalid_argument("goal_tolerance must be finite and nonnegative.");
    ob::ScopedState<> start_state(space);
    ob::ScopedState<> goal_state(space);
    assign_compound_state(start_state.get(), start, t0);
    assign_compound_state(goal_state.get(), goal, t0);
    pdef->setStartAndGoalStates(start_state, goal_state, goal_tolerance);

    auto planner = std::make_shared<PatchedSTRRTstar>(si);
    planner->setLocalSeed(local_seed);
    if (stay_at_goal && !extra_collision_checker.is_none())
    {
        planner->setGoalValidityChecker([extra_collision_checker, dim, time_upper_bound](const ob::State *state) {
            const auto position = spatial_values(state, dim);
            py::gil_scoped_acquire gil;
            return !py::cast<bool>(extra_collision_checker(
                py_array_from_vector(position), py_array_from_vector(position),
                state_time(state), time_upper_bound));
        });
    }
    const double range = option_double(options, "range", 1.0);
    if (std::isfinite(range) && range > 0.0)
        planner->setRange(range);
    const int batch_size = option_int(options, "batch_size", -1);
    if (batch_size > 0)
        planner->setBatchSize(batch_size);
    const double initial_factor = option_double(options, "initial_time_bound_factor",
                                                std::numeric_limits<double>::quiet_NaN());
    if (std::isfinite(initial_factor) && initial_factor > 0.0)
        planner->setInitialTimeBoundFactor(initial_factor);
    const double increase = option_double(options, "time_bound_factor_increase",
                                          std::numeric_limits<double>::quiet_NaN());
    if (std::isfinite(increase) && increase > 1.0)
        planner->setTimeBoundFactorIncrease(increase);
    const double max_factor = option_double(options, "max_time_bound_factor",
                                            std::numeric_limits<double>::quiet_NaN());
    if (std::isfinite(max_factor))
        planner->setMaxTimeBoundFactor(max_factor);
    const double approx = option_double(options, "optimum_approx_factor", std::numeric_limits<double>::quiet_NaN());
    if (std::isfinite(approx) && approx > 0.0 && approx <= 1.0)
        planner->setOptimumApproxFactor(approx);

    planner->setProblemDefinition(pdef);
    planner->setup();

    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    };

    SolutionSnapshot first_snapshot;
    SolutionSnapshot final_snapshot;

    const double budget = option_double(options, "max_runtime_in_secs", std::numeric_limits<double>::infinity());
    if (!(std::isfinite(budget) && budget > 0.0))
        throw std::invalid_argument("max_runtime_in_secs must be finite and positive.");
    const bool return_first_valid = option_bool(options, "return_first_valid", false);

    std::atomic_bool solution_found{false};
    pdef->setIntermediateSolutionCallback(
        [&solution_found, &first_snapshot, &final_snapshot, dim, &goal, &vlimit, goal_tolerance, &elapsed, budget](
            const ob::Planner *,
            const std::vector<const ob::State *> &states,
            const ob::Cost) {
            const double runtime = elapsed();
            if (runtime > budget)
                return;
            auto snapshot = solution_snapshot(states, dim, goal, vlimit, goal_tolerance, runtime);
            if (snapshot.is_success)
            {
                if (!solution_found.load())
                    first_snapshot = snapshot;
                if (!final_snapshot.is_success || snapshot.cost < final_snapshot.cost)
                    final_snapshot = std::move(snapshot);
                solution_found.store(true);
            }
        });
    const auto timeout_ptc = ob::timedPlannerTerminationCondition(budget);
    const ob::PlannerTerminationCondition first_solution_ptc([&solution_found]() {
        return solution_found.load();
    });
    ob::PlannerTerminationCondition solve_ptc = timeout_ptc;
    if (return_first_valid)
        solve_ptc = ob::plannerOrTerminationCondition(timeout_ptc, first_solution_ptc);

    {
        py::gil_scoped_release release;
        planner->solve(solve_ptc);
    }
    pdef->setIntermediateSolutionCallback(ob::ReportIntermediateSolutionFn());

    // The final measurement consumes the allotted search time; its path is the
    // best exact solution captured before the deadline, excluding shutdown time.
    final_snapshot.runtime = std::min(elapsed(), budget);

    py::dict result;
    result["first"] = solution_payload(first_snapshot);
    result["final"] = solution_payload(final_snapshot);
    result["has_exact_solution"] = pdef->hasExactSolution();
    result["has_approximate_solution"] = pdef->hasApproximateSolution();
    result["solution_difference"] = pdef->getSolutionDifference();
    return result;
}
}  // namespace

PYBIND11_MODULE(_ompl_strrt_star_native, m)
{
    m.doc() = "Official OMPL STRRTstar adapter for stgcs Env with scaled L-infinity distance.";
    m.def("solve_linf_strrt", &solve_linf_strrt, py::arg("environment_payload"), py::arg("extra_collision_checker"),
          py::arg("start"), py::arg("goal"), py::arg("t0"), py::arg("vlimit"), py::arg("lb"), py::arg("ub"),
          py::arg("seed"), py::arg("options"));
}
