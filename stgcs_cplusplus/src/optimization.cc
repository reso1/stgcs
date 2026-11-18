#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <drake/geometry/optimization/point.h>
#include <drake/solvers/solve.h>

#include "internal.h"

namespace stgcs {
namespace {
// Reuse the GCS bindings, substituting one variable for each exit-entry pair.
class PathRestriction {
 public:
  explicit PathRestriction(const std::vector<const GCS::Edge*>& edges) {
    std::vector<const GCS::Vertex*> vertices;
    std::unordered_set<const GCS::Vertex*> seen;
    for (const auto* edge : edges) {
      const auto& u = edge->u();
      const auto& v = edge->v();
      if (u.x().size() != v.x().size() || u.x().size() % 2)
        throw std::invalid_argument("ST-GCS vertices must have equal entry/exit dimensions");
      for (const auto* vertex : {&u, &v})
        if (seen.insert(vertex).second) vertices.push_back(vertex);
      const int n = u.x().size() / 2;
      for (int i = 0; i < n; ++i) {
        const auto entry = representative(v.x()[i]);
        representatives_[entry.get_id()] = representative(u.x()[n + i]);
      }
    }
    const std::unordered_set<GCS::Transcription> transcription{GCS::Transcription::kRestriction};
    for (const auto* vertex : vertices) {
      vertex->set().AddPointInSetConstraints(&program_, variables(vertex->x()));
      for (const auto& binding : vertex->GetConstraints(transcription)) add_constraint(binding);
      for (const auto& binding : vertex->GetCosts(transcription))
        program_.AddCost(drake::solvers::Binding<drake::solvers::Cost>(
            binding.evaluator(), variables(binding.variables())));
    }
    for (const auto* edge : edges) {
      for (const auto& binding : edge->GetConstraints(transcription)) add_constraint(binding);
      for (const auto& binding : edge->GetCosts(transcription))
        program_.AddCost(drake::solvers::Binding<drake::solvers::Cost>(
            binding.evaluator(), variables(binding.variables())));
    }
  }

  drake::solvers::MathematicalProgramResult solve(const GCSOptions& options) {
    const auto& settings = options.restriction_solver_options.value_or(options.solver_options);
    const auto* solver = options.restriction_solver ? options.restriction_solver : options.solver;
    drake::solvers::MathematicalProgramResult result;
    if (solver) solver->Solve(program_, std::nullopt, settings, &result);
    else result = drake::solvers::Solve(program_, std::nullopt, settings);
    if (result.is_success()) {
      // Expand only the returned values; eliminated variables never enter the solve.
      std::unordered_map<drake::symbolic::Variable::Id, int> index;
      Vector values(representatives_.size());
      for (const auto& [id, variable] : representatives_) {
        const int i = index.size();
        index.emplace(id, i);
        values[i] = result.GetSolution(representative(variable));
      }
      result.set_decision_variable_index(std::move(index));
      result.set_x_val(values);
    }
    return result;
  }

 private:
  drake::symbolic::Variable representative(const drake::symbolic::Variable& variable) {
    const auto id = variable.get_id();
    const auto [it, inserted] = representatives_.try_emplace(id, variable);
    auto result = it->second;
    if (result.get_id() != id) {
      result = representative(result);
      representatives_[id] = result;
    }
    return result;
  }

  drake::solvers::VectorXDecisionVariable variables(
      const drake::solvers::VectorXDecisionVariable& original) {
    drake::solvers::VectorXDecisionVariable mapped(original.size());
    for (int i = 0; i < original.size(); ++i) {
      mapped[i] = representative(original[i]);
      if (registered_.insert(mapped[i].get_id()).second)
        program_.AddDecisionVariables(mapped.segment(i, 1));
    }
    return mapped;
  }

  void add_constraint(const drake::solvers::Binding<drake::solvers::Constraint>& binding) {
    const auto mapped = variables(binding.variables());
    const auto equality = std::dynamic_pointer_cast<drake::solvers::LinearEqualityConstraint>(
        binding.evaluator());
    if (!equality) {
      program_.AddConstraint(drake::solvers::Binding<drake::solvers::Constraint>(
          binding.evaluator(), mapped));
      return;
    }
    std::unordered_map<drake::symbolic::Variable::Id, int> index;
    drake::solvers::VectorXDecisionVariable unique(mapped.size());
    for (const auto& variable : mapped)
      if (!index.contains(variable.get_id())) {
        const int i = index.size();
        index.emplace(variable.get_id(), i);
        unique[i] = variable;
      }
    Matrix a = Matrix::Zero(equality->num_constraints(), index.size());
    const Matrix original = equality->GetDenseA();
    for (int i = 0; i < mapped.size(); ++i) a.col(index.at(mapped[i].get_id())) += original.col(i);
    Vector b = equality->lower_bound();
    int rows = 0;
    for (int i = 0; i < a.rows(); ++i)
      if (!a.row(i).isZero(0) || b[i] != 0) {
        a.row(rows) = a.row(i);
        b[rows++] = b[i];
      }
    if (rows)
      program_.AddLinearEqualityConstraint(a.topRows(rows), b.head(rows), unique.head(index.size()));
  }

  drake::solvers::MathematicalProgram program_;
  std::unordered_map<drake::symbolic::Variable::Id, drake::symbolic::Variable> representatives_;
  std::unordered_set<drake::symbolic::Variable::Id> registered_;
};
}  // namespace

std::unique_ptr<OptimizationCache::Model> OptimizationCache::acquire() {
  std::lock_guard lock(mutex_);
  return idle_ ? std::move(idle_) : std::make_unique<Model>();
}

void OptimizationCache::release(std::unique_ptr<Model> model) {
  std::lock_guard lock(mutex_);
  if (!idle_) idle_ = std::move(model);
}

void Query::validate(int dimension) const {
  if (start.size() != dimension || goal.size() != dimension || !start.allFinite() ||
      !goal.allFinite() || !std::isfinite(start_time))
    throw std::invalid_argument("Invalid query coordinates");
}

Optimization::Optimization(const Graph& input, const Query* query)
    : graph(input),
      cache_(input.optimization_cache_),
      model_(cache_->acquire()),
      gcs_(model_->gcs),
      vertices_(model_->vertices),
      edges_(model_->edges) {
  for (const auto& vertex : graph.vertices()) {
    sets.push_back(vertex.set);
    roots.push_back(vertex.root);
  }
  next = graph.edges();
  if (query) {
    const int d = graph.dimension();
    query->validate(d);
    const double earliest =
        query->start_time + (query->start - query->goal).lpNorm<Eigen::Infinity>() / graph.speed();
    if (query->start_time < graph.start_time() || earliest > graph.end_time()) return;
    Vector start(d + 1), lower(d + 1), upper(d + 1);
    start << query->start, query->start_time;
    lower << query->goal, earliest;
    upper << query->goal, graph.end_time();
    std::vector<int> containing;
    for (std::size_t i = 0; i < sets.size(); ++i)
      if (sets[i].PointInSet(start, 1e-8)) {
        containing.push_back(i);
        break;
      }
    if (containing.empty()) return;
    const auto goal = Geometry::box(lower, upper);
    struct Interval {
      int vertex;
      double start, end;
    };
    std::vector<Interval> intervals;
    for (std::size_t i = 0; i < sets.size(); ++i)
      if (auto interval = Geometry::time_interval(sets[i], query->goal, earliest, graph.end_time()))
        intervals.push_back({static_cast<int>(i), interval->first, interval->second});
    double connected_start = graph.end_time();
    if (query->stay_at_goal) {
      std::sort(intervals.begin(), intervals.end(),
                [](const auto& a, const auto& b) { return a.end > b.end; });
      if (intervals.empty() || intervals.front().end < graph.end_time() - kTolerance) return;
      for (const auto& interval : intervals)
        if (interval.end >= connected_start - kTolerance)
          connected_start = std::min(connected_start, interval.start);
    }
    for (const auto& interval : intervals)
      if (!query->stay_at_goal || interval.end >= connected_start - kTolerance)
        goal_parents.push_back(interval.vertex);
    std::sort(goal_parents.begin(), goal_parents.end());
    if (goal_parents.empty()) return;
    source = sets.size();
    target = source + 1;
    sets.push_back(Geometry::point(start));
    sets.push_back(goal);
    roots.push_back(roots[containing.front()]);
    roots.push_back(-1);
    next.resize(sets.size());
    next[source] = std::move(containing);
    for (int parent : goal_parents) {
      const int dummy = sets.size();
      sets.push_back(goal);
      roots.push_back(roots[parent]);
      next.emplace_back(std::vector<int>{target});
      next[parent].push_back(dummy);
    }
  }
  previous.resize(sets.size());
  const auto base_size = graph.vertices().size();
  if (model_->vertices.empty()) {
    for (std::size_t i = 0; i < base_size; ++i) vertices_.push_back(add_vertex(sets[i], true));
    for (std::size_t i = 0; i < graph.edges().size(); ++i)
      for (int j : graph.edges()[i]) edges_[{i, j}] = add_edge(vertices_[i], vertices_[j]);
    model_->vertices = vertices_;
    model_->edges = edges_;
  }
  for (std::size_t i = base_size; i < sets.size(); ++i) {
    if (static_cast<int>(i) == source) {
      Vector point(2 * (graph.dimension() + 1));
      point.head(graph.dimension()) = point.segment(graph.dimension() + 1, graph.dimension()) = query->start;
      point[graph.dimension()] = point[point.size() - 1] = query->start_time;
      auto* vertex = gcs_.AddVertex(drake::geometry::optimization::Point(point));
      vertex->AddConstraint(drake::solvers::Binding<drake::solvers::LinearEqualityConstraint>(
          std::make_shared<drake::solvers::LinearEqualityConstraint>(
              Matrix::Identity(point.size(), point.size()), point), vertex->x()));
      vertices_.push_back(vertex);
    } else {
      vertices_.push_back(add_vertex(sets[i], false, false));
    }
  }
  for (std::size_t i = 0; i < next.size(); ++i)
    for (int j : next[i]) {
      if (i >= base_size || j >= static_cast<int>(base_size))
        edges_[{i, j}] = add_edge(vertices_[i], vertices_[j]);
      previous[j].push_back(i);
    }
}

Optimization::~Optimization() {
  for (std::size_t i = model_->vertices.size(); i < vertices_.size(); ++i)
    gcs_.RemoveVertex(vertices_[i]);
  cache_->release(std::move(model_));
}

GCS::Vertex* Optimization::add_vertex(const Polyhedron& set, bool motion, bool stationary) {
  auto* vertex = gcs_.AddVertex(set.CartesianPower(2));
  const int n = set.ambient_dimension();
  const auto& x = vertex->x();
  if (motion) {
    add_motion_constraints(vertex, graph.speed(), graph.minimum_duration());
  } else if (stationary) {
    for (int i = 0; i < n; ++i) vertex->AddConstraint(x[n + i] == x[i]);
  }
  return vertex;
}

void Optimization::add_motion_constraints(GCS::Vertex* vertex, double speed,
                                          double minimum_duration) {
  const auto& x = vertex->x();
  const int n = x.size() / 2, d = n - 1;
  Matrix a = Matrix::Zero(2*d + 1, 2*n);
  Vector b = Vector::Zero(2*d + 1);
  for (int i = 0; i < d; ++i) {
    a(i, i) = a(d+i, n+i) = 1;
    a(i, n+i) = a(d+i, i) = -1;
    a(i, n-1) = a(d+i, n-1) = speed;
    a(i, 2*n-1) = a(d+i, 2*n-1) = -speed;
  }
  a(2*d, n-1) = 1;
  a(2*d, 2*n-1) = -1;
  b[2*d] = -minimum_duration;
  vertex->AddConstraint(drake::solvers::Binding<drake::solvers::LinearConstraint>(
      std::make_shared<drake::solvers::LinearConstraint>(
          a, Vector::Constant(b.size(), -kInfinity), b), x));
  vertex->AddCost(x[2*n-1] - x[n-1]);
}

GCS::Edge* Optimization::add_edge(GCS::Vertex* from, GCS::Vertex* to) {
  auto* edge = gcs_.AddEdge(from, to);
  const int n = graph.dimension() + 1;
  Matrix a = Matrix::Zero(n, 4*n);
  a.block(0, n, n, n).setIdentity();
  a.block(0, 2*n, n, n) = -Matrix::Identity(n, n);
  drake::solvers::VectorXDecisionVariable x(4*n);
  x << edge->xu(), edge->xv();
  edge->AddConstraint(drake::solvers::Binding<drake::solvers::LinearEqualityConstraint>(
      std::make_shared<drake::solvers::LinearEqualityConstraint>(a, Vector::Zero(n)), x));
  return edge;
}

GCSOptions Optimization::solver_options(double timeout) {
  GCSOptions options;
  options.solver_options = Geometry::solver_options(timeout);
  // Keep presolve enabled, but use the interior-point solution: LP basis
  // recovery can violate thin reservation facets and minimum durations.
  options.restriction_solver_options = options.solver_options;
  options.restriction_solver_options->SetOption(
      drake::solvers::MosekSolver::id(), "MSK_IPAR_INTPNT_BASIS", 0);
  return options;
}

std::optional<Trajectory> Optimization::extract(const std::vector<const GCS::Edge*>& edges,
                                 const std::vector<int>& path,
                                 const drake::solvers::MathematicalProgramResult& result) {
  std::vector<Segment> segments;
  const int n = graph.dimension() + 1;
  if (edges.empty()) return std::nullopt;
  for (std::size_t i = 0; i < edges.size(); ++i) {
    // Query goal variables have no motion constraints. They belong to the
    // optimization model, not to the returned physical trajectory.
    if (target >= 0 && path[i] >= target) continue;
    const Vector x = result.GetSolution(edges[i]->u().x());
    segments.push_back({x.head(n), x.tail(n), path[i]});
  }
  return Trajectory(std::move(segments));
}

std::optional<Trajectory> Optimization::restrict(const std::vector<int>& path,
                                                 const Deadline& deadline,
                                                 const Polyhedron* initial,
                                                 const Vector* final_position,
                                                 double* sampled_duration,
                                                 const Vector* motion_goal,
                                                 double inflation) {
  if (deadline.expired() || path.empty()) return std::nullopt;
  auto restricted_path = path;
  if (target >= 0 && restricted_path.back() == target) restricted_path.pop_back();
  if (restricted_path.empty() || (restricted_path.size() < 2 && !initial)) return std::nullopt;
  std::vector<const GCS::Edge*> edges;
  std::vector<GCS::Vertex*> temporary;
  // Temporary vertices belong to this solve only, including exceptional exits.
  struct Cleanup {
    GCS& graph;
    std::vector<GCS::Vertex*>& vertices;
    ~Cleanup() {
      for (auto* vertex : vertices) graph.RemoveVertex(vertex);
    }
  } cleanup{gcs_, temporary};
  auto full_path = restricted_path;
  if (initial) {
    auto* vertex = add_vertex(*initial, false);
    temporary.push_back(vertex);
    edges.push_back(add_edge(vertex, vertices_.at(restricted_path.front())));
    full_path.insert(full_path.begin(), -1);
  }
  for (std::size_t i = 1; i < restricted_path.size(); ++i)
    edges.push_back(edges_.at({restricted_path[i - 1], restricted_path[i]}));
  if (!final_position) {
    // A prefix ends at the terminal entry: y_v = x_v, with no terminal
    // duration or motion cost. Use a solve-local copy to preserve the graph.
    auto* predecessor = restricted_path.size() > 1
        ? vertices_.at(restricted_path[restricted_path.size() - 2]) : temporary.front();
    auto* terminal = add_vertex(sets.at(restricted_path.back()), false);
    temporary.push_back(terminal);
    auto* edge = add_edge(predecessor, terminal);
    edges.back() = edge;
    if (motion_goal && restricted_path.back() < target) {
      // Share the terminal entry with the motion bound in the same program.
      const auto z = edge->NewSlackVariables(1, "motion_cost_to_go");
      edge->AddConstraint(z[0] >= 0);
      for (int i = 0; i < graph.dimension(); ++i) {
        const auto displacement = (*motion_goal)[i] - edge->xv()[i];
        edge->AddConstraint(displacement <= graph.speed() * z[0]);
        edge->AddConstraint(-displacement <= graph.speed() * z[0]);
      }
      edge->AddCost(inflation * z[0]);
    }
  }
  if (final_position) {
    Vector lo(graph.dimension() + 1), hi(graph.dimension() + 1);
    lo << *final_position, graph.start_time();
    hi << *final_position, graph.end_time();
    auto* vertex = add_vertex(
        Geometry::box(lo, hi).Intersection(sets.at(restricted_path.back())), false, false);
    temporary.push_back(vertex);
    edges.push_back(add_edge(vertices_.at(restricted_path.back()), vertex));
    full_path.push_back(-1);
  }
  PathRestriction restriction(edges);
  const auto result = restriction.solve(solver_options(deadline.remaining()));
  if (!result.is_success()) return std::nullopt;
  if (sampled_duration) {
    const int n = graph.dimension() + 1;
    *sampled_duration = result.GetSolution(edges.back()->u().x())[2*n-1] -
                        result.GetSolution(edges.front()->u().x())[n-1];
  }
  auto trajectory = extract(edges, full_path, result);
  if (initial && trajectory) {
    auto segments = trajectory->segments();
    segments.erase(segments.begin());
    trajectory = Trajectory(std::move(segments));
  }
  return trajectory;
}

std::optional<Trajectory> Optimization::shortest(const PlannerOptions& input,
                                                 const Deadline& deadline) {
  if (!valid() || deadline.expired()) return std::nullopt;
  auto options = solver_options(deadline.remaining());
  options.convex_relaxation = input.method == Method::RoundedRelaxation;
  options.max_rounded_paths = *options.convex_relaxation ? input.rounded_paths : 0;
  options.max_rounding_trials = input.rounded_paths;
  options.rounding_seed = input.seed;
  options.preprocessing = true;
  if (input.method == Method::RoundedRelaxation) {
    // Sample the interior-point flows directly. LP basis recovery can consume
    // the budget after the relaxation has already solved. Let MOSEK choose the
    // primal/dual form, and omit the expensive per-edge preprocessing LPs.
    const auto solver = drake::solvers::MosekSolver::id();
    options.preprocessing = false;
    options.solver_options.SetOption(solver, "MSK_IPAR_INTPNT_BASIS", 0);
    options.solver_options.SetOption(solver, "MSK_IPAR_INTPNT_SOLVE_FORM", 0);
  }
  if (input.method == Method::MixedInteger) {
    // Near-binary flows can distort extracted states beyond trajectory tolerances.
    const auto solver = drake::solvers::MosekSolver::id();
    options.solver_options.SetOption(solver, "MSK_DPAR_MIO_TOL_ABS_RELAX_INT", 1e-9);
    options.solver_options.SetOption(solver, "MSK_DPAR_MIO_TOL_FEAS", 1e-9);
  }
  auto result = gcs_.SolveShortestPath(*vertices_[source], *vertices_[target], options);
  if (!result.is_success()) return std::nullopt;
  auto edges = gcs_.GetSolutionPath(*vertices_[source], *vertices_[target], result);
  std::vector<int> path;
  for (const auto* edge : edges) {
    auto it = std::find(vertices_.begin(), vertices_.end(), &edge->u());
    path.push_back(std::distance(vertices_.begin(), it));
  }
  return extract(edges, path, result);
}

const Polyhedron& Optimization::interface(int previous, int current) const {
  if (previous < 0) return sets.at(current);
  const auto key = std::make_pair(previous, current);
  auto it = interfaces_.find(key);
  if (it == interfaces_.end())
    it = interfaces_.emplace(key, sets.at(previous).Intersection(sets.at(current))).first;
  return it->second;
}

double Optimization::triplet_cost(int u, int v, int w, const Deadline& deadline) const {
  // The dummy already represents arrival at the goal. Its edge to the common target is free.
  if (w == target) return 0;
  drake::solvers::MathematicalProgram program;
  const int d = graph.dimension();
  auto entry = program.NewContinuousVariables(d + 1, "entry");
  auto exit = program.NewContinuousVariables(d + 1, "exit");
  interface(u, v).AddPointInSetConstraints(&program, entry);
  interface(v, w).AddPointInSetConstraints(&program, exit);
  const auto dt = exit[d] - entry[d];
  program.AddLinearConstraint(dt >= 0);
  for (int i = 0; i < d; ++i) {
    program.AddLinearConstraint(exit[i] - entry[i] <= graph.speed() * dt);
    program.AddLinearConstraint(entry[i] - exit[i] <= graph.speed() * dt);
  }
  program.AddLinearCost(dt);
  const auto result = drake::solvers::Solve(program, std::nullopt,
                                            Geometry::solver_options(deadline.remaining()));
  if (result.is_success()) return std::max(0.0, result.get_optimal_cost());
  if (result.get_solution_result() == drake::solvers::SolutionResult::kInfeasibleConstraints)
    return kInfinity;
  return 0;  // An interrupted or inconclusive LP supplies no positive lower bound.
}

Trajectory Optimization::physical(const Trajectory& trajectory) const {
  std::vector<Segment> result;
  for (const auto& segment : trajectory.segments())
    if (segment.vertex >= 0 && segment.vertex < static_cast<int>(graph.vertices().size()))
      result.push_back(segment);
  return Trajectory(std::move(result));
}

Planner::Planner(PlannerOptions options) : options_(std::move(options)) {
  if (!std::isfinite(options_.inflation) || options_.inflation < 1 ||
      std::isnan(options_.timeout) || options_.timeout < 0 || options_.rounded_paths < 1 ||
      options_.position_samples < 1 || options_.heuristics.empty())
    throw std::invalid_argument("Invalid planner options");
  if (options_.key_evaluation == KeyEvaluation::Joint &&
      (options_.method != Method::Search ||
       options_.heuristics != std::vector<Heuristic>{Heuristic::Motion}))
    throw std::invalid_argument("Joint evaluation requires motion-only search");
}

PlanResult Planner::plan(const Graph& graph, const Query& query) const {
  Deadline deadline(options_.timeout);
  PlanResult result;
  if (deadline.expired()) {
    result.status = Status::Timeout;
    return result;
  }
  Deadline construction(kInfinity);
  Optimization model(graph, &query);
  const ProfileEntry gcs_profile{1, construction.elapsed()};
  if (model.valid()) {
    if (options_.method == Method::Search)
      result = SearchEngine::run(model, query, options_, deadline);
    else {
      result.trajectory = model.shortest(options_, deadline);
      result.status = result.trajectory ? Status::Success : Status::Exhausted;
    }
  }
  if (result.trajectory) result.trajectory = model.physical(*result.trajectory);
  result.profile.gcs = gcs_profile;
  if (deadline.expired() && !result.success()) result.status = Status::Timeout;
  result.runtime = deadline.elapsed();
  return result;
}
}  // namespace stgcs
