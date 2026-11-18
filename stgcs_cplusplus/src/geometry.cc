#include "stgcs/geometry.h"

#include <drake/geometry/optimization/vpolytope.h>
#include <drake/solvers/mathematical_program.h>

#include <cddlib/setoper.h>
#include <cddlib/cdd.h>
#include <drake/solvers/clarabel_solver.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/solve.h>

#include <Eigen/SVD>
#include <cmath>
#include <map>
#include <stdexcept>
#include <memory>
#include <mutex>

namespace stgcs {
namespace {
class LinearMinimum {
 public:
  explicit LinearMinimum(const Polyhedron& set)
      : x_(program_.NewContinuousVariables(set.ambient_dimension())),
        cost_(program_.AddLinearCost(Vector::Zero(x_.size()), 0, x_)) {
    set.AddPointInSetConstraints(&program_, x_);
  }
  double solve(const Vector& direction) {
    cost_.evaluator()->UpdateCoefficients(direction);
    const auto result = drake::solvers::Solve(program_, std::nullopt, Geometry::solver_options());
    if (result.is_success()) {
      return result.get_optimal_cost();
    }
    if (result.get_solution_result() == drake::solvers::SolutionResult::kInfeasibleConstraints)
      return kInfinity;
    std::string message = "Polyhedron LP failed: " + result.get_solver_id().name() + " " +
        drake::solvers::to_string(result.get_solution_result());
    if (result.get_solution_result() == drake::solvers::SolutionResult::kSolverSpecificError) {
      // An error is not evidence of emptiness. Check the unchanged LP with an
      // independent solver, accepting only a conclusive infeasibility result.
      const auto check = drake::solvers::ClarabelSolver().Solve(
          program_, std::nullopt, Geometry::solver_options());
      if (check.get_solution_result() == drake::solvers::SolutionResult::kInfeasibleConstraints)
        return kInfinity;
      message += "; independent check: " + drake::solvers::to_string(check.get_solution_result());
    }
    throw std::runtime_error(message);
  }

 private:
  drake::solvers::MathematicalProgram program_;
  drake::solvers::VectorXDecisionVariable x_;
  drake::solvers::Binding<drake::solvers::LinearCost> cost_;
};
}  // namespace

drake::solvers::SolverOptions Geometry::solver_options(double timeout) {
  drake::solvers::SolverOptions options;
  const auto id = drake::solvers::MosekSolver::id();
  options.SetOption(drake::solvers::CommonSolverOption::kPrintToConsole, 0);
  options.SetOption(id, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3);
  options.SetOption(id, "MSK_IPAR_INTPNT_SOLVE_FORM", 1);
  options.SetOption(id, "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3);
  options.SetOption(id, "MSK_IPAR_LOG", 2);
  options.SetOption(id, "MSK_IPAR_NUM_THREADS", 1);
  if (std::isfinite(timeout)) {
    options.SetOption(id, "MSK_DPAR_OPTIMIZER_MAX_TIME", std::max(1e-6, timeout));
    options.SetOption(id, "MSK_DPAR_MIO_MAX_TIME", std::max(1e-6, timeout));
  }
  return options;
}

bool Bounds::overlaps(const Bounds& other, double tolerance) const {
  return (lower.array() <= other.upper.array() + tolerance).all() &&
         (other.lower.array() <= upper.array() + tolerance).all();
}

Bounds Bounds::clip(const Vector& normal, double bound) const {
  Bounds result = *this;
  for (int i = 0; i < normal.size(); ++i) {
    if (normal[i] == 0) continue;
    double remaining = bound;
    for (int j = 0; j < normal.size(); ++j)
      if (j != i)
        remaining -= normal[j] * (normal[j] > 0 ? lower[j] : upper[j]);
    const double endpoint = remaining / normal[i];
    if (normal[i] > 0) result.upper[i] = std::min(upper[i], endpoint);
    else result.lower[i] = std::max(lower[i], endpoint);
  }
  return result;
}

Polyhedron Geometry::box(const Vector& lower, const Vector& upper) {
  if (lower.size() == 0 || lower.size() != upper.size() || !lower.allFinite() ||
      !upper.allFinite() || (lower.array() > upper.array()).any())
    throw std::invalid_argument("Invalid box bounds");
  return Polyhedron::MakeBox(lower, upper);
}

Polyhedron Geometry::point(const Vector& p) { return box(p, p); }

Polyhedron Geometry::extrude(const Polyhedron& spatial, double start, double end) {
  return spatial.CartesianProduct(box(Vector::Constant(1, start), Vector::Constant(1, end)));
}

Polyhedron Geometry::clip(const Polyhedron& set, const Vector& normal, double bound) {
  Matrix a(set.A().rows() + 1, set.ambient_dimension());
  Vector b(a.rows());
  std::map<std::vector<double>, int> rows;
  int count = 0;
  for (int i = 0; i <= set.A().rows(); ++i) {
    const Vector row = i == set.A().rows() ? normal : Vector(set.A().row(i).transpose());
    const double value = i == set.A().rows() ? bound : set.b()[i];
    auto [it, inserted] = rows.try_emplace(
        std::vector<double>(row.data(), row.data() + row.size()), count);
    if (inserted) {
      a.row(count) = row.transpose();
      b[count++] = value;
    } else {
      b[it->second] = std::min(b[it->second], value);
    }
  }
  // Exact coefficient equality only: retain first-row order and the tightest
  // bound. Nearly parallel and differently scaled inequalities stay distinct.
  return Polyhedron(a.topRows(count), b.head(count));
}

Polyhedron Geometry::time_slice(const Polyhedron& set, double start, double end) {
  Vector axis = Vector::Unit(set.ambient_dimension(), set.ambient_dimension() - 1);
  return clip(clip(set, axis, end), -axis, -start);
}

std::optional<std::pair<double, double>> Geometry::time_interval(
    const Polyhedron& set, const Vector& position, double start, double end) {
  const int d = set.ambient_dimension() - 1;
  if (position.size() != d || !position.allFinite() || start > end)
    throw std::invalid_argument("Invalid stationary time interval");
  for (int i = 0; i < set.A().rows(); ++i) {
    const double a = set.A()(i, d);
    const double b = set.b()[i] - set.A().row(i).head(d).dot(position);
    if (a > 0) end = std::min(end, b / a);
    else if (a < 0) start = std::max(start, b / a);
    else if (b < -kTolerance) return std::nullopt;
    if (start > end + kTolerance) return std::nullopt;
  }
  if (start > end) start = end = (start + end) / 2;
  return std::pair{start, end};
}

double Geometry::minimum(const Polyhedron& set, const Vector& cost) {
  return LinearMinimum(set).solve(cost);
}

std::optional<Bounds> Geometry::bounds(const Polyhedron& set,
                                     std::optional<double> minimum_width) {
  if (minimum_width && (!std::isfinite(*minimum_width) || *minimum_width < 0))
    throw std::invalid_argument("Invalid minimum region width");
  if (set.IsEmpty()) return std::nullopt;
  int n = set.ambient_dimension();
  Bounds result{Vector(n), Vector(n)};
  LinearMinimum program(set);
  for (int i = 0; i < n; ++i) {
    result.lower[i] = program.solve(Vector::Unit(n, i));
    if (result.lower[i] == kInfinity) return std::nullopt;
    result.upper[i] = -program.solve(-Vector::Unit(n, i));
    if (result.upper[i] == -kInfinity) return std::nullopt;
    if (minimum_width && result.upper[i] - result.lower[i] <= *minimum_width)
      return std::nullopt;
  }
  return result;
}

double Geometry::travel_time(const Polyhedron& from, const Vector& goal, double speed) {
  if (goal.size() != from.ambient_dimension() - 1 || !goal.allFinite() || !std::isfinite(speed) ||
      speed <= 0)
    throw std::invalid_argument("Invalid travel-time inputs");
  drake::solvers::MathematicalProgram program;
  auto x = program.NewContinuousVariables(from.ambient_dimension());
  auto t = program.NewContinuousVariables<1>();
  from.AddPointInSetConstraints(&program, x);
  program.AddBoundingBoxConstraint(0, kInfinity, t);
  for (int i = 0; i < goal.size(); ++i) {
    program.AddLinearConstraint(x[i] - goal[i] <= speed * t[0]);
    program.AddLinearConstraint(goal[i] - x[i] <= speed * t[0]);
  }
  program.AddLinearCost(t[0]);
  auto result = drake::solvers::Solve(program, std::nullopt, solver_options());
  if (result.is_success()) return std::max(0.0, result.get_optimal_cost());
  if (result.get_solution_result() == drake::solvers::SolutionResult::kInfeasibleConstraints)
    return kInfinity;
  throw std::runtime_error("Travel-time LP failed");
}

Matrix Geometry::affine_basis(const Polyhedron& set) {
  // Match HPolyhedronSampler: floating-point cdd vertices, then an absolute
  // 1e-9 singular-value cutoff. Do not classify facets with separate LPs.
  class CddRuntime {
   public:
    CddRuntime() { dd_set_global_constants(); }
    ~CddRuntime() { dd_free_global_constants(); }
    std::mutex mutex;
  };
  static CddRuntime runtime;
  Matrix vertices;
  {
    std::lock_guard lock(runtime.mutex);
    const int n = set.ambient_dimension();
    std::unique_ptr<dd_MatrixType, decltype(&dd_FreeMatrix)> input(
        dd_CreateMatrix(set.A().rows(), n + 1), dd_FreeMatrix);
    input->representation = dd_Inequality;
    input->numbtype = dd_Real;
    for (int i = 0; i < set.A().rows(); ++i) {
      dd_set_d(input->matrix[i][0], set.b()[i]);
      for (int j = 0; j < n; ++j) dd_set_d(input->matrix[i][j + 1], -set.A()(i, j));
    }
    dd_ErrorType error = dd_NoError;
    std::unique_ptr<dd_PolyhedraType, decltype(&dd_FreePolyhedra)> poly(
        dd_DDMatrix2Poly(input.get(), &error), dd_FreePolyhedra);
    if (error == dd_NoError && poly) {
      std::unique_ptr<dd_MatrixType, decltype(&dd_FreeMatrix)> output(
          dd_CopyGenerators(poly.get()), dd_FreeMatrix);
      if (output) {
        vertices.resize(n, output->rowsize);
        for (int i = 0; i < output->rowsize; ++i)
          for (int j = 0; j < n; ++j)
            vertices(j, i) = dd_get_d(output->matrix[i][j + 1]);
      }
    }
  }
  if (!vertices.cols()) {
    try {
      vertices = drake::geometry::optimization::VPolytope(set).vertices();
    } catch (const std::exception&) {
      return Matrix::Identity(set.ambient_dimension(), set.ambient_dimension());
    }
  }
  if (!vertices.cols()) return Matrix::Identity(set.ambient_dimension(), set.ambient_dimension());
  const Matrix offsets = vertices.colwise() - vertices.col(0);
  const Eigen::JacobiSVD<Matrix> svd(offsets, Eigen::ComputeThinU);
  const int rank = (svd.singularValues().array() > 1e-9).count();
  if (rank == set.ambient_dimension()) return Matrix::Identity(rank, rank);
  return svd.matrixU().leftCols(rank);
}

std::vector<Vector> Geometry::sample(const Polyhedron& set, int count, int seed) {
  drake::RandomGenerator generator(seed);
  std::vector<Vector> result;
  const Matrix basis = affine_basis(set);
  for (int i = 0; i < count; ++i) {
    try {
      result.push_back(basis.cols() == set.ambient_dimension()
          ? set.UniformSample(&generator)
          : set.UniformSample(&generator, 10, basis, 1e-6));
    } catch (const std::exception&) {
      const auto point = set.MaybeGetFeasiblePoint();
      if (!point) throw std::runtime_error("Cannot sample an empty polyhedron");
      result.push_back(*point);
    }
  }
  return result;
}

bool Geometry::cone_contains(const Vector& from, const Vector& to, double speed) {
  const int d = from.size() - 1;
  const double dt = to[d] - from[d];
  return dt >= -kTolerance &&
         (to.head(d) - from.head(d)).lpNorm<Eigen::Infinity>() <= speed * dt + kTolerance;
}

bool Geometry::cone_contains_set(const Vector& from, const Vector& arrival,
                                 const Polyhedron& interface, double speed) {
  return ArrivalSet(interface, arrival, speed).contained_by(from);
}

ArrivalSet::ArrivalSet(const Polyhedron& interface, const Vector& arrival, double speed)
    : arrival_(arrival),
      speed_(speed),
      reachable_(Geometry::clip(interface, -Vector::Unit(arrival.size(), arrival.size() - 1),
                                -arrival[arrival.size() - 1])),
      support_(
          Vector::Constant(2 * (arrival.size() - 1), std::numeric_limits<double>::quiet_NaN())) {}

bool ArrivalSet::contained_by(const Vector& from) const {
  if (!Geometry::cone_contains(from, arrival_, speed_)) return false;
  if (!empty_) empty_ = reachable_.IsEmpty();
  if (*empty_) return true;
  if (reachable_.PointInSet(from, kTolerance)) return false;
  const int d = from.size() - 1;
  // Maximizing each cone halfspace is exact and also handles flat interfaces.
  for (int i = 0; i < d; ++i)
    for (int sign : {-1, 1}) {
      Vector a = sign * Vector::Unit(d + 1, i);
      a[d] = -speed_;
      double& support = support_[2 * i + (sign == 1)];
      if (std::isnan(support)) support = -Geometry::minimum(reachable_, -a);
      if (support > a.dot(from) + kTolerance) return false;
    }
  return true;
}
}  // namespace stgcs
