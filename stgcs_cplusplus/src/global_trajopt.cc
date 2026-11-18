#include "stgcs/global_trajopt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <Eigen/Sparse>
#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/mosek_solver.h>

namespace stgcs::trajopt {
namespace {
using Clock = std::chrono::steady_clock;
using drake::solvers::MathematicalProgram;
using drake::solvers::MosekSolver;
using drake::solvers::VectorXDecisionVariable;

// A sparse constraint row, with the same optional softening policy as the demos.
struct Row {
  std::vector<int> indices;
  std::vector<double> coefficients;
  double lower, upper;
  bool soft{false};
};

class WindowQP {
 public:
  explicit WindowQP(int count, int samples, int dimension)
      : agents(count), times(samples), dim(dimension), variables(count * samples * dimension) {}
  int index(int agent, int time, int axis) const { return (agent * times + time) * dim + axis; }
  void add(std::vector<int> indices, std::vector<double> coefficients,
           double lower, double upper, bool soft = false) {
    rows.push_back({std::move(indices), std::move(coefficients), lower, upper, soft});
  }
  void spatial(int agent, int time, const Vector& normal, double lower,
               double upper, bool soft = false) {
    std::vector<int> indices;
    std::vector<double> coefficients;
    for (int axis = 0; axis < dim; ++axis) {
      indices.push_back(index(agent, time, axis));
      coefficients.push_back(normal[axis]);
    }
    add(std::move(indices), std::move(coefficients), lower, upper, soft);
  }
  static void square(Matrix* Q, Vector* b, const std::vector<int>& indices,
                     const std::vector<double>& coefficients, double offset, double weight,
                     const Matrix& basis) {
    Vector direction = Vector::Zero(basis.cols());
    for (size_t i = 0; i < indices.size(); ++i)
      direction += coefficients[i] * basis.row(indices[i]).transpose();
    *b += (2*weight*offset) * direction;
    *Q += (2*weight) * direction * direction.transpose();
  }
  // Displacements are integrals of velocity increments. Within a span bounded
  // by fixed states, eliminate the velocity on its longest interval to enforce
  // zero total displacement. No division by a tiny interval is required.
  static Matrix position_basis(const std::vector<double>& t, const std::vector<int>& fixed) {
    const int n = t.size();
    Matrix basis = Matrix::Zero(n, n-fixed.size());
    int column = 0;
    if (fixed.empty()) {
      basis.col(column++).setOnes();
    } else {
      for (int j = 1; j <= fixed.front(); ++j)
        basis.col(column++).head(j).setConstant(-(t[j]-t[j-1]));
      for (size_t span = 1; span < fixed.size(); ++span) {
        int left = fixed[span-1], right = fixed[span], pivot = left+1;
        for (int j = left+1; j <= right; ++j)
          if (t[j]-t[j-1] > t[pivot]-t[pivot-1]) pivot = j;
        for (int j = left+1; j <= right; ++j) if (j != pivot) {
          for (int k = left+1; k < right; ++k)
            basis(k,column) = (t[j]-t[j-1]) * (int(j <= k)-int(pivot <= k));
          ++column;
        }
      }
    }
    int last = fixed.empty() ? 0 : fixed.back();
    for (int j = last+1; j < n; ++j)
      basis.col(column++).tail(n-j).setConstant(t[j]-t[j-1]);
    return basis;
  }
  static Matrix local_basis(const std::vector<double>& t, const std::vector<int>& fixed, double min_dt) {
    Matrix basis = Matrix::Zero(t.size(), t.size()-fixed.size());
    int begin = 0, column = 0;
    while (begin < int(t.size())) {
      int end = begin+1;
      while (end < int(t.size()) && t[end]-t[end-1] < min_dt) ++end;
      std::vector<int> local_fixed;
      for (int k : fixed) if (k >= begin && k < end) local_fixed.push_back(k-begin);
      Matrix block = position_basis({t.begin()+begin,t.begin()+end},local_fixed);
      basis.block(begin,column,end-begin,block.cols()) = block;
      column += block.cols(); begin = end;
    }
    return basis;
  }
  void parameterize(const std::vector<double>& t, double min_dt) {
    std::vector<std::set<int>> fixed(agents);
    for (const auto& row : rows)
      if (row.indices.size() == 1 && row.lower == 0 && row.upper == 0)
        fixed[row.indices[0]/(times*dim)].insert((row.indices[0]/dim)%times);
    variables = 0;
    for (int a = 0; a < agents; ++a) {
      // Local velocity increments condition short intervals without making every
      // collision row depend on all earlier velocities in the window.
      Matrix temporal = local_basis(t, {fixed[a].begin(), fixed[a].end()}, min_dt);
      Matrix basis = Matrix::Zero(times*dim, temporal.cols()*dim);
      for (int k = 0; k < times; ++k) for (int j = 0; j < temporal.cols(); ++j)
        for (int d = 0; d < dim; ++d) basis(k*dim+d,j*dim+d) = temporal(k,j);
      offsets.push_back(variables);
      variables += basis.cols();
      bases.push_back(std::move(basis));
    }
    for (auto& row : rows) {
      std::map<int, double> reduced;
      for (size_t k = 0; k < row.indices.size(); ++k) {
        int a = row.indices[k]/(times*dim), local = row.indices[k]%(times*dim);
        for (int j = 0; j < bases[a].cols(); ++j)
          if (bases[a](local,j) != 0)
            reduced[offsets[a]+j] += row.coefficients[k]*bases[a](local,j);
      }
      row.indices.clear(); row.coefficients.clear();
      for (auto [index, value] : reduced) if (value != 0) {
        row.indices.push_back(index); row.coefficients.push_back(value);
      }
    }
  }
  Vector displacement(const Vector& velocity) const {
    Vector result(agents*times*dim);
    for (int a = 0; a < agents; ++a)
      result.segment(a*times*dim,times*dim) = bases[a]*velocity.segment(offsets[a],bases[a].cols());
    return result;
  }
  void objectives(const std::vector<double>& t, const Positions& reference, const Config& c) {
    parameterize(t, c.min_sample_dt);
    for (int a = 0; a < agents; ++a) {
      const auto& basis = bases[a];
      Matrix Q = (2*c.displacement_weight+1e-10)*basis.transpose()*basis;
      Vector b = Vector::Zero(basis.cols());
      for (int k = 2; k < times - 1; ++k) {
        double curr = t[k + 1] - t[k - 1], prev = t[k] - t[k - 2];
        for (int d = 0; d < dim; ++d) {
          double offset = (reference[a](k + 1, d) - reference[a](k - 1, d)) / curr -
                          (reference[a](k, d) - reference[a](k - 2, d)) / prev;
          square(&Q, &b, {(k+1)*dim+d, (k-1)*dim+d, k*dim+d, (k-2)*dim+d},
                 {1/curr, -1/curr, -1/prev, 1/prev}, offset, c.velocity_gradient_weight, basis);
        }
      }
      for (int k = 1; k < times - 1; ++k) {
        double prev = t[k] - t[k - 1], next = t[k + 1] - t[k];
        for (int d = 0; d < dim; ++d)
          square(&Q, &b, {(k-1)*dim+d, k*dim+d, (k+1)*dim+d},
                 {1/prev, -1/prev-1/next, 1/next}, 0, c.displacement_jitter_weight, basis);
      }
      quadratic.push_back(std::move(Q));
      linear.push_back(std::move(b));
    }
    // Equilibrate the reduced variables by objective curvature. Short intervals
    // otherwise give vastly different Hessian diagonals, even in the local
    // velocity basis. Apply x = D*z to the objective, constraints, and output
    // basis together, preserving the problem and all physical tolerances.
    Vector scales(variables);
    for (int a = 0; a < agents; ++a) {
      Vector scale = quadratic[a].diagonal().unaryExpr([](double value) {
        return value > 0 ? 1/std::sqrt(value) : 1.;
      });
      scales.segment(offsets[a], scale.size()) = scale;
      quadratic[a] = scale.asDiagonal()*quadratic[a]*scale.asDiagonal();
      linear[a] = scale.asDiagonal()*linear[a];
      bases[a] = bases[a]*scale.asDiagonal();
    }
    for (auto& row : rows) for (size_t k = 0; k < row.indices.size(); ++k)
      row.coefficients[k] *= scales[row.indices[k]];
  }
  Vector solve(const Config& c, std::string* message) const {
    std::string hard_message;
    for (int attempt = 0; attempt < 2; ++attempt) {
      MathematicalProgram program;
      auto x = program.NewContinuousVariables(variables, "velocity_increment");
      std::vector<int> slack_for_row(rows.size(), -1);
      int slack_count = 0;
      if (attempt) {
        for (size_t r = 0; r < rows.size(); ++r)
          if (rows[r].soft && (std::isfinite(rows[r].lower) || std::isfinite(rows[r].upper)))
            slack_for_row[r] = slack_count++;
        if (c.constraint_slack_penalty <= 0 || !slack_count)
          throw std::runtime_error("Global trajectory QP failed: " + hard_message);
      }
      auto slack = program.NewContinuousVariables(slack_count, "constraint_slack");
      VectorXDecisionVariable all(variables + slack_count);
      all << x, slack;
      std::vector<Eigen::Triplet<double>> entries;
      std::vector<double> lower, upper;
      for (size_t r = 0; r < rows.size(); ++r) {
        const auto& row = rows[r];
        if (row.indices.empty() && row.lower <= c.solver_eps_abs && row.upper >= -c.solver_eps_abs)
          continue;
        // Split a softened two-sided row so one nonnegative slack relaxes both bounds.
        for (int side = 0; side < (slack_for_row[r] >= 0 ? 2 : 1); ++side) {
          double lo = row.lower, hi = row.upper;
          // Allocate half the accepted accuracy to constraint roundoff and reserve
          // the remainder for solver/reconstruction error; physical checks follow.
          if (lo != hi) { lo -= .5*c.solver_eps_abs; hi += .5*c.solver_eps_abs; }
          if (slack_for_row[r] >= 0) {
            if (side == 0 && !std::isfinite(lo)) continue;
            if (side == 1 && !std::isfinite(hi)) continue;
            if (side == 0) hi = kInfinity; else lo = -kInfinity;
            entries.emplace_back(lower.size(), variables + slack_for_row[r], side ? -1 : 1);
          }
          for (size_t k = 0; k < row.indices.size(); ++k)
            if (std::abs(row.coefficients[k]) > 1e-15)
              entries.emplace_back(lower.size(), row.indices[k], row.coefficients[k]);
          lower.push_back(lo); upper.push_back(hi);
        }
      }
      Eigen::SparseMatrix<double> A(lower.size(), all.size());
      A.setFromTriplets(entries.begin(), entries.end());
      if (!variables && !slack_count) {
        if (lower.empty()) { *message = "FixedStates"; return Vector::Zero(agents*times*dim); }
        hard_message = "Infeasible fixed states";
        continue;
      }
      program.AddLinearConstraint(A, Eigen::Map<Vector>(lower.data(), lower.size()),
                                  Eigen::Map<Vector>(upper.data(), upper.size()), all);
      // Independent robot blocks avoid the old dense all-robot Hessian.
      for (int a = 0; a < agents; ++a) {
        if (!bases[a].cols()) continue;
        VectorXDecisionVariable block = x.segment(offsets[a], bases[a].cols());
        program.AddQuadraticCost(quadratic[a], linear[a], block, true);
      }
      if (slack_count) {
        program.AddBoundingBoxConstraint(Vector::Zero(slack_count),
                                         Vector::Constant(slack_count, c.constraint_slack_limit), slack);
        for (int i = 0; i < slack_count; ++i) {
          VectorXDecisionVariable one = slack.segment(i, 1);
          program.AddQuadraticCost(Matrix::Constant(1, 1, 2*c.constraint_slack_penalty),
                                   Vector::Zero(1), one, true);
        }
      }
      program.SetInitialGuess(all, Vector::Zero(all.size()));
      drake::solvers::SolverOptions options;
      options.SetOption(drake::solvers::CommonSolverOption::kPrintToConsole,
                        int(c.solver_print_to_console));
      options.SetOption(MosekSolver::id(), "MSK_IPAR_NUM_THREADS", 1);
      options.SetOption(MosekSolver::id(), "MSK_IPAR_INTPNT_MAX_ITERATIONS", c.solver_max_iter);
      for (const std::string family : {"QO", "CO"}) {
        options.SetOption(MosekSolver::id(), "MSK_DPAR_INTPNT_"+family+"_TOL_PFEAS", std::min(1e-9,.1*c.solver_eps_abs));
        options.SetOption(MosekSolver::id(), "MSK_DPAR_INTPNT_"+family+"_TOL_DFEAS", std::min(1e-9,.1*c.solver_eps_abs));
        options.SetOption(MosekSolver::id(), "MSK_DPAR_INTPNT_"+family+"_TOL_REL_GAP", std::min(1e-9,.1*c.solver_eps_rel));
        options.SetOption(MosekSolver::id(), "MSK_DPAR_INTPNT_"+family+"_TOL_NEAR_REL", 1.);
      }
      auto result = MosekSolver().Solve(program, std::nullopt, options);
      std::ostringstream status;
      status << result.get_solution_result();
      const auto& details = result.get_solver_details<MosekSolver>();
      status << "; solver=MOSEK; optimizer_time=" << details.optimizer_time
             << "; solution_status=" << details.solution_status << "; response_code=" << details.rescode;
      if (result.is_success()) {
        *message = status.str();
        if (attempt) {
          Vector values = result.GetSolution(slack).cwiseMax(0);
          std::ostringstream detail;
          detail << "; constraint_soft_retry_from=" << hard_message
                 << "; constraint_soft_rows=" << slack_count
                 << "; constraint_slack_max=" << values.maxCoeff()
                 << "; constraint_slack_l1=" << values.sum();
          *message += detail.str();
        }
        return displacement(result.GetSolution(x));
      }
      if (attempt) throw std::runtime_error("Global trajectory QP failed: " + status.str() +
                                             " after soft retry from " + hard_message);
      hard_message = status.str();
    }
    throw std::logic_error("Unreachable QP solve");
  }
  int agents, times, dim, variables;
  std::vector<Row> rows;
  std::vector<Matrix> quadratic;
  std::vector<Vector> linear;
  std::vector<Matrix> bases;
  std::vector<int> offsets;
};

}  // namespace

void Optimizer::validate(const Problem& p) {
  if (p.trajectories.empty()) throw std::invalid_argument("Expected at least one trajectory");
  if (p.trajectories.front().empty()) throw std::invalid_argument("Empty trajectory");
  int dim = p.trajectories.front().start().size() - 1;
  for (const auto& path : p.trajectories) {
    if (path.empty() || path.start().size() != dim + 1)
      throw std::invalid_argument("Inconsistent trajectory dimensions");
    for (const auto& s : path.segments())
      // Saved demo trajectories admit solver residuals up to 1e-6, as does
      // trajectory_from_viewer_json; refinement checks nonpositive-duration motion.
      if (s.end_time() < s.start_time()-1e-6) throw std::invalid_argument("Reversed segment time");
  }
  for (const auto& cell : p.cells)
    if (cell.ambient_dimension() != dim + 1) throw std::invalid_argument("Cell dimension mismatch");
  const auto& c = p.config;
  for (double value : {p.speed, c.sample_dt, c.min_sample_dt, c.solver_eps_abs, c.solver_eps_rel})
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Expected finite positive parameter");
  for (double value : {p.radius, c.velocity_gradient_weight, c.displacement_weight,
                       c.displacement_jitter_weight, c.spatial_containment_tolerance,
                       c.spatial_containment_soft_margin, c.constraint_slack_penalty})
    if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Expected finite nonnegative parameter");
  if (std::isnan(c.constraint_slack_limit) || c.constraint_slack_limit <= 0 || c.solver_max_iter <= 0)
    throw std::invalid_argument("Invalid QP settings");
  if (p.environment && (p.environment->lower.size() != dim || p.environment->upper.size() != dim))
    throw std::invalid_argument("Environment dimension mismatch");
}

int Optimizer::segment_index(const Trajectory& path, double time) {
  if (time <= path.start().tail(1)[0]) return 0;
  if (time >= path.end().tail(1)[0]) return path.segments().size()-1;
  // Use actual knot times: rounding can select the preceding cell after a short segment.
  for (size_t i = 0; i < path.segments().size(); ++i) {
    const auto& s = path.segments()[i];
    if (time >= s.start_time() && time <= s.end_time()) return i;
  }
  throw std::invalid_argument("Sample time lies outside trajectory segments");
}

Vector Optimizer::state_at(const Trajectory& path, double time) {
  if (time <= path.start().tail(1)[0]) return path.start();
  if (time >= path.end().tail(1)[0]) return path.end();
  return state_at(path.segments()[segment_index(path, time)], time);
}

Vector Optimizer::state_at(const Segment& s, double time) {
  const double dt = s.end_time()-s.start_time();
  if (dt == 0) return s.start;
  double fraction = std::clamp((time-s.start_time())/dt, 0., 1.);
  return s.start*(1-fraction)+s.end*fraction;
}

std::vector<double> Optimizer::sample_times(const Problem& p) {
  validate(p);
  double begin = kInfinity, end = -kInfinity;
  for (const auto& path : p.trajectories) {
    begin = std::min(begin, path.start().tail(1)[0]);
    end = std::max(end, path.end().tail(1)[0]);
  }
  int count = std::max(1, int(std::ceil((end-begin)/p.config.sample_dt-1e-12)));
  double scale = std::pow(10., p.config.round_time_decimals);
  std::set<double> samples{std::nearbyint(begin*scale)/scale, std::nearbyint(end*scale)/scale};
  if (p.config.include_trajectory_knot_times)
    for (const auto& path : p.trajectories)
      for (const auto& s : path.segments()) {
        samples.insert(std::nearbyint(s.start_time()*scale)/scale);
        samples.insert(std::nearbyint(s.end_time()*scale)/scale);
      }
  double interval = std::min(p.config.min_sample_dt, p.config.sample_dt);
  for (int i = 1; i < count; ++i) {
    double t = std::nearbyint((begin+(end-begin)*i/count)*scale)/scale;
    auto next = samples.lower_bound(t);
    if (next != samples.end() && *next-t < interval) continue;
    if (next != samples.begin() && t-*std::prev(next) < interval) continue;
    samples.insert(t);
  }
  std::vector<double> times(samples.begin(), samples.end());
  if (times.size() < 2) throw std::invalid_argument("Need at least two distinct sample times");
  return times;
}

Positions Optimizer::sample_positions(const std::vector<Trajectory>& paths, const std::vector<double>& times) {
  Positions positions;
  for (const auto& path : paths) {
    int dim = path.start().size() - 1;
    Matrix values(times.size(), dim);
    for (size_t k = 0; k < times.size(); ++k) values.row(k) = state_at(path, times[k]).head(dim).transpose();
    positions.push_back(std::move(values));
  }
  return positions;
}

int Optimizer::containing_cell(const Problem& p, const Trajectory& path, int index) {
  const auto& segment = path.segments()[index];
  if (segment.vertex >= 0 && segment.vertex < int(p.cells.size())) return segment.vertex;
  for (size_t k = 0; k < p.cells.size(); ++k)
    if (p.cells[k].PointInSet(segment.start, 1e-7) && p.cells[k].PointInSet(segment.end, 1e-7)) return k;
  throw std::runtime_error("No ST-GCS cell contains reference segment");
}

Vector Optimizer::closest_point(const Vector& point, const Vector& start, const Vector& end) {
  Vector direction = end-start;
  double norm = direction.squaredNorm();
  return norm <= 1e-12 ? start : Vector(start + std::clamp((point-start).dot(direction)/norm, 0., 1.)*direction);
}

std::pair<Matrix, Vector> Optimizer::inflated_polygon(const Obstacle& o, double radius, double margin) {
  Matrix A = o.A; Vector b = o.b;
  for (int i = 0; i < A.rows(); ++i) {
    double norm = A.row(i).norm();
    if (norm <= 1e-12) throw std::invalid_argument("Near-zero obstacle face normal");
    A.row(i) /= norm;
    b[i] = b[i]/norm + radius*A.row(i).lpNorm<1>() + margin;
  }
  return {A, b};
}

std::pair<Vector, double> Optimizer::separation(const Obstacle& o, const Vector& start,
                                               const Vector& end, double radius, double margin) {
  if (o.center.size()) {
    Vector delta = closest_point(o.center, start, end)-o.center;
    double distance = delta.norm(), required = o.radius+radius+margin;
    if (distance <= required+1e-12) throw std::runtime_error("Colliding reference sphere segment");
    Vector normal = delta/distance;
    return {normal, normal.dot(o.center)+required};
  }
  if (start.size() != 2) throw std::invalid_argument("Polygon avoidance requires 2D");
  auto [A, b] = inflated_polygon(o, radius, margin);
  std::vector<Vector> vertices;
  for (int i = 0; i < A.rows(); ++i) for (int j = i+1; j < A.rows(); ++j) {
    Eigen::Matrix2d pair; pair << A.row(i), A.row(j);
    if (std::abs(pair.determinant()) <= 1e-12) continue;
    Vector point = pair.inverse()*Eigen::Vector2d(b[i], b[j]);
    if ((A*point-b).maxCoeff() > 1e-9) continue;
    bool duplicate = false;
    for (const auto& v : vertices) if ((v-point).norm() <= 1e-9) duplicate = true;
    if (!duplicate) vertices.push_back(point);
  }
  if (vertices.size() < 3) throw std::runtime_error("Cannot recover inflated polygon");
  Vector center = Vector::Zero(2);
  for (const auto& v : vertices) center += v;
  center /= vertices.size();
  std::sort(vertices.begin(), vertices.end(), [&](const auto& a, const auto& b) {
    return std::atan2(a[1]-center[1], a[0]-center[0]) < std::atan2(b[1]-center[1], b[0]-center[0]);
  });
  double distance = kInfinity; Vector best_path, best_obstacle;
  for (size_t k = 0; k < vertices.size(); ++k) {
    for (const Vector* point : {&start, &end}) {
      Vector near = closest_point(*point, vertices[k], vertices[(k+1)%vertices.size()]);
      if ((*point-near).norm() < distance) {
        distance = (*point-near).norm(); best_path = *point; best_obstacle = near;
      }
    }
    Vector near = closest_point(vertices[k], start, end);
    if ((near-vertices[k]).norm() < distance) {
      distance = (near-vertices[k]).norm(); best_path = near; best_obstacle = vertices[k];
    }
  }
  if (distance <= 1e-10) {
    Vector scores = (A*start-b).cwiseMin(A*end-b);
    Eigen::Index face; double score = scores.maxCoeff(&face);
    if (score < -1e-7) throw std::runtime_error("Colliding reference polygon segment");
    return {A.row(face).transpose(), b[face]};
  }
  Vector normal = (best_path-best_obstacle)/distance;
  return {normal, normal.dot(best_obstacle)};
}

bool Optimizer::cells_cover(const std::vector<Polyhedron>& cells, const Vector& start,
                            const Vector& end, double tol) {
  std::vector<std::pair<double, double>> intervals;
  for (const auto& cell : cells) {
    Vector delta = cell.A()*(end-start), rhs = cell.b()-cell.A()*start;
    double lo = 0, hi = 1;
    for (int i = 0; i < rhs.size(); ++i) {
      if (std::abs(delta[i]) <= tol) { if (rhs[i]+tol < 0) {hi = -1; break;} }
      else if (delta[i] > 0) hi = std::min(hi, (rhs[i]+tol)/delta[i]);
      else lo = std::max(lo, (rhs[i]+tol)/delta[i]);
    }
    if (lo <= hi+tol && hi >= -tol && lo <= 1+tol)
      intervals.emplace_back(std::max(lo, 0.), std::min(hi, 1.));
  }
  std::sort(intervals.begin(), intervals.end());
  double covered = 0;
  for (const auto& [lo, hi] : intervals) {
    if (hi < covered-tol) continue;
    if (lo > covered+tol) return false;
    covered = std::max(covered, hi);
    if (covered >= 1-tol) return true;
  }
  return covered >= 1-tol;
}

bool Optimizer::environment_segment_free(const Environment& env, const Vector& start,
                                         const Vector& end, double radius, double tol) {
  if ((start-env.lower).minCoeff() < -tol || (end-env.lower).minCoeff() < -tol ||
      (start-env.upper).maxCoeff() > tol || (end-env.upper).maxCoeff() > tol) return false;
  for (const auto& o : env.obstacles) {
    if (o.center.size()) {
      if ((closest_point(o.center, start, end)-o.center).norm() <= o.radius+radius) return false;
    } else {
      // Match swept axis-aligned robot boxes against the actual polygon,
      // including Minkowski-sum faces that a face-offset approximation misses.
      int dim = start.size(), faces = o.A.rows();
      Matrix A = Matrix::Zero(faces+2*dim+2, dim+1);
      Vector b(faces+2*dim+2);
      A.topLeftCorner(faces, dim) = o.A; b.head(faces) = o.b;
      for (int d = 0; d < dim; ++d) {
        A(faces+2*d,d) = 1; A(faces+2*d,dim) = -(end[d]-start[d]);
        b[faces+2*d] = start[d]+radius;
        A(faces+2*d+1,d) = -1; A(faces+2*d+1,dim) = end[d]-start[d];
        b[faces+2*d+1] = radius-start[d];
      }
      A(faces+2*dim,dim) = 1; b[faces+2*dim] = 1;
      A(faces+2*dim+1,dim) = -1; b[faces+2*dim+1] = 0;
      if (Polyhedron(A,b).MaybeGetFeasiblePoint().has_value()) return false;
    }
  }
  return !env.obstacles.empty() || cells_cover(env.regions, start, end, tol);
}

bool Optimizer::environment_free(const Problem& p, const std::vector<Trajectory>& paths, double tol) {
  for (const auto& path : paths) for (const auto& s : path.segments()) {
    int dim = s.start.size()-1;
    if (p.environment && !environment_segment_free(*p.environment, s.start.head(dim), s.end.head(dim), p.radius, tol))
      return false;
    if (!p.cells.empty() && !cells_cover(p.cells, s.start, s.end, tol)) return false;
  }
  return true;
}

bool Optimizer::pairwise_free(const std::vector<Trajectory>& paths, double radius,
                             const std::vector<bool>& stays, double tol) {
  if (paths.size() != stays.size()) throw std::invalid_argument("Goal-stay count mismatch");
  double begin = kInfinity, end = -kInfinity;
  for (const auto& path : paths) {
    begin = std::min(begin, path.start().tail(1)[0]); end = std::max(end, path.end().tail(1)[0]);
  }
  for (size_t i = 0; i < paths.size(); ++i) for (size_t j = i+1; j < paths.size(); ++j)
    if (Collision::check(paths[i], paths[j], 2*radius, begin, end,
                         {true, stays[i]}, {true, stays[j]}, tol)) return false;
  return true;
}

bool Optimizer::disk_pairwise_free(const std::vector<Trajectory>& paths, double radius,
                                  const std::vector<bool>& stays, double tol) {
  if (paths.size() != stays.size() || radius < 0 || tol < 0)
    throw std::invalid_argument("Invalid disk collision-check parameters");
  double begin = kInfinity, end = -kInfinity;
  for (const auto& path : paths) {
    begin = std::min(begin, path.start().tail(1)[0]);
    end = std::max(end, path.end().tail(1)[0]);
  }
  std::vector<std::vector<Segment>> occupied;
  for (size_t a = 0; a < paths.size(); ++a) {
    std::vector<Segment> segments;
    if (begin < paths[a].start().tail(1)[0]) {
      Vector start = paths[a].start(); start.tail(1)[0] = begin;
      segments.push_back({start,paths[a].start(),-1});
    }
    segments.insert(segments.end(),paths[a].segments().begin(),paths[a].segments().end());
    if (stays[a] && end > paths[a].end().tail(1)[0]) {
      Vector goal = paths[a].end(); goal.tail(1)[0] = end;
      segments.push_back({paths[a].end(),goal,-1});
    }
    occupied.push_back(std::move(segments));
  }
  for (size_t i = 0; i < paths.size(); ++i) for (size_t j = i+1; j < paths.size(); ++j) {
    size_t a = 0, b = 0;
    while (a < occupied[i].size() && b < occupied[j].size()) {
      const auto& first = occupied[i][a]; const auto& second = occupied[j][b];
      double lo = std::max(first.start_time(), second.start_time());
      double hi = std::min(first.end_time(), second.end_time());
      if (lo <= hi) {
        int dim = first.start.size()-1;
        Vector r0 = state_at(first,lo).head(dim)-state_at(second,lo).head(dim);
        Vector delta = state_at(first,hi).head(dim)-state_at(second,hi).head(dim)-r0;
        double s = delta.squaredNorm() == 0 ? 0 : std::clamp(-r0.dot(delta)/delta.squaredNorm(),0.,1.);
        if ((r0+s*delta).norm()+tol < 2*radius) return false;
      }
      if (first.end_time() <= second.end_time()) ++a;
      else ++b;
    }
  }
  return true;
}

std::vector<Trajectory> Optimizer::reference_trajectories(const Problem& p) {
  validate(p);
  Problem uniform = p;
  uniform.config.include_trajectory_knot_times = false;
  uniform.config.min_sample_dt = std::numeric_limits<double>::min();
  const auto grid = sample_times(uniform);
  std::vector<Trajectory> paths;
  for (const auto& path : p.trajectories) {
    double begin = path.start().tail(1)[0], end = path.end().tail(1)[0];
    std::set<double> times{begin, end};
    for (double t : grid) if (t >= begin-1e-9 && t <= end+1e-9) times.insert(t);
    // Subdivide each original segment without removing bends or changing timing.
    std::vector<Segment> segments;
    for (const auto& segment : path.segments()) {
      if (!segments.empty() && segments.back().end_time() < segment.start_time()) {
        Vector previous = segments.back().end;
        if (previous.head(previous.size()-1) != segment.start.head(segment.start.size()-1))
          throw std::invalid_argument("Reference has a spatially discontinuous time gap");
        segments.push_back({previous,segment.start,segment.vertex});
      }
      if (segment.end_time() <= segment.start_time()) {
        if (segment.start != segment.end) throw std::invalid_argument("Reference contains a nonpositive-duration motion");
        continue;
      }
      Vector start = segment.start;
      for (double time : times) {
        if (time <= segment.start_time() || time >= segment.end_time()) continue;
        double fraction = (time-segment.start_time())/(segment.end_time()-segment.start_time());
        Vector end = segment.start*(1-fraction)+segment.end*fraction;
        segments.push_back({start, end, segment.vertex}); start = end;
      }
      segments.push_back({start, segment.end, segment.vertex});
    }
    paths.emplace_back(std::move(segments));
  }
  return paths;
}

Positions Optimizer::solve_window(const Problem& p, const std::vector<double>& times,
                                  const Positions& reference, bool fix_window, std::string* message) {
  const auto& c = p.config;
  int agents = reference.size(), count = times.size(), dim = reference.front().cols();
  WindowQP qp(agents, count, dim);
  // Use conservative displacement bounds to prune pairwise constraints.
  Positions displacement_bounds(agents, Matrix::Constant(count, dim, kInfinity));
  if (fix_window) for (int a = 0; a < agents; ++a) {
    for (int fixed : {0, count-1}) {
      displacement_bounds[a].row(fixed).setZero();
      Vector bound = Vector::Zero(dim);
      for (int k = fixed+1; k < count; ++k) {
        bound += Vector::Constant(dim, p.speed*(times[k]-times[k-1])) +
                 (reference[a].row(k)-reference[a].row(k-1)).cwiseAbs().transpose();
        displacement_bounds[a].row(k) = displacement_bounds[a].row(k).cwiseMin(bound.transpose());
      }
      bound.setZero();
      for (int k = fixed-1; k >= 0; --k) {
        bound += Vector::Constant(dim, p.speed*(times[k+1]-times[k])) +
                 (reference[a].row(k+1)-reference[a].row(k)).cwiseAbs().transpose();
        displacement_bounds[a].row(k) = displacement_bounds[a].row(k).cwiseMin(bound.transpose());
      }
    }
  }
  for (int a = 0; a < agents; ++a) {
    const auto& path = p.trajectories[a];
    for (int k = 0; k < count; ++k) {
      bool terminal = c.fix_terminal_states &&
                      (times[k] <= path.start().tail(1)[0]+1e-9 || times[k] >= path.end().tail(1)[0]-1e-9);
      if (terminal || (fix_window && (k == 0 || k == count-1)))
        for (int d = 0; d < dim; ++d) qp.add({qp.index(a,k,d)}, {1}, 0, 0);
    }
    for (int k = 0; k < count-1; ++k) {
      double dt = times[k+1]-times[k];
      if (dt <= 0) throw std::invalid_argument("Sample times must increase");
      for (int d = 0; d < dim; ++d) {
        double offset = (reference[a](k+1,d)-reference[a](k,d))/dt;
        qp.add({qp.index(a,k+1,d), qp.index(a,k,d)}, {1/dt, -1/dt}, -p.speed-offset, p.speed-offset);
      }
      if (!p.cells.empty()) {
        const auto& cell = p.cells[containing_cell(p, path, segment_index(path, (times[k]+times[k+1])/2))];
        for (int t : {k, k+1}) for (int row = 0; row < cell.A().rows(); ++row) {
          Vector normal = cell.A().row(row).head(dim).transpose();
          if (normal.norm() <= 1e-12) continue;
          double bound = cell.b()[row]-cell.A()(row,dim)*times[t]-normal.dot(reference[a].row(t))+
                         c.spatial_containment_tolerance;
          qp.spatial(a, t, normal, -kInfinity, bound, bound <= c.spatial_containment_soft_margin);
        }
      } else if (p.environment && !p.environment->obstacles.empty()) {
        for (const auto& obstacle : p.environment->obstacles) {
          auto [normal, bound] = separation(obstacle, reference[a].row(k), reference[a].row(k+1), p.radius, c.clearance_margin);
          for (int t : {k,k+1})
            qp.spatial(a, t, normal, bound-normal.dot(reference[a].row(t)), kInfinity, true);
        }
      }
    }
    if (p.cells.empty() && p.environment && !p.environment->obstacles.empty())
      for (int k = 0; k < count; ++k) for (int d = 0; d < dim; ++d)
        qp.add({qp.index(a,k,d)}, {1}, p.environment->lower[d]-reference[a](k,d),
               p.environment->upper[d]-reference[a](k,d));
  }
  double min_sep = 2*p.radius+c.clearance_margin;
  for (int k = 0; k < count; ++k) for (int i = 0; i < agents; ++i) for (int j = i+1; j < agents; ++j) {
    if (c.include_interval_collision_constraints && k == count-1) continue;
    int end = k + int(c.include_interval_collision_constraints);
    Vector diff = (reference[i].row(k)-reference[j].row(k)).transpose();
    Vector next = (reference[i].row(end)-reference[j].row(end)).transpose();
    Vector delta = next-diff;
    double s = delta.squaredNorm() == 0 ? 0 : std::clamp(-diff.dot(delta)/delta.squaredNorm(), 0., 1.);
    Vector relative = (1-s)*diff+s*next;
    double closing = std::max(displacement_bounds[i].row(k).norm(), displacement_bounds[i].row(end).norm())+
                     std::max(displacement_bounds[j].row(k).norm(), displacement_bounds[j].row(end).norm());
    if (fix_window && relative.norm() > min_sep+closing+1e-12) continue;
    Vector normal = relative.norm() > 1e-12 ? relative : diff;
    if (normal.norm() <= 1e-12) {
      for (int offset = 1; offset < count && normal.norm() <= 1e-12; ++offset)
        for (int candidate : {k-offset, k+offset}) {
          if (candidate < 0 || candidate >= count) continue;
          normal = (reference[i].row(candidate)-reference[j].row(candidate)).transpose();
          if (normal.norm() > 1e-12) break;
        }
      if (normal.norm() <= 1e-12) { normal.setZero(); normal[0] = 1; }
    }
    normal.normalize();
    // Interval mode uses only one plane at both endpoints, certifying the whole
    // relative segment. Sampled mode applies one plane at the current sample.
    for (int t = k; t <= end; ++t) {
      std::vector<int> indices; std::vector<double> coefficients;
      for (int d = 0; d < dim; ++d) {
        indices.insert(indices.end(), {qp.index(i,t,d), qp.index(j,t,d)});
        coefficients.insert(coefficients.end(), {normal[d], -normal[d]});
      }
      Vector relative_at_t = (reference[i].row(t)-reference[j].row(t)).transpose();
      qp.add(std::move(indices), std::move(coefficients), min_sep-normal.dot(relative_at_t), kInfinity, true);
    }
  }
  qp.objectives(times, reference, c);
  Vector displacement = qp.solve(c, message);
  Positions optimized = reference;
  for (int a = 0; a < agents; ++a) for (int k = 0; k < count; ++k) for (int d = 0; d < dim; ++d)
    optimized[a](k,d) += displacement[qp.index(a,k,d)];
  return optimized;
}

double Optimizer::min_pairwise_distance(const Positions& p, bool continuous) {
  double minimum = kInfinity;
  for (size_t i = 0; i < p.size(); ++i) for (size_t j = i+1; j < p.size(); ++j)
    for (int k = 0; k < p[i].rows(); ++k) {
      Vector relative = (p[i].row(k)-p[j].row(k)).transpose();
      if (continuous && k+1 < p[i].rows()) {
        Vector delta = (p[i].row(k+1)-p[j].row(k+1)).transpose()-relative;
        double length2 = delta.squaredNorm();
        double s = length2 == 0 ? 0 : std::clamp(-relative.dot(delta)/length2, 0., 1.);
        relative += s*delta;
      }
      minimum = std::min(minimum, relative.norm());
    }
  return minimum;
}

double Optimizer::max_velocity(const Positions& p, const std::vector<double>& times) {
  double maximum = 0;
  for (const auto& agent : p) for (size_t k = 1; k < times.size(); ++k)
    maximum = std::max(maximum, (agent.row(k)-agent.row(k-1)).cwiseAbs().maxCoeff()/(times[k]-times[k-1]));
  return maximum;
}

Result Optimizer::result(const Problem& p, const std::vector<double>& times,
                         const Positions& original, const Positions& optimized) {
  Result result;
  result.times = times; result.original = original; result.optimized = optimized;
  int dim = optimized.front().cols();
  for (size_t a = 0; a < optimized.size(); ++a) {
    std::vector<Segment> segments;
    for (size_t k = 0; k+1 < times.size(); ++k) {
      Vector start(dim+1), end(dim+1);
      start << optimized[a].row(k).transpose(), times[k];
      end << optimized[a].row(k+1).transpose(), times[k+1];
      int vertex = p.trajectories[a].segments()[segment_index(p.trajectories[a], (times[k]+times[k+1])/2)].vertex;
      segments.push_back({start, end, vertex});
    }
    result.trajectories.emplace_back(std::move(segments));
  }
  result.min_distance = min_pairwise_distance(optimized);
  result.min_continuous_distance = min_pairwise_distance(optimized, true);
  result.max_velocity = max_velocity(optimized, times);
  double tolerance = std::max(1e-9, p.config.solver_eps_abs);
  result.sampled_pairwise_ok = result.min_distance+tolerance >= 2*p.radius+p.config.clearance_margin;
  result.continuous_pairwise_ok = result.min_continuous_distance+tolerance >= 2*p.radius+p.config.clearance_margin;
  if (p.environment) {
    bool safe = true;
    for (const auto& agent : optimized) for (int k = 0; k < agent.rows(); ++k)
      if (!environment_segment_free(*p.environment, agent.row(k), agent.row(k), p.radius, 1e-9)) safe = false;
    result.sampled_environment_ok = safe;
  }
  if (p.environment || !p.cells.empty())
    result.continuous_environment_ok = environment_free(p, result.trajectories, tolerance);
  return result;
}

Result Optimizer::inspect(const Problem& p, const std::vector<double>& times) {
  validate(p);
  const auto samples = times.empty() ? sample_times(p) : times;
  auto positions = sample_positions(p.trajectories, samples);
  return result(p, samples, positions, positions);
}

Result Optimizer::optimize(const Problem& p, double window_span, double stride) {
  validate(p);
  bool windowed = window_span != 0 || stride != 0;
  if (windowed && (!std::isfinite(window_span) || !std::isfinite(stride) || stride <= 0 || stride >= window_span))
    throw std::invalid_argument("Require finite 0 < stride < window_span");
  auto started = Clock::now();
  auto times = sample_times(p);
  auto original = sample_positions(p.trajectories, times), optimized = original;
  std::string message;
  if (!windowed) optimized = solve_window(p, times, original, false, &message);
  else {
    std::set<std::string> messages;
    int start = 0, windows = 0;
    while (start < int(times.size())-1) {
      int end = std::upper_bound(times.begin(), times.end(), times[start]+window_span+1e-12)-times.begin()-1;
      end = std::min(int(times.size())-1, std::max(start+1, end));
      int commit = std::upper_bound(times.begin(), times.end(), times[start]+stride+1e-12)-times.begin()-1;
      commit = std::min(end, std::max(start+1, commit));
      std::vector<double> local_times(times.begin()+start, times.begin()+end+1);
      Positions reference;
      for (const auto& agent : optimized) reference.emplace_back(agent.middleRows(start, end-start+1));
      if (p.config.progress) std::cerr << "[global_trajopt C++] window " << windows+1 << ": " << times[start] << ".." << times[end] << '\n';
      auto local = solve_window(p, local_times, reference, true, &message);
      if (p.config.progress) std::cerr << "[global_trajopt C++] " << message
          << "; max_velocity=" << max_velocity(local, local_times)
          << "; min_disk_distance=" << min_pairwise_distance(local, true) << '\n';
      double tolerance = std::max(1e-9, p.config.solver_eps_abs);
      if (min_pairwise_distance(local, true)+tolerance < 2*p.radius+p.config.clearance_margin)
        throw std::runtime_error("Sliding-window optimization produced a continuous disk collision");
      if (max_velocity(local, local_times) > p.speed+tolerance)
        throw std::runtime_error("Sliding-window optimization violated the velocity limit: " +
                                 std::to_string(max_velocity(local, local_times)));
      for (size_t a = 0; a < optimized.size(); ++a)
        optimized[a].middleRows(start, end-start+1) = local[a];
      messages.insert(message); ++windows; start = commit;
    }
    message = "sliding_window_windows="+std::to_string(windows)+"; solver_messages=";
    for (const auto& status : messages) message += status+"; ";
  }
  auto output = result(p, times, original, optimized);
  output.solver_message = message;
  output.runtime = std::chrono::duration<double>(Clock::now()-started).count();
  if (!output.continuous_pairwise_ok)
    throw std::runtime_error("Trajectory optimization produced a continuous disk collision");
  if (p.config.constraint_slack_penalty == 0 && output.continuous_environment_ok == false)
    throw std::runtime_error("Trajectory optimization violated continuous environment containment");
  return output;
}
}  // namespace stgcs::trajopt
