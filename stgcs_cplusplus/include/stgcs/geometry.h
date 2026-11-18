#pragma once

#include <drake/geometry/optimization/hpolyhedron.h>
#include <drake/solvers/solver_options.h>

#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace stgcs {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Polyhedron = drake::geometry::optimization::HPolyhedron;
inline constexpr double kTolerance = 1e-7;
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct Bounds {
  Vector lower, upper;
  bool overlaps(const Bounds& other, double tolerance = kTolerance) const;
  // A conservative box enclosing this box intersected with normal.dot(x) <= bound.
  Bounds clip(const Vector& normal, double bound) const;
};

// Lazy support values for repeated cone tests against one immutable arrival set.
class ArrivalSet {
 public:
  ArrivalSet(const Polyhedron& interface, const Vector& arrival, double speed);
  bool contained_by(const Vector& from) const;

 private:
  Vector arrival_;
  double speed_;
  Polyhedron reachable_;
  mutable std::optional<bool> empty_;
  mutable Vector support_;
};

class Geometry {
 public:
  // Same MOSEK numerical settings as Python's geometry and GCS solvers.
  static drake::solvers::SolverOptions solver_options(double timeout = kInfinity);
  static Polyhedron box(const Vector& lower, const Vector& upper);
  static Polyhedron point(const Vector& point);
  static Polyhedron extrude(const Polyhedron& spatial, double start, double end);
  static Polyhedron clip(const Polyhedron& set, const Vector& normal, double bound);
  static Polyhedron time_slice(const Polyhedron& set, double start, double end);
  // With a minimum width, stop as soon as any axis cannot meet that width.
  // Without one, retain bounds for flat interfaces and points as well.
  static std::optional<Bounds> bounds(
      const Polyhedron& set, std::optional<double> minimum_width = std::nullopt);
  static std::optional<std::pair<double, double>> time_interval(
      const Polyhedron& set, const Vector& position, double start, double end);
  static double minimum(const Polyhedron& set, const Vector& cost);
  static double travel_time(const Polyhedron& from, const Vector& goal, double speed);
  static Matrix affine_basis(const Polyhedron& set);
  // Python position-dominance policy: ambient hit-and-run, with its existing
  // feasible-point substitution when Drake cannot draw a sample.
  static std::vector<Vector> sample(const Polyhedron& set, int count, int seed);
  static bool cone_contains(const Vector& from, const Vector& to, double speed);
  static bool cone_contains_set(const Vector& from, const Vector& arrival,
                                const Polyhedron& interface, double speed);
};
}  // namespace stgcs
