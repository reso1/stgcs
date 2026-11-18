#pragma once

#include <optional>
#include <string>
#include <vector>
#include "stgcs/trajectory.h"

namespace stgcs::trajopt {
struct Config {
  double sample_dt{0.02}, min_sample_dt{1e-4};
  double velocity_gradient_weight{0.05}, displacement_weight{0.2};
  double displacement_jitter_weight{1e-4}, clearance_margin{1e-6};
  double spatial_containment_tolerance{0}, spatial_containment_soft_margin{0};
  double constraint_slack_penalty{0}, constraint_slack_limit{kInfinity};
  bool include_interval_collision_constraints{true}, fix_terminal_states{true};
  bool include_trajectory_knot_times{true};
  int solver_max_iter{100000};
  double solver_eps_abs{1e-6}, solver_eps_rel{1e-6};
  bool solver_print_to_console{false}, progress{false};
  int round_time_decimals{9};
};

struct Obstacle {
  // Spheres use center/radius; polygons use spatial halfspaces.
  Vector center;
  double radius{0};
  Matrix A;
  Vector b;
};

struct Environment {
  Vector lower, upper;
  std::vector<Obstacle> obstacles;
  std::vector<Polyhedron> regions;
};

// Each matrix is samples x spatial dimensions for one robot.
using Positions = std::vector<Matrix>;
struct Problem {
  std::vector<Trajectory> trajectories;
  std::vector<Polyhedron> cells;
  std::optional<Environment> environment;
  double radius{0}, speed{1};
  Config config;
};

struct Result {
  std::vector<Trajectory> trajectories;
  std::vector<double> times;
  Positions original, optimized;
  double runtime{0}, min_distance{kInfinity}, max_velocity{0};
  bool sampled_pairwise_ok{true};
  bool continuous_pairwise_ok{true};
  double min_continuous_distance{kInfinity};
  std::optional<bool> sampled_environment_ok, continuous_environment_ok;
  std::string solver_message;
};

class Optimizer {
 public:
  static Result optimize(const Problem& problem, double window_span = 0, double stride = 0);
  static Result inspect(const Problem& problem, const std::vector<double>& times = {});
  static std::vector<Trajectory> reference_trajectories(const Problem& problem);
  static std::vector<double> sample_times(const Problem& problem);
  static Positions sample_positions(const std::vector<Trajectory>& trajectories,
                                    const std::vector<double>& times);
  static double min_pairwise_distance(const Positions& positions, bool continuous = false);
  static double max_velocity(const Positions& positions, const std::vector<double>& times);
  static bool pairwise_free(const std::vector<Trajectory>& trajectories, double radius,
                            const std::vector<bool>& stays, double tolerance);
  static bool disk_pairwise_free(const std::vector<Trajectory>& trajectories, double radius,
                                const std::vector<bool>& stays, double tolerance);
  static bool environment_free(const Problem& problem, const std::vector<Trajectory>& paths,
                               double tolerance = 1e-9);
  static bool cells_cover(const std::vector<Polyhedron>& cells, const Vector& start,
                          const Vector& end, double tolerance);

 private:
  static void validate(const Problem& problem);
  static Vector state_at(const Trajectory& trajectory, double time);
  static Vector state_at(const Segment& segment, double time);
  static int segment_index(const Trajectory& trajectory, double time);
  static int containing_cell(const Problem& problem, const Trajectory& path, int segment);
  static bool environment_segment_free(const Environment& environment, const Vector& start,
                                       const Vector& end, double radius, double tolerance);
  static std::pair<Vector, double> separation(const Obstacle& obstacle, const Vector& start,
                                             const Vector& end, double radius, double margin);
  static Vector closest_point(const Vector& point, const Vector& start, const Vector& end);
  static std::pair<Matrix, Vector> inflated_polygon(const Obstacle& obstacle,
                                                  double radius, double margin);
  static Positions solve_window(const Problem& problem, const std::vector<double>& times,
                                const Positions& reference, bool fix_window,
                                std::string* message);
  static Result result(const Problem& problem, const std::vector<double>& times,
                       const Positions& original, const Positions& optimized);
};
}  // namespace stgcs::trajopt
