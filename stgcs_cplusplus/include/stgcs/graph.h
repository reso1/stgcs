#pragma once

#include <drake/solvers/mosek_solver.h>

#include <memory>

#include "stgcs/trajectory.h"

namespace stgcs {
class OptimizationCache;

struct Vertex {
  Polyhedron set;
  Bounds bounds;
  int root;
};

struct SweptVolume {
  Polyhedron set;
  Bounds bounds;
};

class Graph {
 public:
  static constexpr double kMinimumRegionWidth = 1e-6;  // Spatial sliver filter.
  Graph(const std::vector<Polyhedron>& spatial_regions, double start, double end, double speed,
        double minimum_duration = 1e-6);
  const std::vector<Vertex>& vertices() const { return vertices_; }
  const std::vector<std::vector<int>>& edges() const { return edges_; }
  int dimension() const { return dimension_; }
  double start_time() const { return start_; }
  double end_time() const { return end_; }
  double speed() const { return speed_; }
  double minimum_duration() const { return minimum_duration_; }
  const void* lineage() const { return lineage_.get(); }
  bool is_base() const { return is_base_; }
  std::size_t edge_count() const;
  // ECD: subtract the moving axis-aligned footprint, retaining root identities.
  Graph reserve(const Trajectory& obstacle, double combined_radius, Occupancy stay = {}) const;
  Graph reserve(const std::vector<std::pair<double, Trajectory>>& obstacles, Occupancy stay = {}) const;
  // Sweeps depend on the trajectory and horizon, not on earlier reservations.
  std::vector<SweptVolume> obstacle_volumes(const Trajectory& obstacle, double combined_radius,
                                           Occupancy stay = {}) const;
  Graph reserve(const std::vector<SweptVolume>& volumes) const;
  Graph time_slice(double start, double end) const;

 private:
  friend class Optimization;
  bool has_width(const Bounds& bounds) const;
  bool add(const Polyhedron& set, int root);
  void connect_pair(int first, int second);
  void connect();
  void connect(const Graph& original, const std::vector<int>& parents,
               const std::vector<bool>& modified);
  void subtract(const std::vector<SweptVolume>& volumes, std::vector<int>& parents,
                std::vector<bool>& modified);
  int dimension_;
  double start_, end_, speed_, minimum_duration_;
  // Shared by derived graphs and caches; outlives all solves using this graph.
  std::shared_ptr<drake::solvers::MosekSolver::License> solver_license_;
  std::shared_ptr<OptimizationCache> optimization_cache_;
  std::vector<Vertex> vertices_;
  std::vector<std::vector<int>> edges_;
  std::shared_ptr<const int> lineage_{std::make_shared<const int>(0)};
  bool is_base_{true};
};
}  // namespace stgcs
