#pragma once

#include "stgcs/planner.h"

namespace stgcs {
enum class Coordination { PBS, Prioritized };
enum class ChildOrder { Lazy, SumOfCosts, Makespan, Conflicts };

struct MultiRobotOptions {
  Coordination method{Coordination::PBS};
  ChildOrder child_order{ChildOrder::Conflicts};
  double radius{0.1};
  double timeout{kInfinity};
  std::vector<int> priority_order;  // Empty means index order for prioritized planning.
};

struct WindowOptions {
  double span{1};
  double execution_fraction{1};
  bool adjust_on_failure{true};
};

struct MultiRobotResult {
  Status status{Status::Infeasible};
  std::vector<Trajectory> trajectories;
  std::vector<std::pair<int, int>> priorities;  // Higher -> lower.
  double runtime{0};
  std::size_t expanded{0}, windows{0};
  // Expanded counts PBS nodes popped; generated excludes the root and counts
  // acyclic children before replanning, including children whose update fails.
  std::size_t generated{0}, update_calls{0}, coordination_calls{0};
  std::size_t low_level_calls{0}, reservation_calls{0}, collision_calls{0};
  std::size_t plan_cache_hits{0}, reservation_cache_hits{0}, collision_cache_hits{0};
  std::size_t obstacle_cache_hits{0}, structural_cache_hits{0};
  double coordination_runtime{0}, low_level_runtime{0}, reservation_runtime{0}, collision_runtime{0};
  SearchProfile profile;
  bool success() const { return status == Status::Success; }
};

class MultiRobotPlanner {
 public:
  MultiRobotPlanner(Planner planner, MultiRobotOptions options = {});
  MultiRobotResult plan(const Graph& graph, const std::vector<Query>& queries) const;
  MultiRobotResult plan_windowed(const Graph& graph, const std::vector<Query>& queries,
                                 WindowOptions window = {}) const;

 private:
  MultiRobotResult coordinate(const Graph& graph, const std::vector<Query>& queries, double horizon,
                              double timeout, const std::vector<bool>& reached = {}) const;
  Planner planner_;
  MultiRobotOptions options_;
};
}  // namespace stgcs
