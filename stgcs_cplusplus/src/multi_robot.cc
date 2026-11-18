#include "stgcs/multi_robot.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

#include "internal.h"

namespace stgcs {
class PriorityNode {
 public:
  struct Plan {
    Trajectory path;
    std::size_t serial;
  };
  explicit PriorityNode(int count) : paths(count), edges(count) {}
  std::vector<int> order() const {
    std::vector<int> indegree(edges.size()), result;
    for (const auto& successors : edges)
      for (int v : successors) ++indegree[v];
    std::set<int> ready;
    for (std::size_t i = 0; i < edges.size(); ++i)
      if (!indegree[i]) ready.insert(i);
    while (!ready.empty()) {
      int u = *ready.begin();
      ready.erase(ready.begin());
      result.push_back(u);
      for (int v : edges[u])
        if (!--indegree[v]) ready.insert(v);
    }
    return result;
  }
  std::vector<int> predecessors(int v) const {
    std::vector<int> result;
    for (std::size_t u = 0; u < edges.size(); ++u)
      if (edges[u].contains(v)) result.push_back(u);
    return result;
  }
  std::set<int> descendants(int u) const {
    std::set<int> result{u};
    std::vector<int> stack{u};
    while (!stack.empty()) {
      int v = stack.back();
      stack.pop_back();
      for (int next : edges[v])
        if (result.insert(next).second) stack.push_back(next);
    }
    return result;
  }
  std::vector<std::shared_ptr<const Plan>> paths;
  std::vector<std::set<int>> edges;
  int pending{-1};
};

// All keys live within one coordination call: graph, queries, radius, planner,
// and window bounds are immutable. Siblings share immutable plan identities.
class CoordinationCache {
 public:
  using ReservationKey = std::vector<std::size_t>;
  using PlanKey = std::pair<int, ReservationKey>;
  using Plan = std::shared_ptr<const PriorityNode::Plan>;

  CoordinationCache(const Graph& graph, const std::vector<Query>& queries, double radius,
                    double horizon, MultiRobotResult& stats)
      : graph_(graph), queries_(queries), radius_(radius), stats_(stats),
        windowed_(std::isfinite(horizon)), end_(std::min(horizon, graph.end_time())),
        reserve_end_(windowed_ ? std::min(graph.end_time(), end_ + graph.minimum_duration())
                              : graph.end_time()) {}

  Plan save(Trajectory path) {
    return std::make_shared<const PriorityNode::Plan>(PriorityNode::Plan{std::move(path), serial_++});
  }
  ReservationKey key(const PriorityNode& node, const std::vector<int>& agents) const {
    ReservationKey result;
    for (int agent : agents) result.push_back(node.paths[agent]->serial);
    return result;
  }
  const Trajectory& occupied(const PriorityNode& node, int agent, bool reserve) {
    const auto& plan = *node.paths[agent];
    const auto key = std::pair{plan.serial, reserve};
    auto it = chunks_.find(key);
    if (it == chunks_.end()) {
      auto path = windowed_
          ? plan.path.chunk(queries_[agent].start_time, reserve ? reserve_end_ : end_, true)
          : Trajectory(Collision::occupied(plan.path, graph_.start_time(), end_,
                                          {true, queries_[agent].stay_at_goal}));
      it = chunks_.emplace(key, std::move(path)).first;
    }
    return it->second;
  }
  bool conflict(const PriorityNode& node, int i, int j, bool reserve = false) {
    auto a = node.paths[i]->serial, b = node.paths[j]->serial;
    const auto key = std::tuple{std::min(a, b), std::max(a, b), reserve};
    if (auto it = collisions_.find(key); it != collisions_.end()) {
      ++stats_.collision_cache_hits;
      return it->second;
    }
    Deadline timer(kInfinity);
    ++stats_.collision_calls;
    // Match source membership: accepting 1e-4 penetration can commit a
    // window endpoint that the next reservation rejects as infeasible.
    bool value = Collision::check(occupied(node, i, reserve), occupied(node, j, reserve),
                                 radius_, graph_.start_time(), reserve ? reserve_end_ : end_,
                                 {false, false}, {false, false}, 1e-8);
    stats_.collision_runtime += timer.elapsed();
    collisions_.emplace(key, value);
    return value;
  }
  const Graph* reserved(const PriorityNode& node, const std::vector<int>& higher,
                        const Deadline& deadline) {
    Deadline timer(kInfinity);
    const Graph* result = &graph_;
    auto prefix = key(node, higher);
    while (!prefix.empty()) {
      auto it = reservations_.find(prefix);
      if (it != reservations_.end()) {
        ++stats_.reservation_cache_hits;
        result = &it->second;
        break;
      }
      prefix.pop_back();
    }
    for (std::size_t offset = prefix.size(); offset < higher.size(); ++offset) {
      if (deadline.expired()) {
        stats_.reservation_runtime += timer.elapsed();
        return nullptr;
      }
      int high = higher[offset];
      auto serial = node.paths[high]->serial;
      auto it = volumes_.find(serial);
      if (it == volumes_.end())
        it = volumes_.emplace(serial, graph_.obstacle_volumes(
            occupied(node, high, true), radius_, {false, false})).first;
      else
        ++stats_.obstacle_cache_hits;
      prefix.push_back(serial);
      result = &reservations_.emplace(prefix, result->reserve(it->second)).first->second;
      ++stats_.reservation_calls;
    }
    stats_.reservation_runtime += timer.elapsed();
    return result;
  }
  std::map<PlanKey, Plan> plans;
  std::set<PlanKey> structural_failures;

 private:
  const Graph& graph_;
  const std::vector<Query>& queries_;
  double radius_;
  MultiRobotResult& stats_;
  bool windowed_;
  double end_, reserve_end_;
  std::size_t serial_{0};
  std::map<std::pair<std::size_t, bool>, Trajectory> chunks_;
  std::map<std::tuple<std::size_t, std::size_t, bool>, bool> collisions_;
  std::map<ReservationKey, Graph> reservations_;
  std::map<std::size_t, std::vector<SweptVolume>> volumes_;
};

MultiRobotPlanner::MultiRobotPlanner(Planner planner, MultiRobotOptions options)
    : planner_(std::move(planner)), options_(std::move(options)) {
  if (!std::isfinite(options_.radius) || options_.radius < 0 || std::isnan(options_.timeout) ||
      options_.timeout < 0)
    throw std::invalid_argument("Invalid multi-robot options");
}

MultiRobotResult MultiRobotPlanner::coordinate(const Graph& graph,
                                               const std::vector<Query>& queries, double horizon,
                                               double timeout,
                                               const std::vector<bool>& reached) const {
  Deadline deadline(timeout);
  MultiRobotResult result;
  result.coordination_calls = 1;
  const int count = queries.size();
  const bool windowed = std::isfinite(horizon);
  CoordinationCache cache(graph, queries, 2 * options_.radius, horizon, result);
  auto conflicts = [&](const PriorityNode& node, std::size_t limit) {
    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < count; ++i)
      for (int j = i + 1; j < count; ++j)
        if (cache.conflict(node, i, j)) {
          result.emplace_back(i, j);
          if (result.size() >= limit) return result;
        }
    return result;
  };
  auto replan = [&](PriorityNode& node, int agent, const std::vector<int>& higher) {
    if (deadline.expired()) return false;
    const CoordinationCache::PlanKey key{agent, cache.key(node, higher)};
    if (auto it = cache.plans.find(key); it != cache.plans.end()) {
      ++result.plan_cache_hits;
      node.paths[agent] = it->second;
      return true;
    }
    if (cache.structural_failures.contains(key)) {
      ++result.structural_cache_hits;
      return false;
    }
    const Graph* reserved = cache.reserved(node, higher, deadline);
    if (!reserved) return false;
    auto options = planner_.options();
    options.retain_upper_bound_solution = true;
    options.timeout = std::min(options.timeout, deadline.remaining());
    auto solution = Planner(options).plan(*reserved, queries[agent]);
    ++result.low_level_calls;
    result.low_level_runtime += solution.runtime;
    result.profile += solution.profile;
    if (!solution.success()) {
      // Search exhaustion and timeouts do not establish structural infeasibility.
      if (solution.status == Status::Infeasible) cache.structural_failures.insert(key);
      return false;
    }
    node.paths[agent] = cache.save(std::move(*solution.trajectory));
    cache.plans.emplace(key, node.paths[agent]);
    return true;
  };
  auto update = [&](PriorityNode& node, int agent) {
    ++result.update_calls;
    auto descendants = node.descendants(agent);
    for (int v : node.order()) {
      if (!descendants.contains(v)) continue;
      auto higher = node.predecessors(v);
      bool needs_plan = !windowed && v == agent;
      for (int u : higher)
        if (needs_plan || cache.conflict(node, u, v, windowed)) {
          needs_plan = true;
          break;
        }
      if (needs_plan && !replan(node, v, higher)) return false;
    }
    return true;
  };
  auto finish = [&](const PriorityNode& node, Status status) {
    result.status = deadline.expired() && status != Status::Success ? Status::Timeout : status;
    result.runtime = deadline.elapsed();
    result.coordination_runtime = result.runtime;
    for (const auto& path : node.paths)
      result.trajectories.push_back(path ? path->path : Trajectory{});
    for (int i = 0; i < count; ++i)
      for (int j : node.edges[i]) result.priorities.emplace_back(i, j);
    return result;
  };
  PriorityNode root(count);
  if (options_.method == Coordination::Prioritized) {
    auto order = options_.priority_order;
    if (order.empty()) {
      order.resize(count);
      std::iota(order.begin(), order.end(), 0);
    }
    auto sorted = order;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.size() != queries.size())
      throw std::invalid_argument("Priority order must include every agent once");
    for (int i = 0; i < count; ++i)
      if (sorted[i] != i) throw std::invalid_argument("Invalid priority order");
    std::vector<int> higher;
    for (int agent : order) {
      for (int high : higher) root.edges[high].insert(agent);
      if (!replan(root, agent, higher)) return finish(root, Status::Exhausted);
      higher.push_back(agent);
    }
    return finish(root, conflicts(root, 1).empty() ? Status::Success : Status::Exhausted);
  }
  for (int i = 0; i < count; ++i)
    if (!replan(root, i, {})) return finish(root, Status::Exhausted);
  std::vector<PriorityNode> stack{root};
  while (!stack.empty() && !deadline.expired()) {
    auto node = std::move(stack.back());
    stack.pop_back();
    ++result.expanded;
    if (node.pending >= 0 && !update(node, node.pending)) continue;
    auto collision = conflicts(node, 1);
    if (collision.empty()) return finish(node, Status::Success);
    auto [i, j] = collision.front();
    // An existing ordering cannot produce a new priority state, and its
    // opposite is cyclic. Do not enqueue the unchanged state indefinitely.
    if (node.edges[i].contains(j) || node.edges[j].contains(i)) continue;
    struct Child {
      int preference;
      double score;
      PriorityNode node;
    };
    std::vector<Child> children;
    std::vector<std::pair<int, int>> orders{{i, j}, {j, i}};
    if (!reached.empty() &&
        !reached[i] && reached[j]) std::reverse(orders.begin(), orders.end());
    for (auto [high, low] : orders) {
      PriorityNode child = node;
      child.edges[high].insert(low);
      if (child.order().size() != queries.size()) continue;
      ++result.generated;
      child.pending = options_.child_order == ChildOrder::Lazy ? low : -1;
      if (child.pending < 0 && !update(child, low)) continue;
      int preference = !reached.empty() && reached[high] && !reached[low];
      double score = 0;
      if (options_.child_order == ChildOrder::Conflicts)
        score = conflicts(child, 64).size();
      else if (options_.child_order == ChildOrder::SumOfCosts)
        for (const auto& path : child.paths) score += path->path.duration();
      else if (options_.child_order == ChildOrder::Makespan)
        for (const auto& path : child.paths) score = std::max(score, path->path.duration());
      children.push_back({preference, score, std::move(child)});
    }
    if (options_.child_order == ChildOrder::Lazy) {
      for (auto& child : children) stack.push_back(std::move(child.node));
      continue;
    }
    std::stable_sort(children.begin(), children.end(), [](const Child& a, const Child& b) {
      return std::tie(a.preference, a.score) < std::tie(b.preference, b.score);
    });
    for (auto it = children.rbegin(); it != children.rend(); ++it)
      stack.push_back(std::move(it->node));
  }
  return finish(root, Status::Exhausted);
}

MultiRobotResult MultiRobotPlanner::plan(const Graph& graph,
                                         const std::vector<Query>& queries) const {
  return coordinate(graph, queries, kInfinity, options_.timeout);
}

MultiRobotResult MultiRobotPlanner::plan_windowed(const Graph& graph,
                                                  const std::vector<Query>& input,
                                                  WindowOptions window) const {
  if (!std::isfinite(window.span) || window.span <= 0 ||
      !std::isfinite(window.execution_fraction) || window.execution_fraction <= 0 ||
      window.execution_fraction > 1)
    throw std::invalid_argument("Invalid execution window");
  Deadline deadline(options_.timeout);
  MultiRobotResult result;
  result.trajectories.resize(input.size());
  if (input.empty()) {
    result.status = Status::Success;
    return result;
  }
  const double first = input.front().start_time;
  for (const auto& q : input) {
    q.validate(graph.dimension());
    if (std::abs(q.start_time - first) > kTolerance)
      throw std::invalid_argument("Windowed planning requires synchronized starts");
  }
  double left = first, right = std::min(graph.end_time(), left + window.span);
  std::vector<Query> queries = input;
  std::vector<bool> reached(input.size(), false);
  using Signature = std::tuple<std::vector<int>, std::vector<int>, std::vector<long long>>;
  std::vector<Signature> history;
  std::vector<Trajectory> committed = result.trajectories;
  double committed_end = first;
  auto at_goal = [](const Vector& position, const Vector& goal) {
    return ((position - goal).array().abs() <= 1e-6 + 1e-5 * goal.array().abs()).all();
  };
  while (left < graph.end_time() - kTolerance && !deadline.expired()) {
    const double execution_end = right >= graph.end_time() - kTolerance
                                     ? right
                                     : left + window.execution_fraction * (right - left);
    for (std::size_t i = 0; i < queries.size(); ++i) {
      queries[i].start_time = left;
      const double earliest =
          left + (queries[i].start - queries[i].goal).lpNorm<Eigen::Infinity>() / graph.speed();
      queries[i].stay_at_goal = reached[i] || earliest <= right + kTolerance;
    }
    // Retain absolute obstacle times but remove expired cells before ECD. A
    // source on their upper time boundary cannot spend a positive duration there.
    const Graph future = graph.time_slice(left, graph.end_time());
    auto plan = coordinate(future, queries, right, deadline.remaining(), reached);
    ++result.windows;
    result.expanded += plan.expanded;
    result.generated += plan.generated;
    result.update_calls += plan.update_calls;
    result.coordination_calls += plan.coordination_calls;
    result.coordination_runtime += plan.coordination_runtime;
    result.low_level_calls += plan.low_level_calls;
    result.low_level_runtime += plan.low_level_runtime;
    result.profile += plan.profile;
    result.reservation_calls += plan.reservation_calls;
    result.reservation_runtime += plan.reservation_runtime;
    result.collision_calls += plan.collision_calls;
    result.collision_runtime += plan.collision_runtime;
    result.plan_cache_hits += plan.plan_cache_hits;
    result.reservation_cache_hits += plan.reservation_cache_hits;
    result.collision_cache_hits += plan.collision_cache_hits;
    result.obstacle_cache_hits += plan.obstacle_cache_hits;
    result.structural_cache_hits += plan.structural_cache_hits;
    std::vector<Trajectory> chunks;
    bool stalled = false;
    if (plan.success()) {
      Signature signature;
      auto& [unreached, order, positions] = signature;
      for (std::size_t i = 0; i < queries.size(); ++i) {
        chunks.push_back(plan.trajectories[i].chunk(left, execution_end, true));
        const Vector position = chunks.back().end().head(graph.dimension());
        reached[i] = at_goal(position, queries[i].goal);
        if (!reached[i]) {
          unreached.push_back(i);
          for (double x : position) positions.push_back(std::llround(x * 1e6));
        }
      }
      if (unreached.size() < 2) {
        history.clear();
      } else {
        PriorityNode priorities(queries.size());
        for (auto [high, low] : plan.priorities)
          if (!reached[high] && !reached[low]) priorities.edges[high].insert(low);
        for (int i : priorities.order())
          if (!reached[i]) order.push_back(i);
        history.push_back(std::move(signature));
        for (std::size_t period = 1; period <= 6 && 2 * period <= history.size(); ++period)
          stalled |= std::equal(history.end() - period, history.end(), history.end() - 2 * period);
        if (history.size() > 12) history.erase(history.begin());
      }
    }
    if (!plan.success() || stalled) {
      if (!window.adjust_on_failure) break;
      double grow = std::min(window.span, graph.end_time() - right);
      double rollback = std::min(2 * window.span - grow, left - first);
      if (grow + rollback <= kTolerance) break;
      right += grow;
      left -= rollback;
      if (left <= first + kTolerance) left = first;
      history.clear();
      for (std::size_t i = 0; i < queries.size(); ++i) {
        if (!result.trajectories[i].empty()) {
          result.trajectories[i] = result.trajectories[i].chunk(first, left);
          queries[i].start = result.trajectories[i].end().head(graph.dimension());
        }
        reached[i] = at_goal(queries[i].start, queries[i].goal);
      }
      continue;
    }
    for (std::size_t i = 0; i < queries.size(); ++i) {
      result.trajectories[i].append(chunks[i]);
      queries[i].start = chunks[i].end().head(graph.dimension());
    }
    if (execution_end >= committed_end) {
      committed = result.trajectories;
      committed_end = execution_end;
    }
    result.priorities = plan.priorities;
    if (std::all_of(reached.begin(), reached.end(), [](bool value) { return value; })) {
      result.status = Status::Success;
      result.runtime = deadline.elapsed();
      return result;
    }
    left = execution_end;
    right = std::min(graph.end_time(), left + window.span);
  }
  result.status = deadline.expired() ? Status::Timeout : Status::Exhausted;
  result.trajectories = std::move(committed);
  result.runtime = deadline.elapsed();
  return result;
}
}  // namespace stgcs
