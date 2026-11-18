#include <algorithm>
#include <stdexcept>

#include "internal.h"

namespace stgcs {
std::map<std::pair<int, int>, double> HeuristicEvaluator::pair_distances(
    const Optimization& model, int target, const std::map<std::tuple<int, int, int>, double>& costs,
    bool allow_reversal) {
  using Item = std::tuple<double, int, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
  std::map<std::pair<int, int>, double> distances;
  for (int pred : model.previous[target]) {
    distances[{pred, target}] = 0;
    queue.emplace(0, pred, target);
  }
  while (!queue.empty()) {
    auto [distance, current, next] = queue.top();
    queue.pop();
    if (distance != distances.at({current, next})) continue;
    for (int pred : model.previous[current]) {
      if (!allow_reversal && (pred == next || pred == target)) continue;
      double cost = 0;
      if (pred != next && pred != model.source) {
        auto it = costs.find({pred, current, next});
        if (it == costs.end()) continue;
        cost = it->second;
      }
      double candidate = distance + std::max(0.0, cost);
      auto pair = std::make_pair(pred, current);
      if (!distances.contains(pair) || candidate < distances[pair]) {
        distances[pair] = candidate;
        queue.emplace(candidate, pred, current);
      }
    }
  }
  return distances;
}

HeuristicCache HeuristicCache::build(const Graph& graph, bool cost_table, double timeout) {
  if (!graph.is_base())
    throw std::invalid_argument("Build caches before reservations or time slicing");
  if (std::isnan(timeout) || timeout < 0) throw std::invalid_argument("Invalid cache timeout");
  HeuristicCache cache;
  cache.base_ = std::make_shared<Graph>(graph);
  cache.data_.has_table = cost_table;
  Deadline deadline(timeout);
  Optimization model(graph);
  std::map<Triple, double> local_costs;
  for (std::size_t u = 0; u < model.sets.size(); ++u)
    for (int v : model.next[u])
      for (int w : model.next[v]) {
        if (deadline.expired()) {
          cache.data_.complete = false;
          return cache;
        }
        if (static_cast<int>(u) == w) continue;
        const double cost = model.triplet_cost(u, v, w, deadline);
        if (!std::isfinite(cost)) continue;
        local_costs[{u, v, w}] = cost;
        cache.data_.triplets[{model.roots[u], model.roots[v], model.roots[w]}] = cost;
      }
  // Pair-state Dijkstra preserves dependence on the incoming interface.
  for (std::size_t target = 0; target < model.sets.size(); ++target) {
    auto distances = HeuristicEvaluator::pair_distances(model, target, local_costs, true);
    cache.data_.sets[{model.roots[target], model.roots[target]}] = 0;
    for (const auto& [pair, cost] : distances) {
      auto triple =
          std::make_tuple(model.roots[pair.first], model.roots[pair.second], model.roots[target]);
      auto [pair_it, pair_inserted] = cache.data_.pairs.try_emplace(triple, cost);
      if (!pair_inserted) pair_it->second = std::min(pair_it->second, cost);
      auto key = std::make_pair(model.roots[pair.first], model.roots[target]);
      auto [it, inserted] = cache.data_.sets.try_emplace(key, cost);
      if (!inserted) it->second = std::min(it->second, cost);
    }
  }
  cache.data_.triplet_seconds = deadline.elapsed();
  if (cost_table)
    for (std::size_t source = 0; source < model.sets.size(); ++source) {
      auto distances = SearchEngine::table(model, source, -1, deadline);
      if (deadline.expired()) {
        cache.data_.complete = false;
        break;
      }
      for (const auto& [target, cost] : distances)
        cache.data_.table[{model.roots[source], model.roots[target]}] = cost;
      for (int pred : model.previous[source]) {
        distances = SearchEngine::table(model, source, pred, deadline);
        if (deadline.expired()) {
          cache.data_.complete = false;
          break;
        }
        for (const auto& [target, cost] : distances)
          cache.data_.incoming[{model.roots[pred], model.roots[source], model.roots[target]}] = cost;
      }
      if (!cache.data_.complete) break;
    }
  cache.data_.table_seconds = deadline.elapsed() - cache.data_.triplet_seconds;
  return cache;
}

HeuristicCache HeuristicCache::restore(const Graph& graph, Data data) {
  if (!graph.is_base() || !data.complete)
    throw std::invalid_argument("Restore requires a base graph and complete cache data");
  HeuristicCache cache;
  cache.base_ = std::make_shared<Graph>(graph);
  cache.data_ = std::move(data);
  return cache;
}

HeuristicEvaluator::HeuristicEvaluator(Optimization& model, const Query& query,
                                       const PlannerOptions& options, const Deadline& deadline)
    : model_(model), query_(query), options_(options) {
  bool updated = false, goal_updated = false, table = false, cached = false;
  for (auto kind : options.heuristics) {
    updated |= kind == Heuristic::UpdatedTriplet;
    goal_updated |= kind == Heuristic::GoalUpdatedTriplet;
    table |= kind == Heuristic::CostTable;
    cached |= kind == Heuristic::Triplet || kind == Heuristic::GoalUpdatedTriplet ||
              kind == Heuristic::CostTable;
  }
  if (cached && (!options.cache || !options.cache->base_ ||
                 options.cache->base_->lineage() != model.graph.lineage()))
    throw std::invalid_argument("Build the heuristic cache on this graph's base regions");
  if (table && !options.cache->data_.has_table)
    throw std::invalid_argument("CostTable requires HeuristicCache::build(graph, true)");
  for (int parent : model.goal_parents) goals_.push_back(model.roots[parent]);
  std::sort(goals_.begin(), goals_.end());
  goals_.erase(std::unique(goals_.begin(), goals_.end()), goals_.end());
  if (updated || goal_updated) {
    std::map<std::tuple<int, int, int>, double> costs;
    for (std::size_t u = 0; u < model.sets.size(); ++u)
      for (int v : model.next[u])
        for (int w : model.next[v]) {
          if (deadline.expired()) return;
          if (static_cast<int>(u) == w || static_cast<int>(u) == model.source) continue;
          double cost;
          if (updated || w >= model.target) {
            cost = model.triplet_cost(u, v, w, deadline);
          } else {
            const int a = model.roots[u], b = model.roots[v], c = model.roots[w];
            if (a == b || b == c || a == c) {
              // Internal transitions and base reversals can have zero relaxed duration.
              cost = 0;
            } else {
              const auto& triplets = options.cache->data_.triplets;
              auto it = triplets.find({a, b, c});
              if (it == triplets.end()) continue;
              cost = it->second;
            }
          }
          if (std::isfinite(cost)) costs[{u, v, w}] = cost;
        }
    updated_ = pair_distances(model, model.target, costs, false);
  }
  if (table)
    for (int parent : model.goal_parents) {
      double correction = kInfinity;
      for (int pred : model.previous[parent])
        correction = std::min(correction, Geometry::travel_time(model.interface(pred, parent),
                                                                query.goal, model.graph.speed()));
      if (!std::isfinite(correction)) correction = 0;
      auto [it, inserted] = correction_.try_emplace(model.roots[parent], correction);
      if (!inserted) it->second = std::min(it->second, correction);
    }
}

double HeuristicEvaluator::value(const Label& label) {
  const int current = label.vertex(), previous = label.predecessor();
  if (current == model_.target) return 0;
  double value = 0;
  const auto lookup = [](const auto& map, const auto& key) {
    auto it = map.find(key);
    return it == map.end() ? 0.0 : it->second;  // Uncomputed entries remain admissible.
  };
  for (auto kind : options_.heuristics) {
    double bound = 0;
    if (kind == Heuristic::Motion) {
      if (current > model_.target) continue;  // Dummy goal has zero motion bound.
      auto key = std::make_pair(previous, current);
      if (!motion_.contains(key))
        motion_[key] = Geometry::travel_time(model_.interface(previous, current), query_.goal,
                                             model_.graph.speed());
      bound = motion_.at(key);
    } else if (kind == Heuristic::UpdatedTriplet) {
      bound = lookup(updated_, std::make_pair(previous, current));
    } else if (kind == Heuristic::GoalUpdatedTriplet) {
      const int root = model_.roots[current];
      if (previous >= 0 && previous != model_.source && model_.roots[previous] == root &&
          std::binary_search(goals_.begin(), goals_.end(), root)) continue;
      auto it = updated_.find({previous, current});
      bound = it == updated_.end() ? kInfinity : it->second;
    } else if (kind == Heuristic::Triplet || kind == Heuristic::CostTable) {
      const auto& cache = *options_.cache;
      const int root = model_.roots[current];
      const int pred = previous < 0 ? -1 : model_.roots[previous];
      bound = kInfinity;
      if (kind == Heuristic::CostTable && previous != model_.source && root == pred) {
        // Skip travel to the first base exit, including a direct exit to the goal.
        if (!exit_bounds_.contains(root)) {
          double best = kInfinity;
          if (correction_.contains(root)) {
            best = 0;
          } else {
            const auto& incoming = cache.data_.incoming;
            auto it = incoming.lower_bound({root, 0, 0});  // Keys are (from, to, goal).
            for (; it != incoming.end() && std::get<0>(it->first) == root; ++it) {
              auto goal = correction_.find(std::get<2>(it->first));
              if (goal != correction_.end()) best = std::min(best, it->second + goal->second);
            }
          }
          exit_bounds_[root] = best;
        }
        value = std::max(value, exit_bounds_.at(root));
        continue;
      }
      for (int goal : goals_) {
        double candidate;
        if (kind == Heuristic::Triplet) {
          candidate = previous == model_.source || pred == root || pred < 0
                          ? lookup(cache.data_.sets, std::make_pair(root, goal))
                          : lookup(cache.data_.pairs, std::make_tuple(pred, root, goal));
        } else {
          candidate = lookup(cache.data_.table, std::make_pair(root, goal)) + correction_[goal];
          if (root == goal)
            candidate = std::min(
                candidate, (label.trajectory.end().head(model_.graph.dimension()) - query_.goal)
                                   .lpNorm<Eigen::Infinity>() /
                               model_.graph.speed());
          if (previous != model_.source && pred >= 0 && pred != root)
            candidate =
                std::max(candidate, lookup(cache.data_.incoming, std::make_tuple(pred, root, goal)) +
                                        correction_[goal]);
        }
        bound = std::min(bound, candidate);
      }
    }
    value = std::max(value, bound);
  }
  return value;
}
}  // namespace stgcs
