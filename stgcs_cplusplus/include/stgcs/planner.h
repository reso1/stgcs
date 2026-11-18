#pragma once

#include <map>
#include <memory>
#include <tuple>

#include "stgcs/graph.h"

namespace stgcs {
struct Query {
  Vector start, goal;
  double start_time{0};
  bool stay_at_goal{true};
  void validate(int dimension) const;
};

enum class Method { Search, MixedInteger, RoundedRelaxation };
enum class Heuristic { Zero, Motion, Triplet, UpdatedTriplet, GoalUpdatedTriplet, CostTable };
enum class KeyEvaluation { Separate, Joint };
enum class Dominance { None, AStar, ArrivalState, Position, SetVertex, SetEdge, SetBoth };
enum class Status { Success, Infeasible, Exhausted, Timeout };

class HeuristicCache {
 public:
  struct Data {
    std::map<std::tuple<int, int, int>, double> pairs, incoming, triplets;
    std::map<std::pair<int, int>, double> sets, table;
    bool complete{true}, has_table{false};
    double triplet_seconds{0}, table_seconds{0};
  };
  // Data is independent of file formats; adapters own persistence and must
  // restore it only onto identical base geometry.
  static HeuristicCache build(const Graph& graph, bool cost_table = false,
                              double timeout = kInfinity);
  static HeuristicCache restore(const Graph& graph, Data data);
  const Data& data() const { return data_; }
  bool complete() const { return data_.complete; }

 private:
  friend class HeuristicEvaluator;
  using Pair = std::pair<int, int>;
  using Triple = std::tuple<int, int, int>;
  Data data_;
  std::shared_ptr<const Graph> base_;
};

struct PlannerOptions {
  Method method{Method::Search};
  std::vector<Heuristic> heuristics{Heuristic::Motion};  // Combine by maximum.
  KeyEvaluation key_evaluation{KeyEvaluation::Separate};
  // Empty uses the main heuristic; otherwise fixes the separate GUB prepass heuristic.
  std::vector<Heuristic> upper_bound_heuristics;
  std::vector<Dominance> dominance{Dominance::SetVertex};
  double inflation{1};
  double timeout{kInfinity};
  bool global_upper_bound{false};  // A* prepass; prune at cost * inflation * 1.05.
  bool retain_upper_bound_solution{false};  // PBS keeps a feasible prepass incumbent.
  bool lifo_ties{false};
  int position_samples{1};
  int rounded_paths{100};
  int seed{0};
  std::shared_ptr<const HeuristicCache> cache;
};

struct ProfileEntry {
  std::size_t calls{0};
  double runtime{0};
  ProfileEntry& operator+=(const ProfileEntry& other) {
    calls += other.calls;
    runtime += other.runtime;
    return *this;
  }
};

// GUB/search include restriction and dominance time; dominance restrictions
// belong to both counters so reports can subtract their overlap explicitly.
struct SearchProfile {
  ProfileEntry gcs, gub, search, convex_restriction, dominance_check, dominance_convex_restriction;
  SearchProfile& operator+=(const SearchProfile& other) {
    gcs += other.gcs;
    gub += other.gub;
    search += other.search;
    convex_restriction += other.convex_restriction;
    dominance_check += other.dominance_check;
    dominance_convex_restriction += other.dominance_convex_restriction;
    return *this;
  }
};

struct PlanResult {
  Status status{Status::Infeasible};
  std::optional<Trajectory> trajectory;
  double runtime{0};
  std::size_t expanded{0}, generated{0}, restrictions{0};
  SearchProfile profile;
  bool success() const { return status == Status::Success; }
};

class Planner {
 public:
  explicit Planner(PlannerOptions options = {});
  PlanResult plan(const Graph& graph, const Query& query) const;
  const PlannerOptions& options() const { return options_; }

 private:
  PlannerOptions options_;
};
}  // namespace stgcs
