#pragma once

#include <drake/geometry/optimization/graph_of_convex_sets.h>
#include <drake/solvers/mosek_solver.h>

#include <chrono>
#include <mutex>
#include <queue>

#include "stgcs/planner.h"

namespace stgcs {
class Deadline {
 public:
  explicit Deadline(double seconds) : limit_(seconds), start_(Clock::now()) {}
  double elapsed() const { return std::chrono::duration<double>(Clock::now() - start_).count(); }
  double remaining() const { return std::max(0.0, limit_ - elapsed()); }
  bool expired() const { return remaining() <= 0; }

 private:
  using Clock = std::chrono::steady_clock;
  double limit_;
  Clock::time_point start_;
};

using GCS = drake::geometry::optimization::GraphOfConvexSets;
using GCSOptions = drake::geometry::optimization::GraphOfConvexSetsOptions;

class ProfileTimer {
 public:
  explicit ProfileTimer(ProfileEntry* entry) : entry_(entry) {
    if (entry_) ++entry_->calls;
  }
  ~ProfileTimer() {
    if (entry_) entry_->runtime += timer_.elapsed();
  }
  ProfileTimer(const ProfileTimer&) = delete;
  ProfileTimer& operator=(const ProfileTimer&) = delete;

 private:
  ProfileEntry* entry_;
  Deadline timer_{kInfinity};
};

class OptimizationCache {
 public:
  struct Model {
    GCS gcs;
    std::vector<GCS::Vertex*> vertices;
    std::map<std::pair<int, int>, const GCS::Edge*> edges;
  };
  std::unique_ptr<Model> acquire();
  void release(std::unique_ptr<Model> model);

 private:
  std::mutex mutex_;
  std::unique_ptr<Model> idle_;
};

class Optimization {
 public:
  Optimization(const Graph& graph, const Query* query = nullptr);
  ~Optimization();
  bool valid() const { return source >= 0 && target >= 0; }
  std::optional<Trajectory> restrict(const std::vector<int>& path, const Deadline& deadline,
                                     const Polyhedron* initial = nullptr,
                                     const Vector* final_position = nullptr,
                                     double* sampled_duration = nullptr,
                                     const Vector* motion_goal = nullptr,
                                     double inflation = 1);
  std::optional<Trajectory> shortest(const PlannerOptions& options, const Deadline& deadline);
  const Polyhedron& interface(int previous, int current) const;
  double triplet_cost(int u, int v, int w, const Deadline& deadline) const;
  static GCSOptions solver_options(double timeout);
  static void add_motion_constraints(GCS::Vertex* vertex, double speed, double minimum_duration);
  Trajectory physical(const Trajectory& trajectory) const;
  const Graph& graph;
  std::vector<Polyhedron> sets;
  std::vector<std::vector<int>> next, previous;
  std::vector<int> roots, goal_parents;
  int source{-1}, target{-1};

 private:
  GCS::Vertex* add_vertex(const Polyhedron& set, bool motion, bool stationary = true);
  GCS::Edge* add_edge(GCS::Vertex* from, GCS::Vertex* to);
  std::optional<Trajectory> extract(const std::vector<const GCS::Edge*>& edges,
                     const std::vector<int>& path,
                     const drake::solvers::MathematicalProgramResult& result);
  std::shared_ptr<OptimizationCache> cache_;
  std::unique_ptr<OptimizationCache::Model> model_;
  GCS& gcs_;
  std::vector<GCS::Vertex*> vertices_;
  std::map<std::pair<int, int>, const GCS::Edge*> edges_;
  mutable std::map<std::pair<int, int>, Polyhedron> interfaces_;
};

struct Label {
  std::vector<int> path;
  Trajectory trajectory;
  double cost{0}, priority{0};
  std::size_t serial{0};
  int vertex() const { return path.back(); }
  int predecessor() const { return path.size() > 1 ? path[path.size() - 2] : -1; }
};

class Frontier {
 public:
  Frontier(Optimization& model, const PlannerOptions& options, SearchProfile* profile = nullptr);
  bool reject(const Label& label, const Deadline& deadline);
  bool stale(const Label& label) const;

 private:
  struct Arrival {
    int predecessor, vertex;
    Vector point;
    std::optional<ArrivalSet> set;
    double time() const { return point[point.size() - 1]; }
  };
  bool contains(const Arrival& from, Arrival& to, bool exact) const;
  bool sample_reject(const Label& label, const Deadline& deadline);
  Optimization& model_;
  const PlannerOptions& options_;
  SearchProfile* profile_;
  std::map<int, double> best_;
  std::map<std::tuple<Dominance, int, int>, std::vector<std::shared_ptr<Arrival>>> frontiers_;
  std::map<int, std::vector<Vector>> samples_, sample_costs_;
};

class HeuristicEvaluator {
 public:
  HeuristicEvaluator(Optimization& model, const Query& query, const PlannerOptions& options,
                     const Deadline& deadline);
  double value(const Label& label);
  static std::map<std::pair<int, int>, double> pair_distances(
      const Optimization& model, int target,
      const std::map<std::tuple<int, int, int>, double>& costs, bool allow_reversal);

 private:
  Optimization& model_;
  const Query& query_;
  const PlannerOptions& options_;
  std::map<std::pair<int, int>, double> motion_, updated_;
  std::map<int, double> correction_, exit_bounds_;
  std::vector<int> goals_;
};

class SearchEngine {
 public:
  static PlanResult run(Optimization& model, const Query& query, const PlannerOptions& options,
                        const Deadline& deadline);
  static std::map<int, double> table(Optimization& model, int source, int predecessor,
                                     const Deadline& deadline);

 private:
  // A joint optimum cannot serve as a minimum-cost prefix in a separate solve.
  using PrefixCache = std::map<std::pair<KeyEvaluation, std::vector<int>>,
                               std::optional<Trajectory>>;
  static std::optional<Trajectory> search(
      Optimization& model, const Query& query, const PlannerOptions& options,
      const Deadline& deadline, HeuristicEvaluator& heuristic, PlanResult& statistics, double upper_bound,
      PrefixCache& cache);
};
}  // namespace stgcs
