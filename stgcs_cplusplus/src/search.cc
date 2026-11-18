#include <algorithm>

#include "internal.h"

namespace stgcs {
Frontier::Frontier(Optimization& model, const PlannerOptions& options, SearchProfile* profile)
    : model_(model), options_(options), profile_(profile) {}

bool Frontier::contains(const Arrival& from, Arrival& to, bool exact) const {
  if (!Geometry::cone_contains(from.point, to.point, model_.graph.speed())) return false;
  if (!exact) return true;
  if (!to.set)
    to.set.emplace(model_.interface(to.predecessor, to.vertex), to.point, model_.graph.speed());
  return to.set->contained_by(from.point);
}

bool Frontier::sample_reject(const Label& label, const Deadline& deadline) {
  const int v = label.vertex();
  auto& samples = samples_[v];
  auto& costs = sample_costs_[v];
  if (samples.empty()) {
    for (const auto& sample : Geometry::sample(model_.sets[v], options_.position_samples, options_.seed))
      samples.push_back(sample.head(model_.graph.dimension()));
  }
  Vector candidate = Vector::Constant(samples.size(), kInfinity);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (deadline.expired()) return true;
    double duration;
    std::optional<Trajectory> solution;
    {
      ProfileTimer optimization(profile_ ? &profile_->convex_restriction : nullptr);
      ProfileTimer dominance(profile_ ? &profile_->dominance_convex_restriction : nullptr);
      solution = model_.restrict(label.path, deadline, nullptr, &samples[i], &duration);
    }
    if (solution) candidate[i] = duration;
  }
  for (const auto& prior : costs)
    if ((prior.array() <= candidate.array()).all()) return true;
  std::erase_if(costs,
                [&](const Vector& prior) { return (candidate.array() <= prior.array()).all(); });
  costs.push_back(std::move(candidate));
  return false;
}

bool Frontier::reject(const Label& label, const Deadline& deadline) {
  ProfileTimer timer(profile_ ? &profile_->dominance_check : nullptr);
  std::shared_ptr<Arrival> arrival;
  for (Dominance kind : options_.dominance) {
    if (kind == Dominance::None) continue;
    if (kind == Dominance::AStar) {
      auto it = best_.find(label.vertex());
      if (it != best_.end() && label.cost >= it->second) return true;
      best_[label.vertex()] = label.cost;
      continue;
    }
    if (kind == Dominance::Position) {
      if (sample_reject(label, deadline)) return true;
      continue;
    }
    const bool exact = kind != Dominance::ArrivalState;
    if (!arrival)
      arrival = std::make_shared<Arrival>(
          Arrival{label.predecessor(), label.vertex(), label.trajectory.end(), std::nullopt});
    std::vector<int> keys{kind == Dominance::SetEdge ? label.predecessor() : -1};
    if (kind == Dominance::SetBoth) keys.push_back(label.predecessor());
    for (int predecessor : keys) {
      auto& frontier = frontiers_[{kind, predecessor, label.vertex()}];
      const auto end =
          std::upper_bound(frontier.begin(), frontier.end(), arrival->time() + kTolerance,
                           [](double time, const auto& item) { return time < item->time(); });
      for (auto it = frontier.begin(); it != end; ++it)
        if (contains(**it, *arrival, exact)) return true;
      const auto begin =
          std::lower_bound(frontier.begin(), frontier.end(), arrival->time() - kTolerance,
                           [](const auto& item, double time) { return item->time() < time; });
      frontier.erase(
          std::remove_if(begin, frontier.end(),
                         [&](const auto& prior) { return contains(*arrival, *prior, exact); }),
          frontier.end());
      const auto position =
          std::upper_bound(frontier.begin(), frontier.end(), arrival->time(),
                           [](double time, const auto& item) { return time < item->time(); });
      frontier.insert(position, arrival);
    }
  }
  return false;
}

bool Frontier::stale(const Label& label) const {
  const auto it = best_.find(label.vertex());
  return it != best_.end() && label.cost > it->second;
}

std::optional<Trajectory> SearchEngine::search(
    Optimization& model, const Query& query, const PlannerOptions& options,
    const Deadline& deadline, HeuristicEvaluator& heuristic, PlanResult& stats, double upper_bound,
    PrefixCache& cache) {
  const bool joint = options.key_evaluation == KeyEvaluation::Joint;
  auto compare = [&](const Label& a, const Label& b) {
    if (a.priority != b.priority) return a.priority > b.priority;
    return options.lifo_ties ? a.serial < b.serial : a.serial > b.serial;
  };
  std::priority_queue<Label, std::vector<Label>, decltype(compare)> queue(compare);
  queue.push(Label{{model.source}, {}, 0, 0, 0});
  Frontier frontier(model, options, &stats.profile);
  std::size_t serial = 0;
  while (!queue.empty() && !deadline.expired()) {
    Label label = queue.top();
    queue.pop();
    if (frontier.stale(label)) continue;
    ++stats.expanded;
    if (label.vertex() == model.target) return label.trajectory;
    for (int next : model.next[label.vertex()]) {
      if (deadline.expired()) break;
      if (std::find(label.path.begin(), label.path.end(), next) != label.path.end()) continue;
      Label child;
      child.path = label.path;
      child.path.push_back(next);
      child.serial = ++serial;
      const auto key = std::make_pair(KeyEvaluation::Separate, child.path);
      auto it = cache.find(key);
      if (it == cache.end()) {
        ++stats.restrictions;
        std::optional<Trajectory> solution;
        {
          ProfileTimer timer(&stats.profile.convex_restriction);
          solution = model.restrict(child.path, deadline);
        }
        it = cache.emplace(key, std::move(solution)).first;
      }
      if (!it->second) continue;
      child.trajectory = *it->second;
      child.cost = std::max(0.0, child.trajectory.duration());
      if (joint && next < model.target) {
        const auto joint_key = std::make_pair(KeyEvaluation::Joint, child.path);
        auto joint_it = cache.find(joint_key);
        if (joint_it == cache.end()) {
          ++stats.restrictions;
          std::optional<Trajectory> solution;
          {
            ProfileTimer timer(&stats.profile.convex_restriction);
            solution = model.restrict(child.path, deadline, nullptr, nullptr, nullptr,
                                      &query.goal, options.inflation);
          }
          joint_it = cache.emplace(joint_key, std::move(solution)).first;
        }
        if (!joint_it->second) continue;
        // Only the priority uses the joint optimizer. Dominance retains the
        // g-only minimum and earliest-arrival witness stored in child.
        const auto& solution = *joint_it->second;
        const double h = (solution.end().head(model.graph.dimension()) - query.goal)
                             .lpNorm<Eigen::Infinity>() / model.graph.speed();
        child.priority = std::max(0.0, solution.duration()) + options.inflation * h;
      } else {
        child.priority = child.cost + options.inflation * heuristic.value(child);
      }
      if (!std::isfinite(child.priority) || child.priority > upper_bound + kTolerance) continue;
      if (next != model.target && frontier.reject(child, deadline)) continue;
      queue.push(std::move(child));
      ++stats.generated;
    }
  }
  return std::nullopt;
}

PlanResult SearchEngine::run(Optimization& model, const Query& query, const PlannerOptions& options,
                             const Deadline& deadline) {
  PlanResult result;
  PrefixCache cache;
  std::optional<HeuristicEvaluator> heuristic;
  double upper_bound = kInfinity;
  std::optional<Trajectory> incumbent;
  if (options.global_upper_bound) {
    ProfileTimer timer(&result.profile.gub);
    auto greedy = options;
    greedy.dominance = {Dominance::AStar};
    greedy.inflation = 1;
    greedy.key_evaluation = KeyEvaluation::Separate;
    if (options.upper_bound_heuristics.empty()) {
      heuristic.emplace(model, query, options, deadline);
      incumbent = search(model, query, greedy, deadline, *heuristic, result, kInfinity, cache);
    } else {
      greedy.heuristics = options.upper_bound_heuristics;
      HeuristicEvaluator prepass_heuristic(model, query, greedy, deadline);
      incumbent = search(model, query, greedy, deadline, prepass_heuristic, result, kInfinity, cache);
    }
    if (incumbent) upper_bound = incumbent->duration() * options.inflation * 1.05;
  }
  {
    ProfileTimer timer(&result.profile.search);
    if (!heuristic) heuristic.emplace(model, query, options, deadline);
    result.trajectory = search(model, query, options, deadline, *heuristic, result, upper_bound, cache);
  }
  if (!result.trajectory && options.retain_upper_bound_solution)
    result.trajectory = std::move(incumbent);
  result.status = result.trajectory ? Status::Success
                                    : (deadline.expired() ? Status::Timeout : Status::Exhausted);
  return result;
}

std::map<int, double> SearchEngine::table(Optimization& model, int source, int predecessor,
                                          const Deadline& deadline) {
  auto compare = [](const Label& a, const Label& b) {
    return std::tie(a.cost, a.serial) > std::tie(b.cost, b.serial);
  };
  std::priority_queue<Label, std::vector<Label>, decltype(compare)> queue(compare);
  queue.push(Label{{source}, {}, 0, 0, 0});
  PlannerOptions options;
  Frontier frontier(model, options);
  std::map<int, double> distances;
  std::optional<Polyhedron> initial;
  if (predecessor >= 0) initial = model.interface(predecessor, source);
  std::size_t serial = 0;
  while (!queue.empty() && !deadline.expired()) {
    auto label = queue.top();
    queue.pop();
    distances.try_emplace(label.vertex(), label.cost);
    if (distances.size() == model.sets.size()) break;
    for (int next : model.next[label.vertex()]) {
      if (deadline.expired()) break;
      if (std::find(label.path.begin(), label.path.end(), next) != label.path.end()) continue;
      Label child;
      child.path = label.path;
      child.path.push_back(next);
      child.serial = ++serial;
      auto solution = model.restrict(child.path, deadline, initial ? &*initial : nullptr);
      if (!solution) continue;
      // Static cache restrictions have a free departure time. Dominance must
      // compare elapsed costs at a common origin, not arbitrary optimizer
      // timestamps; translating toward the base graph's start preserves
      // feasibility on its time-invariant extruded regions.
      child.trajectory = solution->shifted(model.graph.start_time() - solution->start().tail(1)[0]);
      child.cost = std::max(0.0, solution->duration());
      if (!frontier.reject(child, deadline)) queue.push(std::move(child));
    }
  }
  return distances;
}
}  // namespace stgcs
