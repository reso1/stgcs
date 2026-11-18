#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "stgcs/planner.h"
#include "stgcs/multi_robot.h"

#ifndef NDEBUG
#error The ST-GCS planner CLI requires a Release build
#endif

using namespace stgcs;

// Command-line input, output, and dispatch live here, outside the core library.
class PlannerCLI {
 public:
  static Vector vector(std::istream& in, int n) {
    Vector v(n);
    for (int i = 0; i < n; ++i) in >> v[i];
    return v;
  }
  explicit PlannerCLI(const std::string& path) {
    std::ifstream in(path);
    int count;
    in >> dimension_ >> count >> horizon_ >> speed_ >> query_.start_time >> query_.stay_at_goal;
    if (!in || dimension_ < 1 || count < 1) throw std::runtime_error("Invalid instance header");
    query_.start = vector(in, dimension_);
    query_.goal = vector(in, dimension_);
    for (int i = 0; i < count; ++i) {
      int rows;
      in >> rows;
      Matrix a(rows, dimension_);
      for (int j = 0; j < rows; ++j) a.row(j) = vector(in, dimension_).transpose();
      regions_.emplace_back(a, vector(in, rows));
    }
    in >> count;
    for (int i = 0; i < count; ++i) {
      double radius;
      int size;
      in >> radius >> size;
      std::vector<Segment> segments;
      for (int j = 0; j < size; ++j)
        segments.push_back({vector(in, dimension_ + 1), vector(in, dimension_ + 1), -1});
      obstacles_.emplace_back(radius, Trajectory(std::move(segments)));
    }
    if (!in) throw std::runtime_error("Truncated instance");
  }

  template <class T>
  static void write_scalar(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  template <class T>
  static void read_scalar(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
  }
  static void check_cache_encoding() {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    const std::uint32_t one = 1;
    if (*reinterpret_cast<const unsigned char*>(&one) != 1)
      throw std::runtime_error("Native cache requires little-endian encoding");
  }
  template <class Map>
  static void write_map(std::ostream& out, const Map& map) {
    write_scalar(out, static_cast<std::uint64_t>(map.size()));
    for (const auto& [key, value] : map) {
      std::apply([&](auto... indices) { (write_scalar(out, static_cast<std::int32_t>(indices)), ...); }, key);
      write_scalar(out, value);
    }
  }
  template <class Map>
  static void read_map(std::istream& in, Map& map) {
    std::uint64_t size = 0;
    read_scalar(in, size);
    if (!in || size > 100000000) throw std::runtime_error("Invalid cache map");
    for (std::size_t i = 0; i < size; ++i) {
      typename Map::key_type key{};
      double value = 0;
      std::apply([&](auto&... indices) {
        (([&] { std::int32_t index = 0; read_scalar(in, index); indices = index; }()), ...);
      }, key);
      read_scalar(in, value);
      if (!in || !std::isfinite(value) || value < 0)
        throw std::runtime_error("Invalid cache value");
      map.emplace(key, value);
    }
  }
  static void save(const std::string& path, const HeuristicCache::Data& data) {
    check_cache_encoding();
    std::ofstream out(path, std::ios::binary);
    out.write("STGCACHE", 8);
    write_scalar(out, static_cast<std::uint8_t>(data.complete));
    write_scalar(out, static_cast<std::uint8_t>(data.has_table));
    write_scalar(out, data.triplet_seconds);
    write_scalar(out, data.table_seconds);
    write_map(out, data.pairs);
    write_map(out, data.incoming);
    write_map(out, data.sets);
    write_map(out, data.table);
    write_map(out, data.triplets);
    if (!out) throw std::runtime_error("Cannot write cache");
  }
  static HeuristicCache::Data load(const std::string& path) {
    check_cache_encoding();
    std::ifstream in(path, std::ios::binary);
    char signature[8]{};
    std::uint8_t complete = 0, has_table = 0;
    HeuristicCache::Data data;
    in.read(signature, sizeof(signature));
    read_scalar(in, complete);
    read_scalar(in, has_table);
    read_scalar(in, data.triplet_seconds);
    read_scalar(in, data.table_seconds);
    if (!in || std::string(signature, 8) != "STGCACHE" || complete != 1 || has_table > 1)
      throw std::runtime_error("Missing or incomplete native cache");
    data.complete = true;
    data.has_table = has_table != 0;
    read_map(in, data.pairs);
    read_map(in, data.incoming);
    read_map(in, data.sets);
    read_map(in, data.table);
    read_map(in, data.triplets);
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing cache data");
    return data;
  }
  static void array(const Vector& v) {
    std::cout << '[';
    for (int i = 0; i < v.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << v[i];
    }
    std::cout << ']';
  }
  static PlannerOptions search_options(const std::string& heuristic,
                                       const std::string& dominance, double epsilon, double budget) {
    PlannerOptions options;
    if (heuristic == "h_zero") options.heuristics = {Heuristic::Zero};
    else if (heuristic == "h_mot") options.heuristics = {Heuristic::Motion};
    else if (heuristic == "h_tri") options.heuristics = {Heuristic::Triplet};
    else if (heuristic == "h_mot_tri") options.heuristics = {Heuristic::Motion, Heuristic::Triplet};
    else if (heuristic == "h_tab") options.heuristics = {Heuristic::CostTable};
    else if (heuristic == "h_max")
      options.heuristics = {Heuristic::Motion, Heuristic::Triplet, Heuristic::CostTable};
    else throw std::runtime_error("Unknown heuristic");
    if (dominance == "delta_set") options.dominance = {Dominance::SetVertex};
    else if (dominance == "delta_pos") options.dominance = {Dominance::Position};
    else if (dominance == "delta_state") options.dominance = {Dominance::ArrivalState};
    else if (dominance == "none") options.dominance = {Dominance::None};
    else throw std::runtime_error("Unknown dominance");
    options.inflation = epsilon;
    options.timeout = budget;
    options.global_upper_bound = true;
    options.position_samples = 1;
    return options;
  }
  static void segments(const Trajectory& path) {
    std::cout << '[';
    bool first = true;
    for (const auto& segment : path.segments()) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << '[';
      array(segment.start);
      std::cout << ',';
      array(segment.end);
      std::cout << ']';
    }
    std::cout << ']';
  }
  static void profile(const SearchProfile& value) {
    std::cout << '{';
    bool first = true;
    for (const auto& [name, entry] : std::vector<std::pair<const char*, const ProfileEntry*>>{
             {"gcs", &value.gcs}, {"gub", &value.gub}, {"search", &value.search},
             {"cr", &value.convex_restriction}, {"dc", &value.dominance_check},
             {"dc_cr", &value.dominance_convex_restriction}}) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << '\"' << name << "\":[" << entry->calls << ',' << entry->runtime << ']';
    }
    std::cout << '}';
  }
  void run_multi(const Graph& base, int argc, char** argv) const {
    if (argc != 14)
      throw std::runtime_error("mrmp INPUT CACHE HEURISTIC DOMINANCE EPS BUDGET METHOD RULE ALPHA BETA DYNAMIC QUERIES");
    if (!obstacles_.empty()) throw std::runtime_error("MRMP input requires a static base graph");
    auto options = search_options(argv[4], argv[5], std::stod(argv[6]), std::stod(argv[7]));
    // The Python MRMP runner loads h_tri(use_update=True) inside h_max.
    for (auto& heuristic : options.heuristics)
      if (heuristic == Heuristic::Triplet) heuristic = Heuristic::GoalUpdatedTriplet;
    const std::string method = argv[8], rule = argv[9];
    const double alpha = std::stod(argv[10]);
    if (!std::isfinite(alpha) || alpha < 0) throw std::runtime_error("Invalid window factor");
    if (std::string(argv[4]) != "h_zero" && std::string(argv[4]) != "h_mot")
      options.cache = std::make_shared<HeuristicCache>(HeuristicCache::restore(base, load(argv[3])));
    std::ifstream in(argv[13]);
    int count;
    MultiRobotOptions coordination;
    in >> count >> coordination.radius;
    if (!in || count < 1) throw std::runtime_error("Invalid MRMP query header");
    std::vector<Query> queries;
    for (int i = 0; i < count; ++i) {
      Query query;
      in >> query.start_time >> query.stay_at_goal;
      query.start = vector(in, dimension_);
      query.goal = vector(in, dimension_);
      query.validate(dimension_);
      queries.push_back(std::move(query));
    }
    if (!in) throw std::runtime_error("Truncated MRMP queries");
    if (rule == "lazy") coordination.child_order = ChildOrder::Lazy;
    else if (rule == "soc") coordination.child_order = ChildOrder::SumOfCosts;
    else if (rule == "makespan") coordination.child_order = ChildOrder::Makespan;
    else if (rule == "num_conflicts") coordination.child_order = ChildOrder::Conflicts;
    else throw std::runtime_error("Unknown child expansion rule");
    coordination.timeout = options.timeout;
    MultiRobotResult result;
    if (method == "pp") coordination.method = Coordination::Prioritized;
    else if (method != "pbs") throw std::runtime_error("Unknown MRMP method");
    MultiRobotPlanner planner(Planner(options), coordination);
    result = alpha == 0 ? planner.plan(base, queries)
                       : planner.plan_windowed(base, queries,
                           {alpha * coordination.radius / speed_, std::stod(argv[11]),
                            std::stoi(argv[12]) != 0});
    std::cout << "{\"event\":\"result\",\"backend\":\"stgcs_cplusplus\",\"success\":"
              << (result.success() ? "true" : "false")
              << ",\"status\":" << static_cast<int>(result.status) << ",\"runtime\":" << result.runtime
              << ",\"expanded\":" << result.expanded << ",\"generated\":" << result.generated
              << ",\"update_calls\":" << result.update_calls << ",\"windows\":" << result.windows
              << ",\"coordination_calls\":" << result.coordination_calls
              << ",\"coordination_runtime\":" << result.coordination_runtime
              << ",\"low_level_calls\":" << result.low_level_calls
              << ",\"low_level_runtime\":" << result.low_level_runtime
              << ",\"reservation_calls\":" << result.reservation_calls
              << ",\"reservation_runtime\":" << result.reservation_runtime
              << ",\"collision_calls\":" << result.collision_calls
              << ",\"collision_runtime\":" << result.collision_runtime
              << ",\"plan_cache_hits\":" << result.plan_cache_hits
              << ",\"reservation_cache_hits\":" << result.reservation_cache_hits
              << ",\"collision_cache_hits\":" << result.collision_cache_hits
              << ",\"obstacle_cache_hits\":" << result.obstacle_cache_hits
              << ",\"structural_cache_hits\":" << result.structural_cache_hits
              << ",\"profile\":";
    profile(result.profile);
    std::cout << ",\"trajectories\":[";
    for (std::size_t i = 0; i < result.trajectories.size(); ++i) {
      if (i) std::cout << ',';
      segments(result.trajectories[i]);
    }
    std::cout << "]}" << std::endl;
  }
  // A witness may cross several encoded cells within one straight segment.
  static bool covered(const Graph& graph, const Segment& segment) {
    std::vector<std::pair<double, double>> intervals;
    for (const auto& vertex : graph.vertices()) {
      double lo = 0, hi = 1;
      const auto& set = vertex.set;
      for (int i = 0; i < set.A().rows() && lo <= hi; ++i) {
        const double slope = set.A().row(i).dot(segment.end - segment.start);
        const double rhs = set.b()[i] + 1e-8 - set.A().row(i).dot(segment.start);
        if (std::abs(slope) < 1e-12) {
          if (rhs < 0) { lo = 1; hi = 0; }
        } else if (slope > 0) hi = std::min(hi, rhs / slope);
        else lo = std::max(lo, rhs / slope);
      }
      if (lo <= hi && hi >= 0 && lo <= 1)
        intervals.emplace_back(std::max(0., lo), std::min(1., hi));
    }
    std::sort(intervals.begin(), intervals.end());
    double end = 0;
    for (auto [lo, hi] : intervals) {
      if (lo > end + 1e-8) return false;
      end = std::max(end, hi);
    }
    return end >= 1 - 1e-8;
  }

  void validate_witness(const Graph& base, const std::string& path) const {
    std::ifstream in(path);
    int count = 0;
    in >> count;
    if (!in || count < 1) throw std::runtime_error("Invalid witness header");
    std::vector<Segment> legs;
    for (int i = 0; i < count; ++i) {
      legs.push_back({vector(in, dimension_ + 1), vector(in, dimension_ + 1), -1});
      if (!in || !legs.back().start.allFinite() || !legs.back().end.allFinite())
        throw std::runtime_error("Invalid witness segment");
    }
    const Trajectory trajectory(legs);
    if ((trajectory.start().head(dimension_) - query_.start).norm() > 1e-8 ||
        (trajectory.end().head(dimension_) - query_.goal).norm() > 1e-8 ||
        std::abs(trajectory.start().tail(1)[0] - query_.start_time) > 1e-8 ||
        trajectory.end().tail(1)[0] > horizon_ + 1e-8)
      throw std::runtime_error("Witness endpoints do not match query");
    for (const auto& [radius, obstacle] : obstacles_)
      if (Collision::check(trajectory, obstacle, radius, query_.start_time, horizon_,
                           {false, query_.stay_at_goal}, {}, 1e-6))
        throw std::runtime_error("Continuous witness collision");
    if (query_.stay_at_goal && trajectory.end().tail(1)[0] < horizon_) {
      auto end = trajectory.end();
      end.tail(1)[0] = horizon_;
      legs.push_back({trajectory.end(), end, -1});
    }
    const auto graph = base.reserve(obstacles_);
    Vector previous = trajectory.start();
    for (const auto& segment : legs) {
      if ((segment.start - previous).lpNorm<Eigen::Infinity>() > 1e-8 ||
          segment.end_time() < segment.start_time() - 1e-8 ||
          (segment.end - segment.start).head(dimension_).lpNorm<Eigen::Infinity>() >
              speed_ * (segment.end_time() - segment.start_time()) + 1e-8)
        throw std::runtime_error("Witness continuity or speed violation");
      if (!covered(graph, segment))
        throw std::runtime_error("Witness not continuously covered by native ST cells");
      previous = segment.end;
    }
    std::cout << "{\"valid\":true,\"vertices\":" << graph.vertices().size()
              << ",\"edges\":" << graph.edge_count()
              << ",\"arrival\":" << trajectory.end().tail(1)[0] << "}" << std::endl;
  }
  void run(int argc, char** argv) const {
    Graph base(regions_, 0, horizon_, speed_);
    if (std::string(argv[1]) == "inspect") {
      if (argc != 3) throw std::runtime_error("inspect INPUT");
      const auto graph = base.reserve(obstacles_);
      std::cout << "{\"st_vertices\":" << graph.vertices().size()
                << ",\"st_edges\":" << graph.edge_count() << "}" << std::endl;
      return;
    }
    if (std::string(argv[1]) == "validate") {
      if (argc != 4) throw std::runtime_error("validate INPUT WITNESS");
      validate_witness(base, argv[3]);
      return;
    }
    if (std::string(argv[1]) == "mrmp") { run_multi(base, argc, argv); return; }
    if (std::string(argv[1]) == "cache") {
      if (argc != 5 && argc != 6) throw std::runtime_error("cache INPUT OUTPUT TIMEOUT [COST_TABLE=1]");
      if (argc == 6 && std::string(argv[5]) != "0" && std::string(argv[5]) != "1")
        throw std::runtime_error("COST_TABLE must be 0 or 1");
      const bool cost_table = argc == 5 || std::string(argv[5]) == "1";
      const auto cache = HeuristicCache::build(base, cost_table, std::stod(argv[4]));
      const auto& data = cache.data();
      const auto encoded = base.reserve(obstacles_);
      std::cout << "{\"complete\":" << (cache.complete() ? "true" : "false")
                << ",\"has_table\":" << (data.has_table ? "true" : "false")
                << ",\"h_tri\":" << data.triplet_seconds << ",\"h_tab\":" << data.table_seconds
                << ",\"vertices\":" << base.vertices().size() << ",\"edges\":" << base.edge_count()
                << ",\"st_vertices\":" << encoded.vertices().size() << ",\"st_edges\":" << encoded.edge_count()
                << "}" << std::endl;
      if (!cache.complete()) throw std::runtime_error("Native heuristic cache timed out");
      save(argv[3], data);
      return;
    }
    if (std::string(argv[1]) != "plan" || argc != 13)
      throw std::runtime_error("plan INPUT CACHE HEURISTIC DOMINANCE EPS BUDGET METHOD SEED ROUNDED KEY_EVALUATION GUB_HEURISTIC");
    auto options = search_options(argv[4], argv[5], std::stod(argv[6]), std::stod(argv[7]));
    const std::string heuristic = argv[4], method = argv[8];
    options.seed = std::stoi(argv[9]);
    options.rounded_paths = std::stoi(argv[10]);
    const std::string evaluation = argv[11], upper_bound_heuristic = argv[12];
    if (evaluation == "joint") options.key_evaluation = KeyEvaluation::Joint;
    else if (evaluation != "separate") throw std::runtime_error("Unknown key evaluation");
    if (upper_bound_heuristic == "h_mot") options.upper_bound_heuristics = {Heuristic::Motion};
    else if (upper_bound_heuristic == "h_max")
      options.upper_bound_heuristics = search_options("h_max", "none", 1, options.timeout).heuristics;
    else if (upper_bound_heuristic != "same") throw std::runtime_error("Unknown GUB heuristic");
    if (method == "micp") options.method = Method::MixedInteger;
    else if (method == "micpg") options.method = Method::RoundedRelaxation;
    else if (method != "search") throw std::runtime_error("Unknown method");
    if (method == "search" && ((heuristic != "h_zero" && heuristic != "h_mot") ||
                               upper_bound_heuristic == "h_max"))
      options.cache = std::make_shared<HeuristicCache>(HeuristicCache::restore(base, load(argv[3])));
    auto graph = base.reserve(obstacles_);
    std::cout << "{\"event\":\"ready\",\"vertices\":" << graph.vertices().size()
              << ",\"edges\":" << graph.edge_count() << "}" << std::endl;
    const auto result = Planner(options).plan(graph, query_);
    const bool success = result.success();
    std::cout << "{\"event\":\"result\",\"success\":" << (success ? "true" : "false")
              << ",\"status\":" << static_cast<int>(result.status)
              << ",\"runtime\":" << result.runtime
              << ",\"cost\":";
    if (success) std::cout << result.trajectory->duration();
    else std::cout << "null";
    std::cout << ",\"expanded\":" << result.expanded << ",\"generated\":" << result.generated
              << ",\"restrictions\":" << result.restrictions
              << ",\"segments\":[";
    if (result.trajectory) {
      bool first = true;
      for (const auto& s : result.trajectory->segments()) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << '[';
        array(s.start);
        std::cout << ',';
        array(s.end);
        std::cout << ']';
      }
    }
    std::cout << "]}" << std::endl;
  }

 private:
  int dimension_{};
  double horizon_{}, speed_{};
  Query query_;
  std::vector<Polyhedron> regions_;
  std::vector<std::pair<double, Trajectory>> obstacles_;
};

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("Expected cache, inspect, validate, plan, or mrmp command and input path");
    std::cout << std::setprecision(17);
    PlannerCLI(argv[2]).run(argc, argv);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
