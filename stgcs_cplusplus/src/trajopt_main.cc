#include "stgcs/global_trajopt.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace stgcs;
using namespace stgcs::trajopt;

// This protocol transports data only; optimization and validation live in C++.
class Runner {
 public:
  static double number(std::istream& in) {
    std::string token;
    if (!(in >> token)) throw std::runtime_error("Truncated native trajopt input");
    return std::stod(token);
  }
  static int count(std::istream& in) {
    double value = number(in);
    if (!std::isfinite(value) || value < 0 || value > 10000000 || value != std::floor(value))
      throw std::runtime_error("Invalid native input count");
    return int(value);
  }
  static Vector vector(std::istream& in, int size) {
    Vector v(size); for (int i=0; i<size; ++i) v[i] = number(in); return v;
  }
  static Problem read(std::istream& in) {
    Problem p;
    p.radius = number(in); p.speed = number(in);
    auto& c = p.config;
    c.sample_dt = number(in); c.min_sample_dt = number(in);
    c.velocity_gradient_weight = number(in); c.displacement_weight = number(in);
    c.displacement_jitter_weight = number(in); c.clearance_margin = number(in);
    c.spatial_containment_tolerance = number(in); c.spatial_containment_soft_margin = number(in);
    c.constraint_slack_penalty = number(in); c.constraint_slack_limit = number(in);
    c.include_interval_collision_constraints = count(in); c.fix_terminal_states = count(in);
    c.include_trajectory_knot_times = count(in); c.solver_max_iter = count(in);
    c.solver_eps_abs = number(in); c.solver_eps_rel = number(in);
    c.solver_print_to_console = count(in); c.progress = count(in); c.round_time_decimals = count(in);
    int dim = count(in), agents = count(in);
    if (!dim || !agents) throw std::runtime_error("Empty native trajopt input");
    for (int a=0; a<agents; ++a) {
      int n = count(in); std::vector<Segment> segments;
      for (int k=0; k<n; ++k) {
        int vertex = int(number(in));
        Vector start = vector(in, dim+1), end = vector(in, dim+1);
        segments.push_back({start, end, vertex});
      }
      p.trajectories.emplace_back(std::move(segments));
    }
    int n = count(in);
    for (int k=0; k<n; ++k) {
      int rows = count(in); Matrix A(rows, dim+1);
      for (int i=0; i<rows; ++i) A.row(i) = vector(in, dim+1).transpose();
      p.cells.emplace_back(A, vector(in, rows));
    }
    if (count(in)) {
      Environment env;
      env.lower = vector(in, dim); env.upper = vector(in, dim);
      n = count(in);
      for (int k=0; k<n; ++k) {
        Obstacle o;
        if (count(in)) { o.center = vector(in, dim); o.radius = number(in); }
        else {
          int rows = count(in); o.A.resize(rows, dim);
          for (int i=0; i<rows; ++i) o.A.row(i) = vector(in, dim).transpose();
          o.b = vector(in, rows);
        }
        env.obstacles.push_back(std::move(o));
      }
      n = count(in);
      for (int k=0; k<n; ++k) {
        int rows = count(in); Matrix A(rows, dim);
        for (int i=0; i<rows; ++i) A.row(i) = vector(in, dim).transpose();
        env.regions.emplace_back(A, vector(in, rows));
      }
      p.environment = std::move(env);
    }
    return p;
  }
  static void number(double value) {
    if (std::isfinite(value)) std::cout << value; else std::cout << "null";
  }
  static void vector(const Vector& v) {
    std::cout << '[';
    for (int i=0; i<v.size(); ++i) { if (i) std::cout << ','; number(v[i]); }
    std::cout << ']';
  }
  static void paths(const std::vector<Trajectory>& trajectories) {
    std::cout << '[';
    for (size_t a=0; a<trajectories.size(); ++a) {
      if (a) std::cout << ',';
      std::cout << "{\"vertices\":[";
      const auto& segments = trajectories[a].segments();
      for (size_t k=0; k<segments.size(); ++k) { if (k) std::cout << ','; std::cout << segments[k].vertex; }
      std::cout << "],\"points\":[";
      for (size_t k=0; k<segments.size(); ++k) {
        if (k) std::cout << ',';
        Vector values(segments[k].start.size()*2); values << segments[k].start, segments[k].end;
        vector(values);
      }
      std::cout << "]}";
    }
    std::cout << ']';
  }
  static void positions(const Positions& p) {
    std::cout << '[';
    for (size_t a=0; a<p.size(); ++a) {
      if (a) std::cout << ',';
      std::cout << '[';
      for (int k=0; k<p[a].rows(); ++k) { if (k) std::cout << ','; vector(p[a].row(k)); }
      std::cout << ']';
    }
    std::cout << ']';
  }
  static void optional(std::optional<bool> value) {
    if (value.has_value()) std::cout << (*value ? "true" : "false"); else std::cout << "null";
  }
  static void result(const Result& r) {
    std::cout << "{\"backend\":\"stgcs_cplusplus\",\"times\":";
    vector(Eigen::Map<const Vector>(r.times.data(), r.times.size()));
    std::cout << ",\"original_positions\":"; positions(r.original);
    std::cout << ",\"optimized_positions\":"; positions(r.optimized);
    std::cout << ",\"trajectories\":"; paths(r.trajectories);
    std::cout << ",\"runtime\":" << r.runtime << ",\"min_pairwise_distance\":"; number(r.min_distance);
    std::cout << ",\"max_velocity_component\":"; number(r.max_velocity);
    std::cout << ",\"sampled_pairwise_collision_free\":" << (r.sampled_pairwise_ok ? "true" : "false");
    std::cout << ",\"continuous_pairwise_collision_free\":" << (r.continuous_pairwise_ok ? "true" : "false");
    std::cout << ",\"min_continuous_pairwise_distance\":"; number(r.min_continuous_distance);
    std::cout << ",\"sampled_environment_collision_free\":"; optional(r.sampled_environment_ok);
    std::cout << ",\"continuous_environment_collision_free\":"; optional(r.continuous_environment_ok);
    std::cout << ",\"solver_message\":" << std::quoted(r.solver_message) << "}\n";
  }
  static void run(const std::string& mode, std::istream& in) {
    Problem p = read(in);
    if (mode == "optimize") {
      double window = number(in), stride = number(in);
      result(Optimizer::optimize(p, window, stride));
    } else if (mode == "inspect") {
      int n = count(in); Vector values = vector(in, n);
      std::vector<double> times(values.data(), values.data()+values.size());
      result(Optimizer::inspect(p, times));
    } else if (mode == "reference") {
      std::cout << "{\"trajectories\":"; paths(Optimizer::reference_trajectories(p)); std::cout << "}\n";
    } else if (mode == "pairwise" || mode == "disk_pairwise") {
      std::vector<bool> stays;
      for (size_t i=0; i<p.trajectories.size(); ++i) stays.push_back(count(in));
      double tolerance = number(in);
      bool ok = mode == "disk_pairwise" ? Optimizer::disk_pairwise_free(p.trajectories, p.radius, stays, tolerance) :
                                         Optimizer::pairwise_free(p.trajectories, p.radius, stays, tolerance);
      std::cout << "{\"ok\":" << (ok ? "true" : "false") << "}\n";
    } else if (mode == "environment") {
      double tolerance = number(in);
      std::cout << "{\"ok\":" << (Optimizer::environment_free(p, p.trajectories, tolerance) ? "true" : "false") << "}\n";
    } else throw std::invalid_argument("Unknown trajopt command");
  }
};

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("stgcs_trajopt COMMAND INPUT");
    std::ifstream in(argv[2]);
    std::cout << std::setprecision(17);
    Runner::run(argv[1], in);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
