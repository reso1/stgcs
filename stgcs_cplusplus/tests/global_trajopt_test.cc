#include "stgcs/global_trajopt.h"

#include <stdexcept>

int main() {
  stgcs::trajopt::Problem problem;
  problem.radius = 0.1;
  problem.speed = 2.0;
  problem.config.sample_dt = 0.25;
  for (double y : {0.0, 2.0}) {
    const Eigen::Vector3d start(-1, y, 0), bend(0, y + 0.6, 1), end(1, y, 2);
    problem.trajectories.emplace_back(std::vector<stgcs::Segment>{{start, bend}, {bend, end}});
  }

  const auto result = stgcs::trajopt::Optimizer::optimize(problem);
  if (result.trajectories.size() != problem.trajectories.size() ||
      !result.continuous_pairwise_ok || result.max_velocity > problem.speed + 1e-6)
    throw std::runtime_error("Expected two collision-free trajectories within the speed limit");
  for (std::size_t i = 0; i < result.trajectories.size(); ++i) {
    const auto& input = problem.trajectories[i];
    const auto& output = result.trajectories[i];
    if (!output.start().isApprox(input.start()) || !output.end().isApprox(input.end()) ||
        !(output.at(1)[1] < input.at(1)[1] - 1e-4))
      throw std::runtime_error("Expected a smoother bend with fixed endpoints");
  }
}
