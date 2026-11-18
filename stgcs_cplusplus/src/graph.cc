#include "stgcs/graph.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "internal.h"

namespace stgcs {
Graph::Graph(const std::vector<Polyhedron>& regions, double start, double end, double speed,
             double dt)
    : dimension_(regions.empty() ? 0 : regions.front().ambient_dimension()),
      start_(start),
      end_(end),
      speed_(speed),
      minimum_duration_(dt) {
  if (dimension_ < 1 || !std::isfinite(start) || !std::isfinite(end) || end <= start ||
      !std::isfinite(speed) || speed <= 0 || !std::isfinite(dt) || dt <= 0)
    throw std::invalid_argument("Require regions, finite time bounds, positive speed and dt");
  using drake::solvers::MosekSolver;
  if (MosekSolver::is_available() && MosekSolver::is_enabled())
    solver_license_ = MosekSolver::AcquireLicense();
  for (std::size_t i = 0; i < regions.size(); ++i) {
    if (regions[i].ambient_dimension() != dimension_ || !regions[i].IsBounded())
      throw std::invalid_argument("Regions must be bounded and have the same dimension");
    add(Geometry::extrude(regions[i], start, end), i);
  }
  connect();
}

bool Graph::has_width(const Bounds& bounds) const {
  const Vector width = bounds.upper - bounds.lower;
  // Keep time slices produced by minimum-duration motions. Applying the
  // spatial 1e-6 cutoff to time erases those slices and their query sources.
  return width.head(dimension_).minCoeff() > kMinimumRegionWidth &&
         width[dimension_] > std::max(kTolerance, minimum_duration_ - kTolerance);
}

bool Graph::add(const Polyhedron& set, int root) {
  auto bounds = Geometry::bounds(set, kTolerance);
  if (!bounds || !has_width(*bounds)) return false;
  vertices_.push_back({set, std::move(*bounds), root});
  return true;
}

void Graph::connect_pair(int first, int second) {
  if (first > second) std::swap(first, second);
  const auto& a = vertices_[first];
  const auto& b = vertices_[second];
  if (!a.bounds.overlaps(b.bounds) || !a.set.IntersectsWith(b.set)) return;
  if (a.bounds.lower[dimension_] < b.bounds.upper[dimension_] - kTolerance)
    edges_[first].push_back(second);
  if (b.bounds.lower[dimension_] < a.bounds.upper[dimension_] - kTolerance)
    edges_[second].push_back(first);
}

void Graph::connect() {
  optimization_cache_ = std::make_shared<OptimizationCache>();
  edges_.assign(vertices_.size(), {});
  for (std::size_t i = 0; i < vertices_.size(); ++i)
    for (std::size_t j = i + 1; j < vertices_.size(); ++j) connect_pair(i, j);
}

void Graph::connect(const Graph& original, const std::vector<int>& parents,
                    const std::vector<bool>& modified) {
  optimization_cache_ = std::make_shared<OptimizationCache>();
  edges_.assign(vertices_.size(), {});
  std::vector<std::vector<int>> children(original.vertices_.size());
  for (std::size_t i = 0; i < parents.size(); ++i) children[parents[i]].push_back(i);
  // Clipping cannot create intersections between previously disjoint cells.
  // Recheck only siblings and descendants of old neighbors, retaining untouched edges.
  for (std::size_t i = 0; i < children.size(); ++i) {
    const auto& family = children[i];
    for (std::size_t u = 0; u < family.size(); ++u)
      for (std::size_t v = u + 1; v < family.size(); ++v) connect_pair(family[u], family[v]);
    for (int j : original.edges_[i]) {
      if (!modified[i] && !modified[j]) {
        edges_[family.front()].push_back(children[j].front());
        continue;
      }
      if (i > static_cast<std::size_t>(j) &&
          std::binary_search(original.edges_[j].begin(), original.edges_[j].end(), i)) continue;
      for (int u : family)
        for (int v : children[j]) connect_pair(u, v);
    }
  }
  // Preserve the full rebuild's deterministic search order.
  for (auto& neighbors : edges_) std::sort(neighbors.begin(), neighbors.end());
}

std::size_t Graph::edge_count() const {
  std::size_t count = 0;
  for (const auto& edges : edges_) count += edges.size();
  return count;
}

Graph Graph::reserve(const Trajectory& obstacle, double radius, Occupancy stay) const {
  return reserve(obstacle_volumes(obstacle, radius, stay));
}

std::vector<SweptVolume> Graph::obstacle_volumes(const Trajectory& obstacle, double radius,
                                                Occupancy stay) const {
  if (!std::isfinite(radius) || radius < 0) throw std::invalid_argument("Invalid safe radius");
  if (obstacle.empty() || radius == 0) return {};
  if (obstacle.start().size() != dimension_ + 1)
    throw std::invalid_argument("Obstacle dimension mismatch");
  std::vector<SweptVolume> volumes;
  std::vector<Segment> segments;
  for (const auto& segment : Collision::occupied(obstacle, start_, end_, stay)) {
    const double duration = segment.end_time() - segment.start_time();
    if (duration <= kTolerance) continue;
    if (!segments.empty()) {
      auto& last = segments.back();
      const Vector before = (last.end - last.start) / (last.end_time() - last.start_time());
      const Vector after = (segment.end - segment.start) / duration;
      if ((before - after).lpNorm<Eigen::Infinity>() < 1e-10 &&
          (last.end - segment.start).lpNorm<Eigen::Infinity>() < kTolerance) {
        last.end = segment.end;
        continue;
      }
    }
    segments.push_back(segment);
  }
  for (const auto& segment : segments) {
    const double lo = std::max(start_, segment.start_time());
    const double hi = std::min(end_, segment.end_time());
    if (hi <= lo + kTolerance) continue;
    const Vector p = segment.at(lo), q = segment.at(hi);
    const Vector velocity = (q.head(dimension_) - p.head(dimension_)) / (hi - lo);
    Matrix a = Matrix::Zero(2 * dimension_ + 2, dimension_ + 1);
    Vector b(2 * dimension_ + 2);
    a(0, dimension_) = -1;
    b[0] = -lo;
    a(1, dimension_) = 1;
    b[1] = hi;
    for (int i = 0; i < dimension_; ++i)
      for (int sign : {-1, 1}) {
        const int row = 2 + 2 * i + (sign == 1);
        a(row, i) = sign;
        a(row, dimension_) = -sign * velocity[i];
        b[row] = sign * (p[i] - velocity[i] * lo) + radius;
      }
    Bounds swept_bounds{p.cwiseMin(q), p.cwiseMax(q)};
    swept_bounds.lower.head(dimension_).array() -= radius;
    swept_bounds.upper.head(dimension_).array() += radius;
    volumes.push_back({Polyhedron(a, b), std::move(swept_bounds)});
  }
  return volumes;
}

void Graph::subtract(const std::vector<SweptVolume>& volumes, std::vector<int>& parents,
                     std::vector<bool>& modified) {
  for (const auto& [swept, swept_bounds] : volumes) {
    const auto& a = swept.A();
    const auto& b = swept.b();
    auto old = std::move(vertices_);
    auto old_parents = std::move(parents);
    vertices_.clear();
    parents.clear();
    for (std::size_t index = 0; index < old.size(); ++index) {
      const auto& vertex = old[index];
      const int parent = old_parents[index];
      if (!vertex.bounds.overlaps(swept_bounds) || !vertex.set.IntersectsWith(swept)) {
        vertices_.push_back(vertex);
        parents.push_back(parent);
        continue;
      }
      modified[parent] = true;
      // Successive halfspaces partition P \ interior(swept) into convex pieces.
      auto inside = vertex.set;
      auto inside_bounds = vertex.bounds;
      for (int i = 0; i < a.rows(); ++i) {
        // Clipping a box gives a conservative enclosure without an LP. Reject
        // numerical slivers before solving any bounds LP on their ill-conditioned
        // constraints, independently of which coordinate is narrow.
        const auto outside_bounds = inside_bounds.clip(-a.row(i).transpose(), -b[i]);
        if (has_width(outside_bounds) &&
            add(Geometry::clip(inside, -a.row(i).transpose(), -b[i]), vertex.root))
          parents.push_back(parent);
        inside_bounds = inside_bounds.clip(a.row(i).transpose(), b[i]);
        if (!has_width(inside_bounds)) break;
        inside = Geometry::clip(inside, a.row(i).transpose(), b[i]);
        if (inside.IsEmpty()) break;
      }
    }
  }
}

Graph Graph::reserve(const std::vector<SweptVolume>& volumes) const {
  if (volumes.empty()) return *this;
  Graph result = *this;
  result.is_base_ = false;
  std::vector<int> parents(vertices_.size());
  std::iota(parents.begin(), parents.end(), 0);
  std::vector<bool> modified(vertices_.size(), false);
  result.subtract(volumes, parents, modified);
  result.connect(*this, parents, modified);
  return result;
}

Graph Graph::reserve(const std::vector<std::pair<double, Trajectory>>& obstacles,
                     Occupancy stay) const {
  Graph result = *this;
  bool changed = false;
  std::vector<int> parents(vertices_.size());
  std::iota(parents.begin(), parents.end(), 0);
  std::vector<bool> modified(vertices_.size(), false);
  for (const auto& [radius, obstacle] : obstacles) {
    if (!std::isfinite(radius) || radius < 0) throw std::invalid_argument("Invalid safe radius");
    if (obstacle.empty() || radius == 0) continue;
    if (obstacle.start().size() != dimension_ + 1)
      throw std::invalid_argument("Obstacle dimension mismatch");
    result.is_base_ = false;
    result.subtract(obstacle_volumes(obstacle, radius, stay), parents, modified);
    changed = true;
  }
  if (changed) result.connect(*this, parents, modified);
  return result;
}

Graph Graph::time_slice(double start, double end) const {
  if (start < start_ || end > end_ || start >= end)
    throw std::invalid_argument("Time slice must lie inside graph horizon");
  Graph result = *this;
  result.start_ = start;
  result.is_base_ = false;
  result.end_ = end;
  result.vertices_.clear();
  std::vector<int> parents;
  for (std::size_t i = 0; i < vertices_.size(); ++i)
    if (result.add(Geometry::time_slice(vertices_[i].set, start, end), vertices_[i].root))
      parents.push_back(i);
  result.connect(*this, parents, std::vector<bool>(vertices_.size(), true));
  return result;
}
}  // namespace stgcs
