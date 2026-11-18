#include "stgcs/trajectory.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stgcs {
Vector Segment::at(double time) const {
  const double dt = end_time() - start_time();
  Vector result =
      dt <= kTolerance
          ? start
          : Vector(start + std::clamp((time - start_time()) / dt, 0.0, 1.0) * (end - start));
  result[result.size() - 1] = time;
  return result;
}

Trajectory::Trajectory(std::vector<Segment> segments) : segments_(std::move(segments)) {
  for (auto& s : segments_) {
    if (s.start.size() < 2 || s.start.size() != s.end.size() || !s.start.allFinite() ||
        !s.end.allFinite())
      throw std::invalid_argument("Invalid trajectory segment");
    // Match STTrajectory._normalize_point: normalize only exactly zero-time
    // segments with spatial residual <= 1e-7. Solver acceptance belongs to the
    // caller, not an extra numerical policy in this data container.
    if (s.end_time() == s.start_time() &&
        (s.end - s.start).head(s.start.size() - 1).norm() <= 1e-7)
      s.end = s.start;
  }
}

const Vector& Trajectory::start() const {
  if (empty()) throw std::logic_error("Empty trajectory has no start");
  return segments_.front().start;
}
const Vector& Trajectory::end() const {
  if (empty()) throw std::logic_error("Empty trajectory has no end");
  return segments_.back().end;
}
double Trajectory::duration() const {
  return empty() ? 0 : segments_.back().end_time() - segments_.front().start_time();
}
Vector Trajectory::at(double time) const {
  if (empty()) throw std::logic_error("Cannot interpolate an empty trajectory");
  for (const auto& s : segments_)
    if (time <= s.end_time()) return s.at(time);
  return segments_.back().at(time);
}

Trajectory Trajectory::chunk(double begin, double finish, bool pad) const {
  if (finish < begin) throw std::invalid_argument("Reversed chunk interval");
  if (empty()) return {};
  std::vector<Segment> result;
  for (const auto& s : Collision::occupied(*this, begin, finish, {pad, pad})) {
    double a = std::max(begin, s.start_time()), b = std::min(finish, s.end_time());
    if (b > a + kTolerance) result.push_back({s.at(a), s.at(b), s.vertex});
  }
  if (result.empty() && begin == finish) result.push_back({at(begin), at(begin), -1});
  return Trajectory(std::move(result));
}

Trajectory Trajectory::shifted(double time) const {
  auto result = segments_;
  for (auto& s : result) {
    s.start[s.start.size() - 1] += time;
    s.end[s.end.size() - 1] += time;
  }
  return Trajectory(std::move(result));
}

void Trajectory::append(const Trajectory& chunk) {
  if (chunk.empty()) return;
  if (!empty() && end().size() != chunk.start().size())
    throw std::invalid_argument("Trajectory dimension mismatch");
  segments_.insert(segments_.end(), chunk.segments_.begin(), chunk.segments_.end());
}

std::vector<Segment> Collision::occupied(const Trajectory& path, double start, double end,
                                         Occupancy stay) {
  if (path.empty()) return {};
  std::vector<Segment> result;
  const auto& first = path.segments().front();
  const auto& last = path.segments().back();
  if (stay.before_start && start < first.start_time())
    result.push_back({first.at(start), first.start, first.vertex});
  result.insert(result.end(), path.segments().begin(), path.segments().end());
  if (stay.after_end && end > last.end_time())
    result.push_back({last.end, last.at(end), last.vertex});
  return result;
}

bool Collision::check(const Trajectory& a, const Trajectory& b, double radius, double start,
                      double end, Occupancy as, Occupancy bs, double tolerance) {
  if (!std::isfinite(radius) || radius < 0 || tolerance < 0 || end < start)
    throw std::invalid_argument("Invalid collision parameters");
  if (a.empty() || b.empty()) return false;
  if (a.start().size() != b.start().size()) throw std::invalid_argument("Dimension mismatch");
  const int d = a.start().size() - 1;
  const double r = std::max(0.0, radius - tolerance);
  if (r == 0) return false;
  for (const auto& sa : occupied(a, start, end, as))
    for (const auto& sb : occupied(b, start, end, bs)) {
      double lo = std::max({start, sa.start_time(), sb.start_time()});
      double hi = std::min({end, sa.end_time(), sb.end_time()});
      if (hi < lo) continue;
      Vector delta = (sa.at(lo) - sb.at(lo)).head(d);
      if ((delta.array().abs() < r).all()) return true;
      if (hi == lo) continue;
      Vector velocity = ((sa.at(hi) - sb.at(hi)).head(d) - delta) / (hi - lo);
      double left = 0, right = hi - lo;
      for (int i = 0; i < d && left <= right; ++i) {
        if (std::abs(velocity[i]) < 1e-14) {
          if (std::abs(delta[i]) >= r) right = -1;
        } else {
          double t1 = (-r - delta[i]) / velocity[i], t2 = (r - delta[i]) / velocity[i];
          left = std::max(left, std::min(t1, t2));
          right = std::min(right, std::max(t1, t2));
        }
      }
      if (left < right) return true;
    }
  return false;
}
}  // namespace stgcs
