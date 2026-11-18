#pragma once

#include <vector>

#include "stgcs/geometry.h"

namespace stgcs {
struct Segment {
  Vector start, end;  // Spatial coordinates followed by time.
  int vertex{-1};
  double start_time() const { return start[start.size() - 1]; }
  double end_time() const { return end[end.size() - 1]; }
  Vector at(double time) const;
};

class Trajectory {
 public:
  Trajectory() = default;
  explicit Trajectory(std::vector<Segment> segments);
  const std::vector<Segment>& segments() const { return segments_; }
  bool empty() const { return segments_.empty(); }
  const Vector& start() const;
  const Vector& end() const;
  double duration() const;
  Vector at(double time) const;
  Trajectory chunk(double start, double end, bool pad = false) const;
  Trajectory shifted(double time) const;
  void append(const Trajectory& chunk);

 private:
  std::vector<Segment> segments_;
};

struct Occupancy {
  bool before_start{true};
  bool after_end{true};
};

class Collision {
 public:
  // Radii are L-infinity half-widths, as in ECD. Contact is permitted.
  static bool check(const Trajectory& a, const Trajectory& b, double combined_radius, double start,
                    double end, Occupancy a_stay = {}, Occupancy b_stay = {},
                    double tolerance = 1e-4);
  static std::vector<Segment> occupied(const Trajectory& trajectory, double start, double end,
                                       Occupancy stay);
};
}  // namespace stgcs
