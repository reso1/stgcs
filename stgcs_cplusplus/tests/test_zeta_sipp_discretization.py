import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from baselines.zeta_sipp import ZetaStarSIPPPlanner
from baselines.zeta_sipp.planner import (_DynamicIntervals, _KeyedHeap, _Node, _ProjectedGrid, _ZetaGeometry,
                                       _ZetaStarSIPP, _ZetaUtils)
from baselines.common import ShortestPathSolution
from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar
from baselines.pbs_zeta_sipp import ZetaSIPPPriorityBasedSearch
from experiments.environment.env import Env
from experiments.environment.maze.maze_env import make_2d_maze_env
from experiments.environment.obstacle import ConcatDynamicSphere, DynamicSphere, StaticPolygon
from experiments.runners.st_baselines import BaselineWorker
from experiments.environment.interval import Interval
from experiments.environment.space_time_graph import STGCS
from stgcs_native.types import MPQuery
from stgcs_native.types import STTrajectory


class KeyedHeapTests(unittest.TestCase):
    def test_old_priority_cannot_resurrect_a_closed_node(self):
        heap = _KeyedHeap(lambda node: node.f)
        node = SimpleNamespace(f=5., closed=False)
        heap.push(node)
        node.f = 3.
        heap.sort_element(node)
        self.assertIs(heap.shift(), node)
        # Revalidation raises the priority back to a previously stored value.
        node.f = 5.
        heap.push(node)
        self.assertIs(heap.shift(), node)
        node.closed = True
        self.assertEqual(heap.size(), 0)
        self.assertEqual(heap.peek_key(), float('inf'))
        with self.assertRaises(IndexError):
            heap.shift()

    def test_resident_updates_and_repeated_pushes_keep_one_entry(self):
        heap = _KeyedHeap(lambda node: node.f)
        node = SimpleNamespace(f=5.)
        for priority in (5., 3., 5., 1., 1., 7.):
            node.f = priority
            heap.push(node)
            heap.sort_element(node)
            self.assertEqual(heap.size(), 1)
            self.assertEqual(heap.peek_key(), priority)
        self.assertIs(heap.shift(), node)
        self.assertEqual(heap.size(), 0)

    def test_infinite_priority_remains_live_until_updated_or_removed(self):
        heap = _KeyedHeap(lambda node: node.f)
        node = SimpleNamespace(f=float('inf'))
        heap.push(node)
        self.assertEqual(heap.size(), 1)
        self.assertEqual(heap.peek_key(), float('inf'))
        self.assertEqual(heap.size(), 1)
        node.f = 2.
        heap.sort_element(node)
        self.assertIs(heap.shift(), node)
        heap.push(node)
        node.f = float('inf')
        heap.sort_element(node)
        self.assertEqual(heap.size(), 1)
        self.assertIs(heap.shift(), node)

    def test_equal_priority_entries_use_update_order(self):
        heap = _KeyedHeap(lambda node: node.f)
        a, b = SimpleNamespace(f=1.), SimpleNamespace(f=1.)
        heap.push(a)
        heap.push(b)
        heap.sort_element(a)
        self.assertIs(heap.shift(), b)
        self.assertIs(heap.shift(), a)
        self.assertEqual(heap.size(), 0)

    def test_mixed_updates_match_a_unique_item_priority_queue(self):
        rng = np.random.default_rng(12)
        nodes = [SimpleNamespace(f=0.) for _ in range(30)]
        heap = _KeyedHeap(lambda node: node.f)
        pending = {}
        sequence = 0
        for _ in range(1000):
            if pending and rng.random() < .35:
                index = min(pending, key=lambda i: pending[i])
                self.assertEqual(heap.peek_key(), pending[index][0])
                self.assertIs(heap.shift(), nodes[index])
                del pending[index]
            else:
                index = int(rng.integers(len(nodes)))
                priority = float(rng.integers(-2, 10)) if rng.random() < .9 else float('inf')
                nodes[index].f = priority
                heap.sort_element(nodes[index])
                pending[index] = (priority, sequence)
                sequence += 1
            self.assertEqual(heap.size(), len(pending))
        for index in sorted(pending, key=lambda i: pending[i]):
            self.assertIs(heap.shift(), nodes[index])
        self.assertEqual(heap.size(), 0)


class ZetaTransitionTimingTests(unittest.TestCase):
    @staticmethod
    def transition(start=(.5, .5), goal=(3.5, .5), source_interval=(0., 20.),
                   target_interval=(0., 20.), horizon=20., arrival=1.):
        env = Env('grid_timing', [ZetaDiscretizationTests.rectangle([0, 0], [4, 1])], .1)
        projection = _ProjectedGrid(env, horizon, 1.)
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        query = MPQuery(start, goal, arrival, False, 1.)
        planner = _ZetaStarSIPP(projection, query, horizon, 5., .001)
        parent = _Node(projection.world_to_grid(start), start, source_interval, projection.cell_for_point(start))
        parent.g = arrival
        node = _Node(projection.world_to_grid(goal), goal, target_interval, projection.cell_for_point(goal))
        return planner, parent, node

    def test_lower_bound_accounts_for_target_cell_entry(self):
        cases = [((.5, .5), (3.5, .5), 7.5),
                 ((.1, .2), (3.9, .8), 7.9),
                 ((0., .5), (2., .5), 7.),
                 ((3., .5), (1., .5), 8.),
                 ((.1, .5), (.8, .5), 7.7),
                 ((.5, .5), (.5, .5), 7.)]
        for start, goal, expected in cases:
            with self.subTest(start=start, goal=goal):
                planner, parent, node = self.transition(start, goal, target_interval=(7., 12.))
                with patch.object(_ProjectedGrid, 'trace_segment', side_effect=AssertionError('bound traced cells')):
                    bound = planner.potential_parent_lower_bound(parent, node)
                self.assertAlmostEqual(bound, expected)
                self.assertGreaterEqual(parent.g + planner.get_cost_sipp(parent, node), bound - 1e-9)

    def test_infeasible_endpoint_and_horizon_bounds_prune_before_tracing(self):
        for settings in ({'source_interval': (0., 1.4)},
                         {'target_interval': (0., 3.9)}, {'horizon': 3.9}):
            with self.subTest(settings=settings):
                planner, parent, node = self.transition(**settings)
                with patch.object(_ProjectedGrid, 'trace_segment', side_effect=AssertionError('pruned edge traced')):
                    planner.add_potential_parent(parent, node)
                    self.assertEqual(planner.get_cost_sipp(parent, node), float('inf'))
                self.assertEqual(node.potential_parents, [])
                self.assertEqual(planner.open_heap.size(), 0)

    def test_pruning_uses_verified_arrival_not_another_optimistic_bound(self):
        planner, parent, node = self.transition(target_interval=(7., 12.))
        node.g_low = 2.  # An optimistic candidate has not established a feasible path.
        planner.add_potential_parent(parent, node)
        self.assertEqual(node.potential_parents, [parent])
        self.assertAlmostEqual(node.g_low_array[0], 7.5)
        for incumbent in (7.4, 7.5):
            planner, parent, node = self.transition(target_interval=(7., 12.))
            node.g = incumbent
            planner.add_potential_parent(parent, node)
            self.assertEqual(node.potential_parents, [])
            self.assertEqual(planner.open_heap.size(), 0)

    def test_trimmed_safe_interval_endpoints_remain_admissible(self):
        planner, parent, node = self.transition(source_interval=(0., 1.5), target_interval=(0., 4.))
        self.assertEqual(planner.potential_parent_lower_bound(parent, node), 4.)
        self.assertEqual(planner.get_cost_sipp(parent, node), 3.)

    def test_endpoint_residence_bounds_match_ray_traversal(self):
        rng = np.random.default_rng(18)
        for _ in range(80):
            points = rng.uniform([0., 0.], [4., 1.], (2, 2))
            planner, parent, node = self.transition(*points, source_interval=(0., 100.),
                                                     target_interval=(80., 100.), horizon=100.)
            travel = planner.heuristic(parent, node)
            trace = planner.projection.trace_segment(parent.grid_point, node.grid_point, travel)
            source, target = parent.cell.base_index, node.cell.base_index
            first_time = max((g['dist'][1] for g in trace if (g['x'], g['y']) == source), default=0.)
            last_time = travel - min((g['dist'][0] for g in trace if (g['x'], g['y']) == target), default=travel)
            wait, limit = planner.endpoint_wait_bounds(parent, node, travel)
            self.assertAlmostEqual(wait, 80. - parent.g - travel + last_time)
            self.assertAlmostEqual(limit, min(100. - parent.g - first_time, 100. - parent.g - travel))

    def test_scalar_waiting_matches_independent_forbidden_departure_oracle(self):
        rng = np.random.default_rng(42)
        planner, parent, node = self.transition(source_interval=(0., 40.), target_interval=(0., 40.),
                                                horizon=40., arrival=.25)
        trace = [{'x': i, 'y': 0} for i in range(4)]
        relative = ((0., .5), (.5, 1.5), (1.5, 2.5), (2.5, 3.))
        for _ in range(60):
            forbidden = []
            for i, (entry, exit_time) in enumerate(relative):
                risks = sorted(rng.uniform(0., 20., 6).tolist())
                planner.projection.get_cell(i, 0).risk_interval = risks
                forbidden.extend((start - parent.g - exit_time, end - parent.g - entry)
                                 for start, end in zip(risks[::2], risks[1::2]))
            limit = float(rng.choice([5., 10., 20., 30.]))
            expected = 0.
            # Solve in departure-time space, independently of cell scan order.
            while True:
                conflicts = [end for start, end in forbidden if start - 1e-9 <= expected <= end + 1e-9]
                if not conflicts:
                    break
                expected = max(conflicts) + planner.time_buffer
                if expected > limit + 1e-9:
                    expected = float('inf')
                    break
            actual = planner.update_wait_time(trace, relative, 0., parent, limit)
            self.assertAlmostEqual(actual, expected)
            self.assertEqual(relative, ((0., .5), (.5, 1.5), (1.5, 2.5), (2.5, 3.)))
            self.assertEqual(planner.update_wait_time(trace, relative, 0., parent, limit), actual)


class ZetaDiscretizationTests(unittest.TestCase):
    @staticmethod
    def rectangle(lo, hi):
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                         [hi[0], hi[1]], [lo[0], hi[1]]], dtype=float)

    def assert_valid_solution(self, env, start, goal, factor, horizon=12):
        planner = ZetaStarSIPPPlanner(env, seed=0, vlimit=1,
                                     cell_size=factor * env.robot_radius, runtime_limit_secs=5)
        solution = planner.solve(start, goal, 0, horizon, is_stay=True)
        self.assertTrue(solution.is_success)
        BaselineWorker.validate_zeta_solution(
            env, MPQuery(start, goal, 0., True, 1.), horizon, solution)
        projection = planner.last_projection
        self.assertEqual(projection.cell_size, factor * env.robot_radius)
        expected_shape = tuple(np.ceil((env.ub - env.lb) / (factor * env.robot_radius)).astype(int))
        self.assertEqual(projection.shape, expected_shape)
        previous, previous_time = start, 0.0
        for segment in solution.trajectory:
            a, t0, b, t1 = segment[:2], segment[2], segment[3:5], segment[5]
            np.testing.assert_allclose(a, previous, atol=1e-8)
            self.assertAlmostEqual(t0, previous_time)
            self.assertGreaterEqual(t1, t0)
            self.assertLessEqual(np.max(np.abs(b - a)), t1 - t0 + 1e-8)
            self.assertTrue(env.is_segment_in_CSpace(a, b))
            self.assertFalse(any(o.is_colliding_lineseg(a, b, env.robot_radius) for o in env.O_Static))
            self.assertFalse(any(o.is_colliding_lineseg(a, b, t0, t1, env.robot_radius)
                                 for o in env.O_Dynamic))
            previous, previous_time = b, t1
        np.testing.assert_allclose(previous, goal, atol=1e-8)
        self.assertAlmostEqual(previous_time, solution.cost)
        self.assertLessEqual(previous_time, horizon)
        self.assertFalse(any(o.is_colliding_lineseg(goal, goal, previous_time, horizon, env.robot_radius)
                             for o in env.O_Dynamic))
        return projection

    def test_free_cell_can_cross_region_seam_but_not_a_hole(self):
        env = Env('grid_union', [self.rectangle([0, 0], [1, 1]),
                                 self.rectangle([1, 0], [2, 1])], .1)
        geometry = _ZetaGeometry(env)
        self.assertTrue(geometry.cell_is_free(np.array([1.2, .4]), .8))
        # All four corners are free; the cell's interior is not.
        ring = [self.rectangle([0, 0], [2, .4]), self.rectangle([0, 1.6], [2, 2]),
                self.rectangle([0, .4], [.4, 1.6]), self.rectangle([1.6, .4], [2, 1.6])]
        self.assertFalse(_ZetaGeometry(Env('grid_hole', ring, .1)).cell_is_free(np.ones(2), 2.))

    def test_whole_cell_static_clearance_in_every_domain(self):
        obstacle = StaticPolygon(self.rectangle([.68, .45], [.8, .55]))
        for name in ('maze_cell', 'grid_cell', 'iris_cell'):
            env = Env(name, [self.rectangle([0, 0], [1, 1])], .1, OStatic=[obstacle])
            geometry = _ZetaGeometry(env)
            self.assertTrue(geometry.point_is_free(np.array([.5, .5])))
            self.assertFalse(geometry.cell_is_free(np.array([.5, .5]), .2))

    def test_cell_safe_intervals_include_dynamic_edge_occupancy(self):
        obstacle = DynamicSphere(np.array([0., .27]), np.array([1., .27]), .01, Interval(0, 10))
        env = Env('maze_intervals', [self.rectangle([0, 0], [1, 1])], .1, ODynamic=[obstacle])
        for factor in (1, 2):
            projection = _ProjectedGrid(env, 10, factor * env.robot_radius)
            cell = projection.cell_for_point(np.array([.51, .11]))
            self.assertTrue(cell.weight)
            # The obstacle never reaches the center's robot footprint, but
            # overlaps positions near this cell's upper edge at t=5.5.
            self.assertEqual(projection.dynamic.blocked(cell.world_point, .1), [])
            self.assertTrue(any(lo <= 5.5 <= hi for lo, hi in
                                zip(cell.risk_interval[::2], cell.risk_interval[1::2])))
            self.assertFalse(any(n.safe_interval[0] <= 5.5 <= n.safe_interval[1] for n in cell.nodes))
            with patch.object(_DynamicIntervals, 'blocked', side_effect=AssertionError('intervals recomputed')):
                safe = projection.safe_intervals_for_point(np.array([.51, .11]))
            self.assertEqual(safe, [n.safe_interval for n in cell.nodes])
            planner = _ZetaStarSIPP(projection,
                MPQuery(np.array([.51, .11]), np.array([.91, .71]), 0., True, 1.), 10., 5., .001)
            start, goal = planner.init()
            self.assertEqual(start.nodes[0].safe_interval, cell.nodes[0].safe_interval)
            self.assertEqual(start.risk_interval, cell.risk_interval)
            np.testing.assert_array_equal(start.world_point, [.51, .11])
            self.assertEqual(goal.nodes[-1].safe_interval[1], 10.)

    def test_grid_trace_corners_and_arbitrary_endpoints(self):
        diagonal = _ZetaUtils.line_seg_on_grid(.5, .5, 2.5, 2.5, 2.)
        self.assertEqual([(g['x'], g['y']) for g in diagonal], [(0, 0), (1, 1), (2, 2)])
        reverse = _ZetaUtils.line_seg_on_grid(2.5, 2.5, .5, .5, 2.)
        self.assertEqual([(g['x'], g['y']) for g in reverse], [(2, 2), (1, 1), (0, 0)])
        # Starting exactly on a cell boundary in the negative direction must
        # enter the neighboring cell without a spurious zero-duration step.
        trace = _ZetaUtils.line_seg_on_grid(2., .25, .25, .25, 1.75)
        self.assertEqual([(g['x'], g['y']) for g in trace], [(1, 0), (0, 0)])
        np.testing.assert_allclose([g['dist'] for g in trace], [[0., 1.], [1., 1.75]])

    def test_grid_only_transitions_reject_static_walls(self):
        obstacle = StaticPolygon(self.rectangle([.8, 0], [1.2, 1]))
        env = Env('maze_wall', [self.rectangle([0, 0], [2, 1])], .1, OStatic=[obstacle])
        projection = _ProjectedGrid(env, 10, .2)
        query = MPQuery(np.array([.3, .3]), np.array([1.7, .3]), 0., False, 1.)
        planner = _ZetaStarSIPP(projection, query, 10., 5., .001)
        start, goal = planner.init()
        start.nodes[0].g = 0.
        with patch.object(_ZetaGeometry, 'segment_is_free', side_effect=AssertionError('exact geometry')):
            self.assertFalse(projection.line_of_sight_grid(start, goal))
            self.assertEqual(planner.transition(start.nodes[0], goal.nodes[0]), float('inf'))

    def test_query_in_blocked_cell_or_outside_domain_is_rejected(self):
        obstacle = StaticPolygon(self.rectangle([.68, .45], [.8, .55]))
        env = Env('maze_endpoint', [self.rectangle([0, 0], [1, 1])], .1, OStatic=[obstacle])
        for point in (np.array([.5, .5]), np.array([-.01, .3]), np.array([1.01, .3])):
            planner = ZetaStarSIPPPlanner(env, 0, 1, .2, 5.)
            solution = planner.solve(point, point, 0., 10.)
            self.assertFalse(solution.is_success)
            self.assertEqual(planner.last_status, 'infeasible_query')

    def test_square_corner_overlap_is_rejected(self):
        obstacle = DynamicSphere(np.array([1.15, 1.15]), np.array([1.15, 1.15]), .1, Interval(0, 10))
        for name in ('maze_corner', 'grid_corner', 'iris_corner'):
            env = Env(name, [self.rectangle([0, 0], [2, 2])], .1, ODynamic=[obstacle])
            solution = ZetaStarSIPPPlanner(env, 0, 1, cell_size=.2).solve(np.ones(2), np.ones(2), 0, 10)
            self.assertFalse(solution.is_success)

    def test_maze_cell_edge_collision_cannot_be_reported_as_success(self):
        obstacle = DynamicSphere(np.array([.4, .49]), np.array([.4, .49]), .1, Interval(0, 10))
        env = Env('maze_cell_gap', [self.rectangle([0, 0], [1, 1])], .1, ODynamic=[obstacle])
        start, goal = np.array([.1, .3]), np.array([.7, .3])
        query = MPQuery(start, goal, 0., True, 1.)
        invalid = ShortestPathSolution(True, .6, 0., [], [np.hstack([start, 0., goal, .6])])
        with self.assertRaisesRegex(ValueError, 'continuous dynamic collision'):
            BaselineWorker.validate_zeta_solution(env, query, 10, invalid)
        for factor in (1, 2):
            solution = ZetaStarSIPPPlanner(env, 0, 1, cell_size=factor*.1).plan(query, 10)
            if solution.is_success:
                BaselineWorker.validate_zeta_solution(env, query, 10, solution)
                self.assertGreater(solution.cost, .6)

    def test_piecewise_square_intervals_match_independent_checker(self):
        pieces = ConcatDynamicSphere(
            [np.array([.2, .3]), np.array([.8, .3])],
            [np.array([.8, .3]), np.array([.8, .9])], [Interval(1, 3), Interval(3, 5)], .1)
        env = Env('intervals', [self.rectangle([0, 0], [2, 2])], .12, ODynamic=[pieces])
        intervals = _DynamicIntervals(env, 8.)
        checker = OfficialOMPLSTRRTStar._dynamic_obstacle_collision_checker(env, 0, 8)
        for point in (np.array([.2, .3]), np.array([.8, .3]), np.array([.8, .9]), np.array([1., .5])):
            blocked = intervals.blocked(point, env.robot_radius)
            for time in np.linspace(0, 8, 81):
                self.assertEqual(any(lo <= time <= hi for lo, hi in blocked),
                                 checker(point, point, time, time), (point, time, blocked))
        # The intermediate waypoint is not held before/after its two motion legs.
        middle = intervals.blocked(np.array([.8, .3]), env.robot_radius)
        self.assertFalse(any(lo <= 0 <= hi or lo <= 8 <= hi for lo, hi in middle))

    def test_pbs_reservations_use_standard_obstacles_and_preserve_goal_holds(self):
        env = Env('pbs_reservation', [self.rectangle([0, 0], [2, 2])], .12, ODynamic=[])
        path = STTrajectory(['a', 'b'], [np.array([.2, .3, 1., .8, .3, 3.]),
                                       np.array([.8, .3, 3., .8, .9, 5.])], 2)
        planner = ZetaSIPPPriorityBasedSearch(STGCS(env.C_Space, tmax=8.), env)
        for stay in (False, True):
            with self.subTest(stay=stay):
                query = MPQuery(path.x0[:-1], path.xT[:-1], 1., stay, 1.)
                reserved = planner._reserved_env_for_node(SimpleNamespace(sols=[path]), [0], [query])
                self.assertEqual(env.O_Dynamic, [])
                self.assertIsInstance(reserved.O_Dynamic[0], ConcatDynamicSphere)
                self.assertEqual(reserved.O_Dynamic[0].reserve_goal, stay)
                dynamic = _DynamicIntervals(reserved, 8.)
                checker = OfficialOMPLSTRRTStar._dynamic_obstacle_collision_checker(reserved, 0, 8)
                for point in (np.array([.38, .48]), np.array([.8, .3]), np.array([.8, .9])):
                    blocked = dynamic.blocked(point, env.robot_radius)
                    for time in np.linspace(0, 8, 81):
                        self.assertEqual(any(lo <= time <= hi for lo, hi in blocked),
                                         checker(point, point, time, time), (point, time, blocked))
                self.assertTrue(any(lo <= 0 <= hi for lo, hi in dynamic.blocked(path.x0[:-1], .12)))
                self.assertEqual(any(lo <= 8 <= hi for lo, hi in dynamic.blocked(path.xT[:-1], .12)), stay)
                self.assertFalse(any(lo <= 0 <= hi or lo <= 8 <= hi
                                     for lo, hi in dynamic.blocked(np.array([.8, .3]), .12)))
        disappearing = ConcatDynamicSphere.from_solution(path, .12, reserve_start=False, reserve_goal=False)
        env.O_Dynamic.append(disappearing)
        dynamic = _DynamicIntervals(env, 8.)
        self.assertFalse(any(lo <= 0 <= hi for lo, hi in dynamic.blocked(path.x0[:-1], .12)))

    def test_static_polygon_sweep_matches_existing_solver(self):
        obstacle = StaticPolygon(np.array([[.9, .4], [1.6, .7], [1.1, 1.5], [.5, 1.2]]))
        env = Env('iris_polygon', [self.rectangle([0, 0], [2, 2])], .1, OStatic=[obstacle])
        geometry = _ZetaGeometry(env)
        rng = np.random.default_rng(4)
        for _ in range(20):
            a, b = rng.uniform(.05, 1.95, (2, 2))
            self.assertEqual(geometry.segment_is_free(a, b),
                             not obstacle.is_colliding_lineseg(a, b, env.robot_radius))

    def test_scaled_maze_corner_contact_is_not_a_roundoff_gap(self):
        obstacle = StaticPolygon(self.rectangle([2.1875, .5208333333333334],
                                                [3.6458333333333335, .7291666666666667]))
        env = Env('maze_scaled', [self.rectangle([0, 0], [5, 5])], 5/48, OStatic=[obstacle])
        a, b = np.array([2.1875, 1.1458333333333335]), np.array([1.9791666666666667, .5208333333333334])
        self.assertTrue(obstacle.is_colliding_lineseg(a, b, env.robot_radius))
        self.assertFalse(_ZetaGeometry(env).segment_is_free(a, b))

    def test_deadline_includes_grid_construction(self):
        env = Env('budget', [self.rectangle([0, 0], [1, 1])], .1)
        clock = [0.]
        original = _ProjectedGrid._build_cells

        def build(projection):
            original(projection)
            clock[0] = .75

        def search(planner):
            self.assertEqual(planner.projection.deadline, 1.)
            self.assertEqual(planner.runtime_limit_secs, .25)
            clock[0] = 1.01
            planner.projection.check_deadline()

        with patch('baselines.zeta_sipp.planner.time.perf_counter', side_effect=lambda: clock[0]), \
                patch.object(_ProjectedGrid, '_build_cells', build), \
                patch.object(_ZetaStarSIPP, 'search', search):
            solution = ZetaStarSIPPPlanner(env, 0, 1, cell_size=.2, runtime_limit_secs=1).solve(
                np.array([.1, .1]), np.array([.9, .9]), 0, 10)
        self.assertFalse(solution.is_success)
        with patch('baselines.zeta_sipp.planner._ZetaGeometry', side_effect=AssertionError('deadline missed')):
            solution = ZetaStarSIPPPlanner(env, 0, 1, runtime_limit_secs=0).solve(
                np.array([.1, .1]), np.array([.9, .9]), 0, 10)
        self.assertFalse(solution.is_success)

    def test_horizon_collision_does_not_create_safe_endpoint(self):
        obstacle = DynamicSphere(np.array([.5, .5]), np.array([.5, .5]), .1, Interval(0, 10))
        env = Env('horizon', [self.rectangle([0, 0], [1, 1])], .1, ODynamic=[obstacle])
        projection = _ProjectedGrid(env, 10, .2)
        self.assertEqual(projection.safe_intervals_for_point(np.array([.5, .5])), [])

    def test_wait_shift_rechecks_later_cells_after_prior_cell_conflict(self):
        env = Env('wait_shift', [self.rectangle([0, 0], [2, 1])], .1)
        projection = _ProjectedGrid(env, 20, 1.)
        projection.get_cell(0, 0).risk_interval = [5., 10.]
        projection.get_cell(1, 0).risk_interval = [0., 6., 10.75, 12.]
        query = MPQuery(np.array([.5, .5]), np.array([1.5, .5]), 0., False, 1.)
        planner = _ZetaStarSIPP(projection, query, 20., 5., .001)
        parent = SimpleNamespace(g=0., safe_interval=(0., 20.),
                                 world_point=query.start, grid_point=query.start)
        node = SimpleNamespace(safe_interval=(0., 20.),
                               world_point=query.goal, grid_point=query.goal)
        cost = planner.get_cost_sipp(parent, node)
        self.assertAlmostEqual(cost, 12.501)
        departure = cost - 1.
        for index, occupied in enumerate([(departure, departure+.5), (departure+.5, cost)]):
            risks = projection.get_cell(index, 0).risk_interval
            for offset in range(0, len(risks), 2):
                self.assertFalse(_ZetaUtils.intersect_intervals(risks[offset:offset+2], occupied))
        parent.safe_interval = (0., 11.)
        self.assertEqual(planner.get_cost_sipp(parent, node), float('inf'))

    def test_both_resolutions_avoid_maze_wall_corners(self):
        env = make_2d_maze_env(np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]), robot_radius=0.1)
        start, goal = np.array([0.5, 0.5]), np.array([2.5, 0.5])
        self.assertTrue(any(o.is_colliding_lineseg(start, goal, env.robot_radius) for o in env.O_Static))
        for factor in (1, 2):
            with self.subTest(factor=factor):
                self.assert_valid_solution(env, start, goal, factor)

    def test_aligned_maze_cells_preserve_free_starts_at_both_resolutions(self):
        env = make_2d_maze_env(np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]), robot_radius=.25)
        start, goal = np.array([.5, .5]), np.array([2.5, .5])
        self.assertTrue(_ZetaGeometry(env).point_is_free(start))
        for factor in (1, 2):
            projection = self.assert_valid_solution(env, start, goal, factor)
            self.assertTrue(projection.wall_aligned)
            self.assertTrue(projection.cell_for_point(start).weight)

    def test_aligned_maze_cells_do_not_allow_wall_edge_or_corner_contacts(self):
        obstacle = StaticPolygon(self.rectangle([1.1, 1.1], [1.9, 1.9]))
        env = Env('maze_aligned', [self.rectangle([0, 0], [3, 3])], .1, OStatic=[obstacle])
        for factor in (1, 2):
            projection = _ProjectedGrid(env, 10., factor * .1)
            self.assertTrue(projection.wall_aligned)
            self.assertTrue(projection.cell_for_point(np.array([.95, 1.3])).weight)
            for a, b in [(np.array([.5, 1.5]), np.array([1.5, .5])),
                         (np.array([1., .3]), np.array([1., 2.5]))]:
                nodes = [SimpleNamespace(grid_point=projection.world_to_grid(p)) for p in (a, b)]
                self.assertFalse(projection.geometry.segment_is_free(a, b))
                self.assertFalse(projection.line_of_sight_grid(*nodes))
                self.assertFalse(projection.line_of_sight_grid(*nodes[::-1]))
            for point in (np.array([1., 1.]), np.array([1., 1.5]), np.array([2., 1.5])):
                self.assertEqual(projection.safe_intervals_for_point(point), [])
            # Domain boundary contact alone is allowed for a robot center.
            self.assertTrue(projection.safe_intervals_for_point(np.array([0., .5])))

    def test_alignment_requires_maze_and_aligned_clearance_geometry(self):
        obstacle = StaticPolygon(self.rectangle([1.1, 1.1], [1.9, 1.9]))
        for name in ('grid_aligned', 'iris_aligned'):
            env = Env(name, [self.rectangle([0, 0], [3, 3])], .1, OStatic=[obstacle])
            projection = _ProjectedGrid(env, 10., .2)
            self.assertFalse(projection.wall_aligned)
            self.assertFalse(projection.cell_for_point(np.array([.95, 1.3])).weight)
        env = Env('maze_unaligned', [self.rectangle([0, 0], [3, 3])], .1,
                  OStatic=[StaticPolygon(obstacle.vertices + .03)])
        self.assertFalse(_ProjectedGrid(env, 10., .2).wall_aligned)

    def test_scaled_aligned_maze_has_no_spurious_outer_blocked_row(self):
        env = make_2d_maze_env(np.zeros((12, 12), dtype=int), robot_radius=5/48, spatial_scale=5/12)
        for factor in (1, 2):
            projection = _ProjectedGrid(env, 10., factor * env.robot_radius)
            self.assertTrue(projection.wall_aligned)
            self.assertEqual(projection.shape, (48 // factor, 48 // factor))
            self.assertTrue(all(cell.weight for cell in projection.cells))

    def test_both_resolutions_avoid_crossing_dynamic_obstacle(self):
        box = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.]])
        obstacle = DynamicSphere(np.array([2., 0.5]), np.array([2., 2.5]), 0.2, Interval(0, 3))
        env = Env('maze_crossing', [box], robot_radius=0.25, ODynamic=[obstacle])
        start, goal = np.array([0.5, 1.5]), np.array([3.5, 1.5])
        self.assertTrue(obstacle.is_colliding_lineseg(start, goal, 0, 3, env.robot_radius))
        for factor in (1, 2):
            with self.subTest(factor=factor), \
                    patch.object(_ZetaGeometry, 'point_is_free', side_effect=AssertionError('point geometry')), \
                    patch.object(_ZetaGeometry, 'segment_is_free', side_effect=AssertionError('segment geometry')):
                self.assert_valid_solution(env, start, goal, factor)


if __name__ == '__main__':
    unittest.main()
