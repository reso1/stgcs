from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import os, time

import numpy as np
from matplotlib.axes import Axes

from experiments.environment.env import Env
from experiments.environment.grid import GridEnvironmentBuilder
from experiments.environment.iris import Iris2DEnvBuilder
from experiments.environment.maze.maze_env import make_random_2d_maze
from experiments.environment.obstacle import ConcatDynamicSphere, DynamicObstacle, DynamicSphere, StaticPolygon

from experiments.environment.space_time_graph import STGCS
from stgcs_native.types import MPQuery
from experiments.environment.interval import Interval

from pydrake.all import RandomGenerator
from scipy.spatial import KDTree


class Instance:

    def __init__(
        self, name:str, seed:int, env:Env, stgcs:STGCS,
    ) -> None:
        self.name = name
        self.seed = seed
        self.env = env
        self.stgcs = stgcs
        self.sampled_pts = []
        self.np_rng = np.random.RandomState(self.seed+100)
        self.drake_rng = RandomGenerator(self.seed+100)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["np_rng"] = None
        state["drake_rng"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.np_rng = np.random.RandomState(self.seed + 100)
        self.drake_rng = RandomGenerator(self.seed + 100)

    def draw(self, ax:Axes) -> None:
        self.env.draw_static(ax, alpha=0.8, draw_CSpace=True, CSpace_text=False)
        for obs in self.env.O_Dynamic:
            self.draw_dynamic_obstacle_spatial_traj(ax, obs, lw=4, alpha=0.5)

        ax.axis('equal')

    @staticmethod
    def draw_dynamic_obstacle_spatial_traj(
        ax: Axes, obstacle: DynamicObstacle, color: str = 'k', lw: float = 2, alpha: float = 1.0
    ) -> None:
        if isinstance(obstacle, DynamicSphere):
            ax.plot(
                [obstacle.x0[0], obstacle.xt[0]],
                [obstacle.x0[1], obstacle.xt[1]],
                '-o',
                color=color,
                lw=lw,
                alpha=alpha,
            )
            return
        if isinstance(obstacle, ConcatDynamicSphere):
            xs, ys = [], []
            for seg in obstacle.segments:
                xs.extend([seg.x0[0], seg.xt[0]])
                ys.extend([seg.x0[1], seg.xt[1]])
            ax.plot(xs, ys, '-o', color=color, lw=lw, alpha=alpha)
            return
        raise TypeError(f"Unsupported dynamic obstacle type {type(obstacle)!r}")

    @staticmethod
    def should_report_progress() -> bool:
        return os.environ.get("STGCS_DISABLE_PROGRESS", "") != "1"

    @staticmethod
    def generate(**kwargs) -> Instance:
        raise NotImplementedError("generate not implemented in base Instance class.")

    @staticmethod
    def build_stgcs_from_env(env:Env, tmax:float, vlimit:float) -> STGCS:
        stgcs = env.build_STGCS(t0=0.0, tmax=tmax, vlimit=vlimit)
        stgcs.make_leaves_roots()
        return stgcs

    @staticmethod
    def dynamic_obstacle_to_spec(obstacle:DynamicObstacle) -> Dict[str, Any]:
        if isinstance(obstacle, DynamicSphere):
            return {
                "type": "sphere",
                "radius": float(obstacle.radius),
                "segments": [{
                    "start": obstacle.x0.tolist(),
                    "goal": obstacle.xt.tolist(),
                    "t_start": float(obstacle.itvl.start),
                    "t_end": float(obstacle.itvl.end),
                }],
            }
        if isinstance(obstacle, ConcatDynamicSphere):
            return {
                "type": "concat_sphere",
                "radius": float(obstacle.radius),
                "segments": [{
                    "start": seg.x0.tolist(),
                    "goal": seg.xt.tolist(),
                    "t_start": float(seg.itvl.start),
                    "t_end": float(seg.itvl.end),
                } for seg in obstacle.segments],
            }
        raise TypeError(f"Unsupported dynamic obstacle type {type(obstacle)!r}")

    @staticmethod
    def dynamic_obstacle_from_spec(spec:Dict[str, Any]) -> DynamicObstacle:
        segments = spec["segments"]
        radius = float(spec["radius"])
        if spec["type"] == "sphere":
            seg = segments[0]
            return DynamicSphere(
                np.asarray(seg["start"], dtype=float),
                np.asarray(seg["goal"], dtype=float),
                radius,
                Interval(float(seg["t_start"]), float(seg["t_end"])),
            )
        if spec["type"] == "concat_sphere":
            return ConcatDynamicSphere(
                [np.asarray(seg["start"], dtype=float) for seg in segments],
                [np.asarray(seg["goal"], dtype=float) for seg in segments],
                [Interval(float(seg["t_start"]), float(seg["t_end"])) for seg in segments],
                radius,
            )
        raise ValueError(f"Unsupported dynamic obstacle spec type {spec['type']!r}")

    @staticmethod
    def dynamic_obstacles_from_specs(specs:List[Dict[str, Any]]) -> List[DynamicObstacle]:
        return [Instance.dynamic_obstacle_from_spec(spec) for spec in specs]

    @staticmethod
    def obstacle_sampled_points(obstacles:List[DynamicObstacle]) -> List[np.ndarray]:
        sampled_pts: List[np.ndarray] = []
        for obstacle in obstacles:
            if isinstance(obstacle, DynamicSphere):
                sampled_pts.extend([obstacle.x0.copy(), obstacle.xt.copy()])
            elif isinstance(obstacle, ConcatDynamicSphere):
                sampled_pts.extend([obstacle.x0.copy(), obstacle.xt.copy()])
            else:
                raise TypeError(f"Unsupported dynamic obstacle type {type(obstacle)!r}")
        return sampled_pts

    @staticmethod
    def add_dynamic_obstacles_to_env(env:Env, num_obs:int, seed:int, vlimit:float) -> List[np.ndarray]:
        np_rng, drake_rng = np.random.RandomState(seed), RandomGenerator(seed)
        sampled_pts = []
        from experiments.native import STGCSPlannerClient

        static_env = env.copy()
        for i in range(num_obs):
            print(f"Adding dynamic obstacle {i+1}/{num_obs}")
            start, goal = env.sample_CSpace(np_rng, drake_rng, size=2)
            sampled_pts.extend([start, goal])
            mp_query = MPQuery(start, goal, 0.0, False, vlimit)
            sol, result = STGCSPlannerClient.solve_single(static_env, mp_query, 1000.0, budget=float("inf"))
            if sol is None:
                raise RuntimeError(f"Could not generate dynamic obstacle {i}: {result}")
            obs = ConcatDynamicSphere.from_solution(sol, radius=env.robot_radius)
            env.O_Dynamic.append(obs)

        return sampled_pts

    def compute_heuristics(self) -> None:
        from experiments.offline_heuristics import BaseOfflineHeuristicStore

        BaseOfflineHeuristicStore.compute(self, cost_table=False)


    def sample_MP_query(self, t0:float=0.0, is_stay:bool=True, timeout_secs:float=6e2) -> Optional[MPQuery]:
        if len(self.sampled_pts) == 0:
            start, goal = self.env.sample_CSpace(self.np_rng, self.drake_rng, size=2)
            self.sampled_pts.extend([start, goal])
            return MPQuery(start, goal, t0, is_stay, self.stgcs.vlimit)

        # sample start and goal that are not too close to existing dynamic obstacles
        kdtree = KDTree(self.sampled_pts)
        ts = time.perf_counter()
        while True:
            start, goal = self.env.sample_CSpace(self.np_rng, self.drake_rng, size=2)
            cc_start = kdtree.query(start)[0]
            cc_goal = kdtree.query(goal)[0]
            if cc_start > 3*self.env.robot_radius and cc_goal > 3*self.env.robot_radius:
                break
            if time.perf_counter() - ts >= timeout_secs:
                print("Timeout in sampling MP query. Using last sampled points.")
                return

        self.sampled_pts.extend([start, goal])
        return MPQuery(start, goal, t0, is_stay, self.stgcs.vlimit)


class GridInstance(Instance):

    def __init__(
        self, seed:int, gridN:int, gridM:int, num_obs:int, env:Env, stgcs:STGCS,
        sampled_pts:List[np.ndarray]=[],
        space_dim:int=2,
    ) -> None:
        if space_dim < 2:
            raise ValueError(f"GridInstance requires space_dim >= 2, got {space_dim}")
        self.grid_shape = (gridN,) + (gridM,) * (space_dim - 1)
        shape_str = "x".join(str(v) for v in self.grid_shape)
        name = f"Grid-{shape_str}-O{num_obs}-S{seed}"
        super().__init__(name, seed, env, stgcs)
        self.gridN = gridN
        self.gridM = gridM
        self.space_dim = space_dim
        self.num_obs = num_obs
        self.sampled_pts = sampled_pts

    @staticmethod
    def build_base_env(seed:int, N:int, M:int, space_dim:int=2, spatial_scale:float=1.0) -> Env:
        if space_dim == 2:
            return GridEnvironmentBuilder.random_2d(seed, N, M, spatial_scale=spatial_scale)
        if space_dim == 3:
            return GridEnvironmentBuilder.random_3d(seed, N, M, spatial_scale=spatial_scale)
        raise ValueError(f"Unsupported space_dim {space_dim}; expected 2 or 3")

    @staticmethod
    def from_env(
        seed:int, N:int, M:int, env:Env, num_obs:int,
        tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        sampled_pts:Optional[List[np.ndarray]]=None,
        space_dim:int=2,
    ) -> GridInstance:
        stgcs = Instance.build_stgcs_from_env(env, tmax=tmax, vlimit=vlimit)
        if Instance.should_report_progress():
            print(f"Grid Instance {env.name} generated. "
                  f"# of sets={stgcs.G.number_of_nodes()}, "
                  f"# of edges={stgcs.G.number_of_edges()}")

        istc = GridInstance(
            seed, N, M, num_obs, env, stgcs,
            sampled_pts=[] if sampled_pts is None else sampled_pts,
            space_dim=space_dim,
        )
        if compute_heuristics:
            istc.compute_heuristics()
        return istc

    @staticmethod
    def generate_base(
        seed:int, N:int, M:int, tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        space_dim:int=2, spatial_scale:float=1.0,
    ) -> GridInstance:
        env = GridInstance.build_base_env(seed, N, M, space_dim=space_dim, spatial_scale=spatial_scale)
        return GridInstance.from_env(
            seed=seed,
            N=N,
            M=M,
            env=env,
            num_obs=0,
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            sampled_pts=[],
            space_dim=space_dim,
        )

    @staticmethod
    def generate_from_obstacle_specs(
        seed:int, N:int, M:int, obstacle_specs:List[Dict[str, Any]],
        tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        space_dim:int=2,
    ) -> GridInstance:
        env = GridInstance.build_base_env(seed, N, M, space_dim=space_dim)
        env.O_Dynamic = Instance.dynamic_obstacles_from_specs(obstacle_specs)
        return GridInstance.from_env(
            seed=seed,
            N=N,
            M=M,
            env=env,
            num_obs=len(obstacle_specs),
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            sampled_pts=Instance.obstacle_sampled_points(env.O_Dynamic),
            space_dim=space_dim,
        )

    @staticmethod
    def generate(
        seed:int, N:int, M:int, num_obs:int, tmax:float=1000.0,
        vlimit:float=1.0,
        compute_heuristics:bool=True,
        space_dim:int=2,
    ) -> GridInstance:
        env = GridInstance.build_base_env(seed, N, M, space_dim=space_dim)
        sampled_pts = Instance.add_dynamic_obstacles_to_env(env, num_obs, seed, vlimit)
        return GridInstance.from_env(
            seed=seed,
            N=N,
            M=M,
            env=env,
            num_obs=num_obs,
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            sampled_pts=sampled_pts,
            space_dim=space_dim,
        )


class Simple2DInstance(Instance):
    DEFAULT_ROBOT_RADIUS = 0.1
    CENTER_SET_INDEX = 4

    def __init__(
        self, seed:int, env:Env, stgcs:STGCS,
        sampled_pts:Optional[List[np.ndarray]]=None,
    ) -> None:
        name = f"SIMPLE2D-S{seed}"
        super().__init__(name, seed, env, stgcs)
        self.sampled_pts = [] if sampled_pts is None else sampled_pts

    @staticmethod
    def cspace_vertices() -> List[np.ndarray]:
        return [
            np.array([[0.0, 0.0], [1.85, 0.0], [1.85, 1.85], [0.0, 1.85]]),
            np.array([[0.0, 2.15], [1.85, 2.15], [1.85, 4.0], [0.0, 4.0]]),
            np.array([[2.15, 0.0], [4.0, 0.0], [4.0, 1.85], [2.15, 1.85]]),
            np.array([[2.15, 2.15], [4.0, 2.15], [4.0, 4.0], [2.15, 4.0]]),
            np.array([[0.75, 1.85], [1.25, 1.85], [1.25, 2.15], [0.75, 2.15]]),
            np.array([[2.75, 1.85], [3.25, 1.85], [3.25, 2.15], [2.75, 2.15]]),
            np.array([[1.85, 0.75], [2.15, 0.75], [2.15, 1.25], [1.85, 1.25]]),
            np.array([[1.85, 2.75], [2.15, 2.75], [2.15, 3.25], [1.85, 3.25]]),
        ]

    @staticmethod
    def static_obstacles() -> List[StaticPolygon]:
        return [
            StaticPolygon(np.array([[1.95, -0.05], [2.05, -0.05], [2.05, 0.65], [1.95, 0.65]])),
            StaticPolygon(np.array([[1.95, 1.35], [2.05, 1.35], [2.05, 2.65], [1.95, 2.65]])),
            StaticPolygon(np.array([[1.95, 3.35], [2.05, 3.35], [2.05, 4.05], [1.95, 4.05]])),
            StaticPolygon(np.array([[-0.05, 1.95], [0.65, 1.95], [0.65, 2.05], [-0.05, 2.05]])),
            StaticPolygon(np.array([[1.35, 1.95], [2.65, 1.95], [2.65, 2.05], [1.35, 2.05]])),
            StaticPolygon(np.array([[3.35, 1.95], [4.05, 1.95], [4.05, 2.05], [3.35, 2.05]])),
        ]

    @classmethod
    def build_base_env(cls, seed:int, robot_radius:float | None=None) -> Env:
        del seed
        return Env(
            name="simple2d",
            CSpace=cls.cspace_vertices(),
            OStatic=cls.static_obstacles(),
            robot_radius=cls.DEFAULT_ROBOT_RADIUS if robot_radius is None else float(robot_radius),
        )

    @classmethod
    def center_set_hpoly(cls, env:Env):
        if len(env._CSpace_hpoly) <= cls.CENTER_SET_INDEX:
            raise ValueError(f"SIMPLE2D center set index {cls.CENTER_SET_INDEX} is not available.")
        return env._CSpace_hpoly[cls.CENTER_SET_INDEX]

    def point_in_center_set(self, point:np.ndarray) -> bool:
        return bool(self.center_set_hpoly(self.env).PointInSet(np.asarray(point, dtype=float)))

    def query_endpoint_uses_center_set(self, query:MPQuery) -> bool:
        return self.point_in_center_set(query.start) or self.point_in_center_set(query.goal)

    @staticmethod
    def from_env(
        seed:int, env:Env, tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        sampled_pts:Optional[List[np.ndarray]]=None,
    ) -> Simple2DInstance:
        stgcs = Instance.build_stgcs_from_env(env, tmax=tmax, vlimit=vlimit)
        if Instance.should_report_progress():
            print(f"SIMPLE2D Instance {env.name} generated. "
                  f"# of sets={stgcs.G.number_of_nodes()}, "
                  f"# of edges={stgcs.G.number_of_edges()}")

        instance = Simple2DInstance(seed, env, stgcs, sampled_pts=sampled_pts)
        if compute_heuristics:
            instance.compute_heuristics()
        return instance

    @staticmethod
    def generate_base(
        seed:int, tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        robot_radius:float | None=None,
    ) -> Simple2DInstance:
        env = Simple2DInstance.build_base_env(seed, robot_radius=robot_radius)
        return Simple2DInstance.from_env(
            seed=seed,
            env=env,
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            sampled_pts=[],
        )


class MazeInstance(Instance):

    def __init__(
        self, seed:int, width:int, height:int, num_obs:int, env:Env, stgcs:STGCS,
        sampled_pts:List[np.ndarray]=[],
    ) -> None:
        name = f"Maze-{width}x{height}-O{num_obs}-S{seed}"
        super().__init__(name, seed, env, stgcs)
        self.width = width
        self.height = height
        self.num_obs = num_obs
        self.sampled_pts = sampled_pts

    @staticmethod
    def build_base_env(seed:int, width:int, height:int, robot_radius:float=0.25,
                       spatial_scale:float=1.0) -> Env:
        return make_random_2d_maze(width, height, seed, robot_radius=robot_radius,
                                   spatial_scale=spatial_scale)

    @staticmethod
    def from_env(
        seed:int, width:int, height:int, env:Env, num_obs:int,
        tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        sampled_pts:Optional[List[np.ndarray]]=None,
    ) -> MazeInstance:
        stgcs = Instance.build_stgcs_from_env(env, tmax=tmax, vlimit=vlimit)
        if Instance.should_report_progress():
            print(f"Maze Instance {env.name} generated. "
                  f"# of sets={stgcs.G.number_of_nodes()}, "
                  f"# of edges={stgcs.G.number_of_edges()}")

        istc = MazeInstance(
            seed, width, height, num_obs, env, stgcs,
            sampled_pts=[] if sampled_pts is None else sampled_pts,
        )
        if compute_heuristics:
            istc.compute_heuristics()
        return istc

    @staticmethod
    def generate_base(
        seed:int, width:int, height:int, tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        robot_radius:float=0.25,
        spatial_scale:float=1.0,
    ) -> MazeInstance:
        env = MazeInstance.build_base_env(seed, width, height, robot_radius=robot_radius,
                                          spatial_scale=spatial_scale)
        return MazeInstance.from_env(
            seed=seed,
            width=width,
            height=height,
            env=env,
            num_obs=0,
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            sampled_pts=[],
        )

    @staticmethod
    def generate_from_obstacle_specs(
        seed:int, width:int, height:int, obstacle_specs:List[Dict[str, Any]],
        tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        robot_radius:float=0.25,
    ) -> MazeInstance:
        env = MazeInstance.build_base_env(seed, width, height, robot_radius=robot_radius)
        env.O_Dynamic = Instance.dynamic_obstacles_from_specs(obstacle_specs)
        return MazeInstance.from_env(
            seed=seed,
            width=width,
            height=height,
            env=env,
            num_obs=len(obstacle_specs),
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            sampled_pts=Instance.obstacle_sampled_points(env.O_Dynamic),
        )

    @staticmethod
    def generate(
        seed:int, width:int, height:int, num_obs:int,
        tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        robot_radius:float=0.25,
    ) -> MazeInstance:
        env = MazeInstance.build_base_env(seed, width, height, robot_radius=robot_radius)
        sampled_pts = Instance.add_dynamic_obstacles_to_env(env, num_obs, seed, vlimit)
        return MazeInstance.from_env(
            seed=seed,
            width=width,
            height=height,
            env=env,
            num_obs=num_obs,
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            sampled_pts=sampled_pts,
        )


class Iris2DInstance(Instance):

    def __init__(
        self, seed:int, num_seed_points:int, num_static_obstacles:int, num_obs:int,
        env:Env, stgcs:STGCS,
        env_params:Optional[Dict[str, Any]]=None,
        sampled_pts:List[np.ndarray]=[],
    ) -> None:
        resolved_env_params = Iris2DEnvBuilder.normalize_env_params(env_params)
        name = f"Iris2D-m{num_static_obstacles}-O{num_obs}-S{seed}"
        super().__init__(name, seed, env, stgcs)
        self.num_seed_points = num_seed_points
        self.num_static_obstacles = num_static_obstacles
        self.num_obs = num_obs
        self.env_params = resolved_env_params
        self.sampled_pts = sampled_pts

    @staticmethod
    def build_base_env(seed:int, env_params:Optional[Dict[str, Any]]=None) -> Env:
        return Iris2DEnvBuilder.build_env(seed=seed, env_params=env_params)

    @staticmethod
    def from_env(
        seed:int, env:Env, num_obs:int,
        tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        env_params:Optional[Dict[str, Any]]=None,
        sampled_pts:Optional[List[np.ndarray]]=None,
    ) -> Iris2DInstance:
        resolved_env_params = Iris2DEnvBuilder.normalize_env_params(env_params)
        stgcs = Instance.build_stgcs_from_env(env, tmax=tmax, vlimit=vlimit)
        if Instance.should_report_progress():
            print(f"IRIS-2D Instance {env.name} generated. "
                  f"# of sets={stgcs.G.number_of_nodes()}, "
                  f"# of edges={stgcs.G.number_of_edges()}")

        seed_points = [
            point.copy()
            for point in getattr(env, "seed_points", np.empty((0, 2)))
        ]
        istc = Iris2DInstance(
            seed,
            int(len(seed_points)),
            int(resolved_env_params["m"]),
            num_obs,
            env,
            stgcs,
            env_params=resolved_env_params,
            sampled_pts=seed_points if sampled_pts is None else sampled_pts,
        )
        if compute_heuristics:
            istc.compute_heuristics()
        return istc

    @staticmethod
    def generate_base(
        seed:int, tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        env_params:Optional[Dict[str, Any]]=None,
    ) -> Iris2DInstance:
        resolved_env_params = Iris2DEnvBuilder.normalize_env_params(env_params)
        env = Iris2DInstance.build_base_env(seed=seed, env_params=resolved_env_params)
        return Iris2DInstance.from_env(
            seed=seed,
            env=env,
            num_obs=0,
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            env_params=resolved_env_params,
        )

    @staticmethod
    def generate_from_obstacle_specs(
        seed:int, obstacle_specs:List[Dict[str, Any]],
        tmax:float=1000.0, vlimit:float=1.0,
        compute_heuristics:bool=True,
        env_params:Optional[Dict[str, Any]]=None,
    ) -> Iris2DInstance:
        resolved_env_params = Iris2DEnvBuilder.normalize_env_params(env_params)
        env = Iris2DInstance.build_base_env(seed=seed, env_params=resolved_env_params)
        env.O_Dynamic = Instance.dynamic_obstacles_from_specs(obstacle_specs)
        seed_points = [
            point.copy()
            for point in getattr(env, "seed_points", np.empty((0, 2)))
        ]
        sampled_pts = seed_points + Instance.obstacle_sampled_points(env.O_Dynamic)
        return Iris2DInstance.from_env(
            seed=seed,
            env=env,
            num_obs=len(obstacle_specs),
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
            env_params=resolved_env_params,
            sampled_pts=sampled_pts,
        )
