from typing import List, Tuple, Dict, Set
import os, time
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial import ConvexHull
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pydrake.all import (
    VPolytope, HPolyhedron, MathematicalProgram, DecomposeLinearExpressions,
    MosekSolver, SolverOptions, CommonSolverOption
)


""" setup mosek solver """
solver = MosekSolver()
solver_options = SolverOptions()
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)
solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_LOG", 0)


def timeit(func):

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Func {func.__name__} took {total_time:.3f} secs')
        return result

    return timeit_wrapper


""" convex set operations """

def make_hpolytope(V) -> HPolyhedron:
    ch = ConvexHull(V)
    return HPolyhedron(ch.equations[:, :-1], -ch.equations[:, -1])


def time_extruded(hpoly:HPolyhedron, t0:float, tf:float) -> HPolyhedron|None:
    if t0 < tf:
        return hpoly.CartesianProduct(HPolyhedron.MakeBox([t0], [tf]))
    
    return None


def crop_time_extruded(
    hpoly:HPolyhedron, t_low:float=-np.inf, t_high:float=np.inf
) -> HPolyhedron|None:
    if t_low >= t_high:
        return None

    if t_low == -np.inf and t_high == np.inf:
        return hpoly
    
    cropping_halfspace = HPolyhedron(
        A = np.block([np.zeros((hpoly.ambient_dimension()-1, 2)), np.array([[-1], [1]])]),
        b = np.array([[-t_low], [t_high]])
    )
    
    return hpoly.Intersection(cropping_halfspace)


def get_hpoly_bounds(
    hpoly:HPolyhedron, dim:int|List[int]
) -> Tuple[np.ndarray|float, np.ndarray|float]:
    num_dims = hpoly.ambient_dimension()
    prog = MathematicalProgram()
    xi = prog.NewContinuousVariables(num_dims, "x")
    prog.AddLinearConstraint(
        A  = hpoly.A(), 
        lb = -np.inf*np.ones_like(hpoly.b()), 
        ub = hpoly.b(),
        vars = xi
    )
    
    if isinstance(dim, int):
        min_xd_cost = prog.AddCost(xi[dim])
        lb = solver.Solve(prog, solver_options=solver_options).GetSolution(xi[dim])
        prog.RemoveCost(min_xd_cost)
        max_xd_cost = prog.AddCost(-xi[dim])
        ub = solver.Solve(prog, solver_options=solver_options).GetSolution(xi[dim])
        return lb, ub

    if isinstance(dim, list):
        lb, ub = np.zeros(len(dim)), np.zeros(len(dim))
        for d in dim:
            # get lower bound
            min_xd_cost = prog.AddCost(xi[d])
            lb[d] = solver.Solve(prog, solver_options=solver_options).GetSolution(xi[d])
            prog.RemoveCost(min_xd_cost)
            # get upper bound
            max_xd_cost = prog.AddCost(-xi[d])
            ub[d] = solver.Solve(prog, solver_options=solver_options).GetSolution(xi[d])
            prog.RemoveCost(max_xd_cost)

    return lb, ub


def find_space_time_intersecting_pts(
    hpoly:HPolyhedron, xp:np.ndarray, xq:np.ndarray, dim:int
) -> Tuple[np.ndarray, np.ndarray]|None:
    tp, tq = xp[-1], xq[-1]

    if tp == tq:
        return xp[:dim], xq[:dim]

    prog = MathematicalProgram()
    xi = prog.NewContinuousVariables(dim, "x")
    
    # time must be in the interval
    prog.AddLinearConstraint(
        A = np.array([[0]* (dim - 1) + [1]]),
        lb = np.array([tp]),
        ub = np.array([tq]),
        vars = xi
    )
    # x must be in the hpoly
    prog.AddLinearConstraint(
        A  = hpoly.A(), 
        lb = -np.inf*np.ones_like(hpoly.b()), 
        ub = hpoly.b(),
        vars = xi
    )
    # x must be on the line segment
    for d in range(dim - 1):
        Aeq = np.zeros((1, dim))
        Aeq[0, d] = 1
        Aeq[0, -1] = (xp[d] - xq[d]) / (tq - tp)
        beq = np.array([tq * xp[d] - tp * xq[d]]) / (tq - tp)
        prog.AddLinearEqualityConstraint(Aeq, beq, xi)

    # get intersections
    min_time_cost = prog.AddCost(xi[-1])
    res = solver.Solve(prog, solver_options=solver_options)
    if not res.is_success():
        return
    lb = res.GetSolution(xi)
    prog.RemoveCost(min_time_cost)
    max_time_cost = prog.AddCost(-xi[-1])
    res = solver.Solve(prog, solver_options=solver_options)
    ub = res.GetSolution(xi)
    if not res.is_success():
        return

    return lb, ub


def is_lineseg_colliding(hpoly:HPolyhedron, xp:np.ndarray, xq:np.ndarray) -> bool:
    prog = MathematicalProgram()
    dim = hpoly.ambient_dimension()
    t = prog.NewContinuousVariables(1, "t")
    # x(t) = (1-t)\cdot xp + t\cdot xq 
    # t \in [0, 1]
    prog.AddLinearConstraint(
        A = np.ones((1, 1)),
        lb = np.array([0]),
        ub = np.array([1]),
        vars = t
    )
   
    # x(t) must be in the hpoly
    prog.AddLinearConstraint(
        A = hpoly.A() @ (xq - xp),
        lb = -np.inf * np.ones_like(hpoly.b()),
        ub = hpoly.b() - hpoly.A() @ xp,
        vars = t
    )
    
    res = solver.Solve(prog, solver_options=solver_options)
    return res.is_success() 


def is_hpoly_pos_fixed(hpoly:HPolyhedron, dim:int) -> bool:
    lb, ub = get_hpoly_bounds(hpoly, [d for d in range(dim-1)])
    return np.allclose(lb, ub)


def collinear(points:np.ndarray, tol=1e-10):
    if len(points) <= 2:
        return True
        
    for i in range(len(points)-2):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        x3, y3 = points[i+2]
        
        area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2
        if area > tol:
            return False

    return True


def squash_multi_points(hpoly:HPolyhedron, dim:int) -> HPolyhedron:
    A, b = hpoly.A(), hpoly.b()
    num_pts = int(A.shape[-1] // dim)
    num_constraints = int(A.shape[0] // num_pts)
    return HPolyhedron(A[:num_constraints, :dim], b[:num_constraints])


""" visualization """

def draw_2d_space_set(obj:np.ndarray|HPolyhedron, ax:Axes, color='k', marker='o', linestyle='-', label=False) -> None:
    if isinstance(obj, HPolyhedron):
        obj = VPolytope(obj).vertices().T

    assert obj.shape[-1] == 2, "2D sets are supported"
    
    for idx, v in enumerate(obj):
        ax.plot(v[0], v[1], f'{color}{marker}')
        if label:
            ax.text(v[0], v[1], str(idx))

    if collinear(obj):
        min_x, min_y = np.min(obj, axis=0)
        max_x, max_y = np.max(obj, axis=0)
        plt.plot([min_x, max_x], [min_y, max_y], 'r-', label='Line')
        ax.plot(obj[:, 0], obj[:, 1], f'{linestyle}{color}')
    else:
        hull = ConvexHull(obj)
        for simplex in hull.simplices:
            ax.plot(obj[simplex, 0], obj[simplex, 1], f'{linestyle}{color}')
            

def draw_3d_space_time_set(obj:np.ndarray|HPolyhedron, ax:Axes3D, 
    alpha:float=0.5, fc='lightgray', ec='k', time_scaler:float=1.0) -> None:
    if isinstance(obj, HPolyhedron):
        obj = VPolytope(obj).vertices().T

    assert obj.shape[-1] == 3, "3D sets are supported (2d space + time)"
    if len(obj) < 3:
        return
    
    obj[:, 2] *= time_scaler
    try:
        hull = ConvexHull(obj)
        hull_faces = []
        for simplex in hull.simplices:
            vertices = obj[simplex]
            hull_faces.append(vertices)

        hull_surface = Poly3DCollection(
            hull_faces, alpha=alpha, facecolor=fc, edgecolor=ec)
        hull_surface.set_edgecolor('none')
        
        ax.add_collection3d(hull_surface)

        for simplex in hull.simplices:
            ax.plot(obj[simplex, 0], obj[simplex, 1], obj[simplex, 2], 'k--', alpha=alpha)
    except:
        vertices = [order_points(obj)]
        plane = Poly3DCollection(vertices, alpha=0.5)
        plane.set_facecolor(fc)
        plane.set_edgecolor(ec)
        plane.set_alpha(alpha)
        ax.add_collection3d(plane)


def draw_cuboid(ax:Axes3D, xp:np.ndarray, xq:np.ndarray, halfsize:float=0.05, color='k', alpha=0.5) -> None:
    p_facet_verts = [[xp[0] - halfsize, xp[1] - halfsize, xp[2]], 
                     [xp[0] + halfsize, xp[1] - halfsize, xp[2]], 
                     [xp[0] + halfsize, xp[1] + halfsize, xp[2]], 
                     [xp[0] - halfsize, xp[1] + halfsize, xp[2]]]
    q_facet_verts = [[xq[0] - halfsize, xq[1] - halfsize, xq[2]],
                     [xq[0] + halfsize, xq[1] - halfsize, xq[2]],
                     [xq[0] + halfsize, xq[1] + halfsize, xq[2]],
                     [xq[0] - halfsize, xq[1] + halfsize, xq[2]]]
    
    # draw cuboid
    cuboid = np.array(p_facet_verts + q_facet_verts)
    draw_3d_space_time_set(cuboid, ax, alpha=alpha, fc=color)

    # draw xp --- xq
    vec = xq - xp
    offset = 0.0 # 0.1
    st = xp - offset * vec
    et = xq + offset * vec
    ax.plot([st[0], et[0]], [st[1], et[1]], [st[2], et[2]], 'x-k', lw=3, ms=10)


def draw_cylinder(ax:Axes3D, P:np.ndarray, Q:np.ndarray, r:float, alpha:float=0.5, num_points=20) -> None:
    P = np.array(P)
    Q = np.array(Q)
    
    v = Q - P
    length = np.linalg.norm(v)
    v = v / length
    
    if v[0] == 0 and v[1] == 0:
        n1 = np.array([1, 0, 0])
    else:
        n1 = np.array([-v[1], v[0], 0])
        n1 = n1 / np.linalg.norm(n1)
    
    n2 = np.cross(v, n1)
    
    theta = np.linspace(0, 2*np.pi, num_points)
    h = np.linspace(0, length, num_points)
    
    theta_grid, h_grid = np.meshgrid(theta, h)
    
    X = P[0] + h_grid*v[0] + r*np.cos(theta_grid)*n1[0] + r*np.sin(theta_grid)*n2[0]
    Y = P[1] + h_grid*v[1] + r*np.cos(theta_grid)*n1[1] + r*np.sin(theta_grid)*n2[1]
    Z = P[2] + h_grid*v[2] + r*np.cos(theta_grid)*n1[2] + r*np.sin(theta_grid)*n2[2]
        
    surf = ax.plot_surface(X, Y, Z, color='k', alpha=alpha)


def order_points(points:np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    centered_points = points - center
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[2]
    projection_matrix = np.eye(3) - normal[:, np.newaxis] * normal
    basis_1, basis_2 = vh[0], vh[1]
    points_2d = np.column_stack([
        np.dot(centered_points, basis_1),
        np.dot(centered_points, basis_2)
    ])
    
    hull = ConvexHull(points_2d)
    return points[hull.vertices]


def anim_rotating_camera(stgcs, sol, name="video") -> None:
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.axis("off")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_zticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_zlim(0, 2)

    ax.view_init(elev=20, azim=0)
    stgcs.draw(ax, set_labels=False)
    for wp in sol.trajectory:
        ax.plot([wp[0], wp[3]], [wp[1], wp[4]], [wp[2], wp[5]], '-ok')

    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return ax,

    anim = animation.FuncAnimation(
        fig, update, np.linspace(0, 360, 120), interval=50,
        blit=True, repeat=True, cache_frame_data=False)

    # FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
    # fp = os.path.join(os.getcwd(), f'{name}.mp4')
    # anim.save(fp, writer=FFwriter, dpi=200,
    #             progress_callback=lambda i, n: print(f'saving frame {i}/{n}'))
