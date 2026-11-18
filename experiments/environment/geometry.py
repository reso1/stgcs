from typing import List, Tuple, Optional

import logging

import numpy as np

import cdd
from scipy.spatial import ConvexHull

from pydrake.all import (
    VPolytope, HPolyhedron, MathematicalProgram,
    MosekSolver, SolverOptions, CommonSolverOption,
    RandomGenerator,
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

logger = logging.getLogger(__name__)


class HPolyhedronSampler:
    @staticmethod
    def _basis_from_vertices(vertices: np.ndarray, ambient_dimension: int, tol: float) -> Optional[np.ndarray]:
        if vertices.size == 0:
            return None
        anchor = np.asarray(vertices[0], dtype=float)
        offsets = (np.asarray(vertices, dtype=float) - anchor).T
        rank = np.linalg.matrix_rank(offsets, tol=tol)
        if rank >= ambient_dimension:
            return None
        if rank == 0:
            return np.empty((ambient_dimension, 0))
        left_singular_vectors, _, _ = np.linalg.svd(offsets, full_matrices=False)
        return left_singular_vectors[:, :rank]

    @classmethod
    def sampling_subspace(cls, hpoly: HPolyhedron, tol: float = 1e-9) -> Optional[np.ndarray]:
        # Avoid Drake's AffineSubspace here: some degenerate HPolyhedra in MRMP delta_pos
        # sampling trip its internal rank check and abort the process.
        vertices = hpoly_to_vrep(hpoly)
        if vertices is None:
            return None
        return cls._basis_from_vertices(vertices, hpoly.ambient_dimension(), tol)

    @classmethod
    def uniform_sample(
        cls,
        hpoly: HPolyhedron,
        generator: RandomGenerator,
        previous_sample: Optional[np.ndarray] = None,
        *,
        tol: float = 1e-6,
        context: Optional[str] = None,
    ) -> np.ndarray:
        subspace = cls.sampling_subspace(hpoly)
        if subspace is None:
            return hpoly.UniformSample(generator)

        if subspace is not None and context:
            logger.debug(
                "Sampling lower-dimensional HPolyhedron in %s with affine dimension %d and ambient dimension %d.",
                context,
                subspace.shape[1],
                hpoly.ambient_dimension(),
            )
        return hpoly.UniformSample(generator, subspace=subspace, tol=tol)


def make_hpolytope(V) -> HPolyhedron:
    dim = V.shape[-1]
    if dim == 1:
        return HPolyhedron.MakeBox([V[0]], [V[1]])

    ch = ConvexHull(V)
    return HPolyhedron(ch.equations[:, :-1], -ch.equations[:, -1])


def time_extruded(hpoly:HPolyhedron, t0:float, tf:float) -> Optional[HPolyhedron]:
    if t0 < tf:
        return hpoly.CartesianProduct(HPolyhedron.MakeBox([t0], [tf]))
    
    return None


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
    
    # check if dim is unbounded
    array = np.hstack([hpoly.b().reshape(-1, 1), -hpoly.A()])
    mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    _, col_basis, _ = cdd.matrix_rank(mat)
    col_basis = set([c - 1 for c in col_basis if c > 0])  # -1 for b column
    
    if isinstance(dim, int):
        if dim not in col_basis:
            return -np.inf, np.inf
        
        min_xd_cost = prog.AddCost(xi[dim])
        lb = solver.Solve(prog, solver_options=solver_options).get_optimal_cost()
        prog.RemoveCost(min_xd_cost)
        max_xd_cost = prog.AddCost(-xi[dim])
        ub = -solver.Solve(prog, solver_options=solver_options).get_optimal_cost()
        return lb, ub

    if isinstance(dim, list):
        lb, ub = -np.inf * np.ones(len(dim)), np.inf * np.ones(len(dim))
        for d in col_basis.intersection(dim):
            # get lower bound
            min_xd_cost = prog.AddCost(xi[d])
            lb[d] = solver.Solve(prog, solver_options=solver_options).get_optimal_cost()
            prog.RemoveCost(min_xd_cost)
            # get upper bound
            max_xd_cost = prog.AddCost(-xi[d])
            ub[d] = -solver.Solve(prog, solver_options=solver_options).get_optimal_cost()
            prog.RemoveCost(max_xd_cost)

    return lb, ub


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


def remove_hpoly_redundancies(hpoly:HPolyhedron) -> HPolyhedron:
    """ Could be slow, use with caution """
    array = np.hstack([hpoly.b().reshape(-1, 1), -hpoly.A()])
    mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    cdd.matrix_canonicalize(mat)
    ret = HPolyhedron(
        A = -np.array(mat.array)[:, 1:],
        b = np.array(mat.array)[:, 0]
    )
    if not ret.IsBounded() and hpoly.IsBounded():
        return hpoly
    return ret


def hpoly_to_vrep(hpoly:HPolyhedron) -> Optional[np.ndarray]:
    array = np.hstack([hpoly.b().reshape(-1, 1), -hpoly.A()])
    mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    try:
        poly = cdd.polyhedron_from_matrix(mat)
        ext = cdd.copy_generators(poly)
        if ext.array != []:       
            return np.array(ext.array)[:, 1:]
    except RuntimeError as e:
        pass
    
    try:
        # hpoly is not full-dimensional
        return VPolytope(hpoly).vertices().T
    except:
        pass

    return None
