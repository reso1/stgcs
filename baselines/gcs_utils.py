from __future__ import annotations
import numpy as np
from pydrake.all import HPolyhedron, Point as DrakePoint

def make_Cartesian_power_hpoly(cvs:HPolyhedron|DrakePoint, power:int) -> HPolyhedron|DrakePoint:
    """Create the Cartesian power of a given HPolyhedron."""
    if isinstance(cvs, DrakePoint):
        return DrakePoint(np.tile(cvs.x(), power))
    elif isinstance(cvs, HPolyhedron):
        return cvs.CartesianPower(power)
    else:
        raise TypeError("Input must be either HPolyhedron or Point.")

def depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn):
    visited_vertices = [source]
    path_vertices = [source]
    path_edges = []
    while path_vertices[-1] != target:
        candidate_edges = getCandidateEdgesFn(path_vertices[-1], visited_vertices)
        if len(candidate_edges) == 0:
            if len(path_vertices) > 0 and len(path_edges) > 0:
                path_vertices.pop()
                path_edges.pop()
        else:
            next_edge, next_vertex = edgeSelectorFn(candidate_edges)
            visited_vertices.append(next_vertex)
            path_vertices.append(next_vertex)
            path_edges.append(next_edge)
    return path_edges

def outgoingEdges(gcs):
    outgoing_edges = {u.id(): [] for u in gcs.Vertices()}
    for e in gcs.Edges():
        outgoing_edges[e.u().id()].append(e)
    return outgoing_edges

def extractEdgeFlows(gcs, result):
    return {e.id(): result.GetSolution(e.phi()) for e in gcs.Edges()}

def randomEdgeSelector(candidate_edges, flows):
    candidate_flows = np.array([flows[e.id()] for e in candidate_edges])
    probabilities = candidate_flows/sum(candidate_flows)
    return np.random.choice(candidate_edges, p=probabilities)

def runTrials(source, target, getCandidateEdgesFn, edgeSelectorFn, max_paths=10, max_trials=1000):
    paths = []
    trials = 0
    while len(paths) < max_paths and trials < max_trials:
        trials += 1
        try:
            path = depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn)
            if path not in paths:
                paths.append(path)
        except IndexError as e:
            pass
    return paths

def randomForwardPathSearch(gcs, result, source, target, max_paths=10, max_trials=100, seed=None, flow_tol=1e-5, **kwargs):

    if seed is not None:
        np.random.seed(seed)

    outgoing_edges = outgoingEdges(gcs)
    flows = extractEdgeFlows(gcs, result)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        keepEdge = lambda e: e.v() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in outgoing_edges[current_vertex.id()] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        e = randomEdgeSelector(candidate_edges, flows)
        return e, e.v()

    return runTrials(source, target, getCandidateEdgesFn, edgeSelectorFn, max_paths, max_trials)
