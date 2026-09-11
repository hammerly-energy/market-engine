"""Buses, branches, and the susceptance matrix B."""

import numpy as np

def incidence(buses, branches):
    """Incidence matrix A for the network
    
    One row per branch: +1 at the from bus, -1 at the to bus.

    The signs define the branch's positive flow direction, so A @ theta
    gives theta_from - theta_to.
    """
    col = {name: j for j, name in enumerate(buses)}
    A = np.zeros((len(branches), len(buses)))
    for i, br in enumerate(branches):
        A[i, col[br.from_bus]] = 1
        A[i, col[br.to_bus]]   = -1
    return A

def b_branch(branches):
    """Diagonal matrix of branch susceptances.
    
    P = b * Δθ

    DC approximation - flat voltage magnitudes, no losses, small angles
    """
    return np.diag([1 / br.reactance_pu for br in branches])

def b_flow(buses, branches):
    """Maps bus angles to line flows. Shape (L, N).

    b @ A, the two halves of P = b * dtheta composed:

        A   which buses each line touches, and which way is forward
        b   how stiff each line is

    The whole DC flow model in one matrix. b_flow(...) @ theta is the MW on
    every line.
    """
    A = incidence(buses, branches)
    b = b_branch(branches)
    return b @ A


def b_bus(buses, branches):
    """Maps bus angles to net bus injections. Shape (N, N).

    A.T sums each line's flow into the buses at its ends, with the sign
    convention incidence() set: it leaves the from bus and arrives at the to
    bus. Rows sum to zero and the matrix is singular -- adding a constant to
    every angle moves nothing. That singularity is the slack, and ptdf.py is
    where it gets resolved.
    """
    A = incidence(buses, branches)
    b = b_branch(branches)
    return A.T @ b @ A


def components(buses, branches):
    """Connected components of the network, as a list of sorted bus-name lists.

    A pure graph question -- reactance and limits play no part, only which
    buses a branch touches. Ordered by the bus order given, so the answer is
    reproducible rather than set-ordered.

    This exists because a disconnected network is the most common thing a user
    editing topology produces, and it is invisible in the linear algebra:

        A --AB-- B        C          B_bus is block diagonal. Each block has
                                     its own zero eigenvalue, so the matrix has
                                     TWO, and the single rank-1 fix in ptdf()
                                     only removes one. The inverse then fails
                                     with "Singular matrix" and names nothing.

    Deleting a line and typing a bus name twice both land on that same message
    from two modules away. Finding the components first turns it into a
    sentence about the network the user actually drew.
    """
    parent = {b: b for b in buses}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    for br in branches:
        a, b = find(br.from_bus), find(br.to_bus)
        if a != b:
            parent[a] = b

    groups = {}
    for b in buses:
        groups.setdefault(find(b), []).append(b)
    # Keyed on a root, which is an implementation detail; return the groups
    # themselves in first-bus order so two runs of the same network agree.
    return sorted(groups.values(), key=lambda g: buses.index(g[0]))
