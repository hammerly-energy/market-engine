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
