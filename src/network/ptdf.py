"""Power transfer distribution factors (shift factors), defined relative to a slack bus.

Changing the slack shifts every LMP by a constant. Price differences are invariant."""

import numpy as np

def ptdf(buses, branches, slack):
    """Power transfer distribution factors (shift factors) for a network.

    PTDFs are the linear sensitivities of line flows to bus injections:
    PTDF[l, i] is the MW that appears on line l when 1 MW is injected at bus i
    AND withdrawn at the slack. It is always a pair of actions -- a lone
    injection has nowhere to come from.

    The slack does two jobs, and both are needed here. It is the angle
    reference, theta_slack = 0, which resolves the singularity b_bus carries
    by construction. And it is the assumed withdrawal point, which is what
    makes a column mean something. Its own column is therefore exactly zero:
    inject at the slack, withdraw at the slack, nothing moves.

    Changing the slack shifts every LMP by a constant. Dispatch, congestion
    rent, and price DIFFERENCES are invariant; only the level of lambda moves
    (CLAUDE.md trap 2). One slack per connected component -- case5 is one.

    Args:
        buses: list of bus names. Fixes the column order.
        branches: list of Branch objects. Fixes the row order.
        slack: name of the slack bus

    Returns:
        (L, N) array. Rows ordered as branches, columns as buses.
    """
    from .topology import b_bus, b_flow

    # PTDF = (b @ A) @ (A.T @ b @ A)^-1
    #      = b_flow @ b_bus^-1

    # b_bus is singular by construction: adding a constant to every angle
    # changes no flow, so the all-ones vector is its null space. Adding 1 to
    # the slack column sends M @ ones = ones instead of 0, which kills that
    # null space and leaves a matrix that inverts.
    slack_index = buses.index(slack)
    e_slack = np.zeros(len(buses))
    e_slack[slack_index] = 1
    # np.outer(ones, e_slack) is all zeros except a column of 1s at the slack.
    B_bus = b_bus(buses, branches) + np.outer(np.ones(len(buses)), e_slack)

    # Compute the PTDFs.
    B_flow = b_flow(buses, branches)
    PTDF = B_flow @ np.linalg.inv(B_bus)

    # The inverse alone answers "inject 1 MW at bus i" with no counterparty,
    # which is not a physical question, and every row comes back floating by
    # an offset. Subtracting the slack column pairs each injection with a
    # withdrawal at the slack -- the actual PTDF definition -- and zeros the
    # slack column for free. Column DIFFERENCES were already correct; this
    # line only sets the anchor.
    PTDF = PTDF - PTDF[:, slack_index][:, None]

    return PTDF
