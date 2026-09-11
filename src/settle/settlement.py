"""Settlement and the primary correctness test.

    sum(load payments) - sum(generator revenue) == sum_over_lines( mu[l] * limit[l] )

Must hold to floating-point tolerance. Run on every solve. A failure points at
PTDF construction, a sign convention, or dual extraction.

The identity says the market does not create or destroy money. Load pays the
LMP at its own bus, generators are paid the LMP at theirs, and when those two
numbers differ the gap is exactly what the binding lines are worth:

      load pays                                       gen is paid
    at bus E, $10                                   at bus A, $14
         |                                                |
         +------------ the gap is congestion rent --------+
                       and it equals sum(mu * limit)

With no congestion every mu is zero, the two sides are equal, and the identity
still holds -- with both halves at zero. That is why the M0 test survives M3
rather than being replaced.
"""

import math


def settle(lmp, load_mw, gen_mw, mu, limits):
    """Money flows for one hour, and the residual that must be zero.

    Args:
        lmp:     {bus: $/MWh}
        load_mw: {bus: MW} demand served
        gen_mw:  {bus: MW} generation dispatched, SUMMED TO THE BUS. Callers
                 holding {generator: MW} must aggregate first -- a generator
                 is paid the price at its bus, so the bus is the only level at
                 which this arithmetic is meaningful.
        mu:      {line: $/MWh} signed congestion price, mu_up - mu_dn
        limits:  {line: MW} thermal rating, may be inf

    Returns:
        {payments, revenue, congestion_rent, mu_times_limit, residual}, $/h.

    Non-finite limits are skipped rather than multiplied. An unlimited line
    always has mu == 0 exactly, so it contributes nothing -- but inf * 0 is
    nan, and one nan would silently destroy the residual that is the whole
    point of this function.
    """
    payments = sum(mw * lmp[b] for b, mw in load_mw.items())
    revenue = sum(mw * lmp[b] for b, mw in gen_mw.items())
    rent = sum(mu[l] * limits[l] for l in mu if math.isfinite(limits[l]))
    return {
        "payments": payments,
        "revenue": revenue,
        # What load paid over what generation collected. Measured from the
        # money side.
        "congestion_rent": payments - revenue,
        # The same quantity derived from the duals instead. Two independent
        # routes to one number; the identity is the claim that they agree.
        "mu_times_limit": rent,
        "residual": payments - revenue - rent,
    }
