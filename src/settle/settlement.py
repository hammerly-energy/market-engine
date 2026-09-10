"""Settlement and the primary correctness test.

    sum(load payments) - sum(generator revenue) == sum_over_lines( mu[l] * limit[l] )

Must hold to floating-point tolerance. Run on every solve. A failure points at
PTDF construction, a sign convention, or dual extraction."""
