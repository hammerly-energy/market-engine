"""Assemble LMPs from dispatch duals.

    LMP[i] = lambda + sum_over_lines( PTDF[l, i] * mu[l] )  [+ loss term]

Sign convention follows how the flow constraint was written. Verify against a
case with a published answer."""
