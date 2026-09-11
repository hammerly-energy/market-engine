"""Assemble LMPs from dispatch duals.

    LMP[i] = lambda + sum_over_lines( PTDF[l, i] * mu[l] )  [+ loss term]

Sign convention follows how the flow constraint was written. Verify against a
case with a published answer."""

def congestion_prices(res):
    """Price of a line's flow capacity constraint. "$XX.XX/MWh is what a 1 MW higher capacity line would be worth."
    {(line, t): $/MWh}. One signed number per line, from two raw duals."""
    # mu = mu_up - mu_dn
    return {
        (l, t): res["mu_up"][l, t] - res["mu_dn"][l, t]
        for (l, t) in res["mu_up"]
    }

def lmps(res, buses, lines, PTDF):
    """Locational Marginal Price - Cost of serving one more MW of load at a bus, including congestion.
    
    {(bus, t): $/MWh}."""

    row = {l: j for j, l in enumerate(lines)} # line name -> PTDF row
    col = {b: j for j, b in enumerate(buses)} # bus name -> PTDF col
    mu = congestion_prices(res) # {(line, t): $/MWh}

    # lmbda[t] + sum_l PTDF[l, i] * mu[l, t]
    return {
        # float() because PTDF is a numpy array and its scalars carry
        # through the sum. np.float64 compares and prints the same, but it
        # leaks numpy into every downstream consumer -- settlement, the
        # figures, a JSON dump of a run -- for no benefit.
        (i, t): float(res["lmbda"][t] + sum(PTDF[row[l], col[i]] * mu[l, t]
                                            for l in lines))
        for i in buses
        for t in res["lmbda"]
    }