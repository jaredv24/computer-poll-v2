"""Computer Poll v2.0 prototype: rating, calibration, components, score, backtest."""
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FCS = "FCS"
W0 = dict(SOR=0.55, QW=0.20, LQ=0.15, GC=0.10)


# ---------------------------------------------------------------- data
def load_season(y):
    d = pd.read_csv(f"{HERE}/data/s{y}.csv")
    d = d[d.completed.astype(str).str.upper() == "TRUE"].dropna(subset=["home_points", "away_points"])
    fbs = set(d.loc[d.home_division == "fbs", "home_team"]) | set(d.loc[d.away_division == "fbs", "away_team"])
    d = d[d.home_team.isin(fbs) | d.away_team.isin(fbs)].copy()
    d["home"] = np.where(d.home_team.isin(fbs), d.home_team, FCS)
    d["away"] = np.where(d.away_team.isin(fbs), d.away_team, FCS)
    d["neutral"] = d.neutral_site.astype(str).str.upper() == "TRUE"
    d["order"] = d.week + np.where(d.season_type == "postseason", 20, 0)
    d["hp"], d["ap"] = d.home_points.astype(int), d.away_points.astype(int)
    d = d[d.hp != d.ap]
    return d[["home", "away", "hp", "ap", "neutral", "order"]].reset_index(drop=True), sorted(fbs)


# ---------------------------------------------------------------- Step 1
def fit_ratings(g, teams, fcs_prior=None, h0=0.3):
    """Win/loss Bradley-Terry with one ghost win + one ghost loss vs a 0-rated team.
    fcs_prior: list of (fixed_fbs_rating, fbs_won, loc_for_fbs) from earlier seasons."""
    idx = {t: i for i, t in enumerate(teams + [FCS])}
    n = len(idx)
    hi = g.home.map(idx).values
    ai = g.away.map(idx).values
    y = (g.hp > g.ap).values.astype(float)          # home won
    loc = np.where(g.neutral.values, 0.0, 1.0)      # home perspective
    if fcs_prior:
        pr = np.array(fcs_prior, dtype=float)
    else:
        pr = np.zeros((0, 3))

    def f(x):
        r, h = x[:n], x[n]
        z = r[hi] - r[ai] + h * loc
        s = np.where(y == 1, z, -z)
        nll = -log_expit(s).sum()
        gs = -expit(-s) * np.where(y == 1, 1, -1)    # dnll/dz
        gr = np.bincount(hi, gs, n) - np.bincount(ai, gs, n)
        gh = (gs * loc).sum()
        # ghost games
        nll += -(log_expit(r) + log_expit(-r)).sum()
        gr += -(expit(-r) - expit(r))
        # FCS prior games: FBS team rating fixed, FCS is r[-1]
        if len(pr):
            z2 = pr[:, 0] - r[-1] + h * pr[:, 2]
            s2 = np.where(pr[:, 1] == 1, z2, -z2)
            nll += -log_expit(s2).sum()
            gs2 = -expit(-s2) * np.where(pr[:, 1] == 1, 1, -1)
            gr[-1] += -gs2.sum()
            gh += (gs2 * pr[:, 2]).sum()
        return nll, np.append(gr, gh)

    x0 = np.append(np.zeros(n), h0)
    res = minimize(f, x0, jac=True, method="L-BFGS-B")
    r, h = res.x[:n], res.x[n]
    shift = r[:-1].mean()
    return {t: r[i] - shift for t, i in idx.items()}, h


def fcs_prior_games(hist):
    """hist: list of (games_df, ratings) for earlier seasons."""
    out = []
    for g, rat in hist:
        for row in g.itertuples():
            if (row.home == FCS) == (row.away == FCS):
                continue
            fbs_home = row.away == FCS
            fbs = row.home if fbs_home else row.away
            fbs_won = (row.hp > row.ap) == fbs_home
            loc = 0.0 if row.neutral else (1.0 if fbs_home else -1.0)
            out.append((rat[fbs] - rat[FCS], float(fbs_won), loc))
    return out


# ---------------------------------------------------------------- per-team game rows
def team_games(g):
    a = pd.DataFrame(dict(team=g.home, opp=g.away, pf=g.hp, pa=g.ap,
                          loc=np.where(g.neutral, 0, 1), order=g.order))
    b = pd.DataFrame(dict(team=g.away, opp=g.home, pf=g.ap, pa=g.hp,
                          loc=np.where(g.neutral, 0, -1), order=g.order))
    t = pd.concat([a, b], ignore_index=True)
    t = t[t.team != FCS]
    t["won"] = t.pf > t.pa
    t["m"] = t.pf - t.pa
    return t


def poisson_binomial(p):
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    return dist


def g_soft(m):
    return 21 * np.tanh(np.asarray(m) / 14)


# ---------------------------------------------------------------- Steps 3-4
def components(g, teams, rat, h_rat, cal, gc):
    a, hp = cal["a"], cal["h"]
    fbs_r = np.array(sorted((rat[t] for t in teams), reverse=True))
    rB = fbs_r[min(24, len(fbs_r) - 1)]
    tg = team_games(g)
    tg["r_opp"] = tg.opp.map(rat)
    tg["r_team"] = tg.team.map(rat)
    tg["p"] = np.clip(expit(a * (rB - tg.r_opp) + hp * tg["loc"]), 0.01, 0.99)
    tg["d"] = 1 - tg.p
    x = tg.r_team - tg.r_opp + gc["h"] * tg["loc"]
    tg["gc"] = g_soft(tg.m) - gc["c"] * np.tanh(gc["b"] * x)
    rows = []
    for t, s in tg.groupby("team"):
        W = int(s.won.sum())
        dist = poisson_binomial(s.p.values)
        S = dist[W + 1:].sum() + 0.5 * dist[W]
        rows.append(dict(team=t, W=W, L=int((~s.won).sum()),
                         SOR=norm.ppf(np.clip(1 - S, 1e-12, 1 - 1e-12)),
                         QW=(s.d[s.won] ** 2).sum(),
                         LQ=-(s.p[~s.won] ** 2).mean() if (~s.won).any() else 0.0,
                         GC=s.gc.mean(), rating=rat[t]))
    c = pd.DataFrame(rows).set_index("team")
    for k in W0:
        c["z" + k] = (c[k] - c[k].mean()) / c[k].std()
    return c, tg


def score(c, w):
    return sum(w[k] * c["z" + k] for k in w)


def h2h_rank(sc, g):
    """Sort by score, then one top-down pass of adjacent head-to-head swaps within the window."""
    order = list(sc.sort_values(ascending=False).index)
    s = sc.to_dict()
    top = [s[t] for t in order[:41]]
    window = 0.5 * np.median(-np.diff(top))
    beat = {}
    for r in g.itertuples():
        w_, l_ = (r.home, r.away) if r.hp > r.ap else (r.away, r.home)
        beat[(w_, l_)] = beat.get((w_, l_), 0) + 1
    i = 0
    while i < len(order) - 1:
        a_, b_ = order[i], order[i + 1]
        if s[a_] - s[b_] <= window and beat.get((b_, a_), 0) > beat.get((a_, b_), 0):
            order[i], order[i + 1] = b_, a_
            i += 2
        else:
            i += 1
    return order


# ---------------------------------------------------------------- baselines
def baselines(g, teams):
    tg = team_games(g)
    fb = tg[tg.opp != FCS]
    wp = tg.groupby("team").won.mean()
    # classic RPI on FBS-vs-FBS games
    wp_f = fb.groupby("team").won.mean()
    owp = {}
    for t, s in fb.groupby("team"):
        vals = []
        for o in s.opp:
            so = fb[(fb.team == o) & (fb.opp != t)]
            if len(so):
                vals.append(so.won.mean())
        owp[t] = np.mean(vals) if vals else 0.5
    owp = pd.Series(owp)
    oowp = fb.groupby("team").opp.apply(lambda os: np.mean([owp.get(o, 0.5) for o in os]))
    rpi = 0.25 * wp_f + 0.5 * owp + 0.25 * oowp
    # Colley on FBS-vs-FBS games
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    C = 2 * np.eye(n)
    b = np.ones(n)
    for r in g.itertuples():
        if r.home == FCS or r.away == FCS:
            continue
        i, j = idx[r.home], idx[r.away]
        C[i, i] += 1; C[j, j] += 1; C[i, j] -= 1; C[j, i] -= 1
        if r.hp > r.ap:
            b[i] += 0.5; b[j] -= 0.5
        else:
            b[i] -= 0.5; b[j] += 0.5
    colley = pd.Series(np.linalg.solve(C, b), index=teams)
    return {"Win %": wp + 1e-6 * rpi.reindex(wp.index).fillna(0),
            "Classic RPI": rpi, "Colley": colley}


# ---------------------------------------------------------------- evaluation
def retro(order, g):
    rank = {t: i for i, t in enumerate(order)}
    ok, n, gap = 0, 0, []
    for r in g.itertuples():
        if r.home == FCS or r.away == FCS:
            continue
        w_, l_ = (r.home, r.away) if r.hp > r.ap else (r.away, r.home)
        if w_ not in rank or l_ not in rank:
            continue
        n += 1
        if rank[w_] < rank[l_]:
            ok += 1
        else:
            gap.append(rank[w_] - rank[l_])
    return ok / n, (np.sum(gap) / n if n else 0.0)


# ---------------------------------------------------------------- driver
def main():
    seasons = list(range(2014, 2027))
    data = {y: load_season(y) for y in seasons}
    final, hist = {}, []
    # pass 1: final ratings per season (sequential for the FCS prior)
    for y in seasons:
        g, teams = data[y]
        rat, h = fit_ratings(g, teams, fcs_prior_games(hist[-3:]))
        final[y] = (rat, h)
        hist.append((g, rat))
    # Step 2 calibration: prior-week ratings, weeks >= 6 and postseason, 2015-2025
    X, Y, L = [], [], []
    for k, y in enumerate(range(2015, 2026)):
        g, teams = data[y]
        hy = [(data[yy][0], final[yy][0]) for yy in range(max(2014, y - 3), y)]
        pri = fcs_prior_games(hy)
        for o in sorted(g.order.unique()):
            if o < 6:
                continue
            past, now = g[g.order < o], g[g.order == o]
            rat, _ = fit_ratings(past, teams, pri)
            X.append((now.home.map(rat) - now.away.map(rat)).values)
            L.append(np.where(now.neutral, 0.0, 1.0))
            Y.append((now.hp > now.ap).values.astype(float))
    X, L, Y = map(np.concatenate, (X, L, Y))
    nll = lambda v: -(Y * log_expit(v[0] * X + v[1] * L) + (1 - Y) * log_expit(-(v[0] * X + v[1] * L))).sum()
    av = minimize(nll, [1.0, 0.3]).x
    cal = dict(a=float(av[0]), h=float(av[1]), n=int(len(Y)))
    # calibration table
    pr = expit(cal["a"] * X + cal["h"] * L)
    bins = pd.cut(pr, np.linspace(0, 1, 11))
    caltab = pd.DataFrame(dict(p=pr, y=Y)).groupby(bins, observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    # Game Control curve on final ratings 2015-2025
    gx, gy = [], []
    for y in range(2015, 2026):
        g, _ = data[y]
        rat, h = final[y]
        gx.append(np.c_[g.home.map(rat) - g.away.map(rat), np.where(g.neutral, 0.0, 1.0)])
        gy.append(g_soft(g.hp - g.ap))
    gx, gy = np.vstack(gx), np.concatenate(gy)
    sse = lambda v: ((gy - v[0] * np.tanh(v[1] * (gx[:, 0] + v[2] * gx[:, 1]))) ** 2).sum()
    gv = minimize(sse, [15, 0.3, 0.3], method="Nelder-Mead", options=dict(maxiter=4000)).x
    gc = dict(c=float(gv[0]), b=float(gv[1]), h=float(gv[2]))

    # per-season components and evaluation
    comps = {}
    results = []
    for y in range(2015, 2026):
        g, teams = data[y]
        rat, h = final[y]
        c, _ = components(g, teams, rat, h, cal, gc)
        comps[y] = c
    # correlations (pooled)
    pooled = pd.concat([comps[y][["SOR", "QW", "LQ", "GC"]].rank(pct=True) for y in comps])
    corr = pooled.corr(method="spearman")

    # weight fit: grid over simplex, leave-one-season-out across 2023-2025
    grid = []
    for a_ in range(0, 21):
        for b_ in range(0, 21 - a_):
            for c_ in range(0, 21 - a_ - b_):
                d_ = 20 - a_ - b_ - c_
                grid.append(dict(SOR=a_ / 20, QW=b_ / 20, LQ=c_ / 20, GC=d_ / 20))
    test_years = [2023, 2024, 2025]
    acc = {}
    for y in test_years:
        g, _ = data[y]
        c = comps[y]
        acc[y] = np.array([retro(list(score(c, w).sort_values(ascending=False).index), g)[0] for w in grid])
    cv = []
    for y in test_years:
        others = [yy for yy in test_years if yy != y]
        best = int(np.argmax(sum(acc[yy] for yy in others)))
        cv.append((y, grid[best], acc[y][best]))
    best_all = grid[int(np.argmax(sum(acc[yy] for yy in test_years)))]

    rows = []
    for y in range(2015, 2026):
        g, teams = data[y]
        c = comps[y]
        rat, _ = final[y]
        row = dict(season=y)
        order = h2h_rank(score(c, W0), g)
        row["v2.0 (starting weights)"] = retro(order, g)
        row["v2.0 SOR only"] = retro(list(c.SOR.sort_values(ascending=False).index), g)
        row["Team Rating (Step 1)"] = retro(sorted(teams, key=lambda t: -rat[t]), g)
        for name, s in baselines(g, teams).items():
            row[name] = retro(list(s.sort_values(ascending=False).index), g)
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("season")

    # 2026 current: rankings + bootstrap rank ranges
    g26, t26 = data[2026]
    hy = [(data[yy][0], final[yy][0]) for yy in (2023, 2024, 2025)]
    pri = fcs_prior_games(hy)
    rat26, h26 = final[2026]
    c26, _ = components(g26, t26, rat26, h26, cal, gc)
    c26["score"] = score(c26, W0)
    order26 = h2h_rank(c26.score, g26)
    rng = np.random.default_rng(7)
    B = int(os.environ.get("BOOT", 300))
    ranks = {t: [] for t in t26}
    gx26 = g26.copy()
    pwin = expit(cal["a"] * (g26.home.map(rat26) - g26.away.map(rat26)) + cal["h"] * np.where(g26.neutral, 0, 1)).values
    for _ in range(B):
        flip = rng.random(len(g26)) > np.where(g26.hp > g26.ap, pwin, 1 - pwin)
        gb = g26.copy()
        gb.loc[flip, ["hp", "ap"]] = gb.loc[flip, ["ap", "hp"]].values
        rb, hb = fit_ratings(gb, t26, pri)
        cb, _ = components(gb, t26, rb, hb, cal, gc)
        ob = h2h_rank(score(cb, W0), gb)
        for i, t in enumerate(ob):
            ranks[t].append(i + 1)
    top = []
    for i, t in enumerate(order26[:30]):
        lo, hi = np.percentile(ranks[t], [10, 90])
        r = c26.loc[t]
        top.append(dict(rank=i + 1, team=t, record=f"{int(r.W)}-{int(r.L)}", score=round(50 + 10 * r.score, 1),
                        range=f"{int(lo)}–{int(hi)}", SOR=round(r.SOR, 2), QW=round(r.QW, 2),
                        LQ=round(r.LQ, 2), GC=round(r.GC, 1)))
    # 2025 final top 15 for a sanity check
    g25, _ = data[2025]
    c25 = comps[2025].copy(); c25["score"] = score(c25, W0)
    o25 = h2h_rank(c25.score, g25)
    top25 = [dict(rank=i + 1, team=t, record=f"{int(c25.loc[t].W)}-{int(c25.loc[t].L)}") for i, t in enumerate(o25[:15])]
    fcs_gap = {y: round(final[y][0][FCS], 2) for y in seasons}

    out = dict(cal=cal, caltab=caltab.reset_index(drop=True).round(3).to_dict("records"), gc=gc,
               corr=corr.round(2).to_dict(), tab={str(k): {kk: [round(v[0] * 100, 1), round(v[1], 3)] for kk, v in r.items()} for k, r in tab.iterrows()},
               cv=[(y, w, round(a * 100, 1)) for y, w, a in cv], best_all=best_all,
               start_acc={y: round(acc[y][grid.index(W0)] * 100, 1) if W0 in grid else None for y in test_years},
               top26=top, top25=top25, fcs=fcs_gap, h_rating={y: round(final[y][1], 3) for y in seasons},
               games26=int(len(g26)), maxweek26=int(g26.order.max()))
    json.dump(out, open(f"{HERE}/results.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
