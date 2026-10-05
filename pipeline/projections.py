"""Game projections: projected scores, point spreads and win chances.

A points model separate from the poll. Each team has an offense rating (points it
scores above average) and a defense rating (points it allows above average), fitted
by ridge regression on every completed game with home field estimated from the data:

    points(team vs opp) = mu + O[team] + D[opp] + h * loc

Early in the season the ratings are pulled toward last season's, regressed 10% to
average, and that pull fades to nothing by week 13. Win chance = Phi(spread / SIGMA).

Settings were chosen by a 2022-2025 week-by-week backtest (3,617 games, each predicted
only from games already played): average miss on the margin 13.0 points (closing
lines: 12.1), winner picked 74.3% (closing lines: 75.5%), 50.1% against the spread.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

FCS = "FCS"
LAMBDA = 1.0        # ridge shrinkage on every team rating
PRIOR_GAMES = 4.0   # extra pull toward last season in week 1, in games
PRIOR_FADE = 12     # weeks for that pull to fade out
PRIOR_KEEP = 0.9    # share of last season's rating carried into this one
SIGMA = 16.75       # spread-to-win-chance scale, fitted on the backtest


def fit_points(g, teams, targets=None, extra=0.0):
    names = list(teams) + [FCS]
    idx = {t: i for i, t in enumerate(names)}
    n = len(names)
    m = len(g)
    X = np.zeros((2 * m, 2 * n + 2))
    y = np.empty(2 * m)
    if m:
        hi, ai = g.home.map(idx).values, g.away.map(idx).values
        loc = np.where(g.neutral.values, 0.0, 1.0)
        r = np.arange(m)
        X[r, hi] = 1; X[r, n + ai] = 1; X[r, 2 * n] = 1; X[r, 2 * n + 1] = loc
        X[m + r, ai] = 1; X[m + r, n + hi] = 1; X[m + r, 2 * n] = 1; X[m + r, 2 * n + 1] = -loc
        y[:m], y[m:] = g.hp.values, g.ap.values
    t = np.zeros(2 * n + 2)
    if targets:
        for tm, i in idx.items():
            o, d = targets.get(tm, targets[FCS])
            t[i], t[n + i] = o, d
    pen = np.full(2 * n + 2, LAMBDA + extra)
    pen[2 * n:] = 0
    if not m:
        # no games yet: ratings are the targets, league average and home edge from last season
        return dict(O={tm: t[i] for tm, i in idx.items()}, D={tm: t[n + i] for tm, i in idx.items()},
                    mu=targets["_mu"], h=targets["_h"])
    beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y + pen * t)
    return dict(O={tm: beta[i] for tm, i in idx.items()}, D={tm: beta[n + i] for tm, i in idx.items()},
                mu=beta[2 * n], h=beta[2 * n + 1])


def last_season_targets(g_last, teams_last):
    mdl = fit_points(g_last, teams_last)
    tg = {t: (PRIOR_KEEP * mdl["O"][t], PRIOR_KEEP * mdl["D"][t]) for t in mdl["O"]}
    tg["_mu"], tg["_h"] = mdl["mu"], mdl["h"]
    return tg


def model_through(g, teams, targets, week):
    extra = PRIOR_GAMES * max(0.0, 1 - (min(week, 16) - 1) / PRIOR_FADE)
    return fit_points(g, teams, targets, extra)


def project(mdl, home, away, neutral):
    O, D = mdl["O"], mdl["D"]
    l = 0.0 if neutral else 1.0
    hs = mdl["mu"] + O.get(home, O[FCS]) + D.get(away, D[FCS]) + mdl["h"] * l
    as_ = mdl["mu"] + O.get(away, O[FCS]) + D.get(home, D[FCS]) - mdl["h"] * l
    spread = hs - as_
    return hs, as_, spread, float(norm.cdf(spread / SIGMA))


def load_upcoming(csv_path, fbs):
    d = pd.read_csv(csv_path)
    d = d[d.completed.astype(str).str.upper() != "TRUE"]
    d = d[d.home_team.isin(fbs) | d.away_team.isin(fbs)].copy()
    if d.empty:
        return d
    d["post"] = d.season_type == "postseason"
    d["order"] = d.week + np.where(d.post, 20, 0)
    d["neutral"] = d.neutral_site.astype(str).str.upper() == "TRUE"
    d["home"] = np.where(d.home_team.isin(fbs), d.home_team, FCS)
    d["away"] = np.where(d.away_team.isin(fbs), d.away_team, FCS)
    return d


def half(x):
    return round(x * 2) / 2


def game_row(mdl, gid, week, date, home, away, home_name, away_name, neutral, rank):
    hs, as_, spread, p = project(mdl, home, away, neutral)
    return dict(id=int(gid), week=int(week), date=date, home=home_name, away=away_name, neutral=bool(neutral),
                homeFCS=home == FCS, awayFCS=away == FCS, hRank=rank.get(home), aRank=rank.get(away),
                hScore=int(round(hs)), aScore=int(round(as_)), spread=half(spread), homeWin=round(p, 3))


def build_projections(g, teams, g_last, teams_last, upcoming, rank, lines=None):
    """g: this season's completed games; rank: team -> current Computer Poll rank."""
    targets = last_season_targets(g_last, teams_last)
    lines = lines or {}
    out = dict(sigma=SIGMA, games=[], nextWeek=None, record=[], hasLines=bool(lines))

    def add_line(row):
        ln = lines.get(row["id"])
        if ln:
            row["book"] = half(ln["margin"])
            row["books"] = len(ln["books"])
        return row

    # How the model did on each completed week, predicting from games before it
    for o in sorted(g.order.unique()):
        past, now = g[g.order < o], g[g.order == o]
        mdl = model_through(past, teams, targets, int(min(o, 16)))
        rows = []
        for r in now.itertuples():
            pr = game_row(mdl, r.game_id, r.week, None, r.home, r.away, r.home_team, r.away_team, r.neutral, {})
            pr.update(hFinal=int(r.hp), aFinal=int(r.ap))
            add_line(pr)
            pr["correct"] = (pr["spread"] > 0) == (r.hp > r.ap) if pr["spread"] != 0 else None
            pr["miss"] = round(abs(pr["spread"] - (r.hp - r.ap)), 1)
            rows.append(pr)
        picks = [x for x in rows if x["correct"] is not None]
        out["record"].append(dict(week=int(o if o < 20 else o - 20), post=bool(o >= 20), n=len(picks),
                                  right=sum(x["correct"] for x in picks),
                                  mae=round(float(np.mean([x["miss"] for x in rows])), 1) if rows else None,
                                  games=rows))

    # Next week's games, from every game played so far
    if upcoming is not None and not upcoming.empty:
        nxt = upcoming.order.min()
        wk = upcoming[upcoming.order == nxt].sort_values("start_date")
        mdl = model_through(g, teams, targets, int(min(nxt, 16)))
        out["nextWeek"] = int(wk.week.iloc[0])
        out["nextPost"] = bool(nxt >= 20)
        out["hfa"] = round(float(mdl["h"]) * 2, 1)
        for r in wk.itertuples():
            out["games"].append(add_line(game_row(mdl, r.game_id, r.week, str(r.start_date), r.home, r.away,
                                                  r.home_team, r.away_team, r.neutral, rank)))
    return out
