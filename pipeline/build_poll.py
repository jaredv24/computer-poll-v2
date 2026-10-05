"""Build the week-by-week Computer Poll for one season.

Writes data/poll-<season>.json: every FBS team ranked after each completed week,
with Strength of Record, Quality Wins, Loss Quality, opponent-adjusted efficiency,
and the AP Top 25 for comparison.

Game results come from the public cfbfastR schedule files. Efficiency stats and
the AP poll come from the CollegeFootballData API when CFBD_API_KEY is set; without
it the poll is built from results alone and the AP comparison is left empty.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "pipeline", ".cache")
FCS = "FCS"
SCHEDULE_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_{y}.csv"
CFBD = "https://api.collegefootballdata.com"

# Fitted on 2015-2025 in prototype/v2.py
CAL_A, CAL_H = 0.81, 0.34
WEIGHTS = dict(SOR=0.55, QW=0.20, LQ=0.15, EFF=0.10)
EFF_RIDGE = 4.0          # shrinkage toward 0 for opponent-adjusted efficiency, in games


# ---------------------------------------------------------------- data
def fetch(url, headers=None, dest=None):
    req = urllib.request.Request(url, headers={"User-Agent": "computer-poll", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    if dest:
        with open(dest, "wb") as f:
            f.write(body)
    return body


def load_season(y, refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"s{y}.csv")
    if refresh or not os.path.exists(path):
        fetch(SCHEDULE_URL.format(y=y), dest=path)
    d = pd.read_csv(path)
    d = d[d.completed.astype(str).str.upper() == "TRUE"].dropna(subset=["home_points", "away_points"])
    fbs = set(d.loc[d.home_division == "fbs", "home_team"]) | set(d.loc[d.away_division == "fbs", "away_team"])
    conf = {}
    for side in ("home", "away"):
        sub = d[d[f"{side}_division"] == "fbs"]
        conf.update(dict(zip(sub[f"{side}_team"], sub[f"{side}_conference"].fillna(""))))
    d = d[d.home_team.isin(fbs) | d.away_team.isin(fbs)].copy()
    d["home"] = np.where(d.home_team.isin(fbs), d.home_team, FCS)
    d["away"] = np.where(d.away_team.isin(fbs), d.away_team, FCS)
    d["neutral"] = d.neutral_site.astype(str).str.upper() == "TRUE"
    d["post"] = d.season_type == "postseason"
    d["order"] = d.week + np.where(d.post, 20, 0)
    d["hp"], d["ap"] = d.home_points.astype(int), d.away_points.astype(int)
    d = d[d.hp != d.ap]
    cols = ["game_id", "week", "post", "home", "away", "hp", "ap", "neutral", "order"]
    return d[cols].reset_index(drop=True), sorted(fbs), conf


def cfbd(path, **params):
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        return None
    url = f"{CFBD}{path}?{urllib.parse.urlencode(params)}"
    return json.loads(fetch(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}))


# ---------------------------------------------------------------- Step 1: rating
def fit_ratings(g, teams, fcs_prior=()):
    """Win/loss Bradley-Terry, one ghost win + one ghost loss per team vs a 0-rated team."""
    idx = {t: i for i, t in enumerate(list(teams) + [FCS])}
    n = len(idx)
    hi, ai = g.home.map(idx).values, g.away.map(idx).values
    y = (g.hp > g.ap).values
    loc = np.where(g.neutral.values, 0.0, 1.0)
    pr = np.array(fcs_prior, dtype=float).reshape(-1, 3)
    sgn = np.where(y, 1.0, -1.0)
    sgn2 = np.where(pr[:, 1] == 1, 1.0, -1.0)

    def f(x):
        r, h = x[:n], x[n]
        s = sgn * (r[hi] - r[ai] + h * loc)
        gs = -expit(-s) * sgn
        nll = -log_expit(s).sum() - (log_expit(r) + log_expit(-r)).sum()
        gr = np.bincount(hi, gs, n) - np.bincount(ai, gs, n) - (expit(-r) - expit(r))
        gh = (gs * loc).sum()
        if len(pr):
            s2 = sgn2 * (pr[:, 0] - r[-1] + h * pr[:, 2])
            gs2 = -expit(-s2) * sgn2
            nll += -log_expit(s2).sum()
            gr[-1] -= gs2.sum()
            gh += (gs2 * pr[:, 2]).sum()
        return nll, np.append(gr, gh)

    res = minimize(f, np.append(np.zeros(n), 0.3), jac=True, method="L-BFGS-B")
    r = res.x[:n]
    shift = r[:-1].mean()
    return {t: float(r[i] - shift) for t, i in idx.items()}


def fcs_prior_games(hist):
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


# ---------------------------------------------------------------- efficiency
def adjusted_efficiency(adv, teams, upto_week):
    """Opponent-adjusted offense and defense per play, by ridge regression on game rows.

    Model for each game row (team on offense vs opponent's defense):
        value = mu + O[team] + D[opp] + home_edge * loc
    D is what a defense allows, so lower is better. Net = O - D.
    """
    if adv is None or adv.empty:
        return None
    rows = adv[adv.week <= upto_week]
    if rows.empty:
        return None
    names = sorted(set(teams) | {FCS})
    idx = {t: i for i, t in enumerate(names)}
    n = len(names)
    out = {}
    for metric in ("ppa", "successRate"):
        r = rows.dropna(subset=[metric])
        o = r.team.map(lambda t: idx.get(t, idx[FCS])).values
        d = r.opponent.map(lambda t: idx.get(t, idx[FCS])).values
        loc = r["loc"].values.astype(float)
        yv = r[metric].values.astype(float)
        m = len(yv)
        X = np.zeros((m, 2 * n + 2))
        X[np.arange(m), o] = 1
        X[np.arange(m), n + d] = 1
        X[:, 2 * n] = 1           # mu
        X[:, 2 * n + 1] = loc     # home edge
        pen = np.full(2 * n + 2, EFF_RIDGE)
        pen[2 * n:] = 0
        beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ yv)
        mu = beta[2 * n]
        out[metric] = {t: (float(mu + beta[idx[t]]), float(mu + beta[n + idx[t]])) for t in teams}
    return out


# ---------------------------------------------------------------- components
def team_games(g):
    a = pd.DataFrame(dict(team=g.home, opp=g.away, pf=g.hp, pa=g.ap, loc=np.where(g.neutral, 0, 1)))
    b = pd.DataFrame(dict(team=g.away, opp=g.home, pf=g.ap, pa=g.hp, loc=np.where(g.neutral, 0, -1)))
    t = pd.concat([a, b], ignore_index=True)
    t = t[t.team != FCS]
    t["won"] = t.pf > t.pa
    return t


def poisson_binomial(p):
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    return dist


def components(g, teams, rat, eff):
    played = sorted(set(g.home) | set(g.away))
    teams = [t for t in teams if t in played]
    fbs_r = sorted((rat[t] for t in teams), reverse=True)
    rB = fbs_r[min(24, len(fbs_r) - 1)]
    tg = team_games(g)
    tg["p"] = np.clip(expit(CAL_A * (rB - tg.opp.map(rat)) + CAL_H * tg["loc"]), 0.01, 0.99)
    rows = []
    for t, s in tg.groupby("team"):
        W, L = int(s.won.sum()), int((~s.won).sum())
        dist = poisson_binomial(s.p.values)
        S = dist[W + 1:].sum() + 0.5 * dist[W]
        row = dict(team=t, W=W, L=L,
                   SOR=float(norm.ppf(np.clip(1 - S, 1e-12, 1 - 1e-12))),
                   QW=float(((1 - s.p[s.won]) ** 2).sum()),
                   LQ=float(-(s.p[~s.won] ** 2).mean()) if L else 0.0,
                   rating=rat[t])
        if eff:
            oe, de = eff["ppa"].get(t, (np.nan, np.nan))
            osr, dsr = eff["successRate"].get(t, (np.nan, np.nan))
            row.update(EFF=oe - de, offEPA=oe, defEPA=de, offSR=osr, defSR=dsr)
        rows.append(row)
    c = pd.DataFrame(rows).set_index("team")
    w = {k: v for k, v in WEIGHTS.items() if k in c and c[k].notna().any()}
    tot = sum(w.values())
    c["score"] = 0.0
    for k, v in w.items():
        z = (c[k] - c[k].mean()) / c[k].std()
        c["score"] += (v / tot) * z.fillna(0)
    return c


def h2h_order(c, g):
    order = list(c.score.sort_values(ascending=False).index)
    s = c.score.to_dict()
    top = [s[t] for t in order[:41]]
    window = 0.5 * float(np.median(-np.diff(top))) if len(top) > 2 else 0.0
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


# ---------------------------------------------------------------- CFBD inputs
def load_advanced(season):
    data = cfbd("/stats/game/advanced", year=season, excludeGarbageTime="true")
    if not data:
        return None
    rows = []
    for r in data:
        off = r.get("offense") or {}
        rows.append(dict(game_id=r.get("gameId"), week=r.get("week"), team=r.get("team"),
                         opponent=r.get("opponent"), ppa=off.get("ppa"),
                         successRate=off.get("successRate"), plays=off.get("plays")))
    adv = pd.DataFrame(rows)
    if adv.empty:
        return adv
    for k in ("ppa", "successRate"):
        adv[k] = pd.to_numeric(adv[k], errors="coerce")
    return adv


def attach_location(adv, g):
    """Home/away/neutral for each advanced-stats row, from the schedule."""
    if adv is None or adv.empty:
        return adv
    sched = g.set_index("game_id")
    loc = []
    for r in adv.itertuples():
        if r.game_id in sched.index:
            s = sched.loc[r.game_id]
            loc.append(0 if s.neutral else (1 if s.home == r.team else -1))
        else:
            loc.append(0)
    adv = adv.copy()
    adv["loc"] = loc
    return adv


def load_ap(season):
    data = cfbd("/rankings", year=season)
    if not data:
        return {}
    polls = {}
    for wk in data:
        if wk.get("seasonType") not in (None, "regular"):
            continue
        for p in wk.get("polls", []):
            if p.get("poll") == "AP Top 25":
                polls[int(wk["week"])] = {r["school"]: int(r["rank"]) for r in p.get("ranks", [])}
    return polls


# ---------------------------------------------------------------- build
def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def build(season):
    hist = []
    for y in range(season - 3, season):
        g, teams, _ = load_season(y)
        hist.append((g, fit_ratings(g, teams, fcs_prior_games(hist))))
    prior = fcs_prior_games(hist)

    g, teams, conf = load_season(season, refresh=True)
    reg = g[~g.post]
    adv = attach_location(load_advanced(season), g)
    ap = load_ap(season)

    weeks_out = []
    prev_rank = {}
    for wk in sorted(reg.week.unique()):
        gw = reg[reg.week <= wk]
        rat = fit_ratings(gw, teams, prior)
        eff = adjusted_efficiency(adv, teams, wk)
        c = components(gw, teams, rat, eff)
        order = h2h_order(c, gw)
        # AP poll released after this week's games carries the next week number in CFBD
        ap_wk = ap.get(int(wk) + 1, {})
        rows = []
        for i, t in enumerate(order):
            r = c.loc[t]
            rows.append(dict(rank=i + 1, team=t, conf=conf.get(t, ""), w=int(r.W), l=int(r.L),
                             score=round(50 + 10 * float(r.score), 1),
                             sor=r3(r.SOR), qw=r3(r.QW), lq=r3(r.LQ), rating=r3(r.rating),
                             eff=r3(r.get("EFF")), offEPA=r3(r.get("offEPA")), defEPA=r3(r.get("defEPA")),
                             offSR=r3(r.get("offSR")), defSR=r3(r.get("defSR")),
                             ap=ap_wk.get(t), prev=prev_rank.get(t)))
        prev_rank = {x["team"]: x["rank"] for x in rows}
        summary = {}
        if ap_wk:
            ours = {x["team"]: x["rank"] for x in rows}
            top25 = {x["team"] for x in rows[:25]}
            in_both = [t for t in ap_wk if t in ours]
            summary = dict(apWeek=int(wk) + 1, overlap=len(top25 & set(ap_wk)),
                           spearman=r3(spearmanr([ap_wk[t] for t in in_both], [ours[t] for t in in_both])[0]) if len(in_both) > 2 else None)
        weeks_out.append(dict(week=int(wk), games=int(len(gw)), teams=rows, vsAP=summary))
        print(f"week {wk}: {len(rows)} teams ranked, AP week {int(wk) + 1}: {'yes' if ap_wk else 'no'}", flush=True)

    out = dict(season=season, updated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
               efficiency=adv is not None and not adv.empty, apAvailable=bool(ap),
               weights=WEIGHTS, weeks=weeks_out)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    path = os.path.join(ROOT, "data", f"poll-{season}.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    with open(os.path.join(ROOT, "data", "latest.json"), "w") as f:
        json.dump(dict(season=season, file=f"poll-{season}.json", updated=out["updated"]), f)
    print(f"wrote {path}")


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    build(int(sys.argv[1]) if len(sys.argv) > 1 else (now.year if now.month >= 7 else now.year - 1))
