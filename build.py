"""Picks Fresh: descarga datos, ajusta el modelo, consulta la API y genera docs/index.html."""
import io, os, re, sys, json, difflib, unicodedata, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

import uclhist

LEAGUES = {
    "bundesliga": dict(name="Bundesliga", code="D1", api=78),
    "premier": dict(name="Premier League", code="E0", api=39),
    "laliga": dict(name="La Liga", code="SP1", api=140),
}
MARKETS = {"corners": ("HC", "AC"), "sot": ("HST", "AST"), "shots": ("HS", "AS")}
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
API = "https://v3.football.api-sports.io"
KEY = os.environ.get("API_FOOTBALL_KEY", "").strip()
HN = dt.timedelta(hours=-6)  # Honduras UTC-6
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ datos
def season_codes(today):
    y = today.year if today.month >= 7 else today.year - 1
    return [f"{k % 100:02d}{(k + 1) % 100:02d}" for k in range(y - 4, y + 1)]


def fetch_frames(code, seasons):
    frames = []
    for ss in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{ss}/{code}.csv"
        try:
            r = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            frames.append(pd.read_csv(io.BytesIO(r.content), encoding="latin-1"))
            log(f"  {code} {ss}: ok")
        except Exception as e:  # noqa: BLE001
            log(f"  {code} {ss}: no se pudo bajar ({e})")
    return frames


def prepare(frames):
    ok = [d for d in frames if {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(d.columns)]
    if not ok:
        raise RuntimeError("sin datos")
    df = pd.concat(ok, ignore_index=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last").copy()
    y = df.Date.dt.year.where(df.Date.dt.month >= 7, df.Date.dt.year - 1)
    df["season"] = (y % 100).astype(str).str.zfill(2) + ((y + 1) % 100).astype(str).str.zfill(2)
    for c in KEEP[3:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[[c for c in KEEP if c in df.columns] + ["season"]].sort_values("Date").reset_index(drop=True)
    df["FTHG"], df["FTAG"] = df.FTHG.astype(int), df.FTAG.astype(int)
    return df


# ------------------------------------------------------------------ modelo
def _disp(y, mu):
    return max(float(np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2)), 1e-4)


def fit_model(train, ref_date, half_life, reg, hcol, acol):
    train = train.dropna(subset=[hcol, acol])
    teams = sorted(set(train.HomeTeam) | set(train.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = train.HomeTeam.map(idx).to_numpy()
    a = train.AwayTeam.map(idx).to_numpy()
    gh = train[hcol].to_numpy(float)
    ga = train[acol].to_numpy(float)
    days = (ref_date - train.Date).dt.days.to_numpy(float)
    w = 0.5 ** (days / half_life)

    def nll(x):
        mu, home = x[0], x[1]
        att, de = x[2:2 + n], x[2 + n:]
        lh = mu + home + att[h] - de[a]
        la = mu + att[a] - de[h]
        elh, ela = np.exp(lh), np.exp(la)
        ll = np.sum(w * (gh * lh - elh + ga * la - ela))
        rh, ra = w * (gh - elh), w * (ga - ela)
        g = np.empty_like(x)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g[2:2 + n] = np.bincount(h, rh, n) + np.bincount(a, ra, n)
        g[2 + n:] = -np.bincount(a, rh, n) - np.bincount(h, ra, n)
        pen = reg * (np.sum(att ** 2) + np.sum(de ** 2))
        gp = 2 * reg * x
        gp[0] = gp[1] = 0.0
        return -ll + pen, -g + gp

    x0 = np.zeros(2 + 2 * n)
    x0[0] = np.log(max((gh.mean() + ga.mean()) / 2, 0.5))
    x = minimize(nll, x0, jac=True, method="L-BFGS-B").x
    att, de = x[2:2 + n], x[2 + n:]
    lh = np.exp(x[0] + x[1] + att[h] - de[a])
    la = np.exp(x[0] + att[a] - de[h])
    return {
        "mu": float(x[0]), "home": float(x[1]),
        "ah": _disp(gh, lh), "aa": _disp(ga, la), "at": _disp(gh + ga, lh + la),
        "att": {t: round(float(att[i]), 4) for i, t in enumerate(teams)},
        "de": {t: round(float(de[i]), 4) for i, t in enumerate(teams)},
    }


def league_params(df, ref):
    models = {"goles": fit_model(df, ref, 365, 4.0, "FTHG", "FTAG")}
    for k, (h, a) in MARKETS.items():
        if h in df.columns and a in df.columns:
            models[k] = fit_model(df, ref, 365, 40.0, h, a)
    last = df[df.season == df.season.max()]
    teams = sorted(set(last.HomeTeam) | set(last.AwayTeam))
    form = {}
    for t in teams:
        g = df[(df.HomeTeam == t) | (df.AwayTeam == t)].tail(5)
        s = ""
        for _, r in g.iterrows():
            gf, ga = (r.FTHG, r.FTAG) if r.HomeTeam == t else (r.FTAG, r.FTHG)
            s += "G" if gf > ga else ("E" if gf == ga else "P")
        form[t] = s
    ht = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    hi = {t: i for i, t in enumerate(ht)}
    nan = pd.Series([np.nan] * len(df), index=df.index)
    hcs, acs = df["HC"] if "HC" in df.columns else nan, df["AC"] if "AC" in df.columns else nan
    def _cards(y, r):  # tarjetas del equipo: amarillas + rojas (-1 si faltan datos)
        if not (pd.notna(y) and pd.notna(r)):
            return -1
        return int(y) + int(r)

    def _col(n):
        return df[n] if n in df.columns else nan

    hks = [_cards(y, r) for y, r in zip(_col("HY"), _col("HR"))]
    aks = [_cards(y, r) for y, r in zip(_col("AY"), _col("AR"))]
    hm = [[int(d.strftime("%Y%m%d")), hi[h], hi[a], int(x), int(y),
           int(c1) if pd.notna(c1) else -1, int(c2) if pd.notna(c2) else -1, k1, k2]
          for d, h, a, x, y, c1, c2, k1, k2 in zip(df.Date, df.HomeTeam, df.AwayTeam, df.FTHG, df.FTAG, hcs, acs, hks, aks)]
    return {"updated": str(df.Date.max().date()), "n": int(len(df)), "models": models,
            "teams": teams, "form": form, "inj": None, "hteams": ht, "hm": hm}


# ------------------------------------------------------------------ nombres
ALIAS = {
    "bayern munchen": "Bayern Munich", "borussia dortmund": "Dortmund", "eintracht frankfurt": "Ein Frankfurt",
    "1 fc koln": "FC Koln", "fc koln": "FC Koln", "koln": "FC Koln", "sc freiburg": "Freiburg",
    "1 fc heidenheim": "Heidenheim", "fc heidenheim": "Heidenheim", "tsg hoffenheim": "Hoffenheim",
    "tsg 1899 hoffenheim": "Hoffenheim", "bayer leverkusen": "Leverkusen", "bayer 04 leverkusen": "Leverkusen",
    "borussia monchengladbach": "M'gladbach", "fsv mainz 05": "Mainz", "1 fsv mainz 05": "Mainz",
    "fc st pauli": "St Pauli", "vfb stuttgart": "Stuttgart", "1 fc union berlin": "Union Berlin",
    "vfl wolfsburg": "Wolfsburg", "hamburger sv": "Hamburg", "fc augsburg": "Augsburg", "vfl bochum": "Bochum",
    "holstein kiel": "Holstein Kiel", "sv werder bremen": "Werder Bremen",
    "manchester city": "Man City", "manchester united": "Man United", "nottingham forest": "Nott'm Forest",
    "tottenham hotspur": "Tottenham", "wolverhampton wanderers": "Wolves", "newcastle united": "Newcastle",
    "west ham united": "West Ham", "brighton hove albion": "Brighton", "leeds united": "Leeds",
    "athletic club": "Ath Bilbao", "athletic bilbao": "Ath Bilbao", "atletico madrid": "Ath Madrid",
    "real betis": "Betis", "celta vigo": "Celta", "espanyol": "Espanol", "real sociedad": "Sociedad",
    "rayo vallecano": "Vallecano", "deportivo alaves": "Alaves", "real oviedo": "Oviedo",
    "real valladolid": "Valladolid", "celta de vigo": "Celta", "rcd mallorca": "Mallorca",
    "rcd espanyol": "Espanol", "ca osasuna": "Osasuna", "sevilla fc": "Sevilla", "valencia cf": "Valencia",
    "villarreal cf": "Villarreal", "girona fc": "Girona", "getafe cf": "Getafe", "elche cf": "Elche",
    "levante ud": "Levante", "fc barcelona": "Barcelona", "ud las palmas": "Las Palmas",
}


def norm(s):
    s = unicodedata.normalize("NFKD", s.replace("ø", "o").replace("Ø", "O").replace("æ", "ae")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("'", ""))
    return re.sub(r"\s+", " ", s).strip()


def match_team(name, teams):
    n = norm(name)
    if n in ALIAS and ALIAS[n] in teams:
        return ALIAS[n]
    for t in teams:
        if norm(t) == n:
            return t
    best = difflib.get_close_matches(n, [norm(t) for t in teams], n=1, cutoff=0.6)
    if best:
        for t in teams:
            if norm(t) == best[0]:
                return t
    for t in teams:
        if norm(t) in n or n in norm(t):
            return t
    return None


# ------------------------------------------------------------------ API
def api(path, params):
    if not KEY:
        return None
    try:
        r = requests.get(API + path, headers={"x-apisports-key": KEY}, params=params, timeout=40)
        j = r.json()
    except Exception as e:  # noqa: BLE001
        log(f"  API {path}: error de conexión ({e})")
        return None
    errs = j.get("errors")
    if errs:
        log(f"  API {path} {params}: {errs}")
        return None
    return j.get("response", [])


def parse_odds(resp):
    acc = {}
    for item in resp or []:
        for bk in item.get("bookmakers", []):
            for bet in bk.get("bets", []):
                n = str(bet.get("name", "")).lower()
                for v in bet.get("values", []):
                    try:
                        o = float(v["odd"])
                    except Exception:  # noqa: BLE001
                        continue
                    val = str(v.get("value", "")).strip().lower()
                    k = None
                    m = re.match(r"(over|under) ([\d.]+)$", val)
                    if n == "match winner":
                        k = {"home": "1", "draw": "X", "away": "2"}.get(val)
                    elif n == "goals over/under" and m:
                        k = f"{m[1][0]}:goals:{m[2]}"
                    elif n == "both teams score":
                        k = {"yes": "btts:y", "no": "btts:n"}.get(val)
                    elif "corner" in n and m and not any(w in n for w in ("1st", "2nd", "half", "asian", "team", "home", "away")):
                        k = f"{m[1][0]}:corners:{m[2]}"
                    if k:
                        acc.setdefault(k, []).append(o)
    return {k: round(sum(v) / len(v), 2) for k, v in acc.items()}


def api_enrich(params, today):
    """Agrega partidos próximos, lesionados y cuotas. Si algo falla, la página funciona igual."""
    note = "Sin clave de API: la página usa solo el modelo y tus cuotas."
    fixtures = []
    if not KEY:
        return fixtures, note
    season = today.year if today.month >= 7 else today.year - 1
    by_id = {v["api"]: k for k, v in LEAGUES.items()}
    calls_ok = 0
    for d in (today, today + dt.timedelta(days=1)):
        resp = api("/fixtures", {"date": d.isoformat(), "timezone": "America/Tegucigalpa"})
        if resp is None:
            continue
        calls_ok += 1
        for f in resp:
            lid = f["league"]["id"]
            if lid not in by_id:
                continue
            key = by_id[lid]
            teams = params[key]["teams"]
            h, a = match_team(f["teams"]["home"]["name"], teams), match_team(f["teams"]["away"]["name"], teams)
            if not h or not a:
                log(f"  sin equivalencia de nombre: {f['teams']['home']['name']} / {f['teams']['away']['name']}")
                continue
            fixtures.append({"id": f["fixture"]["id"], "lg": key, "date": f["fixture"]["date"][:16],
                             "status": f["fixture"]["status"]["short"], "h": h, "a": a, "odds": {}})
    for key, v in LEAGUES.items():
        resp = api("/injuries", {"league": v["api"], "season": season})
        if resp is None:
            continue
        inj = {}
        for it in resp:
            t = match_team(it["team"]["name"], params[key]["teams"])
            if t:
                pl = it["player"]
                row = [pl.get("name", "?"), pl.get("reason") or pl.get("type") or ""]
                if row not in inj.setdefault(t, []):
                    inj[t].append(row)
        params[key]["inj"] = inj
    n_odds = 0
    for f in sorted(fixtures, key=lambda x: x["date"])[:25]:
        if f["status"] not in ("NS", "TBD"):
            continue
        resp = api("/odds", {"fixture": f["id"]})
        if resp:
            f["odds"] = parse_odds(resp)
            n_odds += bool(f["odds"])
    fixtures.sort(key=lambda x: x["date"])
    if calls_ok == 0:
        errs = [l.strip() for l in LOG if l.strip().startswith("API")]
        note = "La API no respondió (revisa la clave o el plan). La página usa solo el modelo y tus cuotas. Detalle: " + (errs[0][:300] if errs else "sin detalle")
    elif not fixtures:
        note = "La API respondió, pero no hay partidos de estas ligas hoy ni mañana."
    else:
        note = f"{len(fixtures)} partidos próximos. Cuotas cargadas en {n_odds}."
    return fixtures, note


def csv_fixtures(params, today, text=None):
    """Próximos partidos y cuotas promedio desde football-data.co.uk/fixtures.csv (sin clave)."""
    from zoneinfo import ZoneInfo
    try:
        if text is None:
            r = requests.get("https://www.football-data.co.uk/fixtures.csv", timeout=40,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
        else:
            df = pd.read_csv(io.StringIO(text))
    except Exception as e:  # noqa: BLE001
        log(f"  fixtures.csv: no se pudo leer ({e})")
        return [], "No se pudo leer la lista de partidos. Elige los equipos a mano."
    df.columns = [re.sub(r"[^\x20-\x7e]", "", str(c)).strip() for c in df.columns]
    code_to_key = {v["code"]: k for k, v in LEAGUES.items()}
    uk, hn = ZoneInfo("Europe/London"), ZoneInfo("America/Tegucigalpa")

    def num(row, cols):
        for c in cols:
            if c in row and pd.notna(row[c]):
                try:
                    v = float(row[c])
                    if v > 1:
                        return round(v, 2)
                except Exception:  # noqa: BLE001
                    pass
        return None

    out = []
    for _, row in df.iterrows():
        key = code_to_key.get(str(row.get("Div", "")).strip())
        if not key or key not in params:
            continue
        d = pd.to_datetime(row.get("Date"), dayfirst=True, errors="coerce")
        if pd.isna(d):
            continue
        tm = str(row.get("Time", "") or "12:00")
        try:
            hh, mm = [int(x) for x in tm.split(":")[:2]]
        except Exception:  # noqa: BLE001
            hh, mm = 12, 0
        loc = dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=uk).astimezone(hn)
        if not (today <= loc.date() <= today + dt.timedelta(days=7)):
            continue
        if loc < dt.datetime.now(hn) - dt.timedelta(hours=2):
            continue
        teams = params[key]["teams"]
        h, a = match_team(str(row["HomeTeam"]), teams), match_team(str(row["AwayTeam"]), teams)
        if not h or not a or h == a:
            log(f"  sin equivalencia: {row.get('HomeTeam')} / {row.get('AwayTeam')}")
            continue
        odds = {}
        for k, cols in {"1": ["AvgH", "B365H"], "X": ["AvgD", "B365D"], "2": ["AvgA", "B365A"],
                        "o:goals:2.5": ["Avg>2.5", "B365>2.5"], "u:goals:2.5": ["Avg<2.5", "B365<2.5"]}.items():
            v = num(row, cols)
            if v:
                odds[k] = v
        out.append({"id": len(out) + 1, "lg": key, "date": loc.strftime("%Y-%m-%dT%H:%M"),
                    "status": "NS", "h": h, "a": a, "odds": odds})
    out.sort(key=lambda x: x["date"])
    if out:
        con = sum(1 for f in out if f["odds"])
        note = f"{len(out)} partidos de hoy a los próximos 7 días (fuente: football-data.co.uk). Cuotas promedio en {con}."
    else:
        info = f"{len(df)} filas; ligas: {sorted(set(str(x) for x in df.get('Div', [])))}; fechas: {list(df.get('Date', []))[:3]}"
        note = "football-data.co.uk no tiene partidos de estas ligas en los próximos 7 días. Elige los equipos a mano. (" + info + ")"
    return out, note



def market_lambdas(odds, start):
    """Goles esperados que reproducen las cuotas 1X2 y más/menos 2.5 (sin margen). Devuelve (local, visitante) o None."""
    try:
        from scipy.optimize import least_squares
        from scipy.stats import poisson
        k = ["1", "X", "2", "o:goals:2.5", "u:goals:2.5"]
        if not all(odds.get(x) and odds[x] > 1 for x in k):
            return None
        q = np.array([1 / odds[x] for x in k[:3]]); q = q / q.sum()
        po = (1 / odds["o:goals:2.5"]) / (1 / odds["o:goals:2.5"] + 1 / odds["u:goals:2.5"])
        tgt = np.r_[q, po]
        G = np.arange(11); i, j = np.indices((11, 11))

        def tg(x):
            m = np.outer(poisson.pmf(G, np.exp(x[0])), poisson.pmf(G, np.exp(x[1]))); m = m / m.sum()
            return np.array([m[i > j].sum(), m[i == j].sum(), m[i < j].sum(), m[(i + j) >= 3].sum()]) - tgt

        r = least_squares(tg, np.log(np.maximum(start, 0.2)))
        lh, la = float(np.exp(r.x[0])), float(np.exp(r.x[1]))
        return [round(lh, 4), round(la, 4)] if 0.1 < lh < 6 and 0.1 < la < 6 else None
    except Exception as e:  # noqa: BLE001
        log(f"  lambdas del mercado: {e}")
        return None


UCL_EXTRA = {"atletico madrid": "Ath Madrid", "borussia dortmund": "Dortmund", "rb leipzig": "RB Leipzig",
             "bayern munich": "Bayern Munich", "real betis": "Betis", "manchester city": "Man City",
             "manchester united": "Man United"}


def ucl_domestic(params):
    """Para cada club de Champions, en qué liga local tenemos su historial (solo Bundesliga, Premier y La Liga)."""
    out = {}
    for t in UCL_TEAMS:
        n = norm(t)
        for key in LEAGUES:
            teams = params.get(key, {}).get("hteams", [])
            cands = [UCL_EXTRA.get(n), ALIAS.get(n), t]
            name = next((c for c in cands if c and c in teams), None)
            if not name:
                name = next((x for x in teams if norm(x) == n), None)
            if name:
                out[t] = [key, name]
                break
    return out

# ------------------------------------------------------------------ Champions League (Elo)
UCL_TEAMS = ["AEK Athens", "Arsenal", "Aston Villa", "Atlético Madrid", "Barcelona", "Bayern Munich", "Bodø/Glimt",
             "Borussia Dortmund", "Club Brugge", "Como", "Feyenoord", "Fenerbahçe", "Galatasaray", "Inter Milan",
             "LASK Linz", "Lens", "Lille", "Liverpool", "Manchester City", "Manchester United", "Napoli",
             "Paris Saint-Germain", "Porto", "PSV Eindhoven", "Real Betis", "Real Madrid", "RB Leipzig", "Roma",
             "Sabah", "Shakhtar Donetsk", "Slavia Prague", "Slovan Bratislava", "Sporting Lisbon", "Stuttgart",
             "Villarreal", "Viking"]
# fecha, hora de Honduras, local, visitante (jornadas 2 y 3 de la fase de liga 2026/27)
UCL_FIXTURES = [
    ("2026-10-13", "10:45", "Lens", "Sporting Lisbon"), ("2026-10-13", "10:45", "Sabah", "Slavia Prague"),
    ("2026-10-13", "13:00", "Arsenal", "Lille"), ("2026-10-13", "13:00", "Atlético Madrid", "Manchester United"),
    ("2026-10-13", "13:00", "Inter Milan", "Club Brugge"), ("2026-10-13", "13:00", "Galatasaray", "Barcelona"),
    ("2026-10-13", "13:00", "RB Leipzig", "PSV Eindhoven"), ("2026-10-13", "13:00", "Viking", "Bayern Munich"),
    ("2026-10-13", "13:00", "Villarreal", "Napoli"),
    ("2026-10-14", "10:45", "Feyenoord", "Como"), ("2026-10-14", "10:45", "LASK Linz", "Liverpool"),
    ("2026-10-14", "13:00", "Roma", "Real Madrid"), ("2026-10-14", "13:00", "Aston Villa", "Fenerbahçe"),
    ("2026-10-14", "13:00", "Shakhtar Donetsk", "AEK Athens"), ("2026-10-14", "13:00", "Bodø/Glimt", "Borussia Dortmund"),
    ("2026-10-14", "13:00", "Manchester City", "Paris Saint-Germain"), ("2026-10-14", "13:00", "Real Betis", "Porto"),
    ("2026-10-14", "13:00", "Slovan Bratislava", "Stuttgart"),
    ("2026-10-20", "10:45", "Fenerbahçe", "Slavia Prague"), ("2026-10-20", "10:45", "Sabah", "Borussia Dortmund"),
    ("2026-10-20", "13:00", "Roma", "Slovan Bratislava"), ("2026-10-20", "13:00", "Porto", "PSV Eindhoven"),
    ("2026-10-20", "13:00", "Liverpool", "Villarreal"), ("2026-10-20", "13:00", "Manchester City", "AEK Athens"),
    ("2026-10-20", "13:00", "Paris Saint-Germain", "Barcelona"), ("2026-10-20", "13:00", "Napoli", "Bodø/Glimt"),
    ("2026-10-20", "13:00", "Stuttgart", "Atlético Madrid"),
    ("2026-10-21", "10:45", "Como", "Manchester United"), ("2026-10-21", "10:45", "Lille", "Galatasaray"),
    ("2026-10-21", "13:00", "Aston Villa", "Viking"), ("2026-10-21", "13:00", "Club Brugge", "Lens"),
    ("2026-10-21", "13:00", "Bayern Munich", "Arsenal"), ("2026-10-21", "13:00", "Inter Milan", "Shakhtar Donetsk"),
    ("2026-10-21", "13:00", "Real Madrid", "RB Leipzig"), ("2026-10-21", "13:00", "Real Betis", "Feyenoord"),
    ("2026-10-21", "13:00", "Sporting Lisbon", "LASK Linz"),
]
ELO_NAMES = {
    "bayern munich": ["bayern"], "manchester city": ["man city"], "manchester united": ["man united"],
    "paris saint germain": ["paris sg", "psg"], "atletico madrid": ["atletico"], "rb leipzig": ["leipzig"],
    "borussia dortmund": ["dortmund"], "inter milan": ["inter"], "sporting lisbon": ["sporting"],
    "shakhtar donetsk": ["shakhtar"], "slavia prague": ["slavia praha", "slavia prague"],
    "bodo glimt": ["bodoe glimt", "bodo glimt"], "fenerbahce": ["fenerbahce"], "real betis": ["betis"],
    "psv eindhoven": ["psv"], "club brugge": ["brugge", "club brugge"], "aek athens": ["aek"],
    "lask linz": ["lask"], "slovan bratislava": ["slovan"], "sabah": ["sabah", "sabah baku"],
}


# Elo copiado a mano de clubelo.com (capturas del 11-oct-2026). Se usa solo si clubelo.com no responde desde GitHub.
STATIC_ELO_DATE = "2026-10-11"
STATIC_ELO = {
    "Bayern Munich": 2049, "Arsenal": 2043, "Manchester City": 2035, "Barcelona": 2026, "Real Madrid": 1975,
    "Paris Saint-Germain": 1958, "Inter Milan": 1949, "Liverpool": 1930, "Atlético Madrid": 1907,
    "Borussia Dortmund": 1895, "Aston Villa": 1894, "Sporting Lisbon": 1877, "Manchester United": 1876,
    "Roma": 1857, "Porto": 1842, "Napoli": 1839, "RB Leipzig": 1825, "Stuttgart": 1824, "Real Betis": 1823,
    "Lens": 1817, "Club Brugge": 1812, "Como": 1807, "Galatasaray": 1798, "Villarreal": 1796,
    "PSV Eindhoven": 1787, "Fenerbahçe": 1785,
    "Lille": 1783, "Bodø/Glimt": 1780, "Slavia Prague": 1764, "Feyenoord": 1724, "AEK Athens": 1708,
    "Viking": 1700, "Shakhtar Donetsk": 1692, "LASK Linz": 1628, "Sabah": 1561, "Slovan Bratislava": 1551,
}


def ucl_params(today, text=None):
    """Elo de cada club (clubelo.com) para la Champions League."""
    if text is None:
        import time
        for back in (0, 1, 3):
            day = (today - dt.timedelta(days=back)).isoformat()
            for url in (f"http://api.clubelo.com/{day}", f"https://api.clubelo.com/{day}"):
                for intento in range(2):
                    try:
                        r = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
                        r.raise_for_status()
                        text = r.text
                        break
                    except Exception as e:  # noqa: BLE001
                        log(f"  clubelo {url}: {str(e)[:120]}")
                        time.sleep(4)
                if text:
                    break
            if text:
                break
    if not text:
        log("  clubelo no responde: se usa el Elo guardado del " + STATIC_ELO_DATE)
        elo = {k: float(v) for k, v in STATIC_ELO.items()}
        return {"name": "Champions League", "kind": "elo", "elo": elo, "hfa": 100, "T": 2.9,
                "updated": STATIC_ELO_DATE, "n": 0, "models": {}, "teams": sorted(elo), "form": {}, "inj": None}
    try:
        df = pd.read_csv(io.StringIO(text))
        table = {norm(str(c)): float(e) for c, e in zip(df["Club"], df["Elo"])}
    except Exception as e:  # noqa: BLE001
        log(f"  clubelo: formato inesperado ({e})")
        return None
    elo = {}
    for team in UCL_TEAMS:
        n = norm(team)
        cands = [n] + ELO_NAMES.get(n, [])
        v = next((table[c] for c in cands if c in table), None)
        if v is None:
            close = difflib.get_close_matches(n, list(table), n=1, cutoff=0.85)
            v = table[close[0]] if close else None
        if v is None:
            log(f"  Elo no encontrado: {team}")
        else:
            elo[team] = round(v, 1)
    if len(elo) < 20:
        log("  Elo: demasiados equipos sin dato; se omite la Champions")
        return None
    return {"name": "Champions League", "kind": "elo", "elo": elo, "hfa": 100, "T": 2.9,
            "updated": today.isoformat(), "n": 0, "models": {}, "teams": sorted(elo), "form": {}, "inj": None}


def ucl_fixtures(params, today):
    if "ucl" not in params:
        return []
    out = []
    now = dt.datetime.now(dt.timezone(HN))
    for d, hm, h, a in UCL_FIXTURES:
        if h not in params["ucl"]["elo"] or a not in params["ucl"]["elo"]:
            continue
        day = dt.date.fromisoformat(d)
        loc = dt.datetime.fromisoformat(f"{d}T{hm}").replace(tzinfo=dt.timezone(HN))
        if not (today <= day <= today + dt.timedelta(days=10)) or loc < now - dt.timedelta(hours=2):
            continue
        out.append({"id": f"ucl-{d}-{h}", "lg": "ucl", "date": f"{d}T{hm}", "status": "NS", "h": h, "a": a, "odds": {}})
    return out


# nombre del club en football-data.co.uk -> nombre en el historial de Champions
DOM2UCL = {"Man City": "Manchester City", "Man United": "Manchester United", "Ath Madrid": "Atlético Madrid",
           "Dortmund": "Borussia Dortmund", "Ein Frankfurt": "Eintracht Frankfurt", "Leverkusen": "Bayer Leverkusen",
           "Ath Bilbao": "Athletic Club", "Betis": "Real Betis", "Newcastle": "Newcastle United", "Paris SG": "Paris Saint-Germain"}


def load_uclh():
    """Partidos de Champions: 2024/25 y 2025/26 (data/ucl_hist.json) + temporada en curso."""
    rows = []
    try:
        rows = json.loads((Path(__file__).parent / "data" / "ucl_hist.json").read_text(encoding="utf8"))["m"]
    except Exception as e:  # noqa: BLE001
        log(f"  historial de Champions no disponible: {e}")
    new = uclhist.md1_rows()
    try:  # si openfootball ya publicó 2026/27, se usa en lugar de la jornada 1 guardada a mano
        r = requests.get("https://raw.githubusercontent.com/openfootball/champions-league/master/2026-27/cl.txt", timeout=20)
        if r.ok:
            got = uclhist.parse(r.text, 2026)
            if len(got) >= 10:
                new = got
                log(f"  Champions 2026/27 desde openfootball: {len(got)} partidos")
    except Exception:  # noqa: BLE001
        pass
    seen, out = set(), []
    for r in rows + new:
        k = (r[0], r[1], r[2])
        if k not in seen:
            seen.add(k)
            out.append(r)
    out.sort(key=lambda r: r[0])
    return out


def ucl_alias(params, rows):
    names = {r[1] for r in rows} | {r[2] for r in rows}
    out = {}
    for key in LEAGUES:
        for t in params.get(key, {}).get("hteams", []):
            c = DOM2UCL.get(t, t)
            if c in names:
                out.setdefault(key, {})[t] = c
    return out


def build(template, out_path="docs/index.html"):
    now = dt.datetime.utcnow()
    local = now + HN
    today = local.date()
    seasons = season_codes(today)
    params = {}
    for key, v in LEAGUES.items():
        log(f"{v['name']}:")
        try:
            df = prepare(fetch_frames(v["code"], seasons))
            ref = df.Date.max() + pd.Timedelta(days=1)
            params[key] = league_params(df, ref)
            params[key]["name"] = v["name"]
        except Exception as e:  # noqa: BLE001
            log(f"  {v['name']} omitida: {e}")
    if not params:
        raise SystemExit("No se pudo cargar ninguna liga; no se actualiza la página.")
    u = ucl_params(today)
    if u:
        u["map"] = ucl_domestic(params)
        log("  Champions con historial local: " + ", ".join(sorted(u["map"])))
        params["ucl"] = u
    if os.environ.get("USE_API_FOOTBALL") == "1":
        fixtures, note = api_enrich(params, today)
    else:
        fixtures, note = csv_fixtures(params, today)
    for f in fixtures:
        M = params.get(f["lg"], {}).get("models", {}).get("goles") if f.get("odds") else None
        if M and f["h"] in M["att"] and f["a"] in M["att"]:
            st = [np.exp(M["mu"] + M["home"] + M["att"][f["h"]] - M["de"][f["a"]]), np.exp(M["mu"] + M["att"][f["a"]] - M["de"][f["h"]])]
            ml = market_lambdas(f["odds"], st)
            if ml:
                f["ml"] = ml
    fixtures = sorted(fixtures + ucl_fixtures(params, today), key=lambda x: x["date"])
    if "ucl" not in params:
        det = [l.strip() for l in LOG if "clubelo" in l or "Elo" in l]
        note += " (Champions: no se pudo leer clubelo.com. " + (det[-1][:260] if det else "sin detalle") + ")"
    if "ucl" in params and params["ucl"]["updated"] == STATIC_ELO_DATE != today.isoformat():
        falta = [t for t in UCL_TEAMS if t not in params["ucl"]["elo"]]
        note += f" Champions: Elo guardado del {STATIC_ELO_DATE}." + (" Faltan: " + ", ".join(falta) + "." if falta else "")
    log(note)
    uclh = load_uclh()
    data = {"leagues": params, "fixtures": fixtures, "note": note, "uclh": uclh, "uclalias": ucl_alias(params, uclh),
            "generated": local.strftime("%Y-%m-%d %H:%M") + " (hora de Honduras)"}
    html = template.replace("__PARAMS__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    log(f"Página escrita en {out_path} ({len(html) // 1024} KB)")


TEMPLATE = r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="theme-color" content="#000000"><style>:root{color-scheme:dark}html,body{margin:0;background:#000}</style>
<title>Picks Fresh</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..900&family=Figtree:wght@400;500;600;700&display=swap">
<style>
/* Diseño "pizarra de entrenador": fondo tiza, verde césped oscuro para lo importante (el marcador de probabilidades) y ámbar de tarjeta para el visitante.
   Tipografía: Archivo condensada para equipos y cifras (como un marcador de estadio), Figtree para el texto.
   Móvil primero: una sola columna, navegación abajo (Partidos, Destacados, Bitácora). */
:root{
  color-scheme:dark;
  --bg:#000000; --paper:#0f1312; --ink:#eef3ef; --muted:#8f9b94; --line:#222a26; --soft:#171d1a;
  --pitch:#0c1512; --pitch2:#16221d; --chalk:#f1f5f2; --chalk2:#b9cdc2;
  --loc:#eef3ef; --draw:#3a4a42; --vis:#f0b02a; --sel:#eef3ef; --on-sel:#06100c;
  --good:#2fd27e; --good-bg:#10321f; --bad:#ff5f6d; --bad-bg:#3a1a20; --warn-bg:#2a2210; --warn:#f0c26a;
  --disp:"Archivo","Arial Narrow","Roboto Condensed",system-ui,sans-serif; --body:"Figtree",system-ui,sans-serif;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;background:#000}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 var(--body);font-variant-numeric:tabular-nums;padding-bottom:calc(88px + env(safe-area-inset-bottom))}
h1,h2,h3,h4,p{margin:0}
button,select,input,summary{font:inherit;color:inherit}
button{cursor:pointer}
[hidden]{display:none!important}
:focus-visible{outline:3px solid var(--vis);outline-offset:2px}
.app{max-width:720px;margin:0 auto;padding:0 16px}
.disp{font-family:var(--disp);font-stretch:72%;font-weight:800;letter-spacing:.005em}

/* ---- cabecera ---- */
.bar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 0 12px}
.brand{display:flex;align-items:center;gap:10px;font:800 26px/1 var(--disp);font-stretch:72%;white-space:nowrap;flex:none}
.brand svg{flex:none}
.upd{font-size:11.5px;line-height:1.3;color:var(--muted);text-align:right;min-width:0}

/* ---- liga y día ---- */
.ctx{display:flex;flex-direction:column;gap:10px;margin-bottom:6px}
.row-scroll{display:flex;gap:8px;overflow-x:auto;scrollbar-width:none;margin:0 -16px;padding:2px 16px}
.row-scroll::-webkit-scrollbar{display:none}
.chip{flex:none;border:1.5px solid var(--line);background:var(--paper);border-radius:999px;padding:9px 16px;font-weight:600;font-size:15px}
.chip[aria-pressed="true"]{background:var(--sel);border-color:var(--sel);color:var(--on-sel)}
.day{flex:none;min-width:84px;border:1.5px solid var(--line);background:var(--paper);border-radius:14px;padding:7px 12px;text-align:center;line-height:1.2}
.day b{display:block;font:800 17px/1.1 var(--disp);font-stretch:72%}
.day small{display:block;font-size:12px;color:var(--muted);margin-top:2px}
.day[aria-pressed="true"]{border-color:var(--sel);box-shadow:inset 0 0 0 1.5px var(--sel)}

/* ---- títulos y texto ---- */
.vh{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin:22px 0 4px}
.vh h2{font:800 28px/1.1 var(--disp);font-stretch:72%}
.sub{font-size:14px;line-height:1.45;color:var(--muted)}
.sub b{color:var(--ink)}
.empty{font-size:15px;color:var(--muted);padding:10px 0}
.foot{font-size:13px;line-height:1.5;color:var(--muted);margin-top:18px}
.box{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:16px;margin-top:12px}
.box>h3{font:800 20px/1.15 var(--disp);font-stretch:75%;margin-bottom:4px}
.box>.sub+*{margin-top:12px}
.callout{background:var(--warn-bg);color:var(--ink);border-radius:12px;padding:11px 13px;font-size:14px;line-height:1.45;margin-top:12px}

/* ---- lista de partidos ---- */
.legend-q{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:6px 0 2px}
.legend-q i,.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.cards{display:flex;flex-direction:column;gap:10px;margin-top:10px}
.mc{display:grid;grid-template-columns:50px minmax(0,1fr) 112px 12px;gap:10px;align-items:center;width:100%;text-align:left;background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:14px}
.mc:active{background:var(--soft)}
.mtime{font:800 20px/1 var(--disp);font-stretch:72%;color:var(--muted)}
.mtt{display:flex;flex-direction:column;gap:2px;min-width:0}
.mtt span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:700;font-size:16px}
.mtt span+span{font-weight:500;color:var(--muted)}
.mq{display:flex;flex-direction:column;gap:5px}
.qb{display:flex;height:8px;border-radius:99px;overflow:hidden;gap:2px}
.qb i{display:block;min-width:3px}
.qb i:nth-child(1){background:var(--loc)} .qb i:nth-child(2){background:var(--draw)} .qb i:nth-child(3){background:var(--vis)}
.qn{display:flex;justify-content:space-between;font:800 15px/1 var(--disp);font-stretch:80%}
.chev{width:9px;height:9px;border-top:2px solid var(--muted);border-right:2px solid var(--muted);transform:rotate(45deg);justify-self:end}
.manual summary{font-weight:600;font-size:15px;padding:2px 0}
.pick{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
label.f{display:flex;flex-direction:column;gap:5px;font-size:13px;color:var(--muted)}
select,input.in{border:1.5px solid var(--line);background:var(--paper);border-radius:10px;padding:10px 11px;font-size:16px;width:100%;min-width:0}
details>summary{cursor:pointer;list-style:none;display:flex;justify-content:space-between;align-items:center;gap:10px}
details>summary::-webkit-details-marker{display:none}
details>summary::after{content:"";width:8px;height:8px;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg);transition:transform .15s;flex:none;margin-right:3px}
details[open]>summary::after{transform:rotate(-135deg)}

/* ---- destacados ---- */
.fp{display:block;width:100%;text-align:left;background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:14px;margin-top:10px}
.fp .ft{display:flex;justify-content:space-between;gap:8px;font-size:13px;color:var(--muted)}
.fp .fm{display:block;font:800 21px/1.15 var(--disp);font-stretch:75%;margin:4px 0 8px}
.fp .fo{font:600 13px var(--body);color:var(--muted);margin-left:10px}
.tl{display:grid;grid-template-columns:minmax(0,92px) 1fr 38px;gap:10px;align-items:center;font-size:14px;margin-top:5px}
.tl .who{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}
.tl b{text-align:right;font:800 16px/1 var(--disp);font-stretch:80%}
.sq{display:grid;grid-template-columns:repeat(10,1fr);gap:3px}
.sq i{height:12px;border-radius:3px;background:var(--line)}
.sq i.y{background:var(--good)} .sq i.n{background:var(--bad)}

/* ---- detalle del partido ---- */
.back{display:inline-flex;align-items:center;gap:8px;border:0;background:transparent;font-weight:700;font-size:16px;padding:8px 4px 8px 0;margin-top:6px}
.back i{width:10px;height:10px;border-left:2.5px solid currentColor;border-bottom:2.5px solid currentColor;transform:rotate(45deg);margin-left:4px}
.mhead{background:var(--pitch);border:1px solid #1d2b25;color:var(--chalk);border-radius:20px;padding:18px 18px 20px;margin-top:6px;position:relative;overflow:hidden}
.mhead::before{content:"";position:absolute;left:50%;top:0;bottom:0;border-left:1.5px solid rgba(244,247,242,.07)}
.mhead::after{content:"";position:absolute;left:50%;top:50%;width:150px;height:150px;margin:-75px 0 0 -75px;border:1.5px solid rgba(244,247,242,.07);border-radius:50%}
.mhead>*{position:relative;z-index:1}
.mmeta{font-size:14px;color:var(--chalk2)}
.mteams{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px;margin-top:10px}
.mt{min-width:0}
.mt.away{text-align:right}
.mn{font:800 34px/1.02 var(--disp);font-stretch:68%;overflow-wrap:break-word}
.mrole{font-size:13px;color:var(--chalk2);margin-top:5px}
.mform{display:flex;gap:4px;margin-top:8px}
.mt.away .mform{justify-content:flex-end}
.mform i{font-style:normal;width:22px;height:22px;border-radius:6px;display:grid;place-items:center;font:700 12px/1 var(--body);color:#fff}
.mform i.G{background:#1f9d6e} .mform i.E{background:#6f8b7e} .mform i.P{background:#cf4a56}
.board{margin-top:20px}
.stack{display:flex;height:16px;border-radius:99px;overflow:hidden;gap:3px}
.stack span{display:block;min-width:4px;transition:flex-grow .35s}
#sH{background:var(--chalk)} #sD{background:#4c8570} #sA{background:#f2b632}
.ticks{position:relative;height:0}
.tick{position:absolute;top:-22px;height:28px;border-left:2px dashed #fff;opacity:.9}
.pnum{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:14px}
.pnum>div{display:flex;flex-direction:column;min-width:0}
.pnum>div:nth-child(2){text-align:center;align-items:center}
.pnum>div:nth-child(3){text-align:right;align-items:flex-end}
.pnum small{font-size:13px;color:var(--chalk2);line-height:1.2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100%}
.pnum b{font:800 46px/1.05 var(--disp);font-stretch:72%}
.pnum .pl{display:flex;align-items:center;gap:6px}
.mkn{font-size:12.5px;color:var(--chalk2);margin-top:12px;line-height:1.4}

.tabs{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0 10px;margin-top:2px}
.tabs button{border:0;background:transparent;border-radius:12px;padding:10px 2px;font-weight:700;font-size:14px;color:var(--muted);white-space:nowrap}
.tabs button[aria-selected="true"]{background:var(--paper);color:var(--ink);box-shadow:inset 0 0 0 1.5px var(--ink)}
.pane>.box:first-child{margin-top:4px}

.tiles{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
.tile{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:12px 14px}
.tile b{display:block;font:800 27px/1.05 var(--disp);font-stretch:72%;white-space:nowrap}
.tile span{display:block;font-size:14px;font-weight:600;margin-top:2px}
.tile small{display:block;font-size:12.5px;color:var(--muted);line-height:1.35;margin-top:3px}
.oddsin{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.oddsin.two{grid-template-columns:1fr 1fr;margin-top:10px}
.oddsin input{text-align:center;font:800 22px/1 var(--disp);font-stretch:80%;padding:12px 6px}
.how p{font-size:14.5px;color:var(--muted);margin-top:8px}
.how summary{font-weight:700;font-size:15px;padding:4px 0}
.adj{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.none{font-size:14px;color:var(--muted)}

/* ---- historial ---- */
.seg{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;background:var(--soft);border:1px solid var(--line);border-radius:14px;padding:4px}
.seg button{border:0;background:transparent;border-radius:10px;padding:10px 2px;font-weight:700;font-size:13px;white-space:nowrap}
.seg button[aria-pressed="true"]{background:var(--sel);color:var(--on-sel)}
.seg.two{grid-template-columns:1fr 1fr;margin-top:12px}
.sc-note{margin-top:8px}
.tcard{margin-top:12px}
.tcard>h3{display:flex;flex-direction:column;font:800 24px/1.1 var(--disp);font-stretch:72%}
.tcard>h3 .role{font:500 13px var(--body);color:var(--muted);margin-top:2px}
.sumg{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0 4px}
.sumg>div{background:var(--soft);border-radius:12px;padding:9px 11px}
.sumg b{display:block;font:800 20px/1.1 var(--disp);font-stretch:75%;white-space:nowrap}
.sumg span{display:block;font-size:12.5px;color:var(--muted);line-height:1.3;margin-top:2px}
.hrow{display:grid;grid-template-columns:28px 1fr;gap:12px;padding:11px 0;border-top:1px solid var(--line)}
.rb{width:28px;height:28px;border-radius:8px;display:grid;place-items:center;font:800 14px/1 var(--body);color:#fff;margin-top:1px}
.rb.G{background:var(--good)} .rb.E{background:#6f8b7e} .rb.P{background:var(--bad)}
.hline{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);gap:8px;align-items:baseline;font-size:15px}
.hline span:first-child{text-align:right}
.hline span:last-child{text-align:left}
.hline b{font-weight:800}
.hsc{font:800 19px/1 var(--disp);font-stretch:80%;padding:0 2px}
.hmeta{font-size:13px;color:var(--muted);margin-top:3px;text-align:center}
.hchips{margin-top:1px;color:var(--ink);opacity:.72}
.hgrid .tcard{display:none}
.hgrid.show-h .tcard[data-t="h"],.hgrid.show-a .tcard[data-t="a"]{display:block}
.h2h .hrow:first-of-type{border-top:0}

/* ---- rachas ---- */
.tcat{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap;margin:18px 0 0}
.tcat h4{font:800 22px/1.1 var(--disp);font-stretch:72%}
.tcat span{font-size:13px;color:var(--muted)}
.trow{border:1px solid var(--line);border-radius:14px;padding:12px;margin-top:8px}
.thead{font-weight:700;font-size:16px;margin-bottom:2px}
.tcmp{font-size:13px;color:var(--muted);line-height:1.4;margin-top:8px}

/* ---- estadísticas ---- */
.seg.three{grid-template-columns:repeat(3,1fr)}
.box.ctl{display:flex;flex-direction:column;gap:8px}
.box.ctl .seg{margin:0}
.lines{margin:0 -16px!important;padding:2px 16px!important}
.lines .chip{padding:8px 16px;min-width:56px;text-align:center}
.cc{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:14px;margin-top:12px}
.cch{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.cch b{font:800 22px/1.1 var(--disp);font-stretch:72%}
.cch span{font-size:13px;color:var(--muted);text-align:right}
.ccs{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-top:8px}
.ccs b{font:800 17px/1 var(--disp);font-stretch:80%;color:var(--ink);margin-left:3px}
.g{color:var(--good)!important} .m{color:var(--vis)!important} .b{color:var(--bad)!important}
.chart{width:100%;height:auto;display:block;margin-top:10px}
.chart .ax{fill:var(--muted);font:10px var(--body)} .chart .dd{font-size:9px}
.chart .vl{fill:var(--ink);font:700 10px var(--body)}
.chart .gl{stroke:var(--line);stroke-width:1}
.chart .bar.y{fill:var(--good)} .chart .bar.n{fill:var(--bad)}
.chart .ln{stroke:var(--vis);stroke-width:1.8;stroke-dasharray:5 4}
.sq.big{margin-top:12px} .sq.big i{height:26px;border-radius:5px}
.rl{display:flex;gap:4px;flex-wrap:wrap;margin-top:12px}
.rl i{font-style:normal;width:26px;height:26px;border-radius:7px;display:grid;place-items:center;font:800 13px/1 var(--body);color:#fff}
.rl i.G{background:#1f9d6e} .rl i.E{background:#6f8b7e} .rl i.P{background:#cf4a56}
.ev .evr{display:grid;grid-template-columns:48px 1fr auto;gap:10px;align-items:baseline;padding:11px 0;border-top:1px solid var(--line);font-size:15px}
.ev h3+.sub+.evr,.ev h3+.evr{border-top:0}
.evc{font:800 18px/1 var(--disp);font-stretch:80%}
.evt{color:var(--ink);line-height:1.35;text-align:left}
.ev,.ev .sub,.ev h3{text-align:left}
.evp{font:800 19px/1 var(--disp);font-stretch:80%}
.evp.g{color:var(--good)} .evp.m{color:var(--vis)} .evp.b{color:var(--bad)}

/* ---- mercados ---- */
.grp{border-top:1px solid var(--line)}
.grp:first-of-type{border-top:0}
.grp>summary{font:800 20px/1.2 var(--disp);font-stretch:75%;padding:14px 0}
.grp .cnt{margin-left:auto;font:600 13px var(--body);color:var(--muted)}
.subh{font-size:13px;font-weight:700;color:var(--muted);padding:6px 0 0}
.mrow{padding:12px 0;border-top:1px solid var(--line)}
.mrow:first-of-type{border-top:0}
.mtop{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.ml{font-weight:600;font-size:16px}
.mp{font:800 24px/1 var(--disp);font-stretch:75%}
.pbar{height:6px;border-radius:99px;background:var(--soft);margin:7px 0 9px;overflow:hidden}
.pbar i{display:block;height:100%;background:var(--loc);border-radius:99px}
.mbot{display:grid;grid-template-columns:1fr 92px 64px auto;gap:8px;align-items:center}
.mf{font-size:13px;color:var(--muted)}
.mbot input{border:1.5px solid var(--line);border-radius:10px;padding:8px;font-size:16px;text-align:center;width:100%;background:var(--paper)}
.ev{font:800 15px/1 var(--disp);font-stretch:80%;text-align:center}
.ev.pos{color:var(--good)} .ev.neg{color:var(--bad)}
.mini{border:1.5px solid var(--ink);background:transparent;border-radius:10px;padding:8px 12px;font-size:14px;font-weight:700;white-space:nowrap}
.mini:disabled{opacity:.3;border-color:var(--line)}
.frow{margin-top:6px}
.flag{display:block;background:var(--warn-bg);color:var(--warn);border-radius:8px;padding:6px 9px;font-size:13px;line-height:1.35}
.mhelp{display:grid;grid-template-columns:1fr 92px 64px auto;gap:8px;font-size:12px;color:var(--muted);padding:2px 0 0}

/* ---- bitácora ---- */
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat{background:var(--soft);border-radius:12px;padding:10px 12px}
.stat .lbl{display:block;font-size:13px;color:var(--muted)}
.stat b{font:800 26px/1.1 var(--disp);font-stretch:75%}
.stat.p b{color:var(--good)} .stat.n b{color:var(--bad)}
.entry{padding:12px 0;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:8px}
.entry .m{font-weight:700;font-size:16px}
.entry .d{font-size:13px;color:var(--muted);line-height:1.4}
.entry .act{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.pill{font-size:13px;font-weight:700;border-radius:99px;padding:4px 11px;background:var(--soft)}
.pill.won{background:var(--good-bg);color:var(--good)} .pill.lost{background:var(--bad-bg);color:var(--bad)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
.btn{border:0;background:var(--sel);color:var(--on-sel);border-radius:12px;padding:11px 16px;font-weight:700;font-size:15px}
.btn.ghost{background:transparent;color:var(--ink);border:1.5px solid var(--line)}

/* ---- navegación inferior ---- */
.nav{position:fixed;left:0;right:0;bottom:0;z-index:20;background:var(--paper);border-top:1px solid var(--line);padding:6px 12px calc(6px + env(safe-area-inset-bottom))}
.nav .in{max-width:720px;margin:0 auto;display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.nav button{display:flex;flex-direction:column;align-items:center;gap:3px;border:0;background:transparent;border-radius:12px;padding:7px 4px;font-size:13px;font-weight:700;color:var(--muted)}
.nav button[aria-current="page"]{color:var(--ink);background:var(--soft)}
.nav svg{width:24px;height:24px}

@media (min-width:900px){
  .hgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .hgrid .tcard{display:block}
  .seg.two{display:none}
}
@media (max-width:380px){
  .mc{grid-template-columns:44px minmax(0,1fr) 96px 10px;gap:8px;padding:12px}
  .mn{font-size:29px} .pnum b{font-size:40px}
  .mbot{grid-template-columns:1fr 84px 58px auto}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head><body>
<div class="app">
  <header class="bar">
    <div class="brand"><svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true"><rect width="30" height="30" rx="8" fill="#0d3a2c"/><path d="M15 5v20" stroke="#f4f7f2" stroke-width="1.8"/><circle cx="15" cy="15" r="5.2" fill="none" stroke="#f4f7f2" stroke-width="1.8"/><circle cx="15" cy="15" r="1.8" fill="#f2b632"/></svg>Picks Fresh</div>
    <div class="upd" id="upd"></div>
  </header>

  <div class="ctx" id="ctx">
    <div class="row-scroll" id="leagueTabs" role="group" aria-label="Liga"></div>
    <div class="row-scroll" id="dayChips" role="group" aria-label="Día"></div>
  </div>

  <main>
    <!-- PARTIDOS -->
    <section id="vList" aria-label="Partidos">
      <div class="vh"><h2>Partidos</h2><span class="sub">Hora de Honduras</span></div>
      <div class="legend-q" aria-hidden="false"><span><i style="background:var(--loc)"></i>Gana el local</span><span><i style="background:var(--draw)"></i>Empate</span><span><i style="background:var(--vis)"></i>Gana el visitante</span></div>
      <div class="cards" id="fxList"></div>
      <details class="box manual" id="manual">
        <summary>¿No ves tu partido? Elige los equipos</summary>
        <div class="pick">
          <label class="f">Local<select id="home"></select></label>
          <label class="f">Visitante<select id="away"></select></label>
        </div>
        <div class="row"><button class="btn" id="goManual" type="button">Ver análisis</button></div>
        <div class="sub" id="manualMsg" style="margin-top:8px"></div>
      </details>
      <p class="foot" id="fxNote"></p>
      <p class="foot">Picks Fresh usa estadísticas de equipos (goles, córners, tarjetas, tiros) de las últimas temporadas. No considera alineaciones, lesiones ni árbitro. Apuesta solo lo que puedas perder y no intentes recuperar pérdidas.</p>
    </section>

    <!-- DESTACADOS -->
    <section id="vFeat" aria-label="Picks destacados" hidden>
      <div class="vh"><h2>Picks destacados</h2></div>
      <div class="sub">Mercados donde los dos equipos cumplieron en 7 o más de sus últimos 10 partidos. Cambia la liga o el día arriba.</div>
      <div id="feat"></div>
      <div class="callout"><b>Una racha no es una probabilidad.</b> En 4.725 partidos pasados, cuando los dos equipos traían 8 o más de 10 en "más de 2.5 goles", se cumplió 69% de las veces, no 80–90%, y las casas ya daban 65%. En córners (más de 9.5) fue 68% contra 52% de base, y en tarjetas (más de 3.5) 74% contra 61%.</div>
    </section>

    <!-- BITÁCORA -->
    <section id="vLog" aria-label="Bitácora de apuestas" hidden>
      <div class="vh"><h2>Bitácora</h2></div>
      <div class="sub">Tus apuestas anotadas, para comparar lo que dijo el modelo contra lo que pasó de verdad.</div>
      <div class="box"><div class="stats" id="stats"></div></div>
      <div class="box" style="padding-top:6px"><div class="log" id="log"></div></div>
      <div class="row"><button class="btn ghost" id="copy" type="button">Copiar bitácora</button><span class="sub" id="copyMsg"></span></div>
      <p class="foot">La bitácora se guarda solo en este navegador. Si borras los datos del sitio se pierde, así que copia el texto de vez en cuando.</p>
    </section>

    <!-- DETALLE DEL PARTIDO -->
    <section id="vMatch" aria-label="Detalle del partido" hidden>
      <button class="back" id="back" type="button"><i></i>Partidos</button>
      <div class="mhead">
        <div class="mmeta" id="hMeta"></div>
        <div class="mteams">
          <div class="mt"><div class="mn" id="hn"></div><div class="mrole">Local</div><div class="mform" id="hf"></div></div>
          <div class="mt away"><div class="mn" id="an"></div><div class="mrole">Visitante</div><div class="mform" id="af"></div></div>
        </div>
        <div class="board">
          <div class="stack" role="img" aria-label="Probabilidad de victoria local, empate y victoria visitante"><span id="sH"></span><span id="sD"></span><span id="sA"></span></div>
          <div class="ticks" id="ticks"></div>
          <div class="pnum">
            <div><small id="lHn"></small><b id="lH"></b></div>
            <div><small>Empate</small><b id="lD"></b></div>
            <div><small id="lAn"></small><b id="lA"></b></div>
          </div>
          <div class="mkn" id="mkNote"></div>
        </div>
      </div>

      <div class="tabs" id="mTabs" role="tablist">
        <button type="button" role="tab" data-t="sum" aria-selected="true">Resumen</button>
        <button type="button" role="tab" data-t="hist" aria-selected="false">Historial</button>
        <button type="button" role="tab" data-t="trend" aria-selected="false">Estadísticas</button>
        <button type="button" role="tab" data-t="mk" aria-selected="false">Apostar</button>
      </div>

      <div class="pane" id="pSum">
        <div class="box">
          <h3>Lo que dice el modelo</h3>
          <div class="sub" id="readout"></div>
          <div class="tiles" id="tiles"></div>
        </div>
        <div class="box">
          <h3>Cuotas de tu casa</h3>
          <div class="sub">Escribe las cuotas que te da tu casa de apuestas y recalculamos las probabilidades quitándoles el margen de la casa.</div>
          <div class="oddsin">
            <label class="f"><span id="ol1">Gana el local</span><input class="in" inputmode="decimal" data-k="1" id="ou1" placeholder="Cuota"></label>
            <label class="f"><span>Empate</span><input class="in" inputmode="decimal" data-k="X" id="ouX" placeholder="Cuota"></label>
            <label class="f"><span id="ol2">Gana el visitante</span><input class="in" inputmode="decimal" data-k="2" id="ou2" placeholder="Cuota"></label>
          </div>
          <div class="oddsin two">
            <label class="f"><span>Más de 2.5 goles</span><input class="in" inputmode="decimal" data-k="o:goals:2.5" id="ouO" placeholder="Cuota (opcional)"></label>
            <label class="f"><span>Menos de 2.5 goles</span><input class="in" inputmode="decimal" data-k="u:goals:2.5" id="ouU" placeholder="Cuota (opcional)"></label>
          </div>
          <div class="sub" id="houseTxt" style="margin-top:12px"></div>
        </div>
        <div class="box how">
          <details id="how"><summary>Cómo se calculó</summary><div id="howTxt"></div></details>
        </div>
        <div class="box how">
          <details><summary>Bajas y ajuste manual</summary>
            <div id="inj" style="margin-top:10px"></div>
            <p>Si hay bajas importantes, puedes bajar el ataque del equipo. Es una opinión tuya y no está validada.</p>
            <div class="adj">
              <label class="f">Local<select id="adjH"><option value="0">Sin ajuste</option><option value="1">1 baja clave (−7% ataque)</option><option value="2">2 bajas clave (−14%)</option><option value="3">3 bajas clave (−21%)</option></select></label>
              <label class="f">Visitante<select id="adjA"><option value="0">Sin ajuste</option><option value="1">1 baja clave (−7% ataque)</option><option value="2">2 bajas clave (−14%)</option><option value="3">3 bajas clave (−21%)</option></select></label>
            </div>
          </details>
        </div>
      </div>

      <div class="pane" id="pHist" hidden>
        <div class="box">
          <h3>Últimos partidos</h3>
          <div class="sub">Elige qué partidos mostrar. G = ganó, E = empató, P = perdió.</div>
          <div id="scopeH"></div>
          <div class="seg two" id="teamSw"></div>
        </div>
        <div class="hgrid show-h" id="hist"></div>
        <div id="h2h"></div>
      </div>

      <div class="pane" id="pTrend" hidden>
        <div class="box">
          <h3>Estadísticas</h3>
          <div class="sub">Cada barra es un partido. Elige qué partidos mirar y qué estadística quieres ver.</div>
          <div id="scopeT"></div>
          <div class="row-scroll" id="metricT" role="group" aria-label="Estadística" style="margin-top:12px"></div>
        </div>
        <div id="statBody"></div>
        <div class="box how"><details><summary>Qué tan confiable es una racha</summary>
          <p>Una racha cuenta lo que pasó, no lo que va a pasar. En 4.725 partidos pasados de Bundesliga, Premier y La Liga, cuando los dos equipos traían 8 o más de 10 en "más de 2.5 goles", se cumplió el 69% de las veces, no el 80–90%. Las casas ya daban 65% en esos partidos.</p>
          <p>En córners (más de 9.5) fue 68% contra 52% de base, y en tarjetas (más de 3.5) fue 74% contra 61% de base. Ahí la racha sí aportó algo, pero no tenemos cuotas de córners ni tarjetas para saber si pagan bien.</p>
          <p>Las tarjetas son amarillas más rojas. Algunas casas cuentan distinto (por ejemplo, la roja como 2), así que revisa cómo cuenta la tuya.</p>
        </details></div>
      </div>

      <div class="pane" id="pMk" hidden>
        <div class="box">
          <h3>Mercados y cuotas</h3>
          <div class="callout" style="margin-top:8px"><b>El modelo no supera a las casas de apuestas.</b> Úsalo como referencia. Una ganancia esperada positiva no es una señal de compra.</div>
          <div class="sub" style="margin-top:10px">Escribe la cuota de tu casa en una fila y toca Anotar para guardarla en la bitácora.</div>
          <div class="sub" id="oddsSrc"></div>
          <div id="mk" style="margin-top:6px"></div>
        </div>
        <div class="box how"><details><summary>Qué tan confiable es el modelo</summary>
          <p>Prueba con 2.300 partidos pasados de Bundesliga, Premier y La Liga: el modelo no superó a las casas de apuestas. Las apuestas donde el modelo veía "valor" perdieron entre 15% y 30%, y fue peor cuanto más grande era el supuesto valor.</p>
          <p>El modelo subestima a los favoritos fuertes: cuando las casas daban 70–80%, el modelo daba cerca de 69% y ganaron cerca de 81%. Córners y tiros salieron bien calibrados.</p>
          <p>Si el modelo y la cuota difieren por 8 puntos o más, casi siempre el error es del modelo.</p>
        </details></div>
      </div>
    </section>
  </main>
</div>

<nav class="nav" aria-label="Secciones"><div class="in">
  <button type="button" data-v="list" aria-current="page"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="3"/><path d="M12 5v14"/><circle cx="12" cy="12" r="3"/></svg>Partidos</button>
  <button type="button" data-v="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.6 5.6 6.1.7-4.5 4.2 1.2 6L12 16.4 6.6 19.5l1.2-6L3.3 9.3l6.1-.7z"/></svg>Destacados</button>
  <button type="button" data-v="log"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3h11a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6z"/><path d="M6 3v18M10 8h6M10 12h6"/></svg>Bitácora</button>
</div></nav>
<script>
const DATA = __PARAMS__;
const $ = id => document.getElementById(id);
const pct = x => (x*100).toFixed(0)+"%";
const f2 = x => x.toFixed(2);
let league = "bundesliga";
const KEY = "scoutpicks.log.v1";

/* ---------- matemática del modelo ---------- */
function pois(l, n){const a=[Math.exp(-l)];for(let k=1;k<=n;k++)a.push(a[k-1]*l/k);return a;}
function scoreMatrix(lh, la){
  const N=10, ph=pois(lh,N), pa=pois(la,N); let s=0; const m=[];
  for(let i=0;i<=N;i++){m.push([]);for(let j=0;j<=N;j++){const v=ph[i]*pa[j];m[i].push(v);s+=v;}}
  return m.map(r=>r.map(v=>v/s));
}
function lam(M, h, a){
  return [Math.exp(M.mu+M.home+M.att[h]-M.de[a]), Math.exp(M.mu+M.att[a]-M.de[h])];
}
function pOver(line, mu, alpha){            // binomial negativa: P(X > floor(line))
  const r=1/Math.max(alpha,1e-4), p=r/(r+mu), K=Math.floor(line);
  let pm=Math.pow(p,r), cdf=pm;
  for(let k=0;k<K;k++){pm=pm*(k+r)/(k+1)*(1-p);cdf+=pm;}
  return Math.max(0,Math.min(1,1-cdf));
}
const LINES = {
  corners:{n:"Córners",tot:[8.5,9.5,10.5,11.5],team:[3.5,4.5,5.5]},
  sot:{n:"Tiros a puerta",tot:[7.5,8.5,9.5,10.5],team:[3.5,4.5,5.5]},
  shots:{n:"Tiros totales",tot:[24.5,26.5,28.5,30.5],team:[11.5,13.5,15.5]}
};

/* ---------- Elo -> goles esperados (Champions) ---------- */
function eloLam(E,T){
  const score=s=>{const m=scoreMatrix((T+s)/2,(T-s)/2);let w=0,d=0;
    for(let i=0;i<m.length;i++)for(let j=0;j<m.length;j++){if(i>j)w+=m[i][j];else if(i===j)d+=m[i][j];}
    return w+0.5*d;};
  let lo=-T*0.95, hi=T*0.95;
  for(let k=0;k<40;k++){const mid=(lo+hi)/2; if(score(mid)<E) lo=mid; else hi=mid;}
  const s=(lo+hi)/2; return [(T+s)/2,(T-s)/2];
}

/* ---------- cálculo de un partido ---------- */
function compute(h, a, fh, fa, ml){
  const L = DATA.leagues[league], out = {rows:[], chips:[]};
  let l0;
  if(ml){ l0=ml; }
  else if(L.kind==="elo"){ const E=1/(1+Math.pow(10,-((L.elo[h]+L.hfa-L.elo[a])/400))); l0=eloLam(E,L.T); }
  else { l0=lam(L.models.goles,h,a); }
  const lh = l0[0]*fh, la = l0[1]*fa, m = scoreMatrix(lh,la);
  let pH=0,pD=0,pA=0,o15=0,o25=0,o35=0,bt=0;
  for(let i=0;i<m.length;i++)for(let j=0;j<m.length;j++){
    const v=m[i][j]; if(i>j)pH+=v; else if(i===j)pD+=v; else pA+=v;
    if(i+j>=2)o15+=v; if(i+j>=3)o25+=v; if(i+j>=4)o35+=v; if(i>=1&&j>=1)bt+=v;
  }
  if(ml){ bt=1/(1+Math.exp(-(Math.log(bt/(1-bt))+0.14))); }   // en pruebas, "ambos anotan" sale ~3 pts más alto de lo que da Poisson
  out.lh=lh; out.la=la; out.p=[pH,pD,pA]; out.o25=o25; out.bt=bt; out.fromMkt=!!ml;
  const R=(sec,label,p,key,fav)=>out.rows.push({sec,label,p,key,fav});
  R("Resultado",h+" gana",pH,"1",true); R("Resultado","Empate",pD,"X"); R("Resultado",a+" gana",pA,"2",true);
  R("Resultado","Doble oportunidad "+h+" o empate",pH+pD); R("Resultado","Doble oportunidad "+a+" o empate",pA+pD);
  R("Goles","Más de 1.5 goles",o15,"o:goals:1.5"); R("Goles","Menos de 1.5 goles",1-o15,"u:goals:1.5");
  R("Goles","Más de 2.5 goles",o25,"o:goals:2.5"); R("Goles","Menos de 2.5 goles",1-o25,"u:goals:2.5");
  R("Goles","Más de 3.5 goles",o35,"o:goals:3.5"); R("Goles","Menos de 3.5 goles",1-o35,"u:goals:3.5");
  R("Goles","Ambos anotan: sí",bt,"btts:y"); R("Goles","Ambos anotan: no",1-bt,"btts:n");
  for(const [k,c] of Object.entries(LINES)){
    const M=L.models[k]; if(!M) continue;
    const e0=lam(M,h,a), ch=e0[0]*fh, ca=e0[1]*fa, tot=ch+ca;
    out.chips.push([c.n, ch.toFixed(1)+" – "+ca.toFixed(1)]);
    for(const x of c.tot){const p=pOver(x,tot,M.at);
      R(c.n+" · total","Más de "+x,p,k==="corners"?"o:corners:"+x:null);
      R(c.n+" · total","Menos de "+x,1-p,k==="corners"?"u:corners:"+x:null);}
    for(const x of c.team){R(c.n+" · "+h,"Más de "+x,pOver(x,ch,M.ah));R(c.n+" · "+a,"Más de "+x,pOver(x,ca,M.aa));}
  }
  return out;
}

/* ---------- interfaz ---------- */
let cur = null;
const DOW=["dom","lun","mar","mié","jue","vie","sáb"], MON=["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"];
let day=null;
function dparts(iso){const p=iso.slice(0,10).split("-").map(Number);const d=new Date(p[0],p[1]-1,p[2]);return {dow:DOW[d.getDay()],d:p[2],mon:MON[p[1]-1]};}
function dtxt(iso){const p=dparts(iso);return p.dow+" "+p.d+" "+p.mon+", "+iso.slice(11,16);}
function fxOfLeague(){return (DATA.fixtures||[]).filter(f=>f.lg===league);}
function daysOfLeague(){const out=[];fxOfLeague().forEach(f=>{const k=f.date.slice(0,10);if(!out.includes(k))out.push(k);});return out;}
function fillTeams(){
  const L=DATA.leagues[league];
  for(const id of ["home","away"]){
    $(id).innerHTML=L.teams.map(t=>`<option>${esc(t)}</option>`).join("");
  }
  const days=daysOfLeague(); if(!days.includes(day)) day=days[0]||null;
  const first=fxOfLeague().find(f=>f.date.slice(0,10)===day);
  const pref=first?[first.h,first.a]:[L.teams[0],L.teams[1]];
  $("home").value=pref[0]; $("away").value=pref[1];
}
function formHtml(t){return (DATA.leagues[league].form[t]||"").split("").map(c=>`<i class="${c}">${c}</i>`).join("");}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function injHtml(h,a){
  const L=DATA.leagues[league];
  if(!L.inj) return `<div class="none">Lesionados: no se cargan automáticamente. Usa el ajuste manual de abajo si hay bajas importantes.</div>`;
  const one=t=>{const l=L.inj[t]||[];
    return l.length?`<div><b>${esc(t)}</b>: ${l.map(x=>esc(x[0])+(x[1]?" ("+esc(x[1])+")":"")).join(", ")}</div>`
                   :`<div class="none"><b>${esc(t)}</b>: sin bajas reportadas</div>`;};
  return one(h)+one(a);
}
/* ---------- recalcular con las cuotas que escribe el usuario ---------- */
let userOdds={}, userMatch="";
function nm(f,x0,iter){
  let P=[x0,[x0[0]+0.4,x0[1]],[x0[0],x0[1]+0.4]].map(x=>({x:x,v:f(x)}));
  for(let k=0;k<iter;k++){
    P.sort((a,b)=>a.v-b.v);
    const c=[(P[0].x[0]+P[1].x[0])/2,(P[0].x[1]+P[1].x[1])/2], w=P[2];
    const r=[2*c[0]-w.x[0],2*c[1]-w.x[1]], fr=f(r);
    if(fr<P[0].v){const e=[3*c[0]-2*w.x[0],3*c[1]-2*w.x[1]],fe=f(e);P[2]=fe<fr?{x:e,v:fe}:{x:r,v:fr};}
    else if(fr<P[1].v){P[2]={x:r,v:fr};}
    else{const t=[(c[0]+w.x[0])/2,(c[1]+w.x[1])/2],ft=f(t);
      if(ft<w.v)P[2]={x:t,v:ft};
      else for(const i of [1,2]){const x=[(P[0].x[0]+P[i].x[0])/2,(P[0].x[1]+P[i].x[1])/2];P[i]={x:x,v:f(x)};}}
  }
  P.sort((a,b)=>a.v-b.v); return P[0];
}
function fairFrom(o){
  if(!(o["1"]>1&&o["X"]>1&&o["2"]>1))return null;
  const q=[1/o["1"],1/o["X"],1/o["2"]],t=q[0]+q[1]+q[2];
  return {p:q.map(x=>x/t),margin:t-1};
}
function mlFromOdds(o){
  const fr=fairFrom(o); if(!fr)return null;
  let po=null; if(o["o:goals:2.5"]>1&&o["u:goals:2.5"]>1){const a=1/o["o:goals:2.5"],b=1/o["u:goals:2.5"];po=a/(a+b);}
  const f=x=>{const m=scoreMatrix(Math.exp(x[0]),Math.exp(x[1]));let w=0,d=0,l=0,ov=0;
    for(let i=0;i<m.length;i++)for(let j=0;j<m.length;j++){const v=m[i][j];if(i>j)w+=v;else if(i===j)d+=v;else l+=v;if(i+j>=3)ov+=v;}
    let e=(w-fr.p[0])**2+(d-fr.p[1])**2+(l-fr.p[2])**2; if(po!==null)e+=(ov-po)**2; return e;};
  const best=nm(f,[Math.log(1.4),Math.log(1.1)],160);
  const lh=Math.exp(best.x[0]), la=Math.exp(best.x[1]);
  return (best.v<2e-3&&lh>0.05&&lh<7&&la>0.05&&la<7)?[lh,la]:null;
}
function rowKey(r){return r.key||(r.sec+"|"+r.label);}
const MONTHS=["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"];
function ymd(n){const y=Math.floor(n/10000),m=Math.floor(n/100)%100,d=n%100;return d+" "+MONTHS[m-1]+" "+String(y).slice(2);}


/* ---------- historial: partidos de liga y de Champions ---------- */
function scTxt(){
  const n=win;
  return {all:"Últimos "+n+" partidos de liga de cada equipo, jugados de local y de visita.",
    home:"Últimos "+n+" partidos de liga en los que cada equipo jugó de local.",
    away:"Últimos "+n+" partidos de liga en los que cada equipo jugó de visita.",
    ucl:"Últimos "+n+" partidos de Champions League de cada equipo, con las rondas previas. Hay datos de 2024/25, 2025/26 y la jornada 1 de 2026/27."}[scope];
}
let scope="all", view="list", prevView="list", tab="sum", histTeam="h", uclSet=null, win=10, met="goals", who="tot", dir="o", lineSel={};
function uclNames(){if(!uclSet){uclSet=new Set();(DATA.uclh||[]).forEach(r=>{uclSet.add(r[1]);uclSet.add(r[2]);});}return uclSet;}
function uclName(team){
  if(league==="ucl")return team;
  const c=((DATA.uclalias||{})[league]||{})[team]||team;
  return uclNames().has(c)?c:null;
}
function resolveHist(team){
  const L=DATA.leagues[league];
  if(L.kind==="elo"){const m=(L.map||{})[team]; return m?{key:m[0],name:m[1],local:true}:null;}
  return L.hm?{key:league,name:team,local:false}:null;
}
function domMatches(r,n,sc){
  const L=DATA.leagues[r.key]; if(!L||!L.hm)return [];
  const i=L.hteams.indexOf(r.name); if(i<0)return [];
  const out=[];
  for(let k=L.hm.length-1;k>=0&&out.length<n;k--){const m=L.hm[k], home=m[1]===i;
    if(!home&&m[2]!==i)continue;
    if(sc==="home"&&!home)continue; if(sc==="away"&&home)continue;
    out.push({d:m[0],h:L.hteams[m[1]],a:L.hteams[m[2]],hg:m[3],ag:m[4],hc:m[5],ac:m[6],kh:m[7]===undefined?-1:m[7],ka:m[8]===undefined?-1:m[8],home:home});}
  return out;
}
function uclMatches(c,n){
  const out=[], M=DATA.uclh||[];
  for(let k=M.length-1;k>=0&&out.length<n;k--){const m=M[k]; if(m[1]!==c&&m[2]!==c)continue;
    out.push({d:m[0],h:m[1],a:m[2],hg:m[3],ag:m[4],hc:-1,ac:-1,kh:-1,ka:-1,home:m[1]===c,stage:m[5],note:m[6]});}
  return out;
}
function seriesFor(team,sc,n){
  n=n||win;
  if(sc==="ucl"){
    const c=uclName(team), ms=c?uclMatches(c,n):[];
    return ms.length?{ok:true,name:c,ms:ms}:{ok:false,msg:"No jugó la Champions League (ni sus rondas previas) en las últimas dos temporadas."};
  }
  const res=resolveHist(team);
  if(!res)return {ok:false,msg:"No tenemos su historial de liga. Solo hay datos de Bundesliga, Premier League y La Liga."};
  const ms=domMatches(res,n,sc);
  return ms.length?{ok:true,name:res.name,ms:ms,local:!!res.local}:{ok:false,msg:"Sin partidos en los datos."};
}
function sumHtml(ms){
  const n=ms.length; let gf=0,gc=0,w=0,d=0,l=0,bt=0,o=0,cf=0,cc=0,nc=0,kf=0,kc=0,nk=0;
  ms.forEach(m=>{const f=m.home?m.hg:m.ag,c=m.home?m.ag:m.hg;gf+=f;gc+=c;if(f>c)w++;else if(f===c)d++;else l++;if(f>0&&c>0)bt++;if(f+c>=3)o++;
    if(m.hc>=0&&m.ac>=0){cf+=m.home?m.hc:m.ac;cc+=m.home?m.ac:m.hc;nc++;}
    if(m.kh>=0&&m.ka>=0){kf+=m.home?m.kh:m.ka;kc+=m.home?m.ka:m.kh;nk++;}});
  const t=(v,lab)=>`<div><b>${v}</b><span>${lab}</span></div>`;
  return `<div class="sumg">${t(w+"G "+d+"E "+l+"P","Ganó, empató y perdió")}${t((gf/n).toFixed(1)+" – "+(gc/n).toFixed(1),"Goles a favor y en contra por partido")}${t(bt+" de "+n,"Partidos donde los dos anotaron")}${t(o+" de "+n,"Partidos con más de 2.5 goles")}${nc?t((cf/nc).toFixed(1)+" – "+(cc/nc).toFixed(1),"Córners a favor y en contra por partido"):""}${nk?t((kf/nk).toFixed(1)+" – "+(kc/nk).toFixed(1),"Tarjetas del equipo y del rival por partido"):""}</div>`;
}
function matchRow(m,me){
  const f=m.home?m.hg:m.ag, c=m.home?m.ag:m.hg, res=f>c?"G":f===c?"E":"P";
  const nx=t=>t===me?`<b>${esc(t)}</b>`:esc(t);
  const chips=[]; if(m.stage)chips.push(m.stage); if(m.note)chips.push(m.note);
  if(m.hc>=0&&m.ac>=0)chips.push("Córners "+m.hc+"–"+m.ac); if(m.kh>=0&&m.ka>=0)chips.push("Tarjetas "+m.kh+"–"+m.ka);
  return `<div class="hrow"><span class="rb ${res}" title="${res==="G"?"Ganó":res==="E"?"Empató":"Perdió"}">${res}</span><div><div class="hline"><span>${nx(m.h)}</span><span class="hsc">${m.hg}–${m.ag}</span><span>${nx(m.a)}</span></div><div class="hmeta">${ymd(m.d)} · ${m.home?"De local":"De visita"}</div>${chips.length?`<div class="hmeta hchips">${chips.map(esc).join(" · ")}</div>`:""}</div></div>`;
}
function scopeHtml(){
  const b=(attr,cur,opts)=>opts.map(([k,t])=>`<button type="button" data-${attr}="${k}" aria-pressed="${String(k)===String(cur)}">${t}</button>`).join("");
  return `<div class="seg" role="group" aria-label="Qué partidos mostrar" style="margin-top:12px">${b("s",scope,[["all","Todos"],["home","De local"],["away","De visita"],["ucl","Champions"]])}</div>
  <div class="seg three" role="group" aria-label="Cuántos partidos" style="margin-top:8px">${b("w",win,[[5,"Últimos 5"],[10,"Últimos 10"],[20,"Últimos 20"]])}</div>
  <div class="sub sc-note">${esc(scTxt())}</div>`;
}
function h2hHtml(h,a){
  let out=[], me=h;
  if(scope==="ucl"){
    const ch=uclName(h), ca=uclName(a); if(!ch||!ca)return ""; me=ch;
    const M=DATA.uclh||[];
    for(let k=M.length-1;k>=0&&out.length<6;k--){const m=M[k];
      if((m[1]===ch&&m[2]===ca)||(m[1]===ca&&m[2]===ch))out.push({d:m[0],h:m[1],a:m[2],hg:m[3],ag:m[4],hc:-1,ac:-1,kh:-1,ka:-1,home:m[1]===ch,stage:m[5],note:m[6]});}
  }else{
    const rh=resolveHist(h), ra=resolveHist(a); if(!rh||!ra||rh.key!==ra.key)return ""; me=rh.name;
    const L=DATA.leagues[rh.key], i=L.hteams.indexOf(rh.name), j=L.hteams.indexOf(ra.name);
    for(let k=L.hm.length-1;k>=0&&out.length<6;k--){const m=L.hm[k];
      if((m[1]===i&&m[2]===j)||(m[1]===j&&m[2]===i))out.push({d:m[0],h:L.hteams[m[1]],a:L.hteams[m[2]],hg:m[3],ag:m[4],hc:m[5],ac:m[6],kh:m[7]===undefined?-1:m[7],ka:m[8]===undefined?-1:m[8],home:m[1]===i});}
  }
  return `<div class="box h2h"><h3>Enfrentamientos directos</h3><div class="sub">Solo informativo: el modelo no usa estos resultados. G, E y P se cuentan desde ${esc(me)}.</div>${out.length?`<div>${out.map(m=>matchRow(m,me)).join("")}</div>`:`<div class="empty">No se han enfrentado en los datos que tenemos.</div>`}</div>`;
}
function renderHist(h,a){
  $("scopeH").innerHTML=scopeHtml();
  $("teamSw").innerHTML=[["h",h],["a",a]].map(([k,t])=>`<button type="button" data-ht="${k}" aria-pressed="${k===histTeam}">${esc(t)}</button>`).join("");
  const col=(team,role,s)=>`<section class="box tcard" data-t="${role}"><h3>${esc(team)}<span class="role">${role==="h"?"Local en este partido":"Visitante en este partido"}</span></h3>${
    s.ok?(s.local?`<div class="sub">Sus partidos de liga local (no incluye Champions).</div>`:"")+sumHtml(s.ms)+`<div>${s.ms.map(m=>matchRow(m,s.name)).join("")}</div>`
        :`<div class="empty">${esc(s.msg)}</div>`}</section>`;
  $("hist").className="hgrid show-"+histTeam;
  $("hist").innerHTML=col(h,"h",seriesFor(h,scope))+col(a,"a",seriesFor(a,scope));
  $("h2h").innerHTML=h2hHtml(h,a);
}

/* ---------- rachas ---------- */
const TM=[
  {cat:"Goles",label:"Más de 2.5 goles",f:m=>m.tot>=3,odds:"o:goals:2.5"},
  {cat:"Goles",label:"Menos de 2.5 goles",f:m=>m.tot<=2,odds:"u:goals:2.5"},
  {cat:"Goles",label:"Ambos anotan",f:m=>m.f>0&&m.c>0},
  {cat:"Goles",label:"No anotan los dos",f:m=>!(m.f>0&&m.c>0)},
  {cat:"Córners",label:"Más de 8.5 córners",need:"cor",ln:8.5,f:m=>m.cor>8.5},
  {cat:"Córners",label:"Más de 9.5 córners",need:"cor",ln:9.5,f:m=>m.cor>9.5},
  {cat:"Córners",label:"Más de 10.5 córners",need:"cor",ln:10.5,f:m=>m.cor>10.5},
  {cat:"Tarjetas",label:"Más de 3.5 tarjetas",need:"crd",ln:3.5,f:m=>m.crd>3.5},
  {cat:"Tarjetas",label:"Más de 4.5 tarjetas",need:"crd",ln:4.5,f:m=>m.crd>4.5},
  {cat:"Tarjetas",label:"Más de 5.5 tarjetas",need:"crd",ln:5.5,f:m=>m.crd>5.5}
];
const TCAT={Goles:"tot",Córners:"cor",Tarjetas:"crd"};
function toTrend(ms){
  return ms.slice().reverse().map(m=>{const f=m.home?m.hg:m.ag,c=m.home?m.ag:m.hg;
    return {f:f,c:c,tot:f+c,cor:(m.hc>=0&&m.ac>=0)?m.hc+m.ac:null,crd:(m.kh>=0&&m.ka>=0)?m.kh+m.ka:null};});
}
function trendHits(ms,def){
  const arr=ms.map(m=>(def.need&&m[def.need]===null)?null:!!def.f(m)), ok=arr.filter(x=>x!==null);
  return {arr:arr,n:ok.length,k:ok.filter(Boolean).length};
}
function sqHtml(t){
  const pad=10-t.arr.length;
  return `<div class="sq" role="img" aria-label="${t.k} de ${t.n}">${Array.from({length:10},(_,i)=>{const v=i<pad?null:t.arr[i-pad];return `<i class="${v===null?"":v?"y":"n"}"></i>`;}).join("")}</div>`;
}
function houseLine(def,fx,comb){
  const o=fx&&fx.odds&&def.odds?fx.odds[def.odds]:null; if(!(o>1))return "";
  const other=def.odds.startsWith("o:")?fx.odds["u:goals:2.5"]:def.odds.startsWith("u:")?fx.odds["o:goals:2.5"]:null;
  let p=1/o,tag="incluye margen"; if(other>1){p=(1/o)/(1/o+1/other);tag="sin margen";}
  return `Cuota ${f2(o)}: la casa da ${pct(p)} (${tag}). Las rachas juntas dan ${pct(comb)}, que no es una probabilidad.`;
}
/* ---------- estadísticas: gráfica por partido y evidencia ---------- */
const MET={
  goals:{n:"goles",tl:[0.5,1.5,2.5,3.5,4.5],td:2.5,el:[0.5,1.5,2.5],ed:0.5},
  cor:{n:"córners",need:"cor",tl:[7.5,8.5,9.5,10.5,11.5],td:9.5,el:[2.5,3.5,4.5,5.5,6.5],ed:4.5},
  crd:{n:"tarjetas",need:"crd",tl:[2.5,3.5,4.5,5.5],td:3.5,el:[0.5,1.5,2.5,3.5],ed:1.5}
};
const METCH=[["goals","Goles"],["cor","Córners"],["crd","Tarjetas"],["btts","Ambos anotan"],["res","Resultado"]];
function nrm(m){
  const f=m.home?m.hg:m.ag, c=m.home?m.ag:m.hg, hc=m.hc>=0&&m.ac>=0, hk=m.kh>=0&&m.ka>=0;
  return {f:f,c:c,tot:f+c,cor:hc?m.hc+m.ac:null,cf:hc?(m.home?m.hc:m.ac):null,
    crd:hk?m.kh+m.ka:null,kf:hk?(m.home?m.kh:m.ka):null,d:m.d,rival:m.home?m.a:m.h,home:m.home};
}
function vfn(k,w){
  if(k==="goals")return w==="tot"?(m=>m.tot):(m=>m.f);
  if(k==="cor")return w==="tot"?(m=>m.cor):(m=>m.cf);
  return w==="tot"?(m=>m.crd):(m=>m.kf);
}
function medianOf(a){const s=a.slice().sort((x,y)=>x-y), i=Math.floor(s.length/2);return s.length%2?s[i]:(s[i-1]+s[i])/2;}
function lineOf(){const c=MET[met], k=met+who; return lineSel[k]!==undefined?lineSel[k]:(who==="tot"?c.td:c.ed);}
function hitV(v,line){return v===null?null:(dir==="o"?v>line:v<line);}
function pcls(p){return p>=0.7?"g":p>=0.5?"m":"b";}
function scLbl(){return {all:"Todos los partidos",home:"De local",away:"De visita",ucl:"Champions"}[scope];}
function code3(t){return t.replace(/[^A-Za-zÀ-ÿ]/g,"").slice(0,3).toUpperCase();}
function dm(n){return String(n%100).padStart(2,"0")+"/"+String(Math.floor(n/100)%100).padStart(2,"0");}
function chartSvg(ms,vf,line){
  const n=ms.length, W=340, H=n<=10?196:176, pl=22, pr=4, pt=16, pb=n<=10?40:36;
  const vals=ms.map(vf), nums=vals.filter(v=>v!==null);
  const top=Math.max(2,Math.ceil(Math.max(line+0.5,...nums,1))), step=top>8?2:1;
  const iy=v=>H-pb-(v/top)*(H-pb-pt), bw=(W-pl-pr)/n, bd=Math.min(bw*0.64,26);
  let g="";
  for(let t=0;t<=top;t+=step)g+=`<line class="gl" x1="${pl}" x2="${W-pr}" y1="${iy(t).toFixed(1)}" y2="${iy(t).toFixed(1)}"/><text class="ax" x="${pl-5}" y="${(iy(t)+3).toFixed(1)}" text-anchor="end">${t}</text>`;
  ms.forEach((m,i)=>{
    const v=vals[i], cx=pl+bw*(i+0.5);
    if(v!==null){const h=Math.max(v/top*(H-pb-pt),3);
      g+=`<rect class="bar ${hitV(v,line)?"y":"n"}" x="${(cx-bd/2).toFixed(1)}" y="${(H-pb-h).toFixed(1)}" width="${bd.toFixed(1)}" height="${h.toFixed(1)}" rx="3"/>`;
      if(n<=10)g+=`<text class="vl" x="${cx.toFixed(1)}" y="${(H-pb-h-4).toFixed(1)}" text-anchor="middle">${v}</text>`;}
    else g+=`<text class="ax" x="${cx.toFixed(1)}" y="${H-pb-4}" text-anchor="middle">–</text>`;
    if(n<=10)g+=`<text class="ax" x="${cx.toFixed(1)}" y="${H-pb+14}" text-anchor="middle" style="font-weight:700">${esc(code3(m.rival))}</text><text class="ax dd" x="${cx.toFixed(1)}" y="${H-pb+26}" text-anchor="middle">${dm(m.d)}</text>`;
    else g+=`<text class="ax dd" transform="translate(${(cx+3).toFixed(1)} ${H-pb+6}) rotate(-90)" text-anchor="end">${dm(m.d)}</text>`;
  });
  g+=`<line class="ln" x1="${pl}" x2="${W-pr}" y1="${iy(line).toFixed(1)}" y2="${iy(line).toFixed(1)}"/>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Valor de cada partido, con la línea de ${line}">${g}</svg>`;
}
function ccHead(team,s){return `<div class="cch"><b>${esc(team)}</b><span>${s.ok?(s.ms.length>1?"Últimos "+s.ms.length:"Último partido")+" · "+scLbl():scLbl()}</span></div>`;}
function chartCard(team,s,vf,line){
  if(!s.ok)return `<div class="cc">${ccHead(team,s)}<div class="empty">${esc(s.msg)}</div></div>`;
  const ms=s.ms.slice().reverse().map(nrm), vs=ms.map(vf).filter(v=>v!==null);
  if(!vs.length)return `<div class="cc">${ccHead(team,s)}<div class="empty">No hay datos de esta estadística para este equipo en estos partidos.</div></div>`;
  const hits=vs.filter(v=>hitV(v,line)).length, avg=(vs.reduce((x,y)=>x+y,0)/vs.length).toFixed(1);
  return `<div class="cc">${ccHead(team,s)}<div class="ccs"><span>Tasa de acierto <b class="${pcls(hits/vs.length)}">${Math.round(100*hits/vs.length)}%</b></span><span>Promedio <b>${avg}</b></span><span>Mediana <b>${medianOf(vs)}</b></span></div>${chartSvg(ms,vf,line)}</div>`;
}
function sqGen(arr){return `<div class="sq big" style="grid-template-columns:repeat(${arr.length},1fr)">${arr.map(v=>`<i class="${v===null?"":v?"y":"n"}"></i>`).join("")}</div>`;}
function boolCard(team,s,test,lbl){
  if(!s.ok)return `<div class="cc">${ccHead(team,s)}<div class="empty">${esc(s.msg)}</div></div>`;
  const arr=s.ms.slice().reverse().map(nrm).map(test), ok=arr.filter(v=>v!==null), x=ok.filter(Boolean).length;
  return `<div class="cc">${ccHead(team,s)}<div class="ccs"><span>${esc(lbl)} <b class="${pcls(x/ok.length)}">${x} de ${ok.length}</b></span></div>${sqGen(arr)}<div class="sub" style="margin-top:6px">El cuadro de la derecha es el partido más reciente.</div></div>`;
}
function resCard(team,s){
  if(!s.ok)return `<div class="cc">${ccHead(team,s)}<div class="empty">${esc(s.msg)}</div></div>`;
  const r=s.ms.slice().reverse().map(nrm).map(m=>m.f>m.c?"G":m.f===m.c?"E":"P"), cn=l=>r.filter(x=>x===l).length;
  return `<div class="cc">${ccHead(team,s)}<div class="ccs"><span>Ganó <b class="g">${cn("G")}</b></span><span>Empató <b>${cn("E")}</b></span><span>Perdió <b class="b">${cn("P")}</b></span></div><div class="rl">${r.map(x=>`<i class="${x}">${x}</i>`).join("")}</div><div class="sub" style="margin-top:6px">La letra de la derecha es el partido más reciente.</div></div>`;
}
function h2hRaw(h,a,k){
  const rh=resolveHist(h), ra=resolveHist(a), out=[];
  if(rh&&ra&&rh.key===ra.key){
    const L=DATA.leagues[rh.key], i=L.hteams.indexOf(rh.name), j=L.hteams.indexOf(ra.name);
    for(let q=L.hm.length-1;q>=0&&out.length<k;q--){const m=L.hm[q];
      if((m[1]===i&&m[2]===j)||(m[1]===j&&m[2]===i))out.push({d:m[0],h:L.hteams[m[1]],a:L.hteams[m[2]],hg:m[3],ag:m[4],hc:m[5],ac:m[6],kh:m[7]===undefined?-1:m[7],ka:m[8]===undefined?-1:m[8],home:m[1]===i});}
    if(out.length)return out;
  }
  const ch=uclName(h), ca=uclName(a), M=DATA.uclh||[];
  if(ch&&ca)for(let q=M.length-1;q>=0&&out.length<k;q--){const m=M[q];
    if((m[1]===ch&&m[2]===ca)||(m[1]===ca&&m[2]===ch))out.push({d:m[0],h:m[1],a:m[2],hg:m[3],ag:m[4],hc:-1,ac:-1,kh:-1,ka:-1,home:m[1]===ch});}
  return out;
}
/* Líneas de evidencia: cuántas veces se cumplió en los partidos del contexto (local en casa, visitante fuera). */
function evRows(h,a,test,o){
  const rows=[], rh=resolveHist(h), ra=resolveHist(a), ph=o.h!==false&&rh, pa=o.a!==false&&ra;
  const add=(arrs,txt)=>{let x=0,n=0;arrs.forEach(ms=>ms.forEach(m=>{const r=test(m);if(r!==null){n++;if(r)x++;}}));if(n)rows.push({x:x,n:n,p:x/n,txt:txt});};
  [5,10,20].forEach(k=>{
    const who=[ph?h+" de local":"",pa?a+" de visitante":""].filter(Boolean).join(" · ");
    add([ph?domMatches(rh,k,"home").map(nrm):[],pa?domMatches(ra,k,"away").map(nrm):[]],"últimos "+k+": "+who);
  });
  const nm2=[ph?h:"",pa?a:""].filter(Boolean).join(" y ");
  add([ph?domMatches(rh,10,"all").map(nrm):[],pa?domMatches(ra,10,"all").map(nrm):[]],"últimos 10 de "+nm2+", local y visita");
  if(o.h2h!==false)add([h2hRaw(h,a,5).map(r=>nrm(o.flip?Object.assign({},r,{home:!r.home}):r))],"enfrentamientos directos (últimos 5)");
  const ch=uclName(h), ca=uclName(a), uh=o.h!==false&&ch, ua=o.a!==false&&ca;
  add([uh?uclMatches(ch,5).map(nrm):[],ua?uclMatches(ca,5).map(nrm):[]],"en Champions: últimos 5 de "+[uh?h:"",ua?a:""].filter(Boolean).join(" y "));
  return rows;
}
function evCard(title,rows,note){
  return `<div class="box ev"><h3>${esc(title)}</h3>${note?`<div class="sub">${esc(note)}</div>`:""}${rows.length?rows.map(r=>`<div class="evr"><span class="evc">${r.x}/${r.n}</span><span class="evt">${esc(r.txt)}</span><span class="evp ${pcls(r.p)}">${Math.round(r.p*100)}%</span></div>`).join(""):`<div class="empty">No hay suficientes partidos para calcular esto.</div>`}</div>`;
}
const EVNOTE="Cuántas veces se cumplió en esos partidos. Verde: 70% o más. Naranja: entre 50% y 69%. Rojo: menos de 50%. Es historia, no una probabilidad.";
function renderStats(h,a,fx){
  $("scopeT").innerHTML=scopeHtml();
  $("metricT").innerHTML=METCH.map(([k,t])=>`<button class="chip" type="button" data-m="${k}" aria-pressed="${k===met}">${t}</button>`).join("");
  const sh=seriesFor(h,scope), sa=seriesFor(a,scope);
  const seg2=(attr,cur,opts)=>`<div class="seg two" role="group">${opts.map(([k,t])=>`<button type="button" data-${attr}="${k}" aria-pressed="${k===cur}">${t}</button>`).join("")}</div>`;
  let body="";
  if(MET[met]){
    const c=MET[met], vf=vfn(met,who), line=lineOf(), lines=who==="tot"?c.tl:c.el;
    let ctl=seg2("who",who,[["tot","Total del partido"],["team","Solo el equipo"]])+seg2("dir",dir,[["o","Más de"],["u","Menos de"]])
      +`<div class="row-scroll lines" role="group" aria-label="Línea">${lines.map(l=>`<button class="chip" type="button" data-line="${l}" aria-pressed="${l===line}">${l}</button>`).join("")}</div>`;
    body+=`<div class="box ctl">${ctl}</div>`;
    if(c.need&&scope==="ucl"){
      body+=`<div class="box"><div class="empty">No tenemos córners ni tarjetas de la Champions League. Prueba con Goles, o cambia a Últimos 10, De local o De visita.</div></div>`;
    }else{
      body+=chartCard(h,sh,vf,line)+chartCard(a,sa,vf,line);
      const test=m=>{const v=vf(m);return v===null?null:hitV(v,line);};
      const lbl=(dir==="o"?"Más de ":"Menos de ")+line+" "+c.n+(who==="team"?" de cada equipo":"");
      body+=evCard(lbl,evRows(h,a,test,{h2h:who==="tot"}),EVNOTE);
    }
  }else if(met==="btts"){
    body+=`<div class="box ctl">${seg2("dir",dir,[["o","Ambos anotan: sí"],["u","Ambos anotan: no"]])}</div>`;
    const test=m=>{const b=m.f>0&&m.c>0;return dir==="o"?b:!b;};
    body+=boolCard(h,sh,test,dir==="o"?"Ambos anotaron":"No anotaron los dos")+boolCard(a,sa,test,dir==="o"?"Ambos anotaron":"No anotaron los dos");
    body+=evCard(dir==="o"?"Ambos anotan: sí":"Ambos anotan: no",evRows(h,a,test,{}),EVNOTE);
  }else{
    body+=resCard(h,sh)+resCard(a,sa);
    body+=evCard("Gana "+h,evRows(h,a,m=>m.f>m.c,{a:false}),EVNOTE);
    body+=evCard("Empate",evRows(h,a,m=>m.f===m.c,{}),null);
    body+=evCard("Gana "+a,evRows(h,a,m=>m.f>m.c,{h:false,flip:true}),null);
  }
  $("statBody").innerHTML=body;
}
function renderFeat(){
  const sc=league==="ucl"?"ucl":"all";
  const list=fxOfLeague().filter(f=>f.date.slice(0,10)===day), items=[];
  list.forEach(f=>{
    const sh=seriesFor(f.h,sc,10), sa=seriesFor(f.a,sc,10); if(!sh.ok||!sa.ok)return;
    const A=toTrend(sh.ms), B=toTrend(sa.ms), best={};
    TM.forEach(def=>{
      const x=trendHits(A,def), y=trendHits(B,def); if(x.n<8||y.n<8)return;
      const rx=x.k/x.n, ry=y.k/y.n; if(Math.min(rx,ry)<0.7)return;
      const it={f:f,def:def,x:x,y:y,score:(rx+ry)/2}, b=best[def.cat];
      if(!b||(def.ln?def.ln>b.def.ln:it.score>b.score))best[def.cat]=it;
    });
    Object.keys(best).forEach(c=>items.push(best[c]));
  });
  items.sort((p,q)=>q.score-p.score);
  $("feat").innerHTML=items.length?items.slice(0,10).map(it=>{
    const f=it.f, o=it.def.odds&&f.odds?f.odds[it.def.odds]:null;
    return `<button class="fp" type="button" data-f="${DATA.fixtures.indexOf(f)}"><span class="ft"><span>${esc(f.h)} vs ${esc(f.a)}</span><span>${esc(dtxt(f.date))}</span></span>
      <span class="fm">${esc(it.def.label)}${o>1?`<small class="fo">cuota ${f2(o)}</small>`:""}</span>
      <span class="tl"><span class="who">${esc(f.h)}</span>${sqHtml(it.x)}<b>${it.x.k}/${it.x.n}</b></span>
      <span class="tl"><span class="who">${esc(f.a)}</span>${sqHtml(it.y)}<b>${it.y.k}/${it.y.n}</b></span></button>`;}).join("")
    :`<div class="empty">Ningún partido de este día tiene una racha de 7 de 10 o más en los dos equipos. Prueba con otro día.</div>`;
}

function onOdds(i){
  const r=cur.rows[i], v=parseFloat($("o"+i).value.replace(",","."));
  const e=$("e"+i), b=$("b"+i), f=$("f"+i);
  if(!(v>1)){e.textContent="";e.className="ev";f.innerHTML="";b.disabled=true;return;}
  const ev=r.p*v-1, gap=r.p-1/v;
  let flag="";
  if(ev>0 && gap>=0.08) flag=`<span class="flag">Diferencia grande con la casa: probable error del modelo</span>`;
  else if(r.fav && r.p<0.5 && ev>0) flag=`<span class="flag">Ojo: el modelo subestima favoritos</span>`;
  e.className="ev "+(ev>0?"pos":"neg");
  e.textContent=(ev>0?"+":"")+(ev*100).toFixed(1)+"%";
  f.innerHTML=flag;
  b.disabled=false;
}


/* ---------- pantalla del partido ---------- */
function updTxt(){
  const L=DATA.leagues[league];
  $("upd").innerHTML=(L.kind==="elo"?"Ranking Elo del "+esc(L.updated):"Datos hasta "+esc(L.updated))+"<br>Actualizado "+esc((DATA.generated||"—").slice(0,16));
}
function fitName(el){const n=el.textContent.length; el.style.fontSize=(n<=7?34:n<=10?29:n<=13?25:21)+"px";}
function render(){
  const h=$("home").value,a=$("away").value,L=DATA.leagues[league];
  updTxt();
  if(h===a)return;
  const fh=1-0.07*(+$("adjH").value), fa=1-0.07*(+$("adjA").value);
  const fx=(DATA.fixtures||[]).find(f=>f.lg===league&&f.h===h&&f.a===a);
  const mk0=league+"|"+h+"|"+a; if(mk0!==userMatch){userOdds={};userMatch=mk0;}
  const uml=mlFromOdds(userOdds);
  const ml=uml||(fx&&fx.ml?fx.ml:null);
  cur=compute(h,a,fh,fa,ml); cur.src=uml?"user":(ml?"fx":null);
  $("hMeta").textContent=L.name+(fx?" · "+dtxt(fx.date):"");
  $("hn").textContent=h; $("an").textContent=a; fitName($("hn")); fitName($("an"));
  $("hf").innerHTML=formHtml(h); $("af").innerHTML=formHtml(a);
  const [pH,pD,pA]=cur.p;
  const seg=(id,p)=>{const e=$(id);e.style.flexGrow=Math.max(p,0.001);};
  seg("sH",pH);seg("sD",pD);seg("sA",pA);
  $("lHn").textContent="Gana "+h; $("lAn").textContent="Gana "+a;
  $("lH").textContent=pct(pH); $("lD").textContent=pct(pD); $("lA").textContent=pct(pA);
  const mo=fairFrom(userOdds)?userOdds:(fx&&fx.odds), fr=mo?fairFrom(mo):null, m=fr?fr.p:null, who=fairFrom(userOdds)?"tu casa":"las casas de apuestas";
  $("ticks").innerHTML=m?`<div class="tick" style="left:${(m[0]*100).toFixed(1)}%"></div><div class="tick" style="left:${((m[0]+m[1])*100).toFixed(1)}%"></div>`:"";
  $("mkNote").textContent=m?`Las líneas punteadas marcan lo que dan ${who}, sin margen: ${h} ${pct(m[0])}, empate ${pct(m[1])}, ${a} ${pct(m[2])}.`:"";
  // resumen
  const top=[[h,pH],["empate",pD],[a,pA]].sort((x,y)=>y[1]-x[1])[0];
  $("readout").textContent=(top[0]==="empate"?"Lo más probable es un empate":"Lo más probable es que gane "+top[0])+" ("+pct(top[1])+"). Es un cálculo con datos pasados, no una garantía.";
  const tile=(v,t,s)=>`<div class="tile"><b>${v}</b><span>${t}</span><small>${s}</small></div>`;
  $("tiles").innerHTML=tile(cur.lh.toFixed(2)+" – "+cur.la.toFixed(2),"Goles esperados","Promedio que se espera de cada equipo. No es un marcador.")
    +tile(pct(cur.o25),"Más de 2.5 goles","Probabilidad de que haya 3 goles o más.")
    +tile(pct(cur.bt),"Ambos anotan","Probabilidad de que los dos equipos marquen.")
    +cur.chips.map(c=>tile(c[1],c[0]+" esperados","Promedio de cada equipo.")).join("");
  $("ol1").textContent="Gana "+h; $("ol2").textContent="Gana "+a;
  [["ou1","1"],["ouX","X"],["ou2","2"],["ouO","o:goals:2.5"],["ouU","u:goals:2.5"]].forEach(([id,k])=>{
    const el=$(id), want=userOdds[k], now=parseFloat(el.value.replace(",","."));
    if(!(want===now||(want===undefined&&!(now>0))))el.value=want===undefined?"":want;
    el.placeholder=fx&&fx.odds&&fx.odds[k]?f2(fx.odds[k]):"Cuota"+((k==="1"||k==="X"||k==="2")?"":" (opcional)");
  });
  const fu=fairFrom(userOdds), ff=fx&&fx.odds?fairFrom(fx.odds):null;
  $("houseTxt").innerHTML=fu?`<b>Tu casa se queda un ${(fu.margin*100).toFixed(1)}% de margen</b> en este mercado. Sin ese margen da: ${esc(h)} ${pct(fu.p[0])}, empate ${pct(fu.p[1])}, ${esc(a)} ${pct(fu.p[2])}. Las probabilidades de arriba ya usan tus cuotas.`
    :(ff?`En gris ves el promedio de varias casas (margen ${(ff.margin*100).toFixed(1)}%). Cámbialo por las cuotas de tu casa y recalculamos todo.`
         :"No hay cuotas de las casas para este partido. Escribe las de tu casa en los tres campos de 1, empate y 2.");
  const how=[];
  if(cur.src==="user") how.push("Estas probabilidades se calcularon a partir de las cuotas que escribiste (resultado y, si las pusiste, más y menos 2.5), sin el margen de la casa. Ya no son las del modelo.");
  else if(L.kind==="elo") how.push("Las probabilidades salen del ranking Elo de los clubes. No están validadas con partidos pasados de Champions, así que el error puede ser mayor que en las ligas. Si escribes las cuotas de tu casa, se recalculan con ellas.");
  else if(m) how.push("En las pruebas con partidos pasados, las casas de apuestas acertaron más que el modelo. Úsalas como referencia principal.");
  else how.push("Estas probabilidades son del modelo estadístico (goles de las últimas temporadas). Sin cuotas de las casas no hay con qué compararlas.");
  if(cur.fromMkt&&cur.src!=="user") how.push("Resultado, goles, más o menos y ambos anotan están calculados a partir de las cuotas de las casas; en las pruebas salieron más precisos que los del modelo. Córners y tiros siguen siendo del modelo.");
  if(cur.src==="user") how.push("Como las probabilidades salen de tu casa, la ganancia esperada de esas filas solo refleja su margen. Córners y tiros siguen siendo del modelo.");
  if(!cur.fromMkt&&L.kind!=="elo") how.push("Rango típico de error del modelo: ±8 puntos en el 1X2 (en 2.300 partidos pasados, la diferencia con las casas fue menor a eso el 80% de las veces).");
  $("howTxt").innerHTML=how.map(x=>`<p>${esc(x)}</p>`).join("");
  $("inj").innerHTML=L.kind==="elo"?`<div class="none">Champions League: las probabilidades salen del ranking Elo de los clubes (sin córners ni tiros, porque no hay datos comparables entre ligas). No están validadas con partidos pasados.</div>`:injHtml(h,a);
  renderHist(h,a); renderStats(h,a,fx);
  // mercados
  let html="",sec="",sub=null;
  cur.rows.forEach((r,i)=>{
    const parts=r.sec.split(" · "), g=parts[0], sb=parts[1]||null;
    if(g!==sec){
      if(sec)html+="</details>";
      sec=g; sub=null; const n=cur.rows.filter(x=>x.sec.split(" · ")[0]===sec).length;
      html+=`<details class="grp"${(sec==="Resultado"||sec==="Goles")?" open":""}><summary>${esc(sec)}<span class="cnt">${n}</span></summary>`;
    }
    if(sb&&sb!==sub){sub=sb; html+=`<div class="subh">${esc(sb==="total"?"Total del partido":sb)}</div>`;}
    html+=`<div class="mrow"><div class="mtop"><span class="ml">${esc(r.label)}</span><span class="mp">${pct(r.p)}</span></div>
      <div class="pbar"><i style="width:${(r.p*100).toFixed(0)}%"></i></div>
      <div class="mbot"><span class="mf">Cuota justa ${f2(1/r.p)}</span><input inputmode="decimal" id="o${i}" data-i="${i}" placeholder="Tu cuota" aria-label="Tu cuota para ${esc(r.label)}"><span class="ev" id="e${i}" title="Ganancia esperada por cada L 1 apostado"></span><button class="mini" id="b${i}" data-i="${i}" disabled type="button">Anotar</button></div>
      <div class="frow" id="f${i}"></div></div>`;
  });
  if(sec)html+="</details>";
  $("mk").innerHTML=html;
  let n=0;
  cur.rows.forEach((r,i)=>{
    const k=rowKey(r); let v=userOdds[k];
    if(v===undefined&&fx&&fx.odds&&r.key&&fx.odds[r.key]){v=fx.odds[r.key];n++;}
    if(v!==undefined){$("o"+i).value=v;onOdds(i);}
  });
  $("oddsSrc").textContent=(n?`Cargamos ${n} cuotas automáticamente (promedio de varias casas). Cámbialas por las de tu casa. `:"")+"Cuota justa: la mínima para que valga la pena. Número verde o rojo: lo que ganarías o perderías en promedio por cada L 1 si el modelo acertara.";
}

/* ---------- pantallas ---------- */
function quick(f){try{return compute(f.h,f.a,1,1,f.ml||null).p;}catch(e){return null;}}
function tabs(){
  $("leagueTabs").innerHTML=Object.entries(DATA.leagues).map(([k,L])=>
    `<button class="chip" type="button" data-k="${k}" aria-pressed="${k===league}">${esc(L.name)}</button>`).join("");
}
function fxList(){
  const days=daysOfLeague(), list=fxOfLeague().filter(f=>f.date.slice(0,10)===day);
  $("fxNote").textContent=DATA.note||"";
  $("dayChips").hidden=!days.length;
  $("dayChips").innerHTML=days.map(k=>{const p=dparts(k),n=fxOfLeague().filter(f=>f.date.slice(0,10)===k).length;
    return `<button class="day" type="button" data-d="${k}" aria-pressed="${k===day}"><b>${p.dow} ${p.d} ${p.mon}</b><small>${n} ${n===1?"partido":"partidos"}</small></button>`;}).join("");
  $("fxList").innerHTML=list.length?list.map(f=>{
    const p=quick(f), q=p?`<span class="mq"><span class="qb"><i style="flex-grow:${p[0].toFixed(3)}"></i><i style="flex-grow:${p[1].toFixed(3)}"></i><i style="flex-grow:${p[2].toFixed(3)}"></i></span><span class="qn"><span>${pct(p[0])}</span><span>${pct(p[1])}</span><span>${pct(p[2])}</span></span></span>`:"<span></span>";
    return `<button class="mc" type="button" data-f="${DATA.fixtures.indexOf(f)}"><span class="mtime">${esc(f.date.slice(11,16))}</span><span class="mtt"><span>${esc(f.h)}</span><span>${esc(f.a)}</span></span>${q}<span class="chev"></span></button>`;}).join("")
    :`<div class="empty">No hay partidos próximos de ${esc(DATA.leagues[league].name)} en la fuente de datos. Puedes elegir los equipos abajo.</div>`;
  $("manual").open=!list.length;
}
function setView(v){
  if(v!=="match")prevView=v;
  view=v;
  [["list","vList"],["feat","vFeat"],["log","vLog"],["match","vMatch"]].forEach(([k,id])=>{$(id).hidden=k!==v;});
  $("ctx").hidden=!(v==="list"||v==="feat");
  document.querySelectorAll(".nav button").forEach(b=>b.setAttribute("aria-current",b.dataset.v===(v==="match"?prevView:v)?"page":"false"));
  window.scrollTo(0,0);
}
function setTab(t){
  tab=t;
  document.querySelectorAll("#mTabs button").forEach(b=>b.setAttribute("aria-selected",b.dataset.t===t?"true":"false"));
  [["sum","pSum"],["hist","pHist"],["trend","pTrend"],["mk","pMk"]].forEach(([k,id])=>{$(id).hidden=k!==t;});
  const mh=document.querySelector(".mhead"), y=mh.getBoundingClientRect().bottom+window.scrollY-8;
  if(window.scrollY>y)window.scrollTo(0,y);
}
function openMatch(h,a,t){
  $("home").value=h; $("away").value=a; $("adjH").value="0"; $("adjA").value="0";
  histTeam="h"; scope=league==="ucl"?"ucl":"all";
  render(); setTab(t||"sum"); setView("match");
  try{history.pushState({m:1},"");}catch(e){}
}
function goBack(){ if(history.state&&history.state.m){history.back();} else setView(prevView); }
window.addEventListener("popstate",()=>{ if(view==="match")setView(prevView); });

$("fxList").addEventListener("click",ev=>{const b=ev.target.closest("button[data-f]"); if(!b)return; const f=DATA.fixtures[+b.dataset.f]; openMatch(f.h,f.a);});
$("feat").addEventListener("click",ev=>{const b=ev.target.closest("button[data-f]"); if(!b)return; const f=DATA.fixtures[+b.dataset.f]; openMatch(f.h,f.a,"trend");});
$("dayChips").addEventListener("click",ev=>{const b=ev.target.closest("button[data-d]"); if(!b)return; day=b.dataset.d; fxList(); renderFeat();});
$("leagueTabs").addEventListener("click",ev=>{
  const k=ev.target.dataset.k; if(!k||k===league)return;
  league=k; day=null; tabs(); fillTeams(); updTxt(); fxList(); renderFeat();
});
$("goManual").addEventListener("click",()=>{
  const h=$("home").value,a=$("away").value;
  if(h===a){$("manualMsg").textContent="Elige dos equipos distintos.";return;}
  $("manualMsg").textContent=""; openMatch(h,a);
});
document.querySelector(".nav").addEventListener("click",ev=>{
  const b=ev.target.closest("button[data-v]"); if(!b)return;
  setView(b.dataset.v); if(b.dataset.v==="feat")renderFeat(); if(b.dataset.v==="log")renderLog();
});
$("back").addEventListener("click",goBack);
$("mTabs").addEventListener("click",ev=>{const b=ev.target.closest("button[data-t]"); if(b)setTab(b.dataset.t);});
function redrawStats(){
  const h=$("home").value,a=$("away").value, fx=(DATA.fixtures||[]).find(f=>f.lg===league&&f.h===h&&f.a===a);
  renderHist(h,a); renderStats(h,a,fx);
}
function onScope(ev){
  const b=ev.target.closest("button[data-s],button[data-w]"); if(!b)return;
  if(b.dataset.s)scope=b.dataset.s; else win=+b.dataset.w;
  redrawStats();
}
$("pHist").addEventListener("click",ev=>{
  onScope(ev);
  const t=ev.target.closest("button[data-ht]"); if(t){histTeam=t.dataset.ht; renderHist($("home").value,$("away").value);}
});
$("pTrend").addEventListener("click",ev=>{
  onScope(ev);
  const b=ev.target.closest("button"); if(!b)return;
  if(b.dataset.m){met=b.dataset.m; dir="o"; who="tot";}
  else if(b.dataset.who)who=b.dataset.who;
  else if(b.dataset.dir)dir=b.dataset.dir;
  else if(b.dataset.line)lineSel[met+who]=parseFloat(b.dataset.line);
  else return;
  redrawStats();
});
$("adjH").addEventListener("change",render); $("adjA").addEventListener("change",render);
$("pSum").addEventListener("input",ev=>{
  const k=ev.target.dataset&&ev.target.dataset.k; if(!k)return;
  const v=parseFloat(ev.target.value.replace(",","."));
  if(v>1)userOdds[k]=v; else delete userOdds[k];
});
$("pSum").addEventListener("change",ev=>{if(ev.target.dataset&&ev.target.dataset.k)render();});
$("mk").addEventListener("input",ev=>{
  if(ev.target.dataset.i===undefined||ev.target.tagName!=="INPUT")return;
  const i=+ev.target.dataset.i, v=parseFloat(ev.target.value.replace(",","."));
  if(v>1)userOdds[rowKey(cur.rows[i])]=v; else delete userOdds[rowKey(cur.rows[i])];
  onOdds(i);
});
$("mk").addEventListener("change",ev=>{
  if(ev.target.dataset.i===undefined||ev.target.tagName!=="INPUT")return;
  if(["1","X","2","o:goals:2.5","u:goals:2.5"].includes(cur.rows[+ev.target.dataset.i].key))render();
});
$("mk").addEventListener("click",ev=>{if(ev.target.tagName==="BUTTON"&&!ev.target.disabled)addPick(+ev.target.dataset.i);});

/* ---------- bitácora ---------- */
function readLog(){try{return JSON.parse(localStorage.getItem(KEY)||"[]");}catch(e){return memLog;}}
let memLog=[];
function writeLog(a){memLog=a;try{localStorage.setItem(KEY,JSON.stringify(a));}catch(e){}}
function addPick(i){
  const r=cur.rows[i], odds=parseFloat($("o"+i).value.replace(",","."));
  const log=readLog();
  log.unshift({id:Date.now(),d:new Date().toISOString().slice(0,10),lg:DATA.leagues[league].name,
    match:$("home").value+" vs "+$("away").value,mk:r.sec+": "+r.label,odds,p:r.p,stake:1,st:"pending"});
  writeLog(log); renderLog();
}
function renderLog(){
  const log=readLog();
  let staked=0,profit=0,w=0,l=0,pSum=0,n=0;
  for(const e of log){
    if(e.st==="pending")continue;
    staked+=e.stake; n++; pSum+=e.p;
    if(e.st==="won"){profit+=e.stake*(e.odds-1);w++;} else {profit-=e.stake;l++;}
  }
  const roi=staked?profit/staked:0;
  $("stats").innerHTML=
    `<div class="stat"><span class="lbl">Resueltas</span><b>${n}</b></div>
     <div class="stat"><span class="lbl">Ganadas</span><b>${w}/${n}</b></div>
     <div class="stat"><span class="lbl">Prob. media del modelo</span><b>${n?pct(pSum/n):"—"}</b></div>
     <div class="stat ${profit>0?"p":profit<0?"n":""}"><span class="lbl">Balance (L)</span><b>${profit>0?"+":""}${profit.toFixed(2)}</b></div>
     <div class="stat ${roi>0?"p":roi<0?"n":""}"><span class="lbl">ROI</span><b>${n?(roi*100).toFixed(0)+"%":"—"}</b></div>`;
  $("log").innerHTML=log.length?log.map(e=>`<div class="entry">
    <div><div class="m">${e.match}</div><div class="d">${e.mk} · cuota ${f2(e.odds)} · modelo ${pct(e.p)} · ${e.lg} · ${e.d}</div></div>
    <div class="act"><span class="pill ${e.st}">${e.st==="won"?"Ganada":e.st==="lost"?"Perdida":"Pendiente"}</span>
      <input style="width:64px;padding:4px 6px;font-size:14px" aria-label="Monto en lempiras" title="Monto (L)" data-stake="${e.id}" value="${e.stake}">
      <button class="mini" data-w="${e.id}" type="button">Ganó</button><button class="mini" data-l="${e.id}" type="button">Perdió</button><button class="mini" data-x="${e.id}" type="button">Borrar</button></div>
  </div>`).join(""):`<div class="empty">Todavía no hay apuestas. Escribe una cuota arriba y toca Anotar para registrar la predicción y comparar después el modelo contra los resultados reales.</div>`;
}
$("log").addEventListener("click",ev=>{
  const t=ev.target,log=readLog();
  const upd=(id,fn)=>{const e=log.find(x=>String(x.id)===id);if(e)fn(e);writeLog(log);renderLog();};
  if(t.dataset.w)upd(t.dataset.w,e=>e.st="won");
  else if(t.dataset.l)upd(t.dataset.l,e=>e.st="lost");
  else if(t.dataset.x){writeLog(log.filter(x=>String(x.id)!==t.dataset.x));renderLog();}
});
$("log").addEventListener("change",ev=>{
  const id=ev.target.dataset.stake; if(!id)return;
  const log=readLog(),e=log.find(x=>String(x.id)===id),v=parseFloat(String(ev.target.value).replace(",","."));
  if(e&&v>0){e.stake=v;writeLog(log);}renderLog();
});
$("copy").addEventListener("click",()=>{
  const txt=readLog().map(e=>`${e.d} | ${e.match} | ${e.mk} | cuota ${f2(e.odds)} | L${e.stake} | modelo ${pct(e.p)} | ${e.st}`).join("\n")||"(vacía)";
  const done=m=>{$("copyMsg").textContent=m;};
  if(navigator.clipboard&&navigator.clipboard.writeText)
    navigator.clipboard.writeText(txt).then(()=>done("Copiado."),()=>done("No se pudo copiar automáticamente."));
  else done("No se pudo copiar automáticamente.");
});


league=(DATA.fixtures&&DATA.fixtures[0]&&DATA.fixtures[0].lg)||Object.keys(DATA.leagues)[0];
tabs(); fillTeams(); updTxt(); fxList(); renderFeat(); renderLog(); setView("list");
</script>

</body></html>
'''


if __name__ == "__main__":
    build(TEMPLATE)
