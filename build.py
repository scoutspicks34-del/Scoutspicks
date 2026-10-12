"""Scout Picks: descarga datos, ajusta el modelo, consulta la API y genera docs/index.html."""
import io, os, re, sys, json, difflib, unicodedata, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

LEAGUES = {
    "bundesliga": dict(name="Bundesliga", code="D1", api=78),
    "premier": dict(name="Premier League", code="E0", api=39),
    "laliga": dict(name="La Liga", code="SP1", api=140),
}
MARKETS = {"corners": ("HC", "AC"), "sot": ("HST", "AST"), "shots": ("HS", "AS")}
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
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
    return {"updated": str(df.Date.max().date()), "n": int(len(df)), "models": models,
            "teams": teams, "form": form, "inj": None}


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
    data = {"leagues": params, "fixtures": fixtures, "note": note,
            "generated": local.strftime("%Y-%m-%d %H:%M") + " (hora de Honduras)"}
    html = template.replace("__PARAMS__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    log(f"Página escrita en {out_path} ({len(html) // 1024} KB)")


TEMPLATE = r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light}body{margin:0}</style>
<title>Scout Picks</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;600;700;800&family=Instrument+Sans:wght@400;500;600&display=swap">
<style>
/* Diseño: la probabilidad va primero. Barra de 3 tramos (local / empate / visitante) con marcas de lo que dan las casas.
   Móvil: una columna. Computadora: riel izquierdo (liga, día, partidos) y panel derecho (análisis, mercados, bitácora). */
:root{
  --bg:#ecedf4; --paper:#ffffff; --ink:#121a3a; --muted:#5a6384; --line:#dcdfeb; --soft:#f4f5fa;
  --home:#2b45f0; --home-ink:#ffffff; --away:#f0902a; --away-ink:#2a1700; --draw:#c9cedd; --draw-ink:#1d2547;
  --good:#0b8a5f; --good-bg:#dff3ea; --bad:#c93a47; --bad-bg:#f9e1e3; --warn:#8a5a00; --warn-bg:#fdf0d4;
  --disp:"Bricolage Grotesque","Trebuchet MS",system-ui,sans-serif; --body:"Instrument Sans",system-ui,sans-serif;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0b1020; --paper:#141a33; --ink:#eaeeff; --muted:#9aa3c7; --line:#262e52; --soft:#1a2142;
  --home:#6f86ff; --home-ink:#0b1020; --away:#ffa24d; --away-ink:#2a1700; --draw:#3b4472; --draw-ink:#eaeeff;
  --good:#3ccf9b; --good-bg:#10332b; --bad:#ff7a86; --bad-bg:#3a1a22; --warn:#f0b958; --warn-bg:#332710; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0b1020; --paper:#141a33; --ink:#eaeeff; --muted:#9aa3c7; --line:#262e52; --soft:#1a2142;
  --home:#6f86ff; --home-ink:#0b1020; --away:#ffa24d; --away-ink:#2a1700; --draw:#3b4472; --draw-ink:#eaeeff;
  --good:#3ccf9b; --good-bg:#10332b; --bad:#ff7a86; --bad-bg:#3a1a22; --warn:#f0b958; --warn-bg:#332710; color-scheme:dark}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body);padding:0 16px 48px;font-variant-numeric:tabular-nums}
h1,h2,h3,p{margin:0}
button,select,input,summary{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--home);outline-offset:2px}
.app{max-width:1040px;margin:0 auto;display:grid;gap:16px;grid-template-columns:minmax(0,1fr)}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;padding-block:18px 2px}
.brand{display:flex;align-items:center;gap:10px;font:700 22px/1 var(--disp);letter-spacing:-.01em}
.logo{display:flex;gap:3px;align-items:flex-end;height:22px}
.logo i{display:block;width:7px;border-radius:2px}
.logo i:nth-child(1){height:22px;background:var(--home)} .logo i:nth-child(2){height:14px;background:var(--draw)} .logo i:nth-child(3){height:9px;background:var(--away)}
.upd{font-size:12px;color:var(--muted);text-align:right;line-height:1.3}
.rail,.main{display:flex;flex-direction:column;gap:16px;min-width:0}
.seg{display:flex;gap:6px;overflow-x:auto;scrollbar-width:none;padding-bottom:2px;-webkit-overflow-scrolling:touch}
.seg::-webkit-scrollbar{display:none}
.tab{flex:none;border:1px solid var(--line);background:var(--paper);padding:9px 15px;border-radius:999px;font-weight:600;font-size:14px;cursor:pointer;white-space:nowrap}
.tab[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--paper)}
.panel{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:16px;display:flex;flex-direction:column;gap:12px;min-width:0}
.panel>h2,.panel-h h2{font:700 19px/1.2 var(--disp);letter-spacing:-.01em}
.panel-h{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.sub{color:var(--muted);font-size:13px;line-height:1.4}
.days{display:flex;gap:6px;overflow-x:auto;scrollbar-width:none}
.days::-webkit-scrollbar{display:none}
.day{flex:none;border:1px solid var(--line);background:var(--soft);border-radius:12px;padding:6px 12px;font-size:13px;font-weight:600;cursor:pointer;text-align:center;line-height:1.25}
.day small{display:block;font-weight:500;color:var(--muted);font-size:11px}
.day[aria-pressed="true"]{border-color:var(--home);background:color-mix(in srgb,var(--home) 12%,var(--paper));color:var(--ink)}
.fxlist{display:flex;flex-direction:column}
.fx{display:grid;grid-template-columns:52px 1fr;gap:10px;align-items:center;text-align:left;width:100%;padding:10px 8px 10px 10px;background:transparent;border:0;border-top:1px solid var(--line);cursor:pointer;border-left:3px solid transparent}
.fx:first-child{border-top:0}
.fx:hover{background:var(--soft)}
.fx[aria-current="true"]{background:color-mix(in srgb,var(--home) 9%,var(--paper));border-left-color:var(--home)}
.fx .tm{font:600 14px var(--disp);color:var(--muted)}
.fx .tt{display:flex;flex-direction:column;gap:1px;font-weight:600;font-size:14.5px;min-width:0}
.fx .tt span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fx .tt .v{font-weight:500;color:var(--muted)}
.empty{color:var(--muted);font-size:14px;padding:6px 0}
details>summary{cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:10px}
details>summary::-webkit-details-marker{display:none}
details>summary::after{content:"";width:8px;height:8px;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg);transition:transform .15s;margin-right:4px;flex:none}
details[open]>summary::after{transform:rotate(-135deg)}
.manual>summary{font-weight:600;font-size:14px;padding:6px 0}
.pick{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding-top:6px}
.pick label,.adj label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--muted);min-width:0}
select,input{padding:10px 11px;border:1px solid var(--line);border-radius:10px;background:var(--soft);width:100%;min-width:0;font-size:16px}
/* análisis del partido */
.hero{background:var(--paper);border:1px solid var(--line);border-radius:26px;padding:20px 18px 18px;display:flex;flex-direction:column;gap:18px;min-width:0}
.meta{font-size:13px;color:var(--muted);font-weight:500}
.teams{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:start}
.nm{font:700 24px/1.05 var(--disp);letter-spacing:-.015em;overflow-wrap:anywhere;text-wrap:balance}
.tm.away{text-align:right}
.tm.away .form{justify-content:flex-end}
.form{display:flex;gap:3px;margin-top:8px}
.form i{font:700 10px var(--disp);font-style:normal;width:17px;height:17px;display:grid;place-items:center;border-radius:5px;color:#fff}
.form .G{background:var(--good)} .form .E{background:#8791b4} .form .P{background:var(--bad)}
.xgbox{display:flex;align-items:baseline;gap:6px;font:800 40px/1 var(--disp);letter-spacing:-.03em;padding-top:2px}
.xgbox .d{color:var(--muted);font-weight:500;font-size:28px}
.xgcap{font-size:12px;color:var(--muted);text-align:center;margin-top:-10px}
.barwrap{position:relative;padding-block:10px}
.stack{display:flex;height:48px;border-radius:14px;overflow:hidden;gap:3px}
.stack span{display:flex;align-items:center;justify-content:center;min-width:0;overflow:hidden;font:700 17px var(--disp);transition:flex-grow .5s cubic-bezier(.2,.8,.2,1);flex:1 1 0}
.stack .h{background:var(--home);color:var(--home-ink)} .stack .x{background:var(--draw);color:var(--draw-ink)} .stack .a{background:var(--away);color:var(--away-ink)}
.ticks{position:absolute;inset:0;pointer-events:none}
.tick{position:absolute;top:2px;bottom:2px;width:0;border-left:2px dashed var(--ink);opacity:.85}
.tick::before{content:"";position:absolute;top:-3px;left:-5px;border:4px solid transparent;border-top:6px solid var(--ink)}
.legend{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.legend div{display:flex;flex-direction:column;gap:2px;min-width:0}
.legend b{font:700 15px/1.2 var(--disp);overflow-wrap:anywhere}
.legend span{font-size:12px;color:var(--muted)}
.legend .l3{text-align:right;align-items:flex-end} .legend .l2{text-align:center;align-items:center}
.legend i{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:6px}
.chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}
.chip{background:var(--soft);border-radius:14px;padding:11px 13px;display:flex;flex-direction:column;gap:3px;font-size:12.5px;color:var(--muted)}
.chip b{font:700 24px/1 var(--disp);letter-spacing:-.02em;color:var(--ink)}
.how>summary{font-size:13px;font-weight:600;color:var(--muted);padding:2px 0}
.how .body{font-size:13px;color:var(--muted);padding-top:8px;display:flex;flex-direction:column;gap:8px}
.how .body b{color:var(--ink)}
.inj{font-size:13px;display:flex;flex-direction:column;gap:4px;padding-top:8px}
.inj .none{color:var(--muted)}
.adj{display:flex;flex-direction:column;gap:8px;padding-top:10px}
.adj .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.lbl{font-size:12px;color:var(--muted)}
/* mercados */
.callout{background:var(--warn-bg);color:var(--ink);border-radius:12px;padding:10px 12px;font-size:13px;line-height:1.4}
.grp{border-top:1px solid var(--line)}
.grp>summary{padding:13px 0;font:700 15px var(--disp)}
.grp>summary .cnt{margin-left:auto;font:500 12px var(--body);color:var(--muted)}
.mrow{padding:9px 0 11px;border-top:1px dashed var(--line)}
.mrow:first-of-type{border-top:0}
.subh{font-size:12px;font-weight:600;color:var(--muted);padding:10px 8px 2px}
.mtop{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:5px 8px;border-radius:8px;background:linear-gradient(90deg,color-mix(in srgb,var(--home) 16%,transparent) var(--p),transparent var(--p))}
.ml{font-weight:500;font-size:14.5px;overflow-wrap:anywhere}
.mp{font:700 17px var(--disp);flex:none}
.mbot{display:grid;grid-template-columns:1fr 82px 66px auto;gap:8px;align-items:center;padding:6px 8px 0}
.mf{font-size:12.5px;color:var(--muted)}
.mbot input{padding:7px 8px;font-size:15px;text-align:right;border-radius:9px}
.ev{font-weight:700;font-size:13.5px;text-align:right}
.ev.pos{color:var(--good)} .ev.neg{color:var(--muted)}
.frow{padding:4px 8px 0}
.flag{display:inline-block;font-size:11.5px;padding:3px 8px;border-radius:7px;background:var(--warn-bg);color:var(--warn)}
.mini{border:1px solid var(--line);background:transparent;border-radius:9px;padding:6px 10px;font-size:12.5px;font-weight:600;cursor:pointer;white-space:nowrap}
.mini:disabled{opacity:.35;cursor:default}
.btn{border:0;background:var(--ink);color:var(--paper);border-radius:11px;padding:9px 14px;font-weight:600;font-size:14px;cursor:pointer}
.btn.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}
/* bitácora */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}
.stat{background:var(--soft);border-radius:14px;padding:10px 12px;display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--muted)}
.stat b{font:700 22px/1 var(--disp);color:var(--ink);letter-spacing:-.02em}
.stat.p b{color:var(--good)} .stat.n b{color:var(--bad)}
.log{display:flex;flex-direction:column}
.entry{display:grid;grid-template-columns:1fr auto;gap:6px 10px;padding:12px 0;border-top:1px solid var(--line)}
.entry .m{font-weight:600;overflow-wrap:anywhere}
.entry .d{font-size:12.5px;color:var(--muted)}
.entry .act{display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.entry .act input{width:64px;padding:5px 7px;font-size:14px}
.pill{font:700 12px var(--disp);padding:4px 9px;border-radius:999px}
.pill.won{background:var(--good-bg);color:var(--good)} .pill.lost{background:var(--bad-bg);color:var(--bad)} .pill.pending{background:var(--soft);color:var(--muted)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.foot{font-size:12px;color:var(--muted);line-height:1.5}
@media (min-width:960px){
  body{padding-inline:28px}
  .app{grid-template-columns:350px minmax(0,1fr);column-gap:24px;align-items:start}
  .top{grid-column:1/-1}
  .rail{position:sticky;top:16px;max-height:calc(100vh - 32px);overflow:auto}
  .seg{flex-wrap:wrap;overflow:visible}
  .nm{font-size:30px} .xgbox{font-size:48px}
  .hero{padding:26px 26px 22px}
}
@media (max-width:600px){
  .teams{grid-template-columns:1fr 1fr;grid-template-areas:"x x" "h a";row-gap:14px}
  .xgbox{grid-area:x;justify-content:center;font-size:36px}
  .tm.home{grid-area:h} .tm.away{grid-area:a}
  .xgcap{margin-top:-8px}
}
@media (max-width:420px){
  .nm{font-size:21px}
  .mbot{grid-template-columns:1fr 78px 58px auto}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

</head><body>
<div class="app">
  <header class="top">
    <div class="brand"><span class="logo" aria-hidden="true"><i></i><i></i><i></i></span>Scout Picks</div>
    <div class="upd" id="upd"></div>
  </header>

  <aside class="rail">
    <div class="seg" id="leagueTabs" role="group" aria-label="Liga"></div>
    <section class="panel" aria-label="Partidos próximos">
      <div class="panel-h"><h2>Partidos</h2><span class="sub">Hora de Honduras</span></div>
      <div class="days" id="dayChips" role="group" aria-label="Día"></div>
      <div class="fxlist" id="fxList"></div>
      <details class="manual" id="manual">
        <summary>Elegir equipos a mano</summary>
        <div class="pick">
          <label>Local<select id="home"></select></label>
          <label>Visitante<select id="away"></select></label>
        </div>
      </details>
    </section>
  </aside>

  <main class="main">
    <section class="hero" aria-label="Análisis del partido">
      <div class="meta" id="hMeta"></div>
      <div class="teams">
        <div class="tm home"><div class="nm" id="hn"></div><div class="form" id="hf"></div></div>
        <div class="xgbox" aria-label="Goles esperados"><span id="xh"></span><span class="d">–</span><span id="xa"></span></div>
        <div class="tm away"><div class="nm" id="an"></div><div class="form" id="af"></div></div>
      </div>
      <div class="xgcap">Goles esperados: un promedio, no un marcador</div>
      <div>
        <div class="barwrap">
          <div class="stack" id="stack" role="img" aria-label="Probabilidad de victoria local, empate y victoria visitante"><span class="h" id="sH"></span><span class="x" id="sD"></span><span class="a" id="sA"></span></div>
          <div class="ticks" id="ticks"></div>
        </div>
        <div class="legend">
          <div class="l1"><b id="lHn"></b><span><i style="background:var(--home)"></i><span id="lH"></span></span></div>
          <div class="l2"><b>Empate</b><span><i style="background:var(--draw)"></i><span id="lD"></span></span></div>
          <div class="l3"><b id="lAn"></b><span><i style="background:var(--away)"></i><span id="lA"></span></span></div>
        </div>
        <div class="sub" id="mktRef" style="margin-top:10px"></div>
      </div>
      <div class="chips" id="chips"></div>
      <details class="how" id="how"><summary>Cómo se calculó</summary><div class="body" id="howTxt"></div></details>
      <details class="how"><summary>Lesionados y ajuste manual</summary>
        <div class="inj" id="inj"></div>
        <div class="adj">
          <span class="lbl">Si hay bajas importantes, puedes bajar el ataque del equipo. Es una opinión tuya y no está validada.</span>
          <div class="row">
            <label>Local<select id="adjH"><option value="0">Sin ajuste</option><option value="1">1 baja clave (−7% ataque)</option><option value="2">2 bajas clave (−14%)</option><option value="3">3 bajas clave (−21%)</option></select></label>
            <label>Visitante<select id="adjA"><option value="0">Sin ajuste</option><option value="1">1 baja clave (−7% ataque)</option><option value="2">2 bajas clave (−14%)</option><option value="3">3 bajas clave (−21%)</option></select></label>
          </div>
        </div>
      </details>
    </section>

    <section class="panel" aria-label="Mercados y cuotas">
      <h2>Mercados y cuotas</h2>
      <div class="callout"><b>El modelo no supera a las casas de apuestas.</b> Úsalo como referencia. Un EV positivo no es una señal de compra.</div>
      <div class="sub" id="oddsSrc"></div>
      <div class="sub">Escribe la cuota de tu casa en cualquier fila para ver la ganancia esperada (EV) por cada L 1 apostado.</div>
      <div id="mk"></div>
      <details class="how"><summary>Qué tan confiable es el modelo</summary><div class="body">
        <p>Prueba con 2.300 partidos pasados de Bundesliga, Premier y La Liga: el modelo no superó a las casas de apuestas. Las apuestas donde el modelo veía "valor" perdieron entre 15% y 30%, y fue peor cuanto más grande era el supuesto valor.</p>
        <p>El modelo subestima a los favoritos fuertes: cuando las casas daban 70–80%, el modelo daba cerca de 69% y ganaron cerca de 81%. Córners y tiros salieron bien calibrados.</p>
        <p>Si el modelo y la cuota difieren por 8 puntos o más, casi siempre el error es del modelo.</p>
      </div></details>
    </section>

    <section class="panel" aria-label="Bitácora">
      <h2>Bitácora de apuestas</h2>
      <div class="stats" id="stats"></div>
      <div class="log" id="log"></div>
      <div class="row">
        <button class="btn ghost" id="copy" type="button">Copiar bitácora</button>
        <span class="foot" id="copyMsg"></span>
      </div>
      <div class="foot">La bitácora se guarda solo en este navegador. Si borras los datos del sitio se pierde, así que copia el texto de vez en cuando.</div>
    </section>

    <p class="foot" id="fxNote"></p>
    <p class="foot">Scout Picks usa estadísticas de equipos (goles, córners, tiros) de las últimas temporadas. No considera alineaciones, lesiones ni árbitro. Apuesta solo lo que puedas perder y no intentes recuperar pérdidas.</p>
  </main>
</div>

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
function render(){
  const h=$("home").value,a=$("away").value,L=DATA.leagues[league];
  $("upd").innerHTML=(L.kind==="elo"?"Ranking Elo del "+esc(L.updated):"Datos hasta "+esc(L.updated))+"<br>Actualizado "+esc((DATA.generated||"—").slice(0,16));
  if(h===a){$("mk").innerHTML=`<div class="empty">Elige dos equipos distintos.</div>`;return;}
  const fh=1-0.07*(+$("adjH").value), fa=1-0.07*(+$("adjA").value);
  const fx=(DATA.fixtures||[]).find(f=>f.lg===league&&f.h===h&&f.a===a);
  cur=compute(h,a,fh,fa,fx&&fx.ml?fx.ml:null);
  $("hMeta").textContent=L.name+(fx?", "+dtxt(fx.date):"");
  $("hn").textContent=h; $("an").textContent=a;
  $("hf").innerHTML=formHtml(h); $("af").innerHTML=formHtml(a);
  $("inj").innerHTML=L.kind==="elo"?`<div class="none">Champions League: las probabilidades salen del ranking Elo de los clubes (sin córners ni tiros, porque no hay datos comparables entre ligas). No están validadas con partidos pasados.</div>`:injHtml(h,a);
  $("xh").textContent=cur.lh.toFixed(2); $("xa").textContent=cur.la.toFixed(2);
  const [pH,pD,pA]=cur.p;
  const seg=(id,p)=>{const e=$(id);e.style.flexGrow=Math.max(p,0.001);e.textContent=p>=0.12?pct(p):"";};
  seg("sH",pH);seg("sD",pD);seg("sA",pA);
  $("lHn").textContent=h; $("lAn").textContent=a;
  $("lH").textContent=pct(pH)+" de ganar"; $("lD").textContent=pct(pD); $("lA").textContent=pct(pA)+" de ganar";
  const o=fx&&fx.odds; let m=null;
  if(o&&o["1"]&&o["X"]&&o["2"]){const q=[1/o["1"],1/o["X"],1/o["2"]],t=q[0]+q[1]+q[2];m=q.map(x=>x/t);}
  $("ticks").innerHTML=m?`<div class="tick" style="left:${(m[0]*100).toFixed(1)}%"></div><div class="tick" style="left:${((m[0]+m[1])*100).toFixed(1)}%"></div>`:"";
  $("mktRef").textContent=m?`Las marcas punteadas muestran lo que dan las casas de apuestas, sin margen: ${h} ${pct(m[0])}, empate ${pct(m[1])}, ${a} ${pct(m[2])}.`
    :(L.kind==="elo"?"Champions League: la fuente de datos no trae cuotas, así que no hay marcas de comparación.":"No hay cuotas de las casas para este partido, así que no hay marcas de comparación.");
  const how=[];
  if(L.kind==="elo") how.push("Las probabilidades salen del ranking Elo de los clubes. No están validadas con partidos pasados de Champions, así que el error puede ser mayor que en las ligas.");
  else if(m) how.push("En las pruebas con partidos pasados, las casas de apuestas acertaron más que el modelo. Úsalas como referencia principal.");
  else how.push("Estas probabilidades son del modelo estadístico (goles de las últimas temporadas). Sin cuotas de las casas no hay con qué compararlas.");
  if(cur.fromMkt) how.push("Resultado, goles, más o menos y ambos anotan están calculados a partir de las cuotas de las casas; en las pruebas salieron más precisos que los del modelo. Córners y tiros siguen siendo del modelo.");
  if(!cur.fromMkt&&L.kind!=="elo") how.push("Rango típico de error del modelo: ±8 puntos en el 1X2 (en 2.300 partidos pasados, la diferencia con las casas fue menor a eso el 80% de las veces).");
  $("howTxt").innerHTML=how.map(x=>`<p>${esc(x)}</p>`).join("");
  $("chips").innerHTML=[["Más de 2.5 goles",pct(cur.o25)],["Ambos anotan",pct(cur.bt)],...cur.chips.map(c=>[c[0]+" esperados",c[1]])]
    .map(c=>`<div class="chip">${c[0]}<b>${c[1]}</b></div>`).join("");
  let html="",sec="",sub=null;
  cur.rows.forEach((r,i)=>{
    const parts=r.sec.split(" · "), g=parts[0], sb=parts[1]||null;
    if(g!==sec){
      if(sec)html+="</details>";
      sec=g; sub=null; const n=cur.rows.filter(x=>x.sec.split(" · ")[0]===sec).length;
      html+=`<details class="grp"${(sec==="Resultado"||sec==="Goles")?" open":""}><summary>${esc(sec)}<span class="cnt">${n}</span></summary>`;
    }
    if(sb&&sb!==sub){sub=sb; html+=`<div class="subh">${esc(sb==="total"?"Total del partido":sb)}</div>`;}
    html+=`<div class="mrow"><div class="mtop" style="--p:${(r.p*100).toFixed(0)}%"><span class="ml">${esc(r.label)}</span><span class="mp">${pct(r.p)}</span></div>
      <div class="mbot"><span class="mf">Cuota justa ${f2(1/r.p)}</span><input inputmode="decimal" id="o${i}" data-i="${i}" placeholder="Tu cuota" aria-label="Cuota ${esc(r.label)}"><span class="ev" id="e${i}"></span><button class="mini" id="b${i}" data-i="${i}" disabled type="button">Anotar</button></div>
      <div class="frow" id="f${i}"></div></div>`;
  });
  if(sec)html+="</details>";
  $("mk").innerHTML=html;
  let n=0;
  if(fx&&fx.odds){
    cur.rows.forEach((r,i)=>{const v=r.key&&fx.odds[r.key]; if(v){$("o"+i).value=v;onOdds(i);n++;}});
  }
  $("oddsSrc").textContent=n?`Cuotas cargadas automáticamente (${n}): promedio de varias casas (football-data.co.uk). Solo vienen 1X2 y más/menos 2.5. Cámbialas por las de tu casa.`:"";
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
function fxList(){
  const days=daysOfLeague(), list=fxOfLeague().filter(f=>f.date.slice(0,10)===day);
  $("fxNote").textContent=DATA.note||"";
  $("dayChips").innerHTML=days.map(k=>{const p=dparts(k),n=fxOfLeague().filter(f=>f.date.slice(0,10)===k).length;
    return `<button class="day" type="button" data-d="${k}" aria-pressed="${k===day}">${p.dow} ${p.d} ${p.mon}<small>${n} ${n===1?"partido":"partidos"}</small></button>`;}).join("");
  const h=$("home").value,a=$("away").value;
  $("fxList").innerHTML=list.length?list.map(f=>`<button class="fx" type="button" data-f="${DATA.fixtures.indexOf(f)}" aria-current="${f.h===h&&f.a===a}"><span class="tm">${esc(f.date.slice(11,16))}</span><span class="tt"><span>${esc(f.h)}</span><span class="v">${esc(f.a)}</span></span></button>`).join("")
    :`<div class="empty">No hay partidos próximos de ${esc(DATA.leagues[league].name)} en la fuente de datos. Elige los equipos a mano.</div>`;
  $("manual").open=!list.length;
}
function pickMatch(f){
  $("home").value=f.h; $("away").value=f.a; $("adjH").value="0"; $("adjA").value="0"; render(); fxList();
}
$("fxList").addEventListener("click",ev=>{
  const b=ev.target.closest("button[data-f]"); if(!b)return;
  pickMatch(DATA.fixtures[+b.dataset.f]);
  if(window.matchMedia("(max-width:959px)").matches)
    document.querySelector(".hero").scrollIntoView({behavior:window.matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth",block:"start"});
});
$("dayChips").addEventListener("click",ev=>{
  const b=ev.target.closest("button[data-d]"); if(!b)return;
  day=b.dataset.d; const first=fxOfLeague().find(f=>f.date.slice(0,10)===day);
  if(first)pickMatch(first); else fxList();
});

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

$("adjH").addEventListener("change",render); $("adjA").addEventListener("change",render);
$("mk").addEventListener("input",ev=>{if(ev.target.dataset.i!==undefined&&ev.target.tagName==="INPUT")onOdds(+ev.target.dataset.i);});
$("mk").addEventListener("click",ev=>{if(ev.target.tagName==="BUTTON"&&!ev.target.disabled)addPick(+ev.target.dataset.i);});
const sync=()=>{render();fxList();}; $("home").addEventListener("change",sync); $("away").addEventListener("change",sync);

function tabs(){
  $("leagueTabs").innerHTML=Object.entries(DATA.leagues).map(([k,L])=>
    `<button class="tab" type="button" data-k="${k}" aria-pressed="${k===league}">${esc(L.name)}</button>`).join("");
}
$("leagueTabs").addEventListener("click",ev=>{
  const k=ev.target.dataset.k; if(!k||k===league)return;
  league=k; day=null; tabs(); fillTeams(); $("adjH").value="0"; $("adjA").value="0"; render(); fxList();
});
league=(DATA.fixtures&&DATA.fixtures[0]&&DATA.fixtures[0].lg)||Object.keys(DATA.leagues)[0];
tabs(); fillTeams(); render(); fxList(); renderLog();
</script>

</body></html>
'''


if __name__ == "__main__":
    build(TEMPLATE)
