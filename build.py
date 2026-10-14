"""Picks Fresh: descarga datos, ajusta el modelo, consulta la API y genera docs/index.html."""
import io, os, re, sys, json, difflib, unicodedata, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

import uclhist

LEAGUES = {
    "bundesliga": dict(name="Bundesliga", code="D1"),
    "premier": dict(name="Premier League", code="E0"),
    "laliga": dict(name="La Liga", code="SP1"),
    # ligas nuevas: solo goles (y en Holanda también córners, tiros y tarjetas); el modelo no se ha validado con partidos pasados
    "eredivisie": dict(name="Eredivisie (Holanda)", code="N1", tbl="euro", new=True),
    "noruega": dict(name="Eliteserien (Noruega)", code="NOR", src="fdnew", cc="NOR", tbl="cal", new=True),
    "arabia": dict(name="Liga de Arabia", code="KSA", src="espn", slug="ksa.1", tbl="euro", new=True),
    "mls": dict(name="MLS (EE. UU.)", code="USA", src="fdnew", cc="USA", tbl="cal", new=True),
    "ligamx": dict(name="Liga MX (México)", code="MEX", src="fdnew", cc="MEX", tbl="split", new=True),
}
BT_W = 0.5   # peso del modelo en "ambos anotan" (el resto va a la tasa histórica de la liga); ver prueba con partidos pasados
REG_M = {"shots": 80.0}   # regularización por mercado (por defecto 40); tiros totales mejoró con 80 en la prueba con partidos pasados
MARKETS = {"corners": ("HC", "AC"), "sot": ("HST", "AST"), "shots": ("HS", "AS")}
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
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


NEW_DIR = Path(__file__).parent / "data" / "raw"   # solo para pruebas sin internet


def fetch_new(v, today, text=None):
    """CSV de 'ligas extra' de football-data.co.uk (un archivo por país, todas las temporadas). Solo goles y cuotas de cierre."""
    if text is None:
        r = requests.get(f"https://www.football-data.co.uk/new/{v['cc']}.csv", timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = r.content.decode("utf-8-sig", errors="replace")
    d = pd.read_csv(io.StringIO(text))
    d.columns = [str(c).strip() for c in d.columns]
    d = d.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
    if "League" in d.columns and len(d):
        d = d[d.League == d.League.iloc[-1]]
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
    d = d.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return [d[d.Date >= pd.Timestamp(today) - pd.Timedelta(days=1100)][["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]]


def espn_hist(key, v, today, evs=None):
    """Historial de una liga que solo está en ESPN (Arabia): marcadores finales de este año y los dos anteriores. Si ESPN falla, usa la copia guardada."""
    cache = Path(__file__).parent / "data" / f"espn_{key}.csv"
    rows = []
    if evs is None:
        for yr in range(today.year - 2, today.year + 1):
            js = _get_json("ESPN", f"https://site.api.espn.com/apis/site/v2/sports/soccer/{v['slug']}/scoreboard?dates={yr}&limit=1000")
            if js:
                rows += parse_espn(js)
    else:
        rows = evs
    df = pd.DataFrame([{"Date": pd.to_datetime(str(r[0])), "HomeTeam": r[1], "AwayTeam": r[2], "FTHG": r[3], "FTAG": r[4]} for r in rows])
    if len(df) >= 100:
        try:
            df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).to_csv(cache, index=False)
        except Exception:  # noqa: BLE001
            pass
    elif cache.exists():
        log(f"  ESPN {key}: no respondió bien, se usa la copia guardada")
        df = pd.read_csv(cache, parse_dates=["Date"])
    if not len(df):
        raise RuntimeError("sin datos de ESPN")
    return [df]


def load_league(key, v, seasons, today):
    src = v.get("src", "fd")
    if src == "fd":
        return prepare(manual_frames(v["code"]) + fetch_frames(v["code"], seasons), cal=v.get("tbl") == "cal")
    if src == "fdnew":
        return prepare(manual_frames(v["code"]) + fetch_new(v, today), cal=v.get("tbl") == "cal")
    return prepare(manual_frames(v["code"]) + espn_hist(key, v, today))


def prepare(frames, cal=False):
    ok = [d for d in frames if {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(d.columns)]
    if not ok:
        raise RuntimeError("sin datos")
    df = pd.concat(ok, ignore_index=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last").copy()
    y = df.Date.dt.year if cal else df.Date.dt.year.where(df.Date.dt.month >= 7, df.Date.dt.year - 1)
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


def rho_dc(gh, ga, lh, la, w):
    """Parámetro de Dixon-Coles (marcadores 0-0, 1-0, 0-1, 1-1) por máxima verosimilitud, con los mismos pesos por antigüedad."""
    best, top = 0.0, -1e18
    for rho in np.arange(-0.25, 0.1001, 0.005):
        t = np.ones_like(lh)
        t = np.where((gh == 0) & (ga == 0), 1 - lh * la * rho, t)
        t = np.where((gh == 0) & (ga == 1), 1 + lh * rho, t)
        t = np.where((gh == 1) & (ga == 0), 1 + la * rho, t)
        t = np.where((gh == 1) & (ga == 1), 1 - rho, t)
        if np.any(t <= 0):
            continue
        v = float(np.sum(w * np.log(t)))
        if v > top:
            best, top = float(rho), v
    return round(best, 3)


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
    out_rho = {"rho": rho_dc(gh, ga, lh, la, w)} if hcol == "FTHG" else {}
    return {
        **out_rho,
        "mu": float(x[0]), "home": float(x[1]),
        "ah": _disp(gh, lh), "aa": _disp(ga, la), "at": _disp(gh + ga, lh + la),
        "att": {t: round(float(att[i]), 4) for i, t in enumerate(teams)},
        "de": {t: round(float(de[i]), 4) for i, t in enumerate(teams)},
    }


def league_params(df, ref):
    models = {"goles": fit_model(df, ref, 365, 4.0, "FTHG", "FTAG")}
    rec = df[df.Date >= ref - pd.Timedelta(days=730)]   # tasa histórica de "ambos anotan" (últimos 2 años) para encoger esa probabilidad
    models["goles"]["btb"] = round(float(((rec.FTHG > 0) & (rec.FTAG > 0)).mean()), 3)
    models["goles"]["btw"] = BT_W
    log(f"  Dixon-Coles rho = {models['goles']['rho']} | ambos anotan base = {models['goles']['btb']}")
    for k, (h, a) in MARKETS.items():
        if h in df.columns and a in df.columns:
            models[k] = fit_model(df, ref, 365, REG_M.get(k, 40.0), h, a)
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
            "teams": teams, "form": form, "hteams": ht, "hm": hm}


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


# ------------------------------------------------------------------ resultados extra
OF_FILES = {"bundesliga": ("deutschland", "1-bundesliga.txt"), "premier": ("england", "1-premierleague.txt"),
            "laliga": ("espana", "1-liga.txt")}
OF_ALIAS = {"fc bayern munchen": "Bayern Munich", "borussia dortmund": "Dortmund", "1 fc koln": "FC Koln",
            "borussia monchengladbach": "M'gladbach", "eintracht frankfurt": "Ein Frankfurt", "sv 07 elversberg": "Elversberg",
            "bayer 04 leverkusen": "Leverkusen", "1 fsv mainz 05": "Mainz", "sc paderborn 07": "Paderborn",
            "tsg 1899 hoffenheim": "Hoffenheim", "hamburger sv": "Hamburg", "fc schalke 04": "Schalke 04",
            "1 fc union berlin": "Union Berlin", "sv werder bremen": "Werder Bremen", "vfb stuttgart": "Stuttgart",
            "sc freiburg": "Freiburg", "fc augsburg": "Augsburg", "rb leipzig": "RB Leipzig",
            "manchester united fc": "Man United", "manchester city fc": "Man City", "tottenham hotspur fc": "Tottenham",
            "newcastle united fc": "Newcastle", "nottingham forest fc": "Nott'm Forest", "brighton hove albion fc": "Brighton",
            "afc bournemouth": "Bournemouth", "leeds united fc": "Leeds", "hull city afc": "Hull", "ipswich town fc": "Ipswich",
            "coventry city fc": "Coventry", "sunderland afc": "Sunderland", "crystal palace fc": "Crystal Palace",
            "everton fc": "Everton", "fulham fc": "Fulham", "chelsea fc": "Chelsea", "arsenal fc": "Arsenal",
            "liverpool fc": "Liverpool", "aston villa fc": "Aston Villa", "brentford fc": "Brentford",
            "malaga cf": "Malaga", "rcd espanyol de barcelona": "Espanol", "rayo vallecano de madrid": "Vallecano",
            "athletic club": "Ath Bilbao", "deportivo alaves": "Alaves", "club atletico de madrid": "Ath Madrid",
            "fc barcelona": "Barcelona", "real madrid cf": "Real Madrid", "villarreal cf": "Villarreal", "elche cf": "Elche",
            "rc celta de vigo": "Celta", "real sociedad de futbol": "Sociedad", "rc deportivo la coruna": "La Coruna",
            "real betis balompie": "Betis", "ca osasuna": "Osasuna", "real racing club de santander": "Santander",
            "valencia cf": "Valencia", "levante ud": "Levante", "sevilla fc": "Sevilla", "getafe cf": "Getafe"}
ALIAS.update(OF_ALIAS)


def manual_frames(code):
    """Resultados cargados a mano (data/resultados_manuales.csv). Si la fuente principal ya trae el partido, gana la fuente principal."""
    p = Path(__file__).parent / "data" / "resultados_manuales.csv"
    try:
        m = pd.read_csv(p, encoding="utf8")
    except Exception as e:  # noqa: BLE001
        log(f"  resultados manuales: no se pudo leer ({e})")
        return []
    m = m[m.Div.astype(str).str.strip() == code].drop(columns=["Div", "fuente"], errors="ignore")
    if len(m):
        log(f"  {code}: {len(m)} resultados manuales")
    return [m] if len(m) else []


def fill_missing(key, df, rows):
    """rows: [(aaaammdd, local, visitante, goles_local, goles_visitante, ...)]. Devuelve un DataFrame con los partidos que
    no están en df (mismo local y visitante a menos de 3 días), con los nombres ya convertidos a los de football-data."""
    cur = df[df.season == df.season.max()]
    teams = set(cur.HomeTeam) | set(cur.AwayTeam)
    have = {}
    for d, h, a in zip(cur.Date, cur.HomeTeam, cur.AwayTeam):
        have.setdefault((h, a), []).append(d)
    out, bad, seen = [], set(), set()
    for r in rows:
        d, h, a, x, y2 = r[:5]
        mh, ma = match_team(h, teams), match_team(a, teams)
        if not mh or not ma or mh == ma:
            bad.add(h if not mh else a)
            continue
        dd = pd.to_datetime(str(d))
        if any(abs((dd - e).days) <= 3 for e in have.get((mh, ma), [])) or (mh, ma) in seen:
            continue
        seen.add((mh, ma))
        out.append({"Date": dd, "HomeTeam": mh, "AwayTeam": ma, "FTHG": int(x), "FTAG": int(y2)})
    if bad:
        log(f"  {key}: sin equivalencia de nombre: {', '.join(sorted(bad))}")
        EXTRA_BAD.extend(f"{b} ({key})" for b in sorted(bad) if f"{b} ({key})" not in EXTRA_BAD)
    return pd.DataFrame(out) if out else None


def of_extra(key, df, today, text=None):
    """Partidos que faltan en football-data y sí están en openfootball (solo goles). Devuelve un DataFrame o None."""
    if key not in OF_FILES:
        return None
    repo, fn = OF_FILES[key]
    y = today.year if today.month >= 7 else today.year - 1
    if text is None:
        for br in ("master", "main"):
            try:
                r = requests.get(f"https://raw.githubusercontent.com/openfootball/{repo}/{br}/{y}-{(y + 1) % 100:02d}/{fn}", timeout=40)
                r.raise_for_status()
                text = r.text
                break
            except Exception as e:  # noqa: BLE001
                log(f"  openfootball {repo} {br}: {str(e)[:100]}")
        if text is None:
            return None
    ex = fill_missing(key, df, uclhist.parse(text, y, raw=True))
    if ex is not None:
        EXTRA_N["openfootball"] = EXTRA_N.get("openfootball", 0) + len(ex)
        log(f"  openfootball {key}: {len(ex)} partidos que faltaban (solo goles)")
    return ex


# Fuentes gratuitas que publican el marcador el mismo día (solo goles); football-data.co.uk trae después las estadísticas completas.
EXTRA_N, EXTRA_ERR, EXTRA_BAD = {}, [], []
ESPN_LG = {"bundesliga": "ger.1", "premier": "eng.1", "laliga": "esp.1", "ucl": "uefa.champions", "eredivisie": "ned.1", "noruega": "nor.1",
           "arabia": "ksa.1", "mls": "usa.1", "ligamx": "mex.1"}
OLDB_LG = {"bundesliga": "bl1"}
FDORG_LG = {"bundesliga": "BL1", "premier": "PL", "laliga": "PD", "ucl": "CL", "eredivisie": "DED"}
UCL_STAGE = {"league-phase": "Fase de liga", "playoffs": "Play-offs", "knockout-round-playoffs": "Play-offs",
             "round-of-16": "Octavos", "quarterfinals": "Cuartos", "semifinals": "Semifinales", "final": "Final"}
UCL_ALIAS = {"sporting cp": "Sporting Lisbon", "inter": "Inter Milan", "internazionale": "Inter Milan", "fc copenhagen": "København",
             "red bull salzburg": "Salzburg", "qarabag": "Qarabağ", "union st gilloise": "Union Saint-Gilloise",
             "bayer leverkusen": "Bayer Leverkusen", "atletico madrid": "Atlético Madrid", "psv": "PSV Eindhoven",
             "bodo glimt": "Bodø/Glimt", "fk bodo glimt": "Bodø/Glimt", "paris saint germain": "Paris Saint-Germain"}


def _ymd(iso):
    return int(str(iso)[:10].replace("-", ""))


def parse_espn(js):
    """Marcadores finales del scoreboard público de ESPN (no oficial)."""
    out = []
    for ev in (js or {}).get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        st = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        if not st.get("completed"):
            continue
        cs = comp.get("competitors") or []
        H = [c for c in cs if c.get("homeAway") == "home"]
        A = [c for c in cs if c.get("homeAway") == "away"]
        if not H or not A:
            continue
        try:
            hg, ag = int(float(H[0].get("score"))), int(float(A[0].get("score")))
        except (TypeError, ValueError):
            continue
        nm = lambda c: (c.get("team") or {}).get("displayName") or (c.get("team") or {}).get("name") or ""
        slug = ((ev.get("season") or {}).get("slug") or "")
        out.append((_ymd(ev.get("date", "")), nm(H[0]), nm(A[0]), hg, ag, UCL_STAGE.get(slug, "Fase de liga")))
    return out


def parse_openliga(js):
    """Marcadores finales de OpenLigaDB (Bundesliga). resultTypeID 2 = resultado final."""
    out = []
    for m in js or []:
        if not m.get("matchIsFinished"):
            continue
        res = [r for r in (m.get("matchResults") or []) if r.get("resultTypeID") == 2] or \
              sorted(m.get("matchResults") or [], key=lambda r: r.get("resultOrderID", 0))[-1:]
        if not res:
            continue
        out.append((_ymd(m.get("matchDateTimeUTC") or m.get("matchDateTime") or ""), (m.get("team1") or {}).get("teamName", ""),
                    (m.get("team2") or {}).get("teamName", ""), int(res[0]["pointsTeam1"]), int(res[0]["pointsTeam2"]), ""))
    return out


def parse_fdorg(js):
    """Marcadores finales de football-data.org v4 (necesita clave gratuita)."""
    out = []
    for m in (js or {}).get("matches", []) or []:
        if m.get("status") != "FINISHED":
            continue
        ft = ((m.get("score") or {}).get("fullTime") or {})
        if ft.get("home") is None or ft.get("away") is None:
            continue
        out.append((_ymd(m.get("utcDate", "")), (m.get("homeTeam") or {}).get("name", ""), (m.get("awayTeam") or {}).get("name", ""),
                    int(ft["home"]), int(ft["away"]), ""))
    return out


TSDB_LG = {"bundesliga": 4331, "premier": 4328, "laliga": 4335, "ucl": 4480, "eredivisie": 4337, "noruega": 4358, "arabia": 4668, "mls": 4346, "ligamx": 4350}
ALIAS.update({
    # MLS (ESPN -> football-data)
    "red bull new york": "New York Red Bulls", "la galaxy": "Los Angeles Galaxy", "lafc": "Los Angeles FC", "atlanta united fc": "Atlanta Utd",
    "st louis city sc": "St. Louis City", "d c united": "DC United", "charlotte fc": "Charlotte", "cf montreal": "CF Montreal",
    "chicago fire fc": "Chicago Fire", "houston dynamo fc": "Houston Dynamo", "inter miami cf": "Inter Miami", "minnesota united fc": "Minnesota United",
    "new york city fc": "New York City", "orlando city sc": "Orlando City", "seattle sounders fc": "Seattle Sounders",
    # Liga MX
    "america": "Club America", "guadalajara": "Guadalajara Chivas", "leon": "Club Leon", "tijuana": "Club Tijuana", "pumas unam": "UNAM Pumas",
    "santos": "Santos Laguna", "fc juarez": "Juarez", "atletico de san luis": "Atl. San Luis", "queretaro": "Queretaro",
    # Noruega
    "sk brann": "Brann", "viking fk": "Viking", "sarpsborg fk": "Sarpsborg 08", "kristiansund bk": "Kristiansund", "ik start": "Start",
    "hamarkameratene": "HamKam", "valerenga": "Valerenga", "kfum oslo": "KFUM Oslo",
    # Holanda
    "ado den haag": "Den Haag", "fortuna sittard": "For Sittard", "sc cambuur": "Cambuur", "ajax amsterdam": "Ajax", "feyenoord rotterdam": "Feyenoord", "fc groningen": "Groningen", "fc twente": "Twente", "fc utrecht": "Utrecht",
    "pec zwolle": "Zwolle", "heracles almelo": "Heracles", "sparta rotterdam": "Sparta Rotterdam", "nac breda": "NAC Breda", "nec nijmegen": "Nijmegen",
    "willem ii": "Willem II", "fc volendam": "Volendam", "go ahead eagles": "Go Ahead Eagles",
    "spurs": "Tottenham", "man utd": "Man United", "racing santander": "Santander", "deportivo la coruna": "La Coruna"})


def parse_fpl(fixtures, boot):
    """Marcadores finales de la API pública de la Fantasy Premier League (solo Premier League)."""
    nm = {t["id"]: t["name"] for t in (boot or {}).get("teams", [])}
    out = []
    for f in fixtures or []:
        if not (f.get("finished") or f.get("finished_provisional")):
            continue
        if f.get("team_h_score") is None or f.get("team_a_score") is None or f.get("team_h") not in nm or f.get("team_a") not in nm:
            continue
        out.append((_ymd(f.get("kickoff_time") or ""), nm[f["team_h"]], nm[f["team_a"]], int(f["team_h_score"]), int(f["team_a_score"]), ""))
    return out


def parse_tsdb(js):
    """Marcadores finales de TheSportsDB (clave gratuita pública)."""
    out = []
    for e in (js or {}).get("events") or []:
        stt = str(e.get("strStatus") or "").lower()
        if not (stt in ("match finished", "ft", "aet", "finished", "after extra time") or "finish" in stt):
            continue
        try:
            hg, ag = int(e["intHomeScore"]), int(e["intAwayScore"])
        except (TypeError, ValueError, KeyError):
            continue
        out.append((_ymd(e.get("dateEvent") or ""), e.get("strHomeTeam") or "", e.get("strAwayTeam") or "", hg, ag, ""))
    return out


def _get_json(label, url, headers=None):
    try:
        r = requests.get(url, timeout=30, headers=headers or {"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        why = getattr(getattr(e, "response", None), "status_code", None) or type(e).__name__
        log(f"  {label}: {str(e)[:120]}")
        if not any(x.startswith(label) for x in EXTRA_ERR):
            EXTRA_ERR.append(f"{label} ({why})")
        return None


def live_rows(key, since, today):
    """[(fuente, filas)] con los partidos terminados entre `since` y hoy, de cada fuente gratuita disponible."""
    out, d0, d1 = [], int(since.strftime("%Y%m%d")), int(today.strftime("%Y%m%d"))
    y = today.year if today.month >= 7 else today.year - 1
    inr = lambda rows: [r for r in rows if d0 <= r[0] <= d1 + 1]
    k = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
    if k and key in FDORG_LG:
        js = _get_json("football-data.org", f"https://api.football-data.org/v4/competitions/{FDORG_LG[key]}/matches?status=FINISHED"
                       f"&dateFrom={since:%Y-%m-%d}&dateTo={today + dt.timedelta(days=1):%Y-%m-%d}", {"X-Auth-Token": k})
        out.append(("football-data.org", inr(parse_fdorg(js))))
    if key in OLDB_LG:
        js = _get_json("OpenLigaDB", f"https://api.openligadb.de/getmatchdata/{OLDB_LG[key]}/{y}")
        out.append(("OpenLigaDB", inr(parse_openliga(js))))
    if key == "premier":
        fx = _get_json("FPL", "https://fantasy.premierleague.com/api/fixtures/")
        bt = _get_json("FPL", "https://fantasy.premierleague.com/api/bootstrap-static/") if fx else None
        out.append(("FPL", inr(parse_fpl(fx, bt))))
    if key in TSDB_LG:
        js = _get_json("TheSportsDB", f"https://www.thesportsdb.com/api/v1/json/3/eventspastleague.php?id={TSDB_LG[key]}")
        out.append(("TheSportsDB", inr(parse_tsdb(js))))
    if key in ESPN_LG:
        rows, day = [], dt.date(d0 // 10000, d0 // 100 % 100, d0 % 100)
        while day <= today:
            js = _get_json("ESPN", f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN_LG[key]}/scoreboard?dates={day:%Y%m%d}")
            if js is None:
                break
            rows += parse_espn(js)
            day += dt.timedelta(days=1)
        out.append(("ESPN", inr(rows)))
    return out


def add_extra(df, extra):
    if extra is None or not len(extra):
        return df
    y = extra.Date.dt.year.where(extra.Date.dt.month >= 7, extra.Date.dt.year - 1)
    extra = extra.copy()
    extra["season"] = (y % 100).astype(str).str.zfill(2) + ((y + 1) % 100).astype(str).str.zfill(2)
    return pd.concat([df, extra], ignore_index=True).sort_values("Date").reset_index(drop=True)


def recent_results(params, uclh, today, days=3):
    """Partidos ya jugados en los últimos días, para mostrar su marcador final en la lista."""
    cut = int((today - dt.timedelta(days=days)).strftime("%Y%m%d"))
    out = []
    for key, P in params.items():
        if key == "ucl":
            rows = [(r[0], r[1], r[2], r[3], r[4]) for r in uclh if r[0] >= cut]
            names = None
        else:
            ht = P["hteams"]
            rows = [(r[0], ht[r[1]], ht[r[2]], r[3], r[4]) for r in P["hm"] if r[0] >= cut]
        for d, h, a, x, y in rows:
            ds = str(d)
            out.append({"id": f"res-{key}-{d}-{h}", "lg": key, "date": f"{ds[:4]}-{ds[4:6]}-{ds[6:]}T00:00", "status": "FT",
                        "h": h, "a": a, "odds": {}, "score": [int(x), int(y)]})
    return out


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



def parse_espn_fx(js):
    """Partidos por jugar del scoreboard de ESPN: [(fecha UTC, local, visitante)]."""
    out = []
    for ev in (js or {}).get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        st = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        if st.get("state") != "pre" or st.get("completed"):
            continue
        cs = comp.get("competitors") or []
        H = [c for c in cs if c.get("homeAway") == "home"]
        A = [c for c in cs if c.get("homeAway") == "away"]
        if not H or not A:
            continue
        nm = lambda c: (c.get("team") or {}).get("displayName") or ""
        try:
            d = dt.datetime.strptime(str(ev.get("date"))[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        out.append((d, nm(H[0]), nm(A[0])))
    return out


def extra_fixtures(params, today, nl_text=None, espn_days=None):
    """Próximos partidos de las ligas nuevas: new_league_fixtures.csv de football-data (con cuotas 1X2) y, si falta algo, el scoreboard de ESPN."""
    from zoneinfo import ZoneInfo
    uk, hn = ZoneInfo("Europe/London"), ZoneInfo("America/Tegucigalpa")
    now = dt.datetime.now(hn)
    out, seen = [], set()

    def add(key, loc, h, a, odds):
        if not (today <= loc.date() <= today + dt.timedelta(days=7)) or loc < now - dt.timedelta(hours=2):
            return
        teams = params[key]["teams"]
        mh, ma = match_team(h, teams), match_team(a, teams)
        if not mh or not ma or mh == ma:
            log(f"  sin equivalencia: {h} / {a} ({key})")
            if key not in ("ucl",) and f"{h}/{a}" not in EXTRA_BAD:
                EXTRA_BAD.append(f"{h}/{a}")
            return
        if (key, mh, ma) in seen:
            return
        seen.add((key, mh, ma))
        out.append({"id": 0, "lg": key, "date": loc.strftime("%Y-%m-%dT%H:%M"), "status": "NS", "h": mh, "a": ma, "odds": odds})

    cc = {v["cc"]: k for k, v in LEAGUES.items() if v.get("src") == "fdnew"}
    country = {"NOR": "Norway", "USA": "USA", "MEX": "Mexico"}
    try:
        if nl_text is None:
            r = requests.get("https://www.football-data.co.uk/new_league_fixtures.csv", timeout=40, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            nl_text = r.content.decode("utf-8-sig", errors="replace")
        nf = pd.read_csv(io.StringIO(nl_text))
        nf.columns = [str(c).strip() for c in nf.columns]
        for _, row in nf.iterrows():
            key = next((k for c, k in cc.items() if country[c] == str(row.get("Country", "")).strip()), None)
            if not key or key not in params:
                continue
            d = pd.to_datetime(row.get("Date"), dayfirst=True, errors="coerce")
            if pd.isna(d):
                continue
            try:
                hh, mm = [int(x) for x in str(row.get("Time", "12:00")).split(":")[:2]]
            except Exception:  # noqa: BLE001
                hh, mm = 12, 0
            loc = dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=uk).astimezone(hn)
            odds = {}
            for k, cols in {"1": ["AvgH", "B365H"], "X": ["AvgD", "B365D"], "2": ["AvgA", "B365A"]}.items():
                for c in cols:
                    try:
                        v = float(row.get(c))
                    except (TypeError, ValueError):
                        continue
                    if v > 1:
                        odds[k] = round(v, 2)
                        break
            add(key, loc, str(row.get("Home")), str(row.get("Away")), odds)
    except Exception as e:  # noqa: BLE001
        log(f"  new_league_fixtures.csv: {str(e)[:100]}")
    # ESPN: completa lo que falte y es la única fuente de Arabia
    for key, v in LEAGUES.items():
        if not v.get("new") or key not in params or key not in ESPN_LG:
            continue
        for i in range(0, 8):
            day = today + dt.timedelta(days=i)
            js = espn_days(key, day) if espn_days else _get_json("ESPN", f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN_LG[key]}/scoreboard?dates={day:%Y%m%d}")
            if js is None:
                break
            for d, h, a in parse_espn_fx(js):
                add(key, d.astimezone(hn), h, a, {})
    return out


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
                "updated": STATIC_ELO_DATE, "n": 0, "models": {}, "teams": sorted(elo), "form": {}}
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
            "updated": today.isoformat(), "n": 0, "models": {}, "teams": sorted(elo), "form": {}}


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


def load_uclh(today=None):
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
    if today is not None:   # marcadores de Champions del mismo día (ESPN / football-data.org), solo goles
        last = max([r[0] for r in rows + new] or [20260701])
        since = dt.date(last // 10000, last // 100 % 100, last % 100) - dt.timedelta(days=1)
        known = set(UCL_TEAMS) | {r[1] for r in rows + new} | {r[2] for r in rows + new}
        for label, got in live_rows("ucl", since, today):
            add = 0
            for d, h, a, x, y2, st in got:
                hh = UCL_ALIAS.get(norm(h)) or match_team(h, known) or uclhist.clean(h)
                aa = UCL_ALIAS.get(norm(a)) or match_team(a, known) or uclhist.clean(a)
                dd = pd.to_datetime(str(d))
                if hh == aa or any((r[1], r[2]) == (hh, aa) and abs((pd.to_datetime(str(r[0])) - dd).days) <= 3 for r in rows + new):
                    continue
                new.append([d, hh, aa, x, y2, st or "Fase de liga", ""])
                add += 1
            if add:
                EXTRA_N[label + " (Champions)"] = EXTRA_N.get(label + " (Champions)", 0) + add
                log(f"  {label} Champions: {add} partidos que faltaban (solo goles)")
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


def h2h_data(params, uclh):
    """Historial de ~10 temporadas (data/h2h_hist.json) solo para los equipos actuales y solo lo que no cubren ya los datos recientes."""
    try:
        raw = json.loads((Path(__file__).parent / "data" / "h2h_hist.json").read_text(encoding="utf8"))
    except Exception as e:  # noqa: BLE001
        log(f"  historial H2H: no se pudo leer ({e})")
        return None

    def pack(src, keep, extra=None):
        used = sorted({src["t"][m[1]] for m in keep} | {src["t"][m[2]] for m in keep})
        ix = {n: i for i, n in enumerate(used)}
        out = {"t": used, "m": [[m[0], ix[src["t"][m[1]]], ix[src["t"][m[2]]]] + m[3:] for m in keep]}
        if extra:
            out.update(extra)
        return out

    res = {"leagues": {}, "ucl": None}
    for key, P in params.items():
        src = raw.get("leagues", {}).get(key)
        if key == "ucl" or not src or not P.get("hm"):
            continue
        cur, cut = set(P["teams"]), min(r[0] for r in P["hm"])
        keep = [m for m in src["m"] if m[0] < cut and src["t"][m[1]] in cur and src["t"][m[2]] in cur]
        res["leagues"][key] = pack(src, keep)
    src = raw.get("ucl")
    if src and uclh:
        names = set(UCL_TEAMS) | {r[1] for r in uclh} | {r[2] for r in uclh}
        cut = min([r[0] for r in uclh if r[5] != "Previa"] or [99999999])
        # Champions: solo lo anterior a los datos recientes (esos ya vienen en uclh). Europa y Conference no están en uclh, van completas.
        keep = [m for m in src["m"] if (m[0] < cut or (len(m) > 6 and m[6] > 0)) and src["t"][m[1]] in names and src["t"][m[2]] in names]
        res["ucl"] = pack(src, keep, {"s": src["s"], "c": src.get("c", ["Champions League"])})
    n = sum(len(v["m"]) for v in res["leagues"].values()) + (len(res["ucl"]["m"]) if res["ucl"] else 0)
    log(f"  historial H2H: {n} partidos de temporadas anteriores")
    return res


def build(template, out_path="docs/index.html"):
    now = dt.datetime.utcnow()
    local = now + HN
    today = local.date()
    seasons = season_codes(today)
    params = {}
    for key, v in LEAGUES.items():
        log(f"{v['name']}:")
        try:
            df = load_league(key, v, seasons, today)
            df = add_extra(df, of_extra(key, df, today))
            for label, rows in ([] if v.get("src") == "espn" else live_rows(key, max((df.Date.max() - pd.Timedelta(days=3)).date(), today - dt.timedelta(days=45)), today)):
                ex = fill_missing(key, df, rows)
                if ex is not None:
                    EXTRA_N[label] = EXTRA_N.get(label, 0) + len(ex)
                    log(f"  {label} {key}: {len(ex)} partidos que faltaban (solo goles)")
                    df = add_extra(df, ex)
            ref = df.Date.max() + pd.Timedelta(days=1)
            params[key] = league_params(df, ref)
            params[key]["name"] = v["name"]
            params[key]["tbl"] = v.get("tbl", "euro")      # cómo empieza la temporada para la tabla: euro (julio), cal (enero), split (julio o enero)
            params[key]["val"] = not v.get("new")           # True: modelo validado con partidos pasados (solo las 3 ligas originales)
        except Exception as e:  # noqa: BLE001
            log(f"  {v['name']} omitida: {e}")
    if not params:
        raise SystemExit("No se pudo cargar ninguna liga; no se actualiza la página.")
    u = ucl_params(today)
    if u:
        u["map"] = ucl_domestic(params)
        log("  Champions con historial local: " + ", ".join(sorted(u["map"])))
        params["ucl"] = u
    fixtures, note = csv_fixtures(params, today)
    fx2 = extra_fixtures(params, today)
    have = {(f["lg"], f["h"], f["a"]) for f in fixtures}
    fixtures = fixtures + [f for f in fx2 if (f["lg"], f["h"], f["a"]) not in have]
    for i, f in enumerate(fixtures):
        f["id"] = i + 1
    if fx2:
        note += f" Ligas nuevas: {len(fx2)} partidos (football-data y ESPN)."
    for f in fixtures:
        M = params.get(f["lg"], {}).get("models", {}).get("goles") if f.get("odds") else None
        if M and f["h"] in M["att"] and f["a"] in M["att"]:
            st = [np.exp(M["mu"] + M["home"] + M["att"][f["h"]] - M["de"][f["a"]]), np.exp(M["mu"] + M["att"][f["a"]] - M["de"][f["h"]])]
            ml = market_lambdas(f["odds"], st)
            if ml:
                f["ml"] = ml
    uclh = load_uclh(today)
    done = recent_results(params, uclh, today)
    seen = {(d["lg"], d["h"], d["a"]) for d in done}
    if EXTRA_N:
        note += " Marcadores al día con " + ", ".join(f"{k} (+{n})" for k, n in EXTRA_N.items()) + " (solo goles; córners y tiros llegan con football-data.co.uk)."
    if EXTRA_ERR:
        note += " Sin respuesta: " + ", ".join(EXTRA_ERR) + "."
    if EXTRA_BAD:
        note += " Sin equivalencia de nombre: " + ", ".join(EXTRA_BAD[:12]) + "."
    fixtures = sorted([f for f in fixtures if (f["lg"], f["h"], f["a"]) not in seen] + ucl_fixtures(params, today) + done, key=lambda x: x["date"])
    if "ucl" not in params:
        det = [l.strip() for l in LOG if "clubelo" in l or "Elo" in l]
        note += " (Champions: no se pudo leer clubelo.com. " + (det[-1][:260] if det else "sin detalle") + ")"
    if "ucl" in params and params["ucl"]["updated"] == STATIC_ELO_DATE != today.isoformat():
        falta = [t for t in UCL_TEAMS if t not in params["ucl"]["elo"]]
        note += f" Champions: Elo guardado del {STATIC_ELO_DATE}." + (" Faltan: " + ", ".join(falta) + "." if falta else "")
    log(note)
    data = {"leagues": params, "fixtures": fixtures, "note": note, "uclh": uclh, "uclalias": ucl_alias(params, uclh), "h2h": h2h_data(params, uclh),
            "generated": local.strftime("%Y-%m-%d %H:%M") + " (hora de Honduras)"}
    html = template.replace("__PARAMS__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    log(f"Página escrita en {out_path} ({len(html) // 1024} KB)")


TEMPLATE = r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="theme-color" content="#000000"><link rel="manifest" href="manifest.webmanifest"><link rel="icon" type="image/png" sizes="32x32" href="icons/favicon-32.png"><link rel="apple-touch-icon" href="icons/apple-touch-icon.png"><meta name="apple-mobile-web-app-capable" content="yes"><meta name="mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-title" content="Picks Fresh"><meta name="apple-mobile-web-app-status-bar-style" content="black"><meta name="format-detection" content="telephone=no"><style>:root{color-scheme:dark}html,body{margin:0;background:#000}</style>
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
.inst{display:flex;align-items:center;gap:10px;padding:10px 14px;padding-top:max(10px,env(safe-area-inset-top));background:var(--good-bg);border-bottom:1px solid var(--good);color:var(--ink);font-size:13.5px;line-height:1.35}
.inst[hidden]{display:none}
.inst .it{flex:1;min-width:0} .inst .it b{display:block;font-size:14px} .inst .it span{color:var(--chalk2)}
.inst .btn{flex:none;padding:8px 14px} .inst .ix{flex:none;background:none;border:0;color:var(--ink);font:400 26px/1 var(--body);padding:2px 8px;cursor:pointer}
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

.tabs{display:flex;justify-content:space-between;gap:0;overflow-x:auto;scrollbar-width:none;position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0 10px;margin-top:2px}
.tabs button{flex:0 0 auto;border:0;background:transparent;border-radius:12px;padding:10px 5px;font-weight:700;font-size:13.5px;color:var(--muted);white-space:nowrap}
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
.sumg.h2 b{font-size:17px;white-space:normal;line-height:1.2}
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

/* ---- tabla, marcador final y qué esperar ---- */
.msc{display:flex;align-items:center;justify-content:center;gap:6px;font:800 26px/1 var(--disp);font-stretch:72%}
.msc span{color:var(--muted);font-size:18px}
.mc.fin .mtime{font-size:15px;letter-spacing:.04em}
.tbl-wrap{overflow-x:auto;margin:10px -16px 0;padding:0 16px;-webkit-overflow-scrolling:touch}
table.st{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-size:13px}
.st th{font:700 10px/1 var(--body);color:var(--muted);text-transform:uppercase;letter-spacing:.02em;padding:8px 3px;text-align:center;border-bottom:1px solid var(--line);background:var(--paper)}
.st td{padding:9px 3px;min-width:26px;text-align:center;border-bottom:1px solid var(--line);background:var(--paper);font-variant-numeric:tabular-nums;white-space:nowrap}
.st .c1{position:sticky;left:0;z-index:2;width:28px;min-width:28px;text-align:center;font-weight:800;color:var(--muted)}
.st .c2{position:sticky;left:28px;z-index:2;text-align:left;width:98px;max-width:98px;min-width:98px;padding-left:4px;overflow:hidden;text-overflow:ellipsis;font-weight:700;color:var(--ink);box-shadow:6px 0 8px -6px rgba(0,0,0,.6)}
.st .pt{font:800 15px/1 var(--disp);color:var(--ink);min-width:34px}
.st .dg{color:var(--muted)}
.st tr.hh td{background:#10301f}.st tr.aa td{background:#33270c}
.st tr.hh .c1{box-shadow:inset 3px 0 0 var(--good)}.st tr.aa .c1{box-shadow:inset 3px 0 0 var(--vis)}
.st tr.hh .c2,.st tr.aa .c2{color:#fff}
.f5{display:inline-flex;gap:3px;vertical-align:middle}
.f5 i{font-style:normal;width:16px;height:16px;border-radius:5px;display:grid;place-items:center;font:800 9px/1 var(--body);color:#fff}
.f5 i.G{background:#1f9d6e}.f5 i.E{background:#6f8b7e}.f5 i.P{background:#cf4a56}
.tleg{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-top:8px}
.tleg i{display:inline-block;width:12px;height:12px;border-radius:4px;margin-right:6px;vertical-align:-2px}
.tpos{display:grid;gap:6px;margin-top:10px}
.tpos div{display:flex;justify-content:space-between;gap:10px;font-size:15px}
.tpos b{font-weight:800}
ul.exp{list-style:none;margin:10px 0 0;padding:0;display:grid;gap:10px}
ul.exp li{font-size:15px;line-height:1.45;padding-left:14px;position:relative}
ul.exp li::before{content:"";position:absolute;left:0;top:.62em;width:6px;height:6px;border-radius:50%;background:var(--good)}
ul.exp b{font-weight:800}
.pc{display:inline-block;font:800 13px/1 var(--disp);font-stretch:80%;padding:3px 7px;border-radius:7px;background:var(--good-bg);color:var(--good);margin-left:3px}
.pc.m{background:var(--warn-bg);color:var(--warn)}
.exptit{font:800 17px/1.2 var(--body)}
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
.chart .bkt{fill:var(--ink);font:700 10px var(--body)}
.bkn{margin-top:8px;line-height:1.4}.bkn b{color:var(--ink)}
.chart .ln{stroke:var(--vis);stroke-width:1.8;stroke-dasharray:5 4}
.sq.big{margin-top:12px} .sq.big i{height:26px;border-radius:5px}
.rl{display:flex;gap:4px;flex-wrap:wrap;margin-top:12px}
.rl i{font-style:normal;width:26px;height:26px;border-radius:7px;display:grid;place-items:center;font:800 13px/1 var(--body);color:#fff}
.rl i.G{background:#1f9d6e} .rl i.E{background:#6f8b7e} .rl i.P{background:#cf4a56}
.ev .evr{display:grid;grid-template-columns:1fr auto;grid-template-areas:"t p" "c p";gap:2px 12px;align-items:center;padding:9px 0;border-top:1px solid var(--line);font-size:15px}
.ev .evg{margin-top:14px;font:700 12px/1.2 var(--body);letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.ev .evr .evt{grid-area:t} .ev .evr .evc{grid-area:c} .ev .evr .evp{grid-area:p;font-size:22px;text-align:right}
.exp .exs{display:block;margin-top:3px;padding-left:2px;color:var(--muted);font-size:13.5px;line-height:1.4}
.evt i{font-style:normal;color:var(--muted);font-size:13.5px}
.evc{font:700 15px/1.2 var(--body);color:var(--ink);text-align:left}
.ev .evg{text-align:left}
.evt{color:var(--ink);line-height:1.35;text-align:left}
.ev,.ev .sub,.ev h3,.ev .evr{text-align:left}
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
.bank{margin-top:8px;padding:12px;border:1.5px solid var(--line);border-radius:12px;background:var(--soft)}
.bank label{display:block;font-size:14px;font-weight:700;margin-bottom:6px}
.bankrow{display:flex;align-items:center;gap:8px}
.bankrow .cur{font:800 18px/1 var(--disp);color:var(--muted)}
.bankrow input{flex:1;min-width:0;border:1.5px solid var(--line);border-radius:10px;padding:10px;font-size:18px;font-weight:700;background:var(--paper)}
.bankk{display:flex;align-items:center;gap:8px;margin-top:10px;font-size:13px;color:var(--muted)}
.nav button{min-width:0}
#pRes .mrow{border-top:1px solid var(--line)} #pRes .mrow:first-of-type{border-top:0}
#vProps .box>.f+.pick{margin-top:12px}
.apibox{margin-top:10px;padding:10px 12px;border:1.5px solid var(--line);border-radius:12px}
.apibox summary{font-weight:700;font-size:14px}
.keyrow{display:grid;grid-template-columns:1fr auto auto;gap:8px;margin-top:10px;align-items:center}
.apirow{margin-top:12px} .apirow .btn{width:100%}
.apirow .btn:disabled{opacity:.55}
#apiSt{margin-top:8px;line-height:1.45} #apiSt.err{color:var(--bad)} #apiSt.ok b{color:var(--good)}
#apiRem{font-size:12px;margin-top:4px}
.linkb{background:none;border:0;padding:0;color:var(--ink);text-decoration:underline;font:inherit;cursor:pointer}
#bajas h3{margin-bottom:2px}
.bjt{font-weight:800;font-size:15px;margin-top:14px}
.bjb{display:block;width:100%;text-align:left;border:1.5px solid var(--line);background:var(--paper);border-radius:12px;padding:11px 12px;margin-top:8px;font-size:14px;font-weight:600;line-height:1.35}
.bjb[aria-pressed="true"]{background:var(--sel);color:var(--on-sel);border-color:var(--sel)}
.bjsum{margin-top:12px;font-size:14px;line-height:1.5}
.bjsum .bf{color:var(--muted)}
.psg{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;background:var(--soft);border:1px solid var(--line);border-radius:14px;padding:4px;margin:12px 0}
.psg button{border:0;background:transparent;border-radius:10px;padding:10px 2px;font-weight:700;font-size:13px}
.psg button[aria-pressed="true"]{background:var(--sel);color:var(--on-sel)}
.picks h3{margin-bottom:2px}
.pk{border:1.5px solid var(--line);border-radius:12px;padding:12px;margin-top:10px;background:var(--soft)}
.pk.val{border-color:var(--good)}
.pcand{border-top:1px solid var(--line);margin-top:10px;padding-top:8px}
.pcand .pm,.why{font-size:12px;color:var(--muted)} .pcand .pl{font-weight:700;font-size:15px}
.why{margin-top:2px;line-height:1.35}
.pkh{font-size:13px;font-weight:800;letter-spacing:.02em;color:var(--muted)}
.pkm{font:800 23px/1.15 var(--disp);font-stretch:80%;margin:5px 0 3px}
.pkn{font-size:14px;line-height:1.45}
.pkn .ev{display:inline;font-size:inherit}
.addb{margin-top:10px;width:100%;border:1.5px solid var(--ink);background:transparent;border-radius:10px;padding:9px 12px;font-weight:700;font-size:14px}
.addb.on{background:var(--sel);color:var(--on-sel);border-color:var(--sel)}
.parl{margin-top:12px}
.parl h3{font-size:17px;margin:0 0 2px}
.pleg{display:grid;grid-template-columns:1fr auto;gap:2px 10px;padding:9px 0;border-top:1px solid var(--line);align-items:center}
.pleg:first-of-type{border-top:0}
.pleg .pm{font-size:12px;color:var(--muted)} .pleg .pl{font-weight:700;font-size:15px}
.pleg .pp{font:800 20px/1 var(--disp);font-stretch:80%;text-align:right}
.pleg .rm{background:none;border:0;color:var(--muted);font-size:13px;text-decoration:underline;padding:0;text-align:right}
.pst{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.pst div{background:var(--soft);border-radius:10px;padding:9px 10px} .pst small{display:block;font-size:12px;color:var(--muted)} .pst b{font:800 20px/1.1 var(--disp);font-stretch:80%}
.pcalc{display:grid;grid-template-columns:1fr 100px 70px;gap:8px;align-items:center;margin-top:10px}
.pcalc input{border:1.5px solid var(--line);border-radius:10px;padding:9px;font-size:16px;text-align:center;width:100%;background:var(--paper)}
.pcalc .mf{font-size:13px;color:var(--muted)}
.parl .row2{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.stk{margin-top:7px;font-size:14px;line-height:1.35}
.stk b{font-weight:800}
.stk.yes b{color:var(--good)} .stk.no{color:var(--muted)}
.stk small{display:block;color:var(--muted);font-size:12px}
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
.nav .in{max-width:720px;margin:0 auto;display:grid;grid-template-columns:repeat(5,1fr);gap:4px}
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
@media (max-width:380px){.tabs button{padding:10px 3px;font-size:13px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head><body>
<div class="inst" id="inst" hidden role="region" aria-label="Instalar la app"><div class="it"><b>Instala Picks Fresh como app</b><span id="instTxt"></span></div><button class="btn" id="instBtn" type="button" hidden>Instalar</button><button class="ix" id="instX" type="button" aria-label="Cerrar">×</button></div>
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
      <div class="psg" role="group" aria-label="Sección"><button type="button" data-ft="play" aria-pressed="true">Jugadas del día</button><button type="button" data-ft="trend" aria-pressed="false">Tendencias</button><button type="button" data-ft="mine" aria-pressed="false">Mi parlay <span id="parlN"></span></button></div>
      <div id="fParl">
        <div class="bank">
          <label>Bankroll (Saldo total):</label>
          <div class="bankrow"><span class="cur">L</span><input class="bankIn" inputmode="decimal" placeholder="Ej. 5000" autocomplete="off" aria-label="Bankroll, saldo total disponible en lempiras"></div>
          <div class="bankk" role="group" aria-label="Fracción de Kelly"><span>Kelly fraccional:</span><button class="chip" type="button" data-k="25" aria-pressed="true">25%</button><button class="chip" type="button" data-k="30" aria-pressed="false">30%</button></div>
          <div class="sub bankTxt"></div>
        </div>
        <div class="sub" id="parlSub"></div>
        <div id="parlPlay"></div>
        <div class="box" id="parlMine" hidden></div>
        <p class="foot">Una combinada solo gana si aciertan todas sus selecciones, y cada selección suma el margen de la casa. Las probabilidades se multiplican como si los partidos fueran independientes, por eso se usa una sola selección por partido. Son cálculos del modelo, no garantías: en pruebas con partidos pasados el modelo no superó a las casas de apuestas.</p>
      </div>
      <div id="fTrend" hidden>
      <div class="sub">Mercados donde los dos equipos cumplieron en 7 o más de sus últimos 10 partidos. Cambia la liga o el día arriba.</div>
      <div id="feat"></div>
      <div class="callout"><b>Una racha no es una probabilidad.</b> En 4.725 partidos pasados, cuando los dos equipos traían 8 o más de 10 en "más de 2.5 goles", se cumplió 69% de las veces, no 80–90%, y las casas ya daban 65%. En córners (más de 9.5) fue 68% contra 52% de base, y en tarjetas (más de 3.5) 74% contra 61%.</div>
      </div>
    </section>

    <!-- BITÁCORA -->
    <section id="vTable" aria-label="Clasificación" hidden>
      <div class="vh"><h2>Clasificación</h2><span class="sub" id="vtS"></span></div>
      <div class="box" style="margin-top:6px"><div class="tbl-wrap" id="vtBody"></div><div class="sub" id="vtN" style="margin-top:10px"></div></div>
    </section>
    <section id="vLog" aria-label="Bitácora de apuestas" hidden>
      <div class="vh"><h2>Bitácora</h2></div>
      <div class="sub">Tus apuestas anotadas, para comparar lo que dijo el modelo contra lo que pasó de verdad.</div>
      <div class="box"><div class="stats" id="stats"></div></div>
      <div class="box" style="padding-top:6px"><div class="log" id="log"></div></div>
      <div class="row"><button class="btn ghost" id="copy" type="button">Copiar bitácora</button><span class="sub" id="copyMsg"></span></div>
      <p class="foot">La bitácora se guarda solo en este navegador. Si borras los datos del sitio se pierde, así que copia el texto de vez en cuando. Si la instalas como app, la app guarda su propia bitácora (en iPhone es distinta a la de Safari), así que copia la de Safari antes de instalarla si ya tienes apuestas anotadas.</p>
    </section>

    <!-- CALCULADORA PROPS -->
    <section id="vProps" aria-label="Calculadora de props" hidden>
      <div class="vh"><h2>Calculadora Props</h2></div>
      <div class="sub">Para tiros, remates, goles de un jugador o córners. Escribe tu promedio esperado y la línea de la casa. Usa la distribución de Poisson y no depende de ningún partido.</div>
      <div class="bank">
            <label>Bankroll (Saldo total):</label>
            <div class="bankrow"><span class="cur">L</span><input class="bankIn" inputmode="decimal" placeholder="Ej. 5000" autocomplete="off" aria-label="Bankroll, saldo total disponible en lempiras"></div>
            <div class="bankk" role="group" aria-label="Fracción de Kelly"><span>Kelly fraccional:</span><button class="chip" type="button" data-k="25" aria-pressed="true">25%</button><button class="chip" type="button" data-k="30" aria-pressed="false">30%</button></div>
            <div class="sub bankTxt"></div>
          </div>
      <div class="box">
        <label class="f">Nombre / Mercado<input class="in" id="pName" type="text" placeholder="Ej. Tiros de Olise" autocomplete="off"></label>
        <div class="pick">
          <label class="f">Promedio esperado<input class="in" id="pAvg" inputmode="decimal" placeholder="Ej. 2.5" autocomplete="off"></label>
          <label class="f">Línea de la casa<input class="in" id="pLine" inputmode="decimal" placeholder="Ej. 1.5" autocomplete="off"></label>
        </div>
        <div class="sub" id="pMsg" style="margin-top:10px">Escribe el promedio por partido y la línea (1.5, 2.5…) para ver las probabilidades.</div>
        <div id="pRes" hidden>
          <div class="mrow" id="pr_o"></div>
          <div class="mrow" id="pr_u"></div>
          <div class="sub" id="pPush"></div>
        </div>
      </div>
      <div class="box how"><details><summary>Cómo se calcula y qué limita</summary>
        <p>Menos de la línea = suma de las probabilidades exactas de 0, 1, 2… hasta el entero por debajo de la línea. Más de la línea = 100% menos eso. Con una línea entera (por ejemplo 2) hay devolución si sale justo ese número: se calcula aparte y la cuota justa ya la descuenta.</p>
        <p>El promedio es tuyo: la calculadora no lo verifica ni lo ajusta por rival, minutos o alineación. Si el promedio está mal, todo lo demás también.</p>
        <p>Poisson supone que los eventos son independientes. Los tiros y córners suelen variar más que eso, así que Poisson tiende a dar de menos las cifras muy altas y muy bajas. Es una estimación sin validar con partidos pasados.</p>
      </details></div>
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
        <button type="button" role="tab" data-t="table" aria-selected="false">Tabla</button>
        <button type="button" role="tab" data-t="mk" aria-selected="false">Apostar</button>
      </div>

      <div class="pane" id="pSum">
        <div class="box">
          <h3>Lo que dice el modelo</h3>
          <div class="sub" id="readout"></div>
          <div class="tiles" id="tiles"></div>
        </div>
        <div class="box" id="expect" hidden></div>
        <div class="box picks" id="picks" hidden></div>
        <div class="box" id="bajas"></div>
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

      <div class="pane" id="pTable" hidden>
        <div class="box">
          <h3>Clasificación</h3>
          <div class="sub" id="tblS"></div>
          <div class="tpos" id="tblPos"></div>
          <div class="tleg"><span><i style="background:#10301f;box-shadow:inset 3px 0 0 var(--good)"></i>Local</span><span><i style="background:#33270c;box-shadow:inset 3px 0 0 var(--vis)"></i>Visitante</span></div>
          <div class="tbl-wrap" id="tblBody"></div>
          <div class="sub" id="tblN" style="margin-top:10px"></div>
        </div>
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
          <details class="apibox" id="apiBox">
            <summary>Cuotas automáticas (The Odds API)</summary>
            <div class="sub" style="margin-top:8px">Pega tu clave gratuita de the-odds-api.com. Se guarda solo en este navegador (no está en el código de la página ni la ve nadie más) y solo se envía a the-odds-api.com cuando tocas el botón.</div>
            <div class="keyrow"><input class="in" id="apiKey" type="password" placeholder="Tu API Key" autocomplete="off" autocapitalize="off" spellcheck="false" aria-label="API Key de The Odds API"><button class="mini" id="apiSave" type="button">Guardar</button><button class="mini" id="apiDel" type="button">Borrar</button></div>
            <div class="sub" id="apiKeySt"></div>
          </details>
          <div class="apirow"><button class="btn" id="apiLoad" type="button">🔄 Cargar cuotas del mercado</button></div>
          <div class="sub" id="apiSt"></div>
          <div class="sub" id="apiRem"></div>
          <div class="bank">
            <label>Bankroll (Saldo total):</label>
            <div class="bankrow"><span class="cur">L</span><input class="bankIn" inputmode="decimal" placeholder="Ej. 5000" autocomplete="off" aria-label="Bankroll, saldo total disponible en lempiras"></div>
            <div class="bankk" role="group" aria-label="Fracción de Kelly"><span>Kelly fraccional:</span><button class="chip" type="button" data-k="25" aria-pressed="true">25%</button><button class="chip" type="button" data-k="30" aria-pressed="false">30%</button></div>
            <div class="sub bankTxt"></div>
          </div>
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
  <button type="button" data-v="table"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M3 15h18M9 4v16"/></svg>Tabla</button>
  <button type="button" data-v="log"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3h11a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6z"/><path d="M6 3v18M10 8h6M10 12h6"/></svg>Bitácora</button>
  <button type="button" data-v="props"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>Props</button>
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
function scoreMatrix(lh, la, rho){          // rho: corrección de Dixon-Coles para 0-0, 1-0, 0-1 y 1-1 (0 = Poisson independiente)
  const N=10, ph=pois(lh,N), pa=pois(la,N); let s=0; const m=[];
  for(let i=0;i<=N;i++){m.push([]);for(let j=0;j<=N;j++){const v=ph[i]*pa[j];m[i].push(v);s+=v;}}
  if(rho){
    m[0][0]*=Math.max(0,1-lh*la*rho); m[0][1]*=Math.max(0,1+lh*rho);
    m[1][0]*=Math.max(0,1+la*rho);    m[1][1]*=Math.max(0,1-rho);
    s=0; m.forEach(r=>r.forEach(v=>{s+=v;}));
  }
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
function compute(h, a, fh, fa, ml, dh, da){
  const L = DATA.leagues[league], out = {rows:[], chips:[]};
  let l0;
  if(ml){ l0=ml; }
  else if(L.kind==="elo"){ const E=1/(1+Math.pow(10,-((L.elo[h]+(dh||0)+L.hfa-L.elo[a]-(da||0))/400))); l0=eloLam(E,L.T); }
  else { l0=lam(L.models.goles,h,a); }
  const rho = (!ml && L.kind!=="elo" && L.models.goles.rho) || 0;   // con cuotas o Elo no se aplica: esos goles esperados ya vienen de otro cálculo
  const lh = l0[0]*fh, la = l0[1]*fa, m = scoreMatrix(lh,la,rho);
  let pH=0,pD=0,pA=0,o15=0,o25=0,o35=0,bt=0; const tg=new Array(2*m.length).fill(0);
  for(let i=0;i<m.length;i++)for(let j=0;j<m.length;j++){
    const v=m[i][j]; tg[i+j]+=v; if(i>j)pH+=v; else if(i===j)pD+=v; else pA+=v;
    if(i+j>=2)o15+=v; if(i+j>=3)o25+=v; if(i+j>=4)o35+=v; if(i>=1&&j>=1)bt+=v;
  }
  if(ml){ bt=1/(1+Math.exp(-(Math.log(bt/(1-bt))+0.14))); }   // en pruebas, "ambos anotan" sale ~3 pts más alto de lo que da Poisson
  if(!ml && L.kind!=="elo" && L.models.goles.btb){ bt=L.models.goles.btw*bt+(1-L.models.goles.btw)*L.models.goles.btb; }   // el modelo solo exagera las diferencias entre partidos en "ambos anotan"
  out.lh=lh; out.la=la; out.p=[pH,pD,pA]; out.o25=o25; out.bt=bt; out.tg=tg; out.mk={}; out.fromMkt=!!ml;
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
    out.chips.push([c.n, ch.toFixed(1)+" – "+ca.toFixed(1)]); out.mk[k]={n:c.n,tot:tot,at:M.at,ch:ch,ca:ca};
    for(const x of c.tot){const p=pOver(x,tot,M.at);
      R(c.n+" · total","Más de "+x,p,k==="corners"?"o:corners:"+x:null);
      R(c.n+" · total","Menos de "+x,1-p,k==="corners"?"u:corners:"+x:null);}
    for(const x of c.team){R(c.n+" · "+h,"Más de "+x,pOver(x,ch,M.ah));R(c.n+" · "+a,"Más de "+x,pOver(x,ca,M.aa));}
  }
  return out;
}

/* ---------- conclusiones rápidas ---------- */
function pickLine(c){                      // c: [{lbl,p}] -> línea con 65-75% (la más cercana a 70%); si no hay, la más cercana por encima
  const inr=c.filter(x=>x.p>=0.65&&x.p<=0.75).sort((a,b)=>Math.abs(a.p-0.70)-Math.abs(b.p-0.70));
  if(inr.length)return inr[0];
  const hi=c.filter(x=>x.p>0.75).sort((a,b)=>a.p-b.p);
  return hi.length?hi[0]:null;
}
function safeLine(c,pick){                 // una línea más segura (80% o más) en la misma dirección
  const hi=c.filter(x=>x.p>=0.80&&(!pick||x.p>pick.p+0.04)).sort((a,b)=>a.p-b.p);
  return hi.length?hi[0]:null;
}
function linesOver(fn,lo,hi,unit){         // fn(línea) = prob. de "más de". Devuelve candidatos de más y de menos
  const ov=[],un=[];
  for(let x=lo;x<=hi;x+=1){const p=fn(x); ov.push({lbl:"Más de "+x+" "+unit,p:p,x:x}); un.push({lbl:"Menos de "+x+" "+unit,p:1-p,x:x});}
  return {ov:ov,un:un};
}
function pcTag(p){return `<span class="pc${p>=0.65?"":" m"}">${pct(p)}</span>`;}
function lineTxt(L,dirOver,central){       // dirección según el lado más probable en la línea central
  const c=dirOver?L.ov:L.un, pk=pickLine(c); if(!pk)return null;
  const sf=safeLine(c,pk);
  return {pk:pk,sf:sf};
}
function expectHtml(h,a,c,fin){
  if(fin)return "";
  const li=[], esc2=esc;
  const [pH,pD,pA]=c.p, fav=pH>=pA?[h,pH,pD+pH,"local"]:[a,pA,pD+pA,"visitante"];
  if(fav[1]>=0.65)li.push(`<li><b>Ganador:</b> alta probabilidad de victoria para ${esc2(fav[0])} ${pcTag(fav[1])}.</li>`);
  else if(Math.max(pH,pD,pA)>=0.5)li.push(`<li><b>Ganador:</b> ${esc2(fav[0])} es favorito ${pcTag(fav[1])}, pero no está asegurado. Opción más segura: ${esc2(fav[0])} o empate ${pcTag(fav[2])}.</li>`);
  else li.push(`<li><b>Ganador:</b> partido parejo, sin favorito claro (${esc2(h)} ${pct(pH)}, empate ${pct(pD)}, ${esc2(a)} ${pct(pA)}). Opción más segura: ${pH>=pA?esc2(h):esc2(a)} o empate ${pcTag(fav[2])}.</li>`);
  // goles: P(total > x) desde la distribución del modelo
  const gp=x=>{let s=0;for(let k=0;k<c.tg.length;k++)if(k>x)s+=c.tg[k];return s;};
  const G=linesOver(x=>gp(x),0.5,5.5,"goles"), gOver=gp(2.5)>=0.5, gl=lineTxt(G,gOver);
  const gtype=c.o25>=0.60?"partido abierto":c.o25<=0.40?"partido cerrado":"partido equilibrado en goles";
  if(gl)li.push(`<li><b>Goles:</b> se espera un ${gtype} (~${(c.lh+c.la).toFixed(1)} goles). Línea sugerida: ${gl.pk.lbl} ${pcTag(gl.pk.p)}.${gl.sf?` Más segura: ${gl.sf.lbl} ${pcTag(gl.sf.p)}.`:""}</li>`);
  const bt=c.bt, yes=bt>=0.5, pl=yes?bt:1-bt;
  const bform=t=>{const sr=seriesFor(t,"all",10); if(!sr||!sr.ok)return `${esc2(t)} sin datos`; const r=sr.ms.map(nrm).map(m=>m.f>0&&m.c>0); return `${esc2(t)} <b>${r.filter(Boolean).length} de ${r.length}</b>`;};
  li.push(`<li><b>Ambos anotan:</b> ${pl>=0.65?(yes?"es probable que los dos marquen":"es probable que al menos uno no marque"):pl>=0.57?`ligera ventaja para "${yes?"sí":"no"}", sin ser una opción clara`:"muy parejo, sin ventaja clara"} (Sí ${pcTag(bt)} · No ${pcTag(1-bt)}).<span class="exs">↳ Historial últimos 10 (partidos donde anotaron los dos): ${bform(h)} · ${bform(a)}.</span></li>`);
  const mk=c.mk||{}, band=(arr)=>arr.filter(x=>x.p>=0.65&&x.p<=0.85).sort((u,v)=>Math.abs(u.p-0.75)-Math.abs(v.p-0.75))[0]||null;
  const one=(k,nm,hiTot)=>{              // mejor "más de" y mejor "menos de" entre 65% y 85% (la más cercana a 75%)
    const M=mk[k]; if(!M)return null;
    const sd=Math.sqrt(M.tot+M.at*M.tot*M.tot), lo=Math.max(0.5,Math.floor(M.tot-2.6*sd)+0.5), hi=Math.floor(M.tot+2.6*sd)+0.5;
    const L={ov:[],un:[]}; for(let x=lo;x<=hi;x+=1){const p=pOver(x,M.tot,M.at); L.ov.push({lbl:"Más de "+x+(nm?" "+nm:""),p:p}); L.un.push({lbl:"Menos de "+x+(nm?" "+nm:""),p:1-p});}
    let ov=band(L.ov), un=band(L.un);
    if(!ov&&!un){const f=pickLine(pOver(Math.floor(M.tot)+0.5,M.tot,M.at)>=0.5?L.ov:L.un); if(!f)return null; if(f.lbl[0]==="M"&&f.lbl[1]==="á")ov=f; else un=f;}
    const high=M.tot>=hiTot, overFirst=high||pOver(Math.floor(M.tot)+0.5,M.tot,M.at)>=0.5;
    const list=(overFirst?[ov,un]:[un,ov]).filter(Boolean);
    return {M:M,list:list,txt:list.map(z=>`${z.lbl} ${pcTag(z.p)}`).join(" · "),high:high};
  };
  const co=one("corners","córners",9.5);
  if(co)li.push(`<li><b>Córners:</b> se proyectan ~${co.M.tot.toFixed(1)} córners en total. Línea${co.list.length>1?"s":""} sugerida${co.list.length>1?"s":""}: ${co.txt}.</li>`);
  const sh=one("shots","",25), so=one("sot","",sh?(sh.high?0:1e9):8.5);
  if(sh||so){
    const parts=[]; if(sh)parts.push(`~${sh.M.tot.toFixed(0)} tiros totales`); if(so)parts.push(`~${so.M.tot.toFixed(0)} a puerta`);
    li.push(`<li><b>Tiros:</b> se esperan ${parts.join(" y ")}.${sh?` Tiros totales: ${sh.txt}.`:""}${so?` A puerta: ${so.txt}.`:""}</li>`);
  }
  const hr=h2hRaw(h,a,5);                  // historial directo: solo si hay al menos 3 duelos previos
  if(hr.length>=3){
    const N=hr.map(nrm), n=N.length, w=N.filter(x=>x.f>x.c).length, d=N.filter(x=>x.f===x.c).length, o=N.filter(x=>x.tot>=3).length;
    li.push(`<li><b>Historial directo (H2H):</b> en sus últimos ${n} duelos hubo un promedio de ${(N.reduce((x,y)=>x+y.tot,0)/n).toFixed(1)} goles (más de 2.5 en ${o} de ${n}). ${esc2(h)} ganó ${w}, empataron ${d} y ${esc2(a)} ganó ${n-w-d}.</li>`);
  }
  return `<div class="exptit">📌 Resumen de tendencias para este partido</div>
    <ul class="exp">${li.join("")}</ul>
    <div class="sub" style="margin-top:12px">En goles, la línea sugerida es la que el modelo ubica entre 65% y 75% (si ninguna cae ahí, la más cercana por encima). ${co||sh||so?`En córners y tiros muestro la mejor línea de \"más de\" y de \"menos de\" entre 65% y 85%, la más cercana a 75%; si el partido promete muchos córners (9.5 o más) o tiros (25 o más), va primero la de \"más de\". `:""}Son probabilidades, no garantías. ${co||sh||so?"De córners y tiros no tenemos cuotas para saber si pagan bien.":""}</div>`;
}

/* ---------- clasificación ---------- */
function standings(lg){
  const L=DATA.leagues[lg]; let games=[], names=[], note="";
  if(lg==="ucl"){
    games=(DATA.uclh||[]).filter(r=>r[5]==="Fase de liga"&&r[0]>=20260701).map(r=>({h:r[1],a:r[2],hg:r[3],ag:r[4],d:r[0]}));
    note="Fase de liga 2026/27 de la Champions. Los clubes con más jornadas jugadas suben primero; la tabla completa se llena jornada a jornada.";
  }else{
    const last=L.hm.length?L.hm[L.hm.length-1][0]:0, y=Math.floor(last/10000), mo=Math.floor(last/100)%100,
      st=L.tbl==="cal"?y*10000+101:L.tbl==="split"?y*10000+(mo>=7?701:101):(mo>=7?y:y-1)*10000+701;
    games=L.hm.filter(m=>m[0]>=st).map(m=>({h:L.hteams[m[1]],a:L.hteams[m[2]],hg:m[3],ag:m[4],d:m[0]}));
    names=L.teams;
    note="Datos hasta "+L.updated+". Orden: puntos, diferencia de goles y goles a favor. Las ligas reales pueden desempatar de otra forma (por ejemplo, el resultado entre los dos equipos en La Liga).";
  }
  const T={}, get=t=>T[t]||(T[t]={t:t,pj:0,g:0,e:0,p:0,gf:0,gc:0,pts:0,res:[]});
  names.forEach(get);
  games.sort((x,y)=>x.d-y.d).forEach(g=>{
    const A=get(g.h), B=get(g.a);
    A.pj++;B.pj++;A.gf+=g.hg;A.gc+=g.ag;B.gf+=g.ag;B.gc+=g.hg;
    if(g.hg>g.ag){A.g++;B.p++;A.pts+=3;A.res.push("G");B.res.push("P");}
    else if(g.hg<g.ag){B.g++;A.p++;B.pts+=3;A.res.push("P");B.res.push("G");}
    else{A.e++;B.e++;A.pts++;B.pts++;A.res.push("E");B.res.push("E");}
  });
  const rows=Object.values(T).sort((x,y)=>y.pts-x.pts||(y.gf-y.gc)-(x.gf-x.gc)||y.gf-x.gf||x.t.localeCompare(y.t));
  rows.forEach((r,i)=>{r.pos=i+1;});
  return {rows:rows,note:note,n:games.length};
}
function tableHtml(lg,hl){                  // hl: [local, visitante] para resaltar
  const S=standings(lg);
  if(!S.rows.length||!S.n)return {html:`<div class="empty">Todavía no hay partidos de esta temporada en la fuente de datos.</div>`,S:S};
  const th=["#","Equipo","PJ","G","E","P","GF","GC","DG","Pts","Últ. 5"].map((x,i)=>`<th class="${i===0?"c1":i===1?"c2":""}">${x}</th>`).join("");
  const body=S.rows.map(r=>{
    const cl=hl&&r.t===hl[0]?"hh":hl&&r.t===hl[1]?"aa":"", dg=r.gf-r.gc;
    return `<tr class="${cl}"><td class="c1">${r.pos}</td><td class="c2" title="${esc(r.t)}">${esc(r.t)}</td><td>${r.pj}</td><td>${r.g}</td><td>${r.e}</td><td>${r.p}</td><td>${r.gf}</td><td>${r.gc}</td><td class="dg">${dg>0?"+":""}${dg}</td><td class="pt">${r.pts}</td><td><span class="f5">${r.res.slice(-5).map(x=>`<i class="${x}">${x}</i>`).join("")}</span></td></tr>`;}).join("");
  return {html:`<table class="st"><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`,S:S};
}
function ord(n){return n+".º";}
function renderTable(h,a){
  const T=tableHtml(league,[h,a]), S=T.S;
  $("tblBody").innerHTML=T.html; $("tblN").textContent=S.note;
  const pj=S.rows.length?Math.max.apply(null,S.rows.map(r=>r.pj)):0;
  $("tblS").textContent=DATA.leagues[league].name+(pj?" · temporada 2026/27 · hasta la jornada "+pj:"");
  const f=t=>S.rows.find(r=>r.t===t);
  const pos=[[h,"hh"],[a,"aa"]].map(([t])=>{const r=f(t);return r?`<div><span>${esc(t)}</span><b>${ord(r.pos)} de ${S.rows.length} · ${r.pts} pts · ${r.gf}-${r.gc} en goles</b></div>`:"";}).join("");
  $("tblPos").innerHTML=pos;
}
function renderTableView(){
  const T=tableHtml(league,null), L=DATA.leagues[league];
  $("vtBody").innerHTML=T.html; $("vtN").textContent=T.S.note;
  $("vtS").textContent=L.name;
}

/* ---------- interfaz ---------- */
let cur = null;
const DOW=["dom","lun","mar","mié","jue","vie","sáb"], MON=["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"];
let day=null;
function dparts(iso){const p=iso.slice(0,10).split("-").map(Number);const d=new Date(p[0],p[1]-1,p[2]);return {dow:DOW[d.getDay()],d:p[2],mon:MON[p[1]-1]};}
function dtxt(iso){const p=dparts(iso);return p.dow+" "+p.d+" "+p.mon+", "+iso.slice(11,16);}
function fxOfLeague(){return (DATA.fixtures||[]).filter(f=>f.lg===league);}
function daysOfLeague(){const out=[];fxOfLeague().forEach(f=>{const k=f.date.slice(0,10);if(!out.includes(k))out.push(k);});return out;}
function defaultDay(days){
  const up=days.find(k=>fxOfLeague().some(f=>f.date.slice(0,10)===k&&!f.score));
  return up||days[days.length-1]||null;
}
function fillTeams(){
  const L=DATA.leagues[league];
  for(const id of ["home","away"]){
    $(id).innerHTML=L.teams.map(t=>`<option>${esc(t)}</option>`).join("");
  }
  const days=daysOfLeague(); if(!days.includes(day)) day=defaultDay(days);
  const first=fxOfLeague().find(f=>f.date.slice(0,10)===day&&!f.score)||fxOfLeague().find(f=>f.date.slice(0,10)===day);
  const pref=first?[first.h,first.a]:[L.teams[0],L.teams[1]];
  $("home").value=pref[0]; $("away").value=pref[1];
}
function formHtml(t){
  let f=(DATA.leagues[league].form||{})[t]||"";
  if(!f&&league==="ucl"){                    // Champions: últimos 5 oficiales (liga local + Champions)
    const sr=seriesFor(t,"all",5);
    if(sr.ok)f=sr.ms.slice().reverse().map(m=>{const x=m.home?m.hg:m.ag,y=m.home?m.ag:m.hg;return x>y?"G":x===y?"E":"P";}).join("");
  }
  return f.split("").filter(Boolean).map(c=>`<i class="${c}">${c}</i>`).join("");
}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
/* ---------- recalcular con las cuotas que escribe el usuario ---------- */
let userOdds={}, userMatch="";
/* cuotas que el usuario escribe, guardadas por partido (solo en este navegador) para que Destacados las vea */
const KEYO="scoutpicks.myodds.v1";
let oddsMem={};
try{const o=JSON.parse(localStorage.getItem(KEYO)||"{}"); if(o&&typeof o==="object"&&!Array.isArray(o))oddsMem=o;}catch(e){}
(function(){                                   // limpia partidos que ya no están en el calendario
  const ok=new Set((DATA.fixtures||[]).filter(f=>!f.score).map(f=>f.lg+"|"+f.h+"|"+f.a)); let ch=false;
  Object.keys(oddsMem).forEach(k=>{if(!ok.has(k)){delete oddsMem[k];ch=true;}});
  if(ch)try{localStorage.setItem(KEYO,JSON.stringify(oddsMem));}catch(e){}
})();
function memSave(key,odds){
  if(odds&&Object.keys(odds).length){delete oddsMem[key]; oddsMem[key]=Object.assign({},odds);} else delete oddsMem[key];
  const ks=Object.keys(oddsMem); while(ks.length>80)delete oddsMem[ks.shift()];
  try{localStorage.setItem(KEYO,JSON.stringify(oddsMem));}catch(e){}
}
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
  return {all:league==="ucl"?"Últimos "+n+" partidos oficiales de cada equipo, con su liga local y la Champions League juntas.":"Últimos "+n+" partidos de liga de cada equipo, jugados de local y de visita.",
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
  if(league==="ucl"&&sc==="all"){          // en un partido de Champions, "Todos" junta su liga local y la Champions
    const c=uclName(team), res0=resolveHist(team);
    const ms=((res0?domMatches(res0,n,"all"):[]).concat(c?uclMatches(c,n):[])).sort((x,y)=>y.d-x.d).slice(0,n);
    if(ms.length)return {ok:true,name:c||res0.name,ms:ms,mixed:true};
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
  const f=m.home?m.hg:m.ag, c=m.home?m.ag:m.hg, res=f>c?"G":f===c?"E":"P", mine=Array.isArray(me)?me:[me];
  const nx=t=>mine.includes(t)?`<b>${esc(t)}</b>`:esc(t);
  const chips=[]; if(m.comp)chips.push(m.comp); if(m.stage)chips.push(m.stage); if(m.note)chips.push(m.note);
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
  const out=h2hRaw(h,a,10), rh=resolveHist(h), me=[h,uclName(h),rh&&rh.name].filter(Boolean);
  let box="";
  if(out.length){
    const N=out.map(nrm), n=N.length, w=N.filter(x=>x.f>x.c).length, d=N.filter(x=>x.f===x.c).length, l=n-w-d;
    const gf=N.reduce((x,y)=>x+y.f,0)/n, gc=N.reduce((x,y)=>x+y.c,0)/n, o=N.filter(x=>x.tot>=3).length, bt=N.filter(x=>x.f>0&&x.c>0).length;
    const t=(v,lab)=>`<div><b>${v}</b><span>${lab}</span></div>`, pc=x=>Math.round(100*x/n)+"%";
    box=`<div class="sumg h2">${t(w+"G "+d+"E "+l+"P","Balance de "+esc(h)+" (ganó, empató, perdió)")}${t(gf.toFixed(1)+" – "+gc.toFixed(1)+" ("+(gf+gc).toFixed(1)+" total)","Goles por duelo ("+esc(h)+" – "+esc(a)+")")}${t(o+" de "+n+" partidos ("+pc(o)+")","Más de 2.5 goles")}${t(bt+" de "+n+" partidos ("+pc(bt)+")","Ambos anotaron")}</div>`;
  }
  return `<div class="box h2h"><h3>Enfrentamientos directos</h3><div class="sub">Últimos ${out.length||10} duelos entre ellos en liga, Champions, Europa League y Conference League, de las últimas ~10 temporadas. Solo informativo: el modelo no usa estos resultados. G, E y P se cuentan desde ${esc(h)}.</div>${box}${out.length?`<div>${out.map(m=>matchRow(m,me)).join("")}</div>`:`<div class="empty">No se han enfrentado en liga ni en torneos europeos (Champions, Europa League, Conference) en los datos que tenemos (unas 10 temporadas).</div>`}</div>`;
}
function renderHist(h,a){
  $("scopeH").innerHTML=scopeHtml();
  $("teamSw").innerHTML=[["h",h],["a",a]].map(([k,t])=>`<button type="button" data-ht="${k}" aria-pressed="${k===histTeam}">${esc(t)}</button>`).join("");
  const col=(team,role,s)=>`<section class="box tcard" data-t="${role}"><h3>${esc(team)}<span class="role">${role==="h"?"Local en este partido":"Visitante en este partido"}</span></h3>${
    s.ok?(s.mixed?`<div class="sub">Liga local y Champions juntas, del más reciente al más antiguo.</div>`:s.local?`<div class="sub">Sus partidos de liga local (no incluye Champions).</div>`:"")+sumHtml(s.ms)+`<div>${s.ms.map(m=>matchRow(m,s.name)).join("")}</div>`
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
const HASMEM={};
function lgHas(){   // ¿la liga trae córners / tarjetas? (las ligas nuevas de football-data solo traen goles)
  const L=DATA.leagues[league]; if(!L||L.kind==="elo"||!L.hm)return {cor:true,crd:true};
  if(!HASMEM[league])HASMEM[league]={cor:L.hm.some(m=>m[5]>=0&&m[6]>=0),crd:L.hm.some(m=>m[7]!==undefined&&m[7]>=0&&m[8]>=0)};
  return HASMEM[league];
}
function nrm(m){
  const f=m.home?m.hg:m.ag, c=m.home?m.ag:m.hg, hc=m.hc>=0&&m.ac>=0, hk=m.kh>=0&&m.ka>=0;
  const cf=hc?(m.home?m.hc:m.ac):null, kf=hk?(m.home?m.kh:m.ka):null;
  return {f:f,c:c,tot:f+c,cor:hc?m.hc+m.ac:null,cf:cf,cc:hc?m.hc+m.ac-cf:null,
    crd:hk?m.kh+m.ka:null,kf:kf,kc:hk?m.kh+m.ka-kf:null,d:m.d,rival:m.home?m.a:m.h,home:m.home};
}
function vfn(k,w){
  if(k==="goals")return w==="tot"?(m=>m.tot):(m=>m.f);
  if(k==="cor")return w==="tot"?(m=>m.cor):(m=>m.cf);
  return w==="tot"?(m=>m.crd):(m=>m.kf);
}
function bkfn(k){   // desglose [a favor, en contra] de cada partido
  if(k==="goals")return m=>[m.f,m.c];
  if(k==="cor")return m=>m.cf===null?null:[m.cf,m.cc];
  return m=>m.kf===null?null:[m.kf,m.kc];
}
function medianOf(a){const s=a.slice().sort((x,y)=>x-y), i=Math.floor(s.length/2);return s.length%2?s[i]:(s[i-1]+s[i])/2;}
function lineOf(){const c=MET[met], k=met+who; return lineSel[k]!==undefined?lineSel[k]:(who==="tot"?c.td:c.ed);}
function hitV(v,line){return v===null?null:(dir==="o"?v>line:v<line);}
function pcls(p){return p>=0.7?"g":p>=0.5?"m":"b";}
function scLbl(){return {all:"Todos los partidos",home:"De local",away:"De visita",ucl:"Champions"}[scope];}
function code3(t){return t.replace(/[^A-Za-zÀ-ÿ]/g,"").slice(0,3).toUpperCase();}
function dm(n){return String(n%100).padStart(2,"0")+"/"+String(Math.floor(n/100)%100).padStart(2,"0");}
function chartSvg(ms,vf,line,bf){
  const n=ms.length, W=340, H=n<=10?212:176, pl=22, pr=4, pt=16, pb=n<=10?56:36;
  const vals=ms.map(vf), nums=vals.filter(v=>v!==null);
  const top=Math.max(2,Math.ceil(Math.max(line+0.5,...nums,1))), step=top>8?2:1;
  const iy=v=>H-pb-(v/top)*(H-pb-pt), bw=(W-pl-pr)/n, bd=Math.min(bw*0.64,26);
  let g="";
  for(let t=0;t<=top;t+=step)g+=`<line class="gl" x1="${pl}" x2="${W-pr}" y1="${iy(t).toFixed(1)}" y2="${iy(t).toFixed(1)}"/><text class="ax" x="${pl-5}" y="${(iy(t)+3).toFixed(1)}" text-anchor="end">${t}</text>`;
  ms.forEach((m,i)=>{
    const v=vals[i], cx=pl+bw*(i+0.5);
    const bk=bf?bf(m):null, tip=bk?`${esc(m.rival)} ${dm(m.d)}: ${m.home?"local":"visita"}, ${bk[0]} a favor – ${bk[1]} en contra`:"";
    if(v!==null){const h=Math.max(v/top*(H-pb-pt),3);
      g+=`<rect class="bar ${hitV(v,line)?"y":"n"}" x="${(cx-bd/2).toFixed(1)}" y="${(H-pb-h).toFixed(1)}" width="${bd.toFixed(1)}" height="${h.toFixed(1)}" rx="3"><title>${tip}</title></rect>`;
      if(n<=10)g+=`<text class="vl" x="${cx.toFixed(1)}" y="${(H-pb-h-4).toFixed(1)}" text-anchor="middle">${v}</text>`;}
    else g+=`<text class="ax" x="${cx.toFixed(1)}" y="${H-pb-4}" text-anchor="middle">–</text>`;
    if(n<=10)g+=`<text class="ax" x="${cx.toFixed(1)}" y="${H-pb+14}" text-anchor="middle" style="font-weight:700">${esc(code3(m.rival))}</text><text class="ax dd" x="${cx.toFixed(1)}" y="${H-pb+26}" text-anchor="middle">${dm(m.d)}</text>${bk?`<text class="bkt" x="${cx.toFixed(1)}" y="${H-pb+40}" text-anchor="middle">${bk[0]}-${bk[1]}</text>`:""}`;
    else g+=`<text class="ax dd" transform="translate(${(cx+3).toFixed(1)} ${H-pb+6}) rotate(-90)" text-anchor="end">${dm(m.d)}</text>`;
  });
  g+=`<line class="ln" x1="${pl}" x2="${W-pr}" y1="${iy(line).toFixed(1)}" y2="${iy(line).toFixed(1)}"/>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Valor de cada partido, con la línea de ${line}">${g}</svg>`;
}
function ccHead(team,s){return `<div class="cch"><b>${esc(team)}</b><span>${s.ok?(s.ms.length>1?"Últimos "+s.ms.length:"Último partido")+" · "+scLbl():scLbl()}</span></div>`;}
function chartCard(team,s,vf,line,bf){
  if(!s.ok)return `<div class="cc">${ccHead(team,s)}<div class="empty">${esc(s.msg)}</div></div>`;
  const ms=s.ms.slice().reverse().map(nrm), vs=ms.map(vf).filter(v=>v!==null);
  if(!vs.length)return `<div class="cc">${ccHead(team,s)}<div class="empty">No hay datos de esta estadística para este equipo en estos partidos.</div></div>`;
  const hits=vs.filter(v=>hitV(v,line)).length, avg=(vs.reduce((x,y)=>x+y,0)/vs.length).toFixed(1);
  const bks=bf?ms.map(bf).filter(Boolean):[], av=i=>(bks.reduce((x,y)=>x+y[i],0)/bks.length).toFixed(1);
  return `<div class="cc">${ccHead(team,s)}<div class="ccs"><span>Tasa de acierto <b class="${pcls(hits/vs.length)}">${Math.round(100*hits/vs.length)}%</b></span><span>Promedio <b>${avg}</b></span><span>Mediana <b>${medianOf(vs)}</b></span></div>${chartSvg(ms,vf,line,bf)}${bks.length?`<div class="sub bkn">Promedio a favor – en contra: <b>${av(0)} – ${av(1)}</b>. ${ms.length<=10?"Bajo cada barra: lo que anotó el equipo – lo que recibió. ":"Toca una barra para ver lo que anotó el equipo y lo que recibió. "}Si el equipo casi no anota y la barra es alta, pasó la línea por lo que recibió.</div>`:""}</div>`;
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
function seasonLbl(d){const y=Math.floor(d/10000),m=Math.floor(d/100)%100,q=m>=7?y:y-1;return q+"/"+String((q+1)%100).padStart(2,"0");}
/* Duelos directos de los dos equipos: liga (si juegan en la misma) + Champions, de las últimas ~10 temporadas, del más reciente al más antiguo. */
function h2hRaw(h,a,k){
  const rh=resolveHist(h), ra=resolveHist(a), out=[], H=DATA.h2h||{};
  if(rh&&ra&&rh.key===ra.key){
    const L=DATA.leagues[rh.key], i=L.hteams.indexOf(rh.name), j=L.hteams.indexOf(ra.name), cn=L.name;
    if(i>=0&&j>=0)for(const m of L.hm)if((m[1]===i&&m[2]===j)||(m[1]===j&&m[2]===i))
      out.push({d:m[0],h:L.hteams[m[1]],a:L.hteams[m[2]],hg:m[3],ag:m[4],hc:m[5],ac:m[6],kh:m[7]===undefined?-1:m[7],ka:m[8]===undefined?-1:m[8],home:m[1]===i,comp:cn+" "+seasonLbl(m[0])});
    const S=(H.leagues||{})[rh.key];
    if(S){const x=S.t.indexOf(rh.name), y=S.t.indexOf(ra.name);
      if(x>=0&&y>=0)for(const m of S.m)if((m[1]===x&&m[2]===y)||(m[1]===y&&m[2]===x))
        out.push({d:m[0],h:S.t[m[1]],a:S.t[m[2]],hg:m[3],ag:m[4],hc:-1,ac:-1,kh:-1,ka:-1,home:m[1]===x,comp:cn+" "+seasonLbl(m[0])});}
  }
  const ch=uclName(h), ca=uclName(a);
  if(ch&&ca&&ch!==ca){
    for(const m of (DATA.uclh||[]))if(m[5]!=="Previa"&&((m[1]===ch&&m[2]===ca)||(m[1]===ca&&m[2]===ch)))
      out.push({d:m[0],h:m[1],a:m[2],hg:m[3],ag:m[4],hc:-1,ac:-1,kh:-1,ka:-1,home:m[1]===ch,stage:m[5],note:m[6],comp:"Champions League "+seasonLbl(m[0])});
    const S=H.ucl;
    if(S){const x=S.t.indexOf(ch), y=S.t.indexOf(ca);
      if(x>=0&&y>=0)for(const m of S.m)if((m[1]===x&&m[2]===y)||(m[1]===y&&m[2]===x))
        out.push({d:m[0],h:S.t[m[1]],a:S.t[m[2]],hg:m[3],ag:m[4],hc:-1,ac:-1,kh:-1,ka:-1,home:m[1]===x,stage:(S.s||[])[m[5]]||"",comp:((S.c||[])[m[6]||0]||"Champions League")+" "+seasonLbl(m[0])});}
  }
  out.sort((p,q)=>q.d-p.d);
  return out.slice(0,k);
}
/* Líneas de evidencia: cuántas veces se cumplió en los partidos del contexto (local en casa, visitante fuera). */
function evRows(h,a,test,o){
  const rows=[], rh=resolveHist(h), ra=resolveHist(a), ph=o.h!==false&&rh, pa=o.a!==false&&ra;
  // una línea por equipo y contexto: "X de Y partidos", sin mezclar a los dos equipos en una sola fracción
  const add1=(ms,team,ctx,grp)=>{let x=0,n=0;ms.forEach(m=>{const r=test(m);if(r!==null){n++;if(r)x++;}});if(n)rows.push({x:x,n:n,p:x/n,team:team,ctx:ctx,grp:grp});};
  [5,10,20].forEach(k=>{
    const g="Últimos "+k+" (local en casa, visitante fuera)";
    if(ph)add1(domMatches(rh,k,"home").map(nrm),h,"de local",g);
    if(pa)add1(domMatches(ra,k,"away").map(nrm),a,"de visitante",g);
  });
  const g2="Últimos 10 en general (local y visita)";
  if(ph)add1(domMatches(rh,10,"all").map(nrm),h,"general",g2);
  if(pa)add1(domMatches(ra,10,"all").map(nrm),a,"general",g2);
  const ch=uclName(h), ca=uclName(a);   // solo aparecen los equipos que sí tienen partidos de Champions; sin ninguno, no hay fila
  if(o.h!==false&&ch)add1(uclMatches(ch,5).map(nrm),h,"en Champions","Champions League (últimos 5)");
  if(o.a!==false&&ca)add1(uclMatches(ca,5).map(nrm),a,"en Champions","Champions League (últimos 5)");
  if(o.h2h!==false){const hr=h2hRaw(h,a,10);if(hr.length)add1(hr.map(r=>nrm(o.flip?Object.assign({},r,{home:!r.home}):r)),"Entre sí","últimos "+hr.length+" duelos","Enfrentamientos directos (H2H)");}
  return rows;
}
function evCard(title,rows,note){
  return `<div class="box ev"><h3>${esc(title)}</h3>${note?`<div class="sub">${esc(note)}</div>`:""}${rows.length?rows.map((r,i)=>(i&&rows[i-1].grp===r.grp?"":`<div class="evg">${esc(r.grp)}</div>`)+`<div class="evr"><span class="evt"><b>${esc(r.team)}</b> <i>(${esc(r.ctx)})</i></span><span class="evc">${r.x} de ${r.n} ${r.n===1?"partido":"partidos"}</span><span class="evp ${pcls(r.p)}">${Math.round(r.p*100)}%</span></div>`).join(""):`<div class="empty">No hay suficientes partidos para calcular esto.</div>`}</div>`;
}
const EVNOTE="Cuántas veces se cumplió en esos partidos. Verde: 70% o más. Naranja: entre 50% y 69%. Rojo: menos de 50%. Es historia, no una probabilidad.";
function renderStats(h,a,fx){
  $("scopeT").innerHTML=scopeHtml();
  const hs=lgHas(), chs=METCH.filter(([k])=>k==="cor"?hs.cor:k==="crd"?hs.crd:true);
  if(!chs.some(([k])=>k===met))met="goals";
  $("metricT").innerHTML=chs.map(([k,t])=>`<button class="chip" type="button" data-m="${k}" aria-pressed="${k===met}">${t}</button>`).join("");
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
      const bf=who==="tot"||met==="goals"?bkfn(met):null;
      body+=chartCard(h,sh,vf,line,bf)+chartCard(a,sa,vf,line,bf);
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
  const list=fxOfLeague().filter(f=>f.date.slice(0,10)===day&&!f.score), items=[];
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
  if(typeof featTab!=="undefined"&&featTab!=="trend"&&view==="feat")schedParl();
}

/* Calculadora de stake: Kelly fraccional. f* = (p·cuota − 1) / (cuota − 1); stake = bankroll × fracción × f* */
const KEYB="scoutpicks.bank.v1";
let bank={v:0,k:25};
try{const o=JSON.parse(localStorage.getItem(KEYB)||"{}"); if(o.v>0)bank.v=o.v; if(o.k===30)bank.k=30;}catch(e){}
function saveBank(){try{localStorage.setItem(KEYB,JSON.stringify(bank));}catch(e){}}
function lps(x){return "L "+Math.round(x).toLocaleString("en-US");}
function stakeHtml(p,v,push){
  push=push||0;                                          // probabilidad de devolución (líneas enteras en Props)
  const ev=p*(v-1)-(1-p-push);                           // sin devolución es p·cuota − 1
  if(!(ev>1e-12))return {cls:"stk no",html:"Stake sugerido: <b>L 0 (No apostar)</b>"};
  const f=(ev/(v-1))*(bank.k/100);                       // fracción del bankroll
  const pc=(f*100).toFixed(f<0.01?2:1)+"% del bankroll";
  if(!(bank.v>0))return {cls:"stk yes",html:`Stake sugerido: <b>${pc}</b><small>Escribe tu bankroll (en Apostar o Props) para verlo en lempiras.</small>`};
  const m=bank.v*f;
  return {cls:"stk yes",html:`Stake sugerido: <b>${m<0.5?"menos de L 1":lps(m)}</b><small>${pc} · Kelly ${bank.k}%</small>`};
}
function bankMsg(){
  const t=bank.v>0?"Con L "+Math.round(bank.v).toLocaleString("en-US")+" y Kelly al "+bank.k+"%, el stake sale de la probabilidad y la cuota de cada fila. Es una referencia, no una garantía: si la probabilidad falla, Kelly apuesta de más.":"Escribe tu saldo para ver cuánto apostar en cada fila con ganancia esperada positiva. Si la ganancia esperada es negativa, el stake es L 0.";
  document.querySelectorAll(".bankTxt").forEach(e=>{e.textContent=t;});
}
function syncBankUI(src){
  document.querySelectorAll(".bankIn").forEach(el=>{if(el!==src)el.value=bank.v>0?String(bank.v):"";});
  document.querySelectorAll(".bank button[data-k]").forEach(x=>x.setAttribute("aria-pressed",String(+x.dataset.k===bank.k)));
}
function updStakes(){
  bankMsg(); schedPicks(); if(view==="feat"&&featTab!=="trend")schedParl();
  if(cur)cur.rows.forEach((r,i)=>{const el=$("o"+i); if(el&&parseFloat(el.value.replace(",","."))>1)onOdds(i);});
  propCalc();
}
document.addEventListener("input",ev=>{
  if(!ev.target.classList||!ev.target.classList.contains("bankIn"))return;
  const n=parseFloat(ev.target.value.replace(/[,\s]/g,"")); bank.v=n>0?n:0; saveBank(); syncBankUI(ev.target); updStakes();
});
document.addEventListener("click",ev=>{
  const b=ev.target.closest&&ev.target.closest(".bank button[data-k]"); if(!b)return;
  bank.k=+b.dataset.k; saveBank(); syncBankUI(); updStakes();
});
syncBankUI(); bankMsg();
function onOdds(i){
  schedPicks();
  const r=cur.rows[i], v=parseFloat($("o"+i).value.replace(",","."));
  const e=$("e"+i), b=$("b"+i), f=$("f"+i);
  const sk=$("s"+i);
  if(!(v>1)){e.textContent="";e.className="ev";f.innerHTML="";b.disabled=true;if(sk){sk.innerHTML="";sk.className="stk";}return;}
  const ev=r.p*v-1, gap=r.p-1/v;
  let flag="";
  if(ev>0 && gap>=0.08) flag=`<span class="flag">Diferencia grande con la casa: probable error del modelo</span>`;
  else if(r.fav && r.p<0.5 && ev>0) flag=`<span class="flag">Ojo: el modelo subestima favoritos</span>`;
  e.className="ev "+(ev>0?"pos":"neg");
  e.textContent=(ev>0?"+":"")+(ev*100).toFixed(1)+"%";
  f.innerHTML=flag;
  b.disabled=false;
  if(sk){const t=stakeHtml(r.p,v); sk.className=t.cls; sk.innerHTML=t.html;}
}


/* ---------- Picks sugeridos y armador de parlays ---------- */
const KEYP="scoutpicks.parlay.v1";
let parlay=[], parlOdds={}, featTab="play", lastPicks=null, picksTimer=null;
try{const o=JSON.parse(localStorage.getItem(KEYP)||"[]"); if(Array.isArray(o))parlay=o.filter(x=>x&&x.id&&x.p>0&&x.p<1);}catch(e){}
function saveParlay(){try{localStorage.setItem(KEYP,JSON.stringify(parlay));}catch(e){}}
function flagFor(r,v){                       // mismas alertas que muestra la fila en Apostar
  if(!(v>1))return null; const ev=r.p*v-1, gap=r.p-1/v;
  if(ev>0&&gap>=0.08)return "big";
  if(r.fav&&r.p<0.5&&ev>0)return "fav";
  return null;
}
const PK_GOALS=new Set(["o:goals:1.5","u:goals:1.5","o:goals:2.5","u:goals:2.5","o:goals:3.5","u:goals:3.5"]);
function pkEligible(r){return r.sec==="Resultado"||PK_GOALS.has(r.key);}     // 1X2, doble oportunidad y goles 1.5 / 2.5 / 3.5
function pkCands(rows,oddsOf){
  return rows.filter(pkEligible).map(r=>{const v=oddsOf(r), ok=v>1;
    return {r:r,p:r.p,label:r.label,fair:1/r.p,v:ok?v:null,ev:ok?r.p*v-1:null,flag:ok?flagFor(r,v):null,id:r.key||(r.sec+"|"+r.label)};});
}
function pkSuggest(cands){
  const val=cands.filter(c=>c.ev!==null&&c.ev>0&&c.p>=0.38&&c.flag!=="big").sort((a,b)=>b.ev-a.ev)[0];
  let main=null;
  if(val)main={kind:"val",c:val};
  else{const pr=cands.filter(c=>c.p>=0.60&&c.p<=0.80&&c.fair>=1.35&&c.fair<=1.85).sort((a,b)=>b.p-a.p)[0]; if(pr)main={kind:"prob",c:pr};}
  const band=cands.filter(c=>c.p>=0.75&&c.p<=0.92), pref=band.filter(c=>/^Más de 1\.5 goles$|^Doble oportunidad/.test(c.label));
  const fijo=(pref.length?pref:band).sort((a,b)=>b.p-a.p)[0]||null;
  return {main:main,fijo:fijo,anyOdds:cands.some(c=>c.v)};
}
function legOf(lg,match,c){return {id:lg+"|"+match+"|"+c.id,lg:lg,match:match,label:c.label,p:c.p,odds:c.v||null};}
function inParlay(id){return parlay.some(x=>x.id===id);}
function addLeg(leg){
  const i=parlay.findIndex(x=>x.lg===leg.lg&&x.match===leg.match);     // una sola selección por partido
  const replaced=i>=0?parlay[i]:null;
  if(i>=0)parlay.splice(i,1);
  parlay.push(leg); saveParlay(); parlN(); return replaced;
}
function parlN(){$("parlN").textContent=parlay.length?"("+parlay.length+")":"";}
function pkBlock(c,kind,idx){
  const odds=c.v?`Cuota cargada <b>${f2(c.v)}</b> · valor esperado <span class="ev ${c.ev>0?"pos":"neg"}">${c.ev>0?"+":""}${(c.ev*100).toFixed(1)}%</span>`:"";
  return {odds:odds};
}
function renderPicks(){
  const box=$("picks"); if(!cur||$("home").value===$("away").value){box.hidden=true;return;}
  const h=$("home").value,a=$("away").value, fxs=(DATA.fixtures||[]).filter(f=>f.lg===league&&f.h===h&&f.a===a);
  if(fxs.length&&fxs.every(f=>f.score)){box.hidden=true;return;}
  const oddsOf=r=>{const i=cur.rows.indexOf(r), el=i>=0?$("o"+i):null; return el?parseFloat(el.value.replace(",",".")):NaN;};
  const cands=pkCands(cur.rows,oddsOf), sg=pkSuggest(cands), match=h+" vs "+a;
  lastPicks={main:sg.main&&legOf(league,match,sg.main.c),fijo:sg.fijo&&legOf(league,match,sg.fijo)};
  let html=`<h3>🎯 Picks sugeridos</h3><div class="sub">Se eligen entre resultado, doble oportunidad y más/menos de 1.5, 2.5 y 3.5 goles.</div>`;
  const btn=(k,leg)=>`<button class="addb${inParlay(leg.id)?" on":""}" type="button" data-pk="${k}">${inParlay(leg.id)?"✓ En tu parlay (quitar)":"+ Agregar a mi Parlay"}</button>`;
  if(sg.main){
    const c=sg.main.c, isV=sg.main.kind==="val", st=isV?stakeHtml(c.p,c.v):null;
    html+=`<div class="pk${isV?" val":""}"><div class="pkh">🎯 PICK PRINCIPAL · ${isV?"PICK DE VALOR":"PICK DE PROBABILIDAD"}</div><div class="pkm">${esc(c.label)}</div>
      <div class="pkn">Probabilidad <b>${pct(c.p)}</b> · cuota justa <b>${f2(c.fair)}</b>${c.v?`<br>Cuota cargada <b>${f2(c.v)}</b> · valor esperado <span class="ev ${c.ev>0?"pos":"neg"}">${c.ev>0?"+":""}${(c.ev*100).toFixed(1)}%</span>`:`<br>Busca una cuota de ${f2(c.fair)} o más; si tu casa paga menos, no vale la pena.`}</div>
      ${isV?`<div class="${st.cls}">${st.html}</div>${c.flag==="fav"?`<div class="flag" style="margin-top:6px">Ojo: el modelo subestima favoritos.</div>`:""}`:`<div class="sub" style="margin-top:6px">${sg.anyOdds?"Con las cuotas cargadas ningún mercado tiene valor esperado positivo (sin alertas), así que va el de mejor equilibrio entre probabilidad (60–74%) y cuota justa (1.35–1.85).":"No hay cuotas cargadas: va el de mejor equilibrio entre probabilidad (60–74%) y cuota justa (1.35–1.85)."}</div>`}
      ${btn("main",lastPicks.main)}</div>`;
  }else html+=`<div class="pk"><div class="pkh">🎯 PICK PRINCIPAL</div><div class="pkn" style="margin-top:6px">Ningún mercado cumple las condiciones en este partido (valor positivo sin alertas, o probabilidad de 60–74% con cuota justa de 1.35 a 1.85). Mejor no forzar uno.</div></div>`;
  if(sg.fijo){
    const c=sg.fijo;
    html+=`<div class="pk"><div class="pkh">🔗 FIJO PARA PARLAY · IDEAL PARA COMBINAR</div><div class="pkm">${esc(c.label)}</div>
      <div class="pkn">Probabilidad <b>${pct(c.p)}</b> · cuota justa <b>${f2(c.fair)}</b>${c.v?`<br>Cuota cargada <b>${f2(c.v)}</b> · valor esperado <span class="ev ${c.ev>0?"pos":"neg"}">${c.ev>0?"+":""}${(c.ev*100).toFixed(1)}%</span>`:""}</div>
      <div class="sub" style="margin-top:6px">Es el mercado más seguro del partido, pero paga poco. Sirve para sumar a una combinada, no para jugarlo solo.</div>
      ${btn("fijo",lastPicks.fijo)}</div>`;
  }else html+=`<div class="pk"><div class="pkh">🔗 FIJO PARA PARLAY</div><div class="pkn" style="margin-top:6px">No hay un mercado con probabilidad de 75% a 92% en este partido.</div></div>`;
  html+=`<div class="sub" style="margin-top:10px">Sugerencias del modelo, no garantías. En pruebas con partidos pasados el modelo no superó a las casas de apuestas.</div>`;
  box.innerHTML=html; box.hidden=false;
}
function schedPicks(){if(picksTimer)return; picksTimer=setTimeout(()=>{picksTimer=null; try{renderPicks();}catch(e){}},0);}
$("picks").addEventListener("click",ev=>{
  const b=ev.target.closest("button[data-pk]"); if(!b||!lastPicks)return;
  const leg=lastPicks[b.dataset.pk]; if(!leg)return;
  if(inParlay(leg.id)){parlay=parlay.filter(x=>x.id!==leg.id); saveParlay(); parlN();}
  else addLeg(leg);
  renderPicks(); renderMine();
});

/* --- escaneo de los partidos del día (todas las ligas) --- */
function parlayDay(){
  if(day)return day;
  const up=(DATA.fixtures||[]).filter(f=>!f.score).map(f=>f.date.slice(0,10)).sort(); return up[0]||"";
}
function fxRows(f){                           // filas del partido con las cuotas de la pantalla del partido (mercado) y filas del modelo puro (sin cuotas)
  const saved=league; league=f.lg;
  try{
    const key=f.lg+"|"+f.h+"|"+f.a, af=apiStore[key], um=key===userMatch?userOdds:(oddsMem[key]||null);
    let odds=Object.assign({},f.odds,af&&af.odds), ml=(af&&af.ml)||f.ml||null;
    if(um){const u=mlFromOdds(um); if(u)ml=u; odds=Object.assign({},odds,um);}
    const ba=key===userMatch?bajArgs(DATA.leagues[f.lg],ml):[1,1,0,0];
    const c=compute(f.h,f.a,ba[0],ba[1],ml,ba[2],ba[3]);
    const pure=ml?compute(f.h,f.a,ba[0],ba[1],null,ba[2],ba[3]):c;
    return {key:key,odds:odds,ml:ml,hasMkt:!!ml,c:c,pure:pure,
      oddsOf:r=>{const v=odds[rowKey(r)]!==undefined?odds[rowKey(r)]:(r.key?odds[r.key]:undefined); return v>1?v:NaN;}};
  }catch(e){return null;}
  finally{league=saved;}
}
function candsOf(fr,f,match,filter){
  const pm=new Map(fr.pure.rows.map(r=>[rowKey(r),r]));
  return fr.c.rows.filter(filter).map(r=>{const v=fr.oddsOf(r), ok=v>1, q=pm.get(rowKey(r));
    return {r:r,p:r.p,pMod:q?q.p:r.p,label:r.label,fair:1/r.p,v:ok?v:null,ev:ok?r.p*v-1:null,flag:ok?flagFor(r,v):null,id:rowKey(r),fx:f,match:match,hasMkt:fr.hasMkt};});
}
function scanDay(){
  const d=parlayDay(), now=Date.now(), items=[];
  (DATA.fixtures||[]).forEach(f=>{
    if(f.score||f.date.slice(0,10)!==d||Date.parse(f.date+":00-06:00")<now)return;
    const fr=fxRows(f); if(!fr)return;
    const match=f.h+" vs "+f.a;
    items.push({f:f,fr:fr,match:match,legs:candsOf(fr,f,match,pkEligible)});
  });
  return {day:d,items:items,legs:[].concat(...items.map(i=>i.legs))};
}
function lgName(k){return k==="ucl"?"Champions":(DATA.leagues[k]||{name:k}).name;}
function legOfC(c){return legOf(c.fx.lg,c.match,c);}

/* 🎯 Pick del Día: mayor valor esperado (EV+) con probabilidad real mayor de 55% y sin alerta de "probable error del modelo" */
function pickOfDay(legs){
  const pool=legs.filter(c=>c.p>0.55), withO=pool.filter(c=>c.v);
  const good=withO.filter(c=>c.ev>0&&c.flag!=="big").sort((a,b)=>b.ev-a.ev);
  const seen=new Set(), cand=pool.filter(c=>c.p<=0.88&&!(good[0]&&c.id===good[0].id&&c.match===good[0].match))
    .sort((a,b)=>((b.ev!==null?b.ev:-0.05)-(a.ev!==null?a.ev:-0.05))||(b.p-a.p))
    .filter(c=>{if(seen.has(c.match))return false; seen.add(c.match); return true;}).slice(0,4);
  return {best:good[0]||null,big:withO.filter(c=>c.ev>0&&c.flag==="big").length,cand:cand,nOdds:withO.length};
}
/* ⚖️ Combinada del Día: 2–3 selecciones de partidos distintos con probabilidad individual entre 62% y 80% */
function comboOfDay(items){
  const cmp=(x,y)=>{const a=x.ev!==null,b=y.ev!==null; if(a!==b)return a?-1:1; return a?y.ev-x.ev:y.p-x.p;};
  const per=[];
  items.forEach(it=>{const b=it.legs.filter(c=>c.p>=0.62&&c.p<=0.80&&c.flag!=="big").sort(cmp); if(b.length)per.push(b[0]);});
  per.sort(cmp);
  if(per.length<2)return {legs:null,n:per.length};
  let pick=per.slice(0,Math.min(3,per.length));
  if(pick.length===3&&pick[0].p*pick[1].p*pick[2].p<0.30)pick=pick.slice(0,2);
  return {legs:pick,n:per.length};
}
/* 🚀 Parlay Soñador: solo "falsos favoritos" y tendencias de goles que el modelo de Poisson subestima */
function dreamPool(items){
  const pool=[], st={n:items.length,fav:0,big:0,trend:0};
  items.forEach(it=>{
    const f=it.f, fr=it.fr, cm=new Map(fr.c.rows.map(r=>[rowKey(r),r])), pmap=new Map(fr.pure.rows.map(r=>[rowKey(r),r])), cands=[];
    if(fr.hasMkt){                                                     // falso favorito: el mercado le da más que el modelo (4 a menos de 8 puntos)
      const m1=cm.get("1"), m2=cm.get("2"), q1=pmap.get("1"), q2=pmap.get("2");
      if(m1&&m2&&q1&&q2){
        const home=m1.p>=m2.p, mf=home?m1:m2, qf=home?q1:q2, gap=mf.p-qf.p;
        if(mf.p>=0.45&&gap>=0.04){
          if(gap>=0.08)st.big++;
          else{
            const k="Resultado|Doble oportunidad "+(home?f.a:f.h)+" o empate", row=cm.get(k), q=pmap.get(k);
            if(row&&q&&Math.min(row.p,q.p)>=0.40){
              st.fav++;
              cands.push({kind:"fav",row:row,q:q,score:gap,note:"Falso favorito: "+(home?f.h:f.a)+" tiene "+pct(mf.p)+" en el mercado y "+pct(qf.p)+" en el modelo."});
            }
          }
        }
      }
    }
    const sc=f.lg==="ucl"?"ucl":"all", saved=league; let sh=null,sa=null;
    league=f.lg; try{sh=seriesFor(f.h,sc,10); sa=seriesFor(f.a,sc,10);}catch(e){} finally{league=saved;}
    if(sh&&sh.ok&&sa&&sa.ok){                                          // tendencia oculta: los dos equipos la cumplen y Poisson da 8 o más puntos menos
      const A=toTrend(sh.ms), B=toTrend(sa.ms);
      [[0,"o:goals:2.5"],[2,"btts:y"],[1,"u:goals:2.5"]].forEach(td=>{
        const x=trendHits(A,TM[td[0]]), y=trendHits(B,TM[td[0]]); if(x.n<8||y.n<8)return;
        const rx=x.k/x.n, ry=y.k/y.n; if(Math.min(rx,ry)<0.7)return;
        const row=cm.get(td[1]), q=pmap.get(td[1]); if(!row||!q)return;
        const gap=(rx+ry)/2-q.p; if(gap<0.08||Math.min(row.p,q.p)<0.40)return;
        st.trend++;
        cands.push({kind:"trend",row:row,q:q,score:gap,note:"Tendencia oculta: "+f.h+" "+x.k+"/"+x.n+" y "+f.a+" "+y.k+"/"+y.n+" partidos; Poisson da "+pct(q.p)+"."});
      });
    }
    if(!cands.length)return;
    cands.sort((a,b)=>b.score-a.score); const c=cands[0], v=fr.oddsOf(c.row);
    pool.push({fx:f,match:it.match,lg:f.lg,label:c.row.label,id:rowKey(c.row),p:Math.min(c.row.p,c.q.p),pMod:c.q.p,pMkt:c.row.p,
      v:v>1?v:null,note:c.note,kind:c.kind,score:c.score});
  });
  return {pool:pool,st:st};
}
function dreamSets(pool){
  const P=pool.slice().sort((a,b)=>b.score-a.score).slice(0,12), n=P.length, best={5:null,8:null};
  for(let m=1;m<(1<<n);m++){
    const idx=[]; for(let i=0;i<n;i++)if(m&(1<<i))idx.push(i);
    if(idx.length<2||idx.length>4)continue;
    const fair=1/idx.reduce((s,i)=>s*P[i].p,1);
    [[5,5,14],[8,8,16]].forEach(t=>{if(fair>=t[1]&&fair<=t[2]){const b=best[t[0]]; if(!b||fair<b.fair-1e-9||(Math.abs(fair-b.fair)<1e-9&&idx.length<b.legs.length))best[t[0]]={fair:fair,legs:idx.map(i=>P[i])};}});
  }
  if(best[8]&&best[5]&&best[8].legs.length===best[5].legs.length&&best[8].legs.every(l=>best[5].legs.includes(l)))best[8]=null;
  return best;
}
function dreamStake(p,v){                     // Kelly fraccional, pero nunca más del 1% del bankroll
  const ev=p*v-1;
  if(!(ev>1e-12))return {cls:"stk no",html:"Stake sugerido: <b>L 0 (No apostar)</b><small>Con esa cuota no hay valor esperado positivo.</small>"};
  const f=Math.min(0.01,(ev/(v-1))*(bank.k/100)), pc=(f*100).toFixed(f<0.01?2:1)+"% del bankroll";
  if(!(bank.v>0))return {cls:"stk yes",html:`Stake sugerido: <b>${pc}</b> (tope 1%)<small>Escribe tu bankroll arriba para verlo en lempiras.</small>`};
  const m=bank.v*f;
  return {cls:"stk yes",html:`Stake sugerido: <b>${m<0.5?"menos de L 1":lps(m)}</b><small>${pc} · tope 1% del bankroll</small>`};
}

/* --- cuota de la casa editable en una selección suelta --- */
let hiMap=[];
function hiRow(c){
  const i=hiMap.push({key:c.fx.lg+"|"+c.fx.h+"|"+c.fx.a,rk:c.id,p:c.p})-1;
  return `<div class="pcalc"><span class="mf">Cuota de la casa${c.v?"":" (escríbela)"}</span><input inputmode="decimal" data-hi="${i}" placeholder="Cuota" value="${c.v?String(c.v):""}" aria-label="Cuota de la casa para ${esc(c.label)}"><span class="ev" id="hiev_${i}"></span></div><div id="hist_${i}"></div>`;
}
function hiUpdate(i){
  const m=hiMap[i], el=document.querySelector('[data-hi="'+i+'"]'), e=$("hiev_"+i), s=$("hist_"+i); if(!m||!el||!e||!s)return;
  const v=parseFloat(el.value.replace(",","."));
  if(!(v>1)){e.textContent="";e.className="ev";s.innerHTML=`<div class="sub" style="margin-top:6px">Solo tiene valor si tu casa paga más de ${f2(1/m.p)}.</div>`;return;}
  const ev=m.p*v-1; e.className="ev "+(ev>0?"pos":"neg"); e.textContent=(ev>0?"+":"")+(ev*100).toFixed(1)+"%";
  const st=stakeHtml(m.p,v); s.innerHTML=`<div class="${st.cls}">${st.html}</div>`;
}
function whereTxt(c){return esc(lgName(c.fx.lg))+" · "+esc(c.match)+" · "+esc(dtxt(c.fx.date));}

function parlCalcHtml(id,legs,dream){
  const pr=legs.reduce((s,x)=>s*x.p,1), fair=1/pr, prod=legs.every(x=>x.odds>1)?legs.reduce((s,x)=>s*x.odds,1):null;
  const typed=parseFloat(String(parlOdds[id]||"").replace(",","."));
  return `<div class="pst"><div><small>${dream?"Probabilidad real (conservadora)":"Probabilidad real del parlay"}</small><b>${(pr*100).toFixed(1)}%</b></div><div><small>Cuota justa combinada</small><b>${f2(fair)}</b></div></div>
    <div class="pcalc"><span class="mf">Cuota que paga la casa por el parlay${prod?` <small>(multiplicando las cuotas cargadas: ${f2(prod)})</small>`:""}</span><input inputmode="decimal" data-pid="${id}" placeholder="${prod?f2(prod):"Cuota"}" value="${typed>1?String(parlOdds[id]):""}" aria-label="Cuota total del parlay"><span class="ev" id="pev_${id}"></span></div>
    <div id="pst_${id}"></div>`;
}
function parlUpdate(id,legs,dream){
  const pr=legs.reduce((s,x)=>s*x.p,1), prod=legs.every(x=>x.odds>1)?legs.reduce((s,x)=>s*x.odds,1):null;
  const typed=parseFloat(String(parlOdds[id]||"").replace(",",".")), v=typed>1?typed:(prod||0);
  const e=$("pev_"+id), s=$("pst_"+id); if(!e||!s)return;
  if(!(v>1)){
    e.textContent="";e.className="ev";
    s.innerHTML=dream?`<div class="${bank.v>0?"stk yes":"stk no"}">Tope de stake: <b>${bank.v>0?lps(bank.v*0.01):"1% del bankroll"}</b><small>Solo si tu casa paga más de ${f2(1/pr)} por el parlay. Escribe la cuota para confirmarlo.</small></div>`
      :`<div class="sub" style="margin-top:6px">Escribe la cuota total que te da tu casa para ver si el parlay tiene valor.</div>`;
    return;
  }
  const ev=pr*v-1; e.className="ev "+(ev>0?"pos":"neg"); e.textContent=(ev>0?"+":"")+(ev*100).toFixed(1)+"%";
  const st=dream?dreamStake(pr,v):stakeHtml(pr,v);
  s.innerHTML=`<div class="${st.cls}">${st.html}</div>${typed>1||!prod?"":`<div class="sub">Cuota calculada con las cuotas de cada selección; las casas suelen pagar menos en una combinada. Escribe la real.</div>`}`;
}
function parlCard(id,title,sub,legs,extra){
  extra=extra||{};
  return `<div class="box parl" data-parl="${id}"><h3>${title}</h3><div class="sub">${sub}</div>
    <div style="margin-top:8px">${legs.map(x=>`<div class="pleg"><div><div class="pm">${esc(lgName(x.lg))} · ${esc(x.match)}</div><div class="pl">${esc(x.label)}${x.odds>1?` <small class="pm">· cuota ${f2(x.odds)}</small>`:""}</div>${x.note?`<div class="why">${esc(x.note)}</div>`:""}</div><div class="pp">${pct(x.p)}${extra.rm?`<br><button class="rm" type="button" data-rm="${esc(x.id)}">quitar</button>`:""}</div></div>`).join("")}</div>
    ${extra.pre||""}${parlCalcHtml(id,legs,extra.dream)}${extra.foot||""}</div>`;
}
let parlSets={}, parlDream={};
function useBtn(id,legs){parlSets[id]=legs; return `<div class="row2"><button class="mini" type="button" data-use="${id}">Usar en mi parlay</button></div>`;}

function pickHtml(P){
  let html="";
  if(P.best){
    const c=P.best;
    html=`<div class="pk val"><div class="pkh">🎯 PICK DEL DÍA · APUESTA SOLITARIA</div><div class="pkm">${esc(c.label)}</div><div class="pkn">${whereTxt(c)}<br>
      Probabilidad real <b>${pct(c.p)}</b> · cuota justa <b>${f2(c.fair)}</b><br><span class="sub">${c.hasMkt?"Probabilidad calculada a partir de las cuotas del mercado.":"Probabilidad del modelo (Elo y Poisson)."}</span></div>
      ${hiRow(c)}${c.flag==="fav"?`<div class="flag" style="margin-top:6px">Ojo: el modelo subestima favoritos.</div>`:""}${useBtn("pick",[legOfC(c)]).replace("Usar en mi parlay","+ Agregar a mi Parlay")}</div>`;
    return html;
  }
  html=`<div class="pk"><div class="pkh">🎯 PICK DEL DÍA · APUESTA SOLITARIA</div><div class="pkn" style="margin-top:6px"><b>Hoy no hay pick.</b> Ningún mercado cumple a la vez valor esperado positivo y probabilidad real mayor de 55%${P.big?` (hay ${P.big} con valor positivo pero con alerta de "probable error del modelo", y se descartan a propósito)`:""}.${P.nOdds?"":" Además no hay cuotas cargadas: carga las del partido o escribe las de tu casa."} No forzar una apuesta también es una decisión.</div>`;
  if(P.cand.length)html+=`<div class="sub" style="margin-top:10px">Mercados más probables de hoy. Escribe la cuota de tu casa: si paga más que la cuota justa, pasa a ser el pick.</div>`+
    P.cand.map(c=>`<div class="pcand"><div class="pm">${whereTxt(c)}</div><div class="pl">${esc(c.label)} · <b>${pct(c.p)}</b> · cuota justa <b>${f2(c.fair)}</b></div>${hiRow(c)}</div>`).join("");
  return html+"</div>";
}
function comboHtml(C){
  if(!C.legs)return `<div class="box parl"><h3>⚖️ Combinada del Día</h3><div class="sub">Hoy no hay 2 selecciones de partidos distintos con probabilidad de 62% a 80% (sin alerta de error del modelo)${C.n===1?": solo hay 1":""}.</div></div>`;
  const legs=C.legs.map(c=>legOfC(c));
  return parlCard("combo","⚖️ Combinada del Día ("+legs.length+" selecciones)","Selecciones de partidos distintos con probabilidad individual entre 62% y 80%, ordenadas por valor esperado cuando hay cuotas cargadas.",legs,{foot:useBtn("combo",legs)});
}
function dreamHtml(D,sets){
  const head=`<div class="sub" style="margin:0 0 6px">Solo mercados donde el modelo ve un <b>falso favorito</b> (el mercado le da 4 a menos de 8 puntos más que el modelo) o una <b>tendencia de goles</b> que cumplen los dos equipos y que Poisson subestima en 8 puntos o más. Máximo 4 selecciones, una por partido, cada una con 40% o más. Es el parlay de mayor riesgo: la probabilidad real mostrada usa la <b>menor</b> entre el modelo y el mercado de cada selección.</div>`;
  const sts=D.st, why=`Partidos escaneados: ${sts.n} · con falso favorito: ${sts.fav} · con tendencia oculta: ${sts.trend}${sts.big?` · descartados por diferencia de 8+ puntos (probable error del modelo): ${sts.big}`:""}.`;
  const cards=[5,8].filter(k=>sets[k]).map(k=>[k,"🚀 Parlay Soñador · cuota justa de "+(sets[k].fair>=8?"8.00":"5.00")+" o más"]).map(t=>{
    const s=sets[t[0]], legs=s.legs.map(l=>({id:l.fx.lg+"|"+l.match+"|"+l.id,lg:l.lg,match:l.match,label:l.label,p:l.p,odds:l.v||null,note:l.note}));
    const model=s.legs.reduce((x,l)=>x*l.pMod,1), mkt=s.legs.reduce((x,l)=>x*l.pMkt,1);
    parlDream["dream"+t[0]]=true;
    return parlCard("dream"+t[0],t[1],"Cuota justa combinada "+f2(s.fair)+". La casa paga menos que la cuota justa: confirma que tu casa pague de verdad "+(s.fair>=8?"más de 8.00":"más de 5.00")+" antes de jugarlo.",legs,
      {dream:true,pre:`<div class="sub" style="margin-top:8px">Solo con el modelo: ${(model*100).toFixed(1)}% · solo con el mercado: ${(mkt*100).toFixed(1)}%.</div>`,foot:useBtn("dream"+t[0],legs)});
  });
  if(!cards.length)return `<div class="box parl"><h3>🚀 Parlay Soñador</h3>${head}<div class="sub"><b>Hoy no hay un parlay de cuota alta bien respaldado.</b> ${why} Con menos de 2 selecciones válidas, o sin una combinación que llegue a 5.00 sin pasar de 4 selecciones, es mejor no armarlo.</div></div>`;
  return `<div class="box parl"><h3>🚀 Parlay Soñador</h3>${head}<div class="sub">${why}</div></div>`+cards.join("")+
    `<div class="callout" style="margin-top:10px"><b>Cuidado.</b> En pruebas con partidos pasados, las diferencias entre el modelo y el mercado casi siempre fueron error del modelo, y una racha de 8 de 10 se cumplió 69% de las veces, no 80–90%. Por eso el stake nunca pasa de 1% del bankroll y es L 0 si la cuota no da valor.</div>`;
}
function renderParl(){
  const sc=scanDay(); parlSets={}; parlDream={}; hiMap=[];
  const dd=sc.day?dtxt(sc.day+"T12:00").slice(0,-7):"—", nO=sc.items.filter(i=>i.legs.some(c=>c.v)).length;
  $("parlSub").innerHTML=sc.items.length?`Se escanean los partidos de <b>todas las ligas</b> del ${esc(dd)} que aún no empiezan (${sc.items.length} partidos). Cambia el día arriba. ${nO?`Hay cuotas cargadas en ${nO} de ellos.`:"No hay cuotas cargadas: sin cuotas no se puede medir el valor (cárgalas en Apostar con el botón de The Odds API, o escríbelas aquí)."}`
    :`No hay partidos pendientes el ${esc(dd)} en ninguna liga. Cambia el día arriba.`;
  if(!sc.items.length){$("parlPlay").innerHTML="";}
  else{
    const P=pickOfDay(sc.legs), C=comboOfDay(sc.items), D=dreamPool(sc.items), S=dreamSets(D.pool);
    $("parlPlay").innerHTML=pickHtml(P)+comboHtml(C)+dreamHtml(D,S);
  }
  renderMine(); showFeat();
  hiMap.forEach((_,i)=>hiUpdate(i));
  document.querySelectorAll("#parlPlay [data-parl]").forEach(el=>{const id=el.dataset.parl, legs=parlSets[id]; if(legs)parlUpdate(id,legs,!!parlDream[id]);});
}
let parlTimer=null;
function schedParl(){if(parlTimer)return; parlTimer=setTimeout(()=>{parlTimer=null; try{renderParl();}catch(e){}},0);}
function showFeat(){
  $("parlSub").hidden=featTab!=="play"; $("parlPlay").hidden=featTab!=="play"; $("parlMine").hidden=featTab!=="mine";
  $("fParl").hidden=featTab==="trend"; $("fTrend").hidden=featTab!=="trend";
}
function renderMine(){
  const box=$("parlMine");
  if(!parlay.length){box.innerHTML=`<h3>🧩 Mi parlay</h3><div class="sub">Aquí se acumulan las selecciones que agregues con "+ Agregar a mi Parlay" desde cualquier partido (pestaña Resumen) o desde Jugadas del día. Una por partido.</div>`; return;}
  const many=parlay.length>=2;
  box.innerHTML=parlCard("mine","🧩 Mi parlay ("+parlay.length+")",many?"Probabilidades guardadas al agregar cada selección.":"Agrega al menos una selección más de otro partido.",parlay,{rm:true,foot:`<div class="row2"><button class="mini" type="button" id="parlClear">Vaciar</button></div>`}).replace('<div class="box parl" data-parl="mine">','<div class="parl" data-parl="mine">');
  if(many)parlUpdate("mine",parlay); else box.querySelectorAll(".pst,.pcalc,#pst_mine").forEach(e=>e.remove());
}
$("fParl").addEventListener("input",ev=>{
  const t=ev.target, d=t.dataset||{};
  if(d.hi!==undefined){
    const m=hiMap[+d.hi]; if(!m)return;
    const v=parseFloat(t.value.replace(",","."));
    if(m.key===userMatch){if(v>1)userOdds[m.rk]=v; else delete userOdds[m.rk]; memSave(m.key,userOdds);}
    else{const o=Object.assign({},oddsMem[m.key]); if(v>1)o[m.rk]=v; else delete o[m.rk]; memSave(m.key,o);}
    hiUpdate(+d.hi); return;
  }
  const id=d.pid; if(!id)return;
  parlOdds[id]=t.value;
  parlUpdate(id,id==="mine"?parlay:parlSets[id]||[],!!parlDream[id]);
});
$("fParl").addEventListener("change",ev=>{                     // al terminar de escribir una cuota se reordena el pick (si el cursor pasó a otro campo, espera)
  if(!(ev.target.dataset&&ev.target.dataset.hi!==undefined))return;
  setTimeout(()=>{const a=document.activeElement; if(a&&a.matches&&a.matches("#fParl input"))return; renderParl();},300);
});
$("fParl").addEventListener("click",ev=>{
  const t=ev.target;
  if(t.dataset.use){const legs=parlSets[t.dataset.use]||[]; legs.forEach(l=>addLeg(l)); renderMine(); parlN(); t.textContent="✓ Agregado a tu parlay"; return;}
  if(t.dataset.rm){parlay=parlay.filter(x=>x.id!==t.dataset.rm); saveParlay(); parlN(); renderMine(); renderPicks(); return;}
  if(t.id==="parlClear"){parlay=[]; saveParlay(); parlN(); renderMine(); renderPicks();}
});
document.querySelector(".psg").addEventListener("click",ev=>{
  const b=ev.target.closest("button[data-ft]"); if(!b)return;
  featTab=b.dataset.ft;
  document.querySelectorAll(".psg button").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));
  showFeat(); if(featTab!=="trend")schedParl();
});
parlN();

/* ---------- Calculadora Props (Poisson) ---------- */
const numIn=el=>parseFloat(String(el.value).replace(",","."));
function propSide(side,label,p,push,fairOdds){
  return `<div class="mtop"><span class="ml">${esc(label)}</span><span class="mp" id="pp_${side}">${pct(p)}</span></div>
    <div class="pbar"><i style="width:${(p*100).toFixed(0)}%"></i></div>
    <div class="mbot"><span class="mf">Cuota justa ${f2(fairOdds)}</span><input inputmode="decimal" id="pi_${side}" data-side="${side}" placeholder="Tu cuota" aria-label="Tu cuota para ${esc(label)}"><span class="ev" id="pe_${side}" title="Ganancia esperada por cada L 1 apostado"></span><button class="mini" id="pa_${side}" data-side="${side}" disabled type="button">Anotar</button></div>
    <div class="stk" id="ps_${side}"></div>`;
}
let prop=null;     // último cálculo válido: {name, mu, line, o:{p,label}, u:{p,label}, push}
function propCalc(){
  const mu=numIn($("pAvg")), line=numIn($("pLine")), msg=$("pMsg"), res=$("pRes");
  const bad=t=>{prop=null; res.hidden=true; msg.textContent=t;};
  if(!$("pAvg").value.trim()||!$("pLine").value.trim())return bad("Escribe el promedio por partido y la línea (1.5, 2.5…) para ver las probabilidades.");
  if(!(mu>0)||mu>60)return bad("El promedio esperado debe ser un número mayor que 0 (y razonable, por ejemplo 2.5).");
  if(!(line>0)||line>100||Math.abs(line*2-Math.round(line*2))>1e-9)return bad("La línea debe terminar en .5 o ser un número entero (por ejemplo 1.5 o 2). Las líneas de cuarto, como 1.25, no están soportadas.");
  const half=Math.abs(line-Math.floor(line)-0.5)<1e-9, k=Math.floor(line);
  const a=pois(mu,k+1);
  const under=a.slice(0,half?k+1:k).reduce((x,y)=>x+y,0), push=half?0:a[k], over=Math.max(0,1-under-push);
  if(!(over>1e-9&&under>1e-9))return bad("Con ese promedio y esa línea una de las dos opciones es casi imposible; revisa los números.");
  const name=$("pName").value.trim()||"Prop sin nombre", ln=String(line);
  const fair=(p)=>push>0?1+(1-p-push)/p:1/p;           // con devolución: p·(cuota−1) = prob. de perder
  prop={name,mu,line,push,o:{p:over,label:"Más de "+ln},u:{p:under,label:"Menos de "+ln}};
  msg.textContent=name+": promedio "+mu+", línea "+ln+". Probabilidades de Poisson.";
  res.hidden=false;
  // se vuelve a dibujar solo si cambia algo que afecta las probabilidades (así no se pierde el foco al escribir tu cuota)
  const keep={o:$("pi_o")&&$("pi_o").value,u:$("pi_u")&&$("pi_u").value};
  $("pr_o").innerHTML=propSide("o",prop.o.label,over,push,fair(over)); $("pr_u").innerHTML=propSide("u",prop.u.label,under,push,fair(under));
  ["o","u"].forEach(sd=>{if(keep[sd])$("pi_"+sd).value=keep[sd]; propOdds(sd);});
  $("pPush").textContent=push>0?"Línea entera: si sale exactamente "+k+" ("+(push*100).toFixed(1)+"%), la casa devuelve la apuesta. La cuota justa ya lo descuenta. No se puede anotar en la bitácora porque ahí no existe la devolución.":"";
}
function propOdds(sd){
  if(!prop)return;
  const side=prop[sd], v=numIn($("pi_"+sd)), e=$("pe_"+sd), b=$("pa_"+sd), sk=$("ps_"+sd);
  if(!(v>1)){e.textContent="";e.className="ev";sk.innerHTML="";sk.className="stk";b.disabled=true;return;}
  const ev=side.p*(v-1)-(1-side.p-prop.push);
  e.className="ev "+(ev>1e-12?"pos":"neg");
  e.textContent=(ev>0?"+":"")+(ev*100).toFixed(1)+"%";
  const t=stakeHtml(side.p,v,prop.push); sk.className=t.cls;
  sk.innerHTML=t.html+(ev>0.25?`<span class="flag" style="margin-top:6px">Ganancia esperada muy alta: casi siempre significa que el promedio está mal estimado o que la casa tiene información que tú no (lesión, minutos, alineación). Revisa el promedio.</span>`:"");
  b.disabled=prop.push>0;
  b.title=prop.push>0?"Con línea entera hay devolución y la bitácora no la maneja":"";
}
["pName","pAvg"].forEach(id=>$(id).addEventListener("input",propCalc));
$("pLine").addEventListener("input",()=>{["pi_o","pi_u"].forEach(id=>{const e=$(id); if(e)e.value="";}); propCalc();});   // otra línea = otras cuotas de la casa
$("pRes").addEventListener("input",ev=>{if(ev.target.dataset.side)propOdds(ev.target.dataset.side);});
$("pRes").addEventListener("click",ev=>{
  const b=ev.target.closest("button[data-side]"); if(!b||b.disabled||!prop)return;
  const sd=b.dataset.side, v=numIn($("pi_"+sd)); if(!(v>1))return;
  const log=readLog();
  log.unshift({id:Date.now(),d:new Date().toISOString().slice(0,10),lg:"Props",match:prop.name,mk:prop[sd].label+" (promedio "+prop.mu+")",odds:v,p:prop[sd].p,stake:1,st:"pending"});
  writeLog(log); renderLog();
  b.textContent="Anotada"; setTimeout(()=>{b.textContent="Anotar";},1500);
});

/* ---------- The Odds API: cuotas del mercado desde el navegador ---------- */
const ODDS_SPORT={bundesliga:"soccer_germany_bundesliga",premier:"soccer_epl",laliga:"soccer_spain_la_liga",eredivisie:"soccer_netherlands_eredivisie",noruega:"soccer_norway_eliteserien",arabia:"soccer_saudi_arabia_pro_league",mls:"soccer_usa_mls",ligamx:"soccer_mexico_ligamx",ucl:"soccer_uefa_champs_league"};
const KEYA="scoutpicks.oddsapikey.v1", KEYC="scoutpicks.oddsapi.v1", ODDS_TTL=30*60*1000;
let apiStore={}, memKeyA="", memCacheA={};
function getApiKey(){try{return localStorage.getItem(KEYA)||memKeyA;}catch(e){return memKeyA;}}
function setApiKey(k){memKeyA=k;try{if(k)localStorage.setItem(KEYA,k);else localStorage.removeItem(KEYA);}catch(e){}}
function readApiCache(){try{return JSON.parse(localStorage.getItem(KEYC)||"{}");}catch(e){return memCacheA;}}
function writeApiCache(c){memCacheA=c;try{localStorage.setItem(KEYC,JSON.stringify(c));}catch(e){}}

/* --- emparejar nombres: sin acentos ni FC/SC/CF…, alias conocidos y coincidencia parcial --- */
/* ODDSAPI-MATCH-BEGIN */
const TN_NOISE=new Set("fc sc cf ac as afc fk sk bk if il cd ud rcd ssc ca sv tsg vfb vfl fsv bsc club de del calcio cp kv the 1 04 05 96 98 1899 1846 1848 1909 1907".split(" "));
const TN_TOK={utd:"united",man:"manchester",nottm:"nottingham",nott:"nottingham",sp:"sporting",munchen:"munich",lisboa:"lisbon",internazionale:"inter",koln:"cologne",kiev:"kyiv",ny:"new york",st:"saint"};
const TN_ALIAS={"ath madrid":"atletico madrid","ath bilbao":"athletic","athletic bilbao":"athletic","athletic club":"athletic","sociedad":"real sociedad","betis":"real betis","espanol":"espanyol","vallecano":"rayo vallecano","celta":"celta vigo","tottenham":"tottenham hotspur","wolves":"wolverhampton wanderers","newcastle":"newcastle united","west ham":"west ham united","leicester":"leicester city","leeds":"leeds united","brighton":"brighton and hove albion","west brom":"west bromwich albion","dortmund":"borussia dortmund","leverkusen":"bayer leverkusen","ein frankfurt":"eintracht frankfurt","mgladbach":"borussia monchengladbach","gladbach":"borussia monchengladbach","monchengladbach":"borussia monchengladbach","bayern munich":"bayern","bayern munchen":"bayern","inter milan":"inter","ac milan":"milan","sporting cp":"sporting lisbon","psv eindhoven":"psv","paris saint germain":"psg","paris sg":"psg","olympique marseille":"marseille","olympique lyonnais":"lyon","red star belgrade":"crvena zvezda","dynamo kiev":"dynamo kyiv","copenhagen":"kobenhavn","fc copenhagen":"kobenhavn","salzburg":"red bull salzburg","rb salzburg":"red bull salzburg","club brugge":"brugge","guadalajara chivas":"guadalajara","chivas":"guadalajara","tigres uanl":"tigres","pumas unam":"unam pumas","la galaxy":"los angeles galaxy","lafc":"los angeles fc","dc united":"dc united","oviedo":"real oviedo","valladolid":"real valladolid","zaragoza":"real zaragoza","vitoria guimaraes":"guimaraes","sheff utd":"sheffield united","sheff wed":"sheffield wednesday","qpr":"queens park rangers","hamburg":"hamburger","hamburger sv":"hamburger","bremen":"werder bremen","fortuna dusseldorf":"dusseldorf","kaiserslautern":"kaiserslautern","stuttgart":"vfb stuttgart"};
function tnNorm(x){
  let s=String(x||"").toLowerCase().replace(/ø/g,"o").replace(/æ/g,"ae").replace(/ß/g,"ss").replace(/ł/g,"l").replace(/đ/g,"d").normalize("NFD").replace(/[\u0300-\u036f]/g,"").replace(/&/g," and ").replace(/['’.]/g,"").replace(/[^a-z0-9]+/g," ").trim();
  s=TN_ALIAS[s]||s;
  let t=s.split(" ").filter(w=>w&&!TN_NOISE.has(w));
  s=t.join(" "); s=TN_ALIAS[s]||s;
  t=[]; s.split(" ").forEach(w=>{if(!w||TN_NOISE.has(w))return; const r=TN_TOK[w]||w; r.split(" ").forEach(z=>{if(!t.includes(z))t.push(z);});});
  return t;
}
function lev(a,b){const m=a.length,n=b.length; if(!m)return n; if(!n)return m; let p=Array.from({length:n+1},(_,i)=>i);
  for(let i=1;i<=m;i++){const c=[i]; for(let j=1;j<=n;j++)c[j]=Math.min(p[j]+1,c[j-1]+1,p[j-1]+(a[i-1]===b[j-1]?0:1)); p=c;} return p[n];}
function tokEq(x,y){return x===y||(x.length>=6&&y.length>=6&&lev(x,y)/Math.max(x.length,y.length)<=0.15);}
function teamScore(a,b){                       // 0..1: 1 = mismo nombre; ≥0.8 se considera el mismo equipo
  const A=tnNorm(a), B=tnNorm(b); if(!A.length||!B.length)return 0;
  const sm=A.length<=B.length?A:B, lg=A.length<=B.length?B:A, used=new Set(); let hit=0;
  sm.forEach(x=>{const j=lg.findIndex((y,k)=>!used.has(k)&&tokEq(x,y)); if(j>=0){used.add(j);hit++;}});
  if(hit===sm.length)return sm.length===lg.length?1:(sm.some(w=>w.length>=4)?0.85:0.5);   // todas las palabras del nombre corto están en el largo ("Tottenham" ⊂ "Tottenham Hotspur")
  return hit/Math.max(A.length,B.length)*0.9;
}
function bestTeam(name,cands){let b=null,bs=0; cands.forEach(c=>{const v=teamScore(name,c); if(v>bs){bs=v;b=c;}}); return bs>=0.8?b:null;}
function findApiEvent(events,h,a,whenMs){
  let best=null,bd=Infinity;
  (events||[]).forEach(ev=>{
    const d1=Math.min(teamScore(h,ev.home),teamScore(a,ev.away)), d2=Math.min(teamScore(h,ev.away),teamScore(a,ev.home));
    const sc=Math.max(d1,d2); if(sc<0.8)return;
    const dist=whenMs?Math.abs(Date.parse(ev.t)-whenMs):Date.parse(ev.t);
    if(dist<bd){bd=dist;best={ev:ev,flip:d2>d1};}
  });
  return best;
}
/* ODDSAPI-MATCH-END */
/* --- cuotas del evento: Pinnacle como referencia; si no está, promedio de las casas --- */
function compactApiEvent(ev){
  const books=ev.bookmakers||[], home=ev.home_team, away=ev.away_team, pin=books.find(b=>b.key==="pinnacle");
  const mk=(b,key)=>(b.markets||[]).find(m=>m.key===key);
  const h2h=b=>{const m=mk(b,"h2h"); if(!m)return null; const g=n=>{const o=(m.outcomes||[]).find(x=>x.name===n);return o&&o.price>1?o.price:null;};
    const r={"1":g(home),"X":g("Draw"),"2":g(away)}; return r["1"]&&r["X"]&&r["2"]?r:null;};
  const tot=(b,x)=>{const m=mk(b,"totals"); if(!m)return null; const f=n=>{const o=(m.outcomes||[]).find(z=>z.name===n&&Math.abs(z.point-x)<1e-9);return o&&o.price>1?o.price:null;};
    const o=f("Over"),u=f("Under"); return o&&u?{o:o,u:u}:null;};
  const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
  const odds={}, used={pin:0,avg:0}; let nAvg=0;
  const one=(get,put)=>{ let r=pin?get(pin):null, from="pin";
    if(!r){const l=books.map(get).filter(Boolean); if(l.length){r=put.avg(l); from="avg"; nAvg=Math.max(nAvg,l.length);}}
    if(r){put.set(r); used[from]++;} };
  one(h2h,{avg:l=>({"1":mean(l.map(z=>z["1"])),"X":mean(l.map(z=>z["X"])),"2":mean(l.map(z=>z["2"]))}),set:r=>{odds["1"]=r["1"];odds["X"]=r["X"];odds["2"]=r["2"];}});
  [1.5,2.5,3.5].forEach(x=>one(b=>tot(b,x),{avg:l=>({o:mean(l.map(z=>z.o)),u:mean(l.map(z=>z.u))}),set:r=>{odds["o:goals:"+x]=r.o;odds["u:goals:"+x]=r.u;}}));
  Object.keys(odds).forEach(k=>{odds[k]=Math.round(odds[k]*100)/100;});
  const src=!used.pin&&!used.avg?"":used.pin&&!used.avg?"Pinnacle":!used.pin?"promedio de "+nAvg+(nAvg===1?" casa":" casas"):"Pinnacle; promedio de las demás casas donde faltó";
  return {home:home,away:away,t:ev.commence_time,odds:odds,src:src};
}
function agoTxt(ts){const m=Math.round((Date.now()-ts)/60000); return m<1?"hace menos de 1 minuto":m<60?"hace "+m+" min":"hace "+Math.round(m/60)+" h";}
function apiMsg(t,cls){const e=$("apiSt"); e.className="sub"+(cls?" "+cls:""); e.innerHTML=t;}
function apiRemTxt(c){const e=$("apiRem"); e.textContent=c&&c.remaining!=null&&c.remaining!==""?"Consultas gratis que te quedan este mes: "+c.remaining+" (cada carga cuesta 2).":(c?"No pude leer cuántas consultas te quedan (el navegador no deja ver ese dato); míralo en tu panel de the-odds-api.com.":"");}
function keyStTxt(){const k=getApiKey(); $("apiKeySt").textContent=k?"Clave guardada en este navegador (termina en …"+k.slice(-4)+").":"Todavía no hay clave guardada."; $("apiBox").open=!k;}
async function apiLoad(force){
  const key=getApiKey(), btn=$("apiLoad");
  if(!key){$("apiBox").open=true; return apiMsg("Primero pega tu API Key arriba y toca Guardar.","err");}
  const sport=ODDS_SPORT[league];
  if(!sport)return apiMsg("Esta liga no tiene cuotas en The Odds API.","err");
  const h=$("home").value, a=$("away").value; if(!h||!a||h===a)return;
  const cache=readApiCache(); let c=cache[sport], used="";
  btn.disabled=true; btn.textContent="Cargando…";
  try{
    if(!force&&c&&Date.now()-c.ts<ODDS_TTL){used="cache";}
    else{
      let res;
      try{res=await fetch("https://api.the-odds-api.com/v4/sports/"+encodeURIComponent(sport)+"/odds/?apiKey="+encodeURIComponent(key)+"&regions=eu&markets=h2h,totals&oddsFormat=decimal");}
      catch(e){ if(c){used="stale";} else return apiMsg("No pude conectar con The Odds API (sin internet o conexión bloqueada). Intenta de nuevo.","err"); }
      if(res){
        if(res.status===401||res.status===403){return apiMsg("The Odds API no aceptó la clave. Revisa que la copiaste completa y que sigue activa.","err");}
        if(res.status===429){return apiMsg("The Odds API dice que hiciste demasiadas consultas seguidas o se acabaron tus créditos. Espera un momento.","err");}
        if(res.status===404||res.status===422){return apiMsg("The Odds API no tiene esta liga ("+esc(sport)+") o no acepta la consulta.","err");}
        if(!res.ok){return apiMsg("The Odds API respondió con un error ("+res.status+"). Intenta más tarde.","err");}
        const rem=res.headers.get("x-requests-remaining"), data=await res.json();
        c={ts:Date.now(),remaining:rem,events:(Array.isArray(data)?data:[]).map(compactApiEvent)};
        cache[sport]=c; writeApiCache(cache);
      }
    }
    apiRemTxt(c);
    const fx0=(DATA.fixtures||[]).find(f=>f.lg===league&&f.h===h&&f.a===a&&!f.score);
    const hit=findApiEvent(c.events,h,a,fx0?Date.parse(fx0.date+":00-06:00"):0);
    if(!hit)return apiMsg(`No encontré <b>${esc(h)} vs ${esc(a)}</b> entre los ${c.events.length} partidos con cuotas de The Odds API (${used==="cache"?"guardados "+agoTxt(c.ts):used==="stale"?"guardados "+agoTxt(c.ts)+", sin conexión":"recién cargados"}). Puede que todavía no tenga cuotas o que el partido ya empezó.`,"err");
    const o=Object.assign({},hit.ev.odds);
    if(hit.flip){const t=o["1"];o["1"]=o["2"];o["2"]=t;}      // la API tiene a los equipos en el orden contrario
    if(!Object.keys(o).length)return apiMsg("The Odds API tiene el partido pero sin cuotas de resultado ni de goles por ahora.","err");
    const merged=Object.assign({},fx0&&fx0.odds,o), ml=mlFromOdds(merged);
    apiStore[league+"|"+h+"|"+a]={odds:o,ml:ml,src:hit.ev.src,date:new Date(Date.parse(hit.ev.t)-6*3600e3).toISOString().slice(0,16)};
    render();
    const lines=[1.5,2.5,3.5].filter(x=>o["o:goals:"+x]).join(", ");
    apiMsg(`<b>✔ Cuotas cargadas</b> (${esc(hit.ev.src)}): resultado${lines?" y goles (líneas "+lines+")":""}. `+(used==="cache"?`Salen de lo guardado ${agoTxt(c.ts)}, así que no gastó créditos. <button class="linkb" id="apiForce" type="button">Actualizar ahora (gasta 2)</button>`:used==="stale"?`Sin conexión: uso lo guardado ${agoTxt(c.ts)}.`:"Consulta nueva.")+`<br>Las probabilidades ahora salen de estas cuotas, así que la ganancia esperada sale casi siempre negativa: es el margen de la casa. Escribe la cuota de tu casa en cada fila para ver si paga mejor.`,"ok");
  }finally{btn.disabled=false; btn.textContent="🔄 Cargar cuotas del mercado";}
}
$("apiLoad").addEventListener("click",()=>apiLoad(false));
$("apiSt").addEventListener("click",ev=>{if(ev.target.id==="apiForce")apiLoad(true);});
$("apiSave").addEventListener("click",()=>{
  const v=$("apiKey").value.trim();
  if(!/^[A-Za-z0-9]{16,64}$/.test(v)){$("apiKeySt").textContent="Esa clave no parece válida: son solo letras y números, sin espacios."; return;}
  setApiKey(v); $("apiKey").value=""; keyStTxt(); $("apiBox").open=false;
});
$("apiDel").addEventListener("click",()=>{setApiKey(""); $("apiKey").value=""; writeApiCache({}); apiStore={}; keyStTxt(); apiMsg("",""); apiRemTxt(null);});
keyStTxt(); { const c0=readApiCache(), ks=Object.keys(c0); if(ks.length){const last=ks.map(k=>c0[k]).sort((x,y)=>y.ts-x.ts)[0]; apiRemTxt(last);} }

/* ---------- pantalla del partido ---------- */
function updTxt(){
  const L=DATA.leagues[league];
  $("upd").innerHTML=(L.kind==="elo"?"Ranking Elo del "+esc(L.updated):"Datos hasta "+esc(L.updated))+"<br>Actualizado "+esc((DATA.generated||"—").slice(0,16));
}
function fitName(el){const n=el.textContent.length; el.style.fontSize=(n<=7?34:n<=10?29:n<=13?25:21)+"px";}
/* ---------- 🚑 Bajas y Sanciones Clave (manual y local) ---------- */
const BJ=[["s","⭐ Baja Estrella Ofensiva (-15% goles esperados / -35 Elo)"],["m","🧠 Baja Mediocampo Clave (-8% goles esperados / +4% goles rival / -20 Elo)"],["d","🛡️ Baja Defensa Titular o Portero (+12% goles rival / -25 Elo)"]];
let baj={h:{s:0,m:0,d:0},a:{s:0,m:0,d:0}};
function bajReset(){baj={h:{s:0,m:0,d:0},a:{s:0,m:0,d:0}};}
function bajAny(){return ["h","a"].some(k=>baj[k].s||baj[k].m||baj[k].d);}
function bajArgs(L,ml){                       // [factor goles local, factor goles visitante, delta Elo local, delta Elo visitante]
  const H=baj.h, A=baj.a;
  const fh=(H.s?0.85:1)*(H.m?0.92:1)*(A.d?1.12:1)*(A.m?1.04:1), fa=(A.s?0.85:1)*(A.m?0.92:1)*(H.d?1.12:1)*(H.m?1.04:1);
  const dh=-(35*H.s+20*H.m+25*H.d), da=-(35*A.s+20*A.m+25*A.d);
  return (L.kind==="elo"&&!ml)?[1,1,dh,da]:[fh,fa,0,0];   // Champions sin cuotas: el ajuste va al Elo; en los demás casos, a los goles esperados
}
function renderBajas(h,a,base,fin){
  const L=DATA.leagues[league], box=$("bajas");
  if(fin){box.innerHTML="";box.hidden=true;return;}
  box.hidden=false;
  const side=(k,t)=>`<div class="bjt">${esc(t)} <span class="sub">(${k==="h"?"local":"visitante"})</span></div>`+BJ.map(([c,lbl])=>`<button class="bjb" type="button" data-k="${k}" data-c="${c}" aria-pressed="${baj[k][c]?"true":"false"}">${lbl}</button>`).join("");
  let sum="";
  if(bajAny()){
    const [pH,pD,pA]=cur.p, [bH,bD,bA]=base.p, d=(x,y)=>`${pct(y)} → <b>${pct(x)}</b>`;
    sum=`<div class="bjsum"><div>Gana ${esc(h)}: <span class="bf">${d(pH,bH)}</span></div><div>Empate: <span class="bf">${d(pD,bD)}</span></div><div>Gana ${esc(a)}: <span class="bf">${d(pA,bA)}</span></div>
      <div>Goles esperados: ${esc(h)} <span class="bf">${base.lh.toFixed(2)} → <b>${cur.lh.toFixed(2)}</b></span> · ${esc(a)} <span class="bf">${base.la.toFixed(2)} → <b>${cur.la.toFixed(2)}</b></span></div></div>`
      +(L.kind==="elo"&&!cur.fromMkt?`<div class="sub" style="margin-top:8px">En la Champions sin cuotas el ajuste mueve el Elo de cada club (y de ahí los goles esperados).</div>`:"")
      +(cur.fromMkt?`<div class="flag" style="margin-top:8px">Las probabilidades salen de cuotas del mercado, que normalmente ya incluyen las bajas conocidas. Si activas aquí una baja que la casa ya conoce, se cuenta dos veces.</div>`:"");
  }
  box.innerHTML=`<h3>🚑 Bajas y Sanciones Clave</h3><div class="sub">Actívalas a mano si falta alguien importante. Se suman si activas varias y recalculan al instante probabilidades, cuotas justas, valor esperado y stake. Son porcentajes fijos, no están validados con partidos pasados.</div>`+side("h",h)+side("a",a)+sum;
}
$("bajas").addEventListener("click",ev=>{
  const b=ev.target.closest("button[data-k]"); if(!b)return;
  baj[b.dataset.k][b.dataset.c]^=1; render();
});

function render(){
  const h=$("home").value,a=$("away").value,L=DATA.leagues[league];
  updTxt();
  if(h===a)return;
  const fxs=(DATA.fixtures||[]).filter(f=>f.lg===league&&f.h===h&&f.a===a), fx0=fxs.find(f=>!f.score)||fxs[0];
  const af=apiStore[league+"|"+h+"|"+a]&&!(fx0&&fx0.score)?apiStore[league+"|"+h+"|"+a]:null;     // cuotas cargadas desde The Odds API (solo en memoria; no tocan DATA)
  const fx=af?Object.assign({},fx0||{lg:league,h:h,a:a,date:af.date,status:"NS"},{odds:Object.assign({},fx0&&fx0.odds,af.odds),ml:af.ml||(fx0&&fx0.ml)||null}):fx0;
  const fin=fx&&fx.score;
  const mk0=league+"|"+h+"|"+a; if(mk0!==userMatch){userOdds=Object.assign({},oddsMem[mk0]||{});userMatch=mk0;bajReset();}
  const uml=mlFromOdds(userOdds);
  const ml=uml||(fx&&fx.ml?fx.ml:null);
  const [fh,fa,dh,da]=bajArgs(L,ml);
  cur=compute(h,a,fh,fa,ml,dh,da); cur.src=uml?"user":(ml?"fx":null);
  $("hMeta").textContent=L.name+(fin?" · Final "+fx.score[0]+"–"+fx.score[1]:fx?" · "+dtxt(fx.date):"");
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
  $("readout").textContent=fin?"Terminó "+h+" "+fx.score[0]+"–"+fx.score[1]+" "+a+". Ya cuenta en el historial y en la tabla de los dos equipos. Las probabilidades de abajo ya incluyen ese resultado, así que no son un pronóstico previo.":(top[0]==="empate"?"Lo más probable es un empate":"Lo más probable es que gane "+top[0])+" ("+pct(top[1])+"). Es un cálculo con datos pasados, no una garantía.";
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
  if(!cur.fromMkt&&L.kind!=="elo"&&L.models.goles.btb) how.push("\"Ambos anotan\" se acerca a lo que pasó en esta liga en los últimos 2 años ("+Math.round(L.models.goles.btb*100)+"%). Con partidos pasados vimos que el modelo solo exageraba las diferencias entre partidos en este mercado.");
  if(!cur.fromMkt&&L.kind!=="elo"&&L.val!==false) how.push("Rango típico de error del modelo: ±8 puntos en el 1X2 (en 2.300 partidos pasados, la diferencia con las casas fue menor a eso el 80% de las veces).");
  if(L.val===false&&!cur.fromMkt) how.push("Esta liga se agregó hace poco y el modelo no se ha validado con partidos pasados como en Bundesliga, Premier y La Liga, así que el error puede ser mayor. "+(L.models.corners?"":"Solo hay modelo de goles: no tenemos córners, tiros ni tarjetas de esta liga."));
  $("howTxt").innerHTML=how.map(x=>`<p>${esc(x)}</p>`).join("");
  renderBajas(h,a,bajAny()?compute(h,a,1,1,ml):null,fin);
  renderHist(h,a); renderStats(h,a,fx); renderTable(h,a);
  const ex=$("expect"); ex.innerHTML=expectHtml(h,a,cur,fin); ex.hidden=!ex.innerHTML;
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
      <div class="stk" id="s${i}"></div>
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
  bankMsg();
  renderPicks();
  $("oddsSrc").textContent=(af?`Cuotas cargadas con The Odds API (${af.src}). Cámbialas por las de tu casa. `:n?`Cargamos ${n} cuotas automáticamente (promedio de varias casas). Cámbialas por las de tu casa. `:"")+"Cuota justa: la mínima para que valga la pena. Número verde o rojo: lo que ganarías o perderías en promedio por cada L 1 si el modelo acertara.";
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
    return `<button class="day" type="button" data-d="${k}" aria-pressed="${k===day}"><b>${p.dow} ${p.d} ${p.mon}</b><small>${fxOfLeague().filter(f=>f.date.slice(0,10)===k).every(f=>f.score)?(n===1?"resultado":"resultados"):n+" "+(n===1?"partido":"partidos")}</small></button>`;}).join("");
  $("fxList").innerHTML=list.length?list.map(f=>{
    const fin=!!f.score, p=fin?null:quick(f), q=fin?`<span class="msc"><b>${f.score[0]}</b><span>–</span><b>${f.score[1]}</b></span>`:p?`<span class="mq"><span class="qb"><i style="flex-grow:${p[0].toFixed(3)}"></i><i style="flex-grow:${p[1].toFixed(3)}"></i><i style="flex-grow:${p[2].toFixed(3)}"></i></span><span class="qn"><span>${pct(p[0])}</span><span>${pct(p[1])}</span><span>${pct(p[2])}</span></span></span>`:"<span></span>";
    return `<button class="mc${fin?" fin":""}" type="button" data-f="${DATA.fixtures.indexOf(f)}"><span class="mtime">${fin?"FIN":esc(f.date.slice(11,16))}</span><span class="mtt"><span>${esc(f.h)}</span><span>${esc(f.a)}</span></span>${q}<span class="chev"></span></button>`;}).join("")
    :`<div class="empty">No hay partidos próximos de ${esc(DATA.leagues[league].name)} en la fuente de datos. Puedes elegir los equipos abajo.</div>`;
  $("manual").open=!list.length;
}
function setView(v){
  if(v!=="match")prevView=v;
  view=v;
  [["list","vList"],["feat","vFeat"],["table","vTable"],["log","vLog"],["props","vProps"],["match","vMatch"]].forEach(([k,id])=>{$(id).hidden=k!==v;});
  $("ctx").hidden=!(v==="list"||v==="feat"||v==="table");
  $("dayChips").hidden=v==="table"||!daysOfLeague().length;
  if(v==="table")renderTableView();
  if(v==="feat"){showFeat(); if(featTab!=="trend")schedParl();}
  document.querySelectorAll(".nav button").forEach(b=>b.setAttribute("aria-current",b.dataset.v===(v==="match"?prevView:v)?"page":"false"));
  window.scrollTo(0,0);
}
function setTab(t){
  if(oddsDirty&&view==="match"&&t!==tab){oddsDirty=false;render();}   // al cambiar de pestaña, se recalculan las probabilidades con las cuotas escritas
  tab=t;
  document.querySelectorAll("#mTabs button").forEach(b=>b.setAttribute("aria-selected",b.dataset.t===t?"true":"false"));
  [["sum","pSum"],["hist","pHist"],["trend","pTrend"],["table","pTable"],["mk","pMk"]].forEach(([k,id])=>{$(id).hidden=k!==t;});
  const mh=document.querySelector(".mhead"), y=mh.getBoundingClientRect().bottom+window.scrollY-8;
  if(window.scrollY>y)window.scrollTo(0,y);
}
function openMatch(h,a,t){
  $("home").value=h; $("away").value=a; bajReset();
  histTeam="h";
  // Champions: si algún equipo tiene menos de 4 partidos de Champions en los datos, se abre en "Todos" (liga local + Champions)
  const nU=t=>{const c=uclName(t);return c?uclMatches(c,4).length:0;};
  scope=league==="ucl"&&nU(h)>=4&&nU(a)>=4?"ucl":"all";
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
  league=k; day=null; tabs(); fillTeams(); updTxt(); fxList(); renderFeat(); if(view==="table")renderTableView();
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
/* Cuotas de Resumen <-> Tu cuota de Apostar: se copian al instante (no hace falta salir del campo) */
const SYNC={"1":"ou1","X":"ouX","2":"ou2","o:goals:2.5":"ouO","u:goals:2.5":"ouU"};
let oddsDirty=false;
$("pSum").addEventListener("input",ev=>{
  const k=ev.target.dataset&&ev.target.dataset.k; if(!k)return;
  const v=parseFloat(ev.target.value.replace(",","."));
  if(v>1)userOdds[k]=v; else delete userOdds[k];
  memSave(userMatch,userOdds);
  oddsDirty=true;
  const i=cur?cur.rows.findIndex(r=>r.key===k):-1, el=i>=0?$("o"+i):null;
  if(el){el.value=ev.target.value; onOdds(i);}
});
$("pSum").addEventListener("change",ev=>{if(ev.target.dataset&&ev.target.dataset.k)render();});
$("mk").addEventListener("input",ev=>{
  if(ev.target.dataset.i===undefined||ev.target.tagName!=="INPUT")return;
  const i=+ev.target.dataset.i, v=parseFloat(ev.target.value.replace(",","."));
  const rk=rowKey(cur.rows[i]);
  if(v>1)userOdds[rk]=v; else delete userOdds[rk];
  memSave(userMatch,userOdds);
  onOdds(i);
  if(SYNC[rk]){oddsDirty=true; $(SYNC[rk]).value=ev.target.value;}
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


league=((DATA.fixtures||[]).find(f=>!f.score)||(DATA.fixtures||[])[0]||{}).lg||Object.keys(DATA.leagues)[0];
tabs(); fillTeams(); updTxt(); fxList(); renderFeat(); renderLog(); setView("list");
</script>
<script>
/* App instalable (PWA): service worker + aviso de instalación (Android: botón; iPhone/iPad: pasos de Safari) */
(function(){
  try{if("serviceWorker" in navigator&&location.protocol.startsWith("http"))window.addEventListener("load",function(){navigator.serviceWorker.register("sw.js").catch(function(){});});}catch(e){}
  var box=document.getElementById("inst"); if(!box)return;
  var KEYX="pf_inst_x", standalone=false, ios=false, closed=false;
  try{standalone=(window.matchMedia&&matchMedia("(display-mode: standalone)").matches)||navigator.standalone===true;}catch(e){}
  try{ios=/iphone|ipad|ipod/i.test(navigator.userAgent)||(navigator.platform==="MacIntel"&&navigator.maxTouchPoints>1);}catch(e){}
  try{closed=localStorage.getItem(KEYX)==="1";}catch(e){}
  if(standalone||closed)return;
  var txt=document.getElementById("instTxt"), btn=document.getElementById("instBtn"), dp=null;
  function show(){box.hidden=false;}
  document.getElementById("instX").addEventListener("click",function(){box.hidden=true;try{localStorage.setItem(KEYX,"1");}catch(e){}});
  if(ios){
    txt.textContent="En Safari toca el botón Compartir (el cuadro con la flecha) y elige \"Añadir a pantalla de inicio\".";
    show();
  }
  window.addEventListener("beforeinstallprompt",function(ev){
    ev.preventDefault(); dp=ev; txt.textContent="Ábrela desde tu pantalla de inicio, a pantalla completa y más rápido."; btn.hidden=false; show();
  });
  btn.addEventListener("click",function(){
    if(!dp)return; dp.prompt(); dp.userChoice.then(function(){dp=null;box.hidden=true;},function(){});
  });
  window.addEventListener("appinstalled",function(){box.hidden=true;});
})();
</script>

</body></html>
'''


if __name__ == "__main__":
    build(TEMPLATE)
