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
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
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
        note = "La API no respondió (revisa la clave o el plan). La página usa solo el modelo y tus cuotas."
    elif not fixtures:
        note = "La API respondió, pero no hay partidos de estas ligas hoy ni mañana."
    else:
        note = f"{len(fixtures)} partidos próximos. Cuotas cargadas en {n_odds}."
    return fixtures, note


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
    fixtures, note = api_enrich(params, today)
    log(note)
    data = {"leagues": params, "fixtures": fixtures, "note": note,
            "generated": local.strftime("%Y-%m-%d %H:%M") + " (hora de Honduras)"}
    html = template.replace("__PARAMS__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    log(f"Página escrita en {out_path} ({len(html) // 1024} KB)")


TEMPLATE = r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light}body{margin:0}</style>
<title>Scout Picks</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap">
<style>
/* Layout: columna única tipo marcador de estadio; selector de partido arriba, mercados en tabla, bitácora al final */
:root{
  --bg:#eef1ef; --card:#ffffff; --ink:#14201b; --muted:#5d6b64; --line:#d5ddd8;
  --accent:#0d6b4b; --accent-ink:#ffffff; --good:#0d6b4b; --warn:#a86a00; --bad:#b3261e;
  --warn-bg:#fbf0d9; --bad-bg:#f8e0de; --good-bg:#dcefe6; --bar:#cfd8d3;
  --disp:"Barlow Condensed","Arial Narrow",sans-serif; --body:"Barlow",system-ui,sans-serif;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e1512; --card:#161f1b; --ink:#e7eee9; --muted:#97a69e; --line:#2a3731;
  --accent:#3fc08e; --accent-ink:#06120d; --good:#3fc08e; --warn:#e0a53a; --bad:#ef7b72;
  --warn-bg:#352a10; --bad-bg:#3a1a18; --good-bg:#13302a; --bar:#2a3731; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e1512; --card:#161f1b; --ink:#e7eee9; --muted:#97a69e; --line:#2a3731;
  --accent:#3fc08e; --accent-ink:#06120d; --good:#3fc08e; --warn:#e0a53a; --bad:#ef7b72;
  --warn-bg:#352a10; --bad-bg:#3a1a18; --good-bg:#13302a; --bar:#2a3731; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.45 var(--body);padding-inline:16px;padding-block:16px 40px}
.wrap{max-width:760px;margin:0 auto;display:flex;flex-direction:column;gap:16px}
h1,h2,h3{font-family:var(--disp);margin:0;letter-spacing:.02em;text-wrap:balance}
header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
h1{font-size:34px;font-weight:700;text-transform:uppercase;line-height:1}
h1 span{color:var(--accent)}
.sub{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:16px;display:flex;flex-direction:column;gap:14px;min-width:0}
h2{font-size:20px;font-weight:600;text-transform:uppercase;color:var(--muted)}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tab{font:600 15px var(--disp);letter-spacing:.04em;text-transform:uppercase;padding:7px 14px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:4px;cursor:pointer}
.tab[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.pick{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:end}
.pick label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;min-width:0}
select,input{font:500 16px var(--body);padding:9px 10px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink);width:100%;min-width:0}
select:focus-visible,input:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.vs{font:600 20px var(--disp);color:var(--muted);padding-bottom:8px}
.board{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;text-align:center}
.board .t{font:600 18px var(--disp);text-transform:uppercase;overflow-wrap:anywhere}
.board .xg{font:700 52px/1 var(--disp);font-variant-numeric:tabular-nums}
.board .dash{font:600 30px var(--disp);color:var(--muted)}
.form{display:flex;gap:3px;justify-content:center;margin-top:4px}
.form i{font:600 11px var(--disp);font-style:normal;width:18px;height:18px;display:grid;place-items:center;border-radius:2px;color:#fff}
.form .G{background:var(--good)} .form .E{background:#7b8780} .form .P{background:var(--bad)}
.lbl{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.stack{display:flex;height:34px;border-radius:4px;overflow:hidden;font:600 16px var(--disp);font-variant-numeric:tabular-nums}
.stack div{display:grid;place-items:center;min-width:0;color:var(--ink);background:var(--bar)}
.stack .h{background:var(--accent);color:var(--accent-ink)}
.stack .a{background:color-mix(in srgb,var(--accent) 35%,var(--bar))}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{border:1px solid var(--line);border-radius:4px;padding:6px 10px;font-size:13px}
.chip b{font:700 18px var(--disp);font-variant-numeric:tabular-nums;margin-left:4px}
.tbl{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{font:600 12px var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
tr.sec td{font:600 14px var(--disp);letter-spacing:.06em;text-transform:uppercase;color:var(--accent);padding-top:14px;border-bottom:2px solid var(--line)}
td input{width:76px;padding:5px 6px;font-size:15px;text-align:right}
.ev{font-weight:600}
.ev.pos{color:var(--good)} .ev.neg{color:var(--muted)}
.flag{display:inline-block;font-size:11px;padding:2px 6px;border-radius:3px;background:var(--warn-bg);color:var(--warn);margin-left:6px;white-space:normal}
.btn{font:600 14px var(--disp);letter-spacing:.04em;text-transform:uppercase;padding:6px 10px;border-radius:4px;border:1px solid var(--accent);background:var(--accent);color:var(--accent-ink);cursor:pointer}
.btn.ghost{background:transparent;color:var(--accent)}
.btn:disabled{opacity:.35;cursor:default}
.note{font-size:13px;color:var(--muted);background:var(--warn-bg);border-left:3px solid var(--warn);padding:10px 12px;border-radius:3px}
.note b{color:var(--ink)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px}
.stat{border:1px solid var(--line);border-radius:4px;padding:8px 10px}
.stat b{display:block;font:700 24px var(--disp);font-variant-numeric:tabular-nums}
.stat.p b{color:var(--good)} .stat.n b{color:var(--bad)}
.log{display:flex;flex-direction:column}
.entry{display:grid;grid-template-columns:1fr auto;gap:4px 10px;padding:10px 0;border-bottom:1px solid var(--line)}
.entry .m{font-weight:600;overflow-wrap:anywhere}
.entry .d{font-size:13px;color:var(--muted)}
.entry .act{display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.pill{font:600 13px var(--disp);text-transform:uppercase;padding:3px 8px;border-radius:3px;letter-spacing:.04em}
.pill.won{background:var(--good-bg);color:var(--good)} .pill.lost{background:var(--bad-bg);color:var(--bad)} .pill.pending{background:var(--bar);color:var(--muted)}
.mini{font:600 13px var(--disp);padding:4px 8px;border-radius:3px;border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer;text-transform:uppercase}
.empty{color:var(--muted);padding:6px 0}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.foot{font-size:12px;color:var(--muted)}
.fxlist{display:flex;flex-direction:column;gap:6px}
.fx{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;text-align:left;font:500 15px var(--body);padding:8px 10px;border:1px solid var(--line);border-radius:4px;background:transparent;color:var(--ink);cursor:pointer;width:100%}
.fx .tm{font:600 14px var(--disp);color:var(--muted);letter-spacing:.04em;text-transform:uppercase;grid-row:span 2;align-self:center;min-width:64px}
.fx .lg{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.fx:hover{border-color:var(--accent)}
.inj{font-size:13px;display:flex;flex-direction:column;gap:4px}
.inj b{font-weight:600}
.inj .none{color:var(--muted)}
.adj{display:flex;flex-direction:column;gap:6px}
.adj label{display:flex;flex-direction:column;gap:4px;flex:1;min-width:150px}
@media (max-width:520px){.board{grid-template-columns:1fr 1fr;grid-template-areas:'s s' 'h a'}.bs{grid-area:s}.bh{grid-area:h}.ba{grid-area:a}.board .xg{font-size:48px} .pick{grid-template-columns:1fr} .vs{display:none}}
</style>

</head><body>
<div class="wrap">
  <header>
    <h1>Scout <span>Picks</span></h1>
    <div class="sub" id="upd"></div>
  </header>

  <section class="card" aria-label="Partidos próximos">
    <h2>Partidos próximos</h2>
    <div class="sub" id="fxNote"></div>
    <div class="fxlist" id="fxList"></div>
  </section>

  <section class="card" aria-label="Partido">
    <div class="tabs" id="leagueTabs" role="group" aria-label="Liga"></div>
    <div class="pick">
      <label>Local<select id="home"></select></label>
      <div class="vs">vs</div>
      <label>Visitante<select id="away"></select></label>
    </div>
    <div class="board">
      <div class="bh"><div class="t" id="hn"></div><div class="form" id="hf"></div></div>
      <div class="row bs" style="justify-content:center;gap:10px;flex-wrap:nowrap"><span class="xg" id="xh"></span><span class="dash">–</span><span class="xg" id="xa"></span></div>
      <div class="ba"><div class="t" id="an"></div><div class="form" id="af"></div></div>
    </div>
    <div class="sub" style="text-align:center;margin-top:-6px">Goles esperados según el modelo (promedio, no es un marcador)</div>
    <div class="inj" id="inj"></div>
    <div class="adj">
      <span class="lbl">Ajuste manual por bajas clave (opinión tuya, no validado)</span>
      <div class="row">
        <label class="sub">Local<select id="adjH"><option value="0">Sin ajuste</option><option value="1">1 baja clave (−7% ataque)</option><option value="2">2 bajas clave (−14%)</option><option value="3">3 bajas clave (−21%)</option></select></label>
        <label class="sub">Visitante<select id="adjA"><option value="0">Sin ajuste</option><option value="1">1 baja clave (−7% ataque)</option><option value="2">2 bajas clave (−14%)</option><option value="3">3 bajas clave (−21%)</option></select></label>
      </div>
    </div>
    <div>
      <div class="lbl" style="margin-bottom:6px">Resultado (1X2)</div>
      <div class="stack" id="stack"></div>
    </div>
    <div class="chips" id="chips"></div>
  </section>

  <section class="card" aria-label="Mercados y cuotas">
    <h2>Mercados y cuotas</h2>
    <div class="sub" id="oddsSrc"></div>
    <div class="sub">Escribe la cuota de tu casa de apuestas. Verás la ganancia esperada (EV) por cada L 1 apostado, según el modelo.</div>
    <div class="tbl"><table><thead><tr><th>Mercado</th><th>Prob.</th><th>Cuota justa</th><th>Tu cuota</th><th>EV</th><th></th></tr></thead><tbody id="mk"></tbody></table></div>
    <div class="note"><b>Cómo leerlo.</b> En las pruebas históricas el modelo no superó a las casas de apuestas. Subestima a los favoritos fuertes y a los equipos que dominan córners y tiros. Si el modelo y la cuota difieren por 8 puntos o más, lo normal es que el error sea del modelo. Un EV positivo no es una señal de compra.</div>
  </section>

  <section class="card" aria-label="Bitácora">
    <h2>Bitácora de apuestas</h2>
    <div class="stats" id="stats"></div>
    <div class="log" id="log"></div>
    <div class="row">
      <button class="btn ghost" id="copy" type="button">Copiar bitácora</button>
      <span class="foot" id="copyMsg"></span>
    </div>
    <div class="foot">La bitácora se guarda solo en este navegador. Si borras los datos del sitio, se pierde; copia el texto de vez en cuando.</div>
  </section>

  <div class="foot">Scout Picks usa solo estadísticas de equipos (goles, córners, tiros) de las últimas temporadas. No considera alineaciones, lesiones ni árbitro. Apuesta solo lo que puedas perder y no intentes recuperar pérdidas.</div>
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

/* ---------- cálculo de un partido ---------- */
function compute(h, a, fh, fa){
  const L = DATA.leagues[league], out = {rows:[], chips:[]};
  const g = L.models.goles, l0 = lam(g,h,a), lh = l0[0]*fh, la = l0[1]*fa, m = scoreMatrix(lh,la);
  let pH=0,pD=0,pA=0,o15=0,o25=0,o35=0,bt=0;
  for(let i=0;i<m.length;i++)for(let j=0;j<m.length;j++){
    const v=m[i][j]; if(i>j)pH+=v; else if(i===j)pD+=v; else pA+=v;
    if(i+j>=2)o15+=v; if(i+j>=3)o25+=v; if(i+j>=4)o35+=v; if(i>=1&&j>=1)bt+=v;
  }
  out.lh=lh; out.la=la; out.p=[pH,pD,pA]; out.o25=o25; out.bt=bt;
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
function fillTeams(){
  const L=DATA.leagues[league];
  for(const id of ["home","away"]){
    $(id).innerHTML=L.teams.map(t=>`<option>${t}</option>`).join("");
  }
  const first=(DATA.fixtures||[]).find(f=>f.lg===league);
  const pref=first?[first.h,first.a]:[L.teams[0],L.teams[1]];
  $("home").value=pref[0]; $("away").value=pref[1];
}
function formHtml(t){return (DATA.leagues[league].form[t]||"").split("").map(c=>`<i class="${c}">${c}</i>`).join("");}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function injHtml(h,a){
  const L=DATA.leagues[league];
  if(!L.inj) return `<div class="none">Lesionados: sin datos (la API no respondió o no hay clave).</div>`;
  const one=t=>{const l=L.inj[t]||[];
    return l.length?`<div><b>${esc(t)}</b>: ${l.map(x=>esc(x[0])+(x[1]?" ("+esc(x[1])+")":"")).join(", ")}</div>`
                   :`<div class="none"><b>${esc(t)}</b>: sin bajas reportadas</div>`;};
  return one(h)+one(a);
}
function render(){
  const h=$("home").value,a=$("away").value,L=DATA.leagues[league];
  $("upd").textContent=`${L.name} · datos hasta ${L.updated} · actualizado ${DATA.generated||"—"}`;
  if(h===a){$("mk").innerHTML=`<tr><td colspan="6" class="empty">Elige dos equipos distintos.</td></tr>`;return;}
  const fh=1-0.07*(+$("adjH").value), fa=1-0.07*(+$("adjA").value);
  cur=compute(h,a,fh,fa);
  $("hn").textContent=h; $("an").textContent=a;
  $("hf").innerHTML=formHtml(h); $("af").innerHTML=formHtml(a);
  $("inj").innerHTML=injHtml(h,a);
  $("xh").textContent=cur.lh.toFixed(2); $("xa").textContent=cur.la.toFixed(2);
  const [pH,pD,pA]=cur.p;
  $("stack").innerHTML=`<div class="h" style="flex:${pH}">${pct(pH)}</div><div style="flex:${pD}">${pct(pD)}</div><div class="a" style="flex:${pA}">${pct(pA)}</div>`;
  $("chips").innerHTML=[["Más de 2.5 goles",pct(cur.o25)],["Ambos anotan",pct(cur.bt)],...cur.chips.map(c=>[c[0]+" esperados",c[1]])]
    .map(c=>`<div class="chip">${c[0]}<b>${c[1]}</b></div>`).join("");
  let html="",sec="";
  cur.rows.forEach((r,i)=>{
    if(r.sec!==sec){sec=r.sec;html+=`<tr class="sec"><td colspan="6">${esc(sec)}</td></tr>`;}
    html+=`<tr data-i="${i}"><td>${esc(r.label)}</td><td>${pct(r.p)}</td><td>${f2(1/r.p)}</td>
      <td><input inputmode="decimal" id="o${i}" data-i="${i}" placeholder="—" aria-label="Cuota ${esc(r.label)}"></td>
      <td class="ev" id="e${i}"></td><td><button class="mini" id="b${i}" data-i="${i}" disabled type="button">Anotar</button></td></tr>`;
  });
  $("mk").innerHTML=html;
  const fx=(DATA.fixtures||[]).find(f=>f.lg===league&&f.h===h&&f.a===a);
  let n=0;
  if(fx&&fx.odds){
    cur.rows.forEach((r,i)=>{const v=r.key&&fx.odds[r.key]; if(v){$("o"+i).value=v;onOdds(i);n++;}});
  }
  $("oddsSrc").textContent=n?`Cuotas cargadas automáticamente (${n}): promedio de varias casas según la API. Puedes cambiarlas por las de tu casa.`:"";
}
function onOdds(i){
  const r=cur.rows[i], v=parseFloat($("o"+i).value.replace(",","."));
  const e=$("e"+i), b=$("b"+i);
  if(!(v>1)){e.innerHTML="";b.disabled=true;return;}
  const ev=r.p*v-1, gap=r.p-1/v;
  let flag="";
  if(ev>0 && gap>=0.08) flag=`<span class="flag">Diferencia grande con la casa: probable error del modelo</span>`;
  else if(r.fav && r.p<0.5 && ev>0) flag=`<span class="flag">Ojo: el modelo subestima favoritos</span>`;
  e.className="ev "+(ev>0?"pos":"neg");
  e.innerHTML=(ev>0?"+":"")+(ev*100).toFixed(1)+"%"+flag;
  b.disabled=false;
}
function fxList(){
  const fx=DATA.fixtures||[];
  $("fxNote").textContent=DATA.note||"";
  $("fxList").innerHTML=fx.map((f,i)=>`<button class="fx" type="button" data-f="${i}"><span class="tm">${esc(f.date.slice(5,10))}<br>${esc(f.date.slice(11,16))}</span><span><b>${esc(f.h)}</b> vs <b>${esc(f.a)}</b></span><span class="lg">${esc(DATA.leagues[f.lg].name)}</span></button>`).join("");
}
$("fxList").addEventListener("click",ev=>{
  const b=ev.target.closest("button[data-f]"); if(!b)return;
  const f=DATA.fixtures[+b.dataset.f];
  league=f.lg; tabs(); fillTeams(); $("home").value=f.h; $("away").value=f.a;
  $("adjH").value="0"; $("adjA").value="0"; render();
  $("home").scrollIntoView({behavior:"smooth",block:"center"});
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
$("home").addEventListener("change",render); $("away").addEventListener("change",render);

function tabs(){
  $("leagueTabs").innerHTML=Object.entries(DATA.leagues).map(([k,L])=>
    `<button class="tab" type="button" data-k="${k}" aria-pressed="${k===league}">${L.name}</button>`).join("");
}
$("leagueTabs").addEventListener("click",ev=>{
  const k=ev.target.dataset.k; if(!k||k===league)return;
  league=k; tabs(); fillTeams(); $("adjH").value="0"; $("adjA").value="0"; render();
});
league=(DATA.fixtures&&DATA.fixtures[0]&&DATA.fixtures[0].lg)||Object.keys(DATA.leagues)[0];
tabs(); fillTeams(); fxList(); render(); renderLog();
</script>

</body></html>
'''


if __name__ == "__main__":
    build(TEMPLATE)
