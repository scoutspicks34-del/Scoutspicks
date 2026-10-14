"""Bajas (lesionados y sancionados) de cada liga, desde páginas públicas y gratuitas (sin claves de API).

Fuente: sportsgambler.com/injuries (su robots.txt permite esas páginas). Se pide una página por liga, con pausa entre
peticiones, desde el flujo programado de GitHub (unas pocas veces al día). Si una página falla, se conserva lo anterior
de esa liga y la app muestra la fecha de la última actualización.

    python bajas.py            -> actualiza data/bajas.json
Transfermarkt (bloquea los robots y prohíbe el acceso automático) y WhoScored (necesita JavaScript) no se usan.
"""
import datetime as dt
import difflib
import json
import os
import re
import sys
import time
import unicodedata

import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bajas.json")
BASE = "https://www.sportsgambler.com/injuries/football/"
SLUG = {
    "premier": "england-premier-league", "bundesliga": "germany-bundesliga", "laliga": "spain-la-liga",
    "ucl": "uefa-champions-league", "eredivisie": "netherlands-eredivisie", "noruega": "norway-eliteserien",
    "mls": "usa-mls", "ligamx": "mexico-liga-mx",
    "uel": "uefa-europa-league", "liga3": "germany-3-liga",   # si la fuente no tiene la página, la comprobación del título la descarta sin afectar a las demás
}
UA = "Mozilla/5.0 (compatible; PicksFreshBot/1.0; +https://github.com/scoutspicks34-del/Scoutspicks)"

PARTE = {  # lo que la fuente escribe en "Info" -> texto en español
    "Thigh": "muslo", "Knee": "rodilla", "Ankle": "tobillo", "Calf": "pantorrilla", "Hamstring": "isquiotibiales", "Groin": "ingle",
    "Foot": "pie", "Low back": "espalda baja", "Upper back": "espalda alta", "Shoulder": "hombro", "Elbow": "codo", "Rib": "costillas",
    "Head": "cabeza", "Hand": "mano", "Wrist": "muñeca", "Abdomen": "abdomen", "Hip": "cadera", "Lower leg": "pierna", "Achilles": "tendón de Aquiles",
    "Back": "espalda", "Neck": "cuello", "Toe": "dedo del pie", "Finger": "dedo", "Shin": "espinilla", "Heel": "talón", "Adductor": "aductor",
}
OTRAS = {
    "Cruciate ligame": "Rotura de ligamento cruzado", "Muscle": "Lesión muscular", "Illness": "Enfermedad", "Knock": "Golpe",
    "Physical discom": "Molestias físicas", "Other": "Lesión (sin detalle)", "Fracture": "Fractura", "Surgery": "Operación", "Virus": "Virus",
    "Ligament": "Lesión de ligamento", "Concussion": "Conmoción", "Personal": "Motivos personales", "Rest": "Descanso",
}
POS = {"G": "Portero", "D": "Defensa", "M": "Mediocampista", "F": "Delantero"}


def _txt(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def _html_un(s):
    return (s or "").replace("&amp;", "&").replace("&#039;", "'").replace("&#39;", "'").replace("&quot;", '"').replace("&rsquo;", "’")


def razon(kind, info):
    """(tipo, motivo en español). tipo: i lesión · d duda · r roja · r2 doble amarilla · y acumulación de amarillas · s otra suspensión."""
    low = (info or "").lower()
    if kind == "redcard":
        if "2nd yellow" in low:
            return "r2", "Suspendido por doble amarilla (expulsión)"
        if "yellow" in low:
            return "y", "Suspendido por acumulación de amarillas"
        if "red" in low:
            return "r", "Suspendido por tarjeta roja"
        return "s", "Suspendido"
    base = OTRAS.get(info) or ("Lesión de " + PARTE[info] if info in PARTE else ("Lesión (" + info + ")" if info and info != "-" else "Lesión (sin detalle)"))
    if kind == "injury-questionmark":
        return "d", "Duda: " + base[0].lower() + base[1:]
    return "i", base


def parse_sg(html):
    """{equipo: [jugador]} desde una página de lesiones de Sportsgambler."""
    out = {}
    for blk in re.split(r'<div class="injury-block">', html)[1:]:
        m = re.search(r'class="injuries-title"[^>]*>(.*?)</h3>', blk, re.S)
        if not m:
            continue
        team = _html_un(_txt(m.group(1)))
        players = []
        for row in re.findall(r'<div class="inj-container">(.*?)</div><!--inj-container-->', blk, re.S):
            ty = re.search(r'inj-type ([^"]*)"', row)
            nm = re.search(r'class="inj-player">([^<]*)<', row)
            if not ty or not nm:
                continue
            f = lambda cls: _html_un(_txt((re.search(r'class="inj-' + cls + r'[^"]*">(.*?)</span>', row, re.S) or [None, ""])[1]))
            info, pos, ret = f("info"), f("position"), f("return")
            k, why = razon(ty.group(1).strip(), info)
            num = lambda x: int(x) if x.isdigit() else None
            players.append({
                "n": _html_un(nm.group(1).strip()), "p": POS.get(pos, ""), "k": k, "w": why,
                "m": num(f("game")), "gl": num(f("goals")), "as": num(f("assist")),
                "r": ret if re.match(r"\d{4}-\d{2}-\d{2}$", ret) else "",
            })
        out[team] = players
    return out


# ---------- emparejar los nombres de la fuente con los nombres de equipos de la app ----------
def _n(s):
    s = unicodedata.normalize("NFD", (s or "").lower().replace("ø", "o").replace("æ", "ae").replace("ß", "ss"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[’'`.]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


NOISE = {"fc", "cf", "sc", "afc", "ac", "as", "ssc", "sv", "fk", "bk", "if", "ik", "sk", "club", "de", "the", "cd", "ud", "rcd", "1", "04", "05", "96", "98", "08", "1899", "1846"}
ALIAS = {  # nombre de la fuente (normalizado) -> nombre normalizado en la app
    "man city": "man city", "man united": "man united", "nottm forest": "nottm forest", "bm gladbach": "mgladbach", "fc koln": "fc koln",
    "sp rotterdam": "sparta rotterdam", "go ahead e": "go ahead eagles", "pumas": "pumas unam", "america": "club america",
    "guadalajara": "guadalajara chivas", "leon": "club leon", "san luis": "atl san luis", "juarez": "juarez", "athletic club": "ath bilbao",
    "atletico madrid": "ath madrid", "real sociedad": "sociedad", "rayo vallecano": "vallecano", "celta vigo": "celta",
    "alaves": "alaves", "racing club": "santander", "frankfurt": "ein frankfurt", "hamburger sv": "hamburg", "mainz 05": "mainz",
    "borussia dortmund": "dortmund", "heerenveen": "sc heerenveen", "psv": "psv eindhoven", "az alkmaar": "az alkmaar", "az": "az alkmaar",
    "deportivo": "la coruna", "psg": "paris saint germain", "la galaxy": "los angeles galaxy", "ny red bulls": "new york red bulls",
    "nycfc": "new york city", "sporting kc": "sporting kansas city",
}


def _toks(s):
    return [t for t in _n(s).split() if t not in NOISE]


def match_team(name, cands):
    """Mejor candidato de `cands` para `name` (o None). Exacto > alias > prefijos de palabras > similitud de texto."""
    nn = _n(name)
    idx = {_n(c): c for c in cands}
    for key in (nn, ALIAS.get(nn)):
        if key and key in idx:
            return idx[key]
    a = _toks(ALIAS.get(nn, nn))
    best, score = None, 0.0
    for c in cands:
        b = _toks(c)
        if not a or not b:
            continue
        pre = lambda x, y: all(any(w2.startswith(w1) or w1.startswith(w2) for w2 in y) for w1 in x)
        s = difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
        if (pre(a, b) or pre(b, a)) and min(len(a), len(b)) >= 1 and (len(a) > 1 or len(b) > 1 or a[0][:4] == b[0][:4]):
            s = max(s, 0.85)
        if s > score:
            best, score = c, s
    return best if score >= 0.82 else None


def attach(raw, teams_by_lg, log=print):
    """Convierte data/bajas.json (nombres de la fuente) en {liga: {t, teams: {equipo de la app: [jugadores]}}} para la página."""
    out = {}
    for lg, blk in (raw or {}).items():
        cands = teams_by_lg.get(lg) or []
        if not cands:
            continue
        mapped, miss = {}, []
        for team, pl in (blk.get("teams") or {}).items():
            m = match_team(team, cands)
            if m is None:
                miss.append(team)
            elif m in mapped:
                miss.append(team + " (repetido)")
            else:
                mapped[m] = pl
        if miss:
            log(f"  bajas {lg}: sin emparejar -> {', '.join(miss)}")
        out[lg] = {"t": blk.get("t", ""), "teams": mapped}
    return out


# ---------- descarga ----------
def fetch(slug):
    r = requests.get(BASE + slug + "/", timeout=40, headers={"User-Agent": UA, "Accept-Language": "en"})
    r.raise_for_status()
    if re.search(r"<title>\s*Football Injuries", r.text):      # la página genérica: esa liga no existe en la fuente
        raise ValueError("liga no disponible en la fuente")
    return r.text


def main():
    try:
        old = json.load(open(OUT, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        old = {}
    now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
    ok = 0
    for lg, slug in SLUG.items():
        try:
            teams = parse_sg(fetch(slug))
        except Exception as e:  # noqa: BLE001
            print(f"  bajas {lg}: {str(e)[:100]} (se conserva lo anterior)")
            teams = {}
        if len(teams) >= 8:
            old[lg] = {"t": now, "teams": teams}
            ok += 1
            print(f"  bajas {lg}: {len(teams)} equipos, {sum(len(v) for v in teams.values())} jugadores")
        else:
            print(f"  bajas {lg}: página sin datos útiles ({len(teams)} equipos); se conserva lo anterior")
        time.sleep(3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(old, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    print(f"bajas: {ok} de {len(SLUG)} ligas actualizadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
