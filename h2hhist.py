"""Historial para enfrentamientos directos (H2H): ~10 temporadas de Bundesliga, Premier League, La Liga y Champions League.
Fuente: openfootball (dominio público). Genera data/h2h_hist.json con solo goles. Se corre a mano una vez por temporada:
    python3 h2hhist.py <carpeta con los clones de openfootball> data/h2h_hist.json
Los nombres se convierten a los que usa la página (football-data para ligas, nombres de Champions para la copa)."""
import json
import re
import sys

import build as B
import uclhist

SEASONS = list(range(2016, 2026))        # 2016/17 ... 2025/26
LOCAL_ALIAS = {"wolverhampton wanderers fc": "Wolves", "west ham united fc": "West Ham", "west bromwich albion fc": "West Brom",
               "leicester city fc": "Leicester", "norwich city fc": "Norwich", "burnley fc": "Burnley", "watford fc": "Watford",
               "southampton fc": "Southampton", "stoke city fc": "Stoke", "swansea city afc": "Swansea", "cardiff city fc": "Cardiff",
               "huddersfield town afc": "Huddersfield", "sheffield united fc": "Sheffield United", "luton town fc": "Luton",
               "queens park rangers fc": "QPR", "middlesbrough fc": "Middlesbrough", "derby county fc": "Derby",
               "fc schalke 04": "Schalke 04", "hannover 96": "Hannover", "sv darmstadt 98": "Darmstadt", "fc ingolstadt 04": "Ingolstadt",
               "vfl wolfsburg": "Wolfsburg", "vfl bochum 1848": "Bochum", "fc st pauli": "St Pauli", "1 fc heidenheim 1846": "Heidenheim",
               "fc union berlin": "Union Berlin", "1 fc union berlin": "Union Berlin", "hertha bsc": "Hertha", "arminia bielefeld": "Bielefeld",
               "fortuna dusseldorf": "Fortuna Dusseldorf", "fc nurnberg": "Nurnberg", "1 fc nurnberg": "Nurnberg",
               "sevilla fc": "Sevilla", "real valladolid cf": "Valladolid", "ud las palmas": "Las Palmas", "cd leganes": "Leganes",
               "sd eibar": "Eibar", "granada cf": "Granada", "cadiz cf": "Cadiz", "ud almeria": "Almeria", "girona fc": "Girona",
               "real zaragoza": "Zaragoza", "sporting de gijon": "Sp Gijon", "sd huesca": "Huesca", "rcd mallorca": "Mallorca",
               "cd alaves": "Alaves", "deportivo alaves": "Alaves", "real oviedo": "Oviedo", "fc cartagena": "Cartagena",
               "athletic club bilbao": "Ath Bilbao", "bor monchengladbach": "M'gladbach", "atletico de madrid": "Ath Madrid", "real sociedad": "Sociedad", "real betis": "Betis"}
B.ALIAS.update(LOCAL_ALIAS)


def parse_league(text, y0):
    """Archivos de liga de openfootball con el formato 'Local  2-1 (1-0)  Visitante' (sin la letra v). Devuelve [fecha, local, visitante, gl, gv]."""
    rows, date = [], None
    for ln in text.splitlines():
        m = re.match(r"^\s{0,2}(?:\w{3}\s+)?(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$", ln)
        if m and m.group(1) in uclhist.MON:
            mo, d = uclhist.MON[m.group(1)], int(m.group(2))
            date = (int(m.group(3)) if m.group(3) else (y0 if mo >= 6 else y0 + 1)) * 10000 + mo * 100 + d
            continue
        m = re.match(r"^\s+(?:\d\d:\d\d\s+)?(.+?)\s{2,}(\d+)-(\d+)(?:\s+\(\d+-\d+\))?\s{2,}(.+?)\s*$", ln)
        if m and date:
            rows.append([date, m.group(1).strip(), m.group(4).strip(), int(m.group(2)), int(m.group(3))])
    return rows


def league_rows(text, y0):
    r = uclhist.parse(text, y0, raw=True)
    return r if r else parse_league(text, y0)


def stage_h2h(st):
    return "Fase de grupos" if st.startswith(("Group", "Gruppe")) else st


def smap(name, known):
    """Conversión estricta de nombres: si no hay equivalencia clara, devuelve None (mejor perder un partido que mezclar clubes)."""
    n = B.norm(name)
    a = B.ALIAS.get(n)
    if a in known:
        return a
    for t in known:
        if B.norm(t) == n:
            return t
    c = [t for t in known if n.startswith(B.norm(t) + " ") or n.endswith(" " + B.norm(t))]
    return c[0] if len(c) == 1 else None


def make(root, known_leagues, known_ucl, out):
    res = {"fuente": "openfootball (dominio público). Solo goles.", "leagues": {}, "ucl": {}}
    for key, (repo, fn) in B.OF_FILES.items():
        known, rows, bad = set(known_leagues[key]), [], {}
        for y in SEASONS:
            try:
                txt = open(f"{root}/{repo}/{y}-{(y + 1) % 100:02d}/{fn}", encoding="utf8").read()
            except OSError:
                continue
            for d, h, a, x, y2, *_ in league_rows(txt, y):
                mh, ma = smap(h, known), smap(a, known)
                if mh and ma and mh != ma:
                    rows.append((d, mh, ma, x, y2))
                else:
                    for nm, m in ((h, mh), (a, ma)):
                        if not m:
                            bad[nm] = bad.get(nm, 0) + 1
        t = sorted({r[1] for r in rows} | {r[2] for r in rows}); ix = {n: i for i, n in enumerate(t)}
        res["leagues"][key] = {"t": t, "m": [[d, ix[h], ix[a], x, y2] for d, h, a, x, y2 in sorted(rows)]}
        print(key, len(rows), "partidos,", len(t), "equipos; sin equivalencia (top):", sorted(bad.items(), key=lambda z: -z[1])[:12])
    known, rows, stages = set(known_ucl), [], []
    for y in SEASONS:
        for fn, pre in (("cl.txt", False), ("clq.txt", True)):
            try:
                txt = open(f"{root}/champions-league/{y}-{(y + 1) % 100:02d}/{fn}", encoding="utf8").read()
            except OSError:
                continue
            for d, h, a, x, y2, st, note in uclhist.parse(txt, y, pre):
                mh = B.UCL_ALIAS.get(B.norm(h)) or (h if h in known else None)
                ma = B.UCL_ALIAS.get(B.norm(a)) or (a if a in known else None)
                if mh in known and ma in known and mh != ma and not pre:
                    rows.append((d, mh, ma, x, y2, stage_h2h(st)))
    t = sorted({r[1] for r in rows} | {r[2] for r in rows}); ix = {n: i for i, n in enumerate(t)}
    for r in rows:
        if r[5] not in stages:
            stages.append(r[5])
    res["ucl"] = {"t": t, "s": stages, "m": [[d, ix[h], ix[a], x, y2, stages.index(st)] for d, h, a, x, y2, st in sorted(rows)]}
    print("ucl", len(rows), "partidos,", len(t), "equipos, fases:", stages)
    json.dump(res, open(out, "w", encoding="utf8"), ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    import glob
    import pandas as pd
    S = "/tmp/claude-0/-home-claude/ca15f016-79f3-52ce-b32c-602dd5479c11/scratchpad/"
    loc = {"bundesliga": S + "bl_data", "premier": S + "pl_data", "laliga": S + "sp_data"}
    kl = {}
    for k, p in loc.items():
        d = B.prepare([pd.read_csv(f, encoding="latin-1") for f in sorted(glob.glob(p + "/*.csv"))])
        kl[k] = set(d.HomeTeam) | set(d.AwayTeam)
    ku = set(B.UCL_TEAMS) | {r[1] for r in json.load(open("data/ucl_hist.json"))["m"]} | {r[2] for r in json.load(open("data/ucl_hist.json"))["m"]}
    make(sys.argv[1], kl, ku, sys.argv[2])
