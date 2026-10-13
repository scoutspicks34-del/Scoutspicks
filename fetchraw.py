"""Descarga datos crudos de las ligas nuevas para probar el código sin internet (se corre en Actions). Guarda en data/raw/."""
import datetime as dt
import json
import os

import requests

H = {"User-Agent": "Mozilla/5.0"}
os.makedirs("data/raw", exist_ok=True)
FD = "https://www.football-data.co.uk/"


def save(name, content):
    open("data/raw/" + name, "wb").write(content)


for u, n in [("mmz4281/2627/N1.csv", "N1_2627.csv"), ("mmz4281/2526/N1.csv", "N1_2526.csv"), ("mmz4281/2425/N1.csv", "N1_2425.csv"),
             ("mmz4281/2324/N1.csv", "N1_2324.csv"), ("mmz4281/2223/N1.csv", "N1_2223.csv"), ("new/NOR.csv", "NOR.csv"), ("new/USA.csv", "USA.csv"),
             ("new/MEX.csv", "MEX.csv"), ("new_league_fixtures.csv", "new_league_fixtures.csv"), ("fixtures.csv", "fixtures.csv")]:
    try:
        r = requests.get(FD + u, timeout=60, headers=H)
        r.raise_for_status()
        save(n, r.content)
    except Exception as e:  # noqa: BLE001
        print(u, e)


def evs(slug, dates):
    r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={dates}&limit=1000", timeout=90, headers=H)
    r.raise_for_status()
    out = []
    for e in r.json().get("events", []):
        c = (e.get("competitions") or [{}])[0]
        t = (c.get("status") or e.get("status") or {}).get("type") or {}
        cs = c.get("competitors") or []
        H_ = [x for x in cs if x.get("homeAway") == "home"]
        A_ = [x for x in cs if x.get("homeAway") == "away"]
        if not H_ or not A_:
            continue
        nm = lambda x: (x.get("team") or {}).get("displayName") or ""
        out.append({"date": e.get("date"), "h": nm(H_[0]), "a": nm(A_[0]), "hs": H_[0].get("score"), "as": A_[0].get("score"),
                    "done": bool(t.get("completed")), "st": t.get("name"), "slug": (e.get("season") or {}).get("slug")})
    return out


sl = {"arabia": "ksa.1", "eredivisie": "ned.1", "noruega": "nor.1", "mls": "usa.1", "ligamx": "mex.1"}
for k, s in sl.items():
    allv = []
    try:
        yrs = [2024, 2025, 2026] if k == "arabia" else [2026]
        for y in yrs:
            allv += evs(s, str(y))
        today = dt.date(2026, 10, 12)
        if k != "arabia":
            for i in range(-30, 9):
                d = today + dt.timedelta(days=i)
                allv += evs(s, d.strftime("%Y%m%d"))
    except Exception as e:  # noqa: BLE001
        print(k, e)
    seen, uniq = set(), []
    for e in allv:
        key = (e["date"], e["h"], e["a"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    json.dump(uniq, open(f"data/raw/espn_{k}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(k, len(uniq))
