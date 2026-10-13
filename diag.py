"""Diagnóstico de fuentes de datos (se corre en GitHub Actions, donde sí hay internet). Guarda un resumen en data/diag.json."""
import json
import requests

H = {"User-Agent": "Mozilla/5.0"}
FD = "https://www.football-data.co.uk/"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/"
TS = "https://www.thesportsdb.com/api/v1/json/3/"
URLS = [FD + "mmz4281/2627/N1.csv", FD + "mmz4281/2526/N1.csv", FD + "new/NOR.csv", FD + "new/USA.csv", FD + "new/MEX.csv",
        FD + "fixtures.csv", FD + "new_league_fixtures.csv", FD + "new/SAU.csv"]
for sl in ["ned.1", "nor.1", "ksa.1", "usa.1", "mex.1"]:
    URLS += [ESPN + sl + "/scoreboard?dates=20261011", ESPN + sl + "/scoreboard?dates=20261016",
             ESPN + sl + "/scoreboard?dates=20260801-20261012", ESPN + sl + "/scoreboard?dates=20260801-20261012&limit=1000",
             "https://site.api.espn.com/apis/v2/sports/soccer/" + sl + "/standings"]
for c in ["Norway", "Netherlands", "Saudi Arabia", "USA", "Mexico"]:
    URLS.append(TS + "search_all_leagues.php?c=" + c.replace(" ", "%20") + "&s=Soccer")
for i in [4337, 4350, 4346, 4668, 4358, 4359]:
    URLS += [TS + f"lookupleague.php?id={i}", TS + f"eventsseason.php?id={i}&s=2026", TS + f"eventsseason.php?id={i}&s=2025-2026", TS + f"eventspastleague.php?id={i}", TS + f"eventsnextleague.php?id={i}"]
out = {}
for u in URLS:
    r = {}
    try:
        x = requests.get(u, timeout=40, headers=H)
        r["status"], r["bytes"] = x.status_code, len(x.content)
        t = x.text
        if u.endswith(".csv"):
            ls = t.splitlines()
            r["lines"], r["head"], r["tail"] = len(ls), ls[0][:300] if ls else "", ls[-1][:200] if ls else ""
            if u.endswith("fixtures.csv"):
                r["divs"] = sorted({l.split(",")[0] for l in ls[1:]})
        else:
            try:
                j = x.json()
                if isinstance(j, dict):
                    r["keys"] = list(j)[:12]
                    for k in ("events", "event", "leagues", "countrys", "countries"):
                        if isinstance(j.get(k), list):
                            r[k] = len(j[k])
                            if j[k]:
                                e = j[k][0]
                                r["first"] = {a: str(e.get(a))[:60] for a in list(e)[:14]} if k == "event" or k == "leagues" else str(e)[:400]
                    if "children" in j:
                        r["children"] = len(j["children"])
            except Exception:  # noqa: BLE001
                r["text"] = t[:200]
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:200]
    out[u] = r
    print(u, r.get("status"), r.get("bytes"), r.get("error", ""), flush=True)
json.dump(out, open("data/diag.json", "w"), ensure_ascii=False, indent=1)
