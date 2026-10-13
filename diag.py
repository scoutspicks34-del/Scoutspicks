"""Diagnóstico de fuentes de datos (se corre en GitHub Actions, donde sí hay internet). Guarda un resumen en data/diag.json."""
import json
import requests

H = {"User-Agent": "Mozilla/5.0"}
FD = "https://www.football-data.co.uk/"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/"
TS = "https://www.thesportsdb.com/api/v1/json/3/"
ESPN2 = "https://site.api.espn.com/apis/site/v2/sports/soccer/"
ESPN2 = "https://site.api.espn.com/apis/site/v2/sports/soccer/"
URLS = [ESPN2 + "ksa.1/scoreboard?dates=202509", ESPN2 + "ksa.1/scoreboard?dates=202510&limit=500", ESPN2 + "ksa.1/scoreboard?dates=2025&limit=1000",
        ESPN2 + "ksa.1/scoreboard?dates=2025&limit=500", ESPN2 + "ksa.1/scoreboard?dates=202603", ESPN2 + "usa.1/scoreboard?dates=202607&limit=500",
        ESPN2 + "mex.1/scoreboard?dates=202608", ESPN2 + "nor.1/scoreboard?dates=202609", ESPN2 + "ksa.1/scoreboard?dates=20261011"]
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
                    if isinstance(j.get("events"), list) and j["events"]:
                        ev = j["events"]
                        r["completed"] = sum(1 for e in ev if (((e.get("competitions") or [{}])[0].get("status") or e.get("status") or {}).get("type") or {}).get("completed"))
                        r["dates"] = [ev[0].get("date"), ev[-1].get("date")]
                        e0 = ev[0]; c0 = (e0.get("competitions") or [{}])[0]
                        r["sample"] = {"name": e0.get("name"), "status": ((c0.get("status") or {}).get("type") or {}).get("name"), "comp": [(c.get("homeAway"), (c.get("team") or {}).get("displayName"), c.get("score")) for c in c0.get("competitors", [])], "odds": c0.get("odds")}
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
json.dump(out, open("data/diag3.json", "w"), ensure_ascii=False, indent=1)
