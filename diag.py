"""Diagnóstico de fuentes de datos (se corre en GitHub Actions, donde sí hay internet). Guarda un resumen en data/diag.json."""
import json
import requests

H = {"User-Agent": "Mozilla/5.0"}
FD = "https://www.football-data.co.uk/"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/"
TS = "https://www.thesportsdb.com/api/v1/json/3/"
ESPN2 = "https://site.api.espn.com/apis/site/v2/sports/soccer/"
URLS = [ESPN2 + "ksa.1/scoreboard?dates=20261005-20261011", ESPN2 + "ksa.1/scoreboard?dates=20260901-20261011", ESPN2 + "ksa.1/scoreboard?dates=20260801-20261011",
        ESPN2 + "ksa.1/scoreboard?dates=2026", ESPN2 + "ksa.1/scoreboard?season=2025&limit=500", ESPN2 + "ksa.1/scoreboard?dates=20260315-20260321&limit=100",
        ESPN2 + "ksa.1/teams", ESPN2 + "ksa.1/scoreboard?dates=20260215",
        "https://sports.core.api.espn.com/v2/sports/soccer/leagues/ksa.1/seasons/2025/types/1/events?limit=1000",
        "https://sports.core.api.espn.com/v2/sports/soccer/leagues/ksa.1/seasons",
        "https://site.api.espn.com/apis/site/v2/sports/soccer/ksa.1/teams/18267/schedule?season=2025",
        "https://site.web.api.espn.com/apis/v2/sports/soccer/ksa.1/standings?season=2025"]
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
json.dump(out, open("data/diag2.json", "w"), ensure_ascii=False, indent=1)
