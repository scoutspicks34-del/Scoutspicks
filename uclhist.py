"""Historial de Champions League: lee archivos de openfootball (Football.TXT) y limpia nombres de clubes."""
import json
import re

MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}

# nombre del archivo -> nombre que usa la página
CANON = {
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "Atalanta BC": "Atalanta", "AFC Ajax": "Ajax",
    "AS Monaco FC": "Monaco", "Bayer 04 Leverkusen": "Bayer Leverkusen", "Bologna FC 1909": "Bologna",
    "Celtic FC": "Celtic", "Chelsea FC": "Chelsea", "Club Atlético de Madrid": "Atlético Madrid",
    "Club Brugge KV": "Club Brugge", "FK Crvena Zvezda": "Crvena Zvezda", "GNK Dinamo Zagreb": "Dinamo Zagreb",
    "FC Barcelona": "Barcelona", "FC Basel 1893": "Basel", "FC Bayern München": "Bayern Munich",
    "FC Internazionale Milano": "Inter Milan", "FC København": "København", "FC Red Bull Salzburg": "Salzburg",
    "RB Salzburg": "Salzburg", "FK Bodø/Glimt": "Bodø/Glimt", "FK Shakhtar Donetsk": "Shakhtar Donetsk",
    "Feyenoord Rotterdam": "Feyenoord", "Galatasaray SK": "Galatasaray", "Girona FC": "Girona",
    "Juventus FC": "Juventus", "Lille OSC": "Lille", "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City", "Newcastle United FC": "Newcastle United", "OGC Nice": "Nice",
    "Olympique de Marseille": "Marseille", "PAE Olympiakos SFP": "Olympiacos", "PAOK Saloniki": "PAOK",
    "PSV": "PSV Eindhoven", "Paphos FC": "Pafos", "Pafos FC": "Pafos", "Paris Saint-Germain FC": "Paris Saint-Germain",
    "Qarabağ Ağdam FK": "Qarabağ", "Qarabağ FK": "Qarabağ", "Real Madrid CF": "Real Madrid",
    "Royale Union Saint-Gilloise": "Union Saint-Gilloise", "SK Slavia Praha": "Slavia Prague",
    "Slavia Praha": "Slavia Prague", "SK Sturm Graz": "Sturm Graz", "SL Benfica": "Benfica",
    "Sport Lisboa e Benfica": "Benfica", "SSC Napoli": "Napoli", "Sporting Clube de Portugal": "Sporting Lisbon",
    "ŠK Slovan Bratislava": "Slovan Bratislava", "Stade Brestois 29": "Brest", "Tottenham Hotspur FC": "Tottenham",
    "VfB Stuttgart": "Stuttgart", "Villarreal CF": "Villarreal", "Dinamo Kiev": "Dynamo Kyiv",
    "Crvena Zvezda": "Crvena Zvezda", "SK Brann": "Brann", "Víkingur": "Víkingur (Islas Feroe)",
    "Víkingur Reykjavík": "Víkingur Reykjavík", "AC Sparta Praha": "Sparta Prague", "Juventus": "Juventus",
    "Real Betis Balompié": "Real Betis", "AS Roma": "Roma", "Manchester United FC": "Manchester United",
    "FC Porto": "Porto", "Racing Club de Lens": "Lens", "RC Lens": "Lens", "SSC Como": "Como", "Como 1907": "Como",
    "Sabah FK": "Sabah", "Sabah Baku": "Sabah", "LASK": "LASK Linz", "FC Viking": "Viking", "Viking FK": "Viking",
    "RasenBallsport Leipzig": "RB Leipzig", "AEK Athína": "AEK Athens", "AEK Athens FC": "AEK Athens",
}
GEN_PRE = {"FC", "FK", "SK", "AFC", "AC", "AS", "SSC", "SL", "KF", "NK", "HNK", "GNK", "BSC", "FCI", "CS", "PFC", "KÍ", "ŠK", "PAE"}
GEN_POST = {"FC", "SK", "FK", "KV", "CF", "BC", "OSC", "TC"}


def clean(name):
    n = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name.strip())
    if n in CANON:
        return CANON[n]
    parts = n.split()
    while len(parts) > 1 and parts[0] in GEN_PRE:
        parts = parts[1:]
    while len(parts) > 1 and parts[-1] in GEN_POST:
        parts = parts[:-1]
    n2 = " ".join(parts)
    return CANON.get(n2, n2)


def stage_es(s, pre):
    if pre:
        return "Previa"
    s = s.strip()
    if s.startswith("League"):
        return "Fase de liga"
    for k, v in [("Playoffs", "Play-offs"), ("Round of 16", "Octavos"), ("Quarter", "Cuartos"),
                 ("Semi", "Semifinales"), ("Final", "Final")]:
        if k in s:
            return v
    return s


def parse(text, y0, pre=False):
    """Devuelve filas [aaaammdd, local, visitante, goles_local, goles_visitante, fase, nota]."""
    rows, stage, date = [], "", None
    for ln in text.splitlines():
        if ln.startswith("▪"):
            stage = stage_es(ln[1:], pre)
            continue
        m = re.match(r"^\s{2}(?:\w{3}\s+)?(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$", ln)
        if m and m.group(1) in MON:
            mo, d = MON[m.group(1)], int(m.group(2))
            y = int(m.group(3)) if m.group(3) else (y0 if mo >= 6 else y0 + 1)
            date = y * 10000 + mo * 100 + d
            continue
        m = re.match(r"^\s+(?:\d\d:\d\d\s+)?(.+?)\s+v\s+(.+?)\s{2,}(\d+)-(\d+)(.*)$", ln)
        if m and date:
            h, a, x, y2, rest = m.groups()
            note = ""
            if "pen." in rest:
                mm = re.search(r"pen\.\s+(\d+)-(\d+)\s+a\.e\.t\.", rest)
                if mm:
                    note = f"penales {x}-{y2}"
                    x, y2 = mm.group(1), mm.group(2)
            elif "a.e.t." in rest:
                note = "prórroga"
            rows.append([date, clean(h), clean(a), int(x), int(y2), stage, note])
    return rows


# Jornada 1 de la fase de liga 2026/27 (8 a 10 de septiembre de 2026). Fuente: uefa.com y nbcsports.com (coinciden).
MD1_2627 = [
    (20260908, "AEK Athens", "LASK Linz", 1, 0), (20260908, "Club Brugge", "Aston Villa", 2, 3),
    (20260908, "Borussia Dortmund", "Villarreal", 3, 2), (20260908, "Porto", "Manchester City", 0, 2),
    (20260908, "Lille", "Real Betis", 2, 3), (20260908, "Real Madrid", "Inter Milan", 2, 1),
    (20260909, "Barcelona", "Feyenoord", 5, 1), (20260909, "Stuttgart", "Viking", 3, 1),
    (20260909, "Liverpool", "Atlético Madrid", 2, 1), (20260909, "Paris Saint-Germain", "Slovan Bratislava", 6, 1),
    (20260909, "Sporting Lisbon", "Galatasaray", 3, 1), (20260909, "Napoli", "Arsenal", 0, 1),
    (20260910, "Fenerbahçe", "Roma", 1, 1), (20260910, "PSV Eindhoven", "Shakhtar Donetsk", 1, 1),
    (20260910, "Como", "RB Leipzig", 4, 1), (20260910, "Bayern Munich", "Bodø/Glimt", 5, 0),
    (20260910, "Manchester United", "Sabah", 4, 0), (20260910, "Slavia Prague", "Lens", 2, 3),
]


def md1_rows():
    return [[d, h, a, x, y, "Fase de liga", ""] for d, h, a, x, y in MD1_2627]


def make_json(repo_dir, out_path):
    """Genera data/ucl_hist.json con 2024/25 y 2025/26 a partir de un clon de openfootball/champions-league."""
    rows = []
    for f, pre, y0 in [("2024-25/clq.txt", True, 2024), ("2024-25/cl.txt", False, 2024),
                       ("2025-26/clq.txt", True, 2025), ("2025-26/cl.txt", False, 2025)]:
        rows += parse(open(f"{repo_dir}/{f}", encoding="utf8").read(), y0, pre)
    rows.sort(key=lambda r: r[0])
    json.dump({"fuente": "openfootball/champions-league (dominio público)", "m": rows},
              open(out_path, "w", encoding="utf8"), ensure_ascii=False, separators=(",", ":"))
    return rows


if __name__ == "__main__":
    import sys
    r = make_json(sys.argv[1], sys.argv[2])
    print(len(r), "partidos")
