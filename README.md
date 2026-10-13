# Picks Fresh

Pronósticos de fútbol (Bundesliga, Premier League, La Liga). `build.py` descarga datos, ajusta el modelo y genera `docs/index.html`. Se ejecuta a diario con GitHub Actions.


## Resultados al día

football-data.co.uk publica los CSV con uno o dos días de retraso. Para que los marcadores, las rachas y la tabla se actualicen el mismo día,
`build.py` completa los partidos que faltan (solo goles) con fuentes gratuitas, en este orden: `data/resultados_manuales.csv`, openfootball,
OpenLigaDB (Bundesliga), ESPN (scoreboard público, no oficial) y, si existe el secreto `FOOTBALL_DATA_ORG_KEY`, football-data.org
(clave gratuita en football-data.org). Cuando football-data.co.uk publica el partido, sus datos completos (córners, tiros, tarjetas) reemplazan los de la fuente rápida.
La portada indica qué fuentes aportaron partidos (o cuáles no respondieron).

## Historial de enfrentamientos directos (H2H)
`data/h2h_hist.json` guarda ~10 temporadas de Bundesliga, Premier League, La Liga y Champions League (solo goles, de openfootball) para
los enfrentamientos directos. Se genera con `PYTHONPATH=. python3 h2hhist.py <carpeta con los clones de openfootball> data/h2h_hist.json`
y conviene regenerarlo una vez por temporada. `build.py` solo incrusta los partidos de equipos actuales y anteriores a los datos recientes.

## App instalable (PWA)
La página se puede instalar como app en Android (botón "Instalar" o menú de Chrome) y en iPhone/iPad (Safari: Compartir → Añadir a pantalla de inicio).
Archivos: `docs/manifest.webmanifest`, `docs/sw.js` (service worker: pide la página a internet y usa la última copia si no hay conexión) e `docs/icons/`.
Si cambias `sw.js`, sube el número de `V` para que los teléfonos descarten la copia vieja.
