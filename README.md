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

## Ligas nuevas (Holanda, Noruega, Arabia, MLS, Liga MX)
- Eredivisie: football-data.co.uk (`N1`), con córners, tiros y tarjetas.
- Noruega, MLS y Liga MX: archivos "new" de football-data.co.uk (solo goles y cuotas); los últimos días se completan con ESPN/TheSportsDB.
- Arabia: ESPN (marcadores del año actual y los dos anteriores; copia en `data/espn_arabia.csv` por si ESPN falla). Sin cuotas.
- El modelo de estas ligas no está validado con partidos pasados (`val: false`); la página lo avisa en "Cómo se calculó".

## Historial europeo en los enfrentamientos directos
`data/h2h_hist.json` incluye Champions, Europa League y Conference League (openfootball, ~10 temporadas, sin previas). Para regenerar solo la parte europea:
`PYTHONPATH=. python3 h2hhist.py <carpeta openfootball> data/h2h_hist.json --ucl-only`

## Calculadora de stake (pestaña Apostar)
Kelly fraccional al 25% o 30%: `stake = bankroll × fracción × (p·cuota − 1) / (cuota − 1)`, con `p` la probabilidad de la fila. Si la ganancia esperada es ≤ 0 muestra "L 0 (No apostar)". El bankroll y la fracción se guardan solo en el navegador de cada persona.

## Calculadora Props
Pestaña "Props" (menú de abajo). Poisson con el promedio y la línea que escribes: Menos de = suma de P(0)…P(entero bajo la línea), Más de = 1 − Menos. Líneas enteras: la devolución se calcula aparte. Usa el mismo bankroll y Kelly que Apostar (se comparten).

## Cuotas automáticas con The Odds API (pestaña Apostar)
Desde el navegador: la clave se pega en la página y vive solo en `localStorage` (nunca va en el código ni en el repositorio). Cada carga cuesta 2 créditos (regions=eu × markets=h2h,totals). La respuesta se guarda 30 min por liga. Referencia: Pinnacle; si no está, promedio de las casas. Llena 1X2 y Más/Menos 1.5, 2.5 y 3.5 (las que la API traiga).

## Picks sugeridos y armador de parlays
- Resumen: tarjeta "Picks sugeridos" por partido (pick de valor si hay EV>0, prob. ≥38% y sin alerta de error del modelo; si no, pick de probabilidad 60–74% con cuota justa 1.35–1.85; y un "fijo para parlay" de 75–92%).
- Destacados → "Jugadas del día" (escanea los partidos del día elegido, todas las ligas, que no hayan empezado; una selección por partido):
  - 🎯 **Pick del Día**: la selección con mayor valor esperado (EV+) y probabilidad real mayor de 55%, sin alerta de "probable error del modelo" (diferencia de 8+ puntos contra la cuota). Muestra cuota justa, cuota de la casa editable, EV en verde/rojo y stake de Kelly en lempiras. Si no hay, lo dice y deja escribir la cuota de tu casa en los mercados más probables.
  - ⚖️ **Combinada del Día**: 2–3 selecciones (partidos distintos) con probabilidad individual de 62% a 80%; probabilidad combinada, cuota justa combinada, campo para la cuota real, EV y stake.
  - 🚀 **Parlay Soñador**: solo con "falso favorito" (el mercado le da 4 a menos de 8 puntos más que el modelo puro; se apuesta la doble oportunidad contra él) o "tendencia de goles oculta" (los dos equipos 7+ de 10 y Poisson 8+ puntos por debajo). Máximo 4 selecciones, cada una 40%+, cuota justa de 5 o de 8 en adelante (tope 14–16). La probabilidad usa la menor entre modelo y mercado; el stake nunca pasa de 1% del bankroll y es L 0 sin valor.
  - Las cuotas que escribes (en el partido o en Destacados) se guardan por partido en `localStorage` (`scoutpicks.myodds.v1`) para que el escaneo las vea.
- Destacados → "Mi parlay": constructor manual (se guarda en `localStorage`).

## Picks y parlays por liga y por partido
- Partidos → panel desplegable "🏆 Picks y Parlays de la Jornada: [liga]" encima de la lista: el mismo motor de Destacados (Pick Solitario, Combinada, Parlay Soñador), pero solo con los partidos de la liga y el día elegidos que aún no empiezan. Se recalcula al cambiar de día o de liga. Destacados no cambia.
- Dentro de cada partido (Resumen): tarjeta "💡 Sugerencias del Modelo para este Partido" con Pick Solitario (1X2, doble oportunidad, goles o ambos anotan), Fijo para Parlay (75% o más) y "🧩 Crear apuesta (mismo partido)": combinación lógica de 2 mercados (por ejemplo doble oportunidad + más de 1.5) con la probabilidad conjunta sacada de la matriz de marcadores de Poisson, cuota justa y casilla para la cuota de la casa (EV y stake). Se recalcula con las cuotas y las bajas activadas.

## Bajas y Sanciones Clave (manual y local)
Reemplaza el panel de lesionados y el ajuste de −7%. Tres botones por equipo (estrella ofensiva −15% goles propios; mediocampo −8% propios y +4% rival; defensa/portero +12% rival). Se acumulan. En ligas con Poisson mueven los goles esperados; en la Champions sin cuotas mueven el Elo (−35, −20, −25). Todo se calcula en el navegador, sin ninguna fuente externa. No están validados con partidos pasados.
No hay conexión con API-Football: se eliminó el código, la clave del workflow y el campo de lesionados.
