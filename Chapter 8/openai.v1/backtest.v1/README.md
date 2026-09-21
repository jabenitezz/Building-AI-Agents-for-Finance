# Chapter 8 / openai.v1 / backtest.v1

Esta versión separa el problema en dos fases:

FASE A — generación de señales (usa LLM y datos históricos point-in-time)
datos históricos -> comité multiagente -> signals.csv

FASE B — motor de backtest (implementado con VectorBT)
signals.csv -> cartera -> PnL -> Sharpe / drawdown / benchmarks

El objetivo es que la fase B pueda ejecutarse muchas veces sin volver a pagar llamadas a modelos.

## Diferencia frente a backtest.py original

El backtest.py del capítulo recorre fechas históricas, pero llama al comité actual, que usa noticias, fundamentales, técnicos y macro actuales. Sirve como demostración de estructura, no como backtest point-in-time.

backtest.v1 pasa una fecha histórica real a toda la capa de datos:

- Técnicos: solo precios hasta as_of_date.
- Macro: VIX, Treasury 10Y y S&P 500 hasta as_of_date.
- Fundamentales: último 10-K SEC que YA estaba presentado en as_of_date.
- Noticias: Alpha Vantage NEWS_SENTIMENT filtrado por ticker y acotado entre la fecha previa y as_of_date.
- Se solicitan hasta 50 candidatos por ventana y se inspeccionan TODOS antes de elegir los 12 finales.
- Una noticia debe mencionar de forma determinista al ticker, compañía o producto específico (por ejemplo NVIDIA/NVDA/B200/H200) en título o resumen.
- Si la coincidencia está en el título basta un relevance score >= 0.20.
- Las noticias con coincidencia solo en el resumen son fallback y exigen relevance score >= 0.85.
- Si hay al menos 5 noticias directas en título, NO se usa ninguna noticia summary-only.
- Si hay menos de 5 directas, se completa únicamente hasta 5 con las mejores summary-only; no se rellena artificialmente hasta 12.
- Dentro de cada grupo se ordena por relevance score y después por recencia.
- Si la fuente histórica de noticias falla, NO se sustituye por noticias actuales.
- La señal se considera generada al final del día de decisión y la operación se ejecuta en la apertura de la siguiente sesión bursátil.

## Por qué usamos datos anuales SEC

Para una primera versión auditable se usan hechos anuales 10-K. Es conservador: puede dejar fuera información trimestral que sí existía, pero evita usar por accidente un dato que todavía no había sido publicado. En una v2 se puede incorporar reconstrucción TTM desde 10-Q/10-K.

## Instalación

Desde Chapter 8/openai.v1:

    uv pip install -r "backtest.v1/requirements.txt"

El script carga primero openai.v1/.env y luego, si existe, backtest.v1/.env.

Necesitas:

    OPENAI_API_KEY=...
    SEC_EDGAR_EMAIL=tu-correo@dominio
    ALPHAVANTAGE_API_KEY=tu-clave-alpha-vantage

SEC_EDGAR_EMAIL no es una clave; se usa para un User-Agent responsable frente a SEC EDGAR.

ALPHAVANTAGE_API_KEY se usa exclusivamente para NEWS_SENTIMENT histórico. La consulta se hace por ticker y con time_from/time_to; las respuestas correctas se guardan en cache/alphavantage_news para no repetir peticiones al reanudar.

Opcionalmente puedes ajustar:

    ALPHAVANTAGE_MIN_RELEVANCE_SCORE=0.20
    ALPHAVANTAGE_SUMMARY_ONLY_MIN_RELEVANCE_SCORE=0.85
    ALPHAVANTAGE_DIRECT_NEWS_TARGET=5
    ALPHAVANTAGE_PROVIDER_NEWS_LIMIT=50

El primer umbral se aplica a coincidencias directas en título. El segundo es el umbral más estricto para candidatos que solo mencionan al ticker/compañía/producto en el resumen. DIRECT_NEWS_TARGET define cuántas noticias directas son suficientes para prescindir completamente del fallback. PROVIDER_NEWS_LIMIT define cuántos candidatos crudos se descargan antes de filtrar.

## Primero: ver el plan sin gastar API

    python "backtest.v1/generate_signals.py" --plan-only

Por defecto:

    tickers = TSLA MSFT NVDA
    start   = 2026-01-01
    end     = 2026-04-01
    freq    = monthly

La frecuencia mensual reduce mucho el coste frente a la semanal.

## Generar señales

    python "backtest.v1/generate_signals.py"

Alternativas:

    python "backtest.v1/generate_signals.py" --freq biweekly
    python "backtest.v1/generate_signals.py" --freq weekly
    python "backtest.v1/generate_signals.py" --tickers MSFT NVDA
    python "backtest.v1/generate_signals.py" --start 2025-01-01 --end 2026-01-01
    python "backtest.v1/generate_signals.py" --show-reports

## Reanudación

El CSV se escribe después de cada ticker. Si se corta la ejecución, vuelve a lanzar el mismo comando: las combinaciones fecha/ticker ya presentes se omiten.

Para empezar desde cero:

    python "backtest.v1/generate_signals.py" --overwrite

## Salida a pantalla

Para cada decisión verás la fecha histórica, filing 10-K usado, técnicos hasta esa fecha, ventana de noticias histórica, cuántos artículos fueron filtrados por relevancia, macro, decisión final y la fecha/precio de apertura de la siguiente sesión donde se ejecutaría la orden.

## CSV

Por defecto:

    backtest.v1/output/signals.csv

Incluye:

- fecha de decisión EOD y siguiente rebalanceo
- ticker
- regla de ejecución, fecha de ejecución y precio de apertura de la siguiente sesión
- BUY / HOLD / SELL
- confianza y tamaño
- precio en la fecha
- fecha de filing SEC y periodo contable
- P/E, P/B, ROE, margen, D/E, current ratio, crecimiento
- SMA50, SMA200, RSI, volatilidad
- proveedor, ventana histórica de noticias, candidatos raw, noticias directas, candidatas summary-only, fallback usado, rechazados, duplicados/vacíos, umbrales de relevancia y titulares finales
- VIX, Treasury 10Y y retorno mensual S&P
- informes de los cuatro especialistas
- tesis del Portfolio Manager
- veredicto del Risk Officer
- JSON crudo de fundamentales/técnicos/macro para auditoría

## Regla temporal de ejecución

La fecha de decisión representa el final del día histórico completo. Por tanto, pueden entrar noticias publicadas después del cierre bursátil pero antes de terminar ese día natural.

Para evitar look-ahead bias, esa señal NO se ejecuta al cierre del mismo día:

    signal_cutoff = end_of_calendar_day
    execution_rule = next_session_open

Ejemplo:

    señal EOD:       2026-01-30
    noticia:         2026-01-30 23:34
    ejecución real:  apertura de la siguiente sesión bursátil

El CSV guarda execution_date y execution_open_price para que el futuro motor de backtest use exactamente esa convención.

## Semántica long-only

    BUY  = abrir/aumentar largo
    HOLD = no añadir
    SELL = salir/no mantener largo

SELL NO significa abrir un short.

## Fase B — backtest con VectorBT

El motor `run_backtest.py` consume exclusivamente `signals.csv`; no vuelve a invocar ningún LLM.

Semántica de cartera:

    BUY  = añadir size_pct del valor TOTAL actual de la cartera
    HOLD = mantener la posición existente
    SELL = liquidar completamente la posición larga

Se utiliza una única bolsa de efectivo compartida entre todos los tickers y se respeta el `execution_open_price` ya guardado en el CSV. Cuando coinciden órdenes el mismo día, se procesan SELL antes de BUY.

La simulación usa VectorBT mediante `Portfolio.from_order_func()`, porque necesitamos sizing incremental dependiente del valor actual de cartera. VectorBT mantiene posiciones, cash, órdenes, comisiones, slippage y curva de equity.

Benchmarks incluidos:

    MAS
    SPY Buy & Hold
    Universe Buy & Hold
    Equal Weight Rebalanced

`Universe Buy & Hold` compra los tickers del universo a pesos iguales al principio y no rebalancea. `Equal Weight Rebalanced` vuelve a 1/N en cada fecha de ejecución de señales.

Si no se especifica `--evaluation-end`, el motor evalúa hasta el fin del mes posterior a la última `decision_date`. Esto da tiempo a actuar a la última señal.

Instalación:

    uv pip install -r "backtest.v1/requirements.txt"

Ejecución base, sin costes:

    python "backtest.v1/run_backtest.py"

Con costes:

    python "backtest.v1/run_backtest.py" \
      --fees-bps 10 \
      --slippage-bps 5

Con fecha final explícita:

    python "backtest.v1/run_backtest.py" \
      --evaluation-end 2026-04-30

Capital inicial distinto:

    python "backtest.v1/run_backtest.py" \
      --initial-cash 10000

Salidas:

    backtest.v1/output/backtest/summary.csv
    backtest.v1/output/backtest/equity_curve.csv
    backtest.v1/output/backtest/mas_orders.csv
    backtest.v1/output/backtest/mas_trades.csv
    backtest.v1/output/backtest/signals_used.csv

Métricas: rentabilidad total/anualizada, volatilidad anualizada, Sharpe con rf=0, max drawdown, exposición media/máxima, turnover, órdenes, trades cerrados e hit rate.


### Compatibilidad Plotly / VectorBT

VectorBT 1.1.0 todavía inicializa plantillas que contienen `scattermapbox`. Plotly 7 elimina ese tipo de traza, por lo que `import vectorbt` puede fallar antes de ejecutar el backtest. Por eso requirements.txt fija:

    plotly>=6.0,<7.0

Si aparece un error `Invalid property ... scattermapbox`, reinstala las dependencias del backtest:

    uv pip install --upgrade -r "backtest.v1/requirements.txt"

Y comprueba:

    python -c "import plotly, vectorbt; print('plotly', plotly.__version__, 'vectorbt', vectorbt.__version__)"
