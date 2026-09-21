# Chapter 8 / openai.v1 / backtest.v1

Esta versión separa el problema en dos fases:

FASE A — generación de señales (usa LLM y datos históricos point-in-time)
datos históricos -> comité multiagente -> signals.csv

FASE B — motor de backtest (siguiente paso)
signals.csv -> cartera -> PnL -> Sharpe / drawdown / benchmarks

El objetivo es que la fase B pueda ejecutarse muchas veces sin volver a pagar llamadas a modelos.

## Diferencia frente a backtest.py original

El backtest.py del capítulo recorre fechas históricas, pero llama al comité actual, que usa noticias, fundamentales, técnicos y macro actuales. Sirve como demostración de estructura, no como backtest point-in-time.

backtest.v1 pasa una fecha histórica real a toda la capa de datos:

- Técnicos: solo precios hasta as_of_date.
- Macro: VIX, Treasury 10Y y S&P 500 hasta as_of_date.
- Fundamentales: último 10-K SEC que YA estaba presentado en as_of_date.
- Noticias: Alpha Vantage NEWS_SENTIMENT filtrado por ticker y acotado entre la fecha previa y as_of_date.
- Se solicitan hasta 50 candidatos por ventana y solo se pasan al agente los que superan un umbral de relevancia del ticker (0.20 por defecto).
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
    ALPHAVANTAGE_PROVIDER_NEWS_LIMIT=50

El primer parámetro elimina artículos tangenciales; el segundo define cuántos candidatos pide al proveedor antes de quedarse con los 12 titulares más relevantes.

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
- proveedor, ventana histórica de noticias, umbral de relevancia, candidatos raw, descartados y titulares finales
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

## Próxima fase

El siguiente componente consumirá signals.csv con un motor de backtesting (VectorBT u otro) y comparará, sin llamadas adicionales al LLM:

    Sistema multiagente
    vs S&P 500
    vs Equal Weight del universo
    vs Buy & Hold del universo

Métricas previstas: rentabilidad, rentabilidad anualizada, volatilidad, Sharpe, max drawdown, hit rate, turnover y número de operaciones.
