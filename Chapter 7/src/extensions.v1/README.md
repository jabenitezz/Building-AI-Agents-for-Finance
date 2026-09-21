# Extensions v1 — comparación paralela de empresas

Esta versión conserva la idea del Lab 5 (analizar varias compañías de forma
concurrente), pero cambia la salida para que sea realmente comparativa y fácil
de leer.

## Qué mejora respecto a `src/extensions/parallel_analysis.py`

- Recoge en paralelo los 8 ratios que usa realmente el capítulo:
  - Return on Assets (ROA)
  - Return on Equity (ROE)
  - Net Profit Margin
  - Gross Margin
  - Current Ratio
  - Quick Ratio
  - Debt-to-Equity Ratio
  - Interest Coverage Ratio
- No mezcla en pantalla las conversaciones de varios agentes ejecutándose a la vez.
- Puntúa los ratios de forma determinista usando los thresholds de `src/common.py`.
- Construye una tabla comparativa entre compañías.
- Calcula una puntuación global didáctica con el mismo peso para los 8 ratios.
- Identifica fortalezas y debilidades por compañía.
- Usa un último LLM únicamente para redactar en español la comparación final,
  sin inventar métricas nuevas.
- Señala cuál aparece como la compañía financieramente más fuerte **según este
  marco concreto de 8 ratios**, no como recomendación de inversión.

## Ejecución

El directorio se llama literalmente `extensions.v1`, por lo que se ejecuta
el script directamente:

```bash
cd "Chapter 7"
source .venv312/bin/activate
python "src/extensions.v1/parallel_comparison.py"
```

Portfolio personalizado:

```bash
python "src/extensions.v1/parallel_comparison.py" AAPL MSFT GOOGL NVDA JPM
```

Modelo usado para la síntesis en español:

```bash
COMPARATOR_MODEL=gpt-4.1-mini python "src/extensions.v1/parallel_comparison.py"
```

## Importante

La puntuación global es una ayuda didáctica: media simple de ocho indicadores.
No pretende sustituir un análisis sectorial ni una valoración de inversión.
Comparar empresas de sectores muy distintos (por ejemplo, un banco y una
empresa tecnológica) tiene limitaciones, porque algunos ratios no tienen el
mismo significado económico en todos los sectores.
