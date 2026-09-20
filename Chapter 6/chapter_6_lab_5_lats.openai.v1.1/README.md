# LATS OpenAI v1.1 — simulación de búsqueda más profunda

Esta versión parte de `chapter_6_lab_5_lats.openai` y mantiene el mismo algoritmo LATS. El objetivo de v1.1 es únicamente hacer **más difícil la simulación** para que pueda verse el comportamiento que en la versión original a veces no aparece porque GPT-4.1-mini alcanza el objetivo en la primera expansión.

## Qué se mantiene

Se conservan:

- `Node`
- UCT
- `select()`
- `expand()`
- `backprop()`
- reflexión y `lessons`
- `trade-evaluator`
- `trade-reflector`
- el entorno simulado 40% → 30%
- `gpt-4.1-mini`
- `k_expand=2`

No cambia el algoritmo LATS.

## Qué cambia en esta simulación

Las acciones están limitadas a:

```text
máximo 2 tickers por acción
percent_each entre 0.5% y 4.0%
```

Con la fórmula del entorno:

```text
reducción = percent_each × nº posiciones × 0.4
```

la máxima reducción de una sola acción es:

```text
4.0 × 2 × 0.4 = 3.2 puntos
```

Por tanto, ya no es posible saltar directamente de 40% a 30% en una sola expansión. Hacen falta varias acciones encadenadas.

Las restricciones aparecen tanto en el prompt como en validación Python, para que no dependan únicamente de que el modelo las respete.

## Búsqueda más larga

La ejecución por defecto pasa de:

```text
3 iteraciones
max_depth = 3
```

a:

```text
16 iteraciones
max_depth = 6
```

Se mantiene:

```text
k_expand = 2
```

La idea es poder observar varias veces:

```text
SELECT
  ↓
EXPAND
  ↓
EVALUATE
  ↓
BACKPROP
  ↓
REFLECT
  ↓
LESSON
  ↓
nueva selección UCT
```

## Coste esperado

En una iteración no terminal suelen producirse aproximadamente:

```text
1 llamada para proponer k candidatos
2 llamadas para evaluar los 2 hijos
1 llamada de reflexión
----------------------------------
≈ 4 llamadas al modelo
```

Con 16 iteraciones, el máximo teórico ronda 64 llamadas, aunque puede terminar antes si aparece una rama terminal.

## Ejecución

```bash
python lats_search.py
```

La llamada por defecto es:

```python
anyio.run(lats_search, 16, 2, 6)
```

equivalente a:

```text
n_iterations = 16
k_expand     = 2
max_depth    = 6
```

## Qué deberías observar

En lugar de:

```text
root 40%
  ├─ 30% [TERMINAL]
  └─ 28% [TERMINAL]
```

deberías ver un árbol que crece durante varias iteraciones, con cambios en `N` y `V`, selección mediante UCT, backpropagation y nuevas `lessons` que vuelven a entrar en las siguientes expansiones.
