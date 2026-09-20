# LATS OpenAI v1.2 — simulación profunda con trazas

Esta versión deriva de `chapter_6_lab_5_lats.openai.v1.1`.

**No cambia el algoritmo LATS.** Añade trazas pedagógicas para ver en pantalla qué está haciendo el algoritmo en cada paso.

## Qué muestra la traza

Durante cada iteración se imprimen:

- nodo inicial de la selección;
- valores `N`, `V` y `UCT` de cada hijo candidato;
- rama elegida por UCT;
- propuestas generadas por el LLM;
- acción aplicada;
- observación del entorno;
- salida visible del evaluador y recompensa normalizada;
- creación del nodo;
- actualizaciones de `N` y `V` durante backpropagation;
- reflexión;
- lesson generada;
- lessons que vuelven al prompt de siguientes expansiones;
- árbol completo después de cada iteración.

Ejemplo conceptual:

    [1] SELECTION
      [TRACE:UCT] root/c1: V=.4 N=1 UCT=1.57
      [TRACE:UCT] root/c2: V=.5 N=2 UCT=1.33
      [TRACE:SELECT] choose root/c1

    [2] EXPANSION + [3] EVALUATION
      [TRACE:PROPOSAL] ...
      [TRACE:ACTION] ...
      [TRACE:OBSERVATION] ...
      [TRACE:EVAL] normalized_reward=0.4

    [5] BACKPROPAGATION
      [TRACE:BACKPROP] root/c1/c1: N 0->1, V 0.000->0.400
      [TRACE:BACKPROP] root/c1: N 1->2, V 0.500->0.450
      [TRACE:BACKPROP] root: N 2->3, V 0.450->0.433

    [6] REFLECTION
      [lesson] When ..., do ...

## Activar o desactivar trazas

Por defecto están activadas:

```env
LATS_TRACE=1
```

Para ejecutar sin el detalle:

```env
LATS_TRACE=0
```

## Simulación

Mantiene las restricciones de v1.1:

```text
máximo 2 posiciones por acción
percent_each entre 0.5% y 4.0%
n_iterations = 16
k_expand     = 2
max_depth    = 6
```

## Ejecución

```bash
python lats_search.py
```

Las trazas muestran **estado y decisiones observables del algoritmo**. No intentan mostrar razonamiento privado del modelo; muestran propuestas, resultados, puntuaciones, UCT, backpropagation y lessons que forman parte explícita del laboratorio.
