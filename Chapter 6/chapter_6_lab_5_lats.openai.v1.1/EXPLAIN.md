# Nota para esta versión v1.1

Esta carpeta utiliza el mismo LATS explicado en la versión base, pero fuerza una simulación más larga para poder observar el algoritmo en funcionamiento.

Las diferencias de la simulación son:

    máximo 2 posiciones por acción
    percent_each = 0.5% .. 4.0%
    n_iterations = 16
    k_expand     = 2
    max_depth    = 6

Como cada acción puede reducir como máximo:

    4.0 × 2 × 0.4 = 3.2 puntos

no puede pasar de 40% a 30% de una sola vez.

Esto permite observar varias veces:

    SELECTION
       |
       v
    EXPANSION
       |
       v
    EVALUATION
       |
       v
    BACKPROPAGATION
       |
       v
    REFLECTION
       |
       v
    LESSON
       |
       v
    nueva iteración

El algoritmo LATS no cambia; cambia únicamente la dificultad del entorno de demostración.

---

# Cómo funciona LATS en este laboratorio

Este documento explica cómo encajan la teoría de **Language Agent Tree Search (LATS)** y el código real de:

- `lats_search.py`
- `lats_helpers.py`

La idea central es:

> **LATS = árbol de búsqueda + LLM + evaluación + UCT + backpropagation + reflexión**

A diferencia de Tree of Thoughts, aquí no se limita a generar unas ramas, podarlas y profundizar. LATS mantiene estadísticas de cada nodo, vuelve a recorrer el árbol en varias iteraciones y decide por dónde seguir usando **UCT**.

---

## 1. Qué problema intenta resolver

El entorno simulado empieza con:

    tech_exposure_pct = 40.0
    target_pct        = 30.0

El objetivo es reducir la exposición tecnológica del 40% al 30%.

El árbol empieza así:

    ROOT
      |
      v
    Exposición = 40%
    Objetivo   = 30%

Cada acción propuesta por el modelo consiste en recortar determinadas posiciones:

    positions = ["AAPL", "MSFT"]
    percent_each = 5.0

La función:

    apply_trim(...)

simula el efecto de esa acción y devuelve:

- nueva exposición;
- slippage;
- observación textual.

---

## 2. Los dos archivos principales

### `lats_search.py`

Contiene el algoritmo de búsqueda:

    Node
    uct()
    select()
    expand()
    backprop()
    lats_search()

Es el **orquestador**.

### `lats_helpers.py`

Contiene:

    INITIAL_MARKET
    apply_trim()
    propose_k_trims()
    evaluate_proposal()
    reflect_on_failure()
    print_tree()
    best_leaf()

Además contiene las llamadas al modelo OpenAI y los parsers.

Una forma sencilla de recordarlo:

    lats_search.py
        =
    lógica del árbol

    lats_helpers.py
        =
    entorno + LLM + evaluación + reflexión

---

## 3. Qué representa un `Node`

Cada nodo del árbol representa un estado después de aplicar una acción.

Tiene:

    state
    parent
    children
    depth
    visits
    value
    is_terminal
    proposal
    observation

Por ejemplo:

    root
      |
      +-- c1
      |    exposure = 34%
      |
      +-- c2
           exposure = 36%

Cada nodo guarda dos valores fundamentales:

    N = visits
    V = value

### N: número de visitas

Indica cuántas veces ha recibido actualizaciones durante la búsqueda.

### V: valor medio

Es la media de las recompensas que se han propagado por ese nodo.

La actualización se hace con:

    V_new =
        (V_old * (N - 1) + reward) / N

Esto está implementado en:

    Node.update(score)

---

## 4. Una iteración completa de LATS

Cada iteración de `lats_search()` sigue este flujo:

    SELECTION
       |
       v
    EXPANSION
       |
       v
    EVALUATION
       |
       v
    ¿TERMINAL?
      /   \
    sí     no
    |       |
    v       v
    FIN   BACKPROPAGATION
            |
            v
         REFLECTION
            |
            v
         LESSON
            |
            v
      siguiente iteración

Vamos paso por paso.

---

## 5. Paso 1: Selection

La función es:

    select(root, k_expand, max_depth)

Su trabajo es decidir qué nodo del árbol se va a expandir.

Al principio solo existe:

    root

Por tanto, en la primera iteración:

    Selected: root

Pero cuando ya existen varias ramas, utiliza:

    uct(...)

para decidir por cuál continuar.

---

## 6. UCT: explotación + exploración

La fórmula utilizada es:

    UCT =
        V
        +
        w * sqrt(
            ln(N_parent) / N_child
        )

donde:

    V
      =
    calidad conocida del nodo

    N_child
      =
    cuántas veces se ha visitado

    N_parent
      =
    visitas del padre

    w = 1.41
      =
    peso de exploración

La intuición es:

    nodo con buen V
        ->
    interesa explotarlo

pero también:

    nodo poco visitado
        ->
    interesa explorarlo

Por eso LATS no siempre continúa por la rama con mayor puntuación actual.

---

## 7. Paso 2: Expansion

Después de seleccionar un nodo:

    leaf = select(...)

se llama:

    expand(
        leaf,
        k_expand,
        lessons
    )

Dentro de `expand()` aparece:

    propose_k_trims(...)

El modelo recibe el estado actual y debe proponer `k` alternativas diferentes.

Con:

    k_expand = 2

podría generar:

    Nodo seleccionado
      |
      +-- c1
      |
      +-- c2

La generación se hace en **una única llamada al LLM**.

Eso es el:

    one-shot k-expand

del laboratorio.

---

## 8. Paso 3: aplicar cada acción al entorno

Para cada propuesta:

    apply_trim(
        node.state,
        positions,
        percent_each
    )

calcula un nuevo estado.

Por ejemplo:

    estado inicial
        =
    40%

    acción
        =
    recortar AAPL y MSFT
    un 5% cada una

    resultado simulado
        =
    nueva exposición
    + slippage

Importante:

> El LLM propone la acción, pero el LLM no modifica directamente el estado.

El entorno Python lo hace de forma determinista.

Flujo:

    LLM
     |
     v
    propone acción
     |
     v
    Python apply_trim()
     |
     v
    nuevo estado

---

## 9. Paso 4: Evaluation

Cada hijo nuevo se evalúa mediante:

    evaluate_proposal(...)

El evaluador usa la skill:

    trade-evaluator

Su rúbrica puntúa de 1 a 10 según:

- progreso hacia el objetivo;
- slippage;
- riesgo residual.

Después:

    _parse_score(...)

convierte:

    SCORE: 4

en:

    reward = 0.4

porque UCT trabaja con valores entre 0 y 1.

Por eso la salida muestra ahora:

    score=4.0/10
    reward=0.40

### Reward no es todavía V

Esto es importante.

Justo después de evaluar:

    reward = 0.40

pero el nodo todavía puede tener:

    V = 0.00
    N = 0

hasta que se ejecute:

    backprop(...)

Por eso se cambió la salida del programa de:

    V=0.40

a:

    reward=0.40

para evitar confundir la recompensa inmediata con el valor acumulado del nodo.

---

## 10. Paso 5: comprobar si el nodo es terminal

Un hijo es terminal cuando:

    tech_exposure_pct <= target_pct

Es decir:

    exposición <= 30%

Entonces:

    child.is_terminal = True

Si aparece al menos un terminal:

    terminal_pairs = ...

el algoritmo elige el terminal con mejor score y termina.

Por eso una ejecución puede terminar en la primera iteración.

Ejemplo:

    root 40%
      |
      +-- c1 -> 30% [TERMINAL]
      |
      +-- c2 -> 28% [TERMINAL]

En ese caso:

    LATS termina

y NO ejecuta:

    backpropagation
    reflection
    nueva iteración

Eso es exactamente lo que ocurrió en una de las primeras ejecuciones con GPT-4.1-mini.

---

## 11. ¿Por qué un terminal puede tener score bajo?

Porque:

    TERMINAL
      !=
    BUENA SOLUCIÓN

Terminal solo significa:

    se alcanzó el objetivo

La evaluación también considera el coste.

Una acción puede llevar:

    40% -> 30%

pero producir un slippage muy alto.

Entonces puede ocurrir:

    exposición = 30%
    TERMINAL
    score = 2/10

Es decir:

> llegó al objetivo, pero de una forma cara o poco eficiente.

---

## 12. Paso 6: Backpropagation

Si ningún hijo alcanza el objetivo:

    for child, score in new_pairs:
        backprop(child, score)

La función recorre:

    hijo
      |
      v
    padre
      |
      v
    abuelo
      |
      v
    root

actualizando en cada nodo:

    visits
    value

Ejemplo:

Antes:

    root
      N=0
      V=0

Se evalúa un hijo con:

    reward = 0.5

Después de backprop:

    child
      N=1
      V=0.5

    root
      N=1
      V=0.5

Si luego llega otra recompensa distinta, V se convierte en una media.

---

## 13. Paso 7: Reflection

Si la iteración no tuvo éxito, el algoritmo toma:

    best_child

entre los hijos recién creados y llama:

    reflect_on_failure(...)

La skill:

    trade-reflector

genera una regla del tipo:

    LESSON:
    When <situation>,
    do <action>.

Ejemplo conceptual:

    When the remaining exposure gap is large,
    do use larger trim sizes.

El parser:

    _parse_lesson(...)

extrae la lección.

Después se guarda en:

    lessons

---

## 14. La memoria de lessons

En la siguiente iteración:

    propose_k_trims(...)

recibe también:

    lessons

y las añade al prompt:

    Lessons from prior attempts:
    - ...
    - ...

Por tanto, el sistema aprende de sus intentos anteriores dentro de la ejecución.

Flujo:

    fallo
      |
      v
    reflexión
      |
      v
    LESSON
      |
      v
    memoria temporal
      |
      v
    siguiente expansión
      |
      v
    nuevas propuestas

Esto es una diferencia muy importante respecto a una búsqueda de árbol clásica.

---

## 15. Qué ocurre entre iteraciones

Supongamos una primera iteración:

    root 40%
      |
      +-- c1 34%  reward=.4
      |
      +-- c2 36%  reward=.5

Ninguno llega al 30%.

Se hace:

    backpropagation

y se guarda una:

    lesson

En la siguiente iteración, `select()` vuelve al root y calcula UCT.

Puede elegir:

    c2

Entonces se expande:

    root
      |
      +-- c1
      |
      +-- c2
           |
           +-- c1
           |
           +-- c2

Después se evalúan esas ramas y sus resultados vuelven hacia arriba mediante backpropagation.

---

## 16. Por qué el algoritmo puede cambiar de rama

Supongamos:

    c1
      V = 0.40

    c2
      V = 0.50

Al principio parece mejor:

    c2

Pero después profundizamos en c2 y sus hijos son malos:

    c2/c1 = 0.40
    c2/c2 = 0.20

Esas recompensas se propagan y pueden bajar el valor medio de c2:

    c2
      V: 0.50 -> 0.37

Mientras c1 puede seguir:

    c1
      V = 0.40

Además c1 tiene menos visitas, así que UCT le da mayor incentivo de exploración.

Entonces una siguiente iteración puede hacer:

    antes
      ->
    explorar c2

    después
      ->
    volver a c1

Esto es el feedback loop de MCTS.

---

## 17. Árbol completo conceptual

Formato vertical para que se vea bien también en visores estrechos:

    ITERACIÓN 1

    root 40%
      |
      +-- c1 34%
      |
      +-- c2 36%

    evaluar
      |
      v
    backprop
      |
      v
    reflection
      |
      v
    lesson

    ITERACIÓN 2

    UCT selecciona c2

    root
      |
      +-- c1 34%
      |
      +-- c2 36%
            |
            +-- c1 33%
            |
            +-- c2 34%

    evaluar
      |
      v
    backprop

    ITERACIÓN 3

    UCT vuelve a comparar
    todas las ramas

          |
          v

    puede continuar por c1
    o por c2 según:

      V
      +
      exploración

---

## 18. Diferencia con Tree of Thoughts

El laboratorio anterior hacía aproximadamente:

    GENERAR
      |
      v
    EVALUAR
      |
      v
    PODAR
      |
      v
    EXPANDIR

Eso es Tree of Thoughts con Beam Search.

LATS añade:

    visitas N
    valores V
    UCT
    backpropagation
    reflexión
    lessons

Por tanto:

    ToT
      =
    explorar alternativas
    y podar

    LATS
      =
    búsqueda iterativa tipo MCTS
    +
    LLM
    +
    reflexión

---

## 19. Correspondencia teoría -> código

| Concepto LATS | Código |
|---|---|
| Estado inicial | `INITIAL_MARKET` |
| Nodo | `Node` |
| Acción del entorno | `apply_trim()` |
| Selección | `select()` |
| UCT | `uct()` |
| Expansión | `expand()` |
| Generar acciones | `propose_k_trims()` |
| Evaluación | `evaluate_proposal()` |
| Recompensa | `score / 10` |
| Estado terminal | `child.is_terminal` |
| Backpropagation | `backprop()` |
| Reflexión | `reflect_on_failure()` |
| Memoria | `lessons` |
| Árbol final | `print_tree()` |
| Mejor hoja | `best_leaf()` |
| Orquestador | `lats_search()` |

---

## 20. Lo fundamental para recordar

Si solo quieres quedarte con una imagen mental:

    1. SELECCIONA
       una rama con UCT

    2. EXPANDE
       creando k acciones

    3. EVALÚA
       cada nueva acción

    4. SI LLEGA AL OBJETIVO
       termina

    5. SI FALLA
       propaga las puntuaciones

    6. REFLEXIONA
       y guarda una lección

    7. REPITE
       usando el árbol actualizado

En una sola línea:

    SELECT
      ->
    EXPAND
      ->
    EVALUATE
      ->
    BACKPROP
      ->
    REFLECT
      ->
    REPEAT

Eso es lo que implementa este laboratorio de **LATS**.
