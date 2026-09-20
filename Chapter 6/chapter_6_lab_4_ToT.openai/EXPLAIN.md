# Cómo funciona `tot.py`: Tree of Thoughts paso a paso

Este documento explica **cómo encaja el patrón Tree of Thoughts (ToT) con el código real de `tot.py`**.

La idea principal es:

> **ToT = GENERAR → EVALUAR → PODAR → EXPANDIR → EVALUAR**

En este laboratorio el problema no se resuelve con una única respuesta del modelo. El programa crea varias alternativas, las puntúa, elimina las menos prometedoras y profundiza solamente en las mejores.

> Nota de visualización: los esquemas de este documento están hechos en formato estrecho y con bloques Markdown clásicos para que se vean correctamente también en visores de libros electrónicos y lectores Markdown que no soportan bien los bloques con triple acento grave.

---

## 1. El problema inicial es la raíz del árbol

El script empieza con:

    portfolio = (
        "5,000 shares of NVDA held outright "
        "(~$5M concentrated single-name position)"
    )

    scenario = (
        "NVDA reports earnings tonight after the close; "
        "weekly implied volatility is elevated"
    )

Conceptualmente:

    RAÍZ
      |
      v
    Posición concentrada en NVDA
      |
      +-- resultados después del cierre
      |
      +-- volatilidad implícita elevada
      |
      v
    ¿Cómo cubrir la posición?

El árbol comienza ahí.

---

## 2. Nivel 1: generar varias alternativas

La primera expansión ocurre en:

    candidates = generate_hedges(
        portfolio,
        scenario,
        n=4
    )

La función `generate_hedges()` pide al modelo **4 candidatos de cobertura distintos**:

    def generate_hedges(portfolio, scenario, n):
        prompt = (
            f"Portfolio: {portfolio}\n"
            f"Stress scenario: {scenario}\n"
            f"Propose {n} distinct hedge candidates..."
        )

El modelo podría devolver, por ejemplo:

    ROOT
      |
      +-- Hedge 1: Comprar puts
      |
      +-- Hedge 2: Put spread
      |
      +-- Hedge 3: Collar
      |
      +-- Hedge 4: Hedge sectorial

Cada alternativa es una **rama del árbol**.

En esta fase todavía no se elige ninguna. Solo se amplía el espacio de soluciones.

---

## 3. El modelo evalúa cada rama

Después, cada candidato pasa por:

    score_hedge(
        portfolio,
        scenario,
        hedge
    )

La función le pide al modelo una puntuación de 1 a 10:

    Score this hedge from 1 to 10
    on protection, cost, and basis risk.

Los criterios son:

- protección;
- coste;
- basis risk.

Una ejecución conceptual podría producir:

    Comprar puts       -> 8
    Put spread         -> 9
    Collar             -> 7
    Hedge sectorial    -> 5

El código combina cada candidato con su evaluación:

    scored = [
        {
            **c,
            **score_hedge(
                portfolio,
                scenario,
                c
            )
        }
        for c in candidates
    ]

Cada nodo pasa a tener aproximadamente:

    {
        "name": "Put spread",
        "rationale": "...",
        "score": 9,
        "reasoning": "..."
    }

Así queda el primer nivel:

    ROOT
      |
      +-- Comprar puts    [8]
      |
      +-- Put spread      [9]
      |
      +-- Collar          [7]
      |
      +-- Hedge sectorial [5]

---

## 4. La poda: Beam Search

Primero el código ordena los candidatos:

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

Ejemplo:

    1. Put spread       9
    2. Comprar puts     8
    3. Collar           7
    4. Hedge sectorial  5

Después:

    retained = scored[:beam_width]

Como por defecto:

    beam_width = 2

solo sobreviven las dos mejores ramas:

    ROOT
      |
      +-- Put spread   [9]  -> CONTINÚA
      |
      +-- Comprar puts [8]  -> CONTINÚA
      |
      +-- Collar       [7]  -> PODADO
      |
      +-- Sector hedge [5]  -> PODADO

Esto es la **poda del árbol**.

La idea de Beam Search es:

> No explorar todas las ramas hasta el final. Profundizar únicamente en las más prometedoras.

---

## 5. Nivel 2: expandir las ramas supervivientes

Ahora el algoritmo profundiza en cada una de las dos mejores coberturas.

La función es:

    generate_sizings(
        portfolio,
        scenario,
        hedge,
        n=3
    )

Para cada cobertura retenida genera **3 variantes de sizing o estructura**.

Por ejemplo:

    Put spread [9]
      |
      +-- S1: 5% / 15% OTM, semanal
      |
      +-- S2: ATM / 10% OTM, semanal
      |
      +-- S3: 10% / 20% OTM, mensual

Y para la segunda rama:

    Comprar puts [8]
      |
      +-- S1
      |
      +-- S2
      |
      +-- S3

El árbol ya tiene dos niveles:

    ROOT
      |
      +-- Put spread [9]
      |     |
      |     +-- S1
      |     +-- S2
      |     +-- S3
      |
      +-- Comprar puts [8]
            |
            +-- S1
            +-- S2
            +-- S3

Las otras dos ramas ya no existen para el resto de la búsqueda porque fueron podadas.

---

## 6. Evaluar las subramas

Cada sizing se evalúa con:

    score_sizing(
        portfolio,
        scenario,
        hedge,
        sizing
    )

En este nivel cambian los criterios:

    Score this sizing from 1 to 10 on:
    - residual exposure to the catalyst
    - premium cost
    - how cleanly it complements the parent hedge

Es decir:

- exposición residual al evento;
- coste de la prima;
- cómo complementa al hedge padre.

Ejemplo:

    Put spread [9]
      |
      +-- S1 -> 7
      |
      +-- S2 -> 9  <- MEJOR
      |
      +-- S3 -> 6

El código ordena:

    scored_sizings.sort(
        key=lambda x: x["score"],
        reverse=True
    )

y conserva el mejor:

    hedge["best_sizing"] = scored_sizings[0]

Lo mismo sucede con la segunda rama retenida.

---

## 7. Las iteraciones reales del algoritmo

### Iteración 1 — Expandir desde la raíz

    ROOT
      |
      +-- Hedge 1
      +-- Hedge 2
      +-- Hedge 3
      +-- Hedge 4

Código:

    generate_hedges(..., n=4)

### Iteración 2 — Evaluar el primer nivel

    Hedge 1 -> score
    Hedge 2 -> score
    Hedge 3 -> score
    Hedge 4 -> score

Código:

    score_hedge(...)

### Iteración 3 — Podar

    4 ramas
       |
       v
    ordenar
       |
       v
    conservar 2

Código:

    scored.sort(...)
    retained = scored[:beam_width]

### Iteración 4 — Expandir cada rama superviviente

    Hedge A
      |
      +-- S1
      +-- S2
      +-- S3

    Hedge B
      |
      +-- S1
      +-- S2
      +-- S3

Código:

    generate_sizings(..., n=3)

### Iteración 5 — Evaluar las nuevas ramas

    Hedge A
      |
      +-- S1 -> score
      +-- S2 -> score
      +-- S3 -> score

    Hedge B
      |
      +-- S1 -> score
      +-- S2 -> score
      +-- S3 -> score

Código:

    score_sizing(...)

### Iteración 6 — Elegir el mejor sizing

    Hedge A
      |
      +-- mejor sizing

    Hedge B
      |
      +-- mejor sizing

Código:

    hedge["best_sizing"] = scored_sizings[0]

---

## 8. El corazón de ToT está en `best_hedges()`

La función:

    best_hedges(...)

es prácticamente el algoritmo Tree of Thoughts completo.

Traducido a pseudocódigo:

    GENERAR 4 coberturas

    PARA cada cobertura:
        EVALUARLA

    ORDENAR por puntuación

    PODAR:
        conservar solo las 2 mejores

    PARA cada cobertura superviviente:
        GENERAR 3 variantes
        EVALUAR las 3
        ORDENAR
        CONSERVAR la mejor variante

En una sola línea:

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
       |
       v
    EVALUAR
       |
       v
    SELECCIONAR

---

## 9. ¿Dónde está el árbol si no hay una clase `Tree`?

El archivo no tiene:

    class Tree:
        ...

ni:

    class Node:
        ...

porque el árbol está representado **de forma implícita** por listas, diccionarios y bucles.

La correspondencia es:

    candidates
        =
    ramas del nivel 1

    retained
        =
    ramas que sobreviven a la poda

    sizings
        =
    ramas del nivel 2

    best_sizing
        =
    mejor hijo de cada hedge retenido

Por tanto:

> Tree of Thoughts no exige implementar físicamente una clase `Tree`. Lo importante es que el algoritmo genere, evalúe, pode y expanda alternativas.

---

## 10. El LLM hace dos trabajos distintos

En este laboratorio, `gpt-4.1-mini` actúa en dos papeles.

### Papel 1 — Generador

Lo hace en:

    generate_hedges()
    generate_sizings()

Su trabajo es:

    "Propón alternativas"

### Papel 2 — Evaluador

Lo hace en:

    score_hedge()
    score_sizing()

Su trabajo es:

    "Evalúa esta alternativa"

Pero la poda **no la realiza el modelo**.

La hace Python:

    scored.sort(...)
    retained = scored[:beam_width]

Visualmente:

    LLM GENERADOR
         |
         v
      crea ramas
         |
         v
    LLM EVALUADOR
         |
         v
     puntúa ramas
         |
         v
       PYTHON
         |
         v
     ordena y poda

Esta separación es importante porque el modelo no controla todo el algoritmo.

---

## 11. ¿Qué hace exactamente `call_structured()`?

Las cuatro funciones que consultan al LLM terminan pasando por:

    call_structured(
        prompt,
        schema,
        schema_name
    )

Esta función centraliza la llamada a OpenAI:

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=2048,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
        store=False,
    )

El objetivo del JSON Schema es evitar una respuesta libre difícil de procesar.

Para candidatos se espera algo parecido a:

    {
        "candidates": [
            {
                "name": "...",
                "rationale": "..."
            }
        ]
    }

Para una evaluación:

    {
        "score": 8,
        "reasoning": "..."
    }

Eso permite que Python haga:

    json.loads(text)

y después pueda ordenar usando directamente:

    x["score"]

---

## 12. Visualización completa del algoritmo

Esta es la misma búsqueda completa, pero en formato vertical para que se vea bien incluso en una página estrecha:

    PORTFOLIO + SCENARIO
             |
             v
      generate_hedges(4)
             |
             v
    +-------------------+
    | Nivel 1           |
    |                   |
    | Hedge A -> 9      |
    | Hedge B -> 8      |
    | Hedge C -> 6      |
    | Hedge D -> 5      |
    +-------------------+
             |
             v
       ordenar scores
             |
             v
        beam_width=2
             |
             v
    +-------------------+
    | RETENIDOS         |
    |                   |
    | Hedge A -> 9      |
    | Hedge B -> 8      |
    +-------------------+
             |
       +-----+-----+
       |           |
       v           v
    Hedge A     Hedge B
       |           |
       v           v
    3 sizings   3 sizings
       |           |
       v           v
    evaluar     evaluar
       |           |
       v           v
    mejor A     mejor B
       |           |
       +-----+-----+
             |
             v
      RESULTADO FINAL

Otra forma de verlo, mostrando las ramas:

    ROOT
      |
      +-- Hedge A [9]
      |     |
      |     +-- A1 [7]
      |     +-- A2 [9] <- mejor
      |     +-- A3 [6]
      |
      +-- Hedge B [8]
      |     |
      |     +-- B1 [8] <- mejor
      |     +-- B2 [6]
      |     +-- B3 [5]
      |
      +-- Hedge C [6] -> PODADO
      |
      +-- Hedge D [5] -> PODADO

---

## 13. Diferencia con Self-Consistency

En Self-Consistency:

    mismo problema
      |
      +-- respuesta 1
      +-- respuesta 2
      +-- respuesta 3
      +-- respuesta 4
      +-- respuesta 5
      |
      v
    votación

Los caminos son esencialmente independientes.

En Tree of Thoughts:

    problema
      |
      v
    generar alternativas
      |
      v
    evaluar
      |
      v
    podar
      |
      v
    profundizar en las mejores
      |
      v
    volver a evaluar

La diferencia clave:

    Self-Consistency
      =
    caminos independientes + consenso

    Tree of Thoughts
      =
    exploración estructurada
      + evaluación
      + poda
      + expansión

---

## 14. Correspondencia directa entre teoría y código

| Concepto ToT | Código en `tot.py` |
|---|---|
| Estado inicial | `portfolio` + `scenario` |
| Generar alternativas | `generate_hedges()` |
| Evaluar alternativas | `score_hedge()` |
| Ordenar ramas | `scored.sort(...)` |
| Poda / Beam Search | `retained = scored[:beam_width]` |
| Expandir supervivientes | `generate_sizings()` |
| Evaluar segundo nivel | `score_sizing()` |
| Elegir mejor hijo | `scored_sizings[0]` |
| Orquestador | `best_hedges()` |

---

## 15. La idea que hay que recordar

Si solo quieres recordar una cosa de este archivo:

    Tree of Thoughts en tot.py

    1. GENERA varias soluciones.
    2. EVALÚA cada solución.
    3. PODA las peores.
    4. EXPANDE las mejores.
    5. EVALÚA las nuevas ramas.
    6. SELECCIONA las mejores.

En este ejercicio:

    NIVEL 1
      |
      v
    ¿Qué instrumento de cobertura utilizar?

    NIVEL 2
      |
      v
    ¿Cómo estructurar exactamente
    ese instrumento?

Y por eso `best_hedges()` es el corazón del laboratorio:

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
       |
       v
    EVALUAR

Ese flujo es la traducción directa del patrón **Tree of Thoughts** al código de `tot.py`.
