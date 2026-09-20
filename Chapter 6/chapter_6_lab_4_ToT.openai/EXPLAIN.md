# Cómo funciona `tot.py`: Tree of Thoughts paso a paso

Este documento explica **cómo encaja el patrón Tree of Thoughts (ToT) con el código real de `tot.py`**.

La idea principal es sencilla:

> **ToT = GENERAR → EVALUAR → PODAR → EXPANDIR → EVALUAR**

En este laboratorio, el problema no se resuelve con una única respuesta del modelo. El programa crea varias alternativas, las puntúa, elimina las menos prometedoras y profundiza solamente en las mejores.

---

## 1. El problema inicial es la raíz del árbol

El script empieza con:

```python
portfolio = "5,000 shares of NVDA held outright (~$5M concentrated single-name position)"
scenario = "NVDA reports earnings tonight after the close; weekly implied volatility is elevated"
```

Conceptualmente:

```text
                         RAÍZ
                          │
                          ▼
      Posición concentrada de ~5 M$ en NVDA
                          +
            resultados después del cierre
                          +
          volatilidad implícita elevada
```

La pregunta implícita es:

```text
¿Cómo puedo cubrir esta posición ante este evento?
```

El árbol comienza ahí.

---

## 2. Nivel 1: generar varias alternativas

La primera expansión ocurre en:

```python
candidates = generate_hedges(portfolio, scenario, n=4)
```

La función `generate_hedges()` pide al modelo **4 candidatos de cobertura distintos**:

```python
def generate_hedges(portfolio: str, scenario: str, n: int) -> list[dict]:
    prompt = (
        f"Portfolio: {portfolio}\nStress scenario: {scenario}\n"
        f"Propose {n} distinct hedge candidates, each with a short rationale."
    )
```

El modelo podría devolver algo de este estilo:

```text
                         ROOT
                          │
              generar 4 alternativas
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
 Comprar puts       Put spread          Collar        Hedge sectorial
```

Cada alternativa es una **rama del árbol**.

En esta fase no se elige todavía ninguna. Solo se amplía el espacio de soluciones.

---

## 3. El modelo evalúa cada rama

Después, cada candidato pasa por:

```python
score_hedge(portfolio, scenario, hedge)
```

La función le pide al modelo una puntuación de 1 a 10:

```python
f"Score this hedge from 1 to 10 on protection, cost, and basis risk."
```

Los criterios son:

- protección;
- coste;
- basis risk.

Una ejecución conceptual podría producir:

```text
Comprar puts        → 8
Put spread          → 9
Collar              → 7
Hedge sectorial     → 5
```

El código combina cada candidato con su evaluación:

```python
scored = [
    {**c, **score_hedge(portfolio, scenario, c)}
    for c in candidates
]
```

Cada nodo pasa a tener aproximadamente esta información:

```json
{
  "name": "Put spread",
  "rationale": "...",
  "score": 9,
  "reasoning": "..."
}
```

Por tanto, en el primer nivel tenemos:

```text
                       ROOT
                        │
           ┌────────────┼────────────┬────────────┐
           │            │            │            │
           ▼            ▼            ▼            ▼
         Put        Put spread     Collar     Sector hedge
          8              9            7             5
```

---

## 4. La poda: Beam Search

Ahora aparece una pieza fundamental del algoritmo.

Primero ordena los candidatos:

```python
scored.sort(key=lambda x: x["score"], reverse=True)
```

Quedarían, por ejemplo:

```text
1. Put spread       9
2. Put              8
3. Collar           7
4. Sector hedge     5
```

Después:

```python
retained = scored[:beam_width]
```

Como por defecto:

```python
beam_width = 2
```

solo sobreviven las dos mejores ramas:

```text
                         ROOT
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
    Put spread          Put             descartadas
       9                 8              Collar (7)
                                          Sector hedge (5)
```

Esta es la **poda del árbol**.

La idea de Beam Search es:

> No explorar todas las ramas hasta el final.  
> Profundizar solo en las más prometedoras.

---

## 5. Nivel 2: expandir las ramas supervivientes

Ahora el algoritmo profundiza en cada una de las dos mejores coberturas.

La función utilizada es:

```python
generate_sizings(portfolio, scenario, hedge, n=3)
```

Para cada cobertura retenida genera **3 variantes de sizing o estructura**.

Por ejemplo:

```text
Put spread
   │
   ├── Sizing A: 5% / 15% OTM, semanal
   ├── Sizing B: ATM / 10% OTM, semanal
   └── Sizing C: 10% / 20% OTM, mensual
```

Y para la otra rama:

```text
Put
   │
   ├── Sizing A
   ├── Sizing B
   └── Sizing C
```

El árbol completo empieza a verse así:

```text
                              ROOT
                               │
                     generar 4 hedges
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
     Put spread               Put              ramas descartadas
        9                      8
         │                     │
    ┌────┼────┐           ┌────┼────┐
    ▼    ▼    ▼           ▼    ▼    ▼
   S1   S2   S3          S1   S2   S3
```

---

## 6. Evaluar las subramas

Cada sizing se evalúa con:

```python
score_sizing(portfolio, scenario, hedge, sizing)
```

Los criterios cambian respecto al nivel anterior:

```python
f"Score this sizing from 1 to 10 on residual exposure to the catalyst, "
f"premium cost, and how cleanly it complements the parent hedge."
```

Es decir:

- exposición residual al evento;
- coste de la prima;
- cómo complementa al hedge padre.

Por ejemplo:

```text
Put spread
   │
   ├── S1 → 7
   ├── S2 → 9
   └── S3 → 6
```

El código ordena:

```python
scored_sizings.sort(key=lambda x: x["score"], reverse=True)
```

y conserva el mejor:

```python
hedge["best_sizing"] = scored_sizings[0]
```

Así:

```text
Put spread
   │
   ├── S1 → 7
   ├── S2 → 9  ← MEJOR
   └── S3 → 6
```

Lo mismo sucede con la segunda rama retenida.

---

## 7. Las iteraciones reales del algoritmo

El proceso completo puede verse como varias iteraciones.

### Iteración 1 — Expandir desde la raíz

```text
ROOT
 │
 ├── Hedge 1
 ├── Hedge 2
 ├── Hedge 3
 └── Hedge 4
```

Código:

```python
generate_hedges(..., n=4)
```

### Iteración 2 — Evaluar el primer nivel

```text
Hedge 1 → score
Hedge 2 → score
Hedge 3 → score
Hedge 4 → score
```

Código:

```python
score_hedge(...)
```

### Iteración 3 — Podar

```text
4 ramas
   ↓
ordenar
   ↓
mantener solo 2
```

Código:

```python
scored.sort(...)
retained = scored[:beam_width]
```

### Iteración 4 — Expandir cada rama superviviente

```text
Hedge A                 Hedge B
  │                       │
  ├── S1                  ├── S1
  ├── S2                  ├── S2
  └── S3                  └── S3
```

Código:

```python
generate_sizings(..., n=3)
```

### Iteración 5 — Evaluar las nuevas ramas

```text
Hedge A
  ├── S1 → score
  ├── S2 → score
  └── S3 → score

Hedge B
  ├── S1 → score
  ├── S2 → score
  └── S3 → score
```

Código:

```python
score_sizing(...)
```

### Iteración 6 — Elegir el mejor sizing de cada hedge

```text
Hedge A → mejor sizing
Hedge B → mejor sizing
```

Código:

```python
hedge["best_sizing"] = scored_sizings[0]
```

---

## 8. El corazón de ToT está en `best_hedges()`

La función:

```python
best_hedges(...)
```

es prácticamente el algoritmo Tree of Thoughts completo.

Traducido a pseudocódigo:

```text
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
```

En una sola línea:

```text
GENERAR → EVALUAR → PODAR → EXPANDIR → EVALUAR → SELECCIONAR
```

---

## 9. ¿Dónde está el árbol si no hay una clase `Tree`?

El archivo no tiene algo como:

```python
class Tree:
    ...
```

ni:

```python
class Node:
    ...
```

porque el árbol está representado **de forma implícita** por listas, diccionarios y bucles.

La correspondencia es:

```text
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
rama ganadora dentro de cada hedge retenido
```

Por tanto:

> Tree of Thoughts no exige implementar físicamente una clase Tree.  
> Lo importante es que el algoritmo genere, evalúe, pode y expanda alternativas.

---

## 10. El LLM hace dos trabajos distintos

En este laboratorio, `gpt-4.1-mini` actúa en dos papeles.

### Papel 1 — Generador

Lo hace en:

```python
generate_hedges()
generate_sizings()
```

Su trabajo es:

```text
"Propón alternativas"
```

### Papel 2 — Evaluador

Lo hace en:

```python
score_hedge()
score_sizing()
```

Su trabajo es:

```text
"Evalúa esta alternativa"
```

Pero la poda no la realiza el modelo.

La hace Python:

```python
scored.sort(...)
retained = scored[:beam_width]
```

Por tanto:

```text
LLM GENERADOR
     │
     ▼
 crea ramas
     │
     ▼
LLM EVALUADOR
     │
     ▼
 puntúa ramas
     │
     ▼
PYTHON
     │
     ▼
ordena y poda
```

Esta separación es importante porque el modelo no controla todo el algoritmo.

---

## 11. ¿Qué hace exactamente `call_structured()`?

Las cuatro funciones que consultan al LLM terminan pasando por:

```python
call_structured(prompt, schema, schema_name)
```

Esta función centraliza la llamada a OpenAI:

```python
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
```

El objetivo del JSON Schema es evitar una respuesta libre difícil de procesar.

Por ejemplo, para candidatos esperamos:

```json
{
  "candidates": [
    {
      "name": "...",
      "rationale": "..."
    }
  ]
}
```

Y para una evaluación:

```json
{
  "score": 8,
  "reasoning": "..."
}
```

Eso permite que Python haga después:

```python
json.loads(text)
```

y pueda ordenar las ramas usando directamente:

```python
x["score"]
```

---

## 12. Visualización completa del algoritmo

```text
                         PORTFOLIO + SCENARIO
                                  │
                                  ▼
                         generate_hedges(n=4)
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
        Hedge A                Hedge B                Hedge C        Hedge D
           │                      │                      │              │
           ▼                      ▼                      ▼              ▼
      score_hedge            score_hedge            score_hedge    score_hedge
           │                      │                      │              │
          9                      8                      6              5
           │                      │                      │              │
           └────────────── ordenar + beam search ───────┴──────────────┘
                                  │
                         beam_width = 2
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
                   ▼                             ▼
               Hedge A                       Hedge B
                   │                             │
         generate_sizings(n=3)          generate_sizings(n=3)
                   │                             │
             ┌─────┼─────┐                 ┌─────┼─────┐
             ▼     ▼     ▼                 ▼     ▼     ▼
            A1    A2    A3                B1    B2    B3
             │     │     │                 │     │     │
             ▼     ▼     ▼                 ▼     ▼     ▼
           score score score             score score score
             │     │     │                 │     │     │
             7     9     6                 8     6     5
                   │                       │
                   ▼                       ▼
              best_sizing A           best_sizing B
                   │                       │
                   └───────────┬───────────┘
                               ▼
                        RESULTADO FINAL
```

---

## 13. Diferencia con Self-Consistency

En Self-Consistency se hace algo parecido a:

```text
mismo problema
    │
    ├── respuesta 1
    ├── respuesta 2
    ├── respuesta 3
    ├── respuesta 4
    └── respuesta 5
             │
             ▼
           votación
```

Los caminos son esencialmente independientes.

En Tree of Thoughts:

```text
problema
   │
   ▼
generar alternativas
   │
   ▼
evaluar
   │
   ▼
podar
   │
   ▼
profundizar únicamente en las mejores
   │
   ▼
volver a evaluar
```

La diferencia clave es:

```text
Self-Consistency
= varios caminos independientes + consenso

Tree of Thoughts
= exploración estructurada + evaluación + poda + expansión
```

---

## 14. Correspondencia directa entre teoría y código

| Concepto ToT | Código en `tot.py` |
|---|---|
| Estado inicial | `portfolio` + `scenario` |
| Generar pensamientos/alternativas | `generate_hedges()` |
| Evaluar pensamientos | `score_hedge()` |
| Seleccionar ramas | `scored.sort(...)` |
| Poda / Beam Search | `retained = scored[:beam_width]` |
| Expandir ramas supervivientes | `generate_sizings()` |
| Evaluar segundo nivel | `score_sizing()` |
| Elegir mejor hijo | `scored_sizings[0]` |
| Orquestador del árbol | `best_hedges()` |

---

## 15. La idea que hay que recordar

Si solo quieres recordar una cosa de este archivo, que sea esta:

```text
Tree of Thoughts en tot.py

1. GENERA varias soluciones.
2. EVALÚA cada solución.
3. PODA las peores.
4. EXPANDE las mejores.
5. EVALÚA las nuevas ramas.
6. SELECCIONA las mejores estructuras finales.
```

En este ejercicio:

```text
Nivel 1:
¿Qué instrumento de cobertura utilizar?

Nivel 2:
¿Cómo estructurar exactamente ese instrumento?
```

Y eso explica por qué `best_hedges()` es el corazón del laboratorio:

```text
GENERAR → EVALUAR → PODAR → EXPANDIR → EVALUAR
```

Ese flujo es la traducción directa del patrón **Tree of Thoughts** al código de `tot.py`.
