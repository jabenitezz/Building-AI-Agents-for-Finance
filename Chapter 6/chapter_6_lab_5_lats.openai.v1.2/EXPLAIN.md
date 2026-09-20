# Cómo leer las trazas de LATS v1.2

Esta versión imprime el recorrido del algoritmo para poder relacionar directamente la teoría con la ejecución.

La secuencia de cada iteración aparece marcada como:

    [1] SELECTION
    [2] EXPANSION + [3] EVALUATION
    [4] TERMINAL CHECK
    [5] BACKPROPAGATION
    [6] REFLECTION

## SELECT / UCT

Cuando veas:

    [TRACE:UCT] root/c1: V=0.400, N=1, parent_N=4, UCT=2.0610
    [TRACE:UCT] root/c2: V=0.500, N=3, parent_N=4, UCT=1.4580

significa que el algoritmo está comparando ramas mediante UCT.

Después:

    [TRACE:SELECT] choose root/c1 because it has max UCT=2.0610

muestra qué rama ha elegido.

## EXPANSION

Una línea:

    [TRACE:PROPOSAL] c1: positions=['NVDA', 'AMD'] pct=4.0 ...

es una acción nueva propuesta por el LLM.

Después:

    [TRACE:ENV] ...

muestra cómo Python aplica esa acción al entorno simulado.

## EVALUATION

La traza:

    [TRACE:EVAL] model_output=...
    [TRACE:EVAL] normalized_reward=0.4

muestra la salida visible del evaluador y su recompensa normalizada.

## BACKPROPAGATION

Por ejemplo:

    [TRACE:BACKPROP] root/c1/c2: N 0->1, V 0.000->0.400
    [TRACE:BACKPROP] root/c1: N 1->2, V 0.500->0.450
    [TRACE:BACKPROP] root: N 4->5, V 0.425->0.420

permite ver cómo la recompensa del hijo se propaga hacia sus antecesores.

## REFLECTION Y LESSONS

La reflexión produce una regla:

    [lesson] When ..., do ...

En iteraciones posteriores puede aparecer:

    [TRACE:MEMORY] lesson[1] = ...

Eso confirma que la lección anterior está volviendo al prompt de expansión.

## Árbol después de cada iteración

Al final de cada vuelta aparece:

    [TRACE:TREE] tree after iteration

seguido del árbol completo con:

    N = visitas
    V = valor medio
    exposure = exposición tecnológica

Esta es la mejor parte para observar cómo LATS va cambiando su opinión sobre las ramas a medida que recibe nueva evidencia.

## Esquema mental

    SELECTION
       |
       v
    calcula UCT
       |
       v
    elige rama
       |
       v
    EXPANSION
       |
       v
    crea hijos
       |
       v
    EVALUATION
       |
       v
    reward
       |
       v
    BACKPROP
       |
       v
    actualiza N y V
       |
       v
    REFLECTION
       |
       v
    LESSON
       |
       v
    siguiente iteración
