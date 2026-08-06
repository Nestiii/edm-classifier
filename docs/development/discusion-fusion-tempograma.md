# Discusión: ¿por qué la fusión con tempograma no mejoró al mel solo?

Documento de análisis para el capítulo de discusión de la memoria técnica.
Aborda una pregunta central de los resultados: el modelo v3 (mel-spectrograma +
tempogramas, siguiendo Hsu et al. 2021, arXiv:2110.08862) es **más sofisticado y
tiene más features** que el modelo v2 (mel-spectrograma solo), y sin embargo v2
lo supera. ¿Por qué el enfoque más rico rinde peor?

## Punto de partida (resultados)

| Modelo | Dataset | Clases | Accuracy (test, track) |
|---|---|---|---|
| Paper (mel + tempograma, late fusion) | 75.000 tracks | 30 | 60.6% (vs 55.4% mel solo → **+5.2%**) |
| Nuestro v2 (mel solo) | 800 tracks | 8 | **80.0%** |
| Nuestro v3.1 (mel + tempograma) | 800 tracks | 8 | 76.7% (**−3.3%** vs v2) |

En el paper el tempograma **suma** (+5%); en nuestro caso **resta** (−3%). La
inversión no es azar: hay causas concretas, ordenadas de mayor a menor peso.

## Causa 1 — Régimen de datos: 30× menos datos (principal)

El paper entrenó con **75.000 tracks**; nosotros con **800** (560 de train).

- Más features implican **más parámetros** (dos ramas de tempograma + cabezal más
  grande). Más capacidad solo ayuda **si hay datos suficientes para estimarla**.
- Con 75k tracks el modelo aprende la señal del tempograma sin memorizar. Con
  560, la capacidad extra **overfittea**: aprende ruido en vez de estructura.
- **Evidencia empírica:** el val accuracy de v3 topó en ~0.677, **por debajo** del
  0.705 de v2, mientras el train loss seguía cayendo — patrón claro de
  sobreajuste.

Es el **compromiso sesgo-varianza**: las features ricas reducen el sesgo pero
elevan la varianza. Con datos escasos, la varianza anula la ganancia.

## Causa 2 — El tempograma discrimina donde nuestras clases no se confunden

El tempograma separa subgéneros de **tempo distinto**. Ahí aparece el desajuste
entre el problema del paper y el nuestro.

- **El paper: 30 subgéneros de todo el espectro EDM**, con tempos muy dispares
  (drum & bass ~170 BPM, dubstep ~140, house ~120, trance ~138, hardcore ~150,
  psy-trance ~140…). El tempo es genuinamente discriminativo entre muchos pares.
- **El nuestro: 8 subgéneros del cluster house/techno con tempos solapados.**

| Subgénero | BPM aprox. |
|---|---|
| deep house | 120-125 |
| melodic techno | 120-126 |
| progressive | 124-130 |
| tech house | 122-128 |
| minimal/deep tech | 125-130 |
| techno peak time | 128-135 |
| trance | 135-140 |
| hard techno | 145-155 |

El **cuello de botella** (deep house / progressive / melodic techno) vive todo en
**~120-128 BPM**. El tempograma **no puede separar lo que comparte tempo**. Y las
clases tempo-distintas (hard techno ~150, trance ~138) ya se clasifican bien sin
él.

En síntesis: **la señal del tempograma está concentrada justo donde no tenemos
problema, y ausente justo donde sí lo tenemos.**

## Causa 3 — Redundancia: el mel ya alcanza para las clases separables

Para las clases separables (tech house, minimal, techno peak time, hard techno)
el mel solo ya da **F1 ≥ 0.9**. Agregar el tempograma ahí es **redundante**: no
aporta información nueva, solo parámetros. Una feature ayuda únicamente si su
señal es **ortogonal** a la que ya se tiene. Para nuestras 8 clases, el
tempograma aporta poco nuevo pero consume capacidad → **neto negativo**.

Matiz honesto: *melodic techno* **sí mejoró** con tempograma (F1 0.81 vs 0.75), o
sea la señal existe y en algún caso ayuda — pero no compensa la pérdida en
*progressive* / *deep house*.

## Causa 4 — Factores menores (higiene metodológica)

- **Nuestra fusión es una simplificación** de la del paper (global pooling vs su
  mean-pool + convolución 2D). Extrae algo menos del tempograma; pero aunque
  fuera óptima, no rompe el límite de la Causa 2.
- **Contexto del tempograma:** el paper lo computa sobre 30 s; nosotros sobre
  ventanas alineadas de ~4 s (calculadas del track completo). Menos contexto =
  estimación de tempo algo más ruidosa.
- **Test pequeño (120 tracks):** cada track pesa 0.83%. La brecha v2–v3.1
  (0.80 vs 0.767) equivale a ~4 tracks; parte es varianza de muestreo. Por eso la
  afirmación robusta es "**no mejoró**", apoyada en el val accuracy
  consistentemente más bajo, no en la magnitud exacta.

## Lección generalizable

El resultado ilustra tres principios clásicos de aprendizaje automático:

1. **No free lunch / navaja de Occam.** Más features no es universalmente mejor.
   Ayuda solo si (a) llevan señal discriminativa **para las clases del problema**
   y (b) hay datos para aprenderlas. Ninguna condición se cumplió aquí.
2. **La utilidad de una feature depende de la estructura del problema.** El
   tempograma es valioso con subgéneros de tempo diverso e inútil con subgéneros
   isócronos.
3. **Capacidad ↔ datos.** Un modelo más grande exige más datos; de lo contrario,
   la varianza domina.

## Conclusión

El resultado **no contradice** al paper: replicamos su método fielmente y
mostramos **cuándo transfiere y cuándo no**. La fusión con tempograma es efectiva
con datasets grandes y subgéneros de tempo diverso (el escenario del paper), pero
**no transfiere** a un dataset chico de subgéneros house/techno con tempos
solapados (nuestro escenario). Delimitar esa frontera de aplicabilidad es una
contribución en sí misma, no un fracaso.

**Decisión de ingeniería:** se conserva **v2 (mel solo)** como modelo final; el
tempograma queda documentado como una palanca cuya utilidad depende de la
separabilidad por tempo de las clases confundibles.

## Trabajo futuro sugerido por este análisis

- **Más datos** por subgénero (o *transfer learning* desde un modelo de
  audio pre-entrenado) para habilitar features más ricas sin overfitting.
- Features que ataquen la confusión real del cluster house — de **timbre/textura
  y armonía** (p. ej. tonalidad/*key*, contenido armónico), no de tempo.
- Reformular el objetivo con **top-2** como métrica primaria, dado que gran parte
  del error es entre pares vecinos musicalmente ambiguos.
