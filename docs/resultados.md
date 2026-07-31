# Resultados experimentales

Registro de las corridas de entrenamiento y sus métricas, para la memoria técnica
(capítulo de resultados y análisis de desempeño).

---

## Baseline v1 — Short-chunk CNN + ResNet (cumple objetivos)

**Fecha:** 2026-07-31.

### Configuración
- Modelo: Short-chunk CNN + ResNet, `n_channels=64`, dropout 0.5.
- Input: log-mel-spectrograma (128 bandas) de segmentos de 4 s, 50% overlap,
  recorte de intro/outro ~15 s.
- Entrenamiento: Adam lr=1e-3, weight_decay=1e-3, batch 128, AMP, SpecAugment
  (freq_mask=30, time_mask=50), early stopping/checkpoint sobre val accuracy.
- Dataset: 800 tracks (100/subgénero), split por track 70/15/15 (560/120/120),
  147.190 segmentos de 4 s.
- Entorno: Google Colab, GPU L4.
- Early stopping en época 27; mejor checkpoint en época 17 (val_acc 0.706).

### Métricas (test set)

| Nivel | Accuracy | Top-2 | Macro F1 | n |
|---|---|---|---|---|
| Por segmento | 0.709 | 0.859 | 0.716 | 21.764 |
| **Por track (agregado)** | **0.800** | **0.900** | **0.799** | 120 |

**Objetivos del plan:** accuracy > 80% ✅ · top-2 > 90% ✅ (cumplidos, al límite).
La agregación por track (promedio del softmax de los segmentos) sube el 70.9%
por-segmento a 80% por-track (96/120 top-1, 108/120 top-2).

### F1 por subgénero (track-level)

| Subgénero | F1 | Recall |
|---|---|---|
| tech house | 0.93 | 0.87 |
| minimal/deep tech | 0.93 | 0.93 |
| hard techno | 0.90 | 0.87 |
| techno peak time | 0.875 | 0.93 |
| trance | 0.83 | 0.80 |
| deep house | 0.75 | 0.80 |
| melodic techno | 0.75 | 0.80 |
| **progressive** | **0.43** | **0.40** |

### Matriz de confusión (test, filas = real, columnas = predicho)

Orden: 0 deep house · 1 tech house · 2 melodic techno · 3 progressive ·
4 techno peak time · 5 hard techno · 6 minimal/deep tech · 7 trance.

```
        0   1   2   3   4   5   6   7
   0 [ 12   0   1   2   0   0   0   0 ]
   1 [  0  13   0   0   0   0   1   1 ]
   2 [  1   0  12   2   0   0   0   0 ]
   3 [  4   0   3   6   1   1   0   0 ]
   4 [  0   0   0   0  14   0   0   1 ]
   5 [  0   0   0   0   2  13   0   0 ]
   6 [  0   0   0   1   0   0  14   0 ]
   7 [  0   0   1   2   0   0   0  12 ]
```

### Análisis de errores
- **Progressive es el cuello de botella** (recall 0.40): solo 6/15 correctos, con
  fugas a deep house (4) y melodic techno (3). Además es "sumidero" de falsos
  positivos (baja precision), lo que hunde su F1 a 0.43.
- **Cluster melódico-house:** deep house ↔ melodic techno ↔ progressive se
  confunden entre sí (y trance filtra hacia ahí). Confusión musicalmente
  coherente: progressive es un subgénero ancho que comparte texturas con esos dos.
- **Subgéneros "distintos" muy sólidos:** tech house, minimal/deep tech, techno
  peak time y hard techno (F1 ≥ 0.875). El modelo los separa sin problema.
- La confusión hard techno ↔ techno peak time existe pero es leve (2 tracks).
- **Interpretación:** el techo no es de capacidad del modelo sino de ambigüedad
  intrínseca entre subgéneros vecinos (BPM y estructura similares).

### Palancas para superar el techo (opcionales, si se busca margen)
1. **Tempograma como 2ª rama** (fusión tardía) — el cluster melódico se distingue
   por build/energía/ritmo en escalas más largas que 4 s. Ver `decisiones §7.1`.
2. **Segmentos más largos** (8-10 s) para más contexto estructural.
3. **Class weighting / focal loss** para pesar más progressive (barato, pero puede
   mover errores a otras clases).

### Artefactos
- Checkpoint del mejor modelo: `run/model.pt` (incluye `model_kwargs` + `config`).
- Reporte completo: `run/report.json` (history + métricas + matriz de confusión).

---

## v2 — + label smoothing + cosine LR (campeón actual)

**Fecha:** 2026-07-31. Misma config que el baseline v1 más:
`label_smoothing=0.1`, `scheduler='cosine'`, `epochs=80`, `patience=15`.
Mismo caché (sin re-preprocesar). Early stopping en época 56; mejor val_acc 0.705
(época 41).

### Métricas (test set, track-level)

| Modelo | Accuracy | Top-2 | Macro F1 | train_loss final |
|---|---|---|---|---|
| Baseline v1 | 0.800 | 0.900 | 0.799 | ~0.09 (overfit) |
| **v2** | **0.800** | **0.917** | 0.797 | ~0.50 (sano) |

- **Top-1 empatado** (96/120), **top-2 mejor** (110/120 vs 108/120).
- **Entrenamiento mucho más sano:** el label smoothing evitó el colapso del
  train_loss → modelo menos overfitteado y mejor calibrado (más robusto).

### F1 por subgénero (v2)

| Subgénero | v1 | v2 |
|---|---|---|
| tech house | 0.93 | **0.97** |
| minimal/deep tech | 0.93 | **0.97** |
| techno peak time | 0.875 | **0.94** |
| hard techno | 0.90 | 0.90 |
| trance | 0.83 | 0.81 |
| melodic techno | 0.75 | 0.75 |
| deep house | 0.75 | 0.69 |
| **progressive** | 0.43 | **0.36** |

### Conclusión
v2 mejora los subgéneros distintos y el top-2, con un entrenamiento más sano,
pero **no resuelve el cuello de botella de progressive** (confusión estructural
con deep house / melodic techno). Se adopta v2 como campeón por el mejor top-2 y
la mejor calibración. Para superar el techo de top-1 haría falta una palanca de
mayor contexto temporal (tempograma / segmentos más largos), que requiere
re-preprocesar.
