# Plan v3 — Fusión mel-spectrograma + tempograma

Plan del próximo experimento de modelo (v3), orientado a superar el techo de v1/v2
—dominado por la confusión de **progressive** con el cluster melódico-house
(deep house / melodic techno)—.

> **Estado: implementado y testeado** (falta correr la experimentación real).
> Código: `features/multifeature.py`, `data/preprocess_v3.py`,
> `models/fusion.py`, `data/torch_dataset.py::CachedMultiFeatureDataset`,
> `training/train_v3.py`.
>
> **Tamaño del caché v3: ~36 GB** (mel 6.5 + fourier 10 + autocorr 20). Requiere
> espacio en Drive **y** en `/content` de Colab. Opción pendiente para reducirlo a
> ~16 GB: pooling del eje BPM de los tempogramas.

## Cómo correr (checklist en Colab)

1. Celda 1 (pull + install) → **Restart session** → Celda 2 (rutas).
2. Preprocess v3 → genera `cache_v3` en Drive (tarda más que v2: 2 tempogramas
   por track). Verificar `result.total_gb()`.
3. Copiar `cache_v3` a `/content/cache_v3` (verificar tamaño completo).
4. Entrenar con `train_fusion_model(LOCAL_V3, SPLITS, run_v3, ...)` — **reusa el
   mismo `SPLITS`** que v1/v2 → comparación apples-to-apples.
5. Comparar `run_v3/report.json` (`test_track`) contra v2 (0.80 / 0.917 top-2) y
   mirar el F1 de progressive + la matriz de confusión.

## Motivación y respaldo bibliográfico

El baseline (solo mel-spectrograma) separa bien los subgéneros de timbre distinto
pero confunde los del cluster melódico-house, que suenan parecido en el mel. La
señal que los distingue es **rítmica/tempo**, no espectral.

Esto está validado por **Hsu, Chen & Yang (2021), arXiv:2110.08862**
(referencia [1] del plan de trabajo), *"Deep Learning Based EDM Subgenre
Classification using Mel-Spectrogram and Tempogram Features"*:

- Mismo backbone que usamos: **Short-chunk CNN + ResNet, 7 bloques**, sr 22050,
  mel 128, win 2048, hop 512.
- Agregan **dos tempogramas** (Fourier + autocorrelación) al mel, con fusión.

| Modelo (30 clases, song-level) | Accuracy |
|---|---|
| Mel-spectrogram solo (30s) | 50.4% |
| Mel-spectrogram solo (120s) | 55.4% |
| Fourier tempograma solo | 34.9% |
| Autocorrelación tempograma solo | 31.2% |
| Early-fusion (mel + 2 tempo) | 60.3% |
| **Late-fusion (mel + 2 tempo)** | **60.6%** |

**Hallazgos clave:**
1. El tempograma **solo** es peor que el mel, pero **sumado** da **+5% (vs mel
   120s) y ~+10% (vs mel 30s)**. Es información complementaria.
2. **Late fusion ≥ early fusion** (marginal, gana late).
3. **Fourier tempograma > autocorrelación**.
4. La mejora se concentra en **pares confundibles en el mel pero distintos en
   tempo** (ej. uplifting-trance vs tech-trance) — análogo a nuestro progressive.
5. Código y checkpoints liberados: `github.com/mir-aidj/EDM-subgenre-classifier`.

## Arquitectura propuesta (late fusion)

```
Mel-spectrograma ──► SCcnn (7 bloques ResNet) ──────────────┐
                                                            ├─► concat ─► clasificador (dense+BN+ReLU+dropout+softmax)
Fourier tempo    ─┐                                         │
                 ├─► rama de fusión (1D convs) ─────────────┘
Autocorr tempo   ─┘
```

- **Rama de tempograma** (según el paper): 4 convoluciones 1-D paralelas (kernels
  3,3,5,5; strides 2,3,3,5, inspiradas en Pons et al.), concatenación, mean-pool,
  una conv 2-D y max-pool → representación combinada.
- **Late fusion:** los dos tempogramas se combinan **después** de las convs 1-D;
  luego se concatena con la salida de la rama mel antes del clasificador.

## Features y parámetros (del paper)

- **Mel-spectrograma:** ya implementado (128 mel, win 2048, hop 512).
- **Fourier tempograma:** `librosa.feature.fourier_tempogram` (hop 512, win 2048).
- **Autocorrelación tempograma:** `librosa.feature.tempogram` (ya implementado en
  `features/extractors.py`).
- Normalización z-score por dimensión.
- El paper computa los tempogramas sobre un segmento de 30s (15s–45s) y luego los
  divide en chunks alineados con el mel (128×200 ≈ 4.6s, muy cerca de nuestros 4s).

## Pasos de implementación (en nuestro codebase)

1. **`features/extractors.py`** — agregar `fourier_tempogram` (autocorrelación ya
   está). Definir la forma/normalización consistente con el mel por segmento.
2. **`features/pipeline.py` + `data/preprocess.py`** — cachear también los dos
   tempogramas por segmento (el caché crece; evaluar tamaño y formato). Mantener
   el streaming a disco para no OOM.
3. **`models/`** — nueva arquitectura de fusión tardía (rama mel = SCcnn actual +
   rama tempograma con convs 1-D + concatenación + clasificador).
4. **`data/torch_dataset.py`** — el dataset debe entregar (mel, fourier, autocorr)
   por segmento.
5. **`training/train.py`** — adaptar el loop a la entrada multi-feature (mínimo;
   el resto se reusa).
6. **Re-preprocesar** el dataset (features nuevas) y **re-entrenar**; comparar
   `test_track` contra v2 (0.80 / 0.917 top-2).

## Riesgos / adaptaciones a validar

- **Estabilidad del tempograma en 4s:** el paper lo computa sobre 30s. La
  estimación de tempo en ventanas de 4s puede ser más ruidosa. Alternativa:
  computar el tempograma sobre una ventana más larga y alinear los chunks.
- **Tamaño del caché:** sumar dos tempogramas (~193×T y ~384×T) aumenta el disco;
  medir y, si hace falta, guardar por-feature.
- **Payoff incierto en 8 clases:** el paper es 30 clases; el margen de mejora
  relativo debería transferir, pero conviene medir contra el baseline honesto.

## Criterio de éxito

v3 supera a v2 si mejora el `test_track.accuracy` y/o el F1 de **progressive**
(y del cluster deep house / melodic techno) sin degradar los subgéneros fuertes.
Si no supera a v2, se conserva v2 como campeón.
