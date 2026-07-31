# Registro de decisiones de diseño

Bitácora de las decisiones técnicas tomadas durante el desarrollo del *Sistema
de clasificación automática de subgéneros de música electrónica* (Trabajo Final,
CEIA-FIUBA), con su contexto y alternativas consideradas.

**Propósito:** material de referencia para la memoria técnica del trabajo final
—en particular para los capítulos de metodología, decisiones de arquitectura y
lecciones aprendidas—. Cada entrada documenta *qué* se decidió, *por qué*, qué
alternativas se evaluaron y dónde quedó implementado.

> Nota: varias decisiones se desvían del plan de proyecto original
> (`plan-de-trabajo.pdf`). Esas desviaciones se marcan explícitamente para poder
> justificarlas en el análisis de cumplimiento del plan (sección "Procesos de
> cierre" del plan).

---

## 1. Arquitectura general

### 1.1 Monorepo con dos servicios desacoplados
- **Decisión:** un único repositorio con dos servicios: `classifier/` (Python) y
  `ui/` (Electron + React + TypeScript), comunicados por una **API HTTP local**
  (FastAPI en `localhost`).
- **Contexto:** el Req 5.1 exige que el módulo de clasificación y la interfaz sean
  componentes independientes comunicados por API. Un monorepo mantiene ambos
  versionados juntos sin acoplarlos.
- **Alternativas:** llamar a Python directamente desde Electron (rechazado: acopla
  los procesos, difícil de testear el clasificador solo). El servidor local
  FastAPI es además el fallback que el propio plan anticipa en la mitigación del
  Riesgo 4.
- **Estado:** estructura creada; el clasificador corre standalone por CLI. La API
  FastAPI (`api/`) queda pendiente de implementación.

### 1.2 Stack tecnológico
- **Decisión:** Python con **uv** (gestor de dependencias), **PyTorch/torchaudio**
  (modelo), **Librosa** (features), **FastAPI** (API), **pytest + ruff**
  (testing/lint). UI con **electron-vite + React + TypeScript**.
- **Contexto:** PyTorch es el framework idiomático para la arquitectura
  Short-chunk CNN de referencia (Won et al.). uv acelera el manejo de entornos.
- **Alternativas evaluadas:** Poetry (vs uv), TensorFlow/Keras (vs PyTorch). Se
  optó por uv + PyTorch.

---

## 2. Reutilización de las experimentaciones previas

### 2.1 No reutilizar el código legacy, sí los conceptos
- **Decisión:** los 8 notebooks en `docs/experiments/` (de experimentaciones
  previas en Colab) se conservan como referencia histórica, pero el proyecto se
  **reescribe desde cero en PyTorch**. Se portan solo 4 conceptos, no código.
- **Contexto:** análisis de los notebooks (2026-07-29). Están en el stack
  equivocado (TensorFlow/Keras + sklearn), sin conexiones residuales, con
  segmentos de 30s/3s (no cortos) y features mayormente solo MFCC. Código
  exploratorio, con rutas hardcodeadas, código muerto y algunos bugs. **Pero**
  comparten el dominio exacto: los mismos 8 subgéneros.
- **Conceptos portados (reimplementados limpio):**
  1. Split a nivel *track* para evitar fuga de segmentos (de `cnn_4`).
  2. Agregación de predicciones por track (promedio del softmax de sus segmentos).
  3. Métricas que *retornan datos* (accuracy, top-2, F1, matriz de confusión).
  4. Baselines clásicos de sklearn como piso de comparación, corrigiendo el
     *data leakage* del `StandardScaler` del `ml_models.ipynb` original.
- **Referencia técnica:** para el Short-chunk CNN + ResNet se toma la
  implementación canónica de Minz Won (*sota-music-tagging-models*), no el port
  incompleto de TF que estaba comentado en `cnn_3`.

---

## 3. Dataset

### 3.1 100 tracks por género (800 total), no 200/1600
- **Decisión:** trabajar con el dataset real disponible: **100 tracks/género × 8
  = 800 tracks**.
- **Desviación del plan:** el Req 3.1 fijaba 200/género (1600 total).
- **Contexto:** el dataset ya existía de las experimentaciones previas (en Google
  Drive). No
  es bloqueante: la segmentación multiplica los ejemplos de entrenamiento a
  decenas de miles. El split se mantiene a nivel track (70/15/15), quedando
  560/120/120 tracks.
- **Implicancia:** con 560 tracks de train el riesgo de *overfitting* es real; se
  mitiga con *data augmentation* (SpecAugment) y *early stopping*.

### 3.2 Alias de nombres de carpeta
- **Decisión:** el indexador acepta **alias** de nombres de carpeta. Se mapea la
  carpeta legacy `minimal` → subgénero `minimal/deep tech`. Las carpetas paralelas
  `*_data` (versión pre-segmentada del dataset viejo) se **ignoran**.
- **Contexto:** el dataset en Drive tenía 7 de 8 nombres coincidentes; solo
  `minimal` difería. En vez de renombrar en Drive (carpeta compartida, de solo
  lectura), se agregó el alias en `config.py`.

### 3.3 Fuente: tracks completos, no la versión pre-segmentada
- **Decisión:** usar las carpetas de **tracks enteros** y segmentar nosotros; se
  descarta la carpeta de "4 segmentos por track" del dataset viejo.
- **Contexto:** la vieja división en 4 segmentos daba pedazos largos (~60s),
  incompatibles con la filosofía *short-chunk* y con muy pocos ejemplos (400/género).

---

## 4. Segmentación de audio

### 4.1 Ventanas de 4 segundos (no 2s), 50% de solapamiento
- **Decisión:** segmentar cada track en ventanas de **4 s** con **50% de overlap**.
- **Desviación del plan:** la sección 1.4 del plan describía segmentos de 2 s.
- **Contexto:** 2 s (~1 compás a 128 BPM) da timbre pero poco patrón rítmico, y el
  groove/kick/hats es clave para distinguir subgéneros de EDM. Se verificó que el
  `cnn_4` legacy en realidad entrenaba con ~3 s (los 30 s eran un paso
  intermedio). 4 s da más contexto y sigue en la familia short-chunk (el canónico
  de Won usa ~3.7 s). Con 50% de overlap sobran muestras (~94k estimadas; 147k
  reales).
- **Estado:** parámetro configurable en `config.py`; se afinará formalmente en el
  *hyperparameter tuning* (WBS 4.5).

### 4.2 Recorte de intro/outro
- **Decisión:** recortar ~15 s del inicio y del final antes de segmentar
  (configurable), con *fallback* que lo omite si el track es demasiado corto.
- **Contexto:** en EDM los intros/outros suelen ser kick pelado o silencio, poco
  representativos. Idea heredada del `create_segments` de `cnn_4` (que muestreaba
  ventanas del medio del track).

### 4.3 Terminología: "segmentación" vs "split"
- **Aclaración registrada:** conviven dos sentidos de "split". (a) *Segmentación*:
  cortar el track en chunks (ocurre dentro de `preprocess`). (b) *Train/val/test
  split*: repartir tracks en particiones (Req 3.4). Son cosas distintas; ambas
  están implementadas.

---

## 5. Pipeline de datos (orden WBS 2 → 3 → 4)

### 5.1 Flujo ordenado: manifest → validate → split → preprocess → train
- **Decisión:** una capa de datos explícita y ordenada, expuesta por CLI para que
  corra igual local y en Colab.
  - **Manifest** (`data/manifest.py`): etiqueta cada track + metadata (formato,
    sample rate, duración, bitrate estimado) → CSV. Incluye columnas
    `source_1`/`source_2` para la doble validación del Req 3.2.
  - **Validate** (`data/validate.py`): chequea 100/clase (Req 3.1), ≥128 kbps
    (Req 3.3), formatos y doble fuente (Req 3.2, como *warning* por defecto).
  - **Split** (`data/splits.py`): estratificado 70/15/15 **a nivel track**,
    persistido a `splits.json` (fijo y reproducible).
  - **Preprocess** (`data/preprocess.py`): precomputa los mel-spectrogramas a un
    caché en disco, una sola vez.
- **Contexto:** separar "de dónde leo" de "cómo entreno"; que el split quede fijo
  evita resultados no reproducibles (un problema de los notebooks legacy).

### 5.2 Split a nivel track (anti-leakage)
- **Decisión:** el split se hace sobre *tracks*, nunca sobre segmentos.
- **Contexto:** como cada track se expande en muchos segmentos, dividir por
  segmento filtraría pedazos de la misma canción entre train/val/test, inflando
  las métricas. Es la lección más valiosa portada del `cnn_4`.

---

## 6. Consumo del dataset y entorno de entrenamiento

### 6.1 Entrenar en la nube (Colab), no local
- **Decisión:** el entrenamiento real corre en **Google Colab (GPU)**; el código
  es *device-agnostic* (`cuda` → `mps` → `cpu`) para correr también en la Mac
  (MPS) o CPU en tests.
- **Contexto:** el dataset ya vive en Drive; el presupuesto del plan asume GPU
  cloud; la experimentación previa era en Colab; una GPU cloud es más rápida que
  el MPS de la Mac. El paquete se instala en Colab (`pip install`) y la notebook
  es solo un *launcher* (sin lógica de proyecto).

### 6.2 No descargar el audio crudo; cachear features
- **Decisión:** el audio crudo (~25-50 GB estimados) **permanece en Drive**; se
  preprocesa una sola vez a un **caché de features compacto** que es lo único que
  se reutiliza.
- **Contexto:** entrenar no usa el audio crudo sino mel-spectrogramas. El caché es
  mucho más chico y portable. Nunca se leen archivos de audio por red en cada
  época (el acceso aleatorio sobre Drive/FUSE es lento).
- **Detalle operativo:** durante el training el caché se **copia a disco local de
  Colab** para lecturas aleatorias rápidas; el original queda en Drive.

### 6.3 Instalación en Colab: preservar el torch con CUDA
- **Decisión:** instalar el paquete **no-editable** y con **`--no-deps`**,
  instalando el resto de las dependencias aparte.
- **Contexto:** (a) el install *editable* (`-e`) registra la ruta vía `.pth` que
  Python solo lee al arrancar el kernel → el módulo no era importable sin
  reiniciar. (b) instalar el paquete arrastraba `torch/torchaudio` y **pisaba** el
  build con CUDA que Colab ya trae, dejándolo roto. `--no-deps` preserva el torch
  de Colab.

### 6.4 Elección de GPU: L4
- **Decisión:** en Colab Pro, usar **L4** con RAM estándar.
- **Contexto:** para una CNN de este tamaño, L4 es el mejor costo/beneficio: mucho
  más rápida que la T4 y muy inferior en consumo de créditos a la A100 (que sería
  overkill). El training no es *RAM-bound* del sistema (memmap + VRAM), así que no
  hace falta High-RAM.

---

## 7. Features y modelo

### 7.1 Input del modelo: solo log-mel-spectrograma
- **Decisión:** la CNN entrena **únicamente sobre el log-mel-spectrograma**
  (128 mel bands, `n_fft=2048`, `hop=512`, escala en dB; shape (1, 128, 173) por
  segmento de 4 s).
- **Contexto:** es el estándar para Short-chunk CNN: la red aprende sus propias
  representaciones de la imagen tiempo-frecuencia, sin features hechas a mano.
- **Sobre el Req 1.7:** las demás features (MFCC, tempograma, spectral
  centroid/rolloff, ZCR, chroma) **están implementadas** (`features/extractors.py`)
  y cumplen el requisito como *capacidad del sistema*, pero alimentan los
  **baselines de sklearn**, no la CNN.
- **Alternativa futura:** arquitectura multi-rama (fusión tardía) que combine el
  mel con tempograma/features, como el `cnn_3` legacy tenía comentado y como
  propone Hsu, Chen & Yang (2021), arXiv:2110.08862 (referencia del plan). Se deja
  como experimento de tuning solo si las métricas se quedan cortas.

### 7.2 Arquitectura: Short-chunk CNN + ResNet
- **Decisión:** `ShortChunkCNNRes` en PyTorch: 7 bloques residuales (canales
  128→256→512), *global pooling* y cabezal MLP con dropout.
- **Contexto:** exigido por el plan; basado en Won et al. El *global pooling* hace
  la cabeza independiente del largo exacto de la entrada.

### 7.3 Baselines clásicos de sklearn
- **Decisión:** LogReg / SVM / RandomForest sobre un vector resumen (media+desvío)
  de las features, con el `StandardScaler` ajustado **solo en train**.
- **Contexto:** piso de comparación honesto para la CNN. Corrige el *data leakage*
  del notebook `ml_models` original (que escalaba sobre todo el dataset). No es
  requisito del plan, pero da material para la memoria técnica.

---

## 8. Entrenamiento (WBS 4.4)

### 8.1 Bucle de entrenamiento
- **Decisión:** `train_model` lee el caché + split persistido, entrena con
  validación por época, **early stopping** (patience) sobre val loss, guarda el
  **mejor checkpoint** y evalúa el test a **dos niveles**: por segmento y por track
  agregado. Logging por época en vivo.
- **Contexto:** device-agnostic; reproducible (semilla fija).

### 8.2 Data augmentation: SpecAugment
- **Decisión:** máscaras aleatorias de frecuencia y de tiempo sobre el
  mel-spectrograma durante el entrenamiento.
- **Contexto:** el plan pide *data augmentation*; con 560 tracks de train ayuda
  contra el overfitting.

### 8.3 Optimizaciones de velocidad
- **Decisión:** **mixed precision (AMP)** con `GradScaler` (solo CUDA),
  `cudnn.benchmark`, `pin_memory` + `non_blocking`, y `persistent_workers` en los
  DataLoaders.
- **Contexto:** el training inicial iba a ~4 min/época en L4. AMP en L4
  (tensor cores fp16) da ~1.5-2×. Se sube el batch a 128 y `num_workers` a 4.
  Opción extra: `n_channels=64` (~4× menos cómputo, poca pérdida de accuracy) para
  iterar rápido.

---

## 9. Detalles de implementación relevantes

### 9.1 Caché de features: streaming a disco (fix de OOM)
- **Decisión:** el `preprocess` escribe los segmentos **track por track** a un
  archivo raw float16 (`segments.f16`), leído luego por `memmap`. Reader
  *backward-compatible* con el formato `.npy` anterior.
- **Contexto:** la primera versión acumulaba todos los segmentos en RAM y hacía
  `np.concatenate` al final → pico de memoria ~2× → **OOM en Colab** con los 800
  tracks (el caché real es ~6.5 GB / 147.190 segmentos). El streaming baja el pico
  de RAM a un solo track.

### 9.2 Formato del caché
- `segments.f16` (raw float16, shape (N, 1, 128, 173)) + `labels.npy` +
  `track_ids.npy` + `index.json` (metadata + registro por track). `track_ids`
  permite tanto el split por track como la agregación de predicciones.

---

## 10. Métricas objetivo (del plan)

- Accuracy general **> 80%** (Req 1.3).
- Top-2 accuracy **> 90%** (Req 1.4).
- F1-score macro y por clase; matriz de confusión (Req 6.2, 6.3).
- Tiempo de procesamiento **< 5 s por track** (Req 1.5).
- La evaluación reporta si se cumplen los targets (`meets_targets`).

---

## 11. Flujo de trabajo del proyecto

- **Control de versiones:** el autor commitea y pushea; al cierre de cada *round*
  de trabajo se entrega un *commit message* listo para usar.
- **Calidad:** cada round cierra con `pytest` (cobertura) + `ruff` (PEP 8) en
  verde antes de dar por terminado.

---

## Desviaciones del plan a documentar en la memoria

| Tema | Plan original | Decisión tomada | Motivo |
|---|---|---|---|
| Tamaño del dataset | 200/género (1600) | 100/género (800) | Dataset disponible; la segmentación compensa |
| Largo de segmento | 2 s | 4 s | Más contexto rítmico para distinguir subgéneros |
| Features del modelo | lista amplia (Req 1.7) | solo mel en la CNN; el resto en baselines | Estándar de Short-chunk CNN; features implementadas como capacidad |
| Doble validación de fuente | Req 3.2 | columnas presentes, *warning* no bloqueante | Manifiesto armado desde archivos; se completa a mano si se requiere |
