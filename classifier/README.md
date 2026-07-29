# classifier

Servicio de clasificación (Python). Extrae features de audio, entrena/aloja el
modelo de deep learning (Short-chunk CNN + ResNet) y expone la inferencia por
API HTTP local y por CLI.

## Stack

- **PyTorch / torchaudio** — modelo y decodificación de audio.
- **Librosa** — extracción de features.
- **FastAPI + Uvicorn** — API HTTP local que consume la app de escritorio.
- **uv** — gestión de dependencias y entornos.
- **pytest + ruff** — testing y linting (PEP 8).

## Requisitos del sistema

Para decodificar MP3, `librosa`/`audioread` puede requerir **ffmpeg** instalado:

```bash
# macOS
brew install ffmpeg
```

## Setup

```bash
cd classifier
uv sync --extra dev        # crea .venv e instala dependencias (incl. dev)
```

## Uso

```bash
uv run edm-classifier --list-subgenres     # lista los 8 subgéneros
uv run edm-classifier --help
```

## Dataset

Layout esperado (una carpeta por subgénero, nombres filesystem-safe). Los audios
**no se versionan** (uso académico, sin distribución — legal 8.1):

```
classifier/data/raw/
├── deep_house/*.{mp3,aiff,wav}
├── tech_house/...
├── melodic_techno/...
├── progressive/...
├── techno_peak_time/...
├── hard_techno/...
├── minimal_deep_tech/...
└── trance/...
```

Flujo de datos ordenado (CLI, corre igual local o en Colab):

```bash
# 1. Etiquetado + metadata → manifiesto (WBS 2.4)
uv run edm-classifier manifest --root data/raw --out data/manifest.csv
# 2. Validación: 100/clase, >=128 kbps, formatos, doble fuente (WBS 2.3)
uv run edm-classifier validate --manifest data/manifest.csv
# 3. Split train/val/test 70/15/15 por track, persistido (WBS 4.3 / Req 3.4)
uv run edm-classifier split --manifest data/manifest.csv --out data/splits.json
# 4. Preprocesamiento: caché de mel-spectrogramas a disco, una sola vez (WBS 4.3)
uv run edm-classifier preprocess --manifest data/manifest.csv --cache data/cache
```

El caché (`segments.npy` float16 + `labels`/`track_ids` + `index.json`) es la
entrada del training; el audio crudo no se vuelve a leer. Segmentación: ventanas
de **4 s** con 50% de solapamiento, recortando ~15 s de intro/outro (configurable
en `config.py`, tuneable en WBS 4.5).

### Entrenar en la nube (Colab)

`colab/edm_classifier_colab.ipynb` es un launcher: clona el repo, instala el
paquete, monta Drive y corre el mismo flujo sobre GPU. El audio crudo se queda en
Drive; solo se preprocesa una vez y el caché (~1 GB) se guarda en Drive.

## Tests

```bash
uv run pytest              # con cobertura (objetivo >=70% en extracción)
uv run ruff check .        # lint PEP 8
```

## Estructura

```
src/edm_classifier/
├── config.py       # subgéneros + parámetros de audio/features/dataset
├── cli.py          # entry point de línea de comandos
├── features/       # extracción con Librosa
├── data/           # construcción del dataset y splits
├── models/         # Short-chunk CNN + ResNet
├── training/       # entrenamiento, tuning, evaluación
├── inference/      # predicción + nivel de confianza
└── api/            # servidor FastAPI
```
