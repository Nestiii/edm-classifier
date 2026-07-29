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
