# ui

Aplicación de escritorio (Electron + React + TypeScript). Permite seleccionar
una carpeta de tracks, dispara la clasificación en el servicio `classifier` vía
API HTTP local, muestra el progreso y organiza los archivos en subcarpetas por
subgénero.

## Stack

- **Electron** — shell de escritorio multiplataforma (Windows/macOS/Linux).
- **React + TypeScript** — interfaz de usuario.
- **electron-vite** — build y dev server.
- **electron-builder** — empaquetado/distribución.

## Setup

```bash
cd ui
npm install
```

## Desarrollo

```bash
npm run dev          # levanta Electron con hot-reload
npm run typecheck    # chequeo de tipos
npm run lint         # ESLint
```

## Build / empaquetado

```bash
npm run build        # compila main + preload + renderer
npm run package      # genera app sin instalador (dir)
npm run dist         # genera instalador con electron-builder
```

## Estructura

```
src/
├── main/       # proceso principal de Electron (ventana, IPC, diálogos)
├── preload/    # puente seguro (contextBridge) hacia el renderer
└── renderer/   # app React
    └── src/    # componentes, estilos
```

## Comunicación con el clasificador

El renderer nunca accede a Node/Electron directamente: usa el objeto `window.api`
expuesto por el preload (selección de carpeta, conteo de audio, abrir carpeta).
La clasificación se delega al servicio `classifier` (FastAPI en
`http://127.0.0.1:8000`) vía `fetch`, manteniendo ambos módulos desacoplados.

Para probar la app end-to-end, primero levantá la API:

```bash
# en classifier/
uv run edm-classifier serve --model path/to/model.pt --port 8000
# en ui/
npm run dev
```

## Pantallas (flujo)

Máquina de estados en `src/renderer/src/App.tsx`:

1. **Startup** — poll `GET /health` hasta que el modelo esté cargado; si no
   responde, muestra el error con "Reintentar".
2. **SelectFolder** — elige carpeta, cuenta audio soportado/ignorado, o el estado
   vacío "sin audio compatible".
3. **Progress** — `POST /jobs` y polling de `GET /jobs/{id}`: progreso, ETA,
   últimos resultados con subgénero + confianza (y 2ª opción si es baja).
4. **Report** — distribución por subcarpeta, confianza media, tiempo total,
   "Abrir carpeta". Los tracks de baja confianza van a `/Revisar`.
