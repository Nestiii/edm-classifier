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
expuesto por el preload. La clasificación se delegará al servicio `classifier`
(FastAPI en `localhost`), manteniendo ambos módulos desacoplados.
