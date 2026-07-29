# edm-classifier

Sistema de clasificación automática de subgéneros de música electrónica.
Trabajo Final — Carrera de Especialización en Inteligencia Artificial (FIUBA).

Clasifica tracks de música electrónica en 8 subgéneros mediante un modelo de
deep learning (Short-chunk CNN + ResNet), expuesto a través de una aplicación
de escritorio multiplataforma.

## Subgéneros

`deep house` · `tech house` · `melodic techno` · `progressive` ·
`techno peak time` · `hard techno` · `minimal/deep tech` · `trance`

## Arquitectura (monorepo)

Dos servicios independientes que se comunican vía API HTTP local:

| Servicio | Stack | Responsabilidad |
|----------|-------|-----------------|
| [`classifier/`](./classifier) | Python · PyTorch · Librosa · FastAPI | Extracción de features de audio, modelo de clasificación, API de inferencia. |
| [`ui/`](./ui) | Electron · React · TypeScript | App de escritorio: selección de directorio, progreso, organización automática de archivos. |

```
┌──────────────────────┐        HTTP (localhost)        ┌───────────────────────┐
│   ui (Electron+React) │  ───────────────────────────▶  │  classifier (FastAPI) │
│  selección · progreso │  ◀───────────────────────────  │  features · modelo    │
│  organización carpetas│         predicción + conf.     │  inferencia           │
└──────────────────────┘                                └───────────────────────┘
```

El clasificador funciona de forma autónoma por CLI, sin depender de la UI
(requerimiento de diseño: componentes independientes comunicados por API).

## Estructura

```
edm-classifier/
├── classifier/     # servicio Python (features + modelo + API)
├── ui/             # app de escritorio Electron + React
└── docs/           # plan de trabajo y documentación
```

## Documentación

- [Plan de trabajo](./docs/plan-de-trabajo.pdf)

## Desarrollo

Cada servicio tiene su propio README con instrucciones de setup:

- [`classifier/README.md`](./classifier/README.md)
- [`ui/README.md`](./ui/README.md)
