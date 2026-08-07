# Manual de usuario — Subgenre Sorter

Aplicación de escritorio para la clasificación automática de subgéneros de música
electrónica. Este manual cubre los requisitos del sistema, la instalación, el uso
paso a paso, la interpretación de los resultados y la resolución de problemas
comunes.

---

## 1. Requisitos del sistema

- **Sistemas operativos:** Windows 10/11, macOS 12+ o Linux (Ubuntu 22.04+).
- **Formatos de audio soportados:** MP3, AIFF, WAV (mínimo 128 kbps).
- **Hardware:** funciona en CPU; una GPU dedicada acelera el análisis pero no es
  obligatoria (ver §5, rendimiento).
- **Servicio de análisis:** la aplicación necesita el módulo clasificador
  ejecutándose localmente (ver §2.2).

---

## 2. Instalación

La solución tiene **dos componentes** que se comunican por una API local:

1. **La aplicación de escritorio** (interfaz gráfica).
2. **El servicio clasificador** (módulo de análisis en Python).

### 2.1 Aplicación de escritorio

Descargar e instalar el paquete correspondiente al sistema operativo:

| Sistema | Archivo |
|---------|---------|
| macOS | `Subgenre Sorter-<versión>.dmg` |
| Windows | `Subgenre Sorter Setup <versión>.exe` |
| Linux | `Subgenre Sorter-<versión>.AppImage` o `.deb` |

En macOS: abrir el `.dmg` y arrastrar la app a *Aplicaciones*. En Windows:
ejecutar el instalador y seguir los pasos. En Linux: dar permisos de ejecución al
`.AppImage` o instalar el `.deb`.

### 2.2 Servicio clasificador

El servicio se levanta desde una terminal, indicando la ruta al modelo entrenado:

```bash
edm-classifier serve --model ruta/al/model.pt --port 8000
```

Al iniciarse muestra `Loaded model from ...` y queda escuchando en
`http://127.0.0.1:8000`. La aplicación de escritorio se conecta automáticamente.

---

## 3. Uso paso a paso

### Inicio
Al abrir la app aparece **"Getting ready"** mientras se conecta con el servicio y
se carga el modelo. Indica también el dispositivo de cómputo (por ejemplo,
"Running on MPS"). Si no logra conectarse, muestra **"Couldn't reach the
classifier"** con un botón **Retry** (ver §6).

### Paso 1 — Elegir carpeta
En **"Sort your library by subgenre"**:
1. Presionar **Choose folder…** y seleccionar el directorio con los tracks.
2. La app muestra cuántos archivos de audio se detectaron y cuántos se ignoraron
   por formato no soportado.
3. Opcionalmente ajustar:
   - **Include subfolders:** analizar también los subdirectorios.
   - **Review threshold:** confianza mínima para aceptar una clasificación. Los
     tracks por debajo de este umbral se apartan en una carpeta *Review* (ver §4).
4. Presionar **Analyze N tracks**.

### Paso 2 — Vista previa
En **"Here's how your library would look"** se muestra el resultado **sin haber
movido ningún archivo todavía**: cuántos irían a subcarpetas, cuántos a revisión,
la confianza media y la lista completa agrupada por subgénero.

- **Organize:** confirma y organiza los archivos.
- **Discard:** vuelve al inicio sin tocar nada.

### Paso 3 — Organización y reporte
Durante la organización se muestra el progreso (archivos procesados, tiempo
restante estimado). Al terminar, el reporte **"N tracks organized"** muestra:
- Confianza media y tiempo total.
- Distribución por subcarpeta creada.
- La cantidad de tracks enviados a *Review* y de archivos que no se pudieron leer.
- **Open folder:** abre la carpeta organizada en el explorador de archivos.
- **Sort another folder:** procesa otra carpeta.

---

## 4. Interpretación de resultados

- **Subgénero y confianza:** cada track se asigna a uno de los 8 subgéneros
  (*deep house, tech house, melodic techno, progressive, techno peak time, hard
  techno, minimal/deep tech, trance*) con un porcentaje de confianza.
- **Segunda opción (top-2):** para los tracks de baja confianza se muestra también
  el segundo subgénero más probable.
- **Carpeta *Review*:** los tracks cuya confianza no supera el umbral configurado
  se colocan aquí en lugar de en una subcarpeta de subgénero, para revisión manual.
  Esto es útil porque ciertos subgéneros comparten características y pueden ser
  ambiguos incluso para el oído humano.
- **Organización en disco:** se crea una subcarpeta por subgénero (por ejemplo
  `deep_house/`, `tech_house/`) y los archivos originales se **mueven** allí. La
  aplicación no conserva copias del audio.

---

## 5. Rendimiento

El tiempo de procesamiento por track (cargar audio, extraer características y
clasificar) se mantiene muy por debajo del objetivo de 5 segundos:

| Hardware | Tiempo medio por track |
|----------|------------------------|
| GPU (Apple MPS) | ~0.25 s |
| CPU | ~1.0 s |

*(Medido sobre tracks de ~4 minutos.)*

---

## 6. Resolución de problemas

**"Couldn't reach the classifier"**
La app no encontró el servicio de análisis. Verificar que el comando
`edm-classifier serve …` esté en ejecución y volver a intentar con **Retry**.

**"No supported audio in this folder"**
La carpeta seleccionada no contiene MP3, AIFF ni WAV. Elegir otra carpeta. Si los
archivos están en subcarpetas, activar **Include subfolders**.

**"We couldn't finish"**
Ocurrió un error durante el análisis (por ejemplo, un archivo corrupto). El
mensaje describe la causa. Presionar **Back** y reintentar.

**Muchos tracks van a *Review***
El umbral de revisión es demasiado alto. Bajarlo desde la pantalla de selección.

**macOS bloquea la aplicación ("no se puede abrir")**
Si la app no está firmada/notarizada, macOS puede bloquearla la primera vez.
Abrirla con clic derecho → *Abrir*, o autorizarla desde *Ajustes → Privacidad y
seguridad*.
