import { join, extname } from 'path'
import { readdirSync, statSync } from 'fs'
import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'

// Audio formats the classifier supports (mirrors config.SUPPORTED_EXTENSIONS).
const SUPPORTED_EXTENSIONS = new Set(['.mp3', '.aiff', '.aif', '.wav'])

function createWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1000,
    height: 720,
    show: false,
    autoHideMenuBar: true,
    title: 'EDM Classifier',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true
    }
  })

  mainWindow.on('ready-to-show', () => mainWindow.show())

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // In development, load the Vite dev server; in production, the built HTML.
  if (process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// IPC: let the renderer open a native directory picker (Req 2.1).
ipcMain.handle('dialog:selectDirectory', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory']
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

// IPC: count supported vs ignored audio files directly in a directory.
ipcMain.handle('fs:countAudio', (_event, dir: string) => {
  let supported = 0
  let ignored = 0
  try {
    for (const name of readdirSync(dir)) {
      const full = join(dir, name)
      if (!statSync(full).isFile()) continue
      if (SUPPORTED_EXTENSIONS.has(extname(name).toLowerCase())) supported++
      else ignored++
    }
  } catch {
    return { supported: 0, ignored: 0 }
  }
  return { supported, ignored }
})

// IPC: reveal a path in the OS file manager (Req 2.5 "Abrir carpeta").
ipcMain.handle('shell:openPath', (_event, path: string) => shell.openPath(path))

app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
