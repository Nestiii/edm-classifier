import { contextBridge, ipcRenderer } from 'electron'

/**
 * Safe, minimal bridge exposed to the renderer. The renderer never touches
 * Node or Electron internals directly — it only sees this typed surface.
 */
const api = {
  /** Opens a native directory picker; resolves to the chosen path or null. */
  selectDirectory: (): Promise<string | null> => ipcRenderer.invoke('dialog:selectDirectory'),

  /** Counts supported/ignored audio files directly in a directory (non-recursive). */
  countAudio: (dir: string): Promise<{ supported: number; ignored: number }> =>
    ipcRenderer.invoke('fs:countAudio', dir),

  /** Reveals a path in the OS file manager. */
  openPath: (path: string): Promise<void> => ipcRenderer.invoke('shell:openPath', path)
}

contextBridge.exposeInMainWorld('api', api)

export type Api = typeof api
