import { contextBridge, ipcRenderer } from 'electron'

/**
 * Safe, minimal bridge exposed to the renderer. The renderer never touches
 * Node or Electron internals directly — it only sees this typed surface.
 */
const api = {
  /** Opens a native directory picker; resolves to the chosen path or null. */
  selectDirectory: (): Promise<string | null> => ipcRenderer.invoke('dialog:selectDirectory')
}

contextBridge.exposeInMainWorld('api', api)

export type Api = typeof api
