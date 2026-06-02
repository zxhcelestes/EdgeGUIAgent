/**
 * Preload script — exposes a minimal, typed API to the renderer process.
 * contextIsolation: true means nothing from Node/Electron leaks into renderer.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Navigate the sandboxed BrowserView
  navigate: (url) => ipcRenderer.invoke('navigate', url),
  getURL:   ()    => ipcRenderer.invoke('get-url'),

  // Receive live agent status updates from main process
  onAgentStatus: (callback) => {
    ipcRenderer.on('agent-status', (_event, data) => callback(data));
  },

  // Remove listener (cleanup)
  removeAgentStatusListener: () => {
    ipcRenderer.removeAllListeners('agent-status');
  },
});
