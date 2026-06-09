/**
 * Electron Main Process
 *
 * Creates:
 *   - A control window (renderer/index.html) showing the agent panel
 *   - A BrowserView (sandboxed) where the agent operates
 *
 * Exposes an Express HTTP server on localhost:7788 for the Python agent:
 *   GET  /screenshot   → PNG base64 of the BrowserView
 *   GET  /dom          → list of interactable DOM elements
 *   GET  /current-url  → current URL of the BrowserView
 *   POST /action       → execute click/type/scroll/key/navigate
 *   POST /navigate     → load a URL in the BrowserView
 *   POST /status       → push step update to renderer via IPC
 */

const { app, BrowserWindow, BrowserView, ipcMain, screen } = require('electron');
const path    = require('path');
const http    = require('http');
const express = require('express');

// ── Config ────────────────────────────────────────────────────────────────────

const BRIDGE_PORT    = 7788;
const SANDBOX_WIDTH  = 1280;
const SANDBOX_HEIGHT = 800;
const CONTROL_WIDTH  = 520;

// ── State ─────────────────────────────────────────────────────────────────────

let mainWindow  = null;
let sandboxView = null;


// ── Window setup ──────────────────────────────────────────────────────────────

function createWindows() {
  const { height: sh } = screen.getPrimaryDisplay().workAreaSize;

  mainWindow = new BrowserWindow({
    width:  CONTROL_WIDTH + SANDBOX_WIDTH,
    height: sh,
    x: 0,
    y: 0,
    title: 'GUI Agent',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  mainWindow.setMenuBarVisibility(false);

  sandboxView = new BrowserView({
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.setBrowserView(sandboxView);
  sandboxView.setBounds({
    x: CONTROL_WIDTH,
    y: 0,
    width:  SANDBOX_WIDTH,
    height: SANDBOX_HEIGHT,
  });
  sandboxView.setAutoResize({ width: false, height: false });
  sandboxView.webContents.loadURL('about:blank');

  mainWindow.on('closed', () => { mainWindow = null; });
}


// ── IPC handlers (renderer → main) ───────────────────────────────────────────

ipcMain.handle('navigate', async (_e, url) => {
  if (!sandboxView) return { ok: false };
  await sandboxView.webContents.loadURL(url);
  return { ok: true };
});

ipcMain.handle('get-url', () => {
  return sandboxView?.webContents.getURL() || '';
});


// ── Screenshot helper ─────────────────────────────────────────────────────────

async function captureScreenshot() {
  if (!sandboxView) throw new Error('No sandbox view');
  const image = await sandboxView.webContents.capturePage();
  return {
    buffer: image.toPNG(),
    width:  image.getSize().width,
    height: image.getSize().height,
  };
}


// ── DOM extraction helper ─────────────────────────────────────────────────────

const DOM_SCRIPT = `
(function() {
  const TAGS = ['a', 'button', 'input', 'select', 'textarea', 'label', '[role="button"]',
                '[role="link"]', '[role="tab"]', '[role="menuitem"]', '[tabindex]'];
  const els = Array.from(document.querySelectorAll(TAGS.join(',')));
  return els.slice(0, 60).map(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return null;
    return {
      tag:  el.tagName.toLowerCase(),
      id:   el.id || null,
      text: (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().slice(0, 80),
      href: el.href || null,
      rect: { top: Math.round(r.top), left: Math.round(r.left), width: Math.round(r.width), height: Math.round(r.height) }
    };
  }).filter(Boolean);
})();
`;

async function getDOMElements() {
  if (!sandboxView) return [];
  try {
    return await sandboxView.webContents.executeJavaScript(DOM_SCRIPT);
  } catch (e) {
    console.error('DOM extraction error:', e.message);
    return [];
  }
}


// ── Action execution ──────────────────────────────────────────────────────────

const INPUT_SCRIPT = (text) => `
(function() {
  const el = document.activeElement;
  if (!el) return false;
  const nativeInput = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
  if (nativeInput) {
    nativeInput.set.call(el, ${JSON.stringify(text)});
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  } else {
    el.value = ${JSON.stringify(text)};
  }
  return true;
})();
`;

// React-compatible value setter — clears then sets value, dispatches input+change
const CLEAR_AND_SET_SCRIPT = (text) => `
(function() {
  const el = document.activeElement;
  if (!el) return false;
  const nativeSetter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value'
  ) || Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype, 'value'
  );
  if (nativeSetter && nativeSetter.set) {
    nativeSetter.set.call(el, '');
    el.dispatchEvent(new Event('input', { bubbles: true }));
    nativeSetter.set.call(el, ${JSON.stringify(text)});
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  } else {
    el.value = ${JSON.stringify(text)};
  }
  return true;
})();
`;

// JS-level click — works for React/SPA links that don't respond to sendInputEvent
const JS_CLICK_SCRIPT = (px, py) => `
(function() {
  const el = document.elementFromPoint(${px}, ${py});
  if (!el) return false;
  let target = el;
  for (let i = 0; i < 5; i++) {
    if (!target) break;
    const tag = target.tagName && target.tagName.toLowerCase();
    if (tag === 'a' || tag === 'button' || target.onclick ||
        target.getAttribute('role') === 'button' ||
        target.getAttribute('role') === 'link') {
      break;
    }
    target = target.parentElement;
  }
  if (!target) target = el;
  target.focus();
  target.click();
  ['pointerdown','mousedown','pointerup','mouseup','click'].forEach(evt => {
    target.dispatchEvent(new MouseEvent(evt, {
      bubbles: true, cancelable: true, view: window,
      clientX: ${px}, clientY: ${py}
    }));
  });
  return target.tagName || true;
})();
`;

async function executeAction(action) {
  if (!sandboxView) throw new Error('No sandbox view');
  const wc     = sandboxView.webContents;
  const bounds = sandboxView.getBounds();
  const W = bounds.width;
  const H = bounds.height;

  switch (action.type) {
    case 'click': {
      const px = Math.round((action.x ?? 0.5) * W);
      const py = Math.round((action.y ?? 0.5) * H);

      const href = await wc.executeJavaScript(`
        (function() {
          const el = document.elementFromPoint(${px}, ${py});
          if (!el) return null;
          let target = el;
          for (let i = 0; i < 5; i++) {
            if (!target) break;
            if (target.tagName && target.tagName.toLowerCase() === 'a' && target.href) {
              return target.href;
            }
            target = target.parentElement;
          }
          return null;
        })();
      `).catch(() => null);

      if (href && href.startsWith('http')) {
        console.log(`[action] navigating via loadURL: ${href}`);
        await wc.loadURL(href);
      } else {
        const jsResult = await wc.executeJavaScript(JS_CLICK_SCRIPT(px, py)).catch(() => null);
        console.log(`[action] click (${px},${py}) js=${jsResult}`);
        await new Promise(r => setTimeout(r, 100));
        wc.sendInputEvent({ type: 'mouseMove', x: px, y: py });
        wc.sendInputEvent({ type: 'mouseDown', button: 'left', x: px, y: py, clickCount: 1 });
        wc.sendInputEvent({ type: 'mouseUp',   button: 'left', x: px, y: py, clickCount: 1 });
      }
      break;
    }
    case 'type': {
      if (action.text) {
        const px = action.x != null ? Math.round(action.x * W) : null;
        const py = action.y != null ? Math.round(action.y * H) : null;

        if (px != null && py != null) {
          await wc.executeJavaScript(`
            (function() {
              const el = document.elementFromPoint(${px}, ${py});
              if (el) { el.focus(); }
            })();
          `).catch(() => {});
          await new Promise(r => setTimeout(r, 100));
          wc.sendInputEvent({ type: 'mouseDown', button: 'left', x: px, y: py, clickCount: 1 });
          wc.sendInputEvent({ type: 'mouseUp',   button: 'left', x: px, y: py, clickCount: 1 });
          await new Promise(r => setTimeout(r, 150));
        }

        // Clear existing content
        wc.sendInputEvent({ type: 'keyDown', keyCode: 'a', modifiers: ['meta'] });
        wc.sendInputEvent({ type: 'keyUp',   keyCode: 'a', modifiers: ['meta'] });
        await new Promise(r => setTimeout(r, 50));
        wc.sendInputEvent({ type: 'keyDown', keyCode: 'Backspace' });
        wc.sendInputEvent({ type: 'keyUp',   keyCode: 'Backspace' });
        await new Promise(r => setTimeout(r, 50));

        // Set value via React-compatible setter ONLY — no char events
        await wc.executeJavaScript(CLEAR_AND_SET_SCRIPT(action.text)).catch(() => {});

        // Also send char events — required for sites that listen to keydown (Wikipedia)
        for (const char of action.text) {
          wc.sendInputEvent({ type: 'char', keyCode: char });
          await new Promise(r => setTimeout(r, 10));
        }
      }
      break;
    }
    case 'scroll': {
      const px = Math.round((action.x ?? 0.5) * W);
      const py = Math.round((action.y ?? 0.5) * H);
      const delta = (action.direction === 'down' ? 1 : -1) * (action.amount ?? 300);
      wc.sendInputEvent({ type: 'mouseWheel', x: px, y: py, deltaX: 0, deltaY: delta });
      break;
    }
    case 'navigate': {
      if (action.url) await wc.loadURL(action.url);
      break;
    }
    case 'key': {
      if (action.key) {
        const KEY_MAP = {
          'Enter': 'Return', 'Tab': 'Tab', 'Escape': 'Escape',
        };
        const key = action.key;
        if (key.toLowerCase() === 'enter') {
          wc.sendInputEvent({ type: 'keyDown', keyCode: 'Return' });
          wc.sendInputEvent({ type: 'keyUp',   keyCode: 'Return' });
        } else if (key.includes('+')) {
          const parts    = key.split('+');
          const modifiers = parts.slice(0, -1).map(m => m.toLowerCase());
          const mainKey  = parts[parts.length - 1];
          wc.sendInputEvent({ type: 'keyDown', keyCode: mainKey, modifiers });
          wc.sendInputEvent({ type: 'keyUp',   keyCode: mainKey, modifiers });
        } else {
          const kc = KEY_MAP[key] || key;
          wc.sendInputEvent({ type: 'keyDown', keyCode: kc });
          wc.sendInputEvent({ type: 'keyUp',   keyCode: kc });
        }
      }
      break;
    }
    default:
      break;
  }

  await new Promise(r => setTimeout(r, 300));
  return { ok: true };
}


// ── Express HTTP Bridge ───────────────────────────────────────────────────────

function startBridge() {
  const expressApp = express();
  expressApp.use(express.json({ limit: '10mb' }));

  expressApp.get('/screenshot', async (req, res) => {
    try {
      const { buffer, width, height } = await captureScreenshot();
      res.json({ image: buffer.toString('base64'), width, height });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  expressApp.get('/dom', async (req, res) => {
    try {
      const elements = await getDOMElements();
      res.json({ elements });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── NEW: current URL endpoint ─────────────────────────────────────────────
  expressApp.get('/current-url', (req, res) => {
    try {
      const url = sandboxView?.webContents.getURL() || '';
      res.json({ url });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  expressApp.post('/action', async (req, res) => {
    try {
      const result = await executeAction(req.body);
      res.json(result);
    } catch (e) {
      console.error('[action] error:', e.message, e.stack);
      res.status(500).json({ error: e.message });
    }
  });

  expressApp.post('/navigate', async (req, res) => {
    try {
      const { url } = req.body;
      await sandboxView.webContents.loadURL(url);
      res.json({ ok: true });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  expressApp.post('/status', (req, res) => {
    if (mainWindow) {
      mainWindow.webContents.send('agent-status', req.body);
    }
    res.json({ ok: true });
  });

  const server = http.createServer(expressApp);
  server.listen(BRIDGE_PORT, '127.0.0.1', () => {
    console.log(`Bridge listening on http://127.0.0.1:${BRIDGE_PORT}`);
  });
  return server;
}


// ── App lifecycle ─────────────────────────────────────────────────────────────

app.whenReady().then(() => {
  createWindows();
  startBridge();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindows();
});