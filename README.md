# GUI Agent Demo — Edge Device GUI Automation

A local-first GUI agent built on **Qwen2.5-VL-7B** (via Ollama) and **Electron**, with a pluggable hybrid mode using **Claude (Anthropic API)** as a remote planner. Screenshots never leave the device in local modes.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Electron App                                        │
│  ┌─────────────────┐  ┌──────────────────────────┐   │
│  │  Control Panel  │  │  BrowserView (sandboxed) │   │
│  │  (renderer/)    │  │  — agent operates here   │   │
│  └────────┬────────┘  └──────────────────────────┘   │
│           │ IPC                    ↑                  │
│  ┌────────▼────────────────────────────────────────┐  │
│  │  Main Process  (src/main.js)                    │  │
│  │  Express bridge  :7788                          │  │
│  │  /screenshot  /dom  /action  /navigate          │  │
│  │  /current-url                                   │  │
│  └────────────────────────┬────────────────────────┘  │
└───────────────────────────│────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼────────────────────────────┐
│  Python Agent Server  :8000                            │
│                                                        │
│  server.py                                            │
│  ┌──────────────┐  ┌────────────────────────────────┐  │
│  │  Executor    │  │  VLM Client                    │  │
│  │  perceive →  │  │  ┌──────────────┐              │  │
│  │  plan →      │──▶  │ Qwen2.5-VL-7B│  (Ollama)    │  │
│  │  act loop    │  │  └──────────────┘              │  │
│  │              │  │  ┌──────────────┐              │  │
│  │  evaluator   │  │  │ Claude       │  (hybrid     │  │
│  │  confirms    │◀──  │ claude-opus  │   planner +  │  │
│  │  "done"      │  │  │ -4-5         │   evaluator) │  │
│  └──────────────┘  │  └──────────────┘              │  │
│                    └────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### Agent Modes

| Mode | Perception | Planner | Data leaves device? |
|------|-----------|---------|---------------------|
| `screenshot` | PNG only | Qwen2.5-VL-7B | ✗ Never |
| `dom` | PNG + DOM context | Qwen2.5-VL-7B | ✗ Never |
| `hybrid` | PNG + DOM context | Claude (claude-opus-4-5) | ⚠ Screenshots |

In `hybrid` mode, Claude is used both as the step-by-step planner and as the
completion evaluator (it confirms or rejects the model's "done" signal against
the current screenshot).

---

## Prerequisites

- **macOS** 12+ (tested on M2 16GB) or **Windows** 10+
- **Node.js** ≥ 18
- **Python** ≥ 3.11 (tested on 3.9 as well)
- **Ollama** — [ollama.com](https://ollama.com)
- 8 GB+ unified memory recommended (7B model peaks higher than 3B — see Known Limitations)
- (Optional, for hybrid mode) an **Anthropic API key**

---

## Installation

### 1. Pull the vision model

```bash
ollama pull qwen2.5vl:7b
```

### 2. Install Python dependencies

```bash
cd agent
pip install -r requirements.txt
```

`requirements.txt` now depends on `anthropic` (replacing the earlier
`google-genai` dependency) for hybrid mode.

### 3. Install Node dependencies

```bash
cd electron-app
npm install --registry https://registry.npmmirror.com
```

> If Electron download is slow, set the mirror first:
> ```bash
> export ELECTRON_MIRROR="https://npmmirror.com/mirrors/electron/"
> npm install
> ```

---

## Running the Agent

Start three components in separate terminals:

**Terminal 1 — Python agent server**
```bash
cd agent
OLLAMA_MODEL=qwen2.5vl:7b STEP_DELAY=0.5 uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

With Claude hybrid mode:
```bash
OLLAMA_MODEL=qwen2.5vl:7b ANTHROPIC_API_KEY=sk-ant-... STEP_DELAY=0.5 uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Electron app**
```bash
cd electron-app
npx electron .
```

**Terminal 3 — (optional) watch agent logs**
```bash
curl -N http://localhost:8000/status/stream
```

Alternatively, use the startup script to launch everything at once:
```bash
chmod +x start_dev.sh
./start_dev.sh

# With Claude hybrid mode
ANTHROPIC_API_KEY=sk-ant-... ./start_dev.sh
```

### Health check

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","ollama":true,"claude":false,"running":false}
```

### Verify Electron bridge

```bash
curl http://localhost:7788/screenshot | python3 -c "
import json,sys,base64
d=json.load(sys.stdin)
print('screenshot size:', d['width'], 'x', d['height'])
"
```

---

## Running the Benchmark

```bash
cd benchmark

# Screenshot mode only
python benchmark.py --modes screenshot

# Screenshot vs DOM comparison
python benchmark.py --modes screenshot dom

# With Claude hybrid mode
ANTHROPIC_API_KEY=sk-ant-... python benchmark.py --modes screenshot hybrid

# Results saved to benchmark/results/ as JSON + Markdown
```

### Task Suite

The current suite focuses on two multi-step information-extraction tasks that
exercise search → navigate → read flows:

| ID | Task | Category | Playwright? |
|----|------|----------|-------------|
| `huggingface_model_info` | Search HuggingFace for a model, open the first result, report its download count | multi_step | ✗ |
| `wikipedia_cross_page` | Search Wikipedia, open an article, report its first sentence | multi_step | ✓ |

### Benchmark Results (M2 16GB, Qwen2.5-VL-7B)

| Mode | Success Rate | Avg Steps | Avg Total Time | Avg Step Latency |
|------|-------------|-----------|----------------|-------------------|
| Screenshot | 100% (2/2) | 3 | 66.4s | 13.0s |
| Hybrid (Claude) | 100% (2/2) | 3.5 | 34.9s | 4.2s |
| DOM | 50% (1/2) | 8 | 135.8s | 12.8s |

**Screenshot mode is the most reliable local-only configuration** with the
7B model — both tasks complete in a small number of steps. **Hybrid mode is
both more reliable and faster per step** (Claude's lower latency more than
offsets the extra DOM-context payload). **DOM mode is the weakest mode** on
this task suite — see [`docs/dev_log.md`](docs/dev_log.md) for the failure
analysis and why DOM augmentation is not a good fit here.

---

## File Structure

```
gui-agent/
├── agent/
│   ├── vlm_client.py        # Qwen2.5-VL (Ollama) + Claude (Anthropic) VLM wrappers
│   ├── executor.py           # Perceive-plan-act loop + ElectronBridge
│   ├── server.py             # FastAPI server (screenshot/dom/hybrid modes)
│   └── requirements.txt      # Core dependencies
├── benchmark/
│   └── benchmark.py          # Task suite + runner + report generator
├── electron-app/
│   ├── src/
│   │   ├── main.js           # Main process: BrowserView, HTTP bridge, DOM extraction
│   │   └── preload.js        # contextBridge API for renderer
│   ├── renderer/
│   │   └── index.html        # Control panel UI
│   └── package.json
├── docs/
│   └── dev_log.md            # AI-assisted development log and benchmark analysis
├── start_dev.sh
└── README.md
```

---

## Key Design Decisions

**Why Qwen2.5-VL-7B?**
The 7B variant gives noticeably better instruction-following and reduces the
repetitive-loop failures seen with smaller models, at the cost of slower
per-step latency (~10–15s locally on M2).

**Why Claude for hybrid mode (instead of Gemini)?**
Gemini's free tier was unavailable in the development region. Claude is used
via the standard Anthropic Messages API for both step planning and as a
dedicated completion evaluator, and was substantially faster per step in
testing (~4s vs ~13s local).

**Why normalize coordinates to 0–1?**
Resolution-independent. The model sometimes outputs pixel coordinates and
sometimes normalized — the parser auto-detects and converts both.

**Why not use pyautogui for clicks?**
`pyautogui` operates on the full OS screen. Electron's `sendInputEvent` scopes
actions to the BrowserView sandbox, preventing accidental interaction with
anything outside.

**DOM context as prompt augmentation, not replacement**
A filtered element list (inputs, buttons, short nav links — with a small
blacklist for generic chrome links like "Pricing"/"Sign in"/"Donate") is
appended to the prompt alongside the screenshot. In practice, on this task
suite, the extra context did not improve outcomes — see Known Limitations.

**Evaluator-confirmed completion**
When the model signals `done`, a dedicated evaluator call (Claude in hybrid
mode, or the planner model itself otherwise) checks the screenshot against
the task description before accepting termination. This catches premature
"done" signals where the model is on the wrong page.

---

## Known Limitations

- **DOM mode underperforms on this task suite (50% vs 100%).** Two compounding
  causes:
  1. **Information overload at 7B scale.** Qwen2.5-VL-7B's instruction-following
     degrades when the prompt mixes a screenshot with a list of DOM element
     coordinates/labels — on the HuggingFace task the model repeatedly tried
     to `click` an input element instead of `type`-ing into it, something
     that did not happen in screenshot-only mode. Adding more structured
     context did not translate into better decisions for a model this size;
     if anything it added competing signals for the model to reconcile.
  2. **DOM extraction depth vs. latency trade-off.** The Wikipedia task
     requires locating a specific result link on a content-heavy search
     results page. The current DOM extraction caps the element list (to keep
     the prompt small and the 7B model's context manageable), and the target
     link can fall outside that cap. Raising the cap deep enough to
     reliably surface it would mean scanning and serializing a much larger
     DOM tree on every step — a meaningful latency cost on a mode that is
     already the slowest per-step (12.8s avg) and took the most steps overall
     (8 avg, vs 3–3.5 for the other modes). In this run, DOM mode got stuck
     repeatedly clicking the same coordinates `(0.21, 0.54)` on the search
     results page without ever resolving to the correct link.

  **Conclusion:** for this style of task (open-ended search-result navigation
  + content extraction), DOM augmentation is not a good fit at 7B — the
  screenshot-only and Claude-hybrid planners both solved both tasks in 3–5
  steps without it. DOM mode may still be worth revisiting for tasks
  dominated by structured form-filling, where the element list is short and
  directly actionable.

- **Memory pressure:** Processing 1280×800 screenshots with the 7B model uses
  more memory than the 3B variant; on constrained machines (8GB) this can be
  tight alongside Electron overhead.

- **Hybrid mode privacy:** In `hybrid` mode, screenshots are sent to the
  Anthropic API for both planning and evaluation — the control panel UI shows
  a prominent indicator. `screenshot` and `dom` modes remain fully local.

- **Transient UI states:** Tasks requiring rapid sequential interactions (e.g.
  dropdown menus that close during a multi-second inference window) remain
  challenging for local inference regardless of perception mode.

---

## Development Log

See [`docs/dev_log.md`](docs/dev_log.md) for AI-assisted development decisions,
prompt patterns, debugging notes, and the DOM-mode failure analysis underlying
the benchmark results above.
