# GUI Agent Demo — Edge Device GUI Automation

A local-first GUI agent built on **Qwen2.5-VL-3B** (via Ollama) and **Electron**, with a pluggable hybrid mode using Gemini Flash as a remote planner. Screenshots never leave the device in local mode.

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
│  server.py (standard)   server_g2.py (planner+grounder)│
│  ┌──────────────┐  ┌────────────────────────────────┐  │
│  │  Executor    │  │  VLM Client                    │  │
│  │  perceive →  │  │  ┌──────────────┐              │  │
│  │  plan →      │──▶  │ Qwen2.5-VL-3B│  (Ollama)    │  │
│  │  act loop    │  │  └──────────────┘              │  │
│  └──────────────┘  │  ┌──────────────┐              │  │
│                    │  │ Gemini Flash │  (hybrid)    │  │
│                    │  └──────────────┘              │  │
│                    │  ┌──────────────┐              │  │
│                    │  │ GUI-G2-3B    │  (grounder,  │  │
│                    │  │ (CUDA only)  │   optional)  │  │
│                    │  └──────────────┘              │  │
│                    └────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### Agent Modes

| Mode | Perception | Planner | Grounder | Data leaves device? |
|------|-----------|---------|----------|---------------------|
| `screenshot` | PNG only | Qwen2.5-VL-3B | — | ✗ Never |
| `dom` | PNG + DOM context | Qwen2.5-VL-3B | — | ✗ Never |
| `hybrid` | PNG + DOM context | Gemini Flash | — | ⚠ Screenshots |

### Server Variants

| Server | Description |
|--------|-------------|
| `server.py` | Standard server, screenshot/dom/hybrid modes |
| `server_g2.py` | Planner + GUI-G2 grounder (CUDA required) |

---

## Prerequisites

- **macOS** 12+ (tested on M2 16GB) or **Windows** 10+
- **Node.js** ≥ 18
- **Python** ≥ 3.11
- **Ollama** — [ollama.com](https://ollama.com)
- 8 GB+ unified memory recommended

---

## Installation

### 1. Pull the vision model

```bash
ollama pull qwen2.5vl:3b
```

### 2. Install Python dependencies

```bash
cd agent
pip install -r requirements.txt
```

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
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

With Gemini hybrid mode:
```bash
GEMINI_API_KEY=AIza... uvicorn server:app --host 127.0.0.1 --port 8000 --reload
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

# With Gemini hybrid mode
GEMINI_API_KEY=AIza... ./start_dev.sh
```

### Health check

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","ollama":true,"gemini":false,"running":false}
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

# Screenshot vs DOM comparison (recommended)
python benchmark.py --modes screenshot dom

# Run specific tasks
python benchmark.py --modes screenshot dom --tasks form_search nav_github

# With Gemini hybrid mode
GEMINI_API_KEY=AIza... python benchmark.py --modes screenshot dom hybrid

# Results saved to benchmark/results/ as JSON + Markdown
```

> **Note:** Each task takes 100–600s depending on model speed. Running all 8 tasks across 2 modes takes approximately 2–3 hours on M2.

### Task Suite

| ID | Task | Category | Playwright? |
|----|------|----------|-------------|
| `form_contact` | Fill contact form and submit | Form fill | ✓ |
| `form_search` | Search on DuckDuckGo | Form fill | ✓ |
| `nav_github` | Navigate to GitHub Trending | Navigation | ✓ |
| `nav_hacker_news` | Open top HN story | Navigation | ✓ |
| `extract_title` | Extract Wikipedia article title | Extraction | ✓ |
| `multi_github_search` | GitHub repo search + inspect | Multi-step | ✗ |
| `multi_scroll_load` | Count HN front page stories | Multi-step | ✓ |
| `canvas_excalidraw` | Draw shape in Excalidraw | Multi-step | ✗ |

### Benchmark Results (M2 16GB, Qwen2.5-VL-3B)

| Mode | Success Rate | Avg Steps | Avg Step Latency |
|------|-------------|-----------|-----------------|
| Screenshot | 38% (3/8) | 7.9 | 33.5s |
| DOM | 50% (4/8) | 7.6 | 41.6s |

See [`docs/benchmark_report.md`](docs/benchmark_report.md) for full analysis.

---

## GUI-G2 Planner+Grounder Mode (Experimental)

An experimental two-stage architecture is implemented in `server_g2.py`:

1. **Planner** (Qwen2.5-VL-3B via Ollama) — decides action type and describes the target element in natural language
2. **Grounder** (GUI-G2-3B, AAAI 2026) — maps the description to precise coordinates using Gaussian reward-trained grounding

### Setup (CUDA required)

```bash
# Install additional dependencies
pip install -r agent/requirements-g2.txt

# Download GUI-G2-3B (~6GB)
cd agent
huggingface-cli download inclusionAI/GUI-G2-3B --local-dir ./models/GUI-G2-3B

# Start with grounder server
uvicorn server_g2:app --host 127.0.0.1 --port 8000 --reload
```

### Hardware requirement

GUI-G2 requires **CUDA (NVIDIA GPU)**. It is not functional on macOS MPS (bfloat16 unsupported) or CPU (inference >20min per call). On M2, the system automatically falls back to planner-only mode.

---

## File Structure

```
gui-agent/
├── agent/
│   ├── vlm_client.py        # Qwen2.5-VL + Gemini VLM wrappers
│   ├── vlm_client_g2.py     # Planner+Grounder VLM client
│   ├── gui_g2_client.py     # GUI-G2 grounder client
│   ├── executor.py          # Perceive-plan-act loop + ElectronBridge
│   ├── server.py            # Standard FastAPI server
│   ├── server_g2.py         # Planner+Grounder server variant
│   ├── requirements.txt     # Core dependencies
│   └── requirements-g2.txt  # GUI-G2 additional dependencies
├── benchmark/
│   └── benchmark.py         # Task suite + runner + report generator
├── electron-app/
│   ├── src/
│   │   ├── main.js          # Main process: BrowserView, HTTP bridge
│   │   └── preload.js       # contextBridge API for renderer
│   ├── renderer/
│   │   └── index.html       # Control panel UI
│   └── package.json
├── docs/
│   ├── benchmark_report.md  # Full benchmark results and analysis
│   └── dev_log.md           # AI-assisted development log
├── start_dev.sh
└── README.md
```

---

## Key Design Decisions

**Why Qwen2.5-VL-3B instead of UI-TARS?**
UI-TARS-7B failed to run reliably on macOS via Ollama due to model runner crashes under memory pressure. Qwen2.5-VL-3B was selected as a stable alternative with good GUI understanding at lower resource requirements.

**Why normalize coordinates to 0–1?**
Resolution-independent. The model sometimes outputs pixel coordinates and sometimes normalized — the parser auto-detects and converts both.

**Why not use pyautogui for clicks?**
`pyautogui` operates on the full OS screen. Electron's `sendInputEvent` scopes actions to the BrowserView sandbox, preventing accidental interaction with anything outside.

**DOM as prompt augmentation, not replacement**
We append a filtered element list (inputs, buttons, short nav links) to the VLM prompt alongside the screenshot. The model retains visual context while gaining precise anchor points. Long search-result links are excluded to avoid token overflow on 3B models.

**URL-based success detection**
The executor checks the current URL after each step against task-specific heuristics. This reduces reliance on the model's self-termination judgment, which is unreliable at 3B scale. Limitation: measures structural navigation, not semantic task completion.

**Privacy architecture**
In `screenshot` and `dom` modes, all inference is local via Ollama. In `hybrid` mode, screenshots are sent to Gemini — the control panel UI shows a prominent indicator.

---

## Known Limitations

- **Transient UI states:** Dropdown menus close during the 30–110s inference window. Tasks requiring rapid sequential interactions consistently fail in local mode.
- **Memory pressure:** Processing 1280×800 screenshots peaks at ~5–6GB, occasionally crashing the Ollama model runner on M2 16GB.
- **GUI-G2 on macOS:** bfloat16 unsupported on MPS; CPU inference too slow (~20min/call). CUDA required.
- **Gemini regional quota:** Free tier unavailable in mainland China.
- **vLLM macOS incompatible:** Linux + NVIDIA only. Ollama with Metal backend is the only local inference option on Apple Silicon.

---

## Development Log

See [`docs/dev_log.md`](docs/dev_log.md) for AI-assisted development decisions, prompt patterns, debugging notes, and trade-off observations.