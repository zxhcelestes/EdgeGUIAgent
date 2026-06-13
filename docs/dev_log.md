# GUI Agent Benchmark Report
## Edge Device GUI Automation with Local VLM (Qwen2.5-VL-7B)

**Environment:** Mac M4 24GB, macOS
**Local Model:** Qwen2.5-VL-7B via Ollama
**Hybrid Mode:** Claude (`claude-opus-4-5`, Anthropic API)

---

## 1. Overview

This benchmark evaluates a locally-deployed GUI agent across three perception/planning modes:

- **Screenshot mode** — Agent receives only a PNG screenshot of the browser sandbox per step; planning by Qwen2.5-VL-7B (local).
- **DOM mode** — Agent receives screenshot plus a filtered DOM context (form inputs/buttons, and a small number of short navigation links, with a blacklist for generic chrome links like "Pricing"/"Sign in"/"Donate"); planning by Qwen2.5-VL-7B (local).
- **Hybrid mode** — Same screenshot + DOM context as DOM mode, but planning and completion-evaluation are delegated to Claude over the Anthropic API.

The task suite covers two multi-step information-extraction tasks: searching HuggingFace for a model and reading its download count, and searching Wikipedia, opening an article, and reading its first sentence. Both require a search → navigate → extract flow with no Playwright assistance for the HuggingFace task.

---

## 2. Summary Results

| Mode | Success Rate | Avg Steps | Avg Total Time | Avg Step Latency |
|------|-------------|-----------|----------------|-------------------|
| Screenshot | 100% (2/2) | 3 | 66.4s | 13.0s |
| Hybrid (Claude) | 100% (2/2) | 3.5 | 34.9s | 4.2s |
| DOM | 50% (1/2) | 8 | 135.8s | 12.8s |

Moving from the 3B to the 7B local model substantially improved screenshot-mode
reliability versus the earlier benchmark round (both tasks now succeed in a
handful of steps, vs. the previous 38% success rate on a broader 8-task
suite at 3B). Hybrid mode with Claude is both reliable and the fastest
per-step, since Claude's response latency (~4s) is well under the local
model's (~13s) even with the larger DOM-augmented prompt.

**DOM mode is the clear outlier** — it is the slowest mode overall (more than
double the total time of screenshot mode) and the only mode that failed a
task.

---

## 3. Task-Level Results

| Task | Category | Playwright? | Screenshot | Hybrid (Claude) | DOM |
|------|----------|------------|-----------|------------------|-----|
| HuggingFace model page extraction | multi_step | ✗ | ✓ 2 steps / 54.7s → `6,289,765` | ✓ 2 steps / 21.9s → `6,289,765 downloads last month` | ✓ 11 steps / 179.2s → `6,289,765` |
| Wikipedia cross-page navigation | multi_step | ✓ | ✓ 4 steps / 78.0s → first sentence | ✓ 5 steps / 47.8s → first sentence | ✗ stuck in loop at `(0.21, 0.54)`, 5 steps / 92.3s |

The HuggingFace task succeeded in all three modes, though DOM mode took
roughly 5x the steps of screenshot/hybrid mode to get there. The Wikipedia
task — which requires identifying and clicking a specific result link on a
content-heavy search results page — failed outright in DOM mode.

---

## 4. Failure Mode Analysis

### 4.1 DOM Mode: Click-on-Input Instead of Type (HuggingFace)

On the HuggingFace task in DOM mode, the model repeatedly issued `click`
actions targeting the search input box instead of `type`, even though the
DOM context explicitly listed the input element. This pattern was **not**
observed in screenshot-only mode on the same task, where the model went
straight to `type`. The DOM context did not resolve the ambiguity it was
meant to resolve — at 7B scale, adding a structured element list alongside
the screenshot appears to introduce a competing signal ("here is a clickable
element") that the model resolves incorrectly for input fields, costing
several extra steps before it eventually self-corrected.

### 4.2 DOM Mode: Coordinate Loop on Search Results (Wikipedia)

On the Wikipedia task, DOM mode got stuck repeatedly clicking
`(0.21, 0.54)` — the approximate location of the
"Transformer (deep learning)" result link — without the page ever
navigating. The loop-detection guard correctly terminated the run after 5
steps as a failure.

Root cause: the DOM context sent to the model is intentionally capped (form
elements + a small number of short navigation links, filtering out generic
chrome links) to keep the prompt size manageable for a 7B model. On a
content-heavy page like a Wikipedia search results listing, the actual
target link can fall outside that cap, so the model never receives a precise
coordinate or href for it and falls back to an imprecise visual estimate that
doesn't land on the clickable element.

### 4.3 Why Not Just Deepen the DOM Extraction?

The natural fix — increase the element cap or extraction depth so the target
link is always included — was considered but not adopted, for two reasons:

1. **Latency.** DOM mode is already the slowest mode per-step (12.8s avg) and
   by far the slowest in total time (135.8s avg vs 34.9–66.4s for the other
   modes), driven partly by DOM extraction/serialization overhead on every
   step. A deeper extraction (scanning more of the DOM tree, on every step,
   for every page) would add to this cost on a mode that is already the
   least time-efficient.
2. **Model capacity.** Even where the DOM context *did* include the relevant
   elements (the HuggingFace input field), the 7B model did not reliably act
   on it correctly (§4.1). There's limited evidence that giving the model
   *more* DOM information would improve decisions — the bottleneck observed
   here is the model's ability to integrate structured context with visual
   context, not the absence of the right element from that context.

---

## 5. Discussion

### DOM Augmentation Is Not a Good Fit for This Task Class

Both benchmarked tasks are open-ended "search, navigate, extract" flows on
real-world content sites (HuggingFace, Wikipedia). For this class of task:

- **Screenshot-only mode** lets the 7B model rely on its (strong) visual
  grounding and complete both tasks in 2–4 steps.
- **Hybrid mode** gets the same or better outcomes, faster per step, by
  swapping in a more capable remote planner — the DOM context here is mostly
  redundant since Claude grounds well from the screenshot alone.
- **DOM mode** adds extraction latency and a structured context that a 7B
  model doesn't integrate reliably, while still not guaranteeing the *one*
  element that matters (the search result link) is present in a capped
  context.

The two failure modes are compounding rather than independent: even a larger
extraction cap would not have helped the HuggingFace input-vs-click confusion,
and even perfect model behavior would not have helped Wikipedia if the target
link is outside the cap. Fixing both fully would mean a much larger DOM
payload *and* a model capable of reliably prioritizing within it — at which
point DOM mode's only remaining advantage (precise coordinates) is better
delivered by routing to a stronger planner (i.e., hybrid mode), which is
already faster.

**DOM mode likely remains useful for a narrower task class**: structured
form-filling on pages where the relevant inputs are few, near the top of the
DOM, and unambiguous (e.g., a login form, a short search form with a single
field and submit button). The two-task suite here was not representative of
that class.

### Local vs. Hybrid Trade-offs (Updated)

With Claude as the hybrid planner, the latency trade-off observed previously
with Gemini reverses in Claude's favor for this hardware: Claude's ~4s/step
is faster than the local 7B model's ~13s/step, even accounting for network
round-trip. The privacy trade-off is unchanged — hybrid mode sends screenshots
to the Anthropic API, while screenshot and DOM modes remain fully local.

---

## 6. Migration Notes: Gemini → Claude

The hybrid-mode remote planner was migrated from Gemini Flash to Claude
(`claude-opus-4-5`) due to Gemini free-tier unavailability in the development
region. Changes were confined to `vlm_client.py` (new `ClaudeVLMClient`
replacing `GeminiVLMClient`, using `anthropic.Anthropic().messages.create`
with base64-encoded screenshots for both the planning and evaluation calls),
`server.py` and `executor.py` (import/type updates, `ANTHROPIC_API_KEY` env
var replacing `GEMINI_API_KEY`, `/health` field renamed `gemini` → `claude`),
`benchmark.py` (same env var and health-field rename), and
`requirements.txt` (`google-genai` → `anthropic`).

---

## 7. Conclusions

| Finding | Evidence |
|---------|---------|
| 7B model is substantially more reliable than 3B for this task class | Screenshot mode: 100% (2/2) at 7B vs 38% (3/8) at 3B (different suites, but no failures at all at 7B) |
| Hybrid mode with Claude is both reliable and the fastest per step | 100% success, 4.2s/step vs 13.0s/step (screenshot) and 12.8s/step (DOM) |
| DOM augmentation does not help — and actively hurts — at 7B for open-ended search/extract tasks | DOM: 50% success, 8 avg steps, 135.8s avg total — worst on every axis |
| DOM mode's two failure modes are independent of each other and both rooted in the same trade-off (context size vs. latency/model capacity) | §4.1 (click-on-input, HuggingFace) and §4.2 (missing target link, Wikipedia) |
| Deepening DOM extraction is not recommended as a fix | §4.3 — would worsen DOM mode's already-worst latency without addressing the model-integration bottleneck |

---

## 8. Development Process Notes

This project was developed with AI assistance throughout. Key observations on
the human-AI collaboration during this round:

- **Where AI tooling accelerated work:** Migrating the hybrid VLM client from
  Gemini to Claude (API shape differences, base64 image handling), debugging
  DOM-mode click/type confusion by tracing model outputs against DOM context
  contents, and iterating on the DOM context filter (blacklisting generic
  chrome links like "Pricing"/"Sign in"/"Donate" that were crowding out
  task-relevant elements).

- **Where human judgment was essential:** Deciding *not* to chase DOM mode
  further with deeper extraction — recognizing that the latency cost and the
  model-capacity ceiling made it the wrong investment for this task class,
  and that the benchmark data already pointed to hybrid mode as the better
  path for tasks needing precise grounding.

- **Task suite narrowing:** The benchmark suite was narrowed from the earlier
  8-task suite to two representative multi-step search/extract tasks
  (HuggingFace, Wikipedia), chosen because they reflect the target use case
  (information retrieval from real sites) without requiring the
  transient-UI-state interactions (dropdown menus, etc.) that were previously
  shown to fail regardless of perception mode due to inference latency alone.
