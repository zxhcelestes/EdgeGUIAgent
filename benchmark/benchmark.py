"""
Benchmark Runner

Runs all tasks across all modes, collects metrics, outputs a JSON + Markdown report.

Usage:
    python benchmark.py --modes screenshot hybrid --output results/
    python benchmark.py --modes screenshot --tasks form_fill navigation   # subset
"""

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

# ── Task definitions ──────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    name: str
    description: str
    start_url: str
    success_criteria: str           # human-readable, for the report
    category: str                   # "form_fill" | "navigation" | "extraction" | "multi_step"
    playwright_possible: bool = True  # can Playwright handle this with selectors?
    notes: str = ""


TASKS: list[Task] = [
    # --- Form filling ---
    Task(
        id="form_contact",
        name="Fill contact form",
        description="Fill in name='Test User', email='test@example.com', message='Hello world' and submit",
        start_url="https://httpbin.org/forms/post",
        success_criteria="Form submitted, response page shows submitted values",
        category="form_fill",
        playwright_possible=True,
    ),
    Task(
        id="form_search",
        name="Search on DuckDuckGo",
        description="Search for 'open source GUI agents 2025' and confirm results page loaded",
        start_url="https://duckduckgo.com",
        success_criteria="Results page shows relevant results",
        category="form_fill",
        playwright_possible=True,
    ),

    # --- Navigation ---
    Task(
        id="nav_github",
        name="Navigate to GitHub trending",
        description="Go to GitHub and navigate to the Trending, an Open Source repositories page",
        start_url="https://github.com",
        success_criteria="URL contains /trending or page shows trending repos",
        category="navigation",
        playwright_possible=True,
    ),
    Task(
        id="nav_hacker_news",
        name="Open top HN story",
        description="Click on the first story link on Hacker News front page",
        start_url="https://news.ycombinator.com",
        success_criteria="New page loaded at an external URL from HN",
        category="navigation",
        playwright_possible=True,
    ),

    # --- Information extraction ---
    Task(
        id="extract_title",
        name="Extract page title",
        description="Navigate to Wikipedia's 'Artificial intelligence' article and report the first sentence of the lead paragraph",
        start_url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        success_criteria="Agent outputs first sentence correctly",
        category="extraction",
        playwright_possible=True,
    ),

    # --- Multi-step workflows ---
    Task(
        id="multi_github_search",
        name="GitHub repo search and inspect",
        description="Search GitHub for 'gui agent electron', open the most starred result, and find the number of open issues",
        start_url="https://github.com/search?q=gui+agent+electron&type=repositories",
        success_criteria="Agent reports open issue count from the most starred repo",
        category="multi_step",
        playwright_possible=False,  # dynamic results make scripted approach fragile
    ),
    Task(
        id="multi_scroll_load",
        name="Infinite scroll extraction",
        description="On Hacker News, scroll down to load all visible items and count how many stories are on the front page",
        start_url="https://news.ycombinator.com",
        success_criteria="Agent reports a count between 25-35",
        category="multi_step",
        playwright_possible=True,
    ),

    # --- Canvas / no stable selectors (Playwright disadvantage) ---
    Task(
        id="canvas_excalidraw",
        name="Excalidraw canvas interaction",
        description="On Excalidraw, select the rectangle tool and draw a shape",
        start_url="https://excalidraw.com",
        success_criteria="A rectangle appears on the canvas",
        category="multi_step",
        playwright_possible=False,   # canvas-rendered, no DOM selectors
        notes="Key differentiator: DOM-based agents cannot handle this",
    ),
]

TASK_MAP = {t.id: t for t in TASKS}


# ── Benchmark runner ──────────────────────────────────────────────────────────

AGENT_SERVER = os.getenv("AGENT_SERVER", "http://localhost:8000")


@dataclass
class TaskResult:
    task_id: str
    mode: str
    success: bool
    step_count: int
    total_time_s: float
    avg_latency_s: float
    failure_reason: Optional[str]


@dataclass
class BenchmarkReport:
    timestamp: str
    modes: list[str]
    task_results: list[TaskResult] = field(default_factory=list)

    def summary_by_mode(self) -> dict:
        out = {}
        for mode in self.modes:
            mode_results = [r for r in self.task_results if r.mode == mode]
            if not mode_results:
                continue
            successes = [r for r in mode_results if r.success]
            out[mode] = {
                "success_rate": round(len(successes) / len(mode_results), 3),
                "avg_steps": round(statistics.mean(r.step_count for r in mode_results), 2),
                "avg_total_time_s": round(statistics.mean(r.total_time_s for r in mode_results), 2),
                "avg_latency_s": round(statistics.mean(r.avg_latency_s for r in mode_results), 3),
                "n_tasks": len(mode_results),
                "n_success": len(successes),
            }
        return out

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "modes": self.modes,
            "summary_by_mode": self.summary_by_mode(),
            "task_results": [r.__dict__ for r in self.task_results],
        }

    def to_markdown(self) -> str:
        lines = [
            "# GUI Agent Benchmark Report",
            f"\n**Generated:** {self.timestamp}",
            f"**Modes tested:** {', '.join(self.modes)}",
            "\n## Summary\n",
            "| Mode | Success Rate | Avg Steps | Avg Total Time | Avg Step Latency |",
            "|------|-------------|-----------|----------------|-----------------|",
        ]
        for mode, s in self.summary_by_mode().items():
            sr = f"{s['success_rate']*100:.0f}% ({s['n_success']}/{s['n_tasks']})"
            lines.append(
                f"| {mode} | {sr} | {s['avg_steps']} | {s['avg_total_time_s']}s | {s['avg_latency_s']}s |"
            )

        lines += ["\n## Task-level Results\n",
                  "| Task | Category | Playwright? | " + " | ".join(self.modes) + " |",
                  "|------|----------|------------|" + "|".join(["---"] * len(self.modes)) + "|"]

        for task in TASKS:
            row = [task.name, task.category, "✓" if task.playwright_possible else "✗"]
            for mode in self.modes:
                res = next(
                    (r for r in self.task_results if r.task_id == task.id and r.mode == mode),
                    None,
                )
                if res is None:
                    row.append("—")
                elif res.success:
                    row.append(f"✓ {res.step_count}steps {res.total_time_s:.1f}s")
                else:
                    reason = (res.failure_reason or "failed")[:30]
                    row.append(f"✗ {reason}")
            lines.append("| " + " | ".join(row) + " |")

        lines += [
            "\n## Failure Mode Analysis\n",
            "| Task | Mode | Reason |",
            "|------|------|--------|",
        ]
        for r in self.task_results:
            if not r.success:
                lines.append(f"| {r.task_id} | {r.mode} | {r.failure_reason or 'unknown'} |")

        return "\n".join(lines)

def run_task_via_api(
    task: Task,
    mode: str,
    client: httpx.Client,
    timeout: float = 1800.0,
) -> TaskResult:
    print(f"  [{mode}] {task.id}: {task.description[:60]}…")

    for _ in range(60):
        try:
            h = client.get(f"{AGENT_SERVER}/health", timeout=3.0).json()
            if not h.get("running", False):
                break
        except Exception:
            pass
        time.sleep(5.0)

    resp = client.post(
        f"{AGENT_SERVER}/run",
        json={
            "task": task.description,
            "start_url": task.start_url,
            "mode": mode,
            "max_steps": 15,
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    status = resp.json()["status"]
    if status not in ("started", "ok"):
        return TaskResult(
            task_id=task.id, mode=mode, success=False,
            step_count=0, total_time_s=0.0, avg_latency_s=0.0,
            failure_reason=f"server rejected: {status}",
        )

    t0 = time.perf_counter()

    while time.perf_counter() - t0 < timeout:
        time.sleep(10.0)
        elapsed = time.perf_counter() - t0
        try:
            h = client.get(f"{AGENT_SERVER}/health", timeout=3.0).json()
            if not h.get("running", True):
                results_resp = client.get(f"{AGENT_SERVER}/results", timeout=5.0)
                all_results = results_resp.json().get("results", [])
                matching = [r for r in all_results if r.get("task") == task.description]
                if matching:
                    r = matching[-1]
                    print(f"    found result after {elapsed:.0f}s")
                    return TaskResult(
                        task_id=task.id,
                        mode=mode,
                        success=r["success"],
                        step_count=r["step_count"],
                        total_time_s=r["total_time_s"],
                        avg_latency_s=r["avg_latency_s"],
                        failure_reason=r.get("failure_reason"),
                    )
        except Exception:
            continue

    return TaskResult(
        task_id=task.id, mode=mode, success=False,
        step_count=0, total_time_s=timeout, avg_latency_s=0.0,
        failure_reason="benchmark timeout",
    )

def run_benchmark(
    modes: list[str],
    task_ids: Optional[list[str]] = None,
    output_dir: str = "results",
) -> BenchmarkReport:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tasks = [TASK_MAP[t] for t in task_ids] if task_ids else TASKS
    report = BenchmarkReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        modes=modes,
    )

    with httpx.Client(timeout=10.0) as client:
        # Health check
        try:
            health = client.get(f"{AGENT_SERVER}/health").json()
            print(f"Agent server: ollama={health.get('ollama')}, gemini={health.get('gemini')}")
        except Exception as e:
            print(f"Warning: cannot reach agent server at {AGENT_SERVER}: {e}")

        for mode in modes:
            print(f"\n=== Mode: {mode} ===")
            for task in tasks:
                # Skip hybrid tasks if no Gemini key
                if mode == "hybrid" and not os.getenv("GEMINI_API_KEY"):
                    print(f"  [hybrid] skipping {task.id} — no GEMINI_API_KEY")
                    report.task_results.append(TaskResult(
                        task_id=task.id, mode=mode, success=False,
                        step_count=0, total_time_s=0.0, avg_latency_s=0.0,
                        failure_reason="no API key",
                    ))
                    continue

                result = run_task_via_api(task, mode, client)
                report.task_results.append(result)
                symbol = "✓" if result.success else "✗"
                print(f"    {symbol} {result.step_count} steps, {result.total_time_s:.1f}s")
                time.sleep(1.0)   # brief pause between tasks

    # Save outputs
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = Path(output_dir) / f"benchmark_{ts}.json"
    md_path   = Path(output_dir) / f"benchmark_{ts}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    md_path.write_text(report.to_markdown())
    print(f"\nResults saved:\n  {json_path}\n  {md_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GUI agent benchmark")
    parser.add_argument("--modes", nargs="+", default=["screenshot"],
                        choices=["screenshot", "hybrid", "dom"])
    parser.add_argument("--tasks", nargs="*", help="Task IDs to run (default: all)")
    parser.add_argument("--output", default="results")
    parser.add_argument("--server", default=AGENT_SERVER)
    args = parser.parse_args()

    AGENT_SERVER = args.server
    report = run_benchmark(args.modes, args.tasks, args.output)

    print("\n=== Summary ===")
    for mode, s in report.summary_by_mode().items():
        print(f"{mode}: {s['success_rate']*100:.0f}% success, "
              f"avg {s['avg_steps']} steps, avg {s['avg_latency_s']}s/step")
