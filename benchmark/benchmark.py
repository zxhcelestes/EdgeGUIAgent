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
    Task(
        id="github_readme_extract",
        name="GitHub README code block extraction",
        description="Search GitHub for 'ui-tars', open the most starred repository, find the first code block in the README and report its content",
        start_url="https://github.com/search?q=ui-tars&type=repositories&s=stars&o=desc",
        success_criteria="Agent reports the content of the first code block in the README",
        category="multi_step",
        playwright_possible=False,
        notes="Requires: search result selection + page navigation + content extraction",
    ),
    Task(
        id="wikipedia_cross_page",
        name="Wikipedia cross-page navigation",
        description="On the Wikipedia page for 'Transformer (machine learning)', find and click the link to 'Attention mechanism', then report the first sentence of that page",
        start_url="https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        success_criteria="Agent reports the first sentence of the Attention mechanism Wikipedia page",
        category="multi_step",
        playwright_possible=True,
        notes="Tests in-text link navigation and cross-page content extraction",
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
