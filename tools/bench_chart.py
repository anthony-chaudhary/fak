#!/usr/bin/env python3
"""bench_chart.py -- generate benchmark visualizations.

Usage:
  python tools/bench_chart.py throughput --model smollm2-135m
  python tools/bench_chart.py scaling --machines anthony-laptop,mac-m3pro
  python tools/bench_chart.py cost --model smollm2-135m
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "fak" / "experiments" / "benchmark"
CATALOG_PATH = BENCHMARK_DIR / "catalog.json"


def load_catalog() -> Optional[Dict]:
    """Load catalog, return None on error."""
    if not CATALOG_PATH.exists():
        return None
    try:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_batch_results(run_id: str, catalog: Dict) -> Optional[Dict]:
    """Load batch results for a run."""
    for r in catalog.get("runs", []):
        if r["run_id"] == run_id:
            run_dir = ROOT / r["path"]
            batch_path = run_dir / "batch.json"
            if batch_path.exists():
                try:
                    with open(batch_path, encoding="utf-8") as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError):
                    pass
            return None
    return None


def html_header(title: str) -> str:
    """Generate HTML header with Plotly."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; }}
        .chart {{ width: 100%; height: 600px; }}
        .controls {{ margin: 20px 0; }}
        button {{ margin: 5px; padding: 8px 16px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""


def html_footer() -> str:
    """Generate HTML footer."""
    return """</body>
</html>"""


def chart_throughput(runs: List[Dict], output_path: Path) -> bool:
    """Generate throughput vs machine bar chart."""
    if not runs:
        print("[WARN] No runs for throughput chart", file=sys.stderr)
        return False

    # Group by machine
    by_machine: Dict[str, List[Dict]] = {}
    for r in runs:
        mid = r.get("machine_id", "unknown")
        by_machine.setdefault(mid, []).append(r)

    # Get best run per machine
    best_runs = []
    for mid, machine_runs in by_machine.items():
        best = max(machine_runs, key=lambda r: r.get("peak_tok_per_sec") or 0)
        best_runs.append(best)

    best_runs.sort(key=lambda r: r.get("peak_tok_per_sec") or 0, reverse=True)

    # Extract data
    machines = [r.get("machine_id", "?") for r in best_runs]
    peak_vals = [r.get("peak_tok_per_sec", 0) for r in best_runs]
    baseline_vals = [r.get("baseline_tok_per_sec", 0) for r in best_runs]

    # Generate Plotly chart
    data = [
        {
            "x": machines,
            "y": peak_vals,
            "name": "Peak Throughput",
            "type": "bar",
            "marker": {"color": "#2E86AB"}
        },
        {
            "x": machines,
            "y": baseline_vals,
            "name": "Baseline (B=1)",
            "type": "bar",
            "marker": {"color": "#A23B72"}
        }
    ]

    layout = {
        "title": "Throughput vs Machine",
        "xaxis": {"title": "Machine"},
        "yaxis": {"title": "Tokens/sec (aggregate)"},
        "barmode": "group",
        "hovermode": "x unified"
    }

    html = html_header("Throughput vs Machine")
    html += '<div id="chart" class="chart"></div>\n'
    html += f'<script>Plotly.newPlot("chart", {json.dumps(data)}, {json.dumps(layout)});</script>\n'
    html += html_footer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[bench_chart] Wrote {output_path}", file=sys.stderr)
    return True


def chart_scaling(runs: List[Dict], catalog: Dict, output_path: Path) -> bool:
    """Generate scaling curves (batch size vs throughput)."""
    if not runs:
        print("[WARN] No runs for scaling chart", file=sys.stderr)
        return False

    # Load batch results for each run
    series = []
    for r in runs:
        batch = load_batch_results(r["run_id"], catalog)
        if not batch:
            continue
        points = batch.get("points", [])
        if not points:
            continue

        x_vals = [p.get("batch", 0) for p in points]
        y_vals = [p.get("agg_tok_per_sec", 0) for p in points]

        series.append({
            "name": r.get("machine_id", r["run_id"]),
            "x": x_vals,
            "y": y_vals
        })

    if not series:
        print("[WARN] No batch data for scaling chart", file=sys.stderr)
        return False

    # Generate Plotly chart
    traces = []
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    for i, s in enumerate(series):
        traces.append({
            "x": s["x"],
            "y": s["y"],
            "name": s["name"],
            "type": "scatter",
            "mode": "lines+markers",
            "line": {"color": colors[i % len(colors)]}
        })

    layout = {
        "title": "Batch Scaling Curves",
        "xaxis": {"title": "Batch Size", "type": "log"},
        "yaxis": {"title": "Aggregate Tokens/sec"},
        "hovermode": "x unified"
    }

    html = html_header("Batch Scaling Curves")
    html += '<div id="chart" class="chart"></div>\n'
    html += f'<script>Plotly.newPlot("chart", {json.dumps(traces)}, {json.dumps(layout)});</script>\n'
    html += html_footer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[bench_chart] Wrote {output_path}", file=sys.stderr)
    return True


def chart_prefill_decode(runs: List[Dict], output_path: Path) -> bool:
    """Generate prefill vs decode grouped bar chart."""
    if not runs:
        print("[WARN] No runs for prefill/decode chart", file=sys.stderr)
        return False

    # This would require loading kernel.json for each run
    # For now, return placeholder
    print("[WARN] Prefill/decode chart not yet implemented", file=sys.stderr)
    return False


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", help="Chart type")

    # throughput
    throughput_p = sub.add_parser("throughput", help="Throughput vs machine")
    throughput_p.add_argument("--model", help="Filter by model")
    throughput_p.add_argument("--precision", help="Filter by precision")
    throughput_p.add_argument("--machines", help="Comma-separated machine IDs")
    throughput_p.add_argument("-o", "--output", type=Path, default=BENCHMARK_DIR / "charts" / "throughput.html",
                             help="Output HTML file")

    # scaling
    scaling_p = sub.add_parser("scaling", help="Batch scaling curves")
    scaling_p.add_argument("--model", help="Filter by model")
    scaling_p.add_argument("--machines", help="Comma-separated machine IDs")
    scaling_p.add_argument("-o", "--output", type=Path, default=BENCHMARK_DIR / "charts" / "scaling.html",
                           help="Output HTML file")

    # prefill-decode
    sub.add_parser("prefill-decode", help="Prefill vs decode comparison")

    # all
    all_p = sub.add_parser("all", help="Generate all charts")
    all_p.add_argument("--model", help="Filter by model")
    all_p.add_argument("-o", "--output-dir", type=Path, default=BENCHMARK_DIR / "charts",
                      help="Output directory")

    args = ap.parse_args(argv)

    if not args.command:
        ap.print_help()
        return 1

    catalog = load_catalog()
    if not catalog:
        print("[ERROR] Catalog not found. Run 'python tools/bench_catalog.py build' first", file=sys.stderr)
        return 1

    runs = catalog.get("runs", [])

    # Apply filters
    if hasattr(args, "model") and args.model:
        runs = [r for r in runs if r.get("model") == args.model]
    if hasattr(args, "precision") and args.precision:
        runs = [r for r in runs if r.get("precision") == args.precision]
    if hasattr(args, "machines") and args.machines:
        machine_list = args.machines.split(",")
        runs = [r for r in runs if r.get("machine_id") in machine_list]

    if args.command == "throughput":
        ok = chart_throughput(runs, args.output)
        return 0 if ok else 1

    if args.command == "scaling":
        ok = chart_scaling(runs, catalog, args.output)
        return 0 if ok else 1

    if args.command == "prefill-decode":
        ok = chart_prefill_decode(runs, args.output if hasattr(args, "output") else BENCHMARK_DIR / "charts" / "prefill-decode.html")
        return 0 if ok else 1

    if args.command == "all":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        chart_throughput(runs, args.output_dir / "throughput.html")
        chart_scaling(runs, catalog, args.output_dir / "scaling.html")
        chart_prefill_decode(runs, args.output_dir / "prefill-decode.html")
        print(f"[bench_chart] Generated charts in {args.output_dir}", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
