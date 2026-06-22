#!/usr/bin/env python3
"""
fleet_value_projection.py -- executive value sweep for fleet scale.

This joins the current measured FAK artifacts into one reusable projection:

  * measured syscall / turn-tax / fleet-cache inputs from checked-in JSON/CSV;
  * a scale ladder from 1 agent, 25 turns, 20k context to
    100 agents, 250 turns, 500k context, with intermediate levels;
  * transparent split between measured tool-turn savings and theoretical
    long-context/resident-KV savings;
  * Markdown + CSV + JSON + PNG/SVG/PDF/JPG charts that can be regenerated with one command.

Run:
  python tools/fleet_value_projection.py

Optional:
  python tools/fleet_value_projection.py --levels my-levels.json --no-plots

The optional levels file is a JSON array of objects with:
  {"level":"L0", "agents":1, "turns_per_agent":25, "context_tokens":20000}
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any

import fleet_version


ROOT = Path(__file__).resolve().parents[1]
FLEET_DIR = ROOT / "fak" / "experiments" / "fleet"
OUT_DIR = ROOT / "fak" / "experiments" / "value-sweep"
MARKDOWN = ROOT / "fak" / "FLEET-VALUE-PROJECTION.md"


# The requested sweep: endpoints plus six intermediate levels.
# Edit this block or pass --levels for a different ladder.
DEFAULT_LEVELS = [
    {"level": "L0 solo", "agents": 1, "turns_per_agent": 25, "context_tokens": 20_000},
    {"level": "L1 pair", "agents": 2, "turns_per_agent": 40, "context_tokens": 35_000},
    {"level": "L2 small team", "agents": 5, "turns_per_agent": 60, "context_tokens": 60_000},
    {"level": "L3 pod", "agents": 10, "turns_per_agent": 90, "context_tokens": 100_000},
    {"level": "L4 squad", "agents": 20, "turns_per_agent": 120, "context_tokens": 160_000},
    {"level": "L5 fleet", "agents": 40, "turns_per_agent": 160, "context_tokens": 250_000},
    {"level": "L6 large fleet", "agents": 70, "turns_per_agent": 210, "context_tokens": 360_000},
    {"level": "L7 requested max", "agents": 100, "turns_per_agent": 250, "context_tokens": 500_000},
]


# Long-context residual model. These are not new measurements; they are deliberately
# surfaced as assumptions and written into the JSON/Markdown.
LONG_CONTEXT = {
    "effective_kv_miss_rate": 0.0579,
    "source": "grounded_recalc.py / grounded-recalc-results.md: 30-day telemetry, cache hit 94.21%",
    "fresh_input_dollars_per_mtok": 3.0,
    "note": (
        "The context axis is a theory projection. The realistic-workload profile "
        "provides measured prefix bounds; ladder rows above that measured max extrapolate."
    ),
}


# Default self-host hardware profile inherited from inline_tool_roi.py. Hardware
# changes only the resident-KV / cold-prefill rows, not the measured turn counts.
HARDWARE = {
    "id": "70B_8xH100_default",
    "prefill_tokens_per_second_per_replica": 20_000,
    "gpus_per_replica": 8,
    "dollars_per_gpu_hour": 3.0,
    "watts_per_gpu": 700,
    "source": "inline_tool_roi.py STACK default; sensitivity knob, not a fresh benchmark",
}


VISUAL_FORMATS = ("png", "svg", "pdf", "jpg")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def money(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.1f}k"
    if abs(x) >= 10:
        return f"${x:,.0f}"
    if abs(x) >= 1:
        return f"${x:,.2f}"
    if abs(x) >= 0.01:
        return f"${x:,.3f}"
    return f"${x:,.2f}"


def human_tokens(x: float) -> str:
    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"


def duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def percent(x: float) -> str:
    return f"{x:.0%}"


def wrap_bullets(lines: list[str], width: int = 52) -> str:
    out: list[str] = []
    for line in lines:
        wrapped = textwrap.wrap(line, width=width)
        if not wrapped:
            out.append("-")
            continue
        out.append(f"- {wrapped[0]}")
        out.extend(f"  {w}" for w in wrapped[1:])
    return "\n".join(out)


def chart_text(s: str) -> str:
    return s.replace("$", r"\$")


def r2_score(y: list[float], pred: list[float]) -> float:
    mean = sum(y) / len(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def load_inputs() -> dict[str, Any]:
    report = load_json(ROOT / "fak" / "report.json")
    turntax = load_json(ROOT / "fak" / "experiments" / "turn-tax" / "turntax-airline.json")
    fleet_rows = load_csv(FLEET_DIR / "readfleet-50x50.csv")
    profile = load_json(FLEET_DIR / "readfleet-50x50.json").get("profile", {})
    realistic_profile = load_json(
        ROOT / "fak" / "experiments" / "agent-live" / "realistic-workload" / "profile.json"
    )

    measured = {}
    calls = []
    shared = []
    cross = []
    coupon = []
    pool = float(profile.get("shared_pool", 8))
    p_shared = float(profile.get("p_shared", 0.45))

    for r in fleet_rows:
        t = int(float(r["turns"]))
        a = int(float(r["agents"]))
        c = float(r["calls"])
        s = float(r["shared_saved_mean"])
        x = float(r["cross_uplift_mean"])
        measured[(t, a)] = {
            "shared_saved": s,
            "cross_uplift": x,
            "source": "measured",
        }
        calls.append(c)
        shared.append(s)
        cross.append(x)
        coupon.append(coupon_cross_uplift(t, a, pool, p_shared))

    # Through-origin fit for total shared turn savings. This is the same near-bilinear
    # result described in FLEET-SWEEP-RESULTS, recomputed from checked-in CSV.
    shared_rate = sum(c * s for c, s in zip(calls, shared)) / sum(c * c for c in calls)
    shared_pred = [min(c, shared_rate * c) for c in calls]
    shared_r2 = r2_score(shared, shared_pred)
    coupon_r2 = r2_score(cross, coupon)

    cost_model = turntax["cost_model"]
    per_turn_cost = (
        cost_model["prompt_tokens_per_turn"] * cost_model["dollars_per_mtok_in"]
        + cost_model["completion_tokens_per_turn"] * cost_model["dollars_per_mtok_out"]
    ) / 1_000_000
    per_turn_tokens = cost_model["prompt_tokens_per_turn"] + cost_model["completion_tokens_per_turn"]

    return {
        "report": report,
        "turntax": turntax,
        "fleet_measured": measured,
        "fleet_profile": profile,
        "realistic_profile": realistic_profile,
        "shared_saved_rate": shared_rate,
        "shared_saved_r2": shared_r2,
        "coupon_r2": coupon_r2,
        "pool": pool,
        "p_shared": p_shared,
        "per_turn_cost": per_turn_cost,
        "per_turn_tokens": per_turn_tokens,
        "model_turn_latency_s": cost_model["model_turn_latency_ms"] / 1000,
    }


def coupon_cross_uplift(turns: float, agents: float, pool: float, p_shared: float) -> float:
    if agents <= 1 or pool <= 0 or p_shared <= 0:
        return 0.0
    cover = pool * (1 - (1 - 1 / pool) ** (p_shared * turns))
    return (agents - 1) * cover


def project(level: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    app_ver = fleet_version.app_version(ROOT)
    agents = int(level["agents"])
    turns = int(level["turns_per_agent"])
    context = int(level["context_tokens"])
    calls = agents * turns
    measured_cell = inputs["fleet_measured"].get((turns, agents))

    if measured_cell is not None:
        shared_saved = measured_cell["shared_saved"]
        cross_uplift = measured_cell["cross_uplift"]
        turn_evidence = "measured_exact_cell"
    else:
        shared_saved = min(calls, inputs["shared_saved_rate"] * calls)
        cross_uplift = coupon_cross_uplift(turns, agents, inputs["pool"], inputs["p_shared"])
        turn_evidence = "measured_supported_projection"

    per_turn_cost = inputs["per_turn_cost"]
    per_turn_tokens = inputs["per_turn_tokens"]
    turn_latency = inputs["model_turn_latency_s"]
    miss = LONG_CONTEXT["effective_kv_miss_rate"]
    rho = HARDWARE["prefill_tokens_per_second_per_replica"]
    gpus = HARDWARE["gpus_per_replica"]
    gpu_price = HARDWARE["dollars_per_gpu_hour"]
    watts = HARDWARE["watts_per_gpu"]

    tool_tokens_saved = shared_saved * per_turn_tokens
    tool_dollars_saved = shared_saved * per_turn_cost
    tool_agent_seconds_saved = shared_saved * turn_latency
    control_plane_seconds_saved = calls * (
        inputs["report"]["spawned_hook_baseline"]["p50_ns"] - inputs["report"]["vdso_on"]["p50_ns"]
    ) / 1_000_000_000

    cold_context_tokens = calls * miss * context
    context_api_dollars = cold_context_tokens * LONG_CONTEXT["fresh_input_dollars_per_mtok"] / 1_000_000
    context_replica_seconds = cold_context_tokens / rho
    context_gpu_hours = context_replica_seconds / 3600 * gpus
    context_gpu_dollars = context_gpu_hours * gpu_price
    context_kwh = context_replica_seconds / 3600 * gpus * watts / 1000

    baseline_parallel_wall_s = turns * turn_latency + turns * miss * context / rho
    avg_tool_latency_s = (shared_saved / agents) * turn_latency if agents else 0
    avg_context_latency_s = turns * miss * context / rho
    avg_latency_saved_s = avg_tool_latency_s + avg_context_latency_s
    projected_parallel_wall_s = max(0.0, baseline_parallel_wall_s - avg_latency_saved_s)

    total_api_equiv_value = tool_dollars_saved + context_api_dollars
    total_self_host_value = tool_dollars_saved + context_gpu_dollars

    return {
        "version": app_ver,
        "level": level["level"],
        "agents": agents,
        "turns_per_agent": turns,
        "context_tokens": context,
        "calls": calls,
        "turn_evidence": turn_evidence,
        "shared_saved_turns": shared_saved,
        "cross_agent_uplift_turns": cross_uplift,
        "shared_saved_pct_of_calls": shared_saved / calls if calls else 0,
        "cross_uplift_pct_of_calls": cross_uplift / calls if calls else 0,
        "tool_tokens_saved": tool_tokens_saved,
        "tool_dollars_saved": tool_dollars_saved,
        "tool_agent_hours_saved": tool_agent_seconds_saved / 3600,
        "control_plane_seconds_saved_measured": control_plane_seconds_saved,
        "cold_context_tokens_avoided_theory": cold_context_tokens,
        "context_api_equiv_dollars_theory": context_api_dollars,
        "context_replica_hours_avoided_theory": context_replica_seconds / 3600,
        "context_gpu_hours_avoided_theory": context_gpu_hours,
        "context_gpu_dollars_avoided_theory": context_gpu_dollars,
        "context_kwh_avoided_theory": context_kwh,
        "avg_latency_saved_per_agent_s": avg_latency_saved_s,
        "baseline_parallel_wall_s": baseline_parallel_wall_s,
        "projected_parallel_wall_s": projected_parallel_wall_s,
        "total_api_equiv_value_per_run": total_api_equiv_value,
        "total_self_host_value_per_run": total_self_host_value,
    }


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    cols = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json_payload(inputs: dict[str, Any], rows: list[dict[str, Any]], out: Path) -> None:
    payload = {
        "schema": "fak.value-sweep.v1",
        "app_version": fleet_version.app_version(ROOT),
        "generated_by": "tools/fleet_value_projection.py",
        "scale_ladder": {
            "min": {"agents": 1, "turns_per_agent": 25, "context_tokens": 20_000},
            "max": {"agents": 100, "turns_per_agent": 250, "context_tokens": 500_000},
            "levels": len(rows),
        },
        "measured_inputs": {
            "syscall_report": "fak/report.json",
            "turntax": "fak/experiments/turn-tax/turntax-airline.json",
            "fleet_csv": "fak/experiments/fleet/readfleet-50x50.csv",
            "realistic_workload": "fak/experiments/agent-live/realistic-workload/profile.json",
            "spawned_hook_p50_ns": inputs["report"]["spawned_hook_baseline"]["p50_ns"],
            "inproc_p50_ns": inputs["report"]["vdso_on"]["p50_ns"],
            "model_turn_latency_s": inputs["model_turn_latency_s"],
            "per_turn_cost": inputs["per_turn_cost"],
            "per_turn_tokens": inputs["per_turn_tokens"],
            "fleet_shared_saved_rate_fit": inputs["shared_saved_rate"],
            "fleet_shared_saved_rate_r2": inputs["shared_saved_r2"],
            "fleet_coupon_cross_uplift_r2": inputs["coupon_r2"],
        },
        "theory_assumptions": {
            "long_context": LONG_CONTEXT,
            "hardware": HARDWARE,
        },
        "rows": rows,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_markdown(inputs: dict[str, Any], rows: list[dict[str, Any]], out: Path) -> None:
    top = rows[-1]
    profile = inputs["realistic_profile"]
    L: list[str] = []
    a = L.append
    a("# Fleet Value Projection - executive sweep\n")
    a(
        "> Generated by `python tools/fleet_value_projection.py`. The two images are "
        "plain-English summary sheets: one takeaway per image, with the assumptions printed "
        "on the image. The audit table below keeps the detailed numbers.\n"
    )

    a("## Executive read\n")
    a(
        f"At the requested top end (**{top['agents']} agents x {top['turns_per_agent']} turns x "
        f"{top['context_tokens']:,} context tokens**), the read-heavy cache-favorable projection "
        f"deletes **{top['shared_saved_turns']:,.0f} of {top['calls']:,} tool round-trips** "
        f"({top['shared_saved_pct_of_calls']:.1%}). The fleet-only portion is "
        f"**{top['cross_agent_uplift_turns']:,.0f} turns** beyond agents run in isolation."
    )
    a(
        f"Priced two ways, that is **{money(top['tool_dollars_saved'])}/run** from measured "
        f"tool-turn deletion plus a theoretical long-context residual of "
        f"**{money(top['context_api_equiv_dollars_theory'])}/run API-equivalent fresh-input** "
        f"or **{money(top['context_gpu_dollars_avoided_theory'])}/run self-host GPU** "
        f"on the default `{HARDWARE['id']}` profile. The latency view is "
        f"**{duration(top['avg_latency_saved_per_agent_s'])} average per-agent latency removed**, "
        "but that wall-clock row is theory because it assumes the saved turns land on the "
        "critical path evenly across agents.\n"
    )

    a("![headline value sweep](experiments/value-sweep/value-sweep-headline.png)\n")
    a("![hardware sensitivity](experiments/value-sweep/value-sweep-hardware.png)\n")
    a(
        "Front doors: [README](../README.md), "
        "[benchmark visuals](../VISUALS-benchmarking-status-2026-06-18.md), and "
        "[raw value-sweep exports](experiments/value-sweep/)."
    )
    a(
        "Export formats: "
        "[headline PNG](experiments/value-sweep/value-sweep-headline.png) / "
        "[SVG](experiments/value-sweep/value-sweep-headline.svg) / "
        "[PDF](experiments/value-sweep/value-sweep-headline.pdf) / "
        "[JPG](experiments/value-sweep/value-sweep-headline.jpg); "
        "[hardware PNG](experiments/value-sweep/value-sweep-hardware.png) / "
        "[SVG](experiments/value-sweep/value-sweep-hardware.svg) / "
        "[PDF](experiments/value-sweep/value-sweep-hardware.pdf) / "
        "[JPG](experiments/value-sweep/value-sweep-hardware.jpg).\n"
    )

    a("## Scale ladder\n")
    a("| level | agents | turns/agent | context | calls | evidence | turns deleted | cross-agent only | API-equiv value/run | self-host value/run | parallel wall baseline->projected | avg latency saved/agent |")
    a("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        a(
            f"| {r['level']} | {r['agents']} | {r['turns_per_agent']} | "
            f"{r['context_tokens']:,} | {r['calls']:,} | {r['turn_evidence']} | "
            f"{r['shared_saved_turns']:,.0f} | {r['cross_agent_uplift_turns']:,.0f} | "
            f"{money(r['total_api_equiv_value_per_run'])} | {money(r['total_self_host_value_per_run'])} | "
            f"{duration(r['baseline_parallel_wall_s'])}->{duration(r['projected_parallel_wall_s'])} | "
            f"{duration(r['avg_latency_saved_per_agent_s'])} |"
        )

    a("\n## Assumptions in plain English\n")
    a("| assumption | what it means |")
    a("|---|---|")
    a(f"| Scale ladder | The generated sweep uses the requested range: 1->{top['agents']} agents, 25->{top['turns_per_agent']} turns per agent, and 20k->{top['context_tokens']:,} context tokens. |")
    a("| Workload shape | The headline applies to a read-heavy, cache-friendly agent workload. Write-heavy fleets need scoped invalidation; global invalidation loses value around 1% writes. |")
    a(f"| Measured cells | Small rows inside the 50x50 fleet grid use measured results from `readfleet-50x50.csv`; larger rows use the measured-supported fit, R2={inputs['shared_saved_r2']:.5f}. |")
    a(f"| Tool-turn dollars | The measured-turn value uses `turntax-airline.json`: {inputs['per_turn_tokens']:,} tokens, {money(inputs['per_turn_cost'])}, and {inputs['model_turn_latency_s']:.1f}s per tool round-trip. |")
    a(f"| Long-context theory | Context dollars use a {LONG_CONTEXT['effective_kv_miss_rate']:.2%} fresh-context miss rate and {money(LONG_CONTEXT['fresh_input_dollars_per_mtok'])}/million input tokens. Rows above the measured prefix max are extrapolations. |")
    a(f"| Hardware theory | Self-host values assume `{HARDWARE['id']}`: {HARDWARE['prefill_tokens_per_second_per_replica']:,} prefill tokens/sec/replica, {HARDWARE['gpus_per_replica']} GPUs, {money(HARDWARE['dollars_per_gpu_hour'])}/GPU-hour, {HARDWARE['watts_per_gpu']} W/GPU. |")
    a("| Not claimed | The charts do not prove the model reasons the same percent less, do not price engineering work, and do not claim an API user can delete provider-internal prefill unless the provider exposes that mechanism. |")

    a("\n## What is measured vs theory\n")
    a("| row | source | how it is used | status |")
    a("|---|---|---|---|")
    a(
        f"| In-process syscall cost | `fak/report.json` | control-plane overhead: "
        f"{inputs['report']['vdso_on']['p50_ns']:,} ns p50 vs "
        f"{inputs['report']['spawned_hook_baseline']['p50_ns'] / 1e6:.3f} ms spawned-hook p50 | **measured on this box** |"
    )
    a(
        f"| Per-turn price and latency | `turntax-airline.json` | "
        f"{inputs['per_turn_tokens']:,} tok/turn, {money(inputs['per_turn_cost'])}/turn, "
        f"{inputs['model_turn_latency_s']:.1f}s/turn | **measured artifact + cost-model knobs** |"
    )
    a(
        f"| Fleet turn deletion | `readfleet-50x50.csv` | exact cells inside the 50x50 grid; "
        f"outside it, total turn deletion uses a through-origin fit "
        f"(rate={inputs['shared_saved_rate']:.3f}, R2={inputs['shared_saved_r2']:.5f}) | "
        "**measured-supported projection** |"
    )
    a(
        f"| Cross-agent-only uplift | `readfleet-50x50.csv` + coupon law | "
        f"`(A-1)*pool*(1-(1-1/pool)^(p_shared*T))`, pool={inputs['pool']:.0f}, "
        f"p_shared={inputs['p_shared']:.2f}, R2={inputs['coupon_r2']:.5f} on measured grid | "
        "**derived from measured mechanism** |"
    )
    a(
        f"| Long-context residual | `grounded_recalc.py` | miss={LONG_CONTEXT['effective_kv_miss_rate']:.2%}; "
        f"prices fresh context misses across the requested 20k->500k context ladder | "
        "**theory; grounded by observed miss rate** |"
    )
    a(
        f"| Realistic profile bounds | `realistic-workload/profile.json` | measured prefix median "
        f"{profile['prefix_tokens']['median']:,}, p90 {profile['prefix_tokens']['p90']:,}, "
        f"max {profile['prefix_tokens']['max']:,}; requested ladder reaches "
        f"{top['context_tokens']:,}, above this measured max | "
        "**measured bounds, extrapolated above max** |"
    )

    a("\n## Hardware note\n")
    a(
        f"Hardware changes only the resident-KV / cold-prefill rows. With the default "
        f"`{HARDWARE['id']}` profile, avoided context work is "
        f"`cold_context_tokens / {HARDWARE['prefill_tokens_per_second_per_replica']:,}` "
        f"replica-seconds, then multiplied by {HARDWARE['gpus_per_replica']} GPUs, "
        f"{money(HARDWARE['dollars_per_gpu_hour'])}/GPU-hour, and "
        f"{HARDWARE['watts_per_gpu']} W/GPU. If a B200/H200/local-GPU profile changes "
        "prefill throughput, price, or power, the self-host dollars/watts move linearly. "
        "The measured turn counts do not change.\n"
    )

    a("## Caveats before quoting\n")
    a("- The turn-deletion surface is a **read-heavy, cache-favorable** workload. Write-heavy fleets need the scoped eraser; global invalidation flips negative around 1% writes.")
    a("- `shared_saved_turns` are tool round-trips deleted, not proof that a frontier model reasons the same fraction fewer semantic steps on arbitrary tasks.")
    a("- The long-context row is a resident-KV / provider-ships-this / self-host projection. A plain API consumer still gets the trust floor and gateway caching, but cannot delete the provider's internal prefill cost unless the provider exposes that mechanism.")
    a("- The requested 500k context endpoint is beyond the repo's measured realistic transcript profile; it is included because the user asked for that planning horizon.\n")

    a("## Regenerate\n")
    a("```powershell")
    a("python tools\\fleet_value_projection.py")
    a("```")
    a("\nOutputs: `fak/experiments/value-sweep/value-sweep.{csv,json}`, both charts as `.png`/`.svg`/`.pdf`/`.jpg`, and this Markdown report.")

    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(L))
        f.write("\n")


def plot(rows: list[dict[str, Any]], out_dir: Path, inputs: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    top = rows[-1]
    y = list(range(len(rows)))
    labels = [
        f"{r['level'].split()[0]}  {r['agents']} agents, {r['turns_per_agent']} turns"
        for r in rows
    ]
    pct_saved = [r["shared_saved_pct_of_calls"] * 100 for r in rows]

    fig = plt.figure(figsize=(14.5, 8.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.22, 0.78], width_ratios=[1.23, 0.77], hspace=0.06, wspace=0.24)

    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    ax_head.text(
        0.0,
        0.92,
        "Plain English: FAK skips duplicate tool trips in read-heavy fleets",
        fontsize=20,
        fontweight="bold",
        color="#111827",
        va="top",
    )
    ax_head.text(
        0.0,
        0.48,
        (
            f"At the max row: {top['shared_saved_turns']:,.0f} of {top['calls']:,} tool round-trips "
            f"are avoided ({percent(top['shared_saved_pct_of_calls'])}). "
            f"Measured-turn value: {chart_text(money(top['tool_dollars_saved']))}/run. "
            f"Long-context theory: {chart_text(money(top['context_api_equiv_dollars_theory']))}/run API-equivalent."
        ),
        fontsize=12.5,
        color="#374151",
        va="top",
    )

    ax = fig.add_subplot(gs[1, 0])
    bars = ax.barh(y, pct_saved, color="#2563eb", edgecolor="#1e40af", height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("percent of tool round-trips skipped")
    ax.set_title("One thing to read: the share of duplicate tool trips removed", loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, r in zip(bars, rows):
        label = f"{r['shared_saved_turns']:,.0f}/{r['calls']:,} ({percent(r['shared_saved_pct_of_calls'])})"
        ax.text(min(bar.get_width() + 1.2, 97), bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9.5, color="#111827")

    ax_note = fig.add_subplot(gs[1, 1])
    ax_note.axis("off")
    ax_note.add_patch(Rectangle((0, 0), 1, 1, transform=ax_note.transAxes, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0))
    ax_note.text(0.05, 0.95, "Assumptions printed on the chart", fontsize=13, fontweight="bold", va="top", color="#111827", transform=ax_note.transAxes)
    assumption_text = wrap_bullets([
        f"Scale is chosen for this planning sweep: 1->{top['agents']} agents, 25->{top['turns_per_agent']} turns, 20k->{top['context_tokens'] // 1000}k context tokens.",
        "This is a read-heavy, cache-friendly workload. Write-heavy fleets need scoped invalidation.",
        f"Rows inside the 50x50 fleet grid are measured. Larger rows use the measured fit, R2={inputs['shared_saved_r2']:.5f}.",
        f"Dollar value from skipped tool trips uses {chart_text(money(rows[0]['tool_dollars_saved'] / rows[0]['shared_saved_turns']))} per avoided tool round-trip.",
        f"Long-context value is theory: {LONG_CONTEXT['effective_kv_miss_rate']:.2%} fresh-context miss rate and {chart_text(money(LONG_CONTEXT['fresh_input_dollars_per_mtok']))}/million input tokens.",
        "Not a claim that the model reasons this percent less; this only counts avoidable tool trips.",
    ], width=48)
    ax_note.text(0.05, 0.86, assumption_text, fontsize=10.2, va="top", color="#1f2937", linespacing=1.27, transform=ax_note.transAxes)

    fig.subplots_adjust(left=0.14, right=0.96, top=0.94, bottom=0.08)
    save_figure_all(fig, out_dir, "value-sweep-headline")
    plt.close(fig)

    fig2 = plt.figure(figsize=(14.5, 7.4), facecolor="white")
    gs2 = fig2.add_gridspec(2, 3, height_ratios=[0.20, 0.80], wspace=0.18, hspace=0.08)

    ax2_head = fig2.add_subplot(gs2[0, :])
    ax2_head.axis("off")
    ax2_head.text(0.0, 0.93, "Hardware sheet: this only prices the long-context theory row", fontsize=20, fontweight="bold", color="#111827", va="top")
    ax2_head.text(
        0.0,
        0.47,
        "The measured tool-trip savings do not change with GPU choice. Only the self-host cold-prefill math moves.",
        fontsize=12.5,
        color="#374151",
        va="top",
    )

    hardware_axes = [fig2.add_subplot(gs2[1, i]) for i in range(3)]
    for axh in hardware_axes:
        axh.axis("off")
        axh.add_patch(Rectangle((0, 0), 1, 1, transform=axh.transAxes, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0))

    ax_in, ax_hw, ax_out = hardware_axes
    ax_in.text(0.06, 0.94, "1. Workload inputs", fontsize=13, fontweight="bold", va="top", color="#111827", transform=ax_in.transAxes)
    ax_in.text(
        0.06,
        0.84,
        wrap_bullets([
            f"Max row: {top['agents']} agents doing {top['turns_per_agent']} turns each.",
            f"That is {top['calls']:,} total tool round-trips.",
            f"Context size is {top['context_tokens']:,} tokens.",
            f"Fresh-context miss rate is {LONG_CONTEXT['effective_kv_miss_rate']:.2%}.",
            f"Cold context avoided: {human_tokens(top['cold_context_tokens_avoided_theory'])} tokens.",
        ], width=43),
        fontsize=10.8,
        va="top",
        color="#1f2937",
        linespacing=1.35,
        transform=ax_in.transAxes,
    )

    ax_hw.text(0.06, 0.94, "2. Hardware assumptions", fontsize=13, fontweight="bold", va="top", color="#111827", transform=ax_hw.transAxes)
    ax_hw.text(
        0.06,
        0.84,
        wrap_bullets([
            f"Profile: {HARDWARE['id']}.",
            f"Prefill speed: {HARDWARE['prefill_tokens_per_second_per_replica']:,} tokens/sec per replica.",
            f"Replica size: {HARDWARE['gpus_per_replica']} GPUs.",
            f"GPU price: {chart_text(money(HARDWARE['dollars_per_gpu_hour']))} per GPU-hour.",
            f"Power: {HARDWARE['watts_per_gpu']} watts per GPU.",
            "If any of these change, GPU-hours, dollars, and kWh move linearly.",
        ], width=43),
        fontsize=10.8,
        va="top",
        color="#1f2937",
        linespacing=1.35,
        transform=ax_hw.transAxes,
    )

    ax_out.text(0.06, 0.94, "3. What comes out", fontsize=13, fontweight="bold", va="top", color="#111827", transform=ax_out.transAxes)
    outputs = [
        ("GPU-hours avoided", f"{top['context_gpu_hours_avoided_theory']:.1f}"),
        ("Self-host dollars avoided", f"{chart_text(money(top['context_gpu_dollars_avoided_theory']))}/run"),
        ("Power avoided", f"{top['context_kwh_avoided_theory']:.1f} kWh"),
    ]
    y0 = 0.80
    for idx, (label, value) in enumerate(outputs):
        y_card = y0 - idx * 0.20
        ax_out.add_patch(Rectangle((0.06, y_card - 0.10), 0.88, 0.14, transform=ax_out.transAxes, facecolor="white", edgecolor="#bfdbfe", linewidth=1.0))
        ax_out.text(0.10, y_card, value, fontsize=19, fontweight="bold", color="#1d4ed8", va="center", transform=ax_out.transAxes)
        ax_out.text(0.10, y_card - 0.065, label, fontsize=10.3, color="#374151", va="center", transform=ax_out.transAxes)
    ax_out.text(
        0.06,
        0.22,
        wrap_bullets([
            "Not included: engineering cost, idle capacity, cluster overhead, utilization, or model-quality risk.",
            "This is not an API bill claim unless the provider exposes a way to avoid this prefill work.",
        ], width=43),
        fontsize=10.1,
        va="top",
        color="#1f2937",
        linespacing=1.32,
        transform=ax_out.transAxes,
    )

    fig2.subplots_adjust(left=0.04, right=0.97, top=0.92, bottom=0.08)
    save_figure_all(fig2, out_dir, "value-sweep-hardware")
    plt.close(fig2)


def save_figure_all(fig: Any, out_dir: Path, stem: str) -> None:
    for ext in VISUAL_FORMATS:
        kwargs: dict[str, Any] = {"facecolor": "white", "bbox_inches": "tight"}
        if ext in ("png", "jpg"):
            kwargs["dpi"] = 120
        if ext == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95}
        fig.savefig(out_dir / f"{stem}.{ext}", **kwargs)


def read_levels(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_LEVELS
    with open(path, encoding="utf-8") as f:
        levels = json.load(f)
    if not isinstance(levels, list):
        raise SystemExit("--levels must point to a JSON array")
    for idx, row in enumerate(levels):
        for key in ("level", "agents", "turns_per_agent", "context_tokens"):
            if key not in row:
                raise SystemExit(f"--levels row {idx} missing {key}")
    return levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", help="optional JSON scale ladder")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--markdown", default=str(MARKDOWN))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    rows = [project(level, inputs) for level in read_levels(args.levels)]

    write_csv(rows, out_dir / "value-sweep.csv")
    write_json_payload(inputs, rows, out_dir / "value-sweep.json")
    if not args.no_plots:
        plot(rows, out_dir, inputs)
    write_markdown(inputs, rows, Path(args.markdown))

    print(f"wrote {out_dir / 'value-sweep.csv'}")
    print(f"wrote {out_dir / 'value-sweep.json'}")
    if not args.no_plots:
        for stem in ("value-sweep-headline", "value-sweep-hardware"):
            paths = ", ".join(str(out_dir / f"{stem}.{ext}") for ext in VISUAL_FORMATS)
            print(f"wrote {paths}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
