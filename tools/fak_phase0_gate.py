#!/usr/bin/env python3
"""Validate a Phase 0 fak production benchmark artifact directory.

The checker verifies the machine-readable pieces of the Phase 0 contract:
required raw artifacts, uncapped recorded-workload replay, the full batch curve,
and the batched decode speedup threshold. The clean-node requirement is a
provenance assertion, so callers must pass --clean-node for a canonical run.
"""
import argparse
import json
import os
import sys

import fleet_version


REQUIRED_ARTIFACTS = (
    "node-info.json",
    "q8kernel.txt",
    "modelprof.json",
    "modelbench-q8.json",
    "batchbench-q8.json",
    "fleetbench.json",
    "fleetbench.csv",
    "production-readiness-manifest.json",
)

FULL_BATCHES = (1, 4, 8, 16, 32, 64, 128, 256)


def load_json(path, failures):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        failures.append(f"missing JSON artifact: {rel(path)}")
    except json.JSONDecodeError as exc:
        failures.append(f"invalid JSON in {rel(path)}: {exc}")
    except OSError as exc:
        failures.append(f"cannot read {rel(path)}: {exc}")
    return None


def rel(path):
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def as_int(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value):
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def positive_number(value):
    n = as_float(value)
    return n is not None and n > 0


def require(failures, condition, message):
    if not condition:
        failures.append(message)


def validate_manifest(node_dir, manifest, failures):
    require(
        failures,
        manifest.get("schema") == "fak.production-readiness-node.v1",
        "manifest schema is not fak.production-readiness-node.v1",
    )
    require(failures, bool(manifest.get("host")), "manifest host is empty")
    require(failures, bool(manifest.get("git")), "manifest git rev is empty")
    require(failures, as_int(manifest.get("workload_prefill_cap")) == 0,
            "manifest workload_prefill_cap must be 0 for canonical Phase 0")
    require(failures, as_int(manifest.get("workload_prompt_cap")) == 0,
            "manifest workload_prompt_cap must be 0 for canonical Phase 0")
    require(failures, as_int(manifest.get("batch_workload")) == 0,
            "manifest batch_workload must be 0 for the canonical full batch curve")
    require(failures, as_int(manifest.get("decode_steps")) is not None and as_int(manifest.get("decode_steps")) >= 32,
            "manifest decode_steps must be at least 32")
    require(failures, bool(manifest.get("workload")),
            "manifest must record the workload file")

    listed = set(manifest.get("artifacts") or ())
    for name in REQUIRED_ARTIFACTS:
        path = os.path.join(node_dir, name)
        require(failures, os.path.exists(path), f"missing required artifact: {name}")
        if name != "production-readiness-manifest.json":
            require(failures, name in listed, f"manifest artifacts does not list {name}")


def validate_node_info(node_info, manifest, failures):
    for key in ("host", "os", "arch", "cpu", "cores", "go", "git"):
        require(failures, bool(node_info.get(key)), f"node-info.json missing {key}")
    if manifest.get("host") and node_info.get("host"):
        require(failures, manifest.get("host") == node_info.get("host"),
                "manifest host does not match node-info.json")
    if manifest.get("git") and node_info.get("git"):
        require(failures, manifest.get("git") == node_info.get("git"),
                "manifest git rev does not match node-info.json")


def validate_modelbench(modelbench, manifest, failures):
    require(failures, modelbench.get("precision") == "Q8_0",
            "modelbench-q8.json precision must be Q8_0")
    prefill = modelbench.get("prefill")
    require(failures, isinstance(prefill, list) and any(row.get("tokens") == 256 for row in prefill),
            "modelbench prefill must include the canonical 256-token point")
    decode = modelbench.get("decode") or {}
    require(failures, positive_number(decode.get("tok_per_sec")),
            "modelbench decode tok_per_sec must be positive")

    workload = modelbench.get("workload") or {}
    require(failures, workload.get("schema") == "fak.agent-workload.v1",
            "modelbench workload schema must be fak.agent-workload.v1")
    require(failures, as_int(workload.get("prefill_cap")) == 0,
            "modelbench workload prefill_cap must be 0")
    require(failures, as_int(workload.get("decode_steps_cap")) is not None and as_int(workload.get("decode_steps_cap")) >= 32,
            "modelbench workload decode_steps_cap must be at least 32")
    if manifest.get("workload"):
        require(failures, bool(workload.get("path")),
                "modelbench workload path is empty")

    cases = as_int(workload.get("cases"))
    workload_prefill = modelbench.get("workload_prefill")
    workload_decode = modelbench.get("workload_decode")
    require(failures, cases is not None and cases > 0,
            "modelbench workload case count must be positive")
    require(failures, isinstance(workload_prefill, list) and len(workload_prefill) == cases,
            "modelbench workload_prefill must contain every workload case")
    require(failures, isinstance(workload_decode, list) and len(workload_decode) == cases,
            "modelbench workload_decode must contain every workload case")

    for section_name, rows in (("workload_prefill", workload_prefill), ("workload_decode", workload_decode)):
        if not isinstance(rows, list):
            continue
        for i, row in enumerate(rows):
            if section_name == "workload_decode":
                recorded = as_int(row.get("recorded_prompt_tokens"))
                actual = as_int(row.get("prompt_tokens"))
                require(failures, recorded is not None and recorded > 0,
                        f"modelbench {section_name}[{i}] missing recorded_prompt_tokens")
                require(failures, actual == recorded,
                        f"modelbench {section_name}[{i}] is capped: prompt_tokens={actual}, recorded_prompt_tokens={recorded}")
                require(failures, as_int(row.get("decode_steps")) is not None and as_int(row.get("decode_steps")) >= 32,
                        f"modelbench workload_decode[{i}] decode_steps must be at least 32")
                require(failures, positive_number(row.get("tok_per_sec")),
                        f"modelbench workload_decode[{i}] tok_per_sec must be positive")
            else:
                recorded = as_int(row.get("recorded_tokens"))
                actual = as_int(row.get("tokens"))
                require(failures, recorded is not None and recorded > 0,
                        f"modelbench {section_name}[{i}] missing recorded_tokens")
                require(failures, actual == recorded,
                        f"modelbench {section_name}[{i}] is capped: tokens={actual}, recorded_tokens={recorded}")
                require(failures, positive_number(row.get("tok_per_sec")),
                        f"modelbench workload_prefill[{i}] tok_per_sec must be positive")


def validate_batchbench(batchbench, min_speedup, failures):
    require(failures, batchbench.get("precision") == "Q8_0",
            "batchbench-q8.json precision must be Q8_0")
    require(failures, positive_number(batchbench.get("baseline_b1_tok_per_sec")),
            "batchbench baseline_b1_tok_per_sec must be positive")

    points = batchbench.get("points")
    require(failures, isinstance(points, list) and points,
            "batchbench points must be non-empty")
    batches = {as_int(row.get("batch")) for row in points or []}
    missing = [b for b in FULL_BATCHES if b not in batches]
    require(failures, not missing,
            "batchbench full curve missing batch sizes: " + ", ".join(str(b) for b in missing))
    for i, row in enumerate(points or []):
        require(failures, as_int(row.get("decode_steps")) is not None and as_int(row.get("decode_steps")) >= 32,
                f"batchbench points[{i}] decode_steps must be at least 32")
        require(failures, positive_number(row.get("agg_tok_per_sec")),
                f"batchbench points[{i}] agg_tok_per_sec must be positive")

    peak = batchbench.get("peak") or {}
    speedup = as_float(peak.get("speedup_vs_naive_serial"))
    require(failures, as_int(peak.get("batch")) in batches,
            "batchbench peak batch must appear in points")
    require(failures, positive_number(peak.get("agg_tok_per_sec")),
            "batchbench peak agg_tok_per_sec must be positive")
    require(failures, speedup is not None and speedup >= min_speedup,
            f"batchbench peak speedup_vs_naive_serial must be >= {min_speedup:g}x")


def validate(args):
    node_dir = os.path.abspath(args.node_dir)
    failures = []
    require(failures, os.path.isdir(node_dir), f"node directory does not exist: {node_dir}")
    if not args.clean_node:
        failures.append("clean-node provenance missing: pass --clean-node only for a non-fleet benchmark node")

    manifest = load_json(os.path.join(node_dir, "production-readiness-manifest.json"), failures) or {}
    node_info = load_json(os.path.join(node_dir, "node-info.json"), failures) or {}
    modelbench = load_json(os.path.join(node_dir, "modelbench-q8.json"), failures) or {}
    batchbench = load_json(os.path.join(node_dir, "batchbench-q8.json"), failures) or {}

    if manifest:
        validate_manifest(node_dir, manifest, failures)
    if node_info:
        validate_node_info(node_info, manifest, failures)
    if modelbench:
        validate_modelbench(modelbench, manifest, failures)
    if batchbench:
        validate_batchbench(batchbench, args.min_speedup, failures)

    return {
        "app_version": fleet_version.app_version(),
        "node_dir": node_dir,
        "host": manifest.get("host") or node_info.get("host") or os.path.basename(node_dir),
        "clean_node_asserted": args.clean_node,
        "min_speedup": args.min_speedup,
        "speedup_vs_naive_serial": ((batchbench.get("peak") or {}).get("speedup_vs_naive_serial")
                                    if batchbench else None),
        "passed": not failures,
        "failures": failures,
    }


def main(argv):
    parser = argparse.ArgumentParser(
        description="Validate a fak Phase 0 production-readiness node artifact directory."
    )
    parser.add_argument("node_dir", help="fak/experiments/fleet-nodes/<host> artifact directory")
    parser.add_argument("--clean-node", action="store_true",
                        help="assert this run came from a node not running the live fleet")
    parser.add_argument("--min-speedup", type=float, default=45.0,
                        help="minimum peak speedup_vs_naive_serial required for the 45x claim")
    parser.add_argument("--json", action="store_true", help="emit machine-readable verdict")
    args = parser.parse_args(argv)

    result = validate(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["passed"]:
        print(
            "PASS phase0 gate: "
            f"{result['host']} speedup={result['speedup_vs_naive_serial']:.3f}x"
        )
    else:
        print(f"FAIL phase0 gate: {result['host']}", file=sys.stderr)
        for failure in result["failures"]:
            print(f"  - {failure}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
