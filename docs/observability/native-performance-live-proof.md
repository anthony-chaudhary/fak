---
title: "Native-performance Grafana panel proof"
description: "Reproduce the deterministic panel coverage matrix, distinguish fixtures from live fak-native evidence, and preserve honest unavailable states."
---

# Native-performance Grafana panel proof

Issue #9817 binds the four native-performance dashboards to their contracts,
controlled Prometheus fixtures, deterministic visual witnesses, and a separate live
fak-native Qwen3.8 receipt:

- `fak-native-kernel-performance.json`
- `fak-native-backends.json`
- `fak-native-artifacts.json`
- `fak-native-slo.json`

The deterministic fixture proof is complete only when every query-bearing panel appears
in the matrix and every static row/text panel appears as a non-query surface. Fixture
success is **not** a live performance claim. Live completion additionally requires one
fresh Qwen3.8 execution whose scrubbed receipt names `fak-native`, the `inkernel` runtime,
an explicit native forward path, and zero fallback.

## Reproduce the matrix

Run from the repository root:

```bash
go test ./internal/nativeperfcoverage -count=1 -v
```

The test parses all four dashboard and contract JSON files, extracts each panel target,
parses the controlled Prometheus fixtures, validates every metric and bounded label, and
prints a stable dashboard-by-dashboard and panel-by-panel matrix. The suite then evaluates
all 54 panel targets, both annotations, and all five query variables through `promtool test
rules` over multi-point controlled data. It also audits annotations and dashboard
variables so a query can neither hide outside a panel nor drift from the same metric
contract.

The package test suite includes negative controls for:

- a renamed or unknown metric;
- a missing supervised fixture job;
- invalid or unsupported PromQL;
- stale evidence;
- `vector(0)` or another missing-as-zero coercion;
- an unavailable panel without an honest `UNAVAILABLE`/reason state;
- a non-native or fallback engine; and
- the complete successful matrix.

PromQL is parsed with `promtool`; the controlled fixture query check uses a temporary
Prometheus data directory and never contacts an external service. Validate the committed
fixture and alert files separately as part of the repository gate:

```bash
for fixture in tools/grafana/provisioning/fixtures/fak-native-*.prom; do
  promtool check metrics < "$fixture"
done
promtool check rules tools/grafana/native-performance-alerts.yml
promtool test rules internal/nativeperfslo/testdata/native-performance-alerts.test.yml
```

## What the visual witnesses prove

Existing Grafana captures remain the detailed dashboard witnesses for kernel, backend,
and SLO states. The aggregate populated and unavailable PNGs cover the complete final set,
including the artifacts dashboard:

- `tools/grafana/provisioning/witnesses/fak-native-panel-coverage-populated.png`
- `tools/grafana/provisioning/witnesses/fak-native-panel-coverage-unavailable.png`
- `tools/grafana/provisioning/witnesses/fak-native-panel-coverage-manifest.json`

The populated image is a visual index of every panel whose query produced a real
fixture-derived series in the controlled Prometheus evaluation; `or vector(-1)` is removed
for that populated assertion so the sentinel cannot make a row green. The unavailable
image is the matching coverage index for each panel's required unavailable path. The
detailed Grafana captures prove kernel, backend, and SLO UI states; the aggregate index
adds the artifacts dashboard and binds the complete four-dashboard set to the matrix. The
aggregate images are not pixel captures of Grafana, not screenshots of a live model run,
and not benchmark evidence.

## Fixture versus live evidence

| Evidence | Proves | Does not prove |
|---|---|---|
| Contract + fixture matrix | Query syntax, metric names, supervised jobs, bounded labels, panel coverage, and missing-data semantics | A model executed or a device produced the displayed values |
| Deterministic PNGs | Populated and explicit-unavailable presentation for the complete dashboard set | Grafana availability on a production host or current performance |
| Scrubbed live receipt | One exact Qwen3.8 request executed through fak-native without fallback | A comparative gain, SLO promotion, or general hardware capacity |

The live receipt location is
`tools/grafana/provisioning/witnesses/fak-native-qwen38-live-proof-unavailable.json`.
As of **August 28, 2026**, it honestly records `status: unavailable`: the local native
Metal route refused before load because its current streamed-Q4_K admission requirement
exceeded host memory, the sanctioned GCP probe lacked a refreshable credential, and the
private-lab bridge had no authorized control channel. No live Qwen3.8 result is claimed
from those attempts, and no private hostname, channel, credential, path, or raw log is
committed.

## Complete the live half on sanctioned compute

First select an eligible route using `docs/fleet-compute-nodes.md`. For the public cloud
route:

```bash
python3 tools/gcp_gpu_probe.py --all-tiers
python3 tools/gcp_bench.py \
  --dry-run \
  --tier <ELIGIBLE_TIER> \
  --engine fak-cuda-q8 \
  --hf-repo unsloth/Qwen3.8-27B-GGUF \
  --hf-file Qwen3.8-27B-Q4_K_M.gguf \
  --fak-ref <ISSUE_9817_COMMIT>
```

Continue only when the probe reports an authenticated tier with quota and the dry run
still names Qwen3.8 plus the fak CUDA engine. On an authorized private-lab route, follow
`docs/private-comms-channel.md` to reach the private command authority, run
`dgxbridge doctor`, then `dgxbridge doctor -probe`, and require a fixed-token readback
before dispatching the same native-only workload.

The compute-side serve and bounded request are:

```bash
FAK_Q4K=1 fak serve \
  --addr 127.0.0.1:8137 \
  --engine inkernel \
  --gguf qwen38:27b \
  --model qwen38:27b \
  --backend cuda \
  --context-budget-tokens 4096

curl -fsS http://127.0.0.1:8137/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"qwen38:27b","max_tokens":8,"messages":[{"role":"user","content":"Reply with exactly: Q38"}]}'
```

Add the repository's normal authentication flag/header when the listener is not isolated
to loopback. Accept the live half only when readiness and `/v1/models` identify the exact
model, the response passes the fixed-output check, and the scrubbed receipt records:

- immutable Qwen3.8 artifact revision and SHA-256;
- `execution.engine: "fak-native"`;
- runtime/planner `inkernel`, model owner `fak`, and a native CUDA or Metal forward path;
- `fallback_count: 0`, `fallback_active: false`, and `llama_cpp_used: false`;
- bounded controls, positive wall time, token counts, and exact-output quality;
- unavailable memory/energy as `N/A` with a reason, never numeric zero; and
- no endpoint, hostname, credential, private path, prompt text, or raw log.

Replace the unavailable receipt with that scrubbed artifact and rerun the full matrix.
Until then, the deterministic Grafana contract is proven and the live dependency remains
explicitly pending.
