# Python session-audit parity retirement

Issue #8495 retires `tools/session_audit.py` as an executable, importable oracle. The maintained operational contract is now `fak trajectory audit`; its pinned Claude and Codex fixtures are exercised by `audit_test.go` and its CLI projection by `cmd/fak/trajectory_audit_test.go`.

## Behavior inventory

| Python behavior | Disposition |
|---|---|
| Discover Claude transcript JSONL by root, namespace, age, and subagent scope | **Required.** Go discovers Claude and Codex roots, applies `--since`, records missing roots and source denominators, and classifies Claude fixture/subagent paths explicitly. |
| Deduplicate Claude billed turns by `message.id`; preserve id-less usage rows | **Required.** Pinned Claude fixtures assert exact applied/duplicate counts and token totals. |
| Fold exact input, output, cache-create, and cache-read buckets | **Required.** Cross-harness fixtures assert disjoint exact totals; malformed or unknown usage shapes refuse rather than estimate. |
| Parse Codex cumulative token snapshots and task-boundary resets | **Required.** Codex fixtures cover monotonic snapshots, typed reset boundaries, and fail-closed untyped decreases. |
| Count tools, errors, repeated failures, mutation churn, and hook latency | **Required.** Session/summary fixture assertions and CLI JSONL/Markdown tests cover these operator-facing fields. |
| Rank heavy sessions and compare baselines | **Required.** Deterministic bottleneck and higher-is-worse delta rows are fixture-tested. |
| Emit machine-readable and Markdown reports | **Required.** `--jsonl` and `--md` are the maintained outputs; byte order and schema are fixture-tested. |
| `discover`, `audit`, `deep`, `trend`, and `capture-hooks` Python subcommands | **Omitted.** Discovery and audit are the Go verb. `deep`/`trend` are presentation-specific projections over the versioned JSONL, and hook command wrapping is not transcript auditing; preserving them would keep a second operational CLI. |
| Static model pricing and cross-provider dollar totals | **Omitted.** Prices were editable assumptions, not transcript facts. The Go contract reports exact vendor-neutral token buckets and deliberately refuses a fabricated blended price. |
| Python-only percentile tables, namespace rollups, model-tier labels, gates, and formatting | **Omitted.** They are presentation quirks derivable from JSONL, not the audit contract. Required regressions use typed baseline rows. |
| Import-only helper API used by legacy Python experiments | **Omitted.** It was never a supported audit contract. Operational audit instructions use `fak trajectory audit`; experiments must consume its versioned JSONL rather than import parser internals. |

## Generation retirement evidence

- **Promotion evidence:** the Go verb has cross-harness exact-accounting fixtures, unsupported-shape refusals, deterministic JSONL/Markdown/CLI tests, and the real baseline captured by #8494.
- **Demotion/retirement evidence:** the Python executable, its tests, and its pythongate allowance are removed; the trajectory skill names only the Go verb.
- **Invalidating assumption:** retirement is invalid if a supported transcript shape cannot be represented exactly by `fak-trajectory-audit/1`; that must produce `TRAJECTORY_SCHEMA_REFUSED` and a new Go fixture, not revive the Python parser.
