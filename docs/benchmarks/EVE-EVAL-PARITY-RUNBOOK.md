---
title: "Eve eval parity runbook (GATED)"
description: "The command path and evidence contract for running a Vercel Eve eval suite raw and fak-routed, preserving gate failures and --strict soft scores, before any parity number is claimed."
---

# Eve eval parity runbook (GATED)

Status: **CI-runnable fixture parity harness shipped** (`internal/eveparity`, resolving
[#2605](https://github.com/anthony-chaudhary/fak/issues/2605)); the **upstream `eve` npm
CLI arm remains host-gated**. This is the command path and evidence contract for running a
[Vercel Eve](https://github.com/vercel/eve) eval suite once raw and once routed through
fak's gateway, and proving the two arms agree. The fak-mediation invariant — routing the
identical fixture suite through fak's real gateway preserves every gate verdict, reason,
and `--strict` threshold, and never downgrades a hard gate into a soft observation — is now
witnessed by a Go test and a golden artifact. What is NOT yet witnessed, and makes no claim
here, is a run against the *actual* upstream `eve` binary (it is not installed in CI).

Upstream sources: [Eve evals overview](https://github.com/vercel/eve/blob/main/docs/evals/overview.mdx),
[Eve CLI reference](https://github.com/vercel/eve/blob/main/docs/reference/cli.md).
Parent epic: [#2600](https://github.com/anthony-chaudhary/fak/issues/2600) (eve integration bridge).
Sibling child tickets: [#2601](https://github.com/anthony-chaudhary/fak/issues/2601)–[#2606](https://github.com/anthony-chaudhary/fak/issues/2606).

## The claim boundary (read before quoting anything)

You may quote no "eve parity" claim, no "fak preserves Eve gate failures" claim, until
all of these hold:

1. a fixture Eve eval suite exists in this repo (`t.succeeded`, `t.calledTool`, and a
   deterministic content check) driven by a deterministic mock model — no external model
   key required in CI;
2. the raw arm (the fixture suite evaluated unmediated) and the fak arm (the same suite
   routed through fak's gateway) both produce a result artifact for the same fixture
   suite; and
3. a Go test asserts the two arms' pass/fail status is equal, a deliberately-failing gate
   fails both arms with the original Eve failure reason preserved in fak's result artifact,
   and `--strict` soft-threshold behavior is preserved.

**All three now hold** for the fak-mediation arm: `internal/eveparity` implements the
fixture suite, both arms, the comparator, and the golden witness
(`internal/eveparity/testdata/parity-witness.golden.json`). The one boundary that
remains: the raw arm here is an in-repo evaluator that models the Eve eval semantics, **not**
the upstream `eve` npm CLI (`eve eval --json --junit`), which is not installed in CI. So this
repo witnesses that *fak's mediation is transparent to Eve-shaped gates*; it does not yet
witness a byte diff against the upstream binary's own JSON/JUnit output. That upstream-binary
arm is the named residual — run it on a host with `eve` installed (§2/§3 below).

## Shipped vs residual

| Piece | State | Evidence / residual |
|---|---|---|
| OpenAI-compatible fak gateway | shipped | `fak serve` exposes `/v1/chat/completions`; the fak arm points the fixture suite at a real `gateway.New` proxy. |
| Fixture Eve eval suite (`t.succeeded`, `t.calledTool`, deterministic content check) | shipped | `internal/eveparity` `FixtureSuite()` — three positive cases plus one deliberately-failing hard gate and one strict-sensitive soft score. |
| Deterministic mock-model fixture (no external key needed) | shipped | `internal/eveparity` `FixtureUpstream()` serves scripted OpenAI chat replies; no model key required. |
| Raw-arm runner | shipped | `RunArm("raw", …)` drives the fixture suite unmediated against the fixture upstream. |
| fak-routed arm runner | shipped | `RunArm("fak", …)` drives the identical suite through a real `gateway.New` proxy (session admission + policy + upstream proxy); the fixture `search` tool is admitted through fak's floor. |
| Comparison / parity verdict + witness artifact | shipped | `Compare()` emits `fak.eve-eval-parity-contract.v1`; golden at `internal/eveparity/testdata/parity-witness.golden.json`. |
| Go test asserting equal pass/fail across arms | shipped | `TestEveParityRawVsFak` (both `--strict` modes) + `TestCompareCatchesSilentDowngrade` (adversarial no-downgrade). |
| Raw arm against the **upstream `eve` npm CLI** (`eve eval --json --junit`) | host-gated | `eve` is not installed in CI; §2/§3 give the exact command shape to run on a host that has it. |

## 1. Fixture suite shape

The fixture suite is a small, deterministic Eve eval file covering exactly the cases
#2605 names:

- `t.succeeded` — the fixture agent completes without error.
- `t.calledTool` — the fixture agent calls a named tool at least once.
- a deterministic content check — the fixture agent's final output matches an exact
  string or regex (no model sampling variance, since the model is a fixture).
- one deliberately-failing case — a gate that the fixture agent cannot pass, so both
  arms are proven to *fail* the same way, not just to agree on passing.

Because the model is a deterministic fixture, the suite must not depend on live sampling;
every assertion should be satisfiable by a fixed, scripted response sequence.

## 2. Raw arm — unmediated

Run the official Eve CLI directly against the fixture agent, no fak in the path:

```bash
eve eval --json --junit --strict \
  --agent testdata/eve/fixture-agent \
  --suite testdata/eve/fixture-suite.json \
  --out experiments/eve/raw-result.json
```

This is the baseline result artifact. `--strict` is included because #2605 requires
strict soft-threshold behavior to be preserved identically on both arms.

## 3. fak arm — same suite, routed through fak

Start a fak gateway in front of the same fixture agent's model/tool surface, then point
the identical suite at it:

```bash
fak serve --provider openai --base-url "$UPSTREAM_BASE_URL" \
  --model fixture --addr 127.0.0.1:8080

eve eval --json --junit --strict \
  --agent testdata/eve/fixture-agent \
  --suite testdata/eve/fixture-suite.json \
  --base-url http://127.0.0.1:8080/v1 \
  --out experiments/eve/fak-result.json
```

Task id, fixture suite, model identity, and `--strict` must be identical to the raw arm.
Any difference in the two result artifacts must come from fak's mediation, never from a
changed workload — the same rule the FrontierSWE and LiveCodeBench runbooks hold to.

## 4. Compare and witness

The comparison step (not yet built) must fold both result artifacts into one witness
record, preserving:

- session ids, tool calls, and timing/token metadata from both arms;
- the original Eve gate failure reason on the deliberately-failing case, unchanged by
  fak — a fak result that silently downgrades a hard gate failure into a soft observation
  is the specific regression #2605 exists to catch;
- the `--strict` soft-score threshold verdict from both arms.

Proposed schema, `fak.eve-eval-parity-contract.v1`:

```json
{
  "schema": "fak.eve-eval-parity-contract.v1",
  "issue": 2605,
  "suite": "testdata/eve/fixture-suite.json",
  "raw": {"command": "eve eval --json --junit --strict ...", "result_path": "experiments/eve/raw-result.json"},
  "fak": {"command": "eve eval --json --junit --strict --base-url http://127.0.0.1:8080/v1 ...", "result_path": "experiments/eve/fak-result.json"},
  "parity_verdict": "pending",
  "gate_failure_preserved": null,
  "strict_threshold_preserved": null
}
```

`parity_verdict` stays `"pending"` until a Go test has actually run both arms and compared
them; a doc alone can never set it to `"pass"`.

## 5. The smallest honest win — landed

One fixture suite (the three positive cases plus the one deliberately-failing gate), both
arms run, a Go test asserting equal pass/fail status, and the failing gate's original Eve
reason string byte-identical in both result artifacts. **That point is now witnessed** by
`internal/eveparity`:

- `go test ./internal/eveparity/ -count=1` runs the fixture suite raw and fak-routed (the
  fak arm through a real `gateway.New` proxy) under both `--strict` modes.
- The deliberately-failing gate (`t.calledTool("write")`) fails BOTH arms with the
  byte-identical reason `t.calledTool("write"): tool not called`, marked a `hard_gate` in
  the golden — never a soft observation.
- `TestCompareCatchesSilentDowngrade` proves the comparator *fails* if a fak arm ever turned
  that hard gate into a pass — the golden artifact's whole reason to exist.

Regenerate the golden with `go test ./internal/eveparity/ -run Golden -update`.

The remaining residual is the upstream-binary arm: replace the in-repo evaluator's raw arm
with a real `eve eval --json --junit` run (§2) on a host that has the `eve` CLI, and diff its
JSON/JUnit against fak's artifact byte-for-byte.

## 6. Provenance

- House benchmark discipline this runbook follows: [`BENCHMARK-CONTRACT-MAP.md`](BENCHMARK-CONTRACT-MAP.md),
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md), [`BENCHMARK-GOVERNANCE.md`](../../BENCHMARK-GOVERNANCE.md).
- The fixture harness (`internal/eveparity`) is now in the tree and green in CI; §2/§3's
  upstream-`eve`-binary commands remain proposed shapes (no `eve` CLI in CI), and that
  upstream-binary diff is the residual #2605's harness does not yet capture.
