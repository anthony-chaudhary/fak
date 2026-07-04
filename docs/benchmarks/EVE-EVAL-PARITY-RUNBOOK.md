---
title: "Eve eval parity runbook (GATED)"
description: "The command path and evidence contract for running a Vercel Eve eval suite raw and fak-routed, preserving gate failures and --strict soft scores, before any parity number is claimed."
---

# Eve eval parity runbook (GATED)

Status: **contract written, no fixture harness yet**. This is the command path and evidence
contract for [#2605](https://github.com/anthony-chaudhary/fak/issues/2605) — running a
[Vercel Eve](https://github.com/vercel/eve) eval suite once raw and once routed through
`fak serve`, and proving the two arms agree. No parity number, and no claim that fak
preserves an Eve gate failure, is made until the fixture harness below exists and both
arms have actually run.

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
2. the raw arm (`eve eval --json --junit` against the fixture agent, unmediated) and the
   fak arm (the same eval target routed through `fak serve`) both produce a result artifact
   for the same fixture suite; and
3. a Go test asserts the two arms' pass/fail status is equal, a deliberately-failing gate
   fails both arms with the original Eve failure reason preserved in fak's result artifact,
   and `--strict` soft-threshold behavior is preserved.

None of the three exist yet. This runbook fixes the contract now so the implementation has
one target instead of an ad hoc shape.

## Shipped vs residual

| Piece | State | Evidence / residual |
|---|---|---|
| OpenAI-compatible fak gateway | shipped | `fak serve` exposes `/v1/chat/completions`; an Eve eval target can point at it once the fixture agent is authored. |
| Fixture Eve eval suite (`t.succeeded`, `t.calledTool`, deterministic content check) | pending | no `testdata` fixture suite exists in this repo yet. |
| Deterministic mock-model fixture (no external key needed) | pending | fak already has deterministic model fixtures elsewhere in the tree (e.g. `internal/model` test doubles); none are wired to an Eve-shaped agent yet. |
| Raw-arm runner (`eve eval --json --junit`) | pending | no `fak eve` verb exists (`cmd/fak` has no `eve.go`). |
| fak-routed arm runner | pending | same residual — no adapter routes the eval target through `fak serve` yet. |
| Comparison / parity verdict + witness artifact | pending | schema proposed below (`fak.eve-eval-parity-contract.v1`); nothing produces it yet. |
| Go test asserting equal pass/fail across arms | pending | this is the acceptance witness #2605 asks for; not written. |

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

## 5. The smallest honest win

One fixture suite (the three positive cases plus the one deliberately-failing gate), both
arms run, a Go test asserting equal pass/fail status, and the failing gate's original Eve
reason string byte-identical in both result artifacts. That is the first witnessed point.
Until it lands, this repo makes no Eve eval parity claim.

## 6. Provenance

- House benchmark discipline this runbook follows: [`BENCHMARK-CONTRACT-MAP.md`](BENCHMARK-CONTRACT-MAP.md),
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md), [`BENCHMARK-GOVERNANCE.md`](../../BENCHMARK-GOVERNANCE.md).
- Written with no `eve` CLI installed and no fixture harness in the tree. Every step above
  is a proposed command shape pinned to the issue's own text, not a captured run — that is
  the residual #2605 still owns.
