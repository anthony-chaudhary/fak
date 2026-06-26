# Terminal-Bench 2.1 submission packet — assembly index

Status: `BLOCKED_PRECREDENTIAL` — no result claim, no authority row yet.
Date: 2026-06-26.
Issue: https://github.com/anthony-chaudhary/fak/issues/902
Parent epic: https://github.com/anthony-chaudhary/fak/issues/897

This is the assembly index for the Terminal-Bench 2.1 leaderboard submission
packet. It ties every checked-in campaign artifact to the machine-readable
promotion gate, pins each artifact by SHA-256 so the packet is reproducible from
this file alone, and records exactly which evidence is still missing before any
public number or `BENCHMARK-AUTHORITY.md` row may exist.

It makes **no** Terminal-Bench result claim. `result_claim_allowed` is `false`
across every artifact below, and this index does not add an authority row. The
row template in the last section is a fill-after-evidence form, not a claim.

## The evidence gate (why there is no number here)

The campaign hard fences (#897) block a result claim until command logs, fak
verdict logs, official grader output, a raw-vs-fak compare artifact, and the
submission-packet hashes all exist. As of this date the rehearsal has **not**
cleared the go/no-go bar: the credentialed raw and fak arms have not run, so no
compare artifact and no gateway-traffic witness exist. The packet therefore
stays precredential.

Target bar to clear before submission, from the epic (#897), shown here only as
the bar — **not** as a fak result: the official Terminal-Bench 2.1 leaderboard
lists Codex CLI + GPT-5.5 at **83.4% ± 2.2** at rank 1 (as of 2026-06-26), and
the campaign go/no-go threshold is a stricter **≥ 86.0% mean pass rate** on the
official 2.1 task set, unless the organizers confirm a different ranking
statistic.

## Checked-in artifacts (hash-pinned)

Every artifact below is tracked in this repo. The packet is reproducible from
this index plus these hashes: re-derive them with `sha256sum <path>` and compare.

| Artifact | Role in the packet | SHA-256 |
|---|---|---|
| `experiments/agent-live/terminalbench-official-run-contract-20260626.json` | Official-run contract (machine-readable gate): task selection, both arm commands, score-evidence link, gates, required-before-claim, `result_claim_allowed=false`. | `c89798de6f3205a695268b79e46efdd3d3192eb6fa72fbc07cd4ea7d4da7bd94` |
| `experiments/agent-live/terminalbench-official-run-contract-20260626.md` | Human-readable render of the contract above. | `99f4f595604e8c17081e35008bdb4dce5605f34ee205cf1eb157b8ab49bee65f` |
| `experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json` | Adapter smoke (`SIMULATED_LOCAL_FIXTURE`): raw-vs-fak command-boundary shape over a recorded trace. Adapter evidence only — never a leaderboard number. | `2db95bdaa778e2df0733de41973fc895bfae01ea911ffd5a1a8f8d521c43f31c` |
| `experiments/agent-live/terminalbench-command-boundary-smoke-20260625.md` | Human-readable render of the smoke above. | `a334c9c6a94576f82e006ee19a7192ada36c36a0ba0379dbc793ef51a9375691` |
| `docs/benchmarks/TERMINAL-BENCH-2.1-FAILURE-TAXONOMY.md` | Failure taxonomy + legal retry policy (#901): the closed-vocabulary classifier the compare artifact tallies by. | `768a2eea696bc8dcb1a82034d724c7c89b79ea96e2ddd92a32fd7cb08ed4a52c` |
| `testdata/terminalbench/command_boundary_smoke.json` | Terminal-Bench-shaped candidate suite the contract draws its candidate task ids from. | `73b6481228ded6c092f36883a38386f0d20ee686774d1d8c1f9306c796737e31` |

Re-render the two generated artifacts (no key, no network). Each stamps a fresh
`generated_at` from the wall clock, so a re-render produces an **equivalent**
artifact with a **different** hash — the hashes above pin the committed snapshot;
this command reproduces its shape and content fields, not the byte hash:

```bash
go run ./cmd/terminalbench --contract \
  --out experiments/agent-live/terminalbench-official-run-contract-20260626.json \
  --md  experiments/agent-live/terminalbench-official-run-contract-20260626.md
go run ./cmd/terminalbench \
  --out experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json \
  --md  experiments/agent-live/terminalbench-command-boundary-smoke-20260625.md
```

Verify every committed hash in one pass (this is the byte-exact gate):

```bash
sha256sum \
  experiments/agent-live/terminalbench-official-run-contract-20260626.json \
  experiments/agent-live/terminalbench-official-run-contract-20260626.md \
  experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json \
  experiments/agent-live/terminalbench-command-boundary-smoke-20260625.md \
  docs/benchmarks/TERMINAL-BENCH-2.1-FAILURE-TAXONOMY.md \
  testdata/terminalbench/command_boundary_smoke.json
```

## Missing evidence (required before any result claim)

The contract's `required_before_claim` list is the authority on what is still
missing. None of the following exist yet; each must be checked in and hashed
into this index before the packet leaves precredential state:

- benchmark-native Terminal-Bench task ids for the selected fixed subset;
- the Terminal-Bench image / environment-setup manifest for each selected task;
- the **raw** arm `tb run` directory — command log + official test output over
  those exact task ids;
- the **fak** arm `tb run` directory — command log + official test output over
  the same task ids, model, image, budget, concurrency, and retry policy;
- a fak per-command verdict/evidence log joined to the tb command log and test
  output (join keys: `task_id`, `turn_or_command_index`, `command`, `cwd`,
  `evidence_id`, `state_hash`);
- a raw-vs-fak **compare artifact** reporting benchmark-native solve separately
  from safe resolve, blocked dangerous actions, unnecessary blocks, runtime, and
  cost/token budget, with evidence completeness;
- a **gateway-traffic witness** proving the Dockerized Terminal-Bench agent
  actually routed through the fak gateway (≥1 successful structured model HTTP
  request and ≥1 gateway inference-turn event), so the fak arm's numbers cannot
  be claimed on boundary booleans alone.

The blocking dependencies are the credentialed live rehearsal (#900) and the fak
Codex `/v1/responses` client wire (#925), plus an `OPENAI_API_KEY` for the raw
baseline. This host has none of these, so the packet cannot be completed here.

## Authority row — fill only after evidence

When — and only when — the official evidence above is checked in and hashed, add
one row to [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md). The row must
keep the official Terminal-Bench pass rate **separate** from fak-specific
evidence/cost/safety metrics, so a reader never mistakes a fak-side safety number
for the leaderboard statistic:

| Field | What it carries | Provenance |
|---|---|---|
| Official Terminal-Bench 2.1 pass rate (raw arm) | mean pass^1 over the fixed official subset | **OFFICIAL** — benchmark-native grader output |
| Official Terminal-Bench 2.1 pass rate (fak arm) | mean pass^1 over the same subset, same model/budget | **OFFICIAL** — benchmark-native grader output |
| Safe-resolve / blocked-dangerous / unnecessary-blocks | fak adjudication outcomes on the same runs | **fak-SPECIFIC** — mediated verdict evidence, not a leaderboard number |
| Cost / token budget per task | raw vs fak | **fak-SPECIFIC** — observed, label whose number it is |
| Artifact paths + SHA-256 | contract, raw dir, fak dir, compare, gateway witness | this index |
| Reproduce command | the exact `tb run` + fak gateway invocation | the contract arms |
| Limitations | parity fences, subset scope, statistic used | plain |

Until that row exists, no Terminal-Bench number may appear in `README.md`, the
hero comparison, or any external claim — the same promotion gate the Authority's
existing benchmark rows already enforce.

## Honesty boundary

This index assembles and hash-pins the precredential packet and documents the
gate. It does not run the benchmark, does not produce a compare artifact, and
does not add an authority row. `result_claim_allowed` stays `false` until the
credentialed rehearsal (#900), the gateway witness, and the official grader
output are checked in and hashed into the manifest above.
