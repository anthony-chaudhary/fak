# Agent-runtime showcase — a runnable governed code-review agent (#3291)

**Status: BUILT & CAPTURED (gen/next dogfood).** This directory holds a small, cloneable reference
**governed-agent app** that runs a real multi-step task where every proposed tool call is
adjudicated by the **real fak kernel**, plus its **captured live witness**. It runs today, on a
plain dev host, with no model and no network.

- [`showcase.py`](./showcase.py) — the runnable app (a governed *code-review* agent).
- [`EXAMPLE-OUTPUT.md`](./EXAMPLE-OUTPUT.md) — a real captured run (the proof): audit tail + a
  visible DENY + cost accounting, all in one session.
- [`showcase-contract.json`](./showcase-contract.json) — the machine-readable acceptance map:
  which of #3291's sub-claims this app's own run discharges, and which remain for full closure.

## Run it (≈1 second, no model, no network)

```
python experiments/agent-runtime-showcase/showcase.py          # human transcript + summary
python experiments/agent-runtime-showcase/showcase.py --json   # the audit tail as JSONL
```

Requires a `fak` binary (repo-root `./fak[.exe]`, `$FAK_BIN`, or `fak` on PATH) and
`examples/presets/coding-agent-safe.json`.

## What it shows (all four acceptance elements, in one run)

A 7-step code-review trajectory under the least-agency preset `coding-agent-safe.json` (#C4):

1. **Multi-step governed task** — Read → Grep → `go test` → Write, each adjudicated by the kernel (ALLOW).
2. **Live DENY of a dangerous action** — two *distinct kernel* refusal paths fire in the same run:
   `git push --force` → `POLICY_BLOCK` (by `gitgate`), and `delete_account` → `DEFAULT_DENY`
   (by `monitor`, the least-agency default-deny).
3. **Cost cap + accounting** — cumulative spend is tracked per turn and the run **halts** the final
   step because it would breach the 50¢ ceiling (`COST_CAP`, by `budget`).
4. **Audit tail** — an append-only JSON object per adjudication.

See `EXAMPLE-OUTPUT.md` for the captured verdicts (4 ALLOW, 3 DENY across all three governance paths).

## What is real vs. illustrative (the honest scope)

The kernel's ALLOW/DENY is **real and model-independent**: `fak preflight` decides each proposed
call by structure, so the same (policy, call) always yields the same verdict. That is why the
*governance* proof needs no model, no gateway, no GPU — it is fully witnessed here.

What a live model *would* add is only the **proposals**. In this app the review trajectory is
**scripted** (a realistic plan), and the per-turn cost is an **illustrative** model-turn estimate —
so the cost cap demonstrates the enforcement *mechanism*, not a real token bill. Binding the cap to
real model tokens is promotion work (needs the runtime seam, #B1). Everything the kernel decides is
the genuine article.

## Why this lives in `experiments/`, not `examples/` (scope boundary)

#3291's full acceptance places the flagship app **in `examples/`** and wires it into the quickstart
(#D1) and `fak init agent` (#D2). Those are a **different lane** (`examples/`) and a command that
**does not exist yet** (#D2). This lane (`experiments`) lands the runnable + captured **core** as a
dogfood, ahead of those deps — proving the governed-agent showcase works on the *existing* kernel
seam rather than waiting on #B1/#B3. Full closure is the promotion path below.

## Promotion / demotion / invalidating assumption (gen/next discipline)

- **Promotion evidence** (moves toward `now`): (1) promote this app into `examples/agent-review/`
  keeping `coding-agent-safe.json` as its floor; (2) capture ONE run driven by a **live model** on a
  model-capable host, so the proposals and the cost bill are real (the ALLOW/DENY/cap structure is
  already proven here); (3) land `#D2 fak init agent` scaffolding it and link it from the `#D1`
  quickstart. Each of these is independently checkable.
- **Demotion / retirement**: retire this `experiments/` copy once the `examples/` app + a live-model
  capture exist and are wired into `#D1`/`#D2` — the promoted app becomes the canonical showcase and
  supersedes this dogfood.
- **Invalidating assumption** (now tested): the contract originally assumed #B1 (runtime pkg) / #B3
  (SDK) were *hard blockers* for any showcase. This app **falsifies** that for the governance core —
  it is built directly on the existing `fak preflight` seam and its captured run discharges the
  multi-step-task, DENY, cost-cap, and audit-tail elements without #B1/#B3. What genuinely remains
  blocked is only the *live-model* capture (real proposals + real token bill) and the `examples/` +
  `#D1`/`#D2` wiring. If a maintainer rules the showcase MUST sit on the #B1 runtime + #B3 SDK before
  it counts, this dogfood is a stepping stone, not the deliverable.

## Smallest next step

Promote `showcase.py` to `examples/agent-review/` (a different lane), swap its scripted trajectory
for a live model driving the same kernel seam, capture one `EXAMPLE-OUTPUT.md` with real proposals +
a real token-bound cost cap, then land `#D2 fak init agent` pointing at it and link it from the `#D1`
quickstart.
