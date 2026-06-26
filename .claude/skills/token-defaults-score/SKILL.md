---
name: token-defaults-score
description: One repeatable pass that keeps fak's OUT-OF-THE-BOX token economy amazing — every stacking token-saving method that can SAFELY default is on by default, honestly noted, and locked against regression. Runs the token-saving-defaults scorecard (`fak token-defaults-scorecard`) over the entrypoint source (cmd/fak/guard.go, cmd/fak/serve.go, the gateway Default* constants, and the audited servewiringData rows), turns each HARD defect into a required fix — turn a WITNESSED-safe bounded-loss saver on by default with an honest note, write the missing regression lock, document a dark lever's gate, align the two front doors — and each off high-value saver into a tracked roadmap item (produce the missing witness, then default it). Retires token-defaults-debt worst-first WITHOUT ever shipping an unwitnessed claim, re-measures to PROVE the debt dropped, and commits only the scorecard lane by explicit path. The out-of-the-box-defaults counterpart of persona-score (who lands) and industry-score (the field). Use after a change to a saver default, the serve-wiring rows, or a new token-saving lever, or on a /loop cadence to keep the no-flag experience maximally cheap.
---

# token-defaults-score — keep the out-of-the-box token economy amazing

This is an instance of the **`scorecard` doctrine** (`.claude/skills/scorecard/SKILL.md`) —
read that for the shared five laws and the anti-gaming rule. This file is the per-surface
loop for **fak's out-of-the-box token-saving defaults**: of every stacking token-saving
method fak knows, which are ON by default when a user runs `fak guard -- claude` / `fak
serve` with no flags — and are the high-value, low-loss ones turned on out of the box, each
honestly noted, none able to regress unnoticed?

The headline metric is **token-defaults-debt** (`corpus.token_defaults_debt`). Drive it to
zero, then — per the `score-2x` doctrine — HARDEN: produce the witness that promotes the
next off lever to a must-default.

## The honesty law that makes this scorecard different

A token saver belongs ON by default the moment it is **demonstrably** safe — the operator's
"if it keeps ~the model's view and saves big, default it" rule. **Demonstrably** is the
load-bearing word: turning an *unwitnessed* lever on would ship an unproven claim, which
[`CLAIMS.md`](../../../CLAIMS.md) / [`CLAUDE.md`](../../../CLAUDE.md) forbid. So:

- A **lossless** saver (provider-cache passthrough, tool-floor pruning, vDSO dedup) has no
  fidelity tradeoff → it MUST be on; off is HARD debt.
- A **bounded-loss** saver (history compaction, oversized-result elision) is high-value, but
  it is hard debt to leave off ONLY when a committed **witness** quantifies its
  savings/fidelity tradeoff AND no gate remains. Unwitnessed → it is a SOFT "needs a
  witness," never a forced flip.
- An **opt-in** saver (the ctxplan view) stays off behind a documented gate; its witness
  (e.g. 13.3× fewer resident, 100% exact recall) makes it the next default to turn on once
  the gate (a watched-live session / live-loop wiring) clears.

The scorecard reads the REAL default from the entrypoint source and cross-checks the witness
against committed docs — you cannot drop debt by editing the roster. **Never turn an
unwitnessed lever on to score.** Produce the witness first.

## The loop (the shared five steps, this surface)

1. **Run it** — `go run ./cmd/fak token-defaults-scorecard` (human work-list);
   `--json` (machine payload); `--markdown` (the committed snapshot). Read the **per-lever
   status table** first — it's the "where each saver stands" view (default, witness,
   blocker, noted, locked).

2. **Retire token-defaults-debt worst-first** — by changing reality:
   - `lossless_stack` / `high_value_defaults` defect → turn the saver ON by default (flip the
     `Default*` const / the flag literal in both `guard.go` and `serve.go`), add an honest
     loss note to the flag help, and regenerate the serve-wiring row (`fak serve-wiring --md`)
     so the audited verdict tracks the flip.
   - `default_on_locked` defect → add a real assertion to `cmd/fak/token_defaults_test.go`
     that pins the default (reds if a peer flips it back).
   - `dark_lever_gated` defect → document why the lever is dark + a safe value + the witness
     it needs, in the flag help / const comment.
   - `entrypoint_parity` defect → make `guard.go` and `serve.go` agree, or regenerate the
     stale servewiring verdict.

3. **Weigh the SOFT signals, then stop** — `witness_status` is the roadmap, not a fix: an
   *unwitnessed* high-value lever needs a measurement (build/run the witness — e.g. a
   deterministic shed-ratio replay, or the watched-live dogfood the gate names), and a
   *witnessed-gated* lever needs its gate cleared. Producing a witness is real work; do it
   deliberately, then the next run promotes that lever to a must-default.

4. **Re-measure + prove** — rerun
   `go run ./cmd/fak token-defaults-scorecard --json` and compare the debt against the
   baseline you recorded; regenerate the snapshot
   (`go run ./cmd/fak token-defaults-scorecard --markdown > docs/serving/token-defaults-scorecard.md`);
   run the locks (`go test ./cmd/fak -run TestTokenDefault`).

5. **Commit only the scorecard lane, by explicit path** — never `git add -A`. End the
   subject with the `(fak <leaf>)` trailer (stamp the dominant leaf — `tools` for a
   scorecard-only pass, `gateway`/`cmd` when the same pass flips a default). e.g.
   `git commit -s -F msg -- cmd/fak/token_defaults.go cmd/fak/token_defaults_test.go docs/serving/token-defaults-scorecard.md`

## The anti-gaming rule (this surface)

Retire a defect by changing the binary's real default / writing a real lock / producing a
real witness — never by editing the roster, loosening the witness regex, or pointing a
sentinel at a vacuous test. A `default_on_locked` defect is fixed by a test that actually
asserts the default and reds when it flips, not by a test that merely names the lever. The
one move that turns this into theater is turning a lever on to score before it is witnessed:
that ships exactly the unproven claim the whole repo refuses.
