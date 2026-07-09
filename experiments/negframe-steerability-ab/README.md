# negframe steerability A/B (#3546)

Does affordance-first ("do this") guard-directive framing lift compliance over the same
directive framed as a prohibition ("don't do that")? This directory is a self-contained,
reproducible design for that A/B test, plus a deterministic offline harness that computes a
**modeled** proxy for the effect.

**Read this first: nothing in this directory is a live-model measurement.** There is no GPU and
no model access in this environment. Every number the harness prints carries an explicit
`MODELED / OFFLINE PROXY` label. Per this repo's net-true-value standard, an unwitnessed gain
stays `not yet` — it is never reported as a shipped, observed result. What is shipped here is the
experiment *design* and a runnable *spine* that will accept a real result the day a live witness
exists (see "Upgrading to a live witness" below).

## Hypothesis

> Agent-steer prose that leads with the affordance (states the action to take) drives higher
> guard/directive compliance than the same instruction framed as a prohibition (states the
> action to avoid), because the reader must perform an extra inversion step to recover the
> actionable instruction from a negative frame.

This is the thesis behind `internal/negframe` (the negation-lexicon / reframe-classification
package backing `fak score negframe`, #3540-#3545): steer prose that says "remember to X" is
processed and followed more reliably than "don't forget to X", even though the two are logically
equivalent.

## The two arms

- **Arm A — negative/prohibition-framed.** A guard directive expressed as a prohibition,
  absence, refusal, or hedge (negframe's four categories) — e.g. `Never force-push to main.`
- **Arm B — affordance-first.** The identical instruction, reframed to lead with the action —
  e.g. `Push to main with a fast-forward only.`

Both arms in a pair carry the **same instruction** — same actionable content, same scope, same
consequence — differing only in framing. This is the controlled variable the A/B isolates.

## The corpus

`fixtures.go` holds 16 paired directives (`fixturePair`), each an `(ArmA, ArmB)` pair. The Arm A
idioms are drawn from this repo's own steer-prose register — the guard-directive style found in
`AGENTS.md`, `CLAUDE.md`, the skills, and the recurring cautionary phrasing in the operator's
memory notes (`don't forget to...`, `never...`, `do not...`, `without a witnessed test...`) —
not synthetic filler. Where `internal/negframe`'s reframe rule produces an unambiguous mechanical
rewrite (e.g. `don't forget to X` → `remember to X`), Arm B uses that reframe verbatim (modulo
the sentence-initial capital letter) and the pair's `ReframeIsMechanical` field is `true`. Where
the negation is judgement-tier (negframe deliberately does not auto-rewrite those — see
`internal/negframe/negframe.go`'s rules table) or the mechanical template's bare-verb capture
would read ungrammatically in context, Arm B is a hand-authored affordance-first rewrite instead,
and `ReframeIsMechanical` is `false`.

`TestFixtureCorpusIntegrity` in `main_test.go` enforces the one property the whole experiment
depends on: every Arm A carries at least one real `negframe.Classify` finding (it is genuinely
negative-framed) and every Arm B carries zero (the reframe genuinely removed what the classifier
detects). A broken pair — a reframe that leaked a stray "don't" — would silently invalidate the
comparison, so this is a hard test failure, not a soft check.

Growing the sample is mechanical: add a `fixturePair` literal to `fixtures.go`. The corpus is
capped at a size a maintainer can eyeball in one file (auditability over volume), consistent with
this repo's preference for a small, honest sample over an opaque large one.

## The compliance metric (and why it is explicitly modeled)

There is no live agent to measure real compliance against, so the harness computes a **linear
cost-model proxy**, not a measured rate:

```
compliance(mechanical, judgement) = clamp(
    ceiling - mechanicalCost*mechanical - judgementCost*judgement,
    floor, ceiling)
```

with `ceiling = 0.97`, `mechanicalCost = 0.07`, `judgementCost = 0.04`, `floor = 0.10` (see the
constants and their doc comments in `main.go`). The idea the constants encode: each negation a
reader must invert before finding the actionable instruction costs some compliance, and a
mechanical idiom ("don't forget to") costs more than a blunt judgement-tier one ("never") because
it is a longer clause to hold in mind while parsing the embedded action. **These constants are
not fit to any data — there is none to fit them to.** They are a stated, monotone, bounded
proxy, chosen for plausibility and to keep the arithmetic legible, nothing more. `main.go` says
this again at the top of the file and at the top of every rendered report so the label survives
copy-pasting just the numbers out of context.

`mechanical`/`judgement` are the real, deterministic `internal/negframe.Classify` finding counts
for each arm's text — the one part of this pipeline that is not modeled. Arm B's finding count is
provably `0` for every fixture (enforced by `TestFixtureCorpusIntegrity`), so Arm B always scores
the modeled ceiling; Arm A's score depends on how many negations its phrasing trips.

## The statistical test

The harness reports the **exact one-sided paired sign test** (a distribution-free binomial test)
over the 16 paired deltas: let `k` be the number of pairs where Arm B's modeled compliance
exceeds Arm A's, `n` the number of non-tied pairs, and test `H0: P(B > A) = 0.5` against
`H1: P(B > A) > 0.5` via `p = P(X >= k | X ~ Binomial(n, 0.5))`, computed by direct binomial-
coefficient summation (`signTestPValue` in `main.go` — no external stats dependency, exact for
the small `n` this harness ever sees).

The sign test was chosen over a paired t-test because it makes no normality assumption, handles
the small sample size cleanly, and is the right tool for the fact that this data is **modeled**,
not the outcome of a real random process — a t-test's assumptions about the underlying
distribution would be a claim this harness cannot back.

On the shipped fixture corpus, because Arm B is constructed to have zero negframe findings, the
sign test result is close to deterministic (all pairs favor Arm B, so `p` is tiny). **That is a
property of the corpus construction, not evidence of a real effect** — see the next section for
what would actually test the hypothesis.

## Running it

```
go run ./experiments/negframe-steerability-ab                # human-readable report
go run ./experiments/negframe-steerability-ab -json           # machine-readable result
go run ./experiments/negframe-steerability-ab -selfcheck      # corpus + direction PASS/FAIL, exit 1 on FAIL
go test ./experiments/negframe-steerability-ab/...            # the pinned regression suite
```

`-selfcheck` is the spine: it re-derives the corpus integrity check and the direction check from
the fixtures on every run and prints one clear `SELFCHECK PASS`/`FAIL` line, exiting non-zero on
failure so it is CI-usable.

## Upgrading to a live witness

What is shipped here is offline by necessity, not by choice. To promote this from **modeled** to
**observed**, run the real experiment:

1. **Sample**: for each of the 16 (or an expanded) fixture pairs, construct a realistic agent
   task where the directive is the operative instruction (e.g. embed it in a guard/system prompt
   alongside a task that exercises it — "stamp the commit" needs a commit to make, "never
   force-push" needs a push scenario, etc.).
2. **Randomize arm order** and run each task N times per arm (N ≥ 30 per arm per pair, more if
   the effect is small) through the same model, same temperature, same tool access, so the ONLY
   varying factor across a pair's two arms is the framing.
3. **Score real compliance**, not a proxy: did the agent actually do the instructed thing? Use
   this repo's own deterministic graders as the compliance oracle where they apply — e.g.
   `fak commit --preview` / `dos_commit_audit` for a "stamp the trailer" directive, a guard-
   refusal check for a "never force-push" directive — so the compliance label itself is
   witnessed, not self-reported by the model.
4. **Log every run** to a JSONL file: `{pair_id, arm, run_index, transcript_ref, compliant: bool,
   model, timestamp, git_sha_of_this_harness}`. That log IS the witness.
5. **Test paired binary outcomes properly**: with real per-run compliance labels, replace the
   modeled sign test with **McNemar's test** (the right paired test for a binary outcome measured
   twice on the same unit, here the same task under both framings) or a mixed-effects logistic
   regression with `pair_id` as a random effect if runs are aggregated across many pairs.
6. **Report with provenance `OBSERVED`**, citing the run log path, the model + date, the sample
   size, and the resulting p-value / effect size — replacing every `MODELED` label in this
   directory's output, not merely adding an observed number alongside it.

Until step 4 produces a real log, any effect size reported from this directory must keep the
`MODELED / OFFLINE PROXY` label. That is a `not yet`, by design, and this design doc is exactly
the checkable next step.

## Files

- `README.md` — this design doc.
- `main.go` — the harness: the modeled compliance-cost model, the experiment runner, the CLI
  (`-json`, `-selfcheck`), and the exact paired sign test.
- `fixtures.go` — the 16-pair Arm A / Arm B corpus.
- `report.go` — human-readable and self-check rendering.
- `main_test.go` — the pinned regression suite (corpus integrity, cost-model monotonicity, sign
  test correctness, determinism, thesis-direction regression, selfcheck/report smoke tests).
