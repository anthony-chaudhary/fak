---
title: "Pre-PR verification attestations after AI moved work left"
description: "Study of Stack72's CI argument: preserve independent CI authority while validating commit-bound evidence produced before submission."
---

# Pre-PR verification attestations after AI moved work left (2026-08-31)

**Source:** Stack72, [“AI Broke the Assumptions Behind CI”](https://stack72.dev/ai-broke-the-assumptions-behind-ci/) (retrieved 2026-08-31).

**Disposition:** **PARTIAL → BORROW THE VALIDATOR, NOT THE TRUST.** fak already has structured local validation, policy-floor attestations, commit witnesses, and journal-bound receipts. It does not yet have one common envelope that lets a later gate verify that the exact candidate commit ran the configured required controls and produced complete, fresh, passing results. Filed [#10463](https://github.com/anthony-chaudhary/fak/issues/10463) under witness epic [#2391](https://github.com/anthony-chaudhary/fak/issues/2391).

## What the article argues

The article's central observation is architectural: traditional CI assumes the first trustworthy execution happens after code is submitted, but agentic workflows can author, build, test, and inspect changes before a PR exists. Re-running every control from zero in a remote queue therefore becomes redundant work when the earlier execution can be represented as verifiable evidence.

Its proposed replacement is a **verification attestation** bound to:

- the commit under review;
- the verification configuration or policy that defined the required controls;
- named step outcomes;
- durations and generation metadata.

The receiving CI system becomes a validator. It checks:

1. **completeness** — every required control appears;
2. **freshness** — evidence is recent enough for the policy;
3. **configuration integrity** — the evidence names the expected verification configuration;

The article also recommends **shadow validation** before enforcement and measuring an **escape rate**: how often attested verification says a change is safe while the authoritative CI execution disagrees.

## Direct claims versus our inference

| Kind | Finding |
|---|---|
| Article claim | AI shifts both code generation and first-pass verification before PR submission. |
| Article claim | CI can validate structured verification evidence instead of unconditionally rediscovering every result. |
| Article claim | Commit/config identity, named results, freshness, and completeness are the essential validation dimensions. |
| Article claim | Adoption should begin in shadow mode and compare attestations with the existing CI result. |
| Our inference | The attestation is initially a **cache hint**, not a new trust root, unless its producer and artifact binding are independently witnessed or signed. |
| Our inference | The largest safe win is selective rerun: accept independently grounded controls and rerun missing, stale, failed, or untrusted ones. |
| Our inference | Repository instruction files can define expected work, but cannot prove that work ran; human review policy should remain independent of an author-produced attestation. |

## Where fak already matches

### Structured local validation

`cmd/fak/validate.go:58-100` defines a machine-readable `validateResult` with the candidate tip, explicit owned paths, tested packages, overall/partial/timeout state, elapsed time, per-phase status and duration, skipped phases, and failures. The repo already treats local verification as structured data rather than only terminal prose.

### Re-checkable policy attestation

`cmd/fak/attest.go:151-159` defines a re-checkable attestation document, and `cmd/fak/attest.go:344-350` emits `fak-attestation/v1` with the fak version, generation time, policy SHA-256, probe results, and summary. That surface proves the policy capability floor, not a candidate commit's build/test/review controls.

### Journal-bound receipts

`internal/agent/receipt.go:60-64` binds receipt fields to a decision-journal chain head, and `internal/agent/receipt.go:169` re-folds the journal during verification. This is adjacent integrity machinery, but it does not bind a general validation result to the candidate commit and verification configuration. Issue [#3191](https://github.com/anthony-chaudhary/fak/issues/3191) already tracks mandatory artifact-digest binding for this receipt family.

### Pipeline provenance

Issue [#3193](https://github.com/anthony-chaudhary/fak/issues/3193) already tracks material/product/command-run attestations across pipeline steps, while [#4010](https://github.com/anthony-chaudhary/fak/issues/4010) tracks a digest and generation for the live policy floor. Those are complementary foundations, not duplicates of a verification-attestation validator.

## Dogfood witness

The capability query:

```text
fak capabilities "validate structured pre-PR verification evidence against commit config required controls and freshness without rerunning CI"
```

returned performance-receipt, fleet, context, and policy-floor capabilities, but no general commit/config-bound verification-attestation validator.

The self-index queries:

```text
fak dev index docs  "pre PR verification attestation commit config required steps freshness"
fak dev index verbs "pre PR verification attestation commit config required steps freshness"
fak dev index claims "pre PR verification attestation commit config required steps freshness"
```

returned `fak validate`, `fak attest`, and witness-adjacent documentation, but no common envelope or validator. GitHub dedup searches for `pre-PR verification`, `verification attestation`, `config integrity`, `shadow validation`, and `escape rate` found adjacent issues but no duplicate. Verdict: **PARTIAL**.

## Borrow for fak

The smallest end-to-end spine is a standard-library-only, read-only validator for a versioned envelope such as `fak-verification-attestation/v1`:

```json
{
  "schema": "fak-verification-attestation/v1",
  "commit_sha": "…",
  "verification_config_digest": "sha256:…",
  "producer": {"tool": "fak validate", "version": "…"},
  "generated_at": "…",
  "steps": [
    {"name": "build", "status": "pass", "elapsed_ms": 1200},
    {"name": "affected-tests", "status": "pass", "elapsed_ms": 4300}
  ]
}
```

Given the expected commit, expected configuration digest, required step set, and freshness policy, the validator should emit a closed verdict:

- `VERIFIED`;
- `COMMIT_MISMATCH`;
- `CONFIG_MISMATCH`;
- `INCOMPLETE`;
- `STALE`;
- `RESULT_FAILED`;
- `MALFORMED`.

The first proof is a fixture where a valid two-step envelope passes, then each binding is mutated independently and produces the matching refusal. Required CI wiring is deliberately later.

## Rollout and measurement

1. **Observe:** generate envelopes locally, validate them in CI, but keep existing CI authoritative.
2. **Compare:** record attestation verdict versus actual CI verdict for the same commit and configuration.
3. **Measure:** track false accepts (the article's escape-rate risk), false rejects, missing/stale evidence, and validation latency.
4. **Admit selectively:** only skip a control when its evidence is commit-bound, configuration-bound, complete, fresh, passing, and grounded in an accepted producer identity.
5. **Strengthen identity later:** reuse the operator-supplied Ed25519/DSSE direction from [the witness signing decision](WITNESS-ATTESTATION-SIGNING-DECISION-2026-06-26.md) after the unsigned envelope and deterministic validator prove useful.

## Honest fences

- Evidence produced by the authoring agent is not independent merely because it is JSON.
- A valid attestation cannot prove test adequacy; it proves that the configured controls ran and reported the recorded results.
- Missing, unknown, stale, failed, commit-mismatched, or config-mismatched evidence must fail closed.
- The validator is read-only and deterministic. Running controls and choosing which producers to trust are separate policy decisions.
- Human review remains an independent control where required; an instruction-file check cannot replace it.

## Companions

- [Issue #10463 — commit-bound pre-PR verification-attestation validator](https://github.com/anthony-chaudhary/fak/issues/10463)
- [Witness epic #2391](https://github.com/anthony-chaudhary/fak/issues/2391)
- [Pipeline material/product attestation #3193](https://github.com/anthony-chaudhary/fak/issues/3193)
- [Mandatory artifact-digest binding #3191](https://github.com/anthony-chaudhary/fak/issues/3191)
- [Policy-floor digest and generation #4010](https://github.com/anthony-chaudhary/fak/issues/4010)
- [Witness attestation signing-key decision](WITNESS-ATTESTATION-SIGNING-DECISION-2026-06-26.md)

