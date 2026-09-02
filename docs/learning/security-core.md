---
title: "fak learning path — L300"
description: "A staged part of the fak learning path, split out of LEARNING-PATH.md so each stage stays a bounded read."
---

# L300 — The Security Core: the in-process default-deny floor and the write-time wall

**Stage 3 of the path** · prev: [the staging page](../../LEARNING-PATH.md) · next: [L400 — The Performance Core](performance-core.md) · back to the [overview and L100–L200](../../LEARNING-PATH.md)

To keep each stage a bounded read, the path ships as this page (overview plus
L100–L200) plus five staged parts under `docs/learning/`. The course numbers,
prerequisites, and checkpoints are continuous across the parts:

- [L300 — The Security Core](security-core.md) — the reference monitor, the policy lifecycle, and the enforcement rungs.
- [L400 — The Performance Core](performance-core.md) — why agents stress the cache and the addressable-eviction answer.
- [L500 — Serving, Integration, and the In-Kernel Model](serving-integration.md) — running and hardening `fak serve`, repointing real agents, the in-kernel model.
- [L600 — Mastery](mastery.md) — benchmarks, honesty discipline, and extending the kernel.
- [The shipped-surface appendix](appendix-shipped-surface.md) — the wrap-up plus the full operator/contributor/package map.


## L300 — The Security Core: the in-process default-deny floor and the write-time wall

**Theme.** The reference monitor, the policy lifecycle, the rungs (preflight, plan-CFI, witness, stewards, rate-limit, escalation), the write-time result gate, canonicalization, IFC, provenance, durability, and code-linting at the same boundary.

**Who joins here.** A security engineer, or anyone who has the Foundations and wants the actual enforcement machinery. Join here if you already understand the KV cache, fail-closed/default-deny, the proofs method, and content addressing, and want to learn how fak adjudicates calls and quarantines results.

**Assumes you can already pass:** **FAK 105**, **FAK 207**.

| Course | Hard prerequisites |
|---|---|
| **FAK 301** — Policy in the Kernel: The First Flip | **FAK 103**, **FAK 207** |
| **FAK 302** — What the Capability Floor Does and Does NOT Bound | **FAK 301** |
| **FAK 303** — The Default-Deny Adjudicator and Closed Refusal Vocabulary | **FAK 301** |
| **FAK 304** — Policy Manifests: Dump, Edit, Check, Load | **FAK 303** |
| **FAK 305** — Preflight Ladder and Grammar Argument-Repair | **FAK 303** |
| **FAK 306** — Plan Control-Flow Integrity (plan-CFI) | **FAK 303** |
| **FAK 307** — The Require-Witness Rung: Effect Verification | **FAK 303** |
| **FAK 308** — Stewards and the Rate-Limit Governor | **FAK 303** |
| **FAK 309** — Graceful Deny: Escalation to a Declared safe_sink | **FAK 304** |
| **FAK 310** — Context-MMU: The Write-Time Tool-Result Gate | **FAK 301** |
| **FAK 311** — Gate Soundness (Regime D): Idempotence and No Gratuitous Mutation | **FAK 310** |
| **FAK 312** — canon: The De-Obfuscating Canonicalizer | **FAK 311** |
| **FAK 313** — normgate: Canonicalize-and-Rescan and Its Honest Limit | **FAK 312** |
| **FAK 314** — IFC: The Taint Lattice and Provenance-Keyed Non-Interference | **FAK 313** |
| **FAK 315** — Provenance: The Model Cannot Author Its Own Trust | **FAK 314** |
| **FAK 316** — Durability Classes and the Expire-by-Default Write Gate | **FAK 203**, **FAK 303**, **FAK 310** |
| **FAK 317** — Hash-Chained Tamper-Evident Audit Journal | **FAK 207** |
| **FAK 318** — codelint: Validating Agent-Written Code at the Same Boundary | **FAK 310** |

### FAK 301 — Policy in the Kernel: The First Flip

**Prerequisites:** **FAK 103**, **FAK 207**

**You'll be able to:**
- Explain why 'the model can't talk past the gate' and 'the default is closed' are properties of WHERE the code runs, not how smart the check is
- Distinguish a fail-closed in-process check from a fail-open out-of-process recognizer
- Sketch which tools in a sample floor are allow-listed and which irreversible ones are deliberately left off

**Read:** [`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md), [`POLICY.md`](../../POLICY.md)

**Lab:**
```bash
go run ./cmd/fak policy --dump  # read the floor; sketch which tools are allow-listed and which irreversible ones are left off (see TestFoldDefaultDenyEmptyPolicy / TestNoOsExecOnHotPath)
```

**Checkpoint:** Explain why 'the model can't talk past the gate' and 'the default is closed' are properties of one address space with no IPC, not of how smart the check is. Name the two independent gates an attacker must beat.

### FAK 302 — What the Capability Floor Does and Does NOT Bound

**Prerequisites:** **FAK 301**

**You'll be able to:**
- Distinguish structural enforcement (refusing a tool NAME) from heuristic detection (argument regex, result flagging)
- Show why allow-listing Bash permits Bash{rm -rf /} and why arg-regex denies are reword-evadable
- State the durable fix: keep irreversible tools off the allow-list

**Read:** [`docs/explainers/policy-in-the-kernel.md`](../explainers/policy-in-the-kernel.md)

**Lab:**
```bash
Given a policy that allow-lists Bash with an RE2 deny on 'rm -rf', invent three rewordings the regex would miss; then state the structural fix (don't allow-list the irreversible tool at all).
```

**Checkpoint:** Classify each as structural or heuristic: (a) refusing an unallowed tool name, (b) the capability deny on the call side, (c) flagging a poisoned result, (d) the result-side quarantine DECISION. State which is the evadable part.

### FAK 303 — The Default-Deny Adjudicator and Closed Refusal Vocabulary

**Prerequisites:** **FAK 301**

**You'll be able to:**
- Explain why an empty policy denies everything and why an arg predicate can never produce an Allow
- State the FoldRank of Deny vs Allow and what happens to an unknown verdict kind
- List several of the 12 reason codes and say which deny is the structural floor (DEFAULT_DENY) vs a policy-pattern deny (POLICY_BLOCK)

**Read:** [`docs/proofs/adjudicator.md`](../proofs/adjudicator.md), [`POLICY.md`](../../POLICY.md), [`examples/adjudication-demo/README.md`](../../examples/adjudication-demo/README.md)

**Lab:**
```bash
go test ./internal/adjudicator/ -count=1 -run 'TestEmptyPolicyDefaultDeny|TestDefaultPolicyUnknownToolDefaultDeny|TestArgPredicatesAreRestrictOnly' -v && fak policy --check policy.json
```

**Checkpoint:** Explain why an empty policy denies everything and why an arg predicate can never Allow. Name the FoldRank of Deny vs Allow, what happens to an unknown verdict kind, and why every deny must cite a code from the fixed vocabulary.

### FAK 304 — Policy Manifests: Dump, Edit, Check, Load

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Explain what makes the loader fail-loud (DisallowUnknownFields, unknown-reason abort) and why that prevents silently loosening the floor
- Show that dump -> check round-trips losslessly
- Ship different floors (coding agent, ops bot, support agent) against the same binary

**Read:** [`POLICY.md`](../../POLICY.md), [`docs/proofs/policy.md`](../proofs/policy.md)

**Lab:**
```bash
fak policy --dump > policy.json && fak policy --check policy.json && fak preflight --policy policy.json --tool delete_account --args '{}'
```

**Checkpoint:** What makes the loader fail-loud and why does that prevent silently loosening the floor? Show that dump->check round-trips losslessly.

### FAK 305 — Preflight Ladder and Grammar Argument-Repair

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Explain why a rung-0 deny stamps RungFailed=0 and never reaches rung 1
- Explain why the grammar rung Defers (not Denies) for a tool with no registered grammar
- Distinguish when the grammar rung Transforms vs Denies

**Read:** [`docs/proofs/preflight.md`](../proofs/preflight.md), [`docs/proofs/grammar.md`](../proofs/grammar.md)

**Lab:**
```bash
go test ./internal/preflight/ -count=1 -run 'TestRung0FailureNeverReachesRung1|TestNegativesRowFields' -v && go test ./internal/grammar/ -count=1 -run 'TestAdjudicatePositionalRepairable|TestAdjudicateNoGrammarDefers' -v
```

**Checkpoint:** Why does a rung-0 deny stamp RungFailed=0 and never reach rung 1? Why does the grammar rung Defer (not Deny) for a tool with no registered grammar, and when does it Transform vs Deny?

### FAK 306 — Plan Control-Flow Integrity (plan-CFI)

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Explain why plan-CFI is opt-in (Defers with no plan declared)
- State what a deviating call returns by default vs in strict mode
- Explain monotone pos advance in Sequence mode and the ROP-gadget analogy

**Read:** [`docs/proofs/plancfi.md`](../proofs/plancfi.md)

**Lab:**
```bash
go test ./internal/plancfi/ -count=1 -run 'TestDeviationEscalates|TestStrictModeDenies|TestSequenceMode|TestConformingCallDefers' -v
```

**Checkpoint:** Why is plan-CFI opt-in and what does a deviating call return by default vs in strict mode? Explain monotone pos advance in Sequence mode and the binary-CFI analogy for an exfil gadget inside an allowed task.

### FAK 307 — The Require-Witness Rung: Effect Verification

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Name the three resolver outcomes (Confirm/Refute/Abstain) and how the kernel folds each
- Explain why a missing git Abstain results in Deny/UNWITNESSED rather than Confirm or Refute
- Corroborate a claimed effect against evidence the agent could not author

**Read:** [`docs/proofs/witness.md`](../proofs/witness.md)

**Lab:**
```bash
go test ./internal/witness/ -count=1 -run 'TestAncestorClaim|TestGitMissingAbstains|TestUnparseableClaimAbstains|TestRealGitAncestor' -v
```

**Checkpoint:** What are the three resolver outcomes and how does the kernel fold each? Why does a missing git Abstain (Deny/UNWITNESSED) rather than Confirm or Refute?

### FAK 308 — Stewards and the Rate-Limit Governor

**Prerequisites:** **FAK 303**

**You'll be able to:**
- Explain why a steward must abstain by default and carry an independently-authored witness
- Explain why check-then-consume ordering makes a denied call cost nothing
- Explain why the limiter is fail-open until configured and denies with RATE_LIMITED (a WAIT)

**Read:** [`docs/proofs/steward.md`](../proofs/steward.md), [`docs/proofs/ratelimit.md`](../proofs/ratelimit.md)

**Lab:**
```bash
go test ./internal/steward/ -count=1 -run 'TestSecretInContext|TestSweepAbstainingStewardNotReported' -v && go test ./internal/ratelimit/ -count=1 -run 'TestQuotaDeniesOverCap|TestDeniedCallConsumesNoBudget|TestInertUntilConfigured' -v
```

**Checkpoint:** Why must a steward abstain by default and carry an independently-authored witness? In the limiter, why is check-then-consume ordering what makes a denied call cost nothing, and why is it fail-open until configured?

### FAK 309 — Graceful Deny: Escalation to a Declared safe_sink

**Prerequisites:** **FAK 304**

**You'll be able to:**
- Explain why the escalation call itself is adjudicated (no side-channel un-sanctioned human-queue tool)
- Explain why the harness, not the kernel, must redact the escalation payload of a denied call
- Route a denied call to the policy's declared safe_sink with a redacted ticket

**Read:** [`examples/escalation-demo/README.md`](../../examples/escalation-demo/README.md)

**Lab:**
```bash
./examples/escalation-demo/run.sh   # build kernel -> serve policy -> catch deny -> route to declared sink -> redacted ticket
```

**Checkpoint:** Why is the escalation call itself adjudicated, and why must the harness (not the kernel) redact the escalation payload of a denied call?

### FAK 310 — Context-MMU: The Write-Time Tool-Result Gate

**Prerequisites:** **FAK 301**

**You'll be able to:**
- Name the three Admit verdicts (Allow / Quarantine / Transform) and which fires for clean, secret-bearing, and small JSON results
- Explain why ctxmmu is the dual of the call-side adjudicator (screening what comes back)
- Explain why PointerMax (2048) is deliberately less than OversizeBytes (4096)

**Read:** [`docs/proofs/ctxmmu.md`](../proofs/ctxmmu.md)

**Lab:**
```bash
go test ./internal/ctxmmu/ -count=1 -timeout 120s -run 'TestAdmit'
```

**Checkpoint:** Name the three Admit verdicts and state which fires for a 6KB clean log line, a body containing an API key, and a 200-byte JSON record. Why is PointerMax deliberately less than OversizeBytes?

### FAK 311 — Gate Soundness (Regime D): Idempotence and No Gratuitous Mutation

**Prerequisites:** **FAK 310**

**You'll be able to:**
- State the two soundness invariants: byte-identical round-trip on Allow, and idempotent page-out
- Explain why re-Admitting a quarantined stub returns Allow without incrementing the quarantine counter
- Identify which property a missing bytes.Equal assertion would leave un-witnessed

**Read:** [`docs/proofs/ctxmmu.md`](../proofs/ctxmmu.md), [`docs/proofs/normgate.md`](../proofs/normgate.md)

**Lab:**
```bash
go test ./internal/ctxmmu/ -count=1 -run 'TestProofPageOutIdempotent|TestProofBenignByteIdentical'
```

**Checkpoint:** Explain why re-Admitting an already-quarantined stub returns Allow and does not increment the quarantine counter (but DOES increment the total call counter). Which property would a missing bytes.Equal assertion leave un-witnessed?

### FAK 312 — canon: The De-Obfuscating Canonicalizer

**Prerequisites:** **FAK 311**

**You'll be able to:**
- Explain why Normalize is idempotent (the property of its output runes that guarantees a fixed point)
- Name one obfuscation family canon folds and the canonical view that catches it
- Explain why a lexical scan must run over the canonical view, not raw bytes

**Read:** [`docs/proofs/canon.md`](../proofs/canon.md)

**Lab:**
```bash
go test ./internal/canon/ -count=1 -run 'TestObfuscatedInjectionCaught|TestNormalizeUndoesObfuscation|TestNormalizeIdempotent_Deterministic' -v
```

**Checkpoint:** Why is Normalize idempotent (what property of its output runes guarantees Normalize(Normalize(x))==Normalize(x))? Give one obfuscation family canon folds and the specific view that catches it.

### FAK 313 — normgate: Canonicalize-and-Rescan and Its Honest Limit

**Prerequisites:** **FAK 312**

**You'll be able to:**
- State the superset theorem (canon flags every body the raw gate flags, plus more) and prove the easy direction informally
- Give an injection string normgate provably does NOT catch (a marker-free paraphrase) and explain why that is an honest limit, not a bug
- Explain why closing the lexical gap needs an IFC/semantic seam

**Read:** [`docs/proofs/normgate.md`](../proofs/normgate.md)

**Lab:**
```bash
go test ./internal/normgate/ -count=1 -run 'TestCanonInjectionSupersetOfRaw_Quick|TestParaphraseEvadesByDesign' -v
```

**Checkpoint:** State the superset theorem and prove the easy direction informally. Then give an injection string normgate provably does NOT catch and explain why that is recorded as an honest limit rather than a bug.

### FAK 314 — IFC: The Taint Lattice and Provenance-Keyed Non-Interference

**Prerequisites:** **FAK 313**

**You'll be able to:**
- Explain why the taint join must be a join-semilattice for the most-restrictive fold to be well-defined
- Trace how a marker-free paraphrase read from an external page still gets its follow-up send_email denied
- Explain declassification as the only sanctioned way tainted data reaches a sink

**Read:** [`docs/proofs/ifc.md`](../proofs/ifc.md)

**Lab:**
```bash
go test ./internal/ifc/ -count=1 -run 'TestParaphrasedExfilBlockedByProvenance|TestForgedSelfTrustCannotEvadeTaint|TestVDSOHitDoesNotLaunderTaint|TestAuthorizeEscape' -v
```

**Checkpoint:** Why must the taint join be a join-semilattice (monotone/commutative/associative/idempotent) for the most-restrictive fold? Trace how a marker-free paraphrase read from an external page still gets its follow-up send_email denied.

### FAK 315 — Provenance: The Model Cannot Author Its Own Trust

**Prerequisites:** **FAK 314**

**You'll be able to:**
- Name the two kernel-controlled facts Taint(c,r) consults and the field it deliberately never reads on a verdict path
- Explain why a forged Meta['provenance'] cannot mint trust and survives only as a forensic signal
- State the honest caveat in Theorem 2: which half of the no-drift claim rests on grep evidence

**Read:** [`docs/proofs/provenance.md`](../proofs/provenance.md), [`docs/proofs/ifc.md`](../proofs/ifc.md)

**Lab:**
```bash
go test ./internal/provenance/ -count=1 -run 'TestModelCannotAuthorTrust|TestTaintBySource|TestRegisterSourceIsHostAuthored' -v
```

**Checkpoint:** What two kernel-controlled facts does Taint(c,r) consult, and which field does it deliberately never read? Explain the honest caveat in Theorem 2: which half of the no-drift claim rests on grep evidence rather than a re-run-on-build assertion?

### FAK 316 — Durability Classes and the Expire-by-Default Write Gate

**Prerequisites:** **FAK 203**, **FAK 303**, **FAK 310**
  ·  **Background:** **FAK 204**

**You'll be able to:**
- Classify every value crossing into durable store as turn/session/bounded/durable at write time
- Justify why an un-classified observation must default to turn (expire), citing the asymmetric error costs
- Locate the attach point: an additive Verdict.Meta['durability'] tag on the ctxmmu Admit seam, fail-closed to 'turn', costing zero frozen-ABI surface
- State precisely what fak claims and does NOT claim vs the named prior art (Tulving, bitemporal SQL:2011, Zhang-Choi 2023, Springdrift, Zep, Cloudflare)

**Read:** [`docs/CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md)

**Lab:**
```bash
Trace the rung-1 bite test by hand: classify 'it's 3pm' and 'the user prefers afternoons' through the ctxmmu gate and state the durability class + promotion verdict each gets; then open internal/abi/types.go and confirm a 'durability' key on the OPEN Meta map does not move TestABIGoldenFreeze.
```

**Checkpoint:** Justify why the default for an un-classified observation must be 'turn' (expire) rather than a centered threshold, citing the asymmetry of the silent false-positive vs the recoverable false-negative; explain why an additive Meta tag (not a new VerdictKind) is the correct attach point; and state the one column where each prior-art system fails to gate on truth-duration at write time.

### FAK 317 — Hash-Chained Tamper-Evident Audit Journal

**Prerequisites:** **FAK 207**

**You'll be able to:**
- Walk through why mutating one content byte trips authenticity AND re-hashing trips the next row's continuity
- Distinguish tamper-evidence from tamper-prevention
- Explain how the durable-flush witness distinguishes per-Emit flush from flush-only-at-Close

**Read:** [`docs/proofs/journal.md`](../proofs/journal.md)

**Lab:**
```bash
go test ./internal/journal/ -count=1 -timeout 120s -run 'TestVerifyDetectsTampering|TestFileJournalReopensAndContinuesChain|TestPerWriteDurableFlush_VerifyWithoutCloseRecoversEveryEmittedRow' -v
```

**Checkpoint:** Walk through why mutating one content byte trips the authenticity check AND why re-hashing to cover it trips the next row's continuity check. Explain how the durable-flush witness distinguishes 'flushed per Emit' from 'flushed only at Close'.

### FAK 318 — codelint: Validating Agent-Written Code at the Same Boundary

**Prerequisites:** **FAK 310**
  ·  **Background:** **FAK 302**

**You'll be able to:**
- Explain why a write_file producing broken code is checkable at the same write-time boundary ctxmmu already runs
- Route a file to the language-server pack that owns its extension and parse/compile-check it
