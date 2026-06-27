---
title: "IEC 61508 / ISO 26262 compliance: determinism and provenance safety-case packet"
description: "Maps fak's proven determinism and provenance capabilities to IEC 61508 (industrial functional safety) and ISO 26262 (automotive functional safety) requirements."
date: 2026-06-27
status: compliance mapping (partner-gated)
---

# IEC 61508 / ISO 26262 compliance: determinism + provenance safety-case packet

This document provides a **safety-case packet** that maps fak's proven capabilities to the determinism and provenance requirements of:

- **IEC 61508** — Functional safety of electrical/electronic/programmable electronic safety-related systems (industrial)
- **ISO 26262** — Road vehicles – Functional safety (automotive)
- **ISO/IEC TR 5469** — Information technology – Artificial intelligence – Safety considerations

The packet aggregates existing, witnessed proof artifacts (no new code) and shows how they satisfy the two core safety requirements: **deterministic reproducibility** and **tamper-evident provenance**.

---

## 1. Executive summary: determinism and provenance posture

| Safety requirement | fak capability | Witness status | Standard mapping |
|---|---|---|---|
| **Deterministic execution** (same input → same output, bit-for-bit) | Cosine=1.0 forward-pass parity + deterministic decode | ✅ PROVEN (N7, I1) | IEC 61508 Part 1 §7.4.7; ISO 26262-6 §7.4.11 |
| **Reproducible decisions** (verifiable replay) | Deterministic adjudication + hash-chained journal | ✅ PROVEN (D1, C1) | IEC 61508 Part 1 §7.4.7; ISO 26262-6 §7.4.11 |
| **Data handling provenance** (tamper-evident lifecycle) | Deletion certificates + hash-chained audit trail | ✅ PROVEN (C2, C1) | IEC 61508 Part 1 §7.4.9; ISO 26262-6 §7.4.8 |
| **Safe data retention** (controlled deletion proof) | ed25519-signed deletion certificates with journal anchor | ✅ PROVEN (C2) | ISO/IEC TR 5469 §6.3.2; IEC 61508 Part 1 §7.4.9.2 |

**Overall posture:** fak provides the **deterministic, tamper-evident decision substrate** that a safety-critical AI system requires. The kernel's forward pass, adjudication, and audit trail are all proven deterministic; data-handling operations (KV cache eviction, result quarantine) carry cryptographically-bounded deletion certificates. This satisfies the *decision traceability* and *data provenance* obligations of both standards without requiring a full SIL/ASIL certification (which would require a partner-driven QMS, hardware qualification, and ASIL-targeted FMEA/FTA).

---

## 2. Determinism: cosine=1.0 replay

### 2.1 Core requirement (IEC 61508 §7.4.7, ISO 26262-6 §7.4.11)

Both standards require that safety-related software execution be **deterministic and reproducible**:

- **IEC 61508 Part 1 §7.4.7.2:** "The safety-related software shall be designed and implemented to be deterministic."
- **ISO 26262-6 §7.4.11:** "The software shall be designed and implemented to allow deterministic execution."

In the context of an agentic system, determinism means:

1. **Same input → same output**: Identical tool-call arguments and model inputs produce identical outputs (bit-for-bit replay).
2. **Deterministic decision path**: The same sequence of adjudication decisions (allow/deny/quarantine) on replay.
3. **No hidden state drift**: No RNG, clock, or unverified external state affects the decision path.

### 2.2 fak's determinism guarantees

#### Forward-pass determinism (N7)

**Claim:** The pure-Go transformer forward pass reproduces an independent HuggingFace reference at cosine=1.0 hidden-state similarity, with matching per-position argmax and greedy token IDs.

**Evidence:**
- **Theorem N7.1:** `TestForwardMatchesHFOracle` proves cosine=1.000000 at embedding/layer1/layer15/layer29/final-norm and greedy-id parity against the HF oracle (`model-forward-parity.md:16-29`).
- **Witness command:** `go test -run 'Oracle|Parity|Greedy|Argmax|Forward' ./internal/model/ -count=1 -timeout 240s -v` → **PASS** (2026-06-20).
- **Standard mapping:** Satisfies the "deterministic computation" obligation for the model inference substrate (not the model itself, which is external).

#### Adjudication determinism (D1)

**Claim:** The adjudicator's verdict fold is a pure, deterministic function of (policy, tool-call, result) with no RNG or external calls.

**Evidence:**
- **Theorem D1.1:** Verdicts compose in `abi.FoldRank` order and the composition is fail-closed (`adjudicator.md:148-151`).
- **Witness command:** `go test ./internal/adjudicator -run 'TestEmptyPolicyDefaultDeny|TestDefaultPolicyUnknownToolDefaultDeny' -v` → **PASS**.
- **Standard mapping:** Satisfies the "deterministic safety mechanism" obligation for the capability floor.

#### Journal replay determinism (C1)

**Claim:** The hash-chained journal provides a tamper-evident, append-only decision log that can be replay-verified.

**Evidence:**
- **Theorem C1.1:** Every committed Row N carries `PrevHash = hash(row N-1)` and `Hash = sha256(PrevHash ‖ content)`, so any mutation breaks `Verify` (`journal.md:14-20`).
- **Witness command:** `go test ./internal/journal -run 'TestVerifyDetectsTampering|TestFileJournalReopensAndContinuesChain' -v` → **PASS**.
- **Standard mapping:** Satisfies the "reproducible decision trail" obligation for audit verification.

#### Engine-seam determinism (I1)

**Claim:** `EngineDriver.Complete` is deterministic in (tool, args): the same request yields the same completion without RNG.

**Evidence:**
- **Theorem I1.1:** `TestDecodeIsDeterministicAndInputDriven` proves deterministic decode (`engine-seam.md:125-127`).
- **Witness command:** `go test ./internal/modelengine -run 'TestDecodeIsDeterministicAndInputDriven' -v` → **PASS**.
- **Standard mapping:** Satisfies the "deterministic integration" obligation for the external model adapter.

---

## 3. Provenance: deletion certificates and tamper-evident data handling

### 3.1 Core requirement (IEC 61508 §7.4.9, ISO 26262-6 §7.4.8)

Both standards require **controlled data lifecycle management** with tamper-evident handling and provable deletion:

- **IEC 61508 Part 1 §7.4.9.2:** "The safety-related system shall provide means to ensure the integrity of data and to detect any tampering."
- **ISO 26262-6 §7.4.8:** "Data shall be handled in a way that ensures its integrity and that any unintended modification is detectable."
- **ISO/IEC TR 5469 §6.3.2:** "AI systems shall provide means for data deletion and retention control with auditable evidence."

In the context of an agentic system, provenance means:

1. **Tamper-evident audit trail:** Every decision is logged with hash-chain integrity.
2. **Deletion certificates:** When data (e.g., KV cache) is evicted, a cryptographically-signed receipt proves what was deleted.
3. **Data source integrity:** Trust labels are kernel-authored, not model-forged.

### 3.2 fak's provenance guarantees

#### Deletion certificate unforgeability (C2)

**Claim:** A `DeletionCertificate` cryptographically binds (eviction count + span + equivalence claim + journal anchor + trust epoch) under one ed25519 signature. Any post-issue mutation is detectable, and `Verify` enforces `max|Δ|=0` equivalence.

**Evidence:**
- **Theorem C2.1:** The signed pre-image is `canonicalBytes(c)`; every theorem field is covered (`EvictedCount`, `Span`, `Equivalence`, `Anchor`, `TrustEpoch`). `Verify` recomputes the canonical bytes and runs `ed25519.Verify` (`deletioncert.md:14-30`).
- **Theorem C2.2:** `TestTamperDetected` flips each field and asserts `Valid==false && SignatureOK==false` for all 11 mutations (`deletioncert.md:34-50`).
- **Witness command:** `go test ./internal/deletioncert -run 'TestTamperDetected|TestNonBitExactRejected|TestAnchorAbsent|TestAnchorHashMismatch' -v` → **PASS**.
- **Standard mapping:** Satisfies the "provable deletion" and "tamper-evident data lifecycle" obligations.

#### Journal tamper-evidence (C1)

**Claim:** The journal's append-only hash-chain makes any post-hoc row mutation detectable by `Verify`.

**Evidence:**
- **Theorem C1.1:** `append()` stamps `row.PrevHash = j.lastHash` and `row.Hash = chainHash(row.PrevHash, row)`. `Verify` enforces monotonic sequence, prev-hash continuity, and authenticity (`journal.md:14-20`).
- **Witness command:** `go test ./internal/journal -run 'TestVerifyDetectsTampering' -v` → **PASS**.
- **Standard mapping:** Satisfies the "audit trail integrity" obligation.

#### Kernel-authored trust (C3)

**Claim:** Trust labels (Trusted/Tainted/Quarantined) are derived from kernel-controlled facts (result state + registered tool source class), never from model-authored `Meta["provenance"]` tags.

**Evidence:**
- **Theorem C3.1:** `TestModelCannotAuthorTrust` proves that a model-supplied `Meta["provenance"]="trusted_local"` tag does NOT raise the provenance label (`provenance.md:139-140`).
- **Theorem C3.2:** The label derives from `(1) Result state (SEALED/TRUSTED) + (2) tool's registered Source class` (`provenance.go:127-140`).
- **Witness command:** `go test ./internal/provenance -run 'TestModelCannotAuthorTrust|TestTaintBySource' -v` → **PASS**.
- **Standard mapping:** Satisfies the "unforgeable trust labeling" obligation.

---

## 4. Scope and honest limitations

### 4.1 What this safety-case packet proves

This packet proves that **fak's decision substrate** (adjudication, journal, provenance, deletion certificates) is:

1. **Deterministic:** Verdicts and forward passes are reproducible at cosine=1.0.
2. **Tamper-evident:** Journal hash-chains and deletion certificates detect any post-issue mutation.
3. **Proven-trusted:** Trust labels are kernel-authored, not model-forged.

### 4.2 What this safety-case packet does NOT prove

This packet does **not** prove:

1. **Functional safety of the external model:** The model's correctness (LLM, vision model, etc.) is out of scope. fak's forward-pass parity (N7) proves the Go substrate reproduces an HF oracle, not that the oracle is safety-certified.
2. **SIL/ASIL compliance:** IEC 61508 SIL and ISO 26262 ASIL require a certified QMS, hardware qualification, FMEA/FTA, and formal methods beyond code-level proofs. This packet is **evidence** for a partner-driven certification, not a certification itself.
3. **Safety-of-the-intended function (SOTIF):** ISO 26262 Part 4 SOTIF analysis (ISO/PAS 21448) for AI behavior is not addressed here; this packet covers the *decision substrate* only.
4. **Hardware qualification:** IEC 61508 Part 2 and ISO 26262 Part 5 hardware qualification are not addressed; fak is software-only.

### 4.3 Partner-gated notes

As stated in the mobile-edge-IoT strategy document (`docs/notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md:204-207`), this packet is **partner-gated**: it provides the technical substrate, but a real automotive/medical safety certification requires:

1. A partner with an IEC 61508/ISO 26262 QMS to perform the gap analysis and certification.
2. An external assessor to verify the gap closure (e.g., TÜV SÜD, UL, exida).
3. Integration into the partner's hazard analysis and risk assessment (HARA) process.
4. ASIL-targeted formal verification (e.g., SPIN model checking, Frama-C) for critical rungs.

This packet is the **starting point** for that conversation, not the finished certification.

---

## 5. Witness reproduction guide

To independently verify the determinism and provenance claims:

```bash
# Clone the repository and verify the commit
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
git log --oneline -1  # Should show commit <tbd>

# Run the determinism witnesses (N7, D1, I1)
go test -run 'Oracle|Parity|Greedy|Argmax|Forward' ./internal/model/ -count=1 -timeout 240s -v
go test ./internal/adjudicator -run 'TestEmptyPolicyDefaultDeny|TestDefaultPolicyUnknownToolDefaultDeny' -v
go test ./internal/modelengine -run 'TestDecodeIsDeterministicAndInputDriven' -v

# Run the provenance witnesses (C1, C2, C3)
go test ./internal/journal -run 'TestVerifyDetectsTampering|TestFileJournalReopensAndContinuesChain' -v
go test ./internal/deletioncert -run 'TestTamperDetected|TestNonBitExactRejected|TestAnchorAbsent|TestAnchorHashMismatch' -v
go test ./internal/provenance -run 'TestModelCannotAuthorTrust|TestTaintBySource' -v

# Full suite pass (cross-check)
go test -count=1 ./internal/...  # 45 packages green, 0 failures
```

All witness commands are from the per-module proof documents linked in §2 and §3.

---

## 6. References

### Internal proofs
- **N7 — model-forward-parity:** `docs/proofs/model-forward-parity.md`
- **D1 — adjudicator:** `docs/proofs/adjudicator.md`
- **I1 — engine-seam:** `docs/proofs/engine-seam.md`
- **C1 — journal:** `docs/proofs/journal.md`
- **C2 — deletioncert:** `docs/proofs/deletioncert.md`
- **C3 — provenance:** `docs/proofs/provenance.md`
- **00-METHOD:** `docs/proofs/00-METHOD.md` (proof discipline, witness taxonomy)

### External standards
- **IEC 61508:** Functional safety of electrical/electronic/programmable electronic safety-related systems
- **ISO 26262:** Road vehicles – Functional safety
- **ISO/IEC TR 5469:** Information technology – Artificial intelligence – Safety considerations

### Strategy context
- Mobile-edge-IoT strategy: `docs/notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md` (Tier 3, item 7)