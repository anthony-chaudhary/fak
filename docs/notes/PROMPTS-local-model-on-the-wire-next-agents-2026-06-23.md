---
title: "Local-model-on-the-wire: handoff prompts for fak's wirescreen spine"
description: "Handoff prompts to extend fak's witnessed-lossy-proposer spine: rungs for a model-backed poison screener, perceptual-hash frame dedup, and PII redaction."
---

# Handoff prompts: extending the local-model-on-the-wire spine

> Tracking epic: #568  ·  Spine: commit `b63264c`  ·  Design note: `docs/notes/RESEARCH-local-model-on-the-wire-2026-06-23.md`

The spine is the witnessed-lossy-proposer pattern for putting a small LOCAL model on the wire as a careful proposer, never the load-bearing answer. It lives in `internal/abi/semscreen.go` (the `SemanticScreen` seam), `internal/ctxmmu/mmu.go` (the MMU consults the screen chain after the regex floor), and `internal/wirescreen/` (the leaf: registry, env-gated `Active()`, `screenAdapter`, reference `heuristicScreener`, `doc.go` roadmap). Rung 1 is SHIPPED in commit `b63264c` (`feat(wirescreen): add the local-model-on-the-wire semantic-screen spine`). Witness culture: every proposer is bounded by the original pinned in CAS plus a gated `PageIn` that restores bytes byte-exact, so a wrong proposal costs one demand-page fault and never a lost fact; every rung is strictly additive/one-sided (only more careful, never weaker than the floor) and DEFAULT-INERT until a real model has weights AND a MEASURED end-to-end latency number. The outbound blocker (`#555`, ctxplan-owned): on the flagship `fak guard -- claude` Anthropic passthrough the upstream gets `req.Raw` verbatim (`gateway/messages.go:219`, `WithRawRequestBody`, to preserve the `cache_control` prefix), so any byte-rewrite rung changes nothing the model reads on the live route — only taint-gate hardening is live there. Repo rules: work directly on the trunk (`main`), commit by explicit path (`git commit -s -- <paths>`, never `git add -A`), end the ship commit with a `(fak <leaf>)` trailer, and ship only once `make ci` is green.

---

## Rung 1 — Model-backed semantic poison screener (issue #569)

You are filling the already-shipped semantic-screen seam with a real local model. This is the single highest-leverage next step: the seam exists and is wired; you add a model-backed `Screener` behind a gate.

**The exact seam.** The leaf is `internal/wirescreen/`. Implement a new `Screener` (interface in `internal/wirescreen/wirescreen.go`: `Name() string`; `Flag(ctx, body []byte, tool string) (bool, string)`) backed by the native CPU model path (`internal/model`, driven the way `cmd/fakchat -gguf` drives it). Register it under a name (e.g. `"model"`) via `Register(name, s)` in `wirescreen.go`'s `init`, selectable through `FAK_WIRE_SCREEN=model`. The existing `screenAdapter` already bridges the selected `Screener` to `abi.SemanticScreen.ScreenResult`; a `Flag`→true becomes `abi.ScreenAdvice{Disposition: abi.ScreenQuarantine, ...}`. `ctxmmu.MMU.Admit` consults `abi.SemanticScreens()` after the regex floor `ScreenBytes` (`mmu.go` ~line 143) and routes a `ScreenQuarantine` through the existing `quarantineResult` (inherits CAS-pin `PinResolved` + PageIn-refused-until-`Clear`). Increment is observable via `MMU.Screened()`. Gate the model `Screener` behind a build tag so the pure-Go default binary stays unchanged.

**Witness / acceptance test.** Prove the additive-superset property: every result the deterministic `heuristicScreener` flags, the model `Screener` must also flag (it may only flag MORE) — the model can turn `Allow`→`Quarantine`, never the reverse. Add an integration test in the style of `internal/wirescreen/integration_test.go` (`TestEndToEndWithHeuristicScreen`) that skips unless `FAK_WIRE_SCREEN=model`, and keep `TestDefaultInertRegistersNoABIScreen` green (default-absent must still register NOTHING with the ABI). `make ci` must pass.

**Honest scope.** On the flagship `fak guard -- claude` passthrough the byte-removal is DEAD (the model reads `req.Raw` verbatim); the live value here is taint-gate hardening — a `ScreenQuarantine` raises the IFC high-water mark that `adjudicateProposed` reads at `gateway/messages.go:228`. Actual byte removal reaches the wire only on the NON-passthrough re-marshal path (OpenAI/xAI proxy, mock, local serve) via `QuarantineOutboundMessages` (`internal/agent/transcript.go:29`). State this in the doc and commit message; do not claim outbound shrink on the flagship route.

**Envelope constraint.** Use the native CPU Q8/Q4_K path in `internal/model`, 1-3B sweet spot (a one-token classify verdict over a short/windowed input is sub-second because prefill >> decode). Do NOT use the compute HAL (f32-only, refuses GGUF-quant — `internal/ggufload/compute_source.go:70-73` — a dead end) and do NOT use the RX 7600 (7.2x slower than CPU at 1-3B, launch-bound). NO measured classify latency exists yet — MEASURE end-to-end admit latency on a representative body before proposing any default-on; default-on is blocked until that number exists.

---

## Rung 2 (do this one second — buildable NOW, zero model) — Multi-modal perceptual-hash frame dedup (issue #571)

Build the perceptual-hash arm of multi-modal screenshot triage. It needs NO model and NO vision encoder, so it is the most buildable rung today.

**The exact seam.** Use the same `ctxmmu` oversize `Transform` branch the digest path targets (`pageToPointer` in `internal/ctxmmu/mmu.go`), but inspect bodies that are base64 image blocks. The reversible collapse: perceptual-hash a frame, and when it matches a recently-seen frame, collapse the unchanged duplicate to a pointer while the original stays pinned in CAS (PageIn restores it byte-exact). Wire it as a `wirescreen.Screener` / `Transform` consistent with the existing leaf so it is `FAK_WIRE_SCREEN`-selectable and DEFAULT-INERT. Do NOT attempt the OCR/VLM or crop-to-ROI arms — `internal/model` is text-only with no vision encoder, so those wait on an encoder (note this and stop).

**Witness / acceptance test.** Prove reversibility: a deduped (paged-out) frame restores byte-exact via the CAS PageIn after a witness `Clear`, and a frame that differs (even by the phash threshold) is NOT collapsed. Add a deterministic test over fixed image fixtures; keep the default-inert ABI test green. `make ci` must pass.

**Honest scope.** This is a reversible `ctxmmu` Transform; it removes bytes only on the NON-passthrough re-marshal wire (same `QuarantineOutboundMessages` / re-marshal path as rung 1). On the flagship `fak guard -- claude` passthrough it changes nothing the model reads (`req.Raw` verbatim). Say so.

**Envelope constraint.** Perceptual hashing is pure CPU and dependency-light; keep it in the pure-Go default style (no GPU, no model). Still MEASURE per-frame hash + compare latency on representative screenshot payloads before defaulting on.

---

## Rung 3 — Useful page-out digest (issue #570)

Wire the RESERVED `ScreenDigest` disposition so an oversize-benign result pages out to a model-authored digest instead of an opaque pointer.

**The exact seam.** `abi.ScreenDigest` already exists in the `ScreenDisposition` enum and `ScreenAdvice.Digest` already carries the summary (`internal/abi/semscreen.go:42`, `:51`) — the interface needs NO change. Today an oversize-benign result pages out to an OPAQUE `{_paged,ref,len}` pointer via `ctxmmu`'s oversize `Transform` branch (`pageToPointer`, `internal/ctxmmu/mmu.go`). Make `MMU.Admit` act on a returned `ScreenDigest` advisory: map it onto the existing `Transform` so the stub carries the model-authored ~200-token digest, with the original retained in CAS. Author the digest from a `wirescreen.Screener`-style model proposer on the native CPU path, gated behind a build tag and `FAK_WIRE_SCREEN`.

**Witness / acceptance test.** The original must remain pinned in CAS and `PageIn` (after a witness `Clear`) must restore it byte-exact — the digest is lossy display, never the witness. Test that a `ScreenDigest` advisory produces a digest-bearing stub AND a byte-exact restore; keep the default-inert and ABI-freeze tests green. `make ci` must pass.

**Honest scope.** BLOCKED on `#555` for the flagship route: the digest reaches the model only on the NON-passthrough re-marshal path; on the `fak guard -- claude` passthrough the model reads `req.Raw` verbatim, so the digest is dead there until the `req.Raw` cache-prefix-preserving transform lands. Ship the `ctxmmu`/non-passthrough behavior and the test; mark the flagship value as `#555`-gated.

**Envelope constraint.** Native CPU Q8/Q4_K path in `internal/model`, 1-3B; a digest is a decode-bound generation (longer than a one-token classify), so MEASURE end-to-end digest latency over a windowed input before any default-on.

---

## Rung 4 — Pre-send PII/secret redaction (issue #572)

A compliance floor: a model proposes spans to redact in place (replace with a placeholder) before bytes leave the box. This is the rung most tightly coupled to the outbound seam.

**The exact seam.** This requires a NEW outbound `req.Raw` transform that does NOT exist yet (the redactor must edit the bytes actually sent). That transform is exactly the `#555` work: a `req.Raw` rewrite that preserves the `cache_control` prefix (`gateway/messages.go:219`, `WithRawRequestBody`). The redaction proposal is a `wirescreen.Screener`-style model proposer (or a deterministic floor) emitting spans; it must be additive/one-sided (only redacts, never injects or weakens). DEFAULT-INERT and gated.

**Witness / acceptance test.** The unredacted original stays pinned in CAS so an authorized PageIn restores byte-exact; redaction is reversible at the witness layer. Test that proposed spans are replaced with placeholders on the outbound bytes and that the CAS original is untouched. Keep default-inert + ABI-freeze tests green. `make ci` must pass.

**Honest scope.** This is a compliance floor, NOT a token saver — do not claim context savings. BLOCKED on `#555`: until the cache-prefix-preserving `req.Raw` transform exists, redaction cannot reach the flagship `fak guard -- claude` wire. It can land on the NON-passthrough re-marshal path first (OpenAI/xAI proxy, mock, local serve) via the same `QuarantineOutboundMessages`-style re-marshal.

**Envelope constraint.** Native CPU path in `internal/model`, 1-3B; redaction is a classify/span task over a windowed input, so MEASURE end-to-end latency before defaulting on. Deterministic regex PII detection is a valid zero-model floor to ship first.

---

## Pointers

- **Rung 4, relevance forecaster — issue `#556`** (ctxplan-owned). Seam: `ctxplan.Forecast.Intents` (`internal/ctxplan/forecast.go`), a DIFFERENT call site (the context planner, not the MMU). A small model authors the predicted reference strings the next turns will touch; the planner keeps the right cold spans resident and demotes the rest, and a miss costs one demand-page fault. BLOCKED on the outbound seam below to affect the flagship wire.
- **The outbound seam — issue `#555`** (ctxplan-owned): build a `req.Raw` transform that preserves the `cache_control` prefix. It is the single seam that unblocks the digest (rung 3), the forecaster (`#556`), and the redactor (rung 4) on the flagship `fak guard -- claude` route. Until it lands, those rungs are taint-gate-hardening / non-passthrough-only.
