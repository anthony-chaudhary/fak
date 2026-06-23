# A local model on the wire, between Claude Code and the Anthropic API

Status: design note, 2026-06-23. Explores where a small local model belongs in the
`fak guard -- claude` path, names the first use case worth building, and gives the
decision framework for when a local-model transform on the wire pays off and when it
does not.

## The thesis

The `fak guard -- claude` gateway already sits on the wire as a disinterested referee,
and it already does heuristic context transforms there: ctxmmu's regex screens decide
what to quarantine, ctxwin's token-budget windowing decides what to demote, and ctxplan's
lexical forecast decides which past turns look relevant. Each of those is a cheap, dumb
predicate that is right most of the time and uncertain at the edges. A small local model
is the cheap semantic brain that upgrades exactly those uncertain edges, and nowhere
else. It earns its place only as a lossy proposer bounded by a witness: the original
bytes stay pinned in the CAS, a gated page-in restores them byte-exact, and the model can
never delete a fact, un-pin a span, or launder a quarantine into an admit.

## When does a local model on the wire make sense?

This is the heart of the note. A local-model transform on the wire is positive ROI only
when ALL of the following hold:

1. Asymmetric economics. The bytes are cheap to process locally and expensive to send
   remotely. The grounded shape of real sessions makes this concrete: tool_result is 92%
   of context, Read alone is 64%, and 77% of that mass is the oldest half of cold
   results. Windowing those (cap@700) is a measured ~2x reduction. When you are trading a
   sub-second local CPU pass against thousands of remote input tokens plus a multi-second
   frontier round-trip, the trade is real.
2. Triage, not authorship. The model emits a routing bit or a rank or a hint, never the
   load-bearing answer. It proposes which reversible representation to inject. Correctness
   must not depend on the model being right.
3. Latency is hidden. The local forward fits in slack: it runs on already-windowed lead
   bytes, or async, or pipelined against the previous turn. It does not sit synchronously
   in front of a warm cache_read where TTFB is already ~760ms.
4. Recoverable and witnessed. Every transform pins the original in the shared CAS, the
   injected stub is a small pointer, and a gated `PageIn` (after a witness `Clear()`)
   restores the exact bytes. A wrong proposal costs one demand-page fault, never a lost
   fact. The divergence class (exact / bounded / lossy) is stamped on the verdict so a
   consumer that needs ground truth knows to page in.
5. A privacy or neutrality dividend. The thing only fak can do because it is NOT the model
   author: keep secrets off the box, screen for injection the provider would otherwise
   ingest, redact before the provider sees anything. The frontier provider structurally
   cannot offer "redaction before itself."

It is NEGATIVE ROI, and you should not ship it, when any of these hold:

- The context is already small or cache-warm. There is no oldest-half mass to demote, so
  the local pass is pure added latency. Worse, demoting a span that sits inside the
  `cache_control` prefix breaks the prefix cache hit and re-bills the prefix, erasing the
  saving the passthrough exists to preserve.
- The transform is unrecoverable or unwitnessed. If a miss leaks bytes off-box with no
  signal (fail-open redaction), the witness proves reversibility of what it caught but
  proves nothing about coverage. That is exactly the silent, unprovable loss the honesty
  culture forbids.
- Local latency exceeds the remote tokens saved. A 1.5B Q8 summary at ~18 tok/s is ~10s
  for 200 tokens. Synchronously, on a small oversize result whose re-Read is cheap, the
  local decode dominates and you lose.
- Determinism is required. A small (1-3B) model is an evadable, foolable predicate. Where
  the existing regex floor or pin priors must hold as a hard invariant, the model can only
  ADD on top of them as advisory, never replace them.

The envelope that makes "cheap local" true on this box: the native CPU Q8/Q4_K path in the
model package (`internal/model/quant_forward.go`, driven by `cmd/fakchat -gguf`), pure-Go
default binary, no llama.cpp, no GPU. The sweet spot is a 1-3B model (1.5B ~18 tok/s
decode, 7B ~8.7 tok/s, 7B is the CPU ceiling on a 36GB box). A 135M-class model decodes at
~177 tok/s, fast enough to sit synchronously for a short output. Prefill is far faster than
decode, so a classify or rank verdict (one decoded token over a short, often
already-windowed input) is dominated by prefill and runs sub-second. The RX 7600 does not
help: it is 7.2x slower than CPU at this size, launch-bound, and reachable only via a
`-tags vulkan` build with `FAK_VULKAN_SPIRV` set. The compute HAL is a dead end for this
work entirely; its GGUF adapter is f32-only and refuses any non-F32 request
(`internal/ggufload/compute_source.go:70-73`), so a quantized triage model cannot ride
`compute.Backend`.

## The recommended first use case: semantic poison screening behind the regex floor

The first thing to build is a semantic injection screen wired as an additive lossy
proposer behind ctxmmu's fail-closed regex floor.

Mechanism. Today `MMU.Admit` calls `ScreenBytes`, a pure regex predicate over a secret
pattern plus a literal list of injection markers ("ignore previous instructions",
"system override", and so on). A social-engineering payload with no literal marker passes
the screen and is admitted as-is. The upgrade inserts ONE additive screen between
`ScreenBytes` and the durability classify: if the regex floor did NOT fire, run the
(capped, leading-window) bytes through a small local triage LM that emits a binary
injection-shaped yes/no. A "yes" routes the SAME bytes into the EXISTING
`quarantineResult(ctx, r, ReasonTrustViolation, body)` path. No new disposition, no new
ABI, no new witness machinery. The model is a pure proposer of additional quarantines: it
can only turn an Allow into a Quarantine, never the reverse. The regex floor runs first
and unconditionally, so the screen stays fail-closed regardless of what the model does or
fails to do.

The exact seam. Inside `MMU.Admit`, immediately after the `ScreenBytes` branch and before
the durability classify (`internal/ctxmmu/mmu.go:139-148`); a positive routes into the
existing `m.quarantineResult(...)` at `mmu.go:168`. `ScreenBytes` itself (`mmu.go:382`)
stays byte-for-byte unchanged. The whole chain is driven from the gateway's
`admitOp -> s.k.AdmitResult` (`internal/gateway/gateway.go:672`), which folds the
priority-10 `ResultAdmitter` registered at `mmu.go:475`. The net-new code is the one call
site from `MMU.Admit` into the model package; there is no in-repo wiring there today.

The witness is inherited, not newly built, which is why this is the cheapest honest rung.
A model-proposed quarantine takes the exact `quarantineResult` path: the original bytes
are pinned verbatim in the shared CAS (`PinResolved`, `mmu.go:177`), the model-visible
payload becomes the tiny `{"_quarantined":true,"id":"q<n>"}` stub (`mmu.go:180-182`), and
page-in is refused until an out-of-band witness calls `Clear(id)` (`mmu.go:264-266`). The
honesty property is strict one-sided additivity: the quarantine-set with the model arm
enabled is a strict superset of the regex-only quarantine-set. A false positive is fully
recoverable (operator `Clear()` + `PageIn` restores the exact bytes); a false negative
degrades to today's behavior (the floor already let it through). The proving test is a
peer of `TestDurabilityTagIsAdditiveOnTransform` (`durability_test.go:76`): assert every
input the floor quarantines is still quarantined with the model arm on, and that a
model-arm quarantine emits a held id that `PageIn` refuses pre-`Clear`.

The economics, and an honest correction. This closes the gap the literal-marker list
leaves open, and that gap matters on the passthrough wire in a specific way. On
`fak guard -- claude` the model ingests `req.Raw` verbatim (`messages.go:218-219` forwards
the original request bytes with `WithRawRequestBody(req.Raw)` to keep the `cache_control`
prefix), so the quarantine byte-rewrite (which targets `req.Messages`, paged out in
`admitInboundResults` at `messages.go:191`) does NOT change the bytes the frontier model
reads. What it DOES do is raise the IFC taint high-water mark that `adjudicateProposed`
reads to refuse the egress tool call (`messages.go:228`). So a model-flagged result still
hardens the proposed-call gate even though the bytes-rewrite is dead on passthrough. Frame
the value as taint-gate hardening, not as "poison never reaches the model" (the in-code
comment at `messages.go:184-190` is optimistic on this point for the passthrough route).
Local cost is one small-model forward per ADMITTED result (only results that survive the
regex floor pay it); cap the screen input to a leading window, since the heavy results are
exactly the ones ctxmmu pages out at `OversizeBytes=4096` anyway.

Why it beats the others. It is the only candidate whose witness, seam, and honesty
property are all already shipped and verified against `mmu.go` and the gateway admit
chain. It reuses an existing disposition with zero new ABI. Its moat fit is the strongest
of the five: a disinterested referee screening for injection the provider would otherwise
ingest is precisely the thing the model author cannot offer about its own input. The
relevance forecaster (candidate B) and the outbound semantic compressor are blocked on an
unbuilt outbound-`req.Raw` transform seam and collide with the cache-prefix invariant;
this candidate needs none of that, because the taint gate it hardens already fires on the
live path.

First-prototype path (smallest shippable rung):

1. Gate the whole arm behind a build tag or env flag, default OFF, like the broader
   local-model integration. The default pure-Go binary stays fast and dependency-light.
2. Load a 1-3B Q8/Q4_K classifier through `cmd/fakchat`'s native path. Prompt it for a
   single binary token over a capped leading window of the candidate body.
3. Insert the one call between `mmu.go:139` and `mmu.go:148`; route a positive into the
   existing `quarantineResult` at `mmu.go:168`.
4. Ship the additive-superset test (floor dominates; model-arm quarantine emits a
   `PageIn`-refused held id) as the gate.
5. Measure end-to-end admit latency on real results before considering default-on. No
   measured classify latency exists yet; only raw tok/s. Treat the model as
   advisory-additive until that number is in hand.

## The multi-modal companion

The user asked about multi-modal explicitly, so here is the honest version. The same
pattern, a lossy-but-recoverable wire transform, applies to pixels as candidate C:
multi-modal screenshot triage as a reversible ctxmmu Transform. A screenshot tool_result
(a base64 image block from a WebVoyager-style run) hits the same registered
`ResultAdmitter`. Instead of always paging raw pixels to an opaque pointer, a proposer
picks one of four reversible collapses: a perceptual-hash match to a prior frame
("unchanged, see frame#k"), OCR/VLM text extraction when text suffices, a crop-to-ROI, or
no-confidence fall-through to today's oversize page-out. Whichever it picks, the original
pixel bytes page into the same CAS via `pageToPointer` (`mmu.go:194-202`), the verdict is
the same `VerdictTransform` carrying a pointer (`mmu.go:151-158`), and `PageIn` after a
witness `Clear` restores the exact frame (`mmu.go:256`). The witness contract is identical;
only the modality of the proposer changes.

It is SECOND, not first, for one blunt reason: the envelope gap. The only arm buildable on
this stack today is the perceptual-hash arm, which uses ZERO local model (it is pure Go
image hashing into the existing Transform/CAS path). The genuinely multi-modal arms (OCR,
VLM collapse-to-text, crop-to-ROI) have no local inference path on this box:
`internal/model` is a text-only causal LM with no vision encoder, and the compute HAL GGUF
adapter is f32-only, so even a quantized vision model could not ride the Backend seam. So
shipping the multi-modal value first would mean shipping a feature with no local model in
it. The right move is to define the proposer interface now so the vision arms slot in
later without changing the witness contract, ship the phash dedup as a reversible
Transform where it pays (screenshot-heavy runs, on the non-passthrough re-marshal wire),
and treat the vision proposer as a later rung gated on a vision encoder that exists in
neither inference stack.

## Runner-up and later candidates

- Relevance triage of conversation history (the ctxplan forecast proposer, #556). SECOND,
  total 18. A small local model authors `Forecast.Intents`
  (`internal/ctxplan/forecast.go:21`) — what the next few turns will ask about — so the
  planner keeps the right cold spans resident and demotes the rest to digests, never
  deleting; a miss costs one demand-page fault (`forecast.go:18`). Makes sense on a long,
  tool-heavy session once an outbound prompt-transform seam exists that re-serializes the
  planned view into the upstream bytes. Blocked today: the flagship passthrough sends
  `req.Raw`, so it demotes nothing the model reads, and demoting a cached-prefix span can
  re-bill the prefix.
- Multi-modal screenshot triage (candidate C, above). LATER, total 16. Ship the phash arm
  only when the workload is screenshot-heavy AND the transform reaches the non-passthrough
  re-marshal wire; the vision arms wait on a vision encoder.
- Outbound tool-result semantic compression, the digest-in-the-stub upgrade. LATER, total
  15. Upgrade ctxmmu's oversize page-out (`mmu.go:151`, today an opaque
  `{_paged, ref, len}` pointer from `pageToPointer`) to carry a model-authored ~200-token
  digest. Makes sense on a non-passthrough upstream where the Transform actually
  re-marshals to the wire (`QuarantineOutboundMessages`, `internal/agent/transcript.go:29`),
  for genuinely large oversize results, with a 135M-class summarizer run async. The digest
  is a non-authoritative hint; `Clear()`+`PageIn` always restores byte-exact. Note it ties
  ctxmmu's page-out on FoldRank with any competing Transform admitter (`registry.go:789`),
  so the digest belongs inside ctxmmu, not as a second admitter racing it.
- Pre-send PII/secret redaction (the privacy compressor). LATER, total 13. Makes sense for
  a compliance-driven buyer who contractually cannot let raw PII reach a third-party model
  and will accept added TTFB. It saves zero tokens by construction, sits on the critical
  outbound hop, and fail-opens on a miss, so it is a compliance floor and
  witness-shape-locker, not a first high-value lever.

## Honest blockers

These come straight from the grounding gaps and bound everything above.

- No vision or OCR path exists. `internal/model` is a text-only causal LM (tokenizer to
  logits). "OCR a tool result" is not buildable on this stack today. Every multi-modal arm
  beyond perceptual hashing is gated on a vision encoder that exists in neither inference
  stack.
- The compute HAL cannot serve a quantized GGUF at all. The adapter is f32-only and
  refuses any non-F32 request (`internal/ggufload/compute_source.go:70-73`); the cpu-ref
  backend is a scalar f32 Reference (`internal/compute/cpuref.go:37-40`). All local triage
  rides the separate native quant path in the model package, not the Backend seam.
- The default pure-Go binary has no GPU. CUDA and Vulkan are `-tags`-gated and excluded
  from `go build ./cmd/fak`; Vulkan also needs `FAK_VULKAN_SPIRV`. On this box the
  realistic default is CPU-only, and the RX 7600 is 7.2x slower than CPU at 1-3B and
  launch-bound (Async and GraphCompile both false in its caps), so there is no GPU triage
  win at the sizes that matter.
- The outbound wire mutates nothing but the stream flag. On the passthrough path the model
  ingests `req.Raw` verbatim (`messages.go:218-219`); the only outbound byte mutation is
  the `stream` flip (`anthropic_stream.go:161-175`). The kernel's inbound-result rewrite
  targets `req.Messages`, which the passthrough never serializes, so the quarantine
  byte-rewrite is a latent dead-path on the live route (it still raises taint that gates
  proposed calls). Any candidate whose value is "shrink or rewrite the outbound prompt"
  needs a new `req.Raw` transform that does not exist, and that transform collides head-on
  with the `cache_control` prefix preservation the passthrough was built for. This is the
  single seam that unblocks the relevance forecaster and the outbound compressor, and
  building it also fixes the dead-rewrite inconsistency.
- No measured local triage latency for a classify or summarize task exists yet, only raw
  decode tok/s. End-to-end cost depends on prefill length, which is fast for short
  results. Measure before defaulting any arm on.

---

How this was derived: five candidate use cases were each deep-designed against the verified
seams above and then scored by an independent skeptic on five axes (value, feasibility-now,
moat-fit, witnessable, latency-hidden). Ranking: poison-screen 19, relevance-forecaster 18,
multi-modal 16, outbound-compression 15, PII-redaction 13. The load-bearing correction —
that an outbound rewrite is dead on the `req.Raw` passthrough and only taint-gate hardening
is live there — was confirmed first-hand against `messages.go:191/218-219/228` and
`messages_stream_passthrough.go`, not taken on an agent's word.
