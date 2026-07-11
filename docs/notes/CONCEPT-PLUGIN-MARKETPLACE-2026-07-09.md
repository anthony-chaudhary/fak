---
title: "Concept: a witness-carrying marketplace for fak plugins"
description: "Design thinking for a discovery + provenance + distribution layer over fak's existing extension seams (compute backends, trajhook scorers, abi Register*, tuiplugin panes) and its self-feature catalog. The differentiator is that every listing ships a re-checkable witness and fak re-verifies it locally before trusting it — a marketplace that does not believe its own listings."
date: 2026-07-09
---

# Concept: a witness-carrying marketplace for fak plugins

Status: concept note / thinking. No new runtime behavior is claimed here. This binds
existing surfaces (the `Register*` seams, `internal/selfquery`, `internal/conformance`,
the capability floor) into one end state and names the gap a marketplace would close.
Parent framing: the popularization concepts in
[`CONCEPT-POPULARIZATION-EPIC-2026-07-02.md`](CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)
(esp. #2 verify-don't-trust/DOS and #5 single-binary drop-in).

## Goal

Let a third party publish an efficiency plugin (a quantization kernel, a cache-eviction
policy, an admission rung), an observability plugin (a trajectory scorer, a console pane,
a verdict kind), or a policy/memory/serving plugin — and let an operator **discover it,
trust it, and install it** without forking the core.

The one-sentence thesis, and the whole reason this is worth building here rather than
copying npm/PyPI/VS-Code-Marketplace:

> **fak already refuses to believe its agents; the marketplace refuses to believe its
> listings.** A listing is not trusted because it is popular or signed — it is trusted
> because it ships a witness the local harness re-checks before the plugin runs. Social
> proof ranks; the witness admits.

This is `dos_verify`'s discipline ("the kernel is the part that doesn't believe the
agents") aimed at the supply chain instead of at a single worker's self-report.

## The tension that shapes everything

fak is **one static Go binary**. Every extension is a source leaf that attaches through a
named `Register*()` seam the kernel *walks but never imports*, blank-imported once in
`internal/registrations`, tier-checked by `internal/architest`, read on the hot path as an
immutable atomic snapshot (the Nth plugin costs the 1st syscall nothing). See
[`EXTENDING.md`](../../EXTENDING.md) and [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

There is **no dynamic loading and no install path today.** "Installing" a leaf means
getting its source into the tree and recompiling. A naive "marketplace" — download a
`.so`/`.wasm`, load it at runtime — collides head-on with the single-binary thesis, the
O(1) snapshot-read contract, and the architest layering gate. So the design cannot be
"VS-Code extensions for fak." It has to respect that fak's in-process seams are
*compile-time* seams.

The resolution is a **two-plane marketplace** (below): the source plane keeps the
compiled-in model and makes "install" a fetch-gate-rebuild; the out-of-process plane uses
the boundary fak *already* adjudicates (MCP) for anything that can live outside the binary.

## What already exists (the marketplace is a layer, not a runtime)

| Surface | Package / entry point | What it already gives the marketplace |
|---|---|---|
| The extension seams | `internal/abi` `Register*` (adjudicator, fastpath, op, verdict, reason, emitter, engine, region/kv/pageout backend, witness resolver, capability); `internal/compute` `Register`; `internal/tuiplugin` `Register` | The attach points. A listing *is* one registration. |
| Scaffolding | `fak new-leaf <name> --tier <t> --register` (`cmd/fak/newleaf.go`) | Green-by-construction plugin skeleton + blank-import wiring. |
| Discovery catalog | `internal/selfquery` `FeatureCard{Kind,Name,Summary,Tags,DetailRef,Effect,RequiresCap,Source,Witness,...}` + `RequestShape`; MCP `fak_feature_query` / `fak_capabilities` | A card model + a query surface. Today every `Source` is first-party. |
| Capability index | `internal/capindex` `CapCard`/`Capability` (content-digest, trigger, `RequiredCaps`, lazy `Resolve`) | Pointer-not-body cards that fault their contents on demand. |
| Correctness+speed gates | the three gates in `EXTENDING.md`: `architest` (plug in) · `compute.CorrectnessClass` + `proofs_witness_test.go` (prove correct) · `shipgate.Evaluate` keep-bit (prove faster) | The admission test. Non-forgeable by construction. |
| Conformance / certification | `internal/conformance` (frozen ABI wire golden + adjudication verdict matrix; `Report.Pass` is one bit); `*_capfloor_conformance_test.go` | A third-party-runnable "does this fork/plugin still meet the floor?" check. |
| Out-of-process boundary | MCP servers (`dos`, `fak`) crossing the adjudicated syscall boundary under the capability floor | A distribution channel that already exists and is already gated. |

The gap is not "invent discovery," "invent trust," or "invent seams." The gap is: the
catalog only knows first-party sources; there is no **provenance** on a card, no **witness
manifest** bundled with a listing, and no **fetch/install** verb.

## The design

### The unit: a Listing = FeatureCard + provenance + witness + conformance + plane

Extend the existing `FeatureCard` (do not invent a parallel type) with the fields a
marketplace needs, all of which fak already produces internally:

```
Listing {
  card         FeatureCard      // Kind/Name/Summary/Tags/Effect/RequiresCap — as today
  plane        source | mcp     // in-process compiled leaf, or out-of-process tool
  publisher    Identity         // signing key + DCO/CLA provenance (who), not (how popular)
  seam         string           // which Register* it attaches to (RegisterAdjudicator, compute.Register, …)
  tier         string           // architest tier it declares (root..integrator)
  witness      WitnessManifest  // the three-gate artifacts, below
  conformance  ConformanceRef   // internal/conformance Report digest + which profile/floor it met
}

WitnessManifest {          // one file, checked locally on install — not trusted from the listing
  architest    green @ <fak version>       // gate 1: layering
  correctness  class + witness digest      // gate 2: Reference max|Δ|=0, or Approx argmax+cosine
  keep_bit     shipgate KEEP + bench artifact ref   // gate 3: strict gain vs honest baseline
  floor        capfloor conformance digest          // cannot drop below the safety floor
}
```

Cards stay pointers, not bodies (the `capindex` lazy-`Resolve` discipline): a search
returns listings; a detail request faults the witness, the source, or the tool schema.

### Two planes

**Source plane (compiled-in) — where efficiency and deep observability live.** Quant
kernels, cache/eviction policies, admission rungs, fastpaths, pageout codecs, trajhook
scorers, verdict kinds, console panes. These *must* be in-process (hot-path cost,
kernel-internal semantics), so a listing here is a **source leaf**, and `fak add <listing>`
is fetch → verify witness → blank-import → rebuild. The single-binary thesis is preserved:
the marketplace is a *source registry with a re-check*, closer to crates.io / Go modules /
Homebrew formulae than to a dynamic-plugin loader.

**Out-of-process plane (MCP) — where anything that can leave the binary goes.** A tool
server, an external scorer, a data-plane exporter. This channel already exists and is
already gated by adjudication + the capability floor, so the marketplace adds only
discovery + provenance + a floor badge, not new runtime. Lower distribution friction, but
it cannot host the in-process seams (a `compute.Backend` over RPC would defeat its purpose).

Picking the plane is not a preference — it is forced by the seam. The `seam` field on the
listing determines the plane.

### The trust model: re-verify locally, don't believe the listing

The marketplace's admission test **is** the three gates, run on the installer's machine
against the shipped witness:

| Marketplace concern | Existing mechanism it reuses | Failure mode it kills |
|---|---|---|
| "Does it even fit the architecture?" | `architest` tier check on `fak add` | a leaf that reaches up a tier / edits the spine |
| "Is it correct?" | `compute.RequireReference` + the witness test re-run | a plausible-but-wrong kernel (Gate 2 catches it) |
| "Is it actually faster?" | `shipgate.Evaluate` keep-bit, non-forgeable | a correct-but-slower "optimization" (Gate 3 reverts it) |
| "Can it exceed its declared powers?" | capability floor + `*_capfloor_conformance_test.go` | an installed plugin that quietly widens its caps |
| "Who shipped it, and is this the bytes they shipped?" | publisher signature + witness digest | a swapped/poisoned artifact; an anonymous drop |

The load-bearing property: none of these is a claim the listing makes about itself. Stars,
downloads, and even the publisher's own witness are *inputs to a local re-check*, never a
substitute for it. A listing with a million downloads whose witness fails `architest`
locally does not install. This is the one thing npm/PyPI/VS-Code structurally cannot do —
they trust the artifact; fak re-derives the trust.

### Supply-chain security is the headline feature, not an afterthought

A plugin marketplace for an *agent kernel* is a supply-chain attack surface — and this
repo already tracks the threat (`RESEARCH-malicious-agent-skills-attention-triage`). fak's
answer falls out of the existing floor: even a malicious installed plugin runs **below**
the capability floor and **behind** adjudication, so its blast radius is bounded by its
declared `RequiresCap` — which the operator saw on the card before installing and which
the floor conformance test proves it cannot widen. "Verify, don't trust" is not marketing
here; it is the reason a marketplace is *safe* to open at all.

## Categories mapped to seams (efficiency, observability, and "similar")

The user's three buckets are already the seam taxonomy:

- **Efficiency** → `compute.Register` (quant/device kernels), `RegisterFastPath`,
  `RegisterRegionBackend` / `RegisterKVBackend` / `RegisterPageOutBackend` (KV / cache /
  page-out), `RegisterAdjudicator` (admission rungs). Trust gate: mostly Gate 3 (keep-bit).
- **Observability** → `internal/trajhook` scorers (`Turn → Finding`), `internal/tuiplugin`
  panes, `RegisterVerdictKind` (open range > 1023), `RegisterEmitter`/steward. Trust gate:
  mostly Gate 2 (deterministic witness) + the floor (a scorer sees turns; can it exfiltrate?).
- **"Similar"** → policy `RegisterReason` (structured refusal vocabulary), memory drivers
  (`internal/memq`), serving `RegisterEngine`. These already lower to `FeatureCard`s in
  `selfquery`, so they are marketplace-ready the day provenance lands.

## The spine (incremental rungs)

Each rung is a shippable increment; none requires dynamic loading.

1. **Provenance on the card.** Add `publisher`, `seam`, `tier`, `plane` to a `Listing`
   view over `selfquery.FeatureCard`. Pure view; no new source of truth. First-party
   listings are just listings whose publisher is `fak`.
2. **The witness manifest format.** Define `WitnessManifest` as the bundle the three
   existing gates already emit (architest result, correctness class + witness digest,
   shipgate keep-bit + bench artifact ref, capfloor digest). A leaf that passes
   `EXTENDING.md`'s gates already has every field.
3. **`fak market` verbs (source plane).** `search` / `show` / `add` / `verify`. `add`
   fetches source, **re-runs the gates locally**, blank-imports, and rebuilds; it refuses
   (structured DOS reason) if any gate reds. This is the fetch-gate-rebuild install.
4. **MCP-plane listings.** Let an out-of-process tool server publish a listing carrying its
   capability-floor badge; discovery + provenance only, since the runtime boundary already
   exists.
5. **The trust ledger.** Bind each installed plugin's witness digest into the journal so
   `dos_verify`-style "did this actually pass its gates on *this* box?" is answerable after
   the fact, not just at install time.
6. **Registry hosting + signing.** The actual index (who hosts it, key management, CLA/DCO
   enforcement). Deliberately last: the value is the re-check, and the re-check works
   against a plain git remote before any hosted index exists.

## Open questions / risks

- **Rebuild-to-install friction (source plane).** `fak add` recompiling the binary is
  correct-but-heavy. Mitigations: a prebuilt-with-witness cache (still re-verified), or a
  curated "featured leaves" set compiled into official releases. Do not solve it by
  reaching for dynamic loading — that trades the whole thesis for convenience.
- **Witness portability.** Gate 2 witnesses are deterministic and cross-platform by design
  (a Mac and a Windows box must agree); Gate 3 bench numbers are *not* portable. The
  keep-bit must be re-measured locally, not trusted from the listing — which is consistent
  with the model but means "faster" is a per-host verdict, and the card should say so.
- **Naming collisions across publishers.** `RegisterOp`/pane/verdict-kind clashes panic at
  startup today (link-time disjointness within one tree). Across publishers this needs a
  namespacing convention on IDs before two third-party listings can coexist.
- **Curation vs. openness.** The re-check makes it *safe* to accept from anyone, but a
  flood of low-value listings still costs discovery. Ranking (social proof) and admission
  (the witness) must stay separate concerns — conflating them reintroduces the trust that
  the design exists to remove.

## The one-line summary

A fak marketplace is not a plugin *loader* — it is a **provenance + witness layer over the
seams and the self-feature catalog that already exist**, whose defining move is to
*re-derive* trust from a local gate re-check instead of *accepting* it from the listing.
That is the only marketplace design consistent with a kernel whose entire identity is
"verify, don't trust."
