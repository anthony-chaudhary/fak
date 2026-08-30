---
title: "TensorBuild deep recheck: measurement contracts worth borrowing"
description: "A snapshot-pinned, whole-tree recheck of TensorBuild that separates transferable evidence and native-runtime contracts from TensorRT-specific machinery."
---

# TensorBuild deep recheck: measurement contracts worth borrowing

**Observed:** 2026-08-29  
**Source:** `local://tensor-build-main`  
**Snapshot:** `snapshot-sha256:64986cf8ff942cdcd6178491d3d9af0199c354ce41f998b89ced2b0286f6772d` over 6,012 regular files / 216,644,047 bytes  
**Durable receipt:** `study_5152edd9388ef5a396f526004fa64ccfd0796171db04810d91a49eed606a17b4`  
**Inventory:** [`local-tensor-build-2026-08-29.json`](../research/inventory/local-tensor-build-2026-08-29.json)  
**Prior study:** [`tensor-build-local-study-2026-08-15.md`](tensor-build-local-study-2026-08-15.md)

## Verdict first

TensorBuild remains most useful to fak as a source of **measurement contracts**, not as a
TensorRT implementation donor. This recheck found 13 bounded, previously unowned mechanisms
worth testing in fak: non-vacuous gate-domain receipts; evidence-lineage independence;
cost-normalized evidence acquisition; population-sampling discipline; directional slice floors;
addition-versus-replacement benchmark topology; offered-versus-exercised tool census; post-pass
GPU-clock coverage; anti-splice vendor evidence; functional kernel-selection identity;
claim/null/prior sensitivity arms; evidence-polarity handling for incomplete inputs; and
comparison criteria derived from run identity.

Seven other mechanisms already have exact fak owners, two are already present without follow-up,
and four are deliberately watched or excluded because they assume TensorRT artifacts that fak's
native execution path does not have. The resulting backlog is issues
#10268-#10271 and #10278-#10286. No source code or expressive prose was copied.

## Provenance and license fence

The source directory is not a trustworthy Git checkout. Git resolves an unrelated enclosing
repository, so that revision is not cited. The declared `github.com/anthony-saronic/tensor-build`
remote was unavailable through authenticated repository lookup and anonymous `ls-remote`; issue,
PR, release, discussion, blame, and commit-history claims are therefore **unavailable**, not
empty. No root `LICENSE`, `COPYING`, `NOTICE`, or CLA is present.

Disposition is **INSPIRE-ONLY**: retain independently specified effects, tests, and operating
boundaries; do not transfer implementation or prose. The snapshot digest and per-file SHA-256
anchors below are the only reproducibility identity available.

## Completeness and change audit

The machine inventory indexes 5,972 files / 888 directories / 215,914,591 bytes / 1,482,541 text
lines after deliberately skipping 40 generated files (729,456 bytes) in `internal/build` and
`internal/conn/web/dist`. It classifies 2,309 runtime files, 1,619 tests, 825 documentation files,
and 16 immediate subsystems.

Two independent deep scopes covered the source:

| Scope | Denominator | Reproducibility pin |
|---|---:|---|
| Worldview, docs, results, operator contracts | 3,174 files / 108,356,033 bytes | `scope-sha256:76b52a40616ddf26327b51587ec58007e3dba7863f549921eb683064045480fc` |
| `go.mod`, `internal/`, and `cmd/` runtime/tests | 2,353 files | `scope-sha256:3f37fa97cce240d75d1fbdbd6f529778b4b6318b90d74ad1333dde907f7139b1` |

All 32 source-file hashes cited by the deep readers were independently recomputed and matched.
The prior raw snapshot had 6,358 files / 218,328,563 bytes. Exact inventory comparison found no
substantive subsystem change: the current source is missing only 344 transient `.dos` files
(1,681,440 bytes) and two transient `.fak` files (3,076 bytes). Runtime/test classifier shifts are
inventory-tool classification changes, not evidence of source churn.

## Architecture and worldview

TensorBuild treats proof shape as part of the product:

1. A gate reports the population it could have judged, not only its verdict.
2. Evidence acquisition is separate from later policy adjudication.
3. Agreement is meaningful only across independent evidence roots.
4. Benchmark arms must match how a treatment is deployed and must prove the measurement is
   sensitive enough to say no.
5. Incomplete inputs affect positive observations and absence claims differently.
6. Hardware and vendor reports carry coverage and coherence, not just selected numbers.

The source also gives agents a command-oriented front door through `AGENTS.md`, `llms.txt`, and
one binary surface. That is already consonant with fak and did not justify another API.

## Candidate matrix

`PRESENT`, `PARTIAL`, and `ABSENT` describe current fak coverage. Disposition says whether the
mechanism belongs in the default product, an optional module, a reusable recipe, a watch list, or
the exclusion set.

| Mechanism | Source anchor | Current fak witness | Verdict / disposition | Owner |
|---|---|---|---|---|
| Claim-to-proof artifact reachability | typed repository graph and reachability audit | docs reachability exists; proof-artifact liveness remains open | PARTIAL / DEFAULT | #6876 |
| Work-class × cost × outcome join | token-spend audit | exact joined accounting shipped | PRESENT / DEFAULT | #6874 |
| Scoped negative/inconclusive experiment ledger | typed result identity | screening-only experiment receipts shipped | PRESENT / DEFAULT | #6875 |
| Gate domain and admit-rate non-vacuity | `AGENTS.md:54-74`; `AD-35:204-230` | hook candidates distinguish unreported from zero but omit eligible/admitted distributions | PARTIAL / DEFAULT | #10268 |
| Persist evidence, reproject policy | ground-truth design `234-268` | replayable immutable audit evidence already owned | PARTIAL / DEFAULT | #3978 |
| Derived-lineage independence | ground-truth design `272-308` | run correlation rejects self-agreement but not shared roots or cycles | PARTIAL / DEFAULT | #10269 |
| Information gain per acquisition cost | ground-truth design `312-365` | tool-call control has gain but no hypothesis-bound cost ranking | PARTIAL / DEFAULT | #10270 |
| Powered population audit | `AD-35:8-17,92-122,234-245` | denominators exist; powered sampling contract does not | ABSENT / RECIPE | #10271 |
| Directional per-slice uncertainty | `M9:121-156` | slice sample counts and vetoes exist without asymmetric bounds | PARTIAL / DEFAULT | #10278 |
| Addition versus replacement topology | `M8:102-120,186-201` | next-best alternative exists without treatment topology | PARTIAL / DEFAULT | #10279 |
| Expected-null walking skeleton | plan `301-315` | extension promotion already requires it | PRESENT / DEFAULT | none |
| Offered versus exercised tool census | `AD-29:92-117` | calls and catalogs are separate; zero rows are lost | PARTIAL / RECIPE | #10280 |
| Revision-range closeout debt | changelog and agent closeout rules | exact range-aware owner exists | PARTIAL / DEFAULT | #5645 |
| Builder-sealed engine inputs | `internal/enginekey` | no equivalent TensorRT artifact; native controls and blobs are already bound | PARTIAL / EXCLUDE | none |
| Cache hit = deserialize + one token | engine cache validation | no serialized native engine-plan cache | PARTIAL / WATCH | none |
| Dynamic min/opt/max shape profiles | engine-key profile tests | TensorRT compilation concern, not fak-native execution | PARTIAL / EXCLUDE | none |
| Bounded binary-plan introspection | engine-plan reader | GGUF structure/bounds preflight plus smoke execution exists | PRESENT / DEFAULT | none |
| Trained versus produced preprocessing identity | vision result contracts | exact identity-bound owner exists | PARTIAL / DEFAULT | #4032 |
| Post-pass GPU clocks and coverage buckets | `internal/gpu/clocklock.go:110-216` | nativeperf environment evidence omits per-pass post clocks | PARTIAL / OPTIONAL-MODULE | #10281 |
| Anti-splice vendor memory evidence | `internal/gpu/enginemem.go:440-821` | memory provenance lacks cross-section pass coherence | PARTIAL / OPTIONAL-MODULE | #10282 |
| Functional kernel-selection digest | `internal/tactic/protocol.go:316-369` | controls exist without one mutation-proven selector identity | PARTIAL / DEFAULT | #10283 |
| Claim/null/prior sensitivity triad | `internal/tactic/remeasure.go:378-421` | nativeperf has control/candidate but no mandatory drifting null | PARTIAL / DEFAULT | #10284 |
| Evidence-polarity incomplete input | `internal/fusion/fusion.go:451` | incomplete evidence cannot pass, but positive-preserve/absence-withdraw is not typed | PARTIAL / DEFAULT | #10285 |
| Byte, numeric, and non-finite equality | export/golden comparator tests | exact non-finite-safe comparison owner exists | PARTIAL / DEFAULT | #8635 |
| Comparator criterion from run identity | `internal/export/parity.go:479-552,774-786` | criterion and identity are recorded but independently supplied | PARTIAL / DEFAULT | #10286 |
| Named order-independent multi-output comparison | export parity | native forward currently exposes one activation/logit output surface | PARTIAL / WATCH | none |

## Source anchors

These anchors make the conceptual transfer falsifiable without reproducing source text:

| Path | SHA-256 |
|---|---|
| `AGENTS.md` | `3f399b6f7f51e57b5c8ae14331ea3697b3be4cf9fe77201d80bc413775a5ee4d` |
| `docs/design/DESIGN-agentic-ground-truth-engine.md` | `f2bb5967679b24ecad2b955717101669e9d6f192c03e5f351fa40aeb54cfb847` |
| `results/agentic-data/AD-35-the-price-of-agreement.md` | `b69f2d2af35b92a1656b8c59ac1d4e09545a19e6abf110eaae72e6c508f8eea3` |
| `results/agentic-data/AD-29-bringup-turns-tools-and-the-conn-loop.md` | `7c046806607637da658c2e3afe859b0ef7a1a522387b052a538a089cd402127a` |
| `results/measurement/M8-baseline-must-be-the-next-best-alternative.md` | `8e245018067f76d0547cd4ab1533838884c05ff33592141a8ce5a42c2b8e2007` |
| `results/measurement/M9-the-agentic-lift-blends-and-the-human-one-does-not.md` | `b88ce7684b3d388f755cac3328f09aca588d1af5d041ca3fa87a884512ae36b4` |
| `internal/gpu/clocklock.go` | `f950841c45569c8b303a48ac70decf1a361dd406c9dc219355692731483f194d` |
| `internal/gpu/enginemem.go` | `857eaf16d921befcadf8b378dda0ba4b8b902ad643535179f59a33a052ba2042` |
| `internal/tactic/protocol.go` | `cc21bcf670e544b60baa908813d3a2c35b2b14b825b6e63295bcc2841ffcf860` |
| `internal/tactic/remeasure.go` | `2e07c8752ea55a935add40e376c48e309a02535dd38984b9839605ec88c3a08f` |
| `internal/fusion/fusion.go` | `bbce0ec9aac19b8c5b0333c33dc18dccd2d4cbdf119ff3cd1038c4a8051868eb` |
| `internal/fusion/verdictpolarity_test.go` | `1dd540c3697f19d74f24e57fb2a62be871ddf6eeaca49ddc0a622109ca73f2f3` |
| `internal/export/parity.go` | `c29297bb051329987f24e321017289a1fcba67eff6dbbea900b958d3d26bd885` |

## Ablation and negative knowledge

The TensorRT-specific candidates were not rejected because they are weak implementations. They
were rejected because their useful effect depends on a compiled engine-plan abstraction that
fak-native inference does not own. Adding builder-shaped keys, min/opt/max optimization profiles,
or deserialize-and-one-token cache validation now would create a false parallel architecture.

Similarly, named multi-output comparison remains a watch item until fak-native forward exposes
multiple named outputs. The correct near-term borrow is comparison identity and non-finite-safe
semantics on the output surface fak actually has.

The native-performance leaves preserve the repository invariant: execution stays fak-native and
the live path prefers Qwen3.8. Hardware witnesses must run on sanctioned lab nodes and name the
engine in the receipt; none may introduce a silent llama.cpp fallback.

## Best-default and coverage frontier

| Axis | Best default now | Bounded superset |
|---|---|---|
| Gate health | domain plus decision counts | constant-decision exemptions only with typed rationale |
| Evidence quorum | independent root lineage | richer evidence classes after a real consumer needs them |
| Tool acquisition | raw gain, cost, hypotheses, and selected/next-best | learned estimates only after calibrated receipts exist |
| Benchmark admission | matched topology, directional floors, sensitivity control | hardware-specific probes as optional modules |
| Partial evidence | polarity plus covered source range | domain-specific parsers behind the common contract |
| Native identity | decoded canonical selector record plus digest | multi-output identity when the runtime exposes it |

## Problem and value check

- **Centrality:** Enabling overall; four candidates directly strengthen native benchmark validity.
- **P1 managed context:** a 6,012-file source is compressed into one inventory, one receipt, source
  hashes, explicit negative knowledge, and dispatchable leaves.
- **P2 net-true efficiency:** the study makes no performance claim. Every performance leaf requires
  setup, probe, verification, and failure cost in its matched envelope.
- **P3 bounded adaptation:** absent, unknown, incomplete, divergent, and present states remain
  separate; optional hardware contracts do not become universal defaults.
- **P4 integrated operations:** every borrow names an existing fak seam or is excluded; no parallel
  TensorRT runtime, benchmark stack, or evidence store is proposed.

**For** fak maintainers and agents; **Problem:** the earlier study captured only three broad
operational mechanisms; **Today:** source-level measurement contracts and their native-runtime
applicability were not deeply reconciled; **Better because:** the complete source denominator,
license fence, ablation decisions, exact FAK witnesses, and issue ownership are durable;
**Witness:** the inventory, immutable study receipt, 32/32 source-hash read-back, and issue set
#10268-#10271 / #10278-#10286.

## Independent witness and remaining uncertainty

Worker summaries were treated as claims. The coordinator recomputed every cited source hash,
read the relevant FAK seams, reran exact open-and-closed issue searches for all 13 new titles,
and read back the created issues. The local source lacks managed worker transcript paths, so no
transcript-level DOS verification was possible; source artifacts, FAK files, GitHub objects, and
the durable study record are the independent evidence surfaces instead.

Refresh when a trustworthy source revision or license appears, when any watched TensorRT-shaped
abstraction gains a real fak-native counterpart, or when one of the filed leaves changes the
on-axis evidence. Do not reinterpret absent upstream metadata as evidence that no upstream work
exists.
