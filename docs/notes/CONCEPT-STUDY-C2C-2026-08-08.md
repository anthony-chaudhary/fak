# Concept study — thu-nics/C2C (Cache-to-Cache / "Rosetta") → witnessed borrows for fak

- **Source:** `https://github.com/thu-nics/C2C` — official implementation of *Cache-to-Cache:
  Direct Semantic Communication Between Large Language Models* (arXiv:2510.03215, ICLR'26).
- **Pinned:** `@113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a` (HEAD at clone; subject
  `[debug] fix a bug of saving kv cache`).
- **License:** Apache-2.0 → INTEGRATE would be *permissible*, but every borrow below is
  **INSPIRE** (PyTorch research code; fak is Go — clean-room or nothing).
- **Mode:** `--draft`. **No issues were filed.** See "Why this is a draft" below.

## Why this study exists — and why it is NOT the requested paper

The operator asked for `https://arxiv.org/html/2608.03893v1`. That paper could not be
acquired, and the honest record of the attempt matters more than a plausible summary:

- **`WebFetch` is refused in this environment as `TRUST_VIOLATION`** (a terminal guard
  boundary). It was not routed around via `curl` — that would be evasion, not adaptation.
- **The citation never resolved.** Six `WebSearch` passes split: three returned a confident
  title (*"Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for
  Prefill Reuse"*) with a full author roster and even neighbouring arXiv ids; three failed to
  find it at all, one stating the id "may not exist". **No pass ever returned the paper's own
  abs/html page as a link**, while the same index did return real links for neighbouring
  August-2026 ids. Consistent detail from a summarizer is not a resolved citation.
- **No implementation repo for it surfaced** in any pass.

Rather than file borrows from an unread — possibly non-existent — pitch, this pass applies
the skill's own arXiv rule ("the PDF is the map, the repo is the ground truth") to the
nearest *verifiable, cloneable* work on the identical axis: C2C projects and fuses one
model's KV-cache into another's. **This is a substituted target the operator did not name**,
which is why it is `--draft`.

## What was read (fan-out coverage + completeness critic)

Read at the pinned sha: `README.md`; `rosetta/model/projector.py` (the fuser: projection,
gating, weighting); `rosetta/model/aligner.py` (`TokenAligner`, in full for the align path);
`rosetta/model/wrapper.py` (`RosettaModel.forward`, hook/monkeypatch injection, the
`kv_cache_index` sharer protocol); `rosetta/baseline/multi_stage.py` (outline — the
text-communication baseline C2C is measured against); `script/evaluation/standard_kvcache_del.py`
and `script/analysis/proportion/auto_kv_cache_evaluation.py` (the degradation/ablation
harnesses); `script/consistency/check_rosetta_consistency.py` (agreement harness).

**Completeness critic — deliberately not opened, with cause:** `rosetta/train/**`,
`rosetta/utils/**`, `rosetta/model/oracle.py`, `sampling.py`, `ablation_projector.py`, and
most of `script/**` (dataset construction, SFT training, plotting) are the machinery for
*producing and grading a trained fuser*. Row 3 below establishes that fak will not train a
fuser, so that whole subtree is off-axis for a borrow — not unread by accident.

## Their worldview (reconstructed — falsifiable, not testimony)

C2C targets a **research/serving user who wants one model to benefit from another's reading
of a prompt without paying text-generation latency**. The evidence, from defaults rather
than marketing: both LLMs are **frozen** and only the fuser trains (README, `SFT_train.py`);
the headline gauges are **accuracy deltas and a latency multiple**, never bit-exactness; the
gate is **learned and stochastic during training** (Gumbel-sigmoid, `projector.py:591`); and
the shipped comparison baseline is **text communication** (`multi_stage.py:25`), i.e. their
"do nothing" is lossy prose, so an *approximate* cache transfer is strictly an upgrade in
their world. Quality is measured as **agreement**, not identity (`check_rosetta_consistency.py:153`
compares greedy label predictions).

That is the axis where fak diverges — and fak says so in its own code, not in my opinion.

## Candidate table

| # | Borrow (one technique) | Source `path:line@sha` | **Axis** | Their-worldview reason | Witness **on-axis** vs fak | Route | Filed |
|---|---|---|---|---|---|---|---|
| 1 | **Positional 1:1 cross-tokenizer alignment**: decode each source token → re-encode in the target tokenizer → on a 1-to-many split, collapse by an explicit `FIRST`/`LONGEST` strategy, keeping one target token per source token; special tokens go through a remap table | `rosetta/model/aligner.py:65-152@113c3a9b` (strategy enum `:14`, special-token remap `:154`) | **Preserving positional correspondence when bridging two tokenizers** — as opposed to dropping the tokens that do not map | They must line up KV *rows* between models, so a length-preserving map is structural, not optional | **PARTIAL** — fak's `VocabMap` is a *set* intersection (id↔id) and `MapContext` (`internal/polymodel/vocabmap.go:220-235`) **drops** unmappable context tokens (`dropped++; continue`), shortening and shifting the drafter's context. fak has no *sequence* alignment that survives a 1-to-many split | INSPIRE | — (draft) |
| 2 | **Per-layer sensitivity profiling to drive precision**: measure per-layer tolerance to a degraded/foreign KV by ablation, then treat layers unequally | `script/evaluation/standard_kvcache_del.py@113c3a9b`; `script/playground/sensitivity_test.py@113c3a9b`; reuse-fraction + front/back order sweep at `script/analysis/proportion/auto_kv_cache_evaluation.py:39-41@113c3a9b` | **Per-layer vs uniform precision assignment inside an already-lossy KV tier** | Layers differ in how much distortion they tolerate; spending equally on all of them wastes budget | **PARTIAL** — fak's `KVPrecision` (`internal/compute/kvprecision.go:28`) is a **whole-cache** tier (`f32`\|`q8`) chosen once by `AutoSelectKVPrecision` (`:114`). `perTokenPerLayerBytes` (`:87`) prices per layer but precision is uniform. **The bit-exactness divergence in row 3 does NOT cover this** — `q8` is already an accepted-lossy lane | INSPIRE | — (draft) |
| 3 | **Learned neural KV projector/fuser** — MLP projects source KV into the target's space, a gate picks which target layers receive it | `rosetta/model/projector.py:862` (`C2CProjector`), `:176` (`AllInOneProjector`), fusion entry `rosetta/model/wrapper.py:350`@`113c3a9b` | **Approximate cross-model KV reuse** | Their floor is lossy text hand-off, so an approximate cache beats it outright | **DIVERGENT — and fak states the tradeoff itself.** `internal/polymodel/vocabmap.go:39-46`: *"WHAT STAYS REFUSED: PREFILL SHARE. CanShare is deliberately NOT relaxed… the reused KV must be BIT-identical, not merely meaningful."* `CanShare` (`internal/polymodel/polymodel.go:583`) admits reuse only on same `Family` **and** same `PrefixDigest`. fak sells replay determinism to an audited fleet; an approximate transfer forfeits exactly that property | — | not filed (earned) |
| 4 | **Per-span sharer bitmask** — `kv_cache_index` selects per sequence-section which sharer(s) contribute cache (`-1` none, bitmask otherwise) | `rosetta/model/wrapper.py:378-382@113c3a9b` | **Per-span selection of which source contributes cache** | Multi-sharer fusion needs span-level provenance | **DIVERGENT (consequent of 3)** — fak's `internal/radixkv/namespace.go` scopes by *trust domain* (never share across), a different question from *blending* sources per span. Blending is only meaningful once row 3 is accepted, which fak refuses | — | not filed |
| 5 | **Greedy label-agreement consistency harness** between the fused path and each reference model | `script/consistency/check_rosetta_consistency.py:153@113c3a9b` | **Statistical agreement as the correctness gauge** | Their transfer is approximate by construction, so agreement is the only available gauge | **PRESENT-on-axis, fak stronger** — fak witnesses *bit-exactness* (`internal/model/kvcache.go:94` `Evict` survivors at `max\|Δ\|=0`; `internal/polymodel/polymodel.go:410` `AcceptGreedy`). An agreement-rate harness is strictly weaker than an identity witness | — | not filed |

## Recommended shape if the operator green-lights filing

Two independently-shippable leaves. **No epic** — they share only a source repo, not a track,
and each ships and proves value alone:

1. **Positionally-aligned cross-vocabulary drafter context** (row 1) — replace `MapContext`'s
   drop with a length-preserving alignment. First checkable step: a test that a bridged pair
   whose tokenizers split differently yields `len(out) == len(committed)`, plus a mean-acceptance
   comparison against today's drop path. **Correctness is free here** — `vocabmap.go:19-37`
   already establishes the verifier gates every token, so this is a throughput lever with a
   floor of "identical to no drafting at all".
2. **Per-layer KV precision from a measured sensitivity profile** (row 2) — let
   `AutoSelectKVPrecision` return a per-layer assignment. First checkable step: a profiler that
   ranks layers by output delta under `q8`, and a witness that a mixed assignment holds a
   target quality bar at strictly fewer resident bytes than uniform `q8`.

## Honest limits of this pass

- **`--depth 1` clone → one commit of history.** Their rationale *evolution* was unreadable;
  the worldview above rests on README + code + defaults only.
- **No issues/discussions/design docs were read** — `WebFetch` is blocked here. The
  worldview reconstruction is therefore weaker than this skill's step-2 ideal, and no row
  was dismissed on a *guessed* motive: rows 3–5 rest on **fak's** stated rationale in fak's
  own code, which is readable and quotable.
- **The requested paper remains unstudied.** Nothing here should be read as a summary of
  arXiv 2608.03893, whose existence this pass could not confirm.

## Companions

- Skill: [`study-repo`](../../.claude/skills/study-repo/SKILL.md); hand-off target
  [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) for a per-capability re-witness
  before any of the above is filed.
- Prior passes on the same seam: [SGLang](CONCEPT-STUDY-SGLANG-2026-07-18.md) (RadixAttention,
  prefix-delta KV handoff #5288), [vLLM](CONCEPT-STUDY-VLLM-2026-07-18.md) (block-hash prefix
  caching, KV offload tiering).
- Related fak surfaces: `internal/polymodel/vocabmap.go` (#4208, epic #4207),
  `internal/compute/kvprecision.go`, `internal/l3kv/partial.go` (#3897).
