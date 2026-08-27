# Zero-day model onboarding: compile release facts before support code

**Observed:** 2026-08-27T15:35:00Z
**Issue:** [#9421](https://github.com/anthony-chaudhary/fak/issues/9421)
**Study receipt:** `study_a4310abb38820a69d2b77ffaa1ada12058d1cde24169bf651806ef1c9baa8d01`
**Current authority:** [Adding a new model to fak](../new-model-playbook.md)

This is the provenance note for the release-to-descriptor compiler. It records the dated external
evidence and the design choice; the playbook carries current operating instructions.

## Source ledger

| Source | State and event | Immutable anchors read | Platform context | Refresh trigger |
|---|---|---|---|---|
| `vllm-project/vllm` | shipped at `76c0c6530e832c758f5502ca4ef37ae88a7ffa13`, 2026-08-27T15:30:22Z | `docs/contributing/model/README.md:3@76c0c6530e832c758f5502ca4ef37ae88a7ffa13`; `docs/contributing/model/registration.md:14@76c0c6530e832c758f5502ca4ef37ae88a7ffa13`; `docs/contributing/model/tests.md:10-27@76c0c6530e832c758f5502ca4ef37ae88a7ffa13`; `vllm/model_executor/models/registry.py:913-998,1183-1295@76c0c6530e832c758f5502ca4ef37ae88a7ffa13`; `tests/models/test_registry.py:31,206@76c0c6530e832c758f5502ca4ef37ae88a7ffa13` | native model registry plus Transformers compatibility backend | vLLM registry/compatibility contract change or a relevant revert follow-up |
| `sgl-project/sglang` | shipped at `20a491d1d311553bbab3f22e19bbafb86ef3c0cc`, observed 2026-08-27 | `python/sglang/srt/models/registry.py:61-134@20a491d1d311553bbab3f22e19bbafb86ef3c0cc`; `docs/docs/supported-models/support_new_models.mdx:8-105,315-433,515-517@20a491d1d311553bbab3f22e19bbafb86ef3c0cc` | SRT architecture registry, processors, tests, docs, optional dependencies, and generic Transformers fallback | SGLang model-registration or fallback-contract change |
| vLLM history | shipped then reverted/migrated, 2026-08-21 through 2026-08-24 | native remove/revert/migrate: `d53b1c2efc0ca7161c3844f51ca9acbcdb7129d5`, `e3f60265032e7c1e158a749d82a8c4dad62f941a`, `48d7132962ef3e2e6982a316013294c0029e3a64`; fusion/revert: `463aa5e30fe0b51a5d6ef22897962fc1be97cb85`, `592e06f2ae115cbb0f7e2e1e776c255a3fe6c3c1` | Hunyuan native/Transformers migration and Kimi-K3 ROCm KDA-prefill fusion | the next native-model removal, backend migration, or architecture-fusion revert |

Both roots were Apache-2.0 at the pinned revisions. The disposition is **INSPIRE-ONLY**: this spine
copies no expressive upstream code, tests, or comments. Any later direct port requires its own
per-file provenance and notice review.

## What the sources prove

vLLM and SGLang optimize for breadth and contributor throughput. A newly released Hugging Face
architecture can often run through a compatibility backend before a native implementation exists.
Native support then fans out across architecture registries, model/config predicates, processors,
weight loaders, tests, docs, dependency gates, and backend-specific behavior.

That is a useful zero-day ingestion strategy, but it is not a durable native-support contract. The
vLLM history above shows two failure patterns that registration alone cannot prevent:

1. native-model removal, revert, and re-migration can happen within days; and
2. an architecture-specific fusion and its device witness can be added and reverted together within
   hours, erasing the runnable evidence along with the implementation.

FAK should borrow the speed of metadata intake without borrowing automatic foreign execution. A
pinned release must become one deterministic packet before any code generation, allocation, or
support claim. The packet keeps provenance, semantic uncertainty, native/backend/test/oracle/docs/
performance work, registration closure, coupling cost, and rollback obligations together.

## Inward witness and seam

The durable study recorded an **ABSENT-on-axis** verdict:

- `fak capabilities "zero day model onboarding compiler"` returned no on-axis capability card on
  2026-08-27.
- `fak dev index docs/leaves "model release onboarding descriptor oracle fixture rollback"`
  surfaced the manual playbook and descriptor leaf, but no release-to-descriptor compiler.
- `internal/modeldescriptor/descriptor.go:1` already defined the declarative validation, digest,
  and coupling report.
- `internal/newmodel/newmodel.go:22` still began at a handwritten scaffold.

Issue #9421 therefore connects those existing seams rather than adding another registry, planner,
or receipt stack.

## Candidate disposition

| Borrow | Axis | Source fact | FAK decision | Disconfirming check |
|---|---|---|---|---|
| Compatibility-first release intake | time to ingest a newly published architecture | both frameworks can recognize or route models before native specialization is complete | **DEFAULT / INSPIRE:** accept pinned facts immediately, but compile only a fak-native work packet | reject if three unrelated releases cannot be expressed without executable hooks |
| One release descriptor | registration closure | architecture, fixture, defaults, dependency, and docs state is spread across handwritten surfaces | **DEFAULT / INSPIRE:** normalize one manifest through `modeldescriptor` and emit exact open obligations | reject if the compiler hides model-specific work or creates a second registry |
| Typed semantic refusal | false support avoidance | generic compatibility fallback can make unknown architecture deltas appear runnable | **DEFAULT / DIVERGENT:** unknown or contradictory deltas stop before allocation; no automatic fallback | reject if a refusal can still emit scaffold/executable behavior |
| Persistent rollback witness | learning across reversions | rapid native and fusion reverts can remove their only witness | **FOLLOW-ON:** keep oracle/performance/rollback obligations independent from an optimization patch | reject if retained witnesses cannot reproduce the reverted failure |

## Completeness and honest boundary

The underlying deep study covered registry/compatibility code, model-contributor docs, registry and
initialization tests, architecture-specific runtime mutation surfaces, recent reverts and repair
history, and root/per-file license/provenance. It did not establish that every current vLLM or SGLang
model is expressible by the FAK manifest; that remains the disconfirming replay for the follow-on
agentic delta extractor. This note makes no runtime, quality, kernel-fusion, or performance claim.

Companions: [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) |
[`study-repo`](../../.claude/skills/study-repo/SKILL.md) |
[#8011](https://github.com/anthony-chaudhary/fak/issues/8011) |
[#9282](https://github.com/anthony-chaudhary/fak/issues/9282) |
[#9421](https://github.com/anthony-chaudhary/fak/issues/9421)
