# Harness model-set dependencies and compatibility resolution research

**Decision date:** 2026-08-19  
**Parent:** [#8132](https://github.com/anthony-chaudhary/fak/issues/8132)  
**Adjacent tracks:** [#8131](https://github.com/anthony-chaudhary/fak/issues/8131) owns bundled-model acquisition/lifecycle; [#8133](https://github.com/anthony-chaudhary/fak/issues/8133) owns the independently composable server builder. This note owns neither.

## Decision

A generated harness should declare a **role-indexed required model set**, resolve that intent against a typed inventory, and commit the result as a deterministic **model-set lock**. Startup should independently re-evaluate the lock against observed inventory and emit a **compatibility receipt**. A role may name ordered compatible alternatives, but selection is permitted only when every declared capability and operational constraint is witnessed; unknown or stale evidence fails closed.

The minimum contract is deliberately narrower than package management or model quality ranking:

1. intent names roles and required capabilities, not a preferred server implementation;
2. inventory records immutable artifact identity plus witnessed serving/capability facts;
3. resolution is deterministic for the same intent, inventory, platform, and policy;
4. the lock records the selected candidate and why rejected alternatives lost;
5. selfcheck verifies the lock without downloading artifacts or constructing servers; and
6. the receipt records target versus witnessed envelopes and actionable incompatibilities.

This gives init, build, inspect, upgrade, CI, and startup one dependency truth while leaving model ownership to #8131 and server construction to #8133.

## Primary sources studied

Sources were read on 2026-08-19 and revisions were pinned before extracting facts.

| System | Pinned primary source | Relevant mechanism |
|---|---|---|
| uv | [`astral-sh/uv@fa2bff1`](https://github.com/astral-sh/uv/tree/fa2bff1ea889090a6628c21c7d73cbdc6cd3618d) · [lockfile concept](https://github.com/astral-sh/uv/blob/fa2bff1ea889090a6628c21c7d73cbdc6cd3618d/docs/concepts/projects/layout.md) · [resolution](https://github.com/astral-sh/uv/blob/fa2bff1ea889090a6628c21c7d73cbdc6cd3618d/docs/concepts/resolution.md) | A universal lock records resolutions across environments; environment markers and resolution forks preserve conditional choices rather than pretending one artifact fits all targets. |
| OCI Image Specification | [`opencontainers/image-spec@af26a05`](https://github.com/opencontainers/image-spec/tree/af26a05fba5ee648512f4ea3c9fda1fcc1b6d6dc) · [descriptor](https://github.com/opencontainers/image-spec/blob/af26a05fba5ee648512f4ea3c9fda1fcc1b6d6dc/descriptor.md) · [image index](https://github.com/opencontainers/image-spec/blob/af26a05fba5ee648512f4ea3c9fda1fcc1b6d6dc/image-index.md) | Content is selected through typed descriptors with digest, size, and media type; indexes provide platform-qualified alternatives. |
| Hugging Face Hub client | [`huggingface/huggingface_hub@7576cce`](https://github.com/huggingface/huggingface_hub/tree/7576ccead92135b25c09fc5353784bb1f53db0df) · [download guide](https://github.com/huggingface/huggingface_hub/blob/7576ccead92135b25c09fc5353784bb1f53db0df/docs/source/en/guides/download.md) · [model metadata API](https://github.com/huggingface/huggingface_hub/blob/7576ccead92135b25c09fc5353784bb1f53db0df/docs/source/en/package_reference/hf_api.md) | Downloads can target a revision/commit and return snapshot-local content; repository metadata exposes revision and file information, but repository presence alone is not a serving-capability proof. |

## Observed facts

### External systems

- **uv:** its project lock is intended to contain a resolution broad enough for supported environments, and its resolver represents environment-specific forks using marker conditions. This is an observed dependency-resolution mechanism, not evidence that Python package semantics transfer directly to models.
- **OCI:** a descriptor binds identity to a digest and qualifies content with a media type; an image index is a list of descriptors and may attach platform fields. Selection and content identity are separate concepts.
- **Hugging Face Hub:** callers can request a named revision or commit hash and obtain repository/file metadata. Hub metadata does not prove tool calling, context length under a particular runtime, memory fit, privacy posture, or endpoint health.

### Current FAK tree at committed `HEAD`

- `internal/modelreg/modelreg.go` maps friendly aliases to `hf://` URIs or local paths and reports cache presence. Its durable entry is `Name`, `Target`, and `Source`; `ListEntry` adds local size and a curated `Coding` boolean. It is an alias/cache registry, not a role-set dependency solver.
- `cmd/fak/model.go` exposes model operations over that registry. It can resolve/download individual model references, but there is no committed multi-role lock with rejection reasons.
- `cmd/fak/harness_resolve.go` resolves harness stack component selections and emits a stack lock. It is useful prior art for deterministic harness materialization, but the component lock does not express role-specific model capabilities or inventory evidence.
- `cmd/fak/harness_stack.go`, `cmd/fak/harness_init.go`, and `cmd/fak/harness_inspect.go` are the current harness lifecycle seams where a model-set lock can be generated and shown.
- `cmd/fak/model_readiness_inventory.go` already frames model readiness as evidence/inventory. It does not currently serve as a typed candidate inventory consumed by a model-set resolver.
- `cmd/fak/serve_backend_preflight.go` and `cmd/fak/doctor_serve.go` are serving checks. They do not independently verify a generated harness's role-indexed lock and emit a compatibility receipt.
- `docs/notes/HARNESS-STACK-RESOLVE-CLI-2026-08-15.md` specifies deterministic stack resolution and atomic lock writing; model-set resolution should compose with that contract rather than overload its component vocabulary.
- `docs/notes/HARNESS-STACK-LOCAL-FIRST-MODEL-2026-08-15.md` makes local-first model selection explicit. Local-first preference is a policy input, not permission to accept an incompatible local candidate.

## Inferences and design choices

These are project decisions inferred from the facts above, not claims about the external systems.

- Borrow uv's separation of declared intent from a reproducible lock, but use **role and capability constraints** instead of package version ranges.
- Borrow OCI's separation of immutable identity from platform qualification: a candidate identity should include provider/repository revision and artifact digest where available, while compatibility facts carry runtime, accelerator, quantization, protocol, and policy envelopes.
- Treat Hub metadata as acquisition identity only. Tool calling, structured output, context, tokenizer/runtime pairing, memory fit, endpoint behavior, privacy, and license acceptance require explicit witnessed facts from FAK-owned probes or operator-attested policy inputs.
- Deterministic ordering should be: satisfy all hard constraints, apply declared policy preference, compare stable candidate keys, and record every rejection. No opaque quality score or network-time “best model” lookup belongs in the spine.
- A lock is stale when its inventory digest, policy digest, target platform, or evidence expiry no longer matches. Stale is incompatible until re-resolution; startup must not silently substitute.
- Alternatives are role-local. A planner candidate cannot silently satisfy an executor role merely because both expose an OpenAI-compatible endpoint.

## Proposed contract

### Intent (`harness.model-set.json`)

Each role declares:

- stable role ID and whether the role is required;
- one or more alternative candidate constraints;
- hard capabilities (for example tool protocol, structured output, minimum context, modality);
- operational constraints (runtime/protocol, platform/accelerator, memory ceiling, locality/privacy, license allowlist);
- explicit preference policy, if any; and
- evidence freshness requirement.

### Inventory (`model-inventory.json`)

Each candidate records immutable identity, availability, artifact/runtime pairing, platform envelope, serving protocol, witnessed capabilities, provenance, witness timestamp/expiry, and policy facts. Unknown fields do not satisfy hard constraints.

### Lock (`harness.model-set.lock.json`)

The canonical lock records schema version, intent digest, inventory digest, policy digest, target envelope, selected candidate per role, immutable identity, evidence references, ordered alternatives, rejection reason codes, and resolver version. Canonical JSON and stable sorting make byte equality meaningful.

### Receipt (`harness.model-set.receipt.json`)

Selfcheck records lock digest, observed inventory digest, target and witnessed envelopes, per-role checks, incompatibility codes, remediation, and overall `compatible|incompatible`. The receipt contains no credentials and does not mutate the lock.

## Minimal working spine

A clean-directory two-role fixture is the first runnable path:

1. `planner` requires tool calling, structured JSON, and a declared minimum context;
2. `executor` requires tool calling, a lower memory ceiling, and local-only execution;
3. inventory contains one valid candidate for each role plus candidates rejected for stale evidence, protocol mismatch, and memory excess;
4. `fak harness model-set resolve --intent ... --inventory ... --out ...` writes a byte-stable lock twice;
5. `fak harness model-set selfcheck --lock ... --inventory ... --receipt ...` exits zero and writes a compatible receipt;
6. replacing the executor inventory with an incompatible candidate exits nonzero, leaves the lock unchanged, and names role, failed constraint, evidence source, and remediation.

This spine exercises real parsing, deterministic resolution, lock persistence, independent read-back, and fail-closed startup evidence without downloading a model, launching a server, or claiming semantic model quality.

## Boundaries

### In scope

- Role-indexed dependency intent and ordered compatible alternatives.
- Immutable candidate identities, capability evidence, platform/runtime constraints, and evidence freshness.
- Deterministic lock generation, stale-lock detection, inspect/upgrade integration, and compatibility receipts.
- Explicit failures for unavailable artifacts, capability mismatch, unsupported runtime/platform, privacy/license rejection, and stale/unknown evidence.

### Out of scope

- Model download, cache eviction, redistribution, license fulfillment, and bundled lifecycle: #8131.
- Building, packaging, launching, or supervising an inference server: #8133.
- Universal semantic benchmarking, automatic quality ranking, pricing optimization, or an unconstrained router.
- Secret storage, endpoint credential acquisition, or silently probing private endpoints.
- Treating provider model names, aliases, or repository cards as capability witnesses.

## Issue decomposition

Exactly seven implementation leaves are required; their domains and file trees are intentionally disjoint:

1. **Intent schema and parser** — role/capability declaration only.
2. **Typed inventory and evidence normalization** — candidate observations only.
3. **Deterministic resolver** — selection/rejection logic only.
4. **Canonical lock and upgrade inspection** — durable selected state only.
5. **Compatibility evaluator and receipt** — independent read-back only.
6. **Harness lifecycle wiring** — init/build/inspect integration only.
7. **Two-role conformance scenario** — end-to-end positive and fail-closed witness only.

Every leaf must preserve all P1-P4 properties, name a target and directly witnessed envelope, and ship an independent test or captured scenario. The live issue URLs are added below after creation so this note remains the durable index.

## Completion criterion for #8132

Research is complete only when this note is visible from pushed `main`, exactly seven live child issues are linked to #8132 and pass `fak-dev issue contract`/`cohort`, and GitHub independently reports those links. No implementation is part of this research issue.
## Live implementation cohort

- [#8146](https://github.com/anthony-chaudhary/fak/issues/8146) — declare role capability intent.
- [#8147](https://github.com/anthony-chaudhary/fak/issues/8147) — normalize candidate evidence inventory.
- [#8148](https://github.com/anthony-chaudhary/fak/issues/8148) — resolve roles with deterministic rejection reasons.
- [#8149](https://github.com/anthony-chaudhary/fak/issues/8149) — persist canonical selection lock.
- [#8150](https://github.com/anthony-chaudhary/fak/issues/8150) — attest compatibility with startup receipt.
- [#8151](https://github.com/anthony-chaudhary/fak/issues/8151) — wire model-set resolve, selfcheck, and inspect.
- [#8152](https://github.com/anthony-chaudhary/fak/issues/8152) — prove two-role model-set resolution.
