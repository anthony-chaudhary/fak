---
title: "Quantization codebook metadata contract"
description: "internal/codebookmeta is a versioned metadata/adjudication leaf for non-uniform quantization. It does not choose a universal format or kernel."
---
# Quantization codebook metadata contract

Issue: [#6251](https://github.com/anthony-chaudhary/fak/issues/6251) · parent [#6221](https://github.com/anthony-chaudhary/fak/issues/6221)

## Scope and neutral outcome

`internal/codebookmeta` is a versioned metadata/adjudication leaf for non-uniform quantization. It does not choose a universal format or kernel. A descriptor produces exactly one typed outcome:

- `supported / CODEBOOK_ACCEPTED`: the schema, payload, packing, digest, and decode requirements are understood locally;
- `unsupported / <reason>`: malformed, unknown, missing, or unavailable requirements fail closed without a fallback;
- `delegate / RUNTIME_DELEGATION_REQUIRED`: a decode feature is unavailable locally but a descriptor-named, capability-approved runtime can handle it.

The contract covers fixed integer grids, the explicitly enumerated 16-entry NF4 codebook, learned codebooks, and parametric codebooks. Learned and parametric forms must carry entries, parameters, or a payload. NF4 is not inferred from “4 bit”: its 16 values are explicit fixture data. Index packing pins ID, version, bits per index, byte order, and optional indices per word. Decode requirements pin placement, scale type, group size, required features, and any routed runtime.

## Provenance pins

Every accepted descriptor names and pins four different things rather than conflating them:

| Boundary | Required fields |
|---|---|
| Artifact | `artifact_id`, SHA-256 `artifact_digest` |
| Recipe | `recipe_id`, `recipe_version`, SHA-256 `recipe_digest` |
| Runtime | `runtime_id`, `runtime_version` |
| Model | `model_id`, immutable `model_revision` |

The codebook has its own SHA-256 over its kind, ID, version, entries, parameters, and payload. A consumer must not interpret a recipe version as an artifact identity or a runtime version as evidence about model quality.

## Research/evaluation record

Evaluation rows are provenance-bearing data, not free-form performance claims. `kind` is either:

- `observed`: the named witness actually read the effect and may record a value, unit, hardware, and dataset;
- `modeled`: a hypothesis or analytical statement; it remains visibly modeled and does not become a measured value.

The checked-in fixtures make one narrow **observed** claim: the descriptor round-trips and is adjudicated by `TestFixtureRoundTripsAndPinsProvenance`. The learned fixture also records the **modeled** statement that quality and performance need a separate hardware-and-corpus witness, with no numeric value. There is no throughput, latency, memory, perplexity, or task-quality claim in this issue. Consequently no accelerator run is needed to validate this metadata contract, and none is invented. A future measured claim must pin the artifact, recipe, runtime, model, hardware, dataset, metric, and independently readable result; GPU work must use the sanctioned lab route described by `docs/private-comms-channel.md`.

## Independently readable witness

The fixtures under `internal/codebookmeta/testdata/` are public JSON effects readable without trusting the implementation author:

1. `integer-grid.json` — fixed signed INT4 grid;
2. `nf4.json` — explicit non-uniform 16-entry NF4 table;
3. `learned.json` — learned entries plus parameters and explicit decode requirements.

The tests read those files from disk, round-trip all fields, verify provenance pins and codebook digests, and exercise typed unsupported/delegate paths. In particular, deleting the learned payload yields `unsupported / MISSING_CODEBOOK_PAYLOAD`; an unavailable feature delegates only to the exact approved runtime; unknown schema, kind, packing, and digest never fall back.

Run the focused witness in WSL:

```sh
go test ./internal/codebookmeta -count=1
```

Then run the repository-owned-path gates over exactly `internal/codebookmeta` and this document.


