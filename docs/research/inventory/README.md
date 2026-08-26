---
title: "Related-system inventory maps"
description: "Reference documentation for Related-system inventory maps, preserving the page's implementation details, evidence, and operating context."
---

# Related-system inventory maps

Machine-assisted maps in this directory are generated from pinned local checkouts with `fak study-inventory`.

Each map is an inventory denominator for a deep `study-repo` pass. It records the local tree shape and explicitly names source classes that still require forge/API read-back or study-pass artifacts. A map is not a borrow decision and not proof that the study is complete.

Registry rows must add `source_evidence` for any source class the map marks `partial` or `external_required`. Bare `source_classes` names are only acceptable when backed by a map status of `covered` or `checked_absent`.

Current maps:

- [`ggml-org-llama-cpp.json`](ggml-org-llama-cpp.json) — pinned tree plus complete open issue/PR corpus and FAK ticket-priority join for `ggml-org/llama.cpp` at `925e1179947ea0c0ebfb0032df18af3a729822be`.
- [`local-tensor-build.json`](local-tensor-build.json) — exhaustive machine map for the local TensorBuild snapshot at `snapshot-sha256:bf4dd9267f31dea48b925602e3d1326f65ca3a1e02d3062afecf414af1614288`.
- [`langchain-ai-open-swe.json`](langchain-ai-open-swe.json) — machine map for `langchain-ai/open-swe` at `a6c360047186cc5b8afe3a74012a12bfc94ae7c7`.
- [`langchain-ai-open-swe.md`](langchain-ai-open-swe.md) — human rendering of the same scan.

- [`ruvnet-ruflo.json`](ruvnet-ruflo.json) — machine map for `ruvnet/ruflo` at `4dcff483482cee316f47552a961bcbaadc89f378`.
- [`ruvnet-ruflo.md`](ruvnet-ruflo.md) — human rendering of the same scan.
- [`vllm-project-speculators.json`](vllm-project-speculators.json) — machine map for `vllm-project/speculators` at `0faffeb3bd547b4451a978d7aaf26a2f01b83d62`.
- [`vllm-project-speculators.md`](vllm-project-speculators.md) — human rendering of the same scan.
- [`obra-superpowers.json`](obra-superpowers.json) — machine map for `obra/superpowers` at `b36e0829c6d0140e93cfef2ca599b1b07d4a7797`.
- [`obra-superpowers.md`](obra-superpowers.md) — human rendering of the same scan.
