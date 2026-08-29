---
title: "runtime - runtime:model-serving identity"
description: "The model-completion implementation behind an engine driver, such as an on-device llama.cpp or Ollama adapter that generates text for one turn. Scope: runtime:model-serving. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# runtime

**Meaning:** The model-completion implementation behind an engine driver, such as an on-device llama.cpp or Ollama adapter that generates text for one turn.

## Do not conflate with

- **model KV cache:** The model-serving runtime executes completion; the model KV cache is attention state owned or reused during that execution.

## Query this identity

```console
$ fak disambiguation query "runtime" --scope-kind "runtime" --scope-value "model-serving" --json
```

## Identity

- **Scope:** `runtime:model-serving`
- **Aliases:** model serving runtime
- **Owner:** `engine / engine`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `runtime-source/1`

## Source witnesses

- `internal/engine/on_device.go` (go-source, revision `runtime-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-runtime-source`)

[Back to the disambiguation index](../INDEX.md)
