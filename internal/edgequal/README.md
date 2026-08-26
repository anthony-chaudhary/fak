# Edgequal offline evidence contract (#8600)

This package is the deterministic, versioned evidence gate for the low-resource
phone-and-laptop spine. It pins exactly one candidate and the runtime seam chosen
by #8131:

- `bartowski/Qwen2.5-1.5B-Instruct-GGUF` at repository revision
  `d6f592509429a0f25fc337a6d05065356c40d2b2`, file
  `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`, LFS SHA-256
  `1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370`.
- `llama.cpp/server` revision
  `2e92ecd0247d25f09797f8fdb044a166522fc05d` (the #8131 reference runtime).
- the embedded `issue-8600/v1` Hindi/Hinglish, Simplified-Chinese, and English
  local-document/tool/injection pack, whose digest is computed from checked-in
  bytes.

Raw physical receipts remain out of tree and privacy-safe. Validate both decoded
receipts with `ValidatePair`; the receipt's immutable raw-artifact URL and SHA-256
bind the retained prompts/outputs. The validator deliberately rejects simulators,
desktop extrapolation, mutable model/runtime names, missing digests, networking
after acquisition, and weights-loaded-only evidence. A typed refusal records an
honest bounded failure and never becomes a pass.

## Lifecycle evidence

Promotion requires two raw receipts: one named physical arm64 Android phone and
one named physical 8 GiB CPU/iGPU laptop, both replaying the same pack/model/runtime
and completing at least 15 minutes offline while recording the declared quality,
safety, latency, RSS, storage, thermal, and energy fields. Demote or retire this
option if either device emits `OOM`, `UNSAFE_TOOL`, `QUALITY_FLOOR`,
`LATENCY_FLOOR`, or `THERMAL_LIMIT`, or if a pinned dependency can no longer be
reacquired by digest. The main invalidating assumption is that this Q4 multilingual
candidate can preserve schema-constrained tool use and injection safety inside a
2K context on both physical device classes.

No physical receipt is checked in yet; therefore this commit defines and tests the
replay/refusal boundary but does **not** claim the two-device run passed.
