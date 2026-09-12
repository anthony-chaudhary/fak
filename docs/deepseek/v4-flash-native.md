---
title: "DeepSeek V4 Flash native readiness"
description: "Evidence-bounded status for native DeepSeek-V4-Flash-0731 support in fak."
---

# DeepSeek V4 Flash native readiness

**Status on 2026-09-09:** the native Flash component tests and the broader DeepSeek/V4 regression tests pass under [fak issue #12659](https://github.com/anthony-chaudhary/fak/issues/12659). `fak` does not yet provide a verified full-token generation path for DeepSeek-V4-Flash-0731.

## Pinned public reference

The compatibility target is the official [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) repository at immutable revision [`7872f01b1d1fe23eabc4c98b48bffcef5a386062`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062). The exact sources used for compatibility work are its pinned [`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json), [reference inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py), [tokenizer code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding/encoding_dsv4.py), and [MIT license](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/LICENSE).

The published configuration identifies `DeepseekV4ForCausalLM` with `model_type: deepseek_v4`, 43 decoder layers, 64 attention heads, 256 routed experts, one shared expert, six selected experts, three initial hash-routed layers, `index_topk: 512`, and `sqrtsoftplus` routed scoring. The reference selects one compression ratio for each layer (`0`, `4`, or `128`) together with sliding-window attention; it does not run CSA and HCA simultaneously in every layer. Its mHC path carries four streams. These details correct older backlog assumptions based on the 384-expert Pro shape, sigmoid routing, or uniform compressed attention. The shared-expert and four-stream mHC execution paths remain missing in `fak`.

**V4.1 source refresh, 2026-09-12:** official [`DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277) is now published. Its [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json) declares a `deepseek_v41` wrapper with nested `deepseek_v41_text`, 40 layers, width 5120, 384 routed experts, Engram, shared KV/index source layers, and 32-by-32 dense FP8 blocks. These are separate from the V4 Flash 0731 profile described here. The earlier artifact-availability hold is superseded; native V4.1 generation remains unverified. The official 48 weight shards total 510,296,708,312 bytes, so metadata or small-fixture tests do not demonstrate a full checkpoint run.

## What the component milestone proves

Issue #12659 adds admission tests for the official Flash configuration and rejection of mixed Flash/Pro tuples. These passing tests establish that the loader recognizes the pinned configuration without silently applying the Pro shape.

The native routed-expert tests exercise routing, selected-expert tensor reads, dequantization, SwiGLU expert execution, and weighted expert composition with small fixtures. The tests also exercise the real Flash packed dimensions: `w1`/`w3` weights `[2048,2048]` with scales `[2048,128]`, and `w2` weights `[4096,1024]` with scales `[4096,64]`. They pin signed FP4 nibble decoding and the 32-value scale boundary while still using deterministic test bytes.

The full `internal/model` suite remains red from four Qwen35 failures reproduced on the unchanged base commit `2cb0d145dfcf9fadf5b5ad7f29811ed93dada6c5`: CPU continuation produces NaNs, and three batched/chunked linear-attention comparisons differ. The related work is tracked in [#12433](https://github.com/anthony-chaudhary/fak/issues/12433) and [#11987](https://github.com/anthony-chaudhary/fak/issues/11987). The scoped Flash and DeepSeek/V4 tests, plus `go vet ./internal/model`, pass; this is not a claim that the whole package is green.

Run the focused component tests on Windows with:

```powershell
.\test.ps1 ./internal/model -run TestDeepSeekV4Flash -count=1
```

On Unix-like systems, run:

```sh
go test ./internal/model -run TestDeepSeekV4Flash -count=1
```

The checked-in configuration fixture is an exact, small metadata fixture derived from the pinned official `config.json`. The expert tests use small synthetic or test-only tensor payloads. They do not contain, download, or execute the full 166 GB-class checkpoint, and a passing component suite is not evidence that the model can generate a token.

## Missing full-token work

A complete native decoder path still needs the checkpoint's per-layer attention behavior and tensor bindings, including:

- mHC pre/post four-stream mixing and head collapse;
- the per-layer compression schedule and the compressed sparse/hybrid attention path;
- grouped low-rank output projections; and
- end-to-end embedding, decoder, normalization, and language-model head execution against the official tensor layout.

DSpark speculative decoding is a separate stage. Disabling or skipping DSpark has not yet been proven to produce a correct target-model token through `fak`, so there is no supported native generation command to document yet.

## Next milestones

1. Restore the broader package gate through the existing Qwen35 work while preserving the passing Flash component tests.
2. Validate the pinned tensor manifest and tensor names without requiring a full weight read.
3. Add an official-layout reference fixture for mHC, per-layer compression, grouped output projection, and next-token parity.
4. Load the full pinned checkpoint and produce a bounded, reproducible target-model token with DSpark disabled, recording a parity receipt.
5. Qualify DSpark independently, including its three published MTP stages and acceptance behavior.
