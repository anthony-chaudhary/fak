---
title: "fak proof: transformer forward-pass HF parity"
description: "Oracle-parity proof for fak's pure-Go transformer forward pass: hidden-state cosine, per-position argmax, and greedy-id match against the HuggingFace reference."
---

# N7 · model/forward-parity

**Regime N (Numerical / linear-algebra).** The `internal/model` package implements a from-scratch, pure-Go transformer forward pass — embedding lookup, the decoder stack (RMSNorm/LayerNorm, RoPE, causal multi-head/GQA/MLA attention, SwiGLU/MoE FFN, residuals, topology-selected norm placement), final norm, and the tied LM head — producing per-layer hidden states and per-position vocab logits. `internal/modelengine` wraps a loaded `Model` for decode/generation. "Correct" here means **oracle parity**: the Go forward pass must reproduce the independently-computed PyTorch/HF reference (exported by `export_oracle.py` to `.cache/oracle-*` / `.cache/<model>`). Per witness-kind §3.1 of [00-METHOD.md](00-METHOD.md), a float forward pass is correct when **hidden-state cosine ≈ 1.0**, the **per-position argmax matches the oracle at every position** (a single ULP of drift cannot flip an argmax the oracle pins), and the **greedy continuation ids match token-for-token**. This doc discharges (1) end-to-end forward parity against the HF oracle, and (2) whether that parity holds across the supported arch families.

> Witness command (this node, native go1.26 darwin/arm64):
> `go test -run 'Oracle|Parity|Greedy|Argmax|Forward' ./internal/model/ -count=1 -timeout 240s -v`
> Package result: `ok github.com/anthony-chaudhary/fak/internal/model 4.816s`. The only oracle fixture present on this node is `internal/model/.cache/smollm2-135m` (`model_type=llama`, 30 layers); every `.cache/oracle-*` family export is absent, so those rungs **SKIP**.

---

## THEOREM 1 — end-to-end forward pass matches the HF oracle (loaded llama family)

**THEOREM.** For the loaded HF oracle (smollm2-135m, `model_type=llama`), the pure-Go `Forward` pass reproduces HF's per-layer hidden states (cosine ≈ 1.0 at the embedding, an early/mid/last decoder layer, and the final-norm slot), matches HF's per-position **argmax at EVERY position** (`go == hf == oracle`), reproduces final logits to `< 0.05` max abs diff, and the cached `NewSession().Prefill` last-position logits agree with cacheless `Forward` within `1e-4`.

**REGIME.** N — oracle parity (cosine + argmax-pin + greedy-id), the gold witness of §3.1.

**PROOF.** `Forward` (`internal/model/forward.go:45`) does the embedding lookup, runs the decoder stack appending one hidden slot per layer (`forward.go:58`, `forward.go:67`), then `finalNorm` (`internal/model/weights.go:1478`) + the tied-head matmul (`forward.go:72-78`) to logits. The witness `assertForwardMatchesHFOracleMode` (`internal/model/oracle_test.go:1016`) reads HF's exported `.hidden.f32` / `.logits.f32`, computes `cosine` (`oracle_test.go:139`) and `Errorf`s when `cs < 0.9999` (`oracle_test.go:1059-1061`), asserts `gotAM == refAM == p.ArgmaxPos[pos]` at every position (`oracle_test.go:1066-1074`), checks last-position logit `max|Δ| ≤ 0.05` (`oracle_test.go:1079-1081`), and checks cached `Prefill` within `1e-4` (`oracle_test.go:1082-1087`). With the smollm2 fixture present, no `Errorf` fired.

**WITNESS.** `go test -run 'Oracle|Parity|Greedy|Argmax|Forward' ./internal/model/ -count=1 -timeout 240s -v` → `TestForwardMatchesHFOracle` / `…/smollm2-135m`. Greedy-id parity is additionally pinned by `TestForwardOnGoDecodedWeights` (`internal/model/safetensors_test.go:77`), which runs `Generate` on Go-decoded (no-torch) weights and asserts each id equals HF's: logged `go-decoded greedy=[7042 30 7042 314 …]` ≡ `hf greedy=[7042 30 7042 314 …]`.

**VERDICT.** **PROVEN** (2026-06-20). `--- PASS: TestForwardMatchesHFOracle (1.79s)` and `…/smollm2-135m (1.79s)`; log shows `cos=1.000000` at embedding/layer1/layer15/layer29/final-norm for all 3 prompts and `logits[last] max|Δ|=6.008e-05 argmax=7042 ✓` (and `4.768e-05`/`4.339e-05` for prompts 1/2). `--- PASS: TestForwardOnGoDecodedWeights (0.63s)`.

**DOS.** bound at ship (audit the proof-shipping commit with `dos commit-audit` from the fleet repo root; mechanism last shipped under `66ccb27` (forward.go) and witness tier `4c3e150` (oracle_test.go)).

---

## THEOREM 2 — parity holds across the supported arch families

**THEOREM.** The same forward-pass HF-oracle parity (hidden cosine ≈ 1, per-position argmax, greedy ids) holds not only for `llama` but across the supported arch families: `qwen3` (QK-norm), `qwen3moe` (hybrid dense/sparse), `gpt_oss` (MXFP4 MoE), `gemma3` (local/global attention, sandwich norm), `glm_moe_dsa` (MLA + Dynamic Sparse Attention + MoE), `mistral` (sliding-window), `llama3` (rope scaling + EOS list), `phi3` (longrope), `deepseek-v2` (MLA).

**REGIME.** N — oracle parity per family.

**PROOF.** The witnesses **exist** and each routes through the same `assertForwardMatchesHFOracle` assertion (`internal/model/oracle_test.go:1011`): `TestOptionalQwen3OracleCoversQKNorm` (`oracle_test.go:304`), `TestOptionalQwen3MoEOracleCoversHybridDenseSparseLayers` (`:330`), `TestOptionalGPTOSSMXFP4ForwardMatchesHFOracle` (`internal/model/safetensors_stream_test.go:488`), `TestOptionalGemma3OracleCoversLocalGlobalAttention` (`:374`), `TestOptionalGLMMoeDsaOracleForwardMatchesHFCacheless` (`:814`) and `…SessionCacheMatchesHF` (`:824`), `TestOptionalMistralSWAOracleNonVacuous` (`:260`), `TestOptionalLlama3OracleCoversScalingAndEOSList` (`:286`), `TestOptionalPhi3LongropeOracleCoversLongFactor` (`:976`), `TestOptionalDeepSeekV2OracleDocumentsMLABoundary` (`:435`). `Forward` really dispatches `glm_moe_dsa` to `layerGLMDsa` (`forward.go:62`, body at `forward.go:130`) and all other families to `layer` (`forward.go:65`), so the per-family code paths are genuine, and each Optional rung additionally asserts the family-distinguishing config axes (QK-norm tensors, MoE dense/sparse split, MLA q_a/q_b geometry, DSA indexer shapes, sliding-window non-vacuity, longrope factor) before the numeric compare. **But the numeric parity is not witnessed here:** on this node only `internal/model/.cache/smollm2-135m` (a `llama` export) is present; every `.cache/oracle-<family>` export is absent, so every Optional rung **SKIPPED** with `no exported weights in .cache/oracle-<family>`. No deterministic witness for a **non-llama** family ran green, so cross-family parity is honestly un-witnessed.

**WITNESS.** Same command as Theorem 1; the family rungs are the `TestOptional*` set. The helper tests `TestOracleMatrixDirsFromEnv` and `TestMissingOracleFamilies` PASS but assert env/string family-matrix derivation, **not** numeric parity, so they do not close this theorem.

**VERDICT.** **OPEN** (2026-06-20). All non-llama family rungs SKIP for want of fixtures (e.g. `--- SKIP: TestOptionalQwen3OracleCoversQKNorm … no exported weights in .cache/oracle-qwen3`; likewise qwen3moe, gpt_oss, gemma3, glm (6 GLM rungs), mistral, llama3, phi3, deepseek-v2). Only the `llama` family (smollm2-135m) is actually witnessed green here. **To close:** export the per-family tiny oracles — `python internal/model/export_oracle.py --out .cache/oracle-qwen3 …` for each, and the GLM rung via the documented `--online --trust-remote-code --model yujiepan/glm-5-tiny-random --out .cache/oracle-glm --prompt-ids-json '…'` hint — then re-run the command so the Optional rungs execute and corroborate cosine/argmax/greedy parity per family. This is an honest OPEN, not a false PROVEN: the llama rung being green is necessary but not sufficient for the family-coverage claim.

**DOS.** bound at ship (no PROVEN to bind; the OPEN is recorded as the residual `dos review` should surface — closing it requires the fixture exports above, which are gitignored and absent on this node).

---

## THEOREM 3 — Qwen3.6 (hybrid Gated-DeltaNet) multi-token greedy parity — **REFUTED**

**THEOREM (under test).** fak's in-kernel greedy decode of Qwen3.6-27B (`qwen35`, hybrid GDN: 48 linear-attn + 16 full-attn layers) reproduces the llama.cpp reference greedy continuation **token-for-token** on the same `q4_k_m` GGUF — i.e. multi-token greedy parity, the decode-side analogue of Theorem 1's llama greedy-id parity.

**REGIME.** N — oracle parity (greedy-id match against an independent engine).

**PROOF / counterexample.** The theorem is **false as stated**. On the exact 22-token ChatML oracle prompt (`You are a helpful assistant.` / `Say OK.`, ids `[248045,8678,198,…,74455,198]`) at `temperature=0, top_k=1`, fak and the oracle agree on tokens 0–1 (`248068 <think>`, `198 \n`) and **diverge at token index 2 (the third token)**: fak emits `8160` (`Here`, logit **23.18**, with `90700` second at 21.43), the oracle emits `90700` (`Thinking`, logprob **−0.547**, with `8160` second at −0.945). Both engines surface the **same top-2 set `{8160, 90700}` ranked oppositely** — a **near-tie argmax flip** at the decision boundary, the signature of accumulated f32 drift (token 0's **argmax matches** — fak logit 28.30, the oracle pins it at logprob −0.0008/near-certain; the **first token is at parity**, the third is not). The drift's root cause (the f32 Gated-DeltaNet recurrent scan, the int8 SDOT decode reduction rounding, or attention/softmax accumulation order vs llama.cpp's Metal kernels) is **not yet isolated** — this row refutes the *parity claim*, it does not diagnose it. Note this is a **strictly harder** claim than Theorem 1 (a 27B hybrid-GDN multi-token continuation, vs a 135M llama single-step + short greedy) and does **not** contradict Theorem 1's PROVEN.

**WITNESS (counter-witness, deterministic, weight-backed / "Optional").** The captured artifact pair is the recorded counterexample:
`fak/experiments/qwen36/native-gguf-q8-multitoken-parity-20260619.json` (`expect_match:false`, `generated_ids:[248068,198,8160]` vs `expected_ids:[248068,198,90700]`) and the oracle `…/llamacpp-qwen36-multitoken-oracle-20260619.json` (`llama-server -m Qwen3.6-27B.q4_k_m.gguf -ngl 99`, `tokens:[248068,198,90700]`). Re-run on a node where the gated 27B GGUF is present — `-expect` self-checks and exits non-zero on the mismatch:
```bash
go -C fak run ./cmd/qwen35check -gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf -n 3 -topk 5 \
  -ids 248045,8678,198,2523,513,264,10631,17313,13,248046,198,248045,846,198,44240,10092,13,248046,198,248045,74455,198 \
  -expect 248068,198,90700
```

**VERDICT.** **REFUTED** (2026-06-20, this M3 Pro node, q4_k_m GGUF). fak's shipped decode is *forward-parity-PROVEN on llama* (Theorem 1) and *first-token-parity on Qwen3.6*, but its **greedy continuation on the hybrid-GDN model fails at token 3** — so fak is **not at correctness parity yet** on this arch. Recorded as a finding, not deleted (00-METHOD §4). Full status + the MLX-bar / swap-contamination measurement caveats: [`../../experiments/qwen36/QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/qwen36/QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md).

**DOS.** bound at ship (the REFUTED row binds to the commit that records it; the counter-witness artifacts are committed under `fak/experiments/qwen36/`).

---

### Honest scope notes

- The smollm2 fixture resolves at `internal/model/.cache/smollm2-135m/` (test cwd is the package dir; `resolveOracleDir` also falls back to the repo-root copy, `oracle_test.go:102`). Its `config.json` is `model_type: llama`, 30 layers — so Theorem 1's PROVEN is specifically the **llama** family.
- `-short` skips the weight-backed rungs by design (`oracle_test.go:80-82`); this run was full (no `-short`), so the llama oracle was the real witness.
- `internal/modelengine` was in scope; its decode tests (`TestCompleteRunsRealDecode`, `TestDecodeIsDeterministicAndInputDriven`) wrap the same `Model` but witness decode determinism/engine wiring, not oracle forward parity, so they are not load-bearing for N7's two theorems and are left to the modelengine proof obligation.
