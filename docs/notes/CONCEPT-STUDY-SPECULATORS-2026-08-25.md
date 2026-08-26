# Speculators study: DSpark confidence, P-EAGLE parallel drafting, and alias-drift guards

> Studied 2026-08-25. Upstream: [vllm-project/speculators](https://github.com/vllm-project/speculators), pinned at [`0faffeb3bd547b4451a978d7aaf26a2f01b83d62`](https://github.com/vllm-project/speculators/commit/0faffeb3bd547b4451a978d7aaf26a2f01b83d62). License: Apache-2.0. No upstream source is copied here.

## Verdict

Speculators is the right current source for speculative-decoding candidate selection because it is a library for training draft models that deploy directly into vLLM, and its current front page names the exact modern seams FAK keeps rediscovering: DSpark, P-EAGLE, MTP fine-tuning, sliding-window draft attention, and hidden-state transfer for online speculator training.

The borrow is not a Python runtime port. FAK already owns the correctness boundary through target verification and rollback. The useful gaps are narrower:

- [#8938](https://github.com/anthony-chaudhary/fak/issues/8938) - make draft-depth control consume calibrated per-position confidence instead of only a scalar recent accept rate.
- [#8939](https://github.com/anthony-chaudhary/fak/issues/8939) - add a P-EAGLE-shaped parallel-depth draft-source witness before treating Qwen3.8 speculation as only sequential EAGLE/MTP.
- [#8940](https://github.com/anthony-chaudhary/fak/issues/8940) - reject speculator checkpoint alias drift before a high-training-accuracy / low-acceptance receipt is trusted.

Per-position acceptance telemetry already landed as [#8258](https://github.com/anthony-chaudhary/fak/issues/8258). Native MTP remains covered by [#5154](https://github.com/anthony-chaudhary/fak/issues/5154), stochastic lossless sampling by [#4202](https://github.com/anthony-chaudhary/fak/issues/4202), and the base EAGLE-style drafter by [#3197](https://github.com/anthony-chaudhary/fak/issues/3197).

## Value frame

- **For:** FAK's native Qwen3.8 decode path and any serving envelope where target verification can amortize more than one accepted token.
- **Problem:** scalar acceptance averages hide why suffix positions fail, and sequential draft loops leave parallel multi-depth ideas untested.
- **Today:** `internal/model` and `internal/polymodel` verify and accept drafts, but the runtime depth governor is scalar and the synthetic draft source in `cmd/polymodelbench` is sequential.
- **Better because:** DSpark exposes calibration targets for cumulative acceptance by position, while P-EAGLE exposes a direct training/runtime shape for proposing multiple depths from one pass.
- **Witness:** #8938, #8939, and #8940 each require a local Go test or bench receipt before any performance claim.

Centrality: **Core** for native decode performance; **Enabling** for training/transfer ideas. P1 managed context stays target-verified; P2 efficiency is net of drafter cost; P3 adaptation is bounded by rollback; P4 operations need explicit model/runtime/checkpoint receipts.

## Dated source ledger

Observed at `2026-08-25T21:18:01Z` via GitHub REST/API read-back, local pinned clone, and public GitHub pages.

| Surface | Pinned evidence | Observation |
|---|---|---|
| Project intent | [`README.md:53-58`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/README.md#L53-L58) | Speculators standardizes draft-model training and deployment into inference engines such as vLLM. |
| Current feature frontier | [`README.md:44-49`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/README.md#L44-L49) | The current listed frontier is hs_connectors, DSpark, P-EAGLE, MTP fine-tuning, sliding-window draft attention, and DFlash support. |
| Qwen3-8B support row | [`README.md:89-92`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/README.md#L89-L92) | Qwen3-8B has EAGLE-3, DFlash, and P-EAGLE training artifacts with vLLM deployment support. |
| Algorithm choice | [`decision_guide.md`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/docs/user_guide/algorithms/decision_guide.md) | The docs separate mature EAGLE-3 from newer P-EAGLE, DFlash, DSpark, and MTP paths. |
| DSpark calibration | [`metrics.py:85-149`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/src/speculators/models/dspark/metrics.py#L85-L149) | Acceptance is computed per position as `1 - TV`, then cumulative acceptance products and confidence calibration metrics are recorded. |
| P-EAGLE draft shape | [`core.py:20-27`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/src/speculators/models/peagle/core.py#L20-L27), [`core.py:94-204`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/src/speculators/models/peagle/core.py#L94-L204) | P-EAGLE extends EAGLE-3 with parallel multi-token prediction, generates COD sample depths, runs masked draft layers, and scores targets at sampled positions. |
| Acceptance telemetry | [`perf_utils.py:326-410`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/scripts/evaluate/perf_utils.py#L326-L410), [`run_vllm.py:40-68`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/tests/e2e/run_vllm.py#L40-L68) | The evaluation path preserves accepted/proposed counts and acceptance by draft position rather than collapsing to one mean. |
| Hidden-state transfer | [`transfer.py:46-117`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/hs_connectors/src/hs_connectors/transfer.py#L46-L117), [`mooncake_store.py:99-299`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/hs_connectors/src/hs_connectors/mooncake_store.py#L99-L299) | Online speculator training treats hidden-state exchange as a backend protocol with manifest validation and partial-write cleanup. |
| Loading failure class | [`loading.py:11-29`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/src/speculators/utils/loading.py#L11-L29), [PR #1038](https://github.com/vllm-project/speculators/pull/1038) | Final norm aliases include `llm.norm.weight` and suffix resolution chooses the shortest matching key after a bug produced low acceptance despite high training accuracy. |
| Tests | [`tests/unit/utils/test_loading.py`](https://github.com/vllm-project/speculators/blob/0faffeb3bd547b4451a978d7aaf26a2f01b83d62/tests/unit/utils/test_loading.py) | Loading alias behavior has unit coverage; FAK should mirror the receipt guard, not the Python loader. |
| Inventory | [`docs/research/inventory/vllm-project-speculators.json`](../research/inventory/vllm-project-speculators.json), [`docs/research/inventory/vllm-project-speculators.md`](../research/inventory/vllm-project-speculators.md) | Local scan: 327 files, 69 directories, 137 runtime files, 104 test files, 78 docs files, 58,757 text lines, 10 subsystems. |
| GitHub surfaces | [issues](https://github.com/vllm-project/speculators/issues), [pull requests](https://github.com/vllm-project/speculators/pulls), [releases](https://github.com/vllm-project/speculators/releases), [discussions](https://github.com/vllm-project/speculators/discussions) | Recent issues, PRs, releases/tags, and discussions were read. Latest merged PR in the bounded read-back was #1038; recent discussions included #529, #139, and #138. |

Repository metadata observed on 2026-08-25: default branch `main`; 762 stars; 200 forks on the public page; 537 commits on the public page; latest push `2026-08-25T20:59:51Z`; releases included `v0.7.0.1` (2026-08-13) and `v0.7.0` (2026-07-30). These are dated observations, not project-controlled guarantees.

## Candidate matrix

| Candidate axis | Speculators evidence | Current FAK witness | Verdict | Disposition |
|---|---|---|---|---|
| Calibrated DSpark depth control | `metrics.py:85-149@0faffeb` records analytical per-position acceptance, cumulative acceptance products, confidence loss, absolute error, predicted mean, and cumulative bias. | `internal/model/selfspecgov.go` governs `DraftDepth` from scalar economics; `internal/polymodel/specdecode.go` records an acceptance profile but no calibrated depth threshold. `fak capabilities` and `fak-dev index` found no DSpark confidence controller. | **ABSENT-on-axis** | Filed [#8938](https://github.com/anthony-chaudhary/fak/issues/8938). |
| P-EAGLE parallel-depth draft source | `core.py:20-27,94-204@0faffeb` trains sampled parallel depths with COD and masked attention. | `internal/model/specbind.go` and `cmd/polymodelbench` draft tokens sequentially; source and issue dedupe found EAGLE/MTP/n-gram work but no P-EAGLE-shaped parallel-depth witness. | **ABSENT-on-axis** | Filed [#8939](https://github.com/anthony-chaudhary/fak/issues/8939). |
| Speculator checkpoint norm-alias drift guard | `loading.py:11-29@0faffeb` and PR #1038 show `llm.norm.weight` alias drift can preserve training-looking metrics while damaging acceptance. | `internal/model/tensor_resolver.go` and `internal/model/materialize.go` require explicit canonical aliases, but no speculator receipt canary ties resolved norm/head provenance to an acceptance receipt. | **PARTIAL-on-axis** | Filed [#8940](https://github.com/anthony-chaudhary/fak/issues/8940). |
| Per-position acceptance telemetry | `perf_utils.py:326-410@0faffeb` and `tests/e2e/run_vllm.py:40-68@0faffeb` preserve position-specific acceptance. | Closed [#8258](https://github.com/anthony-chaudhary/fak/issues/8258) already added per-position proposed/accepted counts and docs. | **PRESENT** | No duplicate issue. |
| Native MTP head handling | `README.md:47@0faffeb` names MTP fine-tuning for native MTP heads. | Open [#5154](https://github.com/anthony-chaudhary/fak/issues/5154) owns DeepSeek V4 MTP depth-1; `internal/model/gguf_config.go` and loader scaffolds fence sidecar materialization until canonical slots exist. | **OWNED elsewhere** | No duplicate issue. |
| Sliding-window draft attention | `README.md:48@0faffeb` says DFlash/DSpark use sliding-window attention by default. | Current speculative issues need a draft source and depth controller first; no standalone FAK issue is justified without a native trained drafter. | **WATCH** | Carry as an acceptance/perf arm for #8938/#8939, not a separate ticket. |
| Hidden-state transfer for online training | `hs_connectors` sources validate sample manifests and abstract backends. | FAK's near-term native performance work is inference-side; online speculator training is outside the immediate serving spine. | **WATCH** | Record as process knowledge; file only if a training-data pipeline becomes the bottleneck. |

## FAK inward map

Self-query and raw source read-back split the existing surface cleanly:

- `internal/model/verify.go` already supplies target verification and rollback; every borrow remains under that losslessness boundary.
- `internal/polymodel/specdecode.go` records acceptance profiles, but the live depth decision path is still scalar.
- `internal/model/specbind.go` drafts greedily one token at a time, which is sufficient for current correctness but not a P-EAGLE parallel-depth test.
- `internal/model/tensor_resolver.go` and `internal/model/materialize.go` are the right place for canonical alias failure, while #8940 keeps speculator receipt provenance from becoming an implicit loader behavior.
- `cmd/polymodelbench` is the correct cheap witness harness before any GPU/device claim. It can prove shape, accounting, and quality parity without pretending to benchmark trained P-EAGLE weights.

## Filed and deduplicated trail

- [#8938](https://github.com/anthony-chaudhary/fak/issues/8938) - **new, open:** DSpark confidence-depth control gated on calibration.
- [#8939](https://github.com/anthony-chaudhary/fak/issues/8939) - **new, open:** P-EAGLE parallel-depth draft-source witness for Qwen3.8.
- [#8940](https://github.com/anthony-chaudhary/fak/issues/8940) - **new, open:** speculator checkpoint norm-alias drift canary.
- [#3197](https://github.com/anthony-chaudhary/fak/issues/3197#issuecomment-5416891936) - **parent updated:** native EAGLE-style draft head now links the three Speculators follow-ons.
- [#8923](https://github.com/anthony-chaudhary/fak/issues/8923#issuecomment-5416891900) - **planning issue updated:** Qwen3.8 borrowed-hot-path work now points at the three concrete Speculators gaps.
- [#8258](https://github.com/anthony-chaudhary/fak/issues/8258), [#4877](https://github.com/anthony-chaudhary/fak/issues/4877), [#5154](https://github.com/anthony-chaudhary/fak/issues/5154), [#5261](https://github.com/anthony-chaudhary/fak/issues/5261), and [#4202](https://github.com/anthony-chaudhary/fak/issues/4202) were read as dedupe boundaries.

## Completion boundary

This study does not claim DSpark, P-EAGLE, MTP, or alias-drift protection are implemented in FAK. It only converts current upstream practice into three exact FAK tickets with pinned sources and local seams. Any speedup claim still needs matched model, tokenizer, weights, target verifier, sampling policy, device, workload, and quality evidence.

## Executable candidate frontier: P-EAGLE parallel-depth source (#8939)

`cmd/polymodelbench -selfcheck` now includes the named `p-eagle-parallel-depth-shape` arm. It makes one logical draft-source call per speculative round, returns `num_depths` future positions together, and feeds that chain to the existing fak-native `polymodel.SpecDecode` + `model.Session.VerifyForward` verifier. The `-bench -out` receipt records the actual engine (`fak-native/internal/model`), target and draft-source provenance, target verify rounds, the current sequential co-resident drafter's target rounds and draft-step cost, sequential steps avoided by the parallel source shape, positional `AcceptanceProfile`, mean acceptance length, and greedy token identity.

This is an **optional draft-source candidate under #3197**, not a shipped speedup. Its proposer is a deterministic synthetic oracle used only to isolate shape and cost accounting. It contains no trained P-EAGLE weights, COD trainer, Qwen3.8 checkpoint result, CUDA run, latency measurement, or GPU claim. A future live claim requires a real Qwen3.8 P-EAGLE checkpoint on a sanctioned node with the receipt still naming the fak-native engine and checkpoint provenance.
