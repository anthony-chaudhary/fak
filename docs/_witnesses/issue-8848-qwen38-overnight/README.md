# Issue #8848 — Qwen3.8-27B overnight hill climb

**Verdict: `HOLD_BELOW_PARITY`.** The campaign retains 120 unique, quality-passing GCP A100 reference-runtime experiments and two quality-constrained `KEEP` decisions, but the matched-enough native check disproves current fak-native parity. The next experiment is to profile and fix cached Qwen GDN session-state restore before rewriting another kernel.

## Results

| Envelope | Arms | Rows | Quality | Median latency | Fold |
|---|---|---:|---:|---:|---|
| Qwen3.8-27B BF16, 2×A100 40 GB, vLLM reference | thinking on → off | 30 + 30 | 60/60 | 3855.989 → 484.570 ms | `KEEP` off (87.43% lower) |
| Qwen3.8-27B FP8, 1×A100 40 GB, vLLM reference | thinking on → off | 30 + 30 | 60/60 | 3876.030 → 502.393 ms | `KEEP` off (87.04% lower) |

These 120 rows are labeled `engine=vllm-reference`. They establish a useful arithmetic-workload setting inside each stated envelope; they are **not** fak-native performance or parity evidence.

The fak-native CUDA check used Qwen3.8-27B Q4_K_M artifact SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. Its identity log records `backend=cuda`, `forward_path=cuda/qwen35-gdn-ssm-decode-v1`, and `q4k=true`, with no delegated reference-runtime marker. All five exact `Q38` answers passed. The first uncached three-token decode reached 12.1 tok/s; four identical cache-hit requests then fell to 0.3 tok/s while preserving correctness.

For explicit parity/reference diagnosis only, a pinned llama.cpp reference (`171974745`) used the same artifact, A100 class, and three-token output count. Its five decode samples were 30.4091, 36.2434, 36.8949, 36.7359, and 36.5491 tok/s (median 36.5491). Against the cached native median, the recorded ratio is 0.0082, so parity is held. Request/template overhead and quality execution differ, making this sufficient to reject parity—not to claim a precise broad speed ratio.

## Hardware coverage

- **GCP A100:** live 120-row BF16/FP8 reference fold plus the fak-native Q4_K_M and pinned llama.cpp checks above.
- **MacBook:** existing native-Metal acceptance is retained at [`../qwen38-27b-2026-08-20/`](../qwen38-27b-2026-08-20/). A separate llama.cpp Metal campaign is retained at [`../issue-8360-qwen38-mac-metal/`](../issue-8360-qwen38-mac-metal/). Neither is counted among these 120 newly run rows.
- **Private accelerator route:** the sanctioned bridge was made ready and a fresh persistent session was created. Inventory had not produced an accepted Qwen3.8 experiment row at this fold; private node/session details are intentionally omitted.
- **Control laptop:** orchestration, hashing, and independent artifact readback only; it has no valid local accelerator/model datum and contributes no fabricated experiment row.

## Artifacts and validation

- `gcp-a100-bf16-reference.jsonl`: 60 immutable BF16 reference rows; SHA-256 `15c4d8e678510b36e5d3377a92ba17e495658a58f037d224891bfa0f1c5a71b8`.
- `gcp-a100-fp8-reference.jsonl`: 60 immutable FP8 reference rows; SHA-256 `fcb4a4bbc37549443fe9d548f5d1b6eda4effd0fe1f7e57608e794f1602c6328`.
- `fold-report.json`: deterministic aggregate, hill-climb decisions, parity hold, and exactly one next experiment.
- `fak-native-q4km-a100-rows.json` and `fak-native-q4km-a100-identity.txt`: scrubbed native observations and engine identity.
- `llamacpp-q4km-a100-bench.json`: explicitly labeled reference-runtime measurements.
- `witness_test.go`: independently re-reads the row ledgers, checks hashes, count/ID uniqueness, engine separation, quality, arm statistics, folds, native identity, exact artifact identity, and parity hold.

Run the witness from the repository root:

```bash
go test ./docs/_witnesses/issue-8848-qwen38-overnight
```

## Next experiment

Profile session-state/cache restore around the fak-native Qwen GDN recurrent state, fix the cache-hit decode collapse, then rerun a matched native-versus-reference quality envelope. No result here authorizes fallback to llama.cpp or vLLM.