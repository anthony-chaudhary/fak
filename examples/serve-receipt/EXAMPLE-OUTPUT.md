# fak serve-receipt — nvidia/Kimi-K3.1-Pro-NVFP4

**Verdict: OK**

GPU: rtx-pro-6000-blackwell-96gb (vram 96 GB, sm 12.0, fp4_native true, nvlink false, pcie 5.0 x16) · config: tp 8 × pp 3 = 24 GPUs, max_num_seqs 32, gpu_mem_util 0.9, activation_reserve_gb 2, max_model_len (none), kv_variant fp8

## Arithmetic

units: VRAM per GPU read as GiB (2^30 B) — vendor "GB" listings treated as GiB (pkg/hwspecs convention); weights/state/KV bytes exact; activation reserve in decimal GB (10^9 B)

- gpus = tp 8 × pp 3 = 24
- weights_per_gpu = ceil(1,605,125,645,632 B / 24) = 66,880,235,235 B
- kv_bytes_per_token_per_gpu = 24 × (576 + 128 + 4) (facts.kv_variants[fp8]) = 16,992 B (kv.tp_replicated=true → no TP divide)
- state_per_gpu_per_seq = 464,601,000 B / tp 8 = 58,075,125 B (state_tp_sharded=true)
- vram_per_gpu = 96 GiB = 103,079,215,104 B × util 0.9 = 92,771,293,593 B
- usable_per_gpu = 92,771,293,593 − 66,880,235,235 = 25,891,058,358 B
- reserved_state = 58,075,125 B × max_num_seqs 32 = 1,858,404,000 B
- activation_reserve = 2 GB = 2,000,000,000 B
- max_cached_tokens = max(0, floor((25,891,058,358 − 1,858,404,000 − 2,000,000,000) / 16,992)) = 1,296,648
- kv_variant fp8: 24 × (576 + 128 + 4) (facts.kv_variants[fp8]) → 16,992 B/token/GPU
- device_pool = 25,891,058,358 − 1,858,404,000 − 2,000,000,000 = 22,032,654,358 B (clamped ≥ 0)
- physical_tokens = floor((device_pool 22032654358 B + usable_host 0 B) / (kv 16992 B/token ÷ dcp 1)) = 1,296,648
- requested max-model-len: (none) → verdict OK

## Warnings

- wide TP all-reduce over PCIe; PP-heavy layouts preferred (measured: TP4 6–7 tok/s vs TP2+PP2 46–49 on 397B MoE, flashinfer#2577)

## Provenance

- observed_at: 2026-11-12
- sources:
  - https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4
  - https://huggingface.co/api/models/nvidia/Kimi-K3.1-Pro-NVFP4?blobs=true
- notes:
  - exact shard sum: 96 safetensors shards at HF rev 3af0462 = 1605125645632 B (1605.13 GB / 1494.89 GiB)
  - DSA sparse indexer has 32 heads across 24 full-attn layers; TP must divide both 96 attention heads and 32 indexer heads

## Witness checklist

1. load witness: checkpoint loads to completion on this TP×PP layout without OOM
2. 3-prompt accuracy spot-check on the served model
3. vllm bench serve sweep at the planned max-num-seqs
4. KV-headroom log cross-check against max_cached_tokens
