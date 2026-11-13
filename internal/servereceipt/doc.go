// Package servereceipt renders on-demand model→GPU serving-config receipts
// (issue #14362): given a model's measured/cited facts and a target GPU
// profile, plan whether a TP×PP serving layout fits, show every intermediate
// byte of the arithmetic, and emit a receipt a peer agent can re-verify
// offline with `fak serve-receipt validate`.
//
// # Facts schema — fak.serve-receipt.facts/1 (JSON input)
//
//	{
//	  "schema": "fak.serve-receipt.facts/1",
//	  "model": "nvidia/Kimi-K3.1-Pro-NVFP4",
//	  "weights_bytes": 1561000000000,            // total checkpoint bytes (decimal)
//	  "observed_at": "2026-11-12",               // date the facts were read
//	  "source_urls": ["https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4"],
//	  "notes": ["provisional weight total; reconcile exact shard sum"],
//	  "num_attention_heads": 0,                  // optional; 0 = unknown → skip divisibility check
//	  "kv": {                                    // per-token KV/latent cache shape (MLA-style latent + indexer)
//	    "full_attn_layers": 24,                  // layers carrying full-attention KV state
//	    "kv_lora_rank": 512,                     // latent KV compression rank
//	    "qk_rope_head_dim": 64,                  // per-token RoPE key dim carried alongside the latent
//	    "indexer_key_dim": 128,                  // per-token indexer key dim
//	    "latent_dtype_bytes": 1,                 // bytes/element of the latent cache dtype
//	    "indexer_key_dtype_bytes": 1,            // bytes/element of the indexer key dtype
//	    "indexer_scale_bytes": 4,                // per-token indexer scale bytes
//	    "tp_replicated": true                    // true: each TP rank holds the full per-token state (no TP divide)
//	  },
//	  "state_per_seq_bytes": 464601000,          // non-KV per-sequence reserved state
//	  "state_tp_sharded": true,                  // true: per-seq state divides by TP
//	  "kv_variants": {                           // OPTIONAL (issue #14438): per-variant per-token component bytes
//	                                             // for ONE full-attention layer, overriding the derived built-ins
//	    "fp8":               {"latent_bytes": 576, "indexer_key_bytes": 128, "indexer_scale_bytes": 4},
//	    "nvfp4":             {"latent_bytes": 352, "indexer_key_bytes": 132, "indexer_scale_bytes": 0},
//	    "nvfp4_mxfp4keys":   {"latent_bytes": 352, "indexer_key_bytes": 68,  "indexer_scale_bytes": 0}
//	  },
//	  "routed_experts": 896,                     // OPTIONAL: routed-expert count (enables the exact expert
//	                                             // offload-rejection arithmetic when a weights-INFEASIBLE verdict fires)
//	  "experts_per_token": 16,                   // OPTIONAL: experts activated per token
//	  "expert_bytes_packed": 1531000000000,      // OPTIONAL: routed-expert weight bytes (scales included when packed together)
//	  "expert_scale_bytes": 0                    // OPTIONAL: separate scale-tensor bytes when stored apart
//	}
//
// # Units (one convention, labeled everywhere)
//
// VRAM per GPU is read as GiB (2^30 bytes) — vendor "GB" listings are treated
// as GiB, the same convention pkg/hwspecs carries. Weights, KV, and state
// sizes are exact bytes. The activation reserve flag is decimal GB (10^9
// bytes). Every rendered receipt carries the UnitsNote string so a reader
// never has to guess which reading produced a number.
//
// # Offload-aware capacity model (issue #14438)
//
// The v1 plan priced device-only KV. The extended plan adds a host KV pool
// (--host-kv-gib), a KV cache variant (--kv-variant), decode-context
// parallelism (--dcp), and the physical capacity they unlock:
//
//	physical_tokens = floor((device_pool_B + usable_host_B) / (kv_per_token_per_gpu / dcp))
//
// where device_pool_B is ONE GPU's device KV leftover (usable budget minus
// reserved state minus activation reserve, clamped ≥ 0) and usable_host_B is
// the per-DCP-rank share of the host pool: host_total × usability ÷ dcp. The
// offloaded cache mirrors the device layout, so with DCP each rank stores 1/N
// of each token on host too — keeping both sides of the ratio per-rank. With
// the host pool at 0 and dcp 1 this reduces exactly to the v1
// max_cached_tokens. The usability default (0.5) is calibrated to the S8
// production-runtime anchor (16×RTX PRO 6000, 32 GiB configured → ~1.02M
// physical FP8 KV tokens observed); the mechanism (per-block metadata,
// alignment, allocator overhead in the native host backend) is not directly
// measured. A device KV shortfall no longer kills the plan: when the host
// pool covers the request the verdict is OFFLOAD_REQUIRED (exit 0, marked) —
// decode reads miss to host at the miss-rate the --curve block prices.
// Routed-expert WEIGHT offload is rejected with arithmetic, not assumed
// viable: every forward activates a large union of experts across the batch,
// so the per-forward host read (union_bytes ÷ host bandwidth) dwarfs a decode
// step; the warning states the seconds-per-forward and keeps KV offload as
// the only viable offload surface.
//
// # Receipt schema — fak.serve-receipt.receipt/1 (JSON output)
//
// The receipt embeds the input facts and config verbatim plus the full
// arithmetic block and verdict, so `validate` can recompute EVERY number from
// the embedded inputs alone and diff field-by-field. Receipts carry no
// wall-clock timestamp (only the facts' observed_at) so identical inputs
// render byte-identical receipts and golden tests stay deterministic.
package servereceipt
