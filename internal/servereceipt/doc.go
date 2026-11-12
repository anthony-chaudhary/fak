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
//	  "state_tp_sharded": true                   // true: per-seq state divides by TP
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
// # Receipt schema — fak.serve-receipt.receipt/1 (JSON output)
//
// The receipt embeds the input facts and config verbatim plus the full
// arithmetic block and verdict, so `validate` can recompute EVERY number from
// the embedded inputs alone and diff field-by-field. Receipts carry no
// wall-clock timestamp (only the facts' observed_at) so identical inputs
// render byte-identical receipts and golden tests stay deterministic.
package servereceipt
