// Package cachevalueledger provides a durable, append-only ledger for cache-value
// observations from fak sessions (run/guard/serve). Each row records a session's
// cacheobs snapshot (turns, prompt_tokens, reused_tokens, reuse_ratio, etc.) so the
// accumulated ledger can be scored by `fak nightrun score` to detect cache-value
// regressions. The gate reports the WITNESSED realized KV-prefix reuse ratio over
// multi-turn sessions (#1066 fences the vs-naive re-prefill multiple), and reports
// INSUFFICIENT rather than failing on a thin corpus. This is the continuous-dogfood side
// of epic #1072.
package cachevalueledger
