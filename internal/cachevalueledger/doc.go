// Package cachevalueledger provides a durable, append-only ledger for cache-value
// observations from fak sessions (run/guard/serve). Each row records a session's
// cacheobs snapshot (turns, prompt_tokens, reused_tokens, reuse_ratio, etc.) so the
// accumulated ledger can be scored by `fak vcache score --ledger` to detect
// cache-value regressions. This is the continuous-dogfood side of epic #1072.
package cachevalueledger