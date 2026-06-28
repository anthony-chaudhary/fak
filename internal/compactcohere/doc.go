// Package compactcohere is the coherence policy for the TWO context managers that
// stack, blind to each other, on the flagship `fak guard -- claude` boundary:
//
//   - the KERNEL manager — fak's cache-PRESERVING levers: the byte-splice history
//     compaction that keeps the provider prompt-cache prefix verbatim
//     (agent.CompactAnthropicHistory, epic #745), the budget-triggered RESET
//     (internal/sessionreset + gateway.ResetOnBudget), the cut-vs-reset hybrid
//     (gateway.ResetScore, #774), the inbound prompt-MMU (internal/promptmmu, #751),
//     and the ctxmmu quarantine/seal of poisoned spans.
//   - the HARNESS manager — Claude Code's OWN auto-compaction: when the conversation
//     nears the context window it summarizes the history and re-emits a brand-new,
//     shorter messages[] array, which REWRITES the cache_control prefix fak depends on.
//     It is cache-DESTROYING (every breakpoint moves → a provider cache_creation burst),
//     it cannot be reliably disabled via settings/env (autoCompactEnabled is silently
//     ignored; the threshold is clamped ~83%; anthropics/claude-code #38483/#42817/#42149),
//     and the ONE dependable lever to suppress it is a PreCompact HOOK that exits 2.
//
// The intersection this package owns is the question the existing levers never ask:
// "is the harness fighting us, and what should fak DO about it this turn?" cachemeta's
// prefix_stability/prefix_coherence detect THAT a prefix broke and break it on PURPOSE
// for world-staleness; gateway.ResetScore chooses cut-vs-reset within fak's own levers.
// Neither ATTRIBUTES a break to the harness-as-second-compactor, reasons about the
// wall-clock prompt-cache TTL, or decides whether to SUPPRESS the harness manager. This
// package is that missing sensor + policy, kept pure so the whole decision is unit-tested
// without a gateway, a provider, a child process, or a clock.
//
// Two surfaces:
//
//   - Classify(prev, cur, ttl) attributes one served turn's prefix event — STABLE,
//     FAK_CUT, FAK_WORLD_BREAK, HARNESS_REWRITE, or COLD_TTL — from cheap, content-free
//     facts the gateway already has (the inbound protected-prefix digest delta, fak's own
//     compaction outcome, the provider cache_read/cache_creation counters, the idle gap).
//   - Coordinator carries the rolling state and emits a per-turn Decision plus a STANDING
//     Posture (block/allow) that a Claude Code PreCompact hook enforces: block the
//     harness's cache-destroying auto-compaction WHILE fak is coping, and yield the net
//     back to the harness once fak's own compaction has bailed for a sustained streak (so
//     suppressing the harness can never strand a session into a hard context overflow).
//
// It is tier-1 (foundation): stdlib-only, imports nothing internal, off the hot path. It
// takes digests as opaque strings (the caller hashes the bytes), so it never touches a
// prompt and a shadow log of any Decision is content-free.
package compactcohere
