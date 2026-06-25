// Package promptmmu is the cache-prefix-preserving inbound prompt MMU: the
// INGRESS dual of the result-side ctxmmu.
//
// ctxmmu gates what comes BACK (tool results: quarantine / page-out). promptmmu
// gates what goes IN (the tool definitions the model is offered). The kernel
// already adjudicates which tool CALLS are denied; a denied tool can never be
// invoked, so dropping its DEFINITION from the advertised tools[] is pure upside
// — the model loses nothing it could have done, and the request carries fewer
// uncached bytes upstream.
//
// CompactInboundTools is a byte-level twin of agent.CompactAnthropicHistory
// (the #555 messages[] compactor), transposed from the messages[] array to the
// tools[] array. It splices on the ORIGINAL bytes so the cached prefix is copied
// verbatim (a memcpy), never re-marshalled — re-marshalling reorders JSON keys
// and destroys the prompt-cache hit. The load-bearing invariant:
//
//	The protected prefix = every byte from offset 0 through the END of the LAST
//	tools[] element that carries a cache_control breakpoint. Only whole tool
//	elements STRICTLY AFTER that index may be dropped. On ANY ambiguity the
//	function returns its input UNCHANGED (fail-safe identity), carrying a named
//	SkipReason so an identity result is auditable, never a silent no-op.
//
// CACHE GEOMETRY — read before changing this. In Claude Code's wire shape
// tools[] sits structurally BEFORE messages[], and the cache_control breakpoint
// typically lands on the LAST tool (the whole static tool block is the cached
// head of the prefix). So the set of tools strictly after the last TOOL-level
// breakpoint is usually EMPTY mid-session — and that is correct: pruning a tool
// at/before the breakpoint would move the cache boundary and bust the whole
// session. The spine therefore fires only when the tool block is being rebuilt
// anyway (session start / post-RESET). Harvesting denied-tool headroom across a
// RESET is a later epic rung; the spine deliberately refuses to do it unsafely.
//
// This is a REQUEST-side transform only. It touches the bytes sent upstream; it
// never touches the decoded request the kernel adjudicates, so the trust
// boundary is unchanged and the kernel still polices the exact same tool set.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate. It
// imports no agent/gateway type: the caller passes a `decode` callback so the
// spine can prove its output re-decodes without an upward import.
// See AGENTS.md and internal/architest for the layering contract.
package promptmmu
