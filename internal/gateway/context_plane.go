package gateway

import (
	"fmt"
	"strings"
)

// The context-plane postures select whether the gateway REWRITES the served
// context on the chat/responses wires: oversize tool-result paging (the
// {"_paged":true} restore stubs) and old-oversized-tool-result elision (the
// "...[fak: tool output elided ...]" markers + the restore-tool auto-
// advertisement). They are the context-plane siblings of the tool-plane
// ToolGuard posture (toolguard.go) and follow its exact idiom: empty resolves
// to the library/harness default (ARMED) so `fak guard`, the demos and every
// existing test keep the savers byte-for-byte, while the serving router
// (`fak serve`) resolves its own product default OFF — the 2026-11-10
// hosting-simplification flip. An explicit unparsable value fails New loud:
// a context rewriter is a model-visible boundary and never silently resolves.
//
// Re-enable paths (bring the savers back without a rebuild):
//   - `fak serve --ctx-paging on` / FAK_CTX_PAGING=on  — oversize paging
//   - `fak serve --chat-elide` / FAK_CHAT_ELIDE=on     — tool-result elision
//   - the per-lever serving flags (--ctx-view-budget, --compact-history-budget,
//     --elide-result-bytes, --elide-stale-reads, --vcache-anchor,
//     --defer-cold-tools) each re-arm their single lever.

const (
	// CtxPagingAuto pages an oversize-but-benign tool result only when the
	// client can restore the stub (the library default).
	CtxPagingAuto = "auto"
	// CtxPagingOn pages oversize tool results regardless of client capability.
	CtxPagingOn = "on"
	// CtxPagingOff never pages: tool results reach the model byte-identical.
	CtxPagingOff = "off"

	// ChatToolResultElideOn arms the chat/responses tool-result elision (the
	// library default).
	ChatToolResultElideOn = "on"
	// ChatToolResultElideOff never rewrites a tool result and never
	// auto-advertises the restore tool.
	ChatToolResultElideOff = "off"
)

// ParseCtxPagingMode validates one explicit CtxPaging posture token. Empty
// resolves to AUTO — the library/harness default — so every existing caller
// keeps the paging behavior byte-for-byte. Accepted spellings mirror the
// repo's env-bool idiom.
func ParseCtxPagingMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "":
		return CtxPagingAuto, nil
	case CtxPagingAuto:
		return CtxPagingAuto, nil
	case CtxPagingOn, "1", "true", "yes", "enabled":
		return CtxPagingOn, nil
	case CtxPagingOff, "0", "false", "no", "disabled", "raw":
		return CtxPagingOff, nil
	}
	return "", fmt.Errorf("unknown ctx paging mode %q (want \"on\", \"off\" or \"auto\")", raw)
}

// ParseChatToolResultElideMode validates one explicit ChatToolResultElide
// posture token. Empty resolves to ON — the library/harness default — so every
// existing caller keeps the elider byte-for-byte.
func ParseChatToolResultElideMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "":
		return ChatToolResultElideOn, nil
	case ChatToolResultElideOn, "1", "true", "yes", "enabled":
		return ChatToolResultElideOn, nil
	case ChatToolResultElideOff, "0", "false", "no", "disabled", "raw":
		return ChatToolResultElideOff, nil
	}
	return "", fmt.Errorf("unknown chat tool-result elide mode %q (want \"on\" or \"off\")", raw)
}

// The `fak serve` door's context-plane product defaults (the 2026-11-10
// hosting-simplification flip). The serving router is a raw inference pipe on
// BOTH planes: the tool plane (ToolGuard, flipped earlier the same day) and
// now the context plane. The guard door (`fak guard`) and every library caller
// keep the armed savers via the empty-config resolution above; each lever here
// is individually re-enableable by its flag/env, so nothing is removed — the
// default is simplified.
const (
	// DefaultServeCtxPaging is the --ctx-paging default: OFF. The serving
	// router does not page tool results into restore stubs by default.
	DefaultServeCtxPaging = CtxPagingOff
	// DefaultServeChatToolResultElide is the --chat-elide default: OFF. The
	// serving router does not elide old tool results or auto-advertise the
	// restore tool by default.
	DefaultServeChatToolResultElide = ChatToolResultElideOff
	// DefaultServeCtxViewBudget is the --ctx-view-budget default: 0 (the
	// ctxplan planner is not wired into the buffered serve loop; the forwarded
	// history is byte-for-byte). The guard door keeps agent.DefaultCtxViewBudget.
	DefaultServeCtxViewBudget = 0
	// DefaultServeCompactHistoryBudget is the --compact-history-budget default:
	// 0 (the outbound body is forwarded byte-for-byte). The guard door keeps
	// gateway.DefaultCompactHistoryBudget (headless: HeadlessCompactHistoryBudget).
	// The compaction SUB-levers (--compact-anchor-head, --assume-session-turns)
	// keep their library defaults: both are inert while the budget is 0, and
	// re-arming the budget brings the full modern compaction (head anchor +
	// horizon prior) rather than a half-configured one.
	DefaultServeCompactHistoryBudget = 0
	// DefaultServeElideResultBytes is the --elide-result-bytes default: 0. The
	// guard door keeps gateway.DefaultElideResultBytes.
	DefaultServeElideResultBytes = 0
	// DefaultServeElideStaleReads is the --elide-stale-reads default: false.
	// The guard door keeps gateway.DefaultElideStaleReads (true).
	DefaultServeElideStaleReads = false
	// DefaultServeVCacheAnchor is the --vcache-anchor default: false. The guard
	// door keeps gateway.DefaultVCacheAnchor (true).
	DefaultServeVCacheAnchor = false
	// DefaultServeDeferColdTools is the --defer-cold-tools default: false. The
	// guard door keeps gateway.DefaultDeferColdTools (true).
	DefaultServeDeferColdTools = false
)
