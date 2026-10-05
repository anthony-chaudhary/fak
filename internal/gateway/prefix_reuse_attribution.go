package gateway

import (
	"context"
	"fmt"
	"hash/maphash"
	"net/http"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// prefix_reuse_attribution.go — one cache scope for every planner path, plus the
// served-path attribution of in-kernel prefix reuse to the agent conversation that paid for
// it.
//
// Scope: subagents launched by a third-party harness (Claude Code, Pi, OpenCode, Codex)
// share their parent's system prompt, tool catalog, and repo context. Those prefixes can
// only be reused when every turn from one principal lands in ONE radix namespace. Binding
// the prefix-cache identity in buffered complete() alone split an authenticated principal
// across two namespaces (streamed turns stayed in the legacy unscoped namespace), so the
// streamed coordinator and its buffered subagents never shared. plannerTurnContext is the
// single binding every planner call site uses.
//
// Attribution: the planner reports per-turn prompt and matched-prefix tokens on the
// Completion (in-kernel turns only). The gateway splits each turn's matched tokens into
// what this conversation could have cached itself (bounded by its own earlier
// prompt+completion length) and the remainder, which another conversation — a sibling
// subagent or another session — prefilled. The cross part also feeds the cross-agent
// reuse ledger keyed on the harness launch session (X-Fak-Session-Id, else X-Trace-Id).
// Only hashes and integer counts are retained; no prompt text crosses.
//
// Hot path: observation is a hash computed outside any lock plus an O(1) map update under
// one short mutex. No I/O, no channel, no allocation proportional to the transcript.

// inKernelProducer is the cachemeta producer the in-kernel planner stamps on its Completion.
const inKernelProducer = "fak-inkernel"

const (
	prefixReuseConvCap    = 4096
	prefixReuseSessionCap = 1024
)

type prefixReuseSessionKey struct{}

// harnessSessionID returns the harness-supplied launch session id: X-Fak-Session-Id, else
// X-Trace-Id. A minted per-request trace is deliberately NOT a fallback — it would make
// every turn its own session and the attribution meaningless.
func harnessSessionID(r *http.Request) string {
	if r == nil {
		return ""
	}
	if id := strings.TrimSpace(r.Header.Get("X-Fak-Session-Id")); id != "" {
		return id
	}
	return strings.TrimSpace(r.Header.Get("X-Trace-Id"))
}

func withPrefixReuseSession(ctx context.Context, session string) context.Context {
	if session == "" {
		return ctx
	}
	return context.WithValue(ctx, prefixReuseSessionKey{}, session)
}

func prefixReuseSessionFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	s, _ := ctx.Value(prefixReuseSessionKey{}).(string)
	return s
}

// plannerTurnContext is the ONE binding applied before every planner call — buffered
// Complete and every streaming CompleteStream. It binds the authenticated principal to the
// in-kernel prefix-cache identity at tenant scope (agent left empty so sibling agents of
// one principal share prefixes), carries the harness session id for attribution, and binds
// the Context Epoch tag last so an epoch turn-over restamps the cache namespace for every
// planner path at once. r may be nil when the caller's ctx already descends from
// beginServedRequest.
func plannerTurnContext(ctx context.Context, r *http.Request, messages []agent.Message, epoch *ContextEpochGate) context.Context {
	if principal := principalFromContext(ctx); principal != "" {
		ctx = agent.WithPrefixCacheIdentity(ctx, principal, "")
	}
	if prefixReuseSessionFromContext(ctx) == "" {
		ctx = withPrefixReuseSession(ctx, harnessSessionID(r))
	}
	return epoch.bind(ctx, messages)
}

// prefixReuseOrigin is the closed classification of one served in-kernel turn.
type prefixReuseOrigin uint8

const (
	prefixReuseCold prefixReuseOrigin = iota
	prefixReuseSameSession
	prefixReuseCrossSession
)

// prefixReuseAttribution is the zero-value-usable per-Server attribution state.
type prefixReuseAttribution struct {
	mu sync.Mutex
	// conv holds each conversation's largest own footprint (prompt+completion tokens).
	conv map[uint64]uint64
	// root holds the first conversation seen under a harness session (its coordinator).
	root   map[string]uint64
	ledger *crossAgentReuseLedger

	sameTokens, crossTokens          uint64
	coldTurns, sameTurns, crossTurns uint64
}

var prefixReuseSeed = maphash.MakeSeed()

// conversationKey identifies an agent conversation by its stable head: the leading system
// messages plus the first non-system message, scoped by principal. A subagent differs from
// its coordinator in that head while sharing the cached system/tool prefix.
func conversationKey(principal string, messages []agent.Message) uint64 {
	var h maphash.Hash
	h.SetSeed(prefixReuseSeed)
	h.WriteString(principal)
	h.WriteByte(0)
	for _, m := range messages {
		h.WriteString(m.Role)
		h.WriteByte(0)
		h.WriteString(m.Content)
		h.WriteByte(0)
		if m.Role != agent.RoleSystem {
			break
		}
	}
	return h.Sum64()
}

// observe classifies one turn and folds it into the counters and the cross-agent ledger.
func (a *prefixReuseAttribution) observe(session string, conv uint64, promptTokens, matchedTokens, completionTokens int) prefixReuseOrigin {
	if promptTokens <= 0 {
		return prefixReuseCold
	}
	if matchedTokens < 0 {
		matchedTokens = 0
	}
	if matchedTokens > promptTokens {
		matchedTokens = promptTokens
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.conv == nil {
		a.conv = map[uint64]uint64{}
		a.root = map[string]uint64{}
		a.ledger = newCrossAgentReuseLedger()
	}
	prior := a.conv[conv]
	own := uint64(matchedTokens)
	if own > prior {
		own = prior
	}
	cross := uint64(matchedTokens) - own
	a.sameTokens += own
	a.crossTokens += cross
	origin := prefixReuseCold
	switch {
	case cross > 0:
		origin = prefixReuseCrossSession
		a.crossTurns++
	case own > 0:
		origin = prefixReuseSameSession
		a.sameTurns++
	default:
		a.coldTurns++
	}
	if footprint := uint64(promptTokens) + uint64(max(completionTokens, 0)); footprint > prior {
		if _, ok := a.conv[conv]; !ok && len(a.conv) >= prefixReuseConvCap {
			evictOne(a.conv)
		}
		a.conv[conv] = footprint
	}
	if session != "" {
		rootConv, ok := a.root[session]
		if !ok {
			if len(a.root) >= prefixReuseSessionCap {
				evictOne(a.root)
			}
			a.root[session] = conv
			rootConv = conv
		}
		child := fmt.Sprintf("c%016x", conv)
		if conv == rootConv {
			child = session // the coordinator's own turn: the ledger excludes it
		}
		a.ledger.ObserveCrossAgentTurn(session, child, "", promptTokens, int(cross))
	}
	return origin
}

func evictOne[K comparable, V any](m map[K]V) {
	for k := range m {
		delete(m, k)
		return
	}
}

type prefixReuseSnapshot struct {
	sameTokens, crossTokens          uint64
	coldTurns, sameTurns, crossTurns uint64
	rollup                           crossAgentRollup
	coordinators                     int
}

func (a *prefixReuseAttribution) snapshot() prefixReuseSnapshot {
	a.mu.Lock()
	snap := prefixReuseSnapshot{
		sameTokens: a.sameTokens, crossTokens: a.crossTokens,
		coldTurns: a.coldTurns, sameTurns: a.sameTurns, crossTurns: a.crossTurns,
	}
	ledger := a.ledger
	a.mu.Unlock()
	snap.rollup = ledger.Rollup()
	snap.coordinators = len(ledger.Snapshot())
	return snap
}

// observePrefixReuseTurn is the served-path hook every planner call site runs after a
// successful turn. Non-in-kernel completions (provider passthrough) carry no local prefix
// reuse and are skipped.
func (s *Server) observePrefixReuseTurn(ctx context.Context, messages []agent.Message, comp *agent.Completion) {
	if s == nil || comp == nil || comp.ProviderCache == nil || comp.ProviderCache.Derivation.Producer != inKernelProducer {
		return
	}
	matched := 0
	if comp.Usage.PromptTokensDetails != nil {
		matched = comp.Usage.PromptTokensDetails.CachedTokens
	}
	conv := conversationKey(principalFromContext(ctx), messages)
	s.prefixReuse.observe(prefixReuseSessionFromContext(ctx), conv, comp.Usage.PromptTokens, matched, comp.Usage.CompletionTokens)
}

// writePrefixReuseAttributionMetrics renders the same-session vs cross-session split of
// in-kernel prefix reuse and the cross-agent ledger rollup.
func (s *Server) writePrefixReuseAttributionMetrics(b *strings.Builder) {
	snap := s.prefixReuse.snapshot()
	writeHelpType(b, "fak_gateway_kv_prefix_reused_tokens_by_origin_total",
		"In-kernel KV-prefix reused tokens split by origin. same_session: within what this agent conversation itself prefilled earlier; cross_session: beyond it, so a sibling subagent or another session paid for the prefix.", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_reused_tokens_by_origin_total{origin=\"same_session\"} %d\n", snap.sameTokens)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_reused_tokens_by_origin_total{origin=\"cross_session\"} %d\n", snap.crossTokens)
	writeHelpType(b, "fak_gateway_kv_prefix_turns_by_origin_total",
		"In-kernel turns classified by prefix-reuse origin: cross_session if any reused token exceeds the conversation's own footprint, same_session if all reuse is its own, cold if nothing was reused.", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_origin_total{origin=\"same_session\"} %d\n", snap.sameTurns)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_origin_total{origin=\"cross_session\"} %d\n", snap.crossTurns)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_origin_total{origin=\"cold\"} %d\n", snap.coldTurns)
	writeHelpType(b, "fak_gateway_kv_prefix_cross_agent_coordinators",
		"Harness launch sessions (X-Fak-Session-Id / X-Trace-Id) with at least one subagent turn in the bounded cross-agent reuse ledger.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_cross_agent_coordinators %d\n", snap.coordinators)
	writeHelpType(b, "fak_gateway_kv_prefix_cross_agent_subagent_turns",
		"Subagent turns (a non-coordinator conversation under one launch session) retained in the cross-agent reuse ledger.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_cross_agent_subagent_turns %d\n", snap.rollup.CrossAgentSubagentCount)
	writeHelpType(b, "fak_gateway_kv_prefix_cross_agent_prompt_tokens",
		"Prompt tokens of the subagent turns retained in the cross-agent reuse ledger.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_cross_agent_prompt_tokens %d\n", snap.rollup.CrossAgentPromptTokens)
	writeHelpType(b, "fak_gateway_kv_prefix_cross_agent_shared_tokens",
		"Subagent prompt tokens served from a prefix another conversation prefilled, retained in the cross-agent reuse ledger.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_cross_agent_shared_tokens %d\n", snap.rollup.CrossAgentSharedTokens)
}
