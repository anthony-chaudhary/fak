package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const defaultStaleProtectTail = 6

// maybeElideStaleReadMessages applies stale-read semantics to decoded OpenAI-compatible
// transcripts. Only a Read result with a later same-path edit is replaced, the newest
// working set is protected, and fak_context_restore retains the exact original.
func (s *Server) maybeElideStaleReadMessages(trace string, messages []agent.Message) []agent.Message {
	if s == nil || !s.elideStaleReads {
		return messages
	}
	return agent.ElideStaleReadMessages(messages, func(id, excerpt string, body []byte) {
		s.stashRestore(trace, id, excerpt, body)
	})
}

func decodedToolPath(arguments string) string {
	var args map[string]json.RawMessage
	if json.Unmarshal([]byte(arguments), &args) != nil {
		return ""
	}
	for _, key := range []string{"file_path", "path", "notebook_path"} {
		var path string
		if json.Unmarshal(args[key], &path) == nil && strings.TrimSpace(path) != "" {
			return path
		}
	}
	return ""
}

// anthropicServedRequest carries the results of the served-path request-side
// transform pipeline back to handleAnthropicMessages.
type anthropicServedRequest struct {
	compacted    bool
	contextEvent bool
	hcoh         harnessCoherenceInputs
	upstreamKey  string
	upstreamBeta string
}

// prepareServedAnthropicRequest runs the ordered, load-bearing request-side transform
// pipeline for the buffered/synth Anthropic served path: capture the pre-transform
// harness-coherence digest and context footprint, sanitize tool references, upgrade the
// managed-cache TTL, plan the ctxplan view, compact/elide history, prune and defer tool
// definitions, and extract the upstream credential and beta set. The ordering here is
// load-bearing (see the per-step comments) and must not be reordered.
func (s *Server) prepareServedAnthropicRequest(ctx context.Context, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string, sessionTurn servedSessionTurn) anthropicServedRequest {
	// Harness-coherence seam (#1132): capture a CONTENT-FREE digest of the inbound protected
	// prefix (bytes through the first cache_control breakpoint) BEFORE any of fak's request-side
	// transforms below. This ordering is load-bearing — fak forwards the inbound protected prefix
	// verbatim, so a change in this digest across turns can only be the harness rewriting its own
	// history (auto-compaction), never fak. The observation is folded after the served turn, once
	// the provider's cache counters are known.
	inboundPrefixDigest := inboundProtectedPrefixDigest(req.Raw)
	// Live footprint (#3233): price the as-sent structural floor (system + built-in
	// tools + MCP tools + history/tail) at the SAME pre-transform anchor as the
	// prefix digest — before maybeCompactInboundTools prunes any tool def — and fold
	// it per-trace for fak_context_value. ESTIMATED, audit-only, off the byte-faithful
	// passthrough (it reads a de-folded COPY, never mutating req).
	s.observeCtxFootprint(reqTrace, req)
	// Tool-reference sanitization (correctness, runs on EVERY wire, before any shrinker or the
	// prefix-digest matters for cache accounting): the Claude Code client emits its INTERNAL
	// `tool_reference` blocks inside a ToolSearch tool_result, which is not a valid Anthropic
	// tool_result.content block type — a body carrying one is 400'd upstream as malformed
	// (witnessed: session b98cf818, which died on turn 2 right after a ToolSearch result). Rewrite
	// each into a wire-valid `text` block naming the tool so the body forwards cleanly and the model
	// still sees which tools the search surfaced. Fail-safe (identity on any ambiguity) and placed
	// AFTER the digest — a tool_reference only ever lives in a late-turn tool_result, never in the
	// cached protected prefix, so this leaves the digest's harness-coherence meaning intact.
	s.sanitizeAnthropicToolReferences(req)
	// M2 star-anchor pre-flight gate (#1493): DEFAULT-ON (--vcache-anchor), applied BEFORE any
	// other body transform so a caller that sent no cache_control has its volatile system blocks
	// hoisted behind a byte-stable anchor and a breakpoint placed on the stable head — earning
	// provider prefix caching by default, DECOUPLED from the compaction budget (the placement in
	// compactAnthropicRawWithReason only runs while --compact-history-budget>0). Running it first
	// also gives the TTL upgrade below a stable-head breakpoint to extend and makes the compaction
	// placement idempotent (it bails already_set). Fail-safe identity when OFF or on any ambiguity.
	// The placement is witnessed on /metrics (observePlacement) and credited per-trace
	// (recordPlacement); it sheds no turns, so it is not folded into the contextEvent signal below.
	s.maybeAnchorAnthropicRaw(req, reqTrace)
	// Managed-cache 1h TTL upgrade (#1850 / epic #1844 C6): when the lever is on
	// (fak guard --managed-cache, auto-on for API-key-billed sessions; or the
	// FAK_ABLATE_TTL_1H ablation arm), extend an existing stable-head cache_control
	// breakpoint to Anthropic's 1h tier before any body shrinker touches req.Raw, so a
	// long session idling past the 5m window re-enters on a cache read. Every attempt is
	// witnessed on /metrics (fak_gateway_cache_ttl_upgrade_total).
	ttl1hUpgraded, ttl1hMessagePrefix := s.maybeUpgradeAnthropicCacheTTL1HScoped(req)
	if ttl1hUpgraded {
		// #2446: the 1h window now applies to this session, so the ctxvalue wakeup
		// horizon reads the 1h TTL instead of the 5m default from here on.
		s.noteCtxValueTTL1hScope(reqTrace, ttl1hMessagePrefix)
	}
	// ctxplan planned VIEW on the Anthropic passthrough (#927 — the deferred #555 req.Raw
	// transform): when --ctx-view-budget is set, plan req.Messages into an O(1) resident
	// view and materialize it onto req.Raw by stubbing each elided middle turn in place
	// (same role → alternation preserved), while the cache_control prefix bytes and every
	// resident message's original bytes stay byte-identical so the upstream cache hit
	// survives. Runs before the shrinkers so it operates on the original conversation body
	// (its content match keys off the decoded req.Messages); the siblings below then see the
	// already-bounded body and bail (under-budget) in the common case. OFF (identity) by
	// default; fail-safe.
	viewPlanned, _ := s.maybePlanAnthropicRaw(ctx, reqTrace, req)
	// Cache-prefix-preserving history compaction (#555): on the Anthropic passthrough,
	// shrink the OUTBOUND body's OLD turns to the configured resident-token budget while
	// keeping the cached-prefix bytes verbatim, so a long conversation forwards far fewer
	// uncached tokens upstream with the cache hit intact. OFF (identity) by default and on
	// every non-passthrough wire. Applied to req.Raw ONLY — the decoded req.Messages the
	// kernel adjudicates below are untouched, so the trust boundary is unchanged. Placed
	// after the pace cap (which only ever rewrites the top-level max_tokens, never the
	// cached message prefix) and before either passthrough consumer of req.Raw.
	//
	// turnsLeft rides sessionTurn.state.Budget.TurnsLeft — the ONLY live session-horizon
	// signal this request boundary carries — through to the --compact-anchor-head burst
	// economics gate (#1407/#1408). Only a genuinely bounded positive value counts as a
	// known horizon; 0 (no DecideSession wired, or a session with no turns left) and -1
	// (session.Unbounded) both leave the gate's TotalTurns unset, so an un-budgeted or
	// unbounded session never bursts the cache on a guess. reqTrace additionally lets the
	// gate consult the OBSERVED per-trace idle gap: a trace that idled past the
	// message-breakpoint cache TTL is provably cold, so a head-anchored fire there carries
	// no marginal penalty and needs no horizon at all — the unflagged long-session firing
	// path (#1407's cold case).
	compacted, compactReason := s.compactAnthropicRawWithReason(req, sessionTurn.state.Budget.TurnsLeft, reqTrace)
	// fakBail is the harness-coherence view of fak's own compaction this turn: "" for a clean fire
	// AND for a healthy under_budget no-op, the real reason for any actual bail. Threaded into the
	// observation below so the coordinator can count a sustained fak-bail streak (when it yields the
	// context net back to the harness).
	fakBail := fakBailReasonFor(compactReason)
	hcoh := harnessCoherenceInputs{inboundPrefixDigest: inboundPrefixDigest, fakBail: fakBail}
	contextEvent := compacted || viewPlanned
	// Oversized tool_result elision (the bounded-loss sibling of compaction): after compaction
	// has dropped whole OLD turns, shrink any remaining oversized tool_result bodies in the
	// un-cached, non-recent middle to a bounded head+tail form, keeping the cached-prefix bytes
	// verbatim and never touching a cache_control-bearing message. OFF (identity) by default and
	// on every non-passthrough wire; fail-safe. Runs on the already-compacted body.
	//
	// Read-lifecycle STALE elision runs FIRST (before the generic size shrinker): it replaces a Read
	// tool_result whose file was Edited/Written in a later turn with a restorable marker, stashing the
	// pre-edit snapshot behind a fak_context_restore handle under reqTrace. Ordering matters — running
	// before the size shrinker means the stashed original is the FULL body, not an already head+tail-
	// shrunk one, so a restore returns full fidelity. Same cache-prefix proof; OFF by default.
	s.maybeElideStaleReads(req, reqTrace)
	s.maybeElideAnthropicRaw(req)
	// Inbound twin of #555: prune tool DEFINITIONS the floor can never admit from the
	// outbound tools[], keeping the cache_control prefix byte-identical (promptmmu). Runs
	// after the history compaction (both rewrite req.Raw; tools[] and messages[] are
	// disjoint regions) and before either passthrough consumer of req.Raw. Identity-safe:
	// nil predicate or no floor-denied advertised tool ⇒ req.Raw untouched. The call records
	// its WITNESSED prune count into /metrics (observeInboundToolPrune), so a turn that shed
	// unreachable tool defs is now visible in the exit summary instead of silently discarded.
	// Also remember the concrete names per trace so a later model proposal of a pruned name is
	// logged once as a floor-vs-observed drift witness.
	s.recordInboundPrunedToolDefinitions(reqTrace, s.maybeCompactInboundTools(req))
	// The system[] twin of the line above, and — until #5446 — the one inbound prune whose
	// result was simply dropped here. It now reports what it did, and the emitter below
	// puts a real prune and a STRUCTURAL failure to read system[] on the log, while the
	// dominant idle turn stays silent. Without this consumption the promptmmu reason split
	// would be correct and still unobservable, which is the ships-unwired shape #5435 tracks.
	s.logInboundSystemPrune(s.maybeCompactInboundSystem(req))
	// The 10x floor lever (#3232): defer the cold tool tail (defer_loading:true) and inject
	// a tool_search_tool on the outbound body, so the provider loads only the hot core into
	// context. OFF by default; deterministic + cache-safe + fail-safe identity. Runs AFTER
	// the deny-prune (disjoint operations on tools[]) and before the passthrough consumers of
	// req.Raw. The current Tool Search contract needs no beta header.
	s.maybeDeferColdTools(req, reqTrace)
	// In passthrough mode the upstream credential is the client's own (transparent
	// hop) UNLESS the gateway pins its own (the subscription path). The inbound
	// anthropic-beta is forwarded so the client's negotiated betas survive the hop.
	// Both extracted here, on the HTTP boundary, since the planner layer never sees
	// the request headers.
	upstreamKey := s.anthropicUpstreamCredential(r)
	upstreamBeta := filterRetiredAnthropicBetas(r.Header.Get("anthropic-beta"))
	migrateRetiredToolSearch(req)
	// Beta union (managed-cache 1h TTL): when this turn upgraded a stable-head breakpoint to
	// the 1h tier, the body now carries cache_control ttl:"1h", which Anthropic accepts only
	// with the extended-cache-ttl beta negotiated. The wrapped claude CLI defaults to the 5m
	// tier and does not send it, so union it in ourselves — mirroring the toolSearchBeta union
	// above. Without this a forced --managed-cache session 400s upstream as malformed (the
	// subscription-OAuth "managed cache — ACTIVE (forced by --managed-cache on)" instant crash).
	if ttl1hUpgraded {
		upstreamBeta = unionBeta(upstreamBeta, extendedCacheTTLBeta)
	}
	return anthropicServedRequest{compacted: compacted, contextEvent: contextEvent, hcoh: hcoh, upstreamKey: upstreamKey, upstreamBeta: upstreamBeta}
}

// readAnthropicMessagesRequest reads the inbound POST /v1/messages body under the
// transcript cap and decodes it into the canonical request. It writes a clean 413
// on a body past the cap and a 400 on malformed JSON; a false return means the
// response has already been written and the caller must return.
func (s *Server) readAnthropicMessagesRequest(w http.ResponseWriter, r *http.Request) (*agent.AnthropicMessagesRequest, bool) {
	r.Body = http.MaxBytesReader(w, r.Body, maxTranscriptBody)
	raw := make([]byte, 0, 4096)
	buf := make([]byte, 32*1024)
	for {
		n, readErr := r.Body.Read(buf)
		raw = append(raw, buf[:n]...)
		if readErr != nil {
			// MaxBytesReader signals overflow with *http.MaxBytesError. Forwarding
			// the truncated prefix upstream yields an opaque 400; surface it as a
			// clean 413 so the operator sees the real cause (a body past the cap),
			// not a malformed-JSON guess.
			var maxErr *http.MaxBytesError
			if errors.As(readErr, &maxErr) {
				writeErr(w, http.StatusRequestEntityTooLarge,
					"request body exceeds the gateway limit ("+strconv.Itoa(maxTranscriptBody>>20)+" MiB); the transcript is too large to forward")
				return nil, false
			}
			break
		}
	}
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return nil, false
	}
	return req, true
}
