package gateway

// messages.go is the native Anthropic Messages front door (POST /v1/messages and
// /v1/messages/count_tokens). It is the Claude-Code-facing twin of
// handleChatCompletions: same planner, same kernel adjudication boundary
// (s.adjudicateProposed), different downstream wire. Point Claude Code at it with
//
//	ANTHROPIC_BASE_URL=http://127.0.0.1:8080   (no /v1 — the client appends it)
//
// and every tool call the locally-served model proposes is dropped/repaired by the
// kernel before Claude Code ever sees it.
//
// The upstream planner is non-streaming (it buffers the whole completion), so when
// the request asks for "stream":true we SYNTHESIZE a well-formed Anthropic SSE
// sequence from the finished, already-adjudicated turn rather than truly streaming
// tokens. Claude Code parses the event sequence identically; the round trip — and
// crucially the tool_use ids it matches results back by — is byte-faithful.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

// anthropicMessageResponse is the buffered (non-streaming) /v1/messages body.
//
// Fak is a non-standard top-level extension carrying the kernel's per-call
// adjudications — the Anthropic-wire twin of ChatResponse.Fak on the OpenAI wire.
// The Anthropic Messages schema is otherwise fixed, but its clients (Claude Code,
// the Anthropic SDKs) tolerate unknown top-level response keys, so a fak-aware
// tool can read structured verdicts here while the standard parser ignores it. The
// human/agent-readable form of the same information rides as an in-band text block
// (see adjudicationNote) — that text is what a coding agent actually reacts to.
type anthropicMessageResponse struct {
	ID           string                    `json:"id"`
	Type         string                    `json:"type"` // always "message"
	Role         string                    `json:"role"` // always "assistant"
	Model        string                    `json:"model"`
	Content      []agent.AnthropicBlockOut `json:"content"`
	StopReason   string                    `json:"stop_reason"`
	StopSequence *string                   `json:"stop_sequence"`
	Usage        anthropicUsage            `json:"usage"`
	Fak          *FakExt                   `json:"fak,omitempty"`
}

// anthropicUsage mirrors the Messages API usage object. The cache counters are
// load-bearing for the anthropic→anthropic proxy: when fak fronts the real API and
// forwards the client's cache_control prefix byte-for-byte, the upstream returns a
// cache_read_input_tokens hit — which the client (Claude Code) needs to see to
// account a turn correctly. They are reported INDEPENDENTLY of input_tokens
// (Anthropic's input_tokens is already the uncached remainder), and omitted when
// zero so a local-model turn's usage shape is unchanged.
type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}

type anthropicTurn struct {
	ID     string
	Model  string
	Blocks []agent.AnthropicBlockOut
	Stop   string
	Usage  anthropicUsage
	// Adjs is the per-call adjudication record for this turn (drops + repairs +
	// allows). It rides back as the response `fak` extension and, when it carries
	// a drop or repair, also as a leading in-band text block so a client that
	// reads only content (Claude Code) still sees what the kernel did.
	Adjs []ToolAdjudication
	// ResultAdmissions is the result-side floor's verdict on each inbound tool
	// result this turn carried (quarantine / transform / allow). It rides the same
	// `fak` extension as the OpenAI wire, and a quarantine also raises an in-band
	// note so the agent knows its tool output was paged out (not lost).
	ResultAdmissions []ResultAdmission
}

var anthropicStreamPingInterval = 15 * time.Second

// anthropicInboundKey extracts the inbound client's own upstream credential from
// the request — the transparent-hop key used on the anthropic→anthropic passthrough
// path, where the client authenticates directly against the real Anthropic API with
// its own secret. Claude Code and the Anthropic SDKs send "x-api-key: <tok>";
// OpenAI/fak-native clients send "Authorization: Bearer <tok>". Reuses the shared
// gatewayCredential extractor so both schemes are honored identically. Empty when
// the client presented no key (loopback dogfood) — passthrough then falls back to
// the planner's configured key.
func anthropicInboundKey(r *http.Request) string {
	if tok, ok := gatewayCredential(r); ok {
		return tok
	}
	return ""
}

// anthropicUpstreamCredential resolves the credential the passthrough hop
// authenticates upstream with. With PinUpstreamCredential set the gateway uses
// its OWN configured key (returns "" so the planner falls back to its configured
// APIKey) and IGNORES the inbound client's key — the subscription path, where fak
// holds the real OAuth token and the wrapped client only sends a placeholder to
// satisfy its own credential check. Otherwise it forwards the inbound client's own
// key (the transparent hop).
func (s *Server) anthropicUpstreamCredential(r *http.Request) string {
	if s.pinUpstreamCredential {
		return ""
	}
	return anthropicInboundKey(r)
}

// handleAnthropicMessages is the adjudication PROXY on the Anthropic wire. It
// decodes the inbound Messages request into the canonical transcript, forwards it
// to the configured model (the same HTTPPlanner/MockPlanner the OpenAI path uses),
// runs every PROPOSED tool call through the kernel, and renders the survivors back
// as an Anthropic message — buffered, or as SSE when the client asked to stream.
func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBody)
	raw := make([]byte, 0, 4096)
	buf := make([]byte, 32*1024)
	for {
		n, err := r.Body.Read(buf)
		raw = append(raw, buf[:n]...)
		if err != nil {
			break
		}
	}
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	ctx := r.Context()
	reqTrace := s.traceFor(r.Header.Get("X-Trace-Id"))
	// Operator control / budget / pace at the served request boundary. With
	// DecideSession wired this mutates the live session table (TurnsLeft debit,
	// budget exhaustion, pace cap); without it the legacy observe-only admission guard
	// still refuses paused/draining/stopped sessions.
	sessionTurn, ok, canceled := s.beginServedSessionTurn(ctx, reqTrace)
	if canceled {
		return
	}
	if !ok {
		// Budget drained: if the host wired the opt-in human-like reset, distill a
		// carryover seed, re-arm a fresh session, and continue transparently on the
		// new trace instead of refusing. Otherwise emit the historical 409 directive.
		if newTrace, seed, reset := s.maybeResetOnBudget(ctx, sessionTurn.state, req.Messages); reset {
			req.Messages = spliceSeed(seed, req.Messages)
			reqTrace = newTrace
			if sessionTurn, ok, canceled = s.beginServedSessionTurn(ctx, reqTrace); canceled {
				return
			}
			if !ok { // the fresh session somehow refuses too — fall back, never loop
				writeSessionRefusal(w, sessionTurn.state)
				return
			}
		} else {
			writeSessionRefusal(w, sessionTurn.state)
			return
		}
	}
	applySessionPaceToAnthropicRequest(req, sessionTurn)
	// ctxplan planned VIEW on the Anthropic passthrough (#927 — the deferred #555 req.Raw
	// transform): when --ctx-view-budget is set, plan req.Messages into an O(1) resident
	// view and materialize it onto req.Raw by stubbing each elided middle turn in place
	// (same role → alternation preserved), while the cache_control prefix bytes and every
	// resident message's original bytes stay byte-identical so the upstream cache hit
	// survives. Runs FIRST so it operates on the original body (its content match keys off
	// the decoded req.Messages); the siblings below then see the already-bounded body and
	// bail (under-budget) in the common case. OFF (identity) by default; fail-safe.
	viewPlanned := s.maybePlanAnthropicRaw(ctx, reqTrace, req)
	// Cache-prefix-preserving history compaction (#555): on the Anthropic passthrough,
	// shrink the OUTBOUND body's OLD turns to the configured resident-token budget while
	// keeping the cached-prefix bytes verbatim, so a long conversation forwards far fewer
	// uncached tokens upstream with the cache hit intact. OFF (identity) by default and on
	// every non-passthrough wire. Applied to req.Raw ONLY — the decoded req.Messages the
	// kernel adjudicates below are untouched, so the trust boundary is unchanged. Placed
	// after the pace cap (which only ever rewrites the top-level max_tokens, never the
	// cached message prefix) and before either passthrough consumer of req.Raw.
	compacted := s.maybeCompactAnthropicRaw(req)
	if viewPlanned {
		compacted = true // the ctxview transform shrunk the body too — record the observed cache_read
	}
	// Inbound twin of #555: prune tool DEFINITIONS the floor can never admit from the
	// outbound tools[], keeping the cache_control prefix byte-identical (promptmmu). Runs
	// after the history compaction (both rewrite req.Raw; tools[] and messages[] are
	// disjoint regions) and before either passthrough consumer of req.Raw. Identity-safe:
	// nil predicate or no floor-denied advertised tool ⇒ req.Raw untouched.
	s.maybeCompactInboundTools(req)
	// In passthrough mode the upstream credential is the client's own (transparent
	// hop) UNLESS the gateway pins its own (the subscription path). The inbound
	// anthropic-beta is forwarded so the client's negotiated betas survive the hop.
	// Both extracted here, on the HTTP boundary, since the planner layer never sees
	// the request headers.
	upstreamKey := s.anthropicUpstreamCredential(r)
	upstreamBeta := r.Header.Get("anthropic-beta")

	// Repetition-loop guard (runs on EVERY wire, before any planner round-trip). A small
	// local model, after a tool refusal, often stops making progress and loops — echoing
	// fak's `[fak] refused …` note back as its own text, or repeating the same refusal
	// prose verbatim — every turn to the harness turn-cap with an empty result. When the
	// replayed history shows an unbroken degenerate tail, short-circuit with a terminal
	// steer turn that breaks the cycle deterministically (no model call). The kernel still
	// adjudicated every real call that got us here; this only stops the dead loop.
	if steer := repetitionLoopSteer(req.Messages, "", s.modelOr(req.Model)); steer != nil {
		s.writeAnthropicTurn(w, req.Stream, steer)
		return
	}

	if req.Stream {
		// When fronting the REAL Anthropic API, relay a TRUE live token stream so the
		// client's first token tracks the model (and the prompt-cache hit's fast prefill
		// is felt, not buffered away). It returns false only if the upstream stream never
		// opened and nothing was written — then fall back to the buffered synth path,
		// which is also the path for a local/mock upstream that cannot stream this wire.
		if s.anthropicPassthrough() && s.streamAnthropicPassthroughLive(w, r, req, reqTrace, sessionTurn, upstreamKey, upstreamBeta, compacted) {
			return
		}
		// For non-Anthropic upstreams that still support the generic planner streaming
		// seam (OpenAI-compatible/vLLM/SGLang), translate live content callbacks into
		// Anthropic text_delta events while holding every proposed tool call until the
		// same whole-turn adjudication gate below can run. A false return means either
		// the planner cannot stream or the writer cannot flush, so the existing
		// ping-then-synthesize fallback remains the behavior.
		if s.streamAnthropicPlannerLive(w, r, req, reqTrace, sessionTurn) {
			return
		}
		s.streamAnthropicPending(w, r, req, reqTrace, sessionTurn, upstreamKey, upstreamBeta, compacted)
		return
	}

	began := time.Now()
	turn, err := s.completeAnthropicTurn(ctx, req, reqTrace, sessionTurn, "", "", upstreamKey, upstreamBeta)
	if err != nil {
		// Classify like the chat-completions path: an in-kernel device OOM becomes a specific,
		// actionable 503; a genuine upstream failure stays the opaque 502 with the raw provider
		// body kept off the wire. writeErrCode with an empty code reproduces the historical
		// code:null body byte-for-byte, so the non-OOM response is unchanged (#346).
		status, code, msg := s.plannerErrorStatus(err)
		s.logf("gateway: messages turn error (%s): %v", code, err)
		writeErrCode(w, status, code, msg)
		return
	}

	// On a turn we actually compacted, record the provider's OBSERVED cache_read (relayed, not a
	// fak claim) so the net effect is visible on /metrics next to the WITNESSED shed.
	if compacted {
		s.metrics.recordCompactionCacheRead(turn.Usage.CacheReadInputTokens)
		s.observeResetHealth(reqTrace, turn.Usage.InputTokens, turn.Usage.CacheReadInputTokens, turn.Usage.CacheCreationInputTokens)
	}
	s.logInferenceTurn(reqTrace, "anthropic_messages", false, agent.Usage{
		PromptTokens:             turn.Usage.InputTokens,
		CompletionTokens:         turn.Usage.OutputTokens,
		CacheReadInputTokens:     turn.Usage.CacheReadInputTokens,
		CacheCreationInputTokens: turn.Usage.CacheCreationInputTokens,
	}, turn.Stop, time.Since(began), compacted)

	writeJSON(w, http.StatusOK, anthropicMessageResponse{
		ID: turn.ID, Type: "message", Role: "assistant", Model: turn.Model,
		Content: turn.Blocks, StopReason: turn.Stop, StopSequence: nil, Usage: turn.Usage,
		Fak: fakExtFrom(turn.Adjs, turn.ResultAdmissions),
	})
}

// modelOr returns the client-requested model, or the gateway's configured model when
// the request omitted one (Anthropic reflects the requested id; we fall back so a
// synthesized turn always names a model).
func (s *Server) modelOr(reqModel string) string {
	if reqModel != "" {
		return reqModel
	}
	return s.model
}

func applySessionPaceToAnthropicRequest(req *agent.AnthropicMessagesRequest, turn servedSessionTurn) {
	if req == nil {
		return
	}
	cap := turn.maxTokensFor(req.MaxTokens)
	if cap <= 0 || cap == req.MaxTokens {
		return
	}
	req.MaxTokens = cap
	if len(req.Raw) == 0 {
		return
	}
	// Cap max_tokens in req.Raw by a TARGETED byte splice, NOT a full unmarshal/re-marshal.
	// On the Anthropic passthrough req.Raw is forwarded byte-for-byte to preserve the client's
	// prompt-cache prefix; re-marshalling a map[string]json.RawMessage sorts the top-level keys
	// (Go map marshal is key-sorted), reordering everything before the messages array and
	// BUSTING the cached prefix on every paced turn (#774 / the F13 cache-bust). spliceMaxTokens
	// replaces only the integer after the existing "max_tokens" key, leaving every other byte —
	// and thus the cached prefix — untouched. If the key is absent or the body cannot be safely
	// spliced, leave req.Raw alone: the decoded req.MaxTokens above already carries the cap for
	// any non-passthrough re-build, and on passthrough the client's original max_tokens riding
	// through unchanged is strictly safer than a cache-busting rewrite.
	if out, ok := spliceMaxTokens(req.Raw, cap); ok {
		req.Raw = out
	}
}

// spliceMaxTokens replaces the integer value of the top-level "max_tokens" key in an Anthropic
// /v1/messages body with cap, by a byte splice that touches ONLY that number — every other byte
// (and so the cache_control prefix) is preserved verbatim. It returns ok=false (caller leaves
// req.Raw unchanged) when the key is absent, the value is not a bare integer, or the splice
// would not re-decode to a valid request — fail-safe identity, never a cache-busting rewrite.
func spliceMaxTokens(raw []byte, cap int) ([]byte, bool) {
	// Locate the "max_tokens" key, then the JSON number that follows its colon. We scan for the
	// key bytes; a false match inside a string value is caught by the re-decode + value check.
	key := []byte(`"max_tokens"`)
	ki := bytes.Index(raw, key)
	if ki < 0 {
		return nil, false
	}
	i := ki + len(key)
	// Skip whitespace and the single ':' separator.
	for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') {
		i++
	}
	if i >= len(raw) || raw[i] != ':' {
		return nil, false
	}
	i++
	for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') {
		i++
	}
	// The value must be a bare JSON integer (digits, optional leading '-').
	start := i
	if i < len(raw) && raw[i] == '-' {
		i++
	}
	digitsStart := i
	for i < len(raw) && raw[i] >= '0' && raw[i] <= '9' {
		i++
	}
	if i == digitsStart { // no digits → not an integer value (e.g. it was a string) — bail
		return nil, false
	}
	var b bytes.Buffer
	b.Grow(len(raw))
	b.Write(raw[:start])
	b.WriteString(itoa(uint64(cap)))
	b.Write(raw[i:])
	out := b.Bytes()
	// Prove the splice produced a valid request before trusting it.
	if _, err := agent.DecodeAnthropicMessagesRequest(out); err != nil {
		return nil, false
	}
	return out, true
}

// maybeCompactAnthropicRaw applies the cache-prefix-preserving history rewrite to the
// outbound passthrough body when --compact-history-budget is set — and, as the offensive twin
// (#806), first places a cache_control breakpoint on the stable head when the caller set none.
// It is a no-op unless the
// gateway is fronting the REAL Anthropic API (s.anthropicPassthrough) — only there is the
// raw body forwarded verbatim, so only there does compacting it reach the wire AND need to
// preserve the cached prefix. On every other wire the body is re-built from req.Messages
// downstream, so touching req.Raw would be pointless. CompactAnthropicHistoryWithOutcome is
// fail-safe: it returns req.Raw unchanged on any ambiguity, so this never breaks a turn.
//
// It records the attempt outcome on /metrics (fired/bailed/off + bail reason + shed — all
// WITNESSED, what fak authored) so a silent failure is visible, and returns whether it FIRED —
// the caller threads that through so the post-response provider cache_read (an OBSERVED value
// fak relays, never a fak claim) is recorded only on turns this actually compacted.
func (s *Server) maybeCompactAnthropicRaw(req *agent.AnthropicMessagesRequest) (fired bool) {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthrough() {
		return false
	}
	if s.compactHistoryBudget <= 0 {
		s.metrics.observeCompaction(agent.CompactOutcome{}, true) // configured OFF
		return false
	}
	// Offensive half (#806): if the caller left NO cache_control breakpoint, place one on the
	// stable system+tools head so the provider caches it — AND so the compaction below has an
	// anchor to protect (a body with no breakpoint bails CompactReasonNoBreakpoint). Fail-safe
	// identity: a body that already carries a breakpoint (the Claude Code shape), or has no stable
	// head, is returned unchanged. Like compaction this only ADDS a caching hint to the FORWARDED
	// bytes; the decoded req.Messages the kernel adjudicates are untouched, so the trust boundary
	// is unchanged. The net effect is visible on the SAME readback: placement makes compaction
	// fire (CompactionFired) and the provider cache_read it relays now covers the cached head.
	req.Raw = agent.PlaceAnthropicCacheBreakpoint(req.Raw)
	out, outcome := agent.CompactAnthropicHistoryWithOutcome(req.Raw, s.compactHistoryBudget)
	req.Raw = out
	s.metrics.observeCompaction(outcome, false)
	return outcome.Reason == agent.CompactReasonNone
}

// maybePlanAnthropicRaw is the ctxplan planned-view req.Raw transform for the Anthropic
// PASSTHROUGH (#927 — the deferred #555 req.Raw step the buffered maybePlanMessages path
// could not reach, because that route forwards req.Raw byte-for-byte). When the view
// planner is enabled (--ctx-view-budget > 0), it plans req.Messages into an O(1)
// resident view and materializes that view onto req.Raw: each message the planner elided
// (beyond the cached prefix) is replaced in place by a same-role stub, while the prefix
// bytes and every resident message's original bytes stay byte-identical so the upstream
// cache hit survives.
//
// A no-op (identity) unless the gateway fronts the REAL Anthropic API and the view
// planner is enabled — so a deploy that leaves --ctx-view-budget at 0 sees the body
// byte-for-byte unchanged (the same posture as the buffered path). Fail-safe:
// agent.CompactAnthropicHistoryToView returns req.Raw unchanged on any ambiguity, so this
// never breaks a turn. Applied to req.Raw ONLY — the decoded req.Messages the kernel
// adjudicates are untouched, so the trust boundary is unchanged.
func (s *Server) maybePlanAnthropicRaw(ctx context.Context, trace string, req *agent.AnthropicMessagesRequest) bool {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthrough() {
		return false
	}
	if s.ctxView == nil || !s.ctxView.Enabled {
		return false
	}
	planned := s.maybePlanMessages(ctx, trace, req.Messages)
	if len(planned) >= len(req.Messages) {
		return false // the planner did not elide anything — nothing to materialize
	}
	out, outcome := agent.CompactAnthropicHistoryToView(req.Raw, planned)
	if outcome.Reason != agent.CompactReasonNone {
		return false // bailed — identity (fail-safe)
	}
	req.Raw = out
	return true
}

// maybeCompactInboundTools is the INBOUND twin of maybeCompactAnthropicRaw: on the
// Anthropic passthrough it prunes tool DEFINITIONS the capability floor can never admit
// (s.toolFloorDenies(name) — a DEFAULT_DENY for every arg, never an arg-conditional tool)
// from the outbound tools[], splicing on the original bytes so the cache_control prefix
// stays byte-identical and the upstream prompt-cache hit survives (promptmmu.CompactInboundTools).
//
// It is behavior-preserving by construction: if the model somehow names a pruned tool, the
// kernel still DEFAULT_DENYs the call — so removing the advertisement only shrinks the
// uncached tool-def tokens, never the reachable action set. A no-op (identity) unless the
// gateway is fronting the real Anthropic API (only there is req.Raw forwarded verbatim) and
// the host supplied the floor predicate. promptmmu is fail-safe: any ambiguity returns
// req.Raw unchanged with a named SkipReason, so this never breaks a turn.
func (s *Server) maybeCompactInboundTools(req *agent.AnthropicMessagesRequest) (pruned []string) {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthrough() || s.toolFloorDenies == nil {
		return nil
	}
	if len(req.Tools) == 0 {
		return nil
	}
	drop := make(map[string]bool, len(req.Tools))
	for _, t := range req.Tools {
		if name := t.Function.Name; name != "" && s.toolFloorDenies(name) {
			drop[name] = true
		}
	}
	if len(drop) == 0 {
		return nil
	}
	res := promptmmu.CompactInboundTools(req.Raw, promptmmu.ToolPlan{Drop: drop}, func(b []byte) error {
		_, err := agent.DecodeAnthropicMessagesRequest(b)
		return err
	})
	if !res.Changed {
		return nil
	}
	req.Raw = res.Body
	return res.Pruned
}

// writeAnthropicTurn renders a fully-formed turn to the wire as either a buffered JSON
// response or a synthesized SSE sequence, matching the request's stream flag. Used by
// the short-circuit guards (e.g. parrotLoopSteer) that produce a complete turn without
// running the planner, so they don't each duplicate the stream/buffered plumbing.
func (s *Server) writeAnthropicTurn(w http.ResponseWriter, stream bool, turn *anthropicTurn) {
	if !stream {
		writeJSON(w, http.StatusOK, anthropicMessageResponse{
			ID: turn.ID, Type: "message", Role: "assistant", Model: turn.Model,
			Content: turn.Blocks, StopReason: turn.Stop, StopSequence: nil, Usage: turn.Usage,
			Fak: fakExtFrom(turn.Adjs, turn.ResultAdmissions),
		})
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		// No flush support: fall back to the buffered shape rather than failing.
		writeJSON(w, http.StatusOK, anthropicMessageResponse{
			ID: turn.ID, Type: "message", Role: "assistant", Model: turn.Model,
			Content: turn.Blocks, StopReason: turn.Stop, StopSequence: nil, Usage: turn.Usage,
		})
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	send := anthropicSSESender(w, flusher)
	send("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": turn.ID, "type": "message", "role": "assistant", "model": turn.Model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]int{"input_tokens": turn.Usage.InputTokens, "output_tokens": 0},
		},
	})
	streamAnthropicBlocks(send, turn.Blocks, turn.Stop, turn.Usage)
}

func (s *Server) completeAnthropicTurn(ctx context.Context, req *agent.AnthropicMessagesRequest, reqTrace string, sessionTurn servedSessionTurn, id, model, upstreamKey, upstreamBeta string) (*anthropicTurn, error) {
	// Arm the RESULT-side floor BEFORE the planner runs — the exact parity the
	// OpenAI proxy has at http.go (#77). DecodeAnthropicMessagesRequest already
	// turned each inbound Anthropic `tool_result` block into a canonical RoleTool
	// message, so admitInboundResults routes each through k.AdmitResult keyed on
	// reqTrace: a poisoned/secret-bearing result is PAGED OUT in place (the model's
	// KV never ingests the poison) and an untrusted-source result RAISES the trace's
	// IFC taint high-water mark. That high-water mark is what adjudicateProposed
	// (k.Decide, keyed on the SAME reqTrace, already wired below) reads to REFUSE an
	// exfil call on a tainted session. Without this, the Anthropic wire — the one
	// Claude Code uses natively — silently had a weaker floor than the OpenAI wire:
	// a tainted result reached the model raw and the egress call was allowed.
	resultAdmissions, err := s.admitInboundResults(ctx, req.Messages, reqTrace)
	if err != nil {
		return nil, err
	}

	// Forward the Claude-Code client's per-request sampling (max_tokens, temperature,
	// top_p, stop_sequences) to the upstream model. max_tokens is REQUIRED on the
	// Anthropic wire, so a real client always sends one — honoring it is what stops a
	// long turn truncating at the planner's 1024 floor (#62). An omitted optional
	// field is a no-op and keeps the planner default.
	var temp *float64
	if req.Temperature != 0 {
		temp = &req.Temperature
	}
	opts := []agent.SampleOpt{
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxTokens)),
		agent.WithTemperature(temp),
		agent.WithTopP(req.TopP),
		agent.WithStop(req.StopSequences),
	}
	// anthropic→anthropic passthrough: when the upstream IS the real Anthropic API,
	// forward the client's ORIGINAL request bytes verbatim (so its cache_control
	// prefix survives → a real cache hit, not a re-billed prefix) and authenticate
	// with the client's OWN key (transparent hop, no second secret). The kernel still
	// adjudicates the RESPONSE's tool calls below; only the request is byte-faithful.
	// WithRawRequestBody makes the sampling opts above no-ops (the client's own values
	// are already in req.Raw; re-injecting them would change the cached prefix bytes).
	if s.anthropicPassthrough() {
		opts = append(opts, agent.WithRawRequestBody(req.Raw), agent.WithUpstreamAPIKey(upstreamKey), agent.WithUpstreamBeta(upstreamBeta))
	}
	comp, err := s.completeServed(ctx, sessionTurn, req.Messages, req.Tools, opts...)
	if err != nil {
		return nil, err
	}

	asst := comp.Message
	asst.Role = agent.RoleAssistant
	kept, adjs, dropped := s.adjudicateProposed(ctx, asst.ToolCalls, reqTrace)
	asst.ToolCalls = kept

	blocks := agent.AnthropicResponseBlocks(asst)
	stop := agent.AnthropicStopReason(comp.FinishReason, len(kept) > 0)

	// Make the kernel's decisions LEGIBLE in-band. On the Anthropic wire Claude
	// Code reads the content blocks (and feeds them back to its model) but not the
	// `fak` extension, so a dropped or repaired call is otherwise invisible — the
	// agent re-proposes a denied call forever, or proceeds unaware its args were
	// rewritten. Whenever a drop or repair happened, prepend a short text note
	// describing it. The all-denied case is just the special case where there is
	// no surviving prose and no tool_use block, so the note becomes the whole turn
	// (the previous denySummary behavior, now generalized to partial denies too).
	if dropped > 0 || anyRepaired(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			blocks = prependTextBlock(blocks, note)
		}
	}
	// If the result-side floor paged out an inbound tool result, say so in-band too:
	// the model is about to read a quarantine stub where its tool output was, and a
	// silent stub reads as a broken tool. Naming the quarantine lets the agent adapt.
	if note := resultAdmissionNote(resultAdmissions); note != "" {
		blocks = prependTextBlock(blocks, note)
	}
	// Echo the model the client asked for (Anthropic reflects the requested id);
	// fall back to the gateway's configured model when the client omitted it.
	if model == "" {
		model = req.Model
	}
	if model == "" {
		model = s.model
	}
	if id == "" {
		id = "msg_fak_" + itoa(uint64(time.Now().UnixNano()))
	}
	return &anthropicTurn{
		ID: id, Model: model, Blocks: blocks, Stop: stop,
		Usage: anthropicUsage{
			InputTokens:              comp.Usage.PromptTokens,
			OutputTokens:             comp.Usage.CompletionTokens,
			CacheReadInputTokens:     comp.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: comp.Usage.CacheCreationInputTokens,
		},
		Adjs:             adjs,
		ResultAdmissions: resultAdmissions,
	}, nil
}

// anthropicPassthrough reports whether the gateway is fronting the REAL Anthropic
// Messages API — i.e. the live planner is an HTTPPlanner configured for the
// anthropic provider wire. Only then is byte-exact request passthrough both
// possible (the inbound and upstream wires match) and necessary (to preserve the
// client's prompt-cache prefix). For the mock, the in-kernel model, or any
// non-Anthropic upstream, this is false and the turn is built exactly as before.
func (s *Server) anthropicPassthrough() bool {
	hp, ok := s.planner.(*agent.HTTPPlanner)
	return ok && hp.Provider == agent.ProviderAnthropic
}

// resultAdmissionNote names any inbound tool result the kernel PAGED OUT, so the
// agent learns its tool output was quarantined (replaced with a stub) rather than
// silently broken. Returns "" when every result was a clean allow.
func resultAdmissionNote(adms []ResultAdmission) string {
	quarantined := make([]string, 0, len(adms))
	for _, a := range adms {
		if a.Verdict.Kind == "QUARANTINE" {
			tool := a.Tool
			if tool == "" {
				tool = "tool_result"
			}
			quarantined = append(quarantined, fmt.Sprintf("%s (%s)", tool, a.Verdict.Reason))
		}
	}
	if len(quarantined) == 0 {
		return ""
	}
	return "[fak] quarantined " + strconv.Itoa(len(quarantined)) +
		" inbound tool result(s) before the model read them: " + strings.Join(quarantined, "; ") +
		". The content was paged out (a stub stands in its place); do not treat the stub as the real result."
}

// anyRepaired reports whether the kernel rewrote any admitted call's arguments.
func anyRepaired(adjs []ToolAdjudication) bool {
	for _, a := range adjs {
		if a.Admitted && a.Verdict.Kind == "TRANSFORM" {
			return true
		}
	}
	return false
}

// loopBreakThreshold is how long a degenerate tail of assistant turns must grow before
// the gateway short-circuits the loop. A capable model never repeats itself verbatim or
// echoes a kernel notice; a small local model fronted by the kernel often does exactly
// that — turn after turn, with no new tool call and no progress — until the harness
// turn-cap ends the session with an empty result. At threshold=2 the third such
// degenerate turn trips the break, reclaiming the rest of the turn budget.
const loopBreakThreshold = 2

// degenerateStreak counts the trailing run of NON-PROGRESSING assistant turns in the
// replayed history. A turn is degenerate when it is text-only (no tool call survived to
// drive work forward) AND it is either:
//
//   - a `[fak]` echo: text the KERNEL originates (adjudicationNote / resultAdmissionNote)
//     that a model should never produce — the model parroting the refusal note back; or
//   - a verbatim repeat of the previous assistant turn — the model emitting the SAME
//     prose every turn (e.g. the same graceful "I can't, policy blocks it" refusal).
//
// Counted from the END so a single stale line in a long healthy transcript never trips
// it; only an unbroken degenerate tail does. A turn that carries a tool_use, or differs
// from its predecessor and is not a `[fak]` echo, is real progress and breaks the run.
// Stateless: it reads exactly what Claude Code replayed.
func degenerateStreak(messages []agent.Message) int {
	// Collect the trailing run of text-only assistant turns (most recent first). A turn
	// carrying a tool_use is forward progress and ends the run.
	var tail []string
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if m.Role != agent.RoleAssistant {
			continue // user / tool turns interleave; skip without breaking the run
		}
		if len(m.ToolCalls) > 0 {
			break
		}
		tail = append(tail, m.Content)
	}
	// Count, from the most recent backward, how many turns are degenerate: a `[fak]`
	// echo (kernel-originated text the model should never produce), or a verbatim repeat
	// of the adjacent more-recent assistant turn (the model emitting the same prose).
	n := 0
	for i, c := range tail {
		isEcho := strings.Contains(c, "[fak]")
		isRepeat := c != "" && ((i > 0 && c == tail[i-1]) || (i+1 < len(tail) && c == tail[i+1]))
		if isEcho || isRepeat {
			n++
			continue
		}
		break
	}
	return n
}

// repetitionLoopSteer returns a terminal corrective turn when the model is stuck in a
// degenerate tail (degenerateStreak ≥ loopBreakThreshold) — echoing the kernel's `[fak]`
// notes or repeating the same prose with no progress — or nil otherwise. The corrective
// turn is a single plain-text assistant message, deliberately NOT prefixed with `[fak]`
// (so it can't itself feed the echo detector) and distinct from any repeated line, that
// ends the turn (end_turn). Returned BEFORE the planner runs, it breaks the loop
// deterministically and cheaply (no model round-trip) and Claude Code reads a normal
// terminal assistant turn so its agent loop settles instead of grinding to the turn-cap.
func repetitionLoopSteer(messages []agent.Message, id, model string) *anthropicTurn {
	if degenerateStreak(messages) < loopBreakThreshold {
		return nil
	}
	const steer = "I was repeating myself without making progress: a tool I tried is " +
		"blocked by the security policy and cannot be used. Stopping that loop. If the " +
		"request needs the blocked tool I cannot complete it; otherwise tell me what to " +
		"answer and I will respond directly."
	if id == "" {
		id = "msg_fak_" + itoa(uint64(time.Now().UnixNano()))
	}
	return &anthropicTurn{
		ID:    id,
		Model: model,
		Blocks: []agent.AnthropicBlockOut{
			{Type: "text", Text: steer},
		},
		Stop: "end_turn",
	}
}

// prependTextBlock inserts an in-band [fak] note as the FIRST content block so a
// client that reads content top-to-bottom (Claude Code) sees the kernel's decision
// before the surviving tool_use blocks it is about to run. The note never replaces
// model prose — existing text/tool_use blocks follow it untouched.
func prependTextBlock(blocks []agent.AnthropicBlockOut, text string) []agent.AnthropicBlockOut {
	out := make([]agent.AnthropicBlockOut, 0, len(blocks)+1)
	out = append(out, agent.AnthropicBlockOut{Type: "text", Text: text})
	return append(out, blocks...)
}

// fakExtFrom builds the response extension from a turn's proposed-call
// adjudications and inbound-result admissions, or nil when there is nothing to
// report (so the `fak` key is omitted on a turn with no tool activity at all).
func fakExtFrom(adjs []ToolAdjudication, results []ResultAdmission) *FakExt {
	if len(adjs) == 0 && len(results) == 0 {
		return nil
	}
	return &FakExt{Adjudications: adjs, ResultAdmissions: results}
}

func (s *Server) streamAnthropicPending(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string, sessionTurn servedSessionTurn, upstreamKey, upstreamBeta string, compacted bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		began := time.Now()
		turn, err := s.completeAnthropicTurn(r.Context(), req, reqTrace, sessionTurn, "", "", upstreamKey, upstreamBeta)
		if err != nil {
			s.logf("gateway: upstream model error (messages): %v", err)
			status, code, msg := s.plannerErrorStatus(err)
			writeErrCode(w, status, code, msg)
			return
		}
		if compacted {
			s.metrics.recordCompactionCacheRead(turn.Usage.CacheReadInputTokens)
			s.observeResetHealth(reqTrace, turn.Usage.InputTokens, turn.Usage.CacheReadInputTokens, turn.Usage.CacheCreationInputTokens)
		}
		s.logInferenceTurn(reqTrace, "anthropic_messages", true, agent.Usage{
			PromptTokens:             turn.Usage.InputTokens,
			CompletionTokens:         turn.Usage.OutputTokens,
			CacheReadInputTokens:     turn.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: turn.Usage.CacheCreationInputTokens,
		}, turn.Stop, time.Since(began), compacted)
		writeJSON(w, http.StatusOK, anthropicMessageResponse{
			ID: turn.ID, Type: "message", Role: "assistant", Model: turn.Model,
			Content: turn.Blocks, StopReason: turn.Stop, StopSequence: nil, Usage: turn.Usage,
			Fak: fakExtFrom(turn.Adjs, turn.ResultAdmissions),
		})
		return
	}
	model := req.Model
	if model == "" {
		model = s.model
	}
	id := "msg_fak_" + itoa(uint64(time.Now().UnixNano()))
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	send := anthropicSSESender(w, flusher)
	send("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]int{"input_tokens": agent.EstimateAnthropicTokens(req), "output_tokens": 0},
		},
	})

	type turnResult struct {
		turn *anthropicTurn
		err  error
	}
	done := make(chan turnResult, 1)
	began := time.Now()
	go func() {
		turn, err := s.completeAnthropicTurn(r.Context(), req, reqTrace, sessionTurn, id, model, upstreamKey, upstreamBeta)
		done <- turnResult{turn: turn, err: err}
	}()

	ticker := time.NewTicker(anthropicStreamPingInterval)
	defer ticker.Stop()
	for {
		select {
		case res := <-done:
			if res.err != nil {
				// The in-kernel decode path Claude Code actually hits: classify so an in-kernel
				// GPU OOM surfaces an actionable message in the SSE error frame instead of the
				// opaque "upstream model error". A genuine upstream error still falls through to
				// the default arm's "upstream model error" (no behavior change, no body leak).
				_, _, msg := s.plannerErrorStatus(res.err)
				s.logf("gateway: messages stream turn error: %v", res.err)
				send("error", map[string]any{
					"type":  "error",
					"error": map[string]any{"type": "api_error", "message": msg},
				})
				return
			}
			if compacted {
				s.metrics.recordCompactionCacheRead(res.turn.Usage.CacheReadInputTokens)
				s.observeResetHealth(reqTrace, res.turn.Usage.InputTokens, res.turn.Usage.CacheReadInputTokens, res.turn.Usage.CacheCreationInputTokens)
			}
			s.logInferenceTurn(reqTrace, "anthropic_messages", true, agent.Usage{
				PromptTokens:             res.turn.Usage.InputTokens,
				CompletionTokens:         res.turn.Usage.OutputTokens,
				CacheReadInputTokens:     res.turn.Usage.CacheReadInputTokens,
				CacheCreationInputTokens: res.turn.Usage.CacheCreationInputTokens,
			}, res.turn.Stop, time.Since(began), compacted)
			streamAnthropicBlocks(send, res.turn.Blocks, res.turn.Stop, res.turn.Usage)
			return
		case <-ticker.C:
			send("ping", map[string]any{"type": "ping"})
		case <-r.Context().Done():
			return
		}
	}
}

func anthropicSSESender(w http.ResponseWriter, flusher http.Flusher) func(event string, data any) {
	return func(event string, data any) {
		b, _ := json.Marshal(data)
		_, _ = w.Write([]byte("event: " + event + "\n"))
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(b)
		_, _ = w.Write([]byte("\n\n"))
		flusher.Flush()
	}
}

func streamAnthropicBlocks(send func(string, any), blocks []agent.AnthropicBlockOut, stop string, usage anthropicUsage) {
	for i, blk := range blocks {
		switch blk.Type {
		case "tool_use":
			send("content_block_start", map[string]any{
				"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "tool_use", "id": blk.ID, "name": blk.Name, "input": map[string]any{}},
			})
			// The whole (already-validated) argument object as one input_json_delta.
			send("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": string(blk.Input)},
			})
		default: // text
			send("content_block_start", map[string]any{
				"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
			send("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "text_delta", "text": blk.Text},
			})
		}
		send("content_block_stop", map[string]any{"type": "content_block_stop", "index": i})
	}

	// The terminal message_delta carries the REAL usage from the finished turn — the
	// message_start figure was a pre-completion estimate. Report the upstream's true
	// input_tokens (the uncached remainder) plus the cache counters, so a passthrough
	// turn's cache hit reaches the client's accounting. Counters are omitted when zero
	// (a local-model turn streams the same shape as before).
	finalUsage := map[string]int{
		"input_tokens":  usage.InputTokens,
		"output_tokens": usage.OutputTokens,
	}
	if usage.CacheReadInputTokens > 0 {
		finalUsage["cache_read_input_tokens"] = usage.CacheReadInputTokens
	}
	if usage.CacheCreationInputTokens > 0 {
		finalUsage["cache_creation_input_tokens"] = usage.CacheCreationInputTokens
	}
	send("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
		"usage": finalUsage,
	})
	send("message_stop", map[string]any{"type": "message_stop"})
}

// handleAnthropicCountTokens answers POST /v1/messages/count_tokens with a cheap,
// tokenizer-free estimate. Claude Code treats this as optional (a 404 is fine), but
// answering it keeps its context-management heuristics from flying blind.
func (s *Server) handleAnthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var raw json.RawMessage
	if err := decodeJSON(w, r, &raw); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"input_tokens": agent.EstimateAnthropicTokens(req)})
}
