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
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
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
	// Compaction is the continuation contract for the compaction boundary this turn
	// crossed (#2422), or nil when it crossed none. It rides the `fak` extension as the
	// orchestrator-readable twin of the in-band `[fak]` boundary note the same record
	// rendered into this turn's leading text block.
	Compaction *CompactionContract
}

// fakExt builds this turn's `fak` response extension: the tool-activity view plus, on a turn
// that crossed a compaction boundary, the continuation contract (#2422). A turn with neither
// yields nil, so the `fak` key stays omitted rather than appearing empty.
func (t *anthropicTurn) fakExt() *FakExt {
	ext := fakExtFrom(t.Adjs, t.ResultAdmissions)
	if t.Compaction == nil {
		return ext
	}
	if ext == nil {
		ext = &FakExt{}
	}
	ext.Compaction = t.Compaction
	return ext
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
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	if s.checkWarmupPending(w) {
		return
	}
	if dl := s.ScalarConfig().CompletionDeadlineMs; dl > 0 {
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(dl)*time.Millisecond)
		defer cancel()
		r = r.WithContext(ctx)
	}
	// Release the EP follower ranks BEFORE this rank enters the decode, onto THIS wire's
	// own route (#5528). Everything below — the passthrough turn, the planner turn, and
	// both streaming arms — blocks for the whole multi-rank decode, and rank-local expert
	// parallelism makes progress only if every rank runs the same forward pass. Without
	// this, an Anthropic request left the front rank alone in a collective the other ranks
	// were never released into. Placement matches the chat and legacy wires: after the
	// method check, before anything reads the body (the helper reads and restores it).
	//
	// Inert on a single-rank serve — with FAK_EP_FANOUT_ADDRS unset there are no follower
	// URLs and this is a no-op, which is why the gap survived. The consequence on real
	// multi-rank hardware (hang, timeout, or a silently degraded single-rank answer) is
	// inferred from the AllReduce contract, not measured.
	//
	// NOT under `serve --native` (#5532). The bridge's unit is one inbound body == one
	// forward pass, which holds for the proxy turn below but NOT for the native branch: a
	// follower rank is another process running the same binary with the same config, so a
	// mirrored body reaches the same s.native branch and runs the WHOLE owned loop — its own
	// N forward passes and its own tool dispatches through the kernel. Measured, not
	// inferred: with one follower configured a two-turn native request drove 4 forward passes
	// and 2 tool-result turns instead of 2 and 1. That is the exact property
	// epFanoutExemptRoutes already refuses to mirror on /v1/fak/agent/sessions
	// (epExemptOwnedLoop, #5528), and the same judgement applies here: a duplicated REAL tool
	// side effect per rank is unrecoverable, while a leader alone in a collective is loud.
	// So the owned-loop arm is left uncovered by the request mirror on purpose. Covering it
	// wants the in-loop rank barrier that already exists — model.EPDecodeCoordinator /
	// model.RunEPFollower (#4835), announced from Session.Prefill and Session.Step — which
	// still has no serve wiring; until that lands, a native multi-rank serve enters its
	// collectives on rank 0 alone.
	if !s.native {
		waitEPFanout, ok := s.startEPFanoutFollowers(w, r, epRouteMessages)
		if !ok {
			return
		}
		defer waitEPFanout()
	}
	req, ok := s.readAnthropicMessagesRequest(w, r)
	if !ok {
		return
	}
	ctx := r.Context()
	reqTrace := s.traceFor(r.Header.Get("X-Trace-Id"))
	appendSessionLedger(reqTrace, "user_message", turnLedgerSummary(req))
	// C1 read-scope floor (#4192): the first turn a trace serves binds its owning principal, so a
	// later fak_context_restore / fak_context_spans read-self op can be scoped to it. First-writer-
	// wins; "" on the no-RequireKey loopback (single-tenant).
	s.bindTraceOwner(reqTrace, principalFor(r, ""))
	// AUTHORITY principal floor (#2412): stamp this turn's inbound authority label so the
	// adjudication seam can type-check consent-shaped acts against it. Absent header => the
	// direct interactive wire => human; a relay (webhook/A2A peer/scheduler) that stamped
	// the class header cannot present as human. Per-turn (last-writer-wins), distinct from
	// the tenant-isolation owner above.
	s.bindTracePrincipal(reqTrace, classifyPrincipal(r.Header.Get(inboundPrincipalHeader)))
	// Native-harness keystone (#1316/#1837): when `fak serve --native` is set, drive
	// fak's OWN agent loop instead of the single-shot proxy turn below. The owned loop
	// runs its own per-turn session gate (WithSessionGate over the same
	// DecideSession/DebitSession hooks), so it is branched BEFORE the request-boundary
	// admission to avoid a double Decide.
	if s.native && req.Stream {
		s.serveNativeMessagesStream(w, r, req, reqTrace)
		return
	}
	if s.native {
		s.serveNativeMessages(w, r, req, reqTrace)
		return
	}
	// The native path above owns its admission. Proxy requests enter the shared
	// model-facing boundary here; useHTTPTrace preserves the established trace.
	ctx, reqTrace, messages, sessionTurn, admitted := s.admitServedRequest(w, r, req.Messages)
	defer sessionTurn.complete()
	if !admitted {
		return
	}
	req.Messages = messages
	applySessionPaceToAnthropicRequest(req, sessionTurn)
	s.injectGuardRecoveryPrompt(req, reqTrace)
	prep := s.prepareServedAnthropicRequest(ctx, r, req, reqTrace, sessionTurn)
	compacted := prep.compacted
	contextEvent := prep.contextEvent
	hcoh := prep.hcoh
	upstreamKey := prep.upstreamKey
	upstreamBeta := prep.upstreamBeta

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
		if s.anthropicPassthroughFor(req.Model) && s.streamAnthropicPassthroughLive(w, r, req, reqTrace, sessionTurn, upstreamKey, upstreamBeta, compacted, contextEvent, hcoh) {
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
		s.streamAnthropicPending(w, r, req, reqTrace, sessionTurn, upstreamKey, upstreamBeta, compacted, contextEvent, hcoh)
		return
	}

	began := time.Now()
	turn, err := s.completeAnthropicTurn(ctx, req, reqTrace, sessionTurn, "", "", upstreamKey, upstreamBeta)
	if err != nil {
		s.renderTurnDebugError(reqTrace, "anthropic_messages", err, time.Since(began))
		// Classify like the chat-completions path: an in-kernel device OOM becomes a specific,
		// actionable 503; a genuine upstream failure stays the opaque 502 with the raw provider
		// body kept off the wire. writeErrCode with an empty code reproduces the historical
		// code:null body byte-for-byte, so the non-OOM response is unchanged (#346).
		// Classify once for the operator log WITHOUT the metric side effect
		// (upstreamErrorStatus is the pure mapper; writeUpstreamErr below does the single
		// metric-observing classify + write, so the failure is counted exactly once).
		status, code, _ := upstreamErrorStatus(err)
		s.logf("gateway: messages turn error (%s, HTTP %d): %v", code, status, err)
		s.writeUpstreamErr(w, err)
		return
	}

	// On a turn we actually compacted, record the provider's OBSERVED cache_read (relayed, not a
	// fak claim) so the net effect is visible on /metrics next to the WITNESSED shed.
	if compacted {
		s.metrics.recordCompactionCacheRead(turn.Usage.CacheReadInputTokens)
		s.observeResetHealth(reqTrace, turn.Usage.InputTokens, turn.Usage.CacheReadInputTokens, turn.Usage.CacheCreationInputTokens)
	}
	// Harness-coherence (#1132): fold this served turn into the per-trace coordinator with the
	// content-free inbound-prefix digest taken before transforms and the provider's relayed cache
	// counters. fakWorldBreak is false here — fak's deliberate cachemeta world-break does not run on
	// this passthrough; a deliberate break, when wired, would set it so the coordinator attributes a
	// changed prefix to fak rather than the harness.
	s.observeHarnessCoherenceAndArm(reqTrace, time.Now(), hcoh.inboundPrefixDigest, compacted, hcoh.fakBail,
		false /*fakWorldBreak*/, false /*sealed*/, int64(turn.Usage.CacheReadInputTokens), int64(turn.Usage.CacheCreationInputTokens), int64(turn.Usage.InputTokens))
	s.logInferenceTurnWithContextEvent(reqTrace, "anthropic_messages", false, agent.Usage{
		PromptTokens:             turn.Usage.InputTokens,
		CompletionTokens:         turn.Usage.OutputTokens,
		CacheReadInputTokens:     turn.Usage.CacheReadInputTokens,
		CacheCreationInputTokens: turn.Usage.CacheCreationInputTokens,
	}, turn.Stop, time.Since(began), compacted, contextEvent)

	writeJSON(w, http.StatusOK, anthropicMessageResponse{
		ID: turn.ID, Type: "message", Role: "assistant", Model: turn.Model,
		Content: turn.Blocks, StopReason: turn.Stop, StopSequence: nil, Usage: turn.Usage,
		Fak: turn.fakExt(),
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

func (s *Server) takeGuardRecoveryPrompt() string {
	if s == nil {
		return ""
	}
	s.guardRecoveryMu.Lock()
	defer s.guardRecoveryMu.Unlock()
	out := strings.TrimSpace(s.guardRecoveryPrompt)
	s.guardRecoveryPrompt = ""
	return out
}

func (s *Server) injectGuardRecoveryPrompt(req *agent.AnthropicMessagesRequest, trace string) bool {
	if req == nil {
		return false
	}
	prompt := s.takeGuardRecoveryPrompt()
	if prompt == "" {
		return false
	}
	if !mergeGuardRecoveryPrompt(req, prompt) {
		req.Messages = append(req.Messages, agent.Message{Role: agent.RoleUser, Content: prompt})
	}
	if len(req.Raw) != 0 {
		if raw, ok := injectAnthropicUserTextRaw(req.Raw, prompt); ok {
			req.Raw = raw
		}
	}
	sessionctl.RecordGuardRecoveryNext(trace, prompt)
	return true
}

func mergeGuardRecoveryPrompt(req *agent.AnthropicMessagesRequest, text string) bool {
	if req == nil {
		return false
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		msg := &req.Messages[i]
		if msg.Role != agent.RoleUser || strings.TrimSpace(msg.Content) == "" {
			continue
		}
		msg.Content = msg.Content + "\n\n" + text
		return true
	}
	return false
}

func injectAnthropicUserTextRaw(raw []byte, text string) ([]byte, bool) {
	text = strings.TrimSpace(text)
	if len(raw) == 0 || text == "" {
		return nil, false
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return nil, false
	}
	messagesRaw, ok := obj["messages"]
	messagesTrimmed := bytes.TrimSpace(messagesRaw)
	if !ok || len(messagesTrimmed) == 0 || messagesTrimmed[0] != '[' {
		return nil, false
	}
	var elems []json.RawMessage
	if json.Unmarshal(messagesRaw, &elems) != nil {
		return nil, false
	}
	if raw, ok := mergeLastAnthropicUserTextRaw(raw, messagesRaw, elems, text); ok {
		return raw, true
	}
	return appendAnthropicUserTextRaw(raw, text)
}

func appendAnthropicUserTextRaw(raw []byte, text string) ([]byte, bool) {
	text = strings.TrimSpace(text)
	if len(raw) == 0 || text == "" {
		return nil, false
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return nil, false
	}
	messagesRaw, ok := obj["messages"]
	messagesTrimmed := bytes.TrimSpace(messagesRaw)
	if !ok || len(messagesTrimmed) == 0 || messagesTrimmed[0] != '[' {
		return nil, false
	}
	var elems []json.RawMessage
	if json.Unmarshal(messagesRaw, &elems) != nil {
		return nil, false
	}
	base := bytes.Index(raw, messagesRaw)
	if base < 0 || len(messagesRaw) < 2 {
		return nil, false
	}
	closeIdx := bytes.LastIndexByte(messagesRaw, ']')
	if closeIdx < 0 {
		return nil, false
	}
	insert := base + closeIdx // before the closing ']'
	msg, err := json.Marshal(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{Role: "user", Content: text})
	if err != nil {
		return nil, false
	}
	var out bytes.Buffer
	out.Grow(len(raw) + len(msg) + 1)
	out.Write(raw[:insert])
	if len(elems) > 0 {
		out.WriteByte(',')
	}
	out.Write(msg)
	out.Write(raw[insert:])
	b := out.Bytes()
	if _, err := agent.DecodeAnthropicMessagesRequest(b); err != nil {
		return nil, false
	}
	return b, true
}

func mergeLastAnthropicUserTextRaw(raw, messagesRaw []byte, elems []json.RawMessage, text string) ([]byte, bool) {
	messagesBase := bytes.Index(raw, messagesRaw)
	if messagesBase < 0 {
		return nil, false
	}
	for i := len(elems) - 1; i >= 0; i-- {
		var msg struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(elems[i], &msg) != nil || msg.Role != "user" {
			continue
		}
		elemOffset := bytes.LastIndex(messagesRaw, elems[i])
		if elemOffset < 0 {
			continue
		}
		contentOffset := bytes.Index(elems[i], msg.Content)
		if contentOffset < 0 {
			continue
		}
		start := messagesBase + elemOffset + contentOffset
		end := start + len(msg.Content)
		if existing, ok := asRawJSONString(msg.Content); ok && strings.TrimSpace(existing) != "" {
			merged := existing + "\n\n" + text
			repl, err := json.Marshal(merged)
			if err != nil {
				return nil, false
			}
			return replaceAnthropicRawRange(raw, start, end, repl)
		}
		var blocks []map[string]json.RawMessage
		if json.Unmarshal(msg.Content, &blocks) != nil {
			continue
		}
		for j := len(blocks) - 1; j >= 0; j-- {
			var typ string
			if json.Unmarshal(blocks[j]["type"], &typ) != nil || typ != "text" {
				continue
			}
			var existing string
			if json.Unmarshal(blocks[j]["text"], &existing) != nil || strings.TrimSpace(existing) == "" {
				continue
			}
			merged, err := json.Marshal(existing + "\n\n" + text)
			if err != nil {
				return nil, false
			}
			blocks[j]["text"] = merged
			content, err := json.Marshal(blocks)
			if err != nil {
				return nil, false
			}
			return replaceAnthropicRawRange(raw, start, end, content)
		}
		blocks = append(blocks, map[string]json.RawMessage{
			"type": json.RawMessage(`"text"`),
			"text": json.RawMessage(mustMarshalJSONString(text)),
		})
		content, err := json.Marshal(blocks)
		if err != nil {
			return nil, false
		}
		return replaceAnthropicRawRange(raw, start, end, content)
	}
	return nil, false
}

func asRawJSONString(raw json.RawMessage) (string, bool) {
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return "", false
	}
	return s, true
}

func mustMarshalJSONString(s string) []byte {
	raw, err := json.Marshal(s)
	if err != nil {
		return []byte(`""`)
	}
	return raw
}

func replaceAnthropicRawRange(raw []byte, start, end int, repl []byte) ([]byte, bool) {
	if start < 0 || end < start || end > len(raw) {
		return nil, false
	}
	var out bytes.Buffer
	out.Grow(len(raw) + len(repl) - (end - start))
	out.Write(raw[:start])
	out.Write(repl)
	out.Write(raw[end:])
	b := out.Bytes()
	if _, err := agent.DecodeAnthropicMessagesRequest(b); err != nil {
		return nil, false
	}
	return b, true
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
//
// Callers outside the live request boundary (tests, and any future non-session caller) have no
// turn horizon or trace identity to offer, so this defaults both — the same conservative
// "unknown horizon, never cold" behavior compactAnthropicRawWithReason has always had.
func (s *Server) maybeCompactAnthropicRaw(req *agent.AnthropicMessagesRequest) (fired bool) {
	fired, _ = s.compactAnthropicRawWithReason(req, 0, "")
	return fired
}

// compactAnthropicRawWithReason is maybeCompactAnthropicRaw with the raw CompactOutcome.Reason
// exposed so the harness-coherence seam (#1132) can build a TurnObservation: it needs to tell a
// clean fire ("") and a healthy under_budget no-op apart from a real bail (prefix_mismatch,
// cached_span, …). A configured-off lever returns ("", false) — no compaction attempt was made,
// so there is no fak-side compaction signal to fold into the observation.
//
// turnsLeft is the session's remaining-turns budget (session.Budget.TurnsLeft on the wire:
// SessionBudget.TurnsLeft), the ONLY session-horizon signal --compact-anchor-head's burst
// economics gate (#1407/#1408) can use. Only turnsLeft > 0 (a genuinely bounded remaining-turn
// count from a wired DecideSession) counts as a known horizon; <= 0 — no DecideSession wired, a
// session with 0 turns left, or session.Unbounded (-1) — leaves the gate's TotalTurns unset, so
// an un-budgeted or unbounded session is never guessed at.
//
// trace keys the OBSERVED per-trace idle-gap cold witness (coldMessageSpanCache): a trace that
// idled past the message-breakpoint cache TTL since its last served turn has a provably expired
// message-span suffix, so a head-anchored fire there carries no marginal cache penalty and the
// gate fires horizon-free. "" (no trace — tests, non-session callers) is never cold.

// headSessionPrior resolves the {TotalTurns, CurrentTurn} horizon the head-anchored burst gate
// (agent.CacheBurstPaysBack) consults. Precedence:
//
//  1. A genuine bounded horizon (turnsLeft>0, a wired DecideSession turn budget) wins, mapped to
//     the same {CurrentTurn=1, TotalTurns=1+turnsLeft} adapter as before — the gate still only
//     fires when the remaining turns repay the one-time burst. This is an adapter, not a guess.
//  2. Otherwise, when the assumed-session-length prior is enabled (assumeSessionTurns>0), an
//     un-budgeted session is PRESUMED to run that many turns, and CurrentTurn is the trace's real
//     served-turn depth + 1. The gate then fires EARLY (low CurrentTurn ⇒ many repaying turns
//     left) and refuses near the presumed end — the same break-even economics, just given a
//     history-based length instead of refusing outright. A large invalidated suffix still refuses
//     regardless (break-even exceeds the headroom), so the economics remain the real guard.
//     The presumed length is VOLUME-AWARE (volumeAwareHorizon): a trace whose observed peak resident
//     window marks it heavy keeps a repaying-turn floor deep into its life, because the break-even it
//     must clear is a TOKEN ratio, not a turn count — a flat turn horizon starves exactly the heavy
//     median session that would in fact repay. Inert for the token-light majority.
//  3. With the prior disabled (assumeSessionTurns<=0) it returns (0,0) — the conservative
//     "unknown horizon ⇒ no fire unless zero-penalty (ColdCache)" behavior, byte-for-byte.
//
// The trade in case 2 is sound because the burst penalty is ONE-TIME and bounded while the shed
// saving is per-turn: an early burst on a session that runs long compounds; a session that ends
// early costs only a single bounded cold re-write of the invalidated suffix.
func (s *Server) headSessionPrior(turnsLeft int, trace string) (totalTurns, currentTurn int) {
	if turnsLeft > 0 {
		return 1 + turnsLeft, 1
	}
	if s.assumeSessionTurns <= 0 {
		return 0, 0
	}
	served := int(s.metrics.servedTurnCount(trace))
	heldPeak := s.metrics.heldResidentPeakTokens(trace)
	return volumeAwareHorizon(s.assumeSessionTurns, served, heldPeak), served + 1
}

// volumeAwareHorizon resolves the presumed session length the head-anchored burst gate assumes,
// keyed on the trace's observed heaviness. base is the configured assumeSessionTurns; served is the
// trace's real served-turn depth; heldPeak is its largest observed resident window (input+cached).
//
// A TOKEN-LIGHT trace (heldPeak below headHorizonHeavyResidentFloor) keeps base verbatim — the
// conservative short-session horizon, so the token-light majority (median 7 turns) is unchanged. A
// HEAVY trace instead retains a repaying-turn floor: TotalTurns = max(base, served+headroom), so it
// still has ~headroom turns of break-even room no matter how deep it already is, instead of winding
// down to zero against a short-session constant. The max() makes the boost INERT while served is
// shallow (served+headroom ≤ base ⇒ base wins), so this only ever LENGTHENS the horizon for a
// demonstrably heavy-AND-deep trace — never shortens it, and never touches an early or thin session.
// The economics downstream still gate the fire (a break-even exceeding the granted headroom refuses).
func volumeAwareHorizon(base, served int, heldPeak int64) int {
	if heldPeak < headHorizonHeavyResidentFloor {
		return base
	}
	if floor := served + headHorizonHeavyHeadroom; floor > base {
		return floor
	}
	return base
}

// earlyFiringBudget ramps the head-anchored compaction's effective budget from a floor fraction
// (earlyFireBudgetFloorFrac) of the configured budget at served-turn depth 1 up to the full
// configured budget by earlyFireRampTurns, linearly in the 1-based step. It NEVER returns more
// than the configured budget (the ceiling and its byte-safety guarantees are preserved) and never
// less than 1. A non-positive configured budget or step returns the input unchanged. See the
// early-firing ramp note in gateway.go for the economics: a lower early budget fires sooner and,
// by dropping more of the middle relative to the fixed recent-breakpoint burst, keeps the fire
// profitable — while CacheBurstPaysBack remains the final safety gate.
func earlyFiringBudget(configured, step int) int {
	if configured <= 0 || step <= 0 || step >= earlyFireRampTurns {
		return configured
	}
	frac := earlyFireBudgetFloorFrac + (1-earlyFireBudgetFloorFrac)*float64(step)/float64(earlyFireRampTurns)
	eff := int(float64(configured) * frac)
	if eff < 1 {
		eff = 1
	}
	if eff > configured {
		eff = configured
	}
	return eff
}

func (s *Server) compactAnthropicRawWithReason(req *agent.AnthropicMessagesRequest, turnsLeft int, trace string) (fired bool, reason string) {
	_ = s.headSessionPrior // silence unused in the (impossible) build shape where the head path is compiled out
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) {
		return false, ""
	}
	scalarCfg := s.ScalarConfig()
	compactBudget := scalarCfg.CompactHistoryBudget
	if compactBudget <= 0 {
		s.metrics.observeCompaction(agent.CompactOutcome{}, true) // configured OFF
		return false, ""
	}
	// Offensive half (#806): if the caller left NO cache_control breakpoint, place one on the
	// stable system+tools head so the provider caches it — AND so the compaction below has an
	// anchor to protect (a body with no breakpoint bails CompactReasonNoBreakpoint). Fail-safe
	// identity: a body that already carries a breakpoint (the Claude Code shape), or has no stable
	// head, is returned unchanged. Like compaction this only ADDS a caching hint to the FORWARDED
	// bytes; the decoded req.Messages the kernel adjudicates are untouched, so the trust boundary
	// is unchanged. The net effect is visible on the SAME readback: placement makes compaction
	// fire (CompactionFired) and the provider cache_read it relays now covers the cached head.
	placed, placement := agent.PlaceAnthropicCacheBreakpointWithOutcome(req.Raw)
	req.Raw = placed
	s.metrics.observePlacement(placement)
	if placement.Reason == agent.BreakpointReasonNone {
		// fak placed a breakpoint the caller did NOT send: the provider cache_read this turn is
		// fak-unlocked (a no-breakpoint caller earns 0 provider cache otherwise). Stash it so the
		// per-turn debug render credits it as fak-authored. Keyed by the same trace the render
		// reads (reqTrace); the empty-trace path (tests, non-session callers) is a safe no-op.
		s.recordPlacement(trace)
	}
	opts := agent.CompactOptions{
		Budget:          compactBudget,
		Anchor:          agent.CompactAnchorFirstBP,
		PositiveResidue: s.positiveResidualSubstitution,
	}
	if scalarCfg.CompactAnchorHead != 0 {
		// #1407/#1408 opt-in: re-anchor on the stable head so anchor-starved sessions can
		// shed. headSessionPrior supplies the {TotalTurns, CurrentTurn} pair the burst gate
		// consults: a genuine bounded horizon (turnsLeft>0) wins as before; otherwise, when the
		// assumed-session-length prior is enabled, it hands the gate a presumed length + the
		// trace's real served-turn depth so a warm continuously-active long session fires early
		// and refuses near the presumed end (see headSessionPrior).
		opts.Anchor = agent.CompactAnchorHead
		opts.TotalTurns, opts.CurrentTurn = s.headSessionPrior(turnsLeft, trace)
		// Early-firing budget ramp (the "fak fires by ~step 5-10" seam, gateway.go): on the
		// un-budgeted default `fak guard -- claude` path, where the assumed-session-length prior
		// supplies the horizon (turnsLeft<=0 AND opts.TotalTurns>0), scale the budget the
		// head-anchored fire targets from a floor up to the configured ceiling over the first
		// earlyFireRampTurns, keyed on the trace's real served-turn depth. A lower early budget sheds
		// the un-cacheable middle sooner (when the per-turn saving compounds over the most future
		// turns, and drops more of the middle relative to the fixed recent-breakpoint burst — so the
		// SAME break-even flips from unprofitable to a clean fire). The CacheBurstPaysBack gate still
		// runs on the real horizon, so the ramp only moves the budget line; it can never force an
		// unprofitable burst. It is deliberately NOT applied when a bounded turn horizon is WIRED
		// (turnsLeft>0): that is an explicit operator budget, so its resident window is respected
		// as-is. The cold-only path (TotalTurns<=0) also keeps the full budget — both are unchanged.
		if turnsLeft <= 0 && opts.TotalTurns > 0 {
			step := int(s.metrics.servedTurnCount(trace)) + 1
			opts.Budget = earlyFiringBudget(compactBudget, step)
		}
		// Observed-cold witness: the per-trace idle gap (the harness-coherence wall clock) has
		// passed the message-breakpoint cache TTL, so the suffix a head fire would invalidate is
		// ALREADY expired and re-bills cold this turn regardless — a zero-penalty burst the gate
		// fires horizon-free. This is what lets a plain, un-budgeted `fak guard -- claude` long
		// session finally shed its sprawled middle (the #1407 cold case), without ever guessing:
		// a warm trace still refuses without a repaying horizon.
		opts.ColdCache = s.metrics.coldMessageSpanCache(trace, time.Now(), s.cacheTTL1H.Load())
		// Context-solvency override (the occupancy axis): hand the gate the trace's OBSERVED peak
		// resident window alongside the armed floor, so a trace that has climbed into the danger
		// band sheds even when the burst does not repay. PEAK (the same heldResidentPeak the
		// volume-aware horizon reads) rather than last-turn resident, for two reasons: it is
		// monotone, so a single small turn cannot disarm a session that has proven it holds a
		// large context; and once a trace is that heavy it SHOULD compact every turn — this is a
		// per-request wire rewrite, so the drop is re-applied each turn and the "burst" the gate
		// prices as one-time-per-fire is in steady state amortized across turns that all forward
		// the same compacted shape. Both values must be positive or the override stays disarmed.
		opts.ResidentTokens = int(s.metrics.heldResidentPeakTokens(trace))
		opts.SolvencyFloorTokens = s.compactSolvencyFloorTokens
	}
	// Survival-class gate (#2421): the byte-level compactor runs UNDER the per-page survival
	// contract rather than on its own. compactWithSurvivalClasses is a drop-in for
	// agent.CompactAnthropicHistoryWithOptions — same pair back, so the metric, restore-handle, and
	// fired/reason handling below are unchanged — that first refuses a budget which cannot hold the
	// PINNED floor (PIN_EVICT_REFUSED, body forwarded unchanged), then verifies a fired plan still
	// carries every pinned page byte-identical, retrying against the evictable set only when it does
	// not. A body with no classifiable eviction domain delegates straight through.
	out, outcome := s.compactWithSurvivalClasses(req.Raw, opts, trace)
	req.Raw = out
	s.metrics.observeCompaction(outcome, false)
	// Restore handle: when this fire tombstoned the session's originating task, the outcome carries
	// the dropped turn's bytes and the sha256-hex handle the stub embedded. Stash digest→bytes under
	// this trace so a model resuming past the compaction can call fak_context_restore(id) to page the
	// full task back in (ctxrestore.go). Empty on every non-tombstone fire (the goal-pin path
	// preserves the task verbatim and mints no handle), where stashRestore no-ops.
	if outcome.Reason == agent.CompactReasonNone && outcome.RestoreID != "" {
		s.stashRestore(trace, outcome.RestoreID, outcome.RestoreExcerpt, outcome.RestoreBytes)
	}
	// Positive-residual substitution preserves the complete dropped span behind its own digest,
	// independent of the originating-task tombstone above.
	if outcome.Reason == agent.CompactReasonNone && outcome.ResidueRestoreID != "" {
		s.stashRestore(trace, outcome.ResidueRestoreID, outcome.PositiveResidue, outcome.ResidueRestoreBytes)
	}
	return outcome.Reason == agent.CompactReasonNone, outcome.Reason
}

// maybeAnchorAnthropicRaw is the DEFAULT-ON M2 star-anchor pre-flight gate (#1493): on the
// flagship Anthropic passthrough it APPLIES cachemeta.RecommendLayout (via
// agent.PlaceAnthropicCacheBreakpointWithOutcome) rather than merely reporting it — hoisting
// volatile system blocks behind a byte-stable cacheable anchor and splicing a cache_control
// breakpoint onto the stable head the caller did NOT send — so a no-breakpoint caller earns
// provider prefix caching BY DEFAULT. Unlike the placement inside compactAnthropicRawWithReason
// (gated on compactHistoryBudget>0, so --compact-history-budget=0 took anchoring down with it),
// this gate is DECOUPLED: it fires whenever s.vcacheAnchor is on, independent of the compaction
// budget and the managed-cache TTL lever. Gated on s.vcacheAnchor (--vcache-anchor, default-on);
// OFF is byte-for-byte identity. Fail-safe: any ambiguity — no stable head, or a volatile-only
// head whose hoist would change the model-visible prefix — returns req.Raw UNCHANGED (the honesty
// guard: a semantics-changing hoist is refused, not silently applied). Idempotent with the
// compaction and TTL-upgrade placements: a body that already carries a breakpoint bails
// already_set, so running the anchor first makes the later gates no-op on the same breakpoint.
// The first natural request then WRITES the anchor to the provider cache and later siblings READ
// it. Anthropic passthrough only; inert on every other wire. Returns whether it PLACED this turn.
func (s *Server) maybeAnchorAnthropicRaw(req *agent.AnthropicMessagesRequest, trace string) (fired bool) {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) {
		return false
	}
	if !s.vcacheAnchor {
		return false // configured OFF (--vcache-anchor=false)
	}
	// A fresh measured provider floor changes the wire decision: do not author a
	// cache breakpoint for a request the provider cannot cache. Missing/stale or
	// observation-only calibration is never wired here, preserving defaults.
	if !s.vcacheCalibration.admitsAnchor(req.Model, agent.EstimateAnthropicTokens(req)) {
		return false
	}
	placed, placement := agent.PlaceAnthropicCacheBreakpointWithOutcome(req.Raw)
	req.Raw = placed
	s.metrics.observePlacement(placement)
	if placement.Reason != agent.BreakpointReasonNone {
		return false // refused (already_set / volatile_head / no_stable_head / …): identity, witnessed
	}
	// fak placed a breakpoint the caller did NOT send — this turn's provider cache_read is
	// fak-unlocked. Credit it so the per-turn debug render attributes it as fak-authored (keyed by
	// the same trace; the empty-trace path is a safe no-op).
	s.recordPlacement(trace)
	return true
}

func (s *Server) maybeUpgradeAnthropicCacheTTL1H(req *agent.AnthropicMessagesRequest) bool {
	upgraded, _ := s.maybeUpgradeAnthropicCacheTTL1HScoped(req)
	return upgraded
}

func (s *Server) maybeUpgradeAnthropicCacheTTL1HScoped(req *agent.AnthropicMessagesRequest) (bool, bool) {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) || !s.cacheTTL1H.Load() || !s.vcacheCalibration.wantsExplicitOneHourTTL(req.Model) {
		return false, false
	}
	upgrade := agent.UpgradeAnthropicStableCacheTTL1hWithMessagePrefixes
	// Explicit control arm for #2186: retain the original system/tools-only
	// behavior while leaving the managed-cache posture enabled. This makes a
	// head-only versus message-prefix sweep independent of the lever-off arm.
	if envEnabled("FAK_ABLATE_TTL_1H_HEAD_ONLY") {
		upgrade = agent.UpgradeAnthropicStableCacheTTL1hHeadOnly
	}
	out, outcome := upgrade(req.Raw)
	if outcome.Reason == agent.TTLUpgradeReasonNoStableBreakpoint {
		// #2175: the flagship lever no-ops forever on a caller that sends zero cache_control,
		// because upgrade only edits an EXISTING stable-head breakpoint. Compose place-then-upgrade
		// as one transform so ACTIVE closes the dead end instead of depending on compaction's
		// (compactHistoryBudget-gated) placement to have run first. Byte-safe: placement itself is
		// fail-safe identity on ambiguity, and the upgrade below re-validates the placed bytes.
		placed, placement := agent.PlaceAnthropicCacheBreakpointWithOutcome(req.Raw)
		s.metrics.observePlacement(placement)
		if placement.Reason == agent.BreakpointReasonNone {
			if upgraded, upgradeOutcome := upgrade(placed); upgradeOutcome.Reason == agent.TTLUpgradeReasonNone {
				req.Raw = upgraded
				// New outcome value distinct from "upgraded" (upgrade-only) and "placed" (placement-only
				// from the compaction path), so a sweep can attribute the composed case on its own row.
				s.metrics.observeCacheTTLUpgrade(cacheTTLUpgradePlacedAndUpgraded)
				return true, upgradeOutcome.UpgradedMessageBreakpoints > 0
			}
		}
	}
	// WITNESSED per attempt while the lever is on (--managed-cache / FAK_ABLATE_TTL_1H), so
	// an active-but-never-eligible session is visible on /metrics instead of silent.
	s.metrics.observeCacheTTLUpgrade(outcome.Reason)
	if outcome.Reason != agent.TTLUpgradeReasonNone {
		return false, false
	}
	req.Raw = out
	return true, outcome.UpgradedMessageBreakpoints > 0
}

// maybeElideAnthropicRaw shrinks oversized tool_result bodies in the outbound passthrough body
// when --elide-result-bytes is set. Like maybeCompactAnthropicRaw it is a no-op unless the gateway
// is fronting the REAL Anthropic API (s.anthropicPassthrough) — only there is the raw body
// forwarded verbatim, so only there does rewriting it reach the wire AND need to preserve the
// cached prefix. agent.ElideAnthropicResultsWithOutcome is fail-safe: it returns req.Raw unchanged
// on any ambiguity, never touches a cache_control-bearing message, and proves the protected prefix
// stays byte-identical, so this never breaks a turn or busts the cache. Returns whether it FIRED.
func (s *Server) maybeElideAnthropicRaw(req *agent.AnthropicMessagesRequest) (fired bool) {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) {
		return false
	}
	if s.elideResultBytes <= 0 {
		return false // configured OFF
	}
	out, outcome := agent.ElideAnthropicResultsWithOutcome(req.Raw, s.elideResultBytes)
	req.Raw = out
	s.metrics.observeUncachedTrim(outcome)
	return outcome.Reason == agent.ElideReasonNone
}

// maybeElideStaleReads is the read-lifecycle sibling of maybeElideAnthropicRaw: on the outbound
// passthrough body it replaces a Read tool_result whose file was Edited/Written in a LATER in-session
// turn (a STALE, superseded snapshot) with a compact restore marker, and stashes the full original
// text behind the marker's content-address so fak_context_restore can page it back in. Like the
// oversized-result path it is a no-op unless the gateway fronts the REAL Anthropic API (only there is
// req.Raw forwarded verbatim), and agent.ElideStaleReadsWithOutcome is fail-safe — identity on any
// ambiguity, never touches a cache_control-bearing message, proves the protected prefix stays
// byte-identical. trace is REQUIRED: the stash lands under it so the same-trace fak_context_restore
// resolves the handle. Returns whether it FIRED.
func (s *Server) maybeElideStaleReads(req *agent.AnthropicMessagesRequest, trace string) (fired bool) {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) {
		return false
	}
	if !s.elideStaleReads {
		return false // configured OFF
	}
	out, outcome := agent.ElideStaleReadsWithOutcome(req.Raw)
	req.Raw = out
	s.metrics.observeStaleElide(outcome.Reason, outcome.Elided, outcome.ShedBytes, outcome.ShedTokens)
	for _, r := range outcome.Restores {
		s.stashRestore(trace, r.ID, r.Excerpt, r.Bytes)
	}
	return outcome.Reason == agent.StaleReasonNone
}

// sanitizeAnthropicToolReferences rewrites the Claude Code client's INTERNAL `tool_reference`
// content blocks (emitted inside a ToolSearch/tool-discovery tool_result) into wire-valid `text`
// blocks before req.Raw is forwarded upstream. `tool_reference` is not a valid Anthropic
// tool_result.content block type, so a body carrying one is 400'd as malformed (witnessed:
// session b98cf818). Unlike the cache-preserving shrinkers this is a CORRECTNESS fix, so it runs
// on EVERY wire and needs no cache anchor — agent.SanitizeAnthropicToolReferences is fail-safe
// (returns req.Raw unchanged on any ambiguity, converts rather than drops, and never edits any
// byte outside a tool_reference block). Runs BEFORE the shrinkers so they operate on an already
// well-formed body. Returns whether it FIRED.
func (s *Server) sanitizeAnthropicToolReferences(req *agent.AnthropicMessagesRequest) (fired bool) {
	if req == nil || len(req.Raw) == 0 {
		return false
	}
	out, outcome := agent.SanitizeAnthropicToolReferences(req.Raw)
	req.Raw = out
	s.metrics.observeToolRefSanitize(outcome)
	// General-form empty-content gate (#3118): the residual backstop. The per-type sanitizer above
	// converts tool_reference blocks; this runs on the ALREADY-converted body and backfills any
	// tool_result whose content array is STILL empty (a future client-internal block type not yet
	// special-cased, or a genuinely empty source result) with a placeholder text block — an empty
	// content array is itself a 400 upstream. Same correctness discipline: every wire, no cache
	// anchor, fail-safe identity on any ambiguity.
	repaired, repairOutcome := agent.RepairEmptyToolResultContent(req.Raw)
	req.Raw = repaired
	s.metrics.observeEmptyContentRepair(repairOutcome)
	return outcome.Reason == agent.ToolRefReasonNone || repairOutcome.Reason == agent.EmptyContentReasonNone
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
func (s *Server) maybePlanAnthropicRaw(ctx context.Context, trace string, req *agent.AnthropicMessagesRequest) (bool, agent.CompactOutcome) {
	if req == nil || len(req.Raw) == 0 || !s.anthropicPassthroughFor(req.Model) {
		return false, agent.CompactOutcome{}
	}
	if s.ctxView == nil || !s.ctxView.Enabled {
		return false, agent.CompactOutcome{}
	}
	planned := s.maybePlanMessages(ctx, trace, req.Messages)
	if len(planned) >= len(req.Messages) {
		return false, agent.CompactOutcome{} // the planner did not elide anything — nothing to materialize
	}
	out, outcome := agent.CompactAnthropicHistoryToView(req.Raw, planned)
	if outcome.Reason != agent.CompactReasonNone {
		return false, outcome // bailed — identity (fail-safe)
	}
	req.Raw = out
	s.metrics.observeCtxViewRewrite(outcome)
	return true, outcome
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
			Fak: turn.fakExt(),
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
	resultAdmissions, err := s.admitInboundResults(ctx, req.Messages, req.Tools, reqTrace)
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
	// The dual planner (local model alongside the API upstream) routes on the
	// requested model, so forward it — ONLY there: on the single-planner paths this
	// wire historically never forwarded the client model, and adding it would change
	// the proxy upstream bytes.
	if _, dual := s.planner.(*DualPlanner); dual {
		opts = append(opts, agent.WithModel(req.Model))
	}
	// anthropic→anthropic passthrough: when the upstream IS the real Anthropic API,
	// forward the client's ORIGINAL request bytes verbatim (so its cache_control
	// prefix survives → a real cache hit, not a re-billed prefix) and authenticate
	// with the client's OWN key (transparent hop, no second secret). The kernel still
	// adjudicates the RESPONSE's tool calls below; only the request is byte-faithful.
	// WithRawRequestBody makes the sampling opts above no-ops (the client's own values
	// are already in req.Raw; re-injecting them would change the cached prefix bytes).
	// Per-request: a dual-mode request addressed to the LOCAL model must decode
	// in-kernel, never ride raw bytes upstream.
	if s.anthropicPassthroughFor(req.Model) {
		opts = append(opts, agent.WithRawRequestBody(req.Raw), agent.WithUpstreamAPIKey(upstreamKey), agent.WithUpstreamBeta(upstreamBeta))
		ctx = withDecodedCtxViewSuppressed(ctx)
	}
	comp, err := s.completeServed(ctx, sessionTurn, req.Messages, req.Tools, opts...)
	if err != nil {
		return nil, err
	}

	asst := comp.Message
	asst.Role = agent.RoleAssistant
	kept, adjs, dropped, servedText, servedHits, bodyRefused := s.adjudicateProposedTurn(ctx, asst, reqTrace)
	asst.ToolCalls = kept
	if bodyRefused {
		asst.Content = ""
	}
	// vDSO served-inline (vDSO live in the hot path): a re-proposed read-only call the
	// vDSO already holds fresh is answered LOCALLY and folded into the assistant text
	// (the only wire-valid surface — a tool_result is a user-turn block). The call was
	// dropped from kept, so the client never re-executes it; the engine round-trip is
	// saved. metrics.recordServedInline attributes it to the gateway seam (not the
	// kernel VDSOHits counter, which only the k.Syscall path bumps).
	if servedText != "" {
		if asst.Content != "" {
			asst.Content += "\n" + servedText
		} else {
			asst.Content = servedText
		}
	}
	if servedHits > 0 {
		s.metrics.recordServedInline(servedHits)
	}
	// Stash this turn's SAFETY delta (blocked/repaired calls + quarantined inbound results) for the
	// per-turn fak-turn debug line, the buffered-wire twin of the streaming flushHeldTools call.
	// resultAdmissions came from admitInboundResults on the SAME reqTrace earlier in this turn.
	s.recordTurnSafety(reqTrace, adjs, resultAdmissions)
	// Fold this turn's adjudication SHAPE into separate turn-control signals. Hard deny-all
	// remains the bounded stop-policy path; retryable tool feedback (for example malformed JSON)
	// continues the agent without counting as a session-stop/give-up reason. On a deny-all turn
	// the fingerprint (same tool+reason) is what the guard Stop hook keys its give-up on.
	signal := adjudicationOutcomeForTurn(adjs, len(kept), servedHits)
	denyFP := ""
	if signal == adjudicationOutcomeDenyAll {
		denyFP = denyAllFingerprint(adjs)
	}
	s.recordAdjudicationOutcome(signal, denyFP)

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
	if dropped > 0 || anyRepaired(adjs) || anyLivelock(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			blocks = prependTextBlock(blocks, note)
		}
	}
	// If the result-side floor paged out an inbound tool result, say so in-band too:
	// the model is about to read a quarantine stub where its tool output was, and a
	// silent stub reads as a broken tool. Naming the quarantine lets the agent adapt.
	if note := resultAdmissionNote(freshAdmissionNotes(resultAdmissions)); note != "" {
		blocks = prependTextBlock(blocks, note)
	}
	// If the client reports the known Windows Bash git/gh exit-143 hang, give the
	// model the closed failure token plus the native PowerShell retry command.
	if note := s.toolFailureNoteOnce(reqTrace, req.Messages); note != "" {
		blocks = prependTextBlock(blocks, note)
	}
	// If the session has become unusually expensive (block-tier per-turn as-sent
	// volume) AND the context-expense gate is armed, tell the model once to
	// checkpoint and end the turn (ctxExpenseNoteOnce; "" when the gate is off or
	// the tier/dedup does not fire, so the default path is byte-for-byte unchanged).
	if note := s.ctxExpenseNoteOnce(reqTrace); note != "" {
		blocks = prependTextBlock(blocks, note)
	}
	// Context-pressure PUSH (#2424): the step_advice verdict ctxvalue.go already computes
	// every turn was pull-only at /v1/fak/ctxvalue, so an agent that never asked never heard
	// it — least of all on the turn its window is about to turn over. When the verdict ENTERS
	// checkpoint or rebuild, say so in-band once (ctxAdviceNoteOnce dedups per state entry),
	// rendered from the SAME CtxStepAdvice the HTTP/MCP read returns so the pushed line and
	// the pulled report for one trace cannot disagree. "" for any/bounded/unknown and for a
	// state already reported, so a steady session's response is byte-for-byte unchanged.
	if note := s.ctxAdviceNoteOnce(reqTrace); note != "" {
		blocks = prependTextBlock(blocks, note)
	}
	// Compaction continuation contract (#2422): if this turn crossed a compaction boundary,
	// tell the model what survived, what stays re-derivable, and — the whole point — that the
	// shortened transcript is not a reason to wrap up. takeCompactionContract CONSUMES the
	// boundary record, so the note fires exactly once per boundary; the same record rides the
	// `fak.compaction` extension below for an orchestrator that reads no prose. Prepended LAST
	// so it lands FIRST, ahead of the per-call notes: the model needs to know its history was
	// shed before it reads a verdict about a single tool call.
	compaction := s.takeCompactionContract(reqTrace)
	if note := compactionContractNote(compaction); note != "" {
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
		// The SAME record the in-band note above rendered, carried on the turn so
		// fakExt lands it under `fak.compaction` (#2422). Both channels must describe
		// one boundary: an orchestrator that reads no prose and a model that reads
		// nothing else have to be told the same thing, and taking the contract twice
		// (once per channel) would let the take-once latch serve one and starve the
		// other. nil on a turn that crossed no boundary, so the key stays omitted.
		Compaction: compaction,
	}, nil
}

// anthropicPassthrough reports whether the gateway is fronting the REAL Anthropic
// Messages API — i.e. the live planner is an HTTPPlanner configured for the
// anthropic provider wire (unwrapped from a dual planner's proxy side). Only then is
// byte-exact request passthrough both possible (the inbound and upstream wires match)
// and necessary (to preserve the client's prompt-cache prefix). For the mock, the
// in-kernel model, or any non-Anthropic upstream, this is false and the turn is built
// exactly as before. This is the PLANNER-level answer; a per-request decision must use
// anthropicPassthroughFor, which also excludes dual-mode requests addressed to the
// local model.
func (s *Server) anthropicPassthrough() bool {
	hp := unwrapHTTPPlanner(s.planner)
	return hp != nil && hp.Provider == agent.ProviderAnthropic
}

// anthropicPassthroughFor is the per-REQUEST passthrough decision: like
// anthropicPassthrough, but in dual mode a request addressed to the in-kernel model is
// NOT a passthrough — its bytes must decode locally, never reach the remote API. Every
// req.Raw-mutating optimization and the byte-preserving relay key on this, so a local
// request cleanly falls to the planner path (where the dual planner routes it in-kernel)
// while API-bound requests keep the passthrough byte-for-byte.
func (s *Server) anthropicPassthroughFor(reqModel string) bool {
	if d, ok := s.planner.(*DualPlanner); ok && d.RoutesLocal(reqModel) {
		return false
	}
	return s.anthropicPassthrough()
}

// unwrapHTTPPlanner returns the direct HTTP proxy planner behind p — p itself, or a
// dual planner's proxy side — and nil for every planner with no single live HTTP
// upstream (mock, in-kernel, replica fleet).
func unwrapHTTPPlanner(p agent.Planner) *agent.HTTPPlanner {
	if d, ok := p.(*DualPlanner); ok {
		p = d.Proxy()
	}
	hp, _ := p.(*agent.HTTPPlanner)
	return hp
}

// reasonSecretExfil / reasonSecretDiscovered are the closed-vocabulary result-quarantine
// reason codes for credential-shaped bytes. They are called out specially by the
// retrievability banner because — unlike every other quarantine class — the page-in gate
// re-screens on release and refuses any bytes that still match, so a secret NEVER pages
// back into context. (Mirror of internal/abi.ReasonSecretExfil / ReasonSecretDiscovered,
// duplicated as local literals to avoid an import solely for two banner strings.)
const (
	reasonSecretExfil      = "SECRET_EXFIL"
	reasonSecretDiscovered = "RESULT_SECRET_DISCOVERED"
	// reasonSecretRedacted is the warn-first default: a credential span was MASKED IN
	// PLACE and the rest of the tool result stays in context (internal/normgate). Unlike
	// the seal classes it does NOT hold the result out, so it gets a one-line WARN, not
	// the "held out of context" banner — and it never baits a re-read (there is nothing
	// paged out to retrieve).
	reasonSecretRedacted = "SECRET_REDACTED"
)

// secretRedactedWarn is the one-line warn for masked-in-place credentials. Empty when
// nothing was redacted, so it composes with the held-out banner (which returns "" when
// no result was held).
func secretRedactedWarn(n int) string {
	return redactedSpanWarn(n, "credential-shaped", reasonSecretRedacted,
		"the credential itself", "fail_closed secret posture")
}

// anthropicTurnIdentity stamps the two identity fields every served Anthropic turn
// carries, under one rule for the buffered and streamed surfaces alike: the model is the
// one the client asked for (Anthropic reflects the requested id back) falling back to the
// gateway's configured model when the client omitted it, and the message id is minted
// fresh per turn from the wall clock under the msg_fak_ prefix that marks a fak-served
// message. Both surfaces must agree here — a client that pipelines a streamed and a
// buffered turn would otherwise see two different models named for the same request.
func (s *Server) anthropicTurnIdentity(reqModel string) (model, id string) {
	model = reqModel
	if model == "" {
		model = s.model
	}
	return model, "msg_fak_" + itoa(uint64(time.Now().UnixNano()))
}

func (s *Server) streamAnthropicPending(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string, sessionTurn servedSessionTurn, upstreamKey, upstreamBeta string, compacted, contextEvent bool, hcoh harnessCoherenceInputs) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		began := time.Now()
		turn, err := s.completeAnthropicTurn(r.Context(), req, reqTrace, sessionTurn, "", "", upstreamKey, upstreamBeta)
		if err != nil {
			s.renderTurnDebugError(reqTrace, "anthropic_messages", err, time.Since(began))
			s.logf("gateway: upstream model error (messages): %v", err)
			s.writeUpstreamErr(w, err)
			return
		}
		if compacted {
			s.metrics.recordCompactionCacheRead(turn.Usage.CacheReadInputTokens)
			s.observeResetHealth(reqTrace, turn.Usage.InputTokens, turn.Usage.CacheReadInputTokens, turn.Usage.CacheCreationInputTokens)
		}
		s.observeHarnessCoherenceAndArm(reqTrace, time.Now(), hcoh.inboundPrefixDigest, compacted, hcoh.fakBail,
			false /*fakWorldBreak*/, false /*sealed*/, int64(turn.Usage.CacheReadInputTokens), int64(turn.Usage.CacheCreationInputTokens), int64(turn.Usage.InputTokens))
		s.logInferenceTurnWithContextEvent(reqTrace, "anthropic_messages", true, agent.Usage{
			PromptTokens:             turn.Usage.InputTokens,
			CompletionTokens:         turn.Usage.OutputTokens,
			CacheReadInputTokens:     turn.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: turn.Usage.CacheCreationInputTokens,
		}, turn.Stop, time.Since(began), compacted, contextEvent)
		writeJSON(w, http.StatusOK, anthropicMessageResponse{
			ID: turn.ID, Type: "message", Role: "assistant", Model: turn.Model,
			Content: turn.Blocks, StopReason: turn.Stop, StopSequence: nil, Usage: turn.Usage,
			Fak: turn.fakExt(),
		})
		return
	}
	model, id := s.anthropicTurnIdentity(req.Model)
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
			s.observeHarnessCoherenceAndArm(reqTrace, time.Now(), hcoh.inboundPrefixDigest, compacted, hcoh.fakBail,
				false /*fakWorldBreak*/, false /*sealed*/, int64(res.turn.Usage.CacheReadInputTokens), int64(res.turn.Usage.CacheCreationInputTokens), int64(res.turn.Usage.InputTokens))
			s.logInferenceTurnWithContextEvent(reqTrace, "anthropic_messages", true, agent.Usage{
				PromptTokens:             res.turn.Usage.InputTokens,
				CompletionTokens:         res.turn.Usage.OutputTokens,
				CacheReadInputTokens:     res.turn.Usage.CacheReadInputTokens,
				CacheCreationInputTokens: res.turn.Usage.CacheCreationInputTokens,
			}, res.turn.Stop, time.Since(began), compacted, contextEvent)
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
	sendAnthropicTerminal(send, stop, usage)
}

// handleAnthropicCountTokens answers POST /v1/messages/count_tokens with a cheap,
// tokenizer-free estimate. Claude Code treats this as optional (a 404 is fine), but
// answering it keeps its context-management heuristics from flying blind.
func (s *Server) handleAnthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	var raw json.RawMessage
	if !decodeRequestBody(w, r, &raw) {
		return
	}
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"input_tokens": agent.EstimateAnthropicTokens(req)})
}
