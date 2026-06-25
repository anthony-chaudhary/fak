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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
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
	// Operator control: refuse a paused/draining/stopped session's next request (the
	// proxy-path enforcement of /v1/fak/session). Fail-open when the route is disabled.
	if ok, st := s.sessionAdmits(ctx, reqTrace); !ok {
		writeSessionRefusal(w, st)
		return
	}
	// In passthrough mode the upstream credential is the client's own (transparent
	// hop) UNLESS the gateway pins its own (the subscription path). The inbound
	// anthropic-beta is forwarded so the client's negotiated betas survive the hop.
	// Both extracted here, on the HTTP boundary, since the planner layer never sees
	// the request headers.
	upstreamKey := s.anthropicUpstreamCredential(r)
	upstreamBeta := r.Header.Get("anthropic-beta")

	if req.Stream {
		// When fronting the REAL Anthropic API, relay a TRUE live token stream so the
		// client's first token tracks the model (and the prompt-cache hit's fast prefill
		// is felt, not buffered away). It returns false only if the upstream stream never
		// opened and nothing was written — then fall back to the buffered synth path,
		// which is also the path for a local/mock upstream that cannot stream this wire.
		if s.anthropicPassthrough() && s.streamAnthropicPassthroughLive(w, r, req, reqTrace, upstreamKey, upstreamBeta) {
			return
		}
		// For non-Anthropic upstreams that still support the generic planner streaming
		// seam (OpenAI-compatible/vLLM/SGLang), translate live content callbacks into
		// Anthropic text_delta events while holding every proposed tool call until the
		// same whole-turn adjudication gate below can run. A false return means either
		// the planner cannot stream or the writer cannot flush, so the existing
		// ping-then-synthesize fallback remains the behavior.
		if s.streamAnthropicPlannerLive(w, r, req, reqTrace) {
			return
		}
		s.streamAnthropicPending(w, r, req, reqTrace, upstreamKey, upstreamBeta)
		return
	}

	turn, err := s.completeAnthropicTurn(ctx, req, reqTrace, "", "", upstreamKey, upstreamBeta)
	if err != nil {
		// Mirror the chat-completions posture: an upstream model failure is a gateway
		// error, and the raw provider body must not cross the trust boundary.
		s.logf("gateway: upstream model error (messages): %v", err)
		writeErr(w, http.StatusBadGateway, "upstream model error")
		return
	}

	writeJSON(w, http.StatusOK, anthropicMessageResponse{
		ID: turn.ID, Type: "message", Role: "assistant", Model: turn.Model,
		Content: turn.Blocks, StopReason: turn.Stop, StopSequence: nil, Usage: turn.Usage,
		Fak: fakExtFrom(turn.Adjs, turn.ResultAdmissions),
	})
}

func (s *Server) completeAnthropicTurn(ctx context.Context, req *agent.AnthropicMessagesRequest, reqTrace, id, model, upstreamKey, upstreamBeta string) (*anthropicTurn, error) {
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
		agent.WithMaxTokens(req.MaxTokens),
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
	comp, err := s.complete(ctx, req.Messages, req.Tools, opts...)
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
		note := adjudicationNote(adjs)
		if note != "" {
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

// streamAnthropic synthesizes the Messages SSE event sequence from a finished,
// already-adjudicated turn: message_start, then a content_block_start /
// (text|input_json)_delta / content_block_stop triple per block, then a
// message_delta carrying the stop_reason + output token count, then message_stop.
// Each event is flushed immediately so the client sees a live stream.
func (s *Server) streamAnthropic(w http.ResponseWriter, id, model string, blocks []agent.AnthropicBlockOut, stop string, usage anthropicUsage) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		// No streaming support on this writer: degrade to a single buffered body
		// rather than hang the client.
		writeJSON(w, http.StatusOK, anthropicMessageResponse{
			ID: id, Type: "message", Role: "assistant", Model: model,
			Content: blocks, StopReason: stop, StopSequence: nil, Usage: usage,
		})
		return
	}
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
			"usage": map[string]int{"input_tokens": usage.InputTokens, "output_tokens": 0},
		},
	})

	streamAnthropicBlocks(send, blocks, stop, usage)
}

func (s *Server) streamAnthropicPending(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace, upstreamKey, upstreamBeta string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		turn, err := s.completeAnthropicTurn(r.Context(), req, reqTrace, "", "", upstreamKey, upstreamBeta)
		if err != nil {
			s.logf("gateway: upstream model error (messages): %v", err)
			writeErr(w, http.StatusBadGateway, "upstream model error")
			return
		}
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
	go func() {
		turn, err := s.completeAnthropicTurn(r.Context(), req, reqTrace, id, model, upstreamKey, upstreamBeta)
		done <- turnResult{turn: turn, err: err}
	}()

	ticker := time.NewTicker(anthropicStreamPingInterval)
	defer ticker.Stop()
	for {
		select {
		case res := <-done:
			if res.err != nil {
				s.logf("gateway: upstream model error (messages): %v", res.err)
				send("error", map[string]any{
					"type":  "error",
					"error": map[string]any{"type": "api_error", "message": "upstream model error"},
				})
				return
			}
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
