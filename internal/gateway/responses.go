package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// responses.go is the inbound OpenAI **Responses API** wire (`POST /v1/responses`)
// — the third client-facing chat surface, alongside /v1/chat/completions and the
// Anthropic /v1/messages. It exists so a Responses-API-native agent (the OpenAI
// Codex CLI, the Terminal-Bench `terminus` agent) can repoint its OpenAI base URL at
// fak and have EVERY proposed tool call cross the kernel's capability floor — the
// same default-deny adjudication the chat wire already runs. Without it a Responses
// client 404s on the gateway and fak cannot sit in its tool loop at all (#925).
//
// This file is SHAPE TRANSLATION only. It maps the Responses request onto the
// gateway's internal agent.Message/agent.ToolDef vocabulary, runs the IDENTICAL
// served-turn core every other wire uses (beginServedSessionTurn -> admitInboundResults
// -> completeServed -> adjudicateProposed), then renders the kept assistant turn back
// into the Responses output-item shape. The verdict pass is reused verbatim; no new
// trust decision lives here. The authoritative field layout is the OUTBOUND Responses
// adapter in internal/agent/adapters.go (openAIResponsesItem/Tool/Response); these
// gateway-local DTOs are its inbound mirror, exactly as ChatRequest mirrors the chat
// adapter.
//
// Streaming is SYNTHESIZED from the buffered turn: the gateway adjudicates the
// complete proposed-tool-call set, then re-serializes a well-formed SSE stream
// (response.created → response.output_item.added → response.output_item.done →
// response.done). This matches the non-tool-path behavior of the chat wire, where
// the kernel's adjudication invariant forces buffering before any byte hits the wire.

// ResponsesRequest is the inbound POST /v1/responses body (the minimal faithful
// subset of the OpenAI Responses API). Input is raw because the Responses wire
// allows EITHER a bare string OR an array of typed input items; decodeResponsesInput
// folds both into the gateway's []agent.Message — the same union trick normalizeStop
// uses for the chat `stop` field. Unknown top-level fields (text/structured-output,
// store, reasoning, metadata, previous_response_id) are accepted and ignored for
// drop-in compatibility; there is, by construction, no Ref field to smuggle.
type ResponsesRequest struct {
	Model           string          `json:"model"`
	Input           json.RawMessage `json:"input"`
	Instructions    string          `json:"instructions,omitempty"`
	Tools           []responsesTool `json:"tools,omitempty"`
	MaxOutputTokens int             `json:"max_output_tokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	Stream          bool            `json:"stream,omitempty"`
	// PreviousResponseID is accepted and ignored: fak does not (yet) persist a
	// server-side response store, so a client threading conversation state must send
	// the full input each turn (the same posture the chat wire has — it is stateless).
	PreviousResponseID string `json:"previous_response_id,omitempty"`
}

// responsesTool is the inbound Responses function-tool declaration. Unlike the chat
// wire's nested {type:"function", function:{name,...}}, the Responses wire FLATTENS
// the function fields to the top level: {type:"function", name, description,
// parameters}. responsesToolsToToolDefs maps it onto agent.ToolDef.
type responsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// responsesInputItem is one element of an `input` array. The Responses wire is a
// tagged union over `type`: a `message` carries a role + content parts; a
// `function_call` is an assistant tool call the client is echoing back; a
// `function_call_output` is a tool RESULT the client executed (the bytes the
// result-side floor must screen). Fields not relevant to a given type are absent.
type responsesInputItem struct {
	Type    string          `json:"type"`
	Role    string          `json:"role,omitempty"`
	Content json.RawMessage `json:"content,omitempty"`
	// function_call fields:
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	ID        string `json:"id,omitempty"`
	// function_call_output field:
	Output string `json:"output,omitempty"`
}

// responsesResponse is the outbound POST /v1/responses body. The `fak` extension
// carries the per-tool-call adjudications for a fak-aware client, exactly as the
// chat/messages wires expose it; a fak-unaware Responses client simply never sees a
// denied call (it is absent from `output`). OutputText is the convenience flattened
// assistant text the Responses SDK surfaces as response.output_text.
type responsesResponse struct {
	ID                string                `json:"id"`
	Object            string                `json:"object"`
	CreatedAt         int64                 `json:"created_at"`
	Model             string                `json:"model"`
	Status            string                `json:"status"`
	IncompleteDetails *responsesIncomplete  `json:"incomplete_details,omitempty"`
	Output            []responsesOutputItem `json:"output"`
	OutputText        string                `json:"output_text,omitempty"`
	Usage             responsesUsage        `json:"usage"`
	Fak               *FakExt               `json:"fak,omitempty"`
}

type responsesIncomplete struct {
	Reason string `json:"reason"`
}

// responsesOutputItem is one element of the `output` array: a `message` item
// (assistant prose, content = output_text parts) or a `function_call` item (one
// KEPT tool call, carrying call_id so the client matches its next
// function_call_output). The two shapes share a struct; only the fields for the
// active type are populated (the rest are omitempty).
type responsesOutputItem struct {
	Type   string `json:"type"`
	ID     string `json:"id,omitempty"`
	Status string `json:"status,omitempty"`
	// message fields:
	Role    string                 `json:"role,omitempty"`
	Content []responsesContentPart `json:"content,omitempty"`
	// function_call fields:
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type responsesContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// responsesUsage is the Responses-shaped token accounting (input/output/total),
// projected from the gateway's internal agent.Usage.
type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// handleResponses serves POST /v1/responses. Its spine is handleChatCompletions
// step-for-step over the same served-turn core; only the request decode and the
// response render differ (the Responses shape vs the chat shape).
func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req ResponsesRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	messages, err := decodeResponsesInput(req.Input, req.Instructions)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "input: "+err.Error())
		return
	}
	// An empty/missing input is a CLIENT error, mirroring the chat wire's
	// empty-messages floor — reject here rather than spending an upstream round-trip
	// on a degenerate request.
	if len(messages) == 0 {
		writeErr(w, http.StatusBadRequest, "input: field required")
		return
	}
	if msg := validateResponsesSampling(req); msg != "" {
		writeErr(w, http.StatusBadRequest, msg)
		return
	}
	tools := responsesToolsToToolDefs(req.Tools)

	ctx := r.Context()
	reqModel := req.Model
	if reqModel == "" {
		reqModel = s.model
	}

	reqTrace := s.useHTTPTrace(w, r, "")
	sessionTurn, ok, canceled := s.beginServedSessionTurn(ctx, reqTrace)
	if canceled {
		return
	}
	if !ok {
		if newTrace, seed, reset := s.maybeResetOnBudget(ctx, sessionTurn.state, messages); reset {
			messages = spliceSeed(seed, messages)
			reqTrace = newTrace
			if sessionTurn, ok, canceled = s.beginServedSessionTurn(ctx, reqTrace); canceled {
				return
			}
			if !ok {
				writeSessionRefusal(w, sessionTurn.state)
				return
			}
		} else {
			writeSessionRefusal(w, sessionTurn.state)
			return
		}
	}

	resultAdmissions, err := s.admitInboundResults(ctx, messages, reqTrace)
	if err != nil {
		writeErr(w, http.StatusBadGateway, "upstream cache invalidation failed")
		return
	}

	began := time.Now()
	comp, err := s.completeServed(ctx, sessionTurn, messages, tools,
		agent.WithModel(req.Model),
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxOutputTokens)),
		agent.WithTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
	)
	if err != nil {
		s.logf("gateway: upstream model error: %v", err)
		status, code, msg := s.plannerErrorStatus(err)
		writeErrCode(w, status, code, msg)
		return
	}

	asst := comp.Message
	asst.Role = agent.RoleAssistant

	// Tool-call conformance fail-closed (mirrors handleChatCompletions): the upstream
	// announced tool calls but none survived parsing — refusing here is the only way
	// to keep an unparsed call from crossing the gateway WITHOUT adjudication.
	if comp.ToolCallsDropped && len(asst.ToolCalls) == 0 {
		s.logf("gateway: upstream announced tool_calls but none parsed (conformance fail-closed); model=%s", s.model)
		writeErr(w, http.StatusBadGateway, "upstream tool-call format not recognized; refusing to skip adjudication")
		return
	}

	kept, adjs, dropped := s.adjudicateProposed(ctx, asst.ToolCalls, reqTrace)
	asst.ToolCalls = kept

	// finishReason drives both the logged turn classification and the Responses
	// status: a length-truncated turn is `incomplete`, everything else `completed`.
	finish := comp.FinishReason
	if len(kept) > 0 {
		finish = "tool_calls"
	} else if dropped > 0 {
		// Every proposed call was refused: give even a fak-unaware client an
		// actionable in-band message instead of an empty turn (mirrors the chat wire).
		finish = "stop"
		if asst.Content == "" {
			asst.Content = denySummary(adjs)
		}
	}

	respModel := comp.Model
	if respModel == "" {
		respModel = reqModel
	}
	s.logInferenceTurn(reqTrace, "openai_responses", false, comp.Usage, finish, time.Since(began), false)

	resp := responsesResponse{
		ID:         "resp_fak_" + itoa(uint64(time.Now().UnixNano())),
		Object:     "response",
		CreatedAt:  time.Now().Unix(),
		Model:      respModel,
		Status:     responsesStatusFor(comp.FinishReason),
		Output:     responsesOutputFromAssistant(asst),
		OutputText: asst.Content,
		Usage: responsesUsage{
			InputTokens:  comp.Usage.PromptTokens,
			OutputTokens: comp.Usage.CompletionTokens,
			TotalTokens:  comp.Usage.TotalTokens,
		},
	}
	if comp.FinishReason == "length" {
		resp.IncompleteDetails = &responsesIncomplete{Reason: "max_output_tokens"}
	}
	if len(adjs) > 0 || len(resultAdmissions) > 0 {
		resp.Fak = &FakExt{Adjudications: adjs, ResultAdmissions: resultAdmissions}
	}
	if req.Stream {
		s.writeResponsesStream(w, resp)
	} else {
		writeJSON(w, http.StatusOK, resp)
	}
}

// decodeResponsesInput folds the Responses `input` (a bare string OR an array of
// typed input items) plus the top-level `instructions` into the gateway's
// []agent.Message. A bare string is one user message. An array is walked item by
// item: a `message` becomes a role-tagged message with its content parts flattened;
// a `function_call` is folded into an assistant message's ToolCalls; a
// `function_call_output` becomes a RoleTool message keyed by call_id (so
// admitInboundResults screens it like any other inbound tool result); an unknown
// type is skipped (drop-in tolerance). `instructions`, when present, is prepended as
// a leading RoleSystem message — the Responses analogue of the chat system turn.
func decodeResponsesInput(raw json.RawMessage, instructions string) ([]agent.Message, error) {
	var msgs []agent.Message
	if instructions != "" {
		msgs = append(msgs, agent.Message{Role: agent.RoleSystem, Content: instructions})
	}

	b := trimLeadingWS(raw)
	if len(b) == 0 {
		return msgs, nil
	}
	switch b[0] {
	case '"':
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		if s != "" {
			msgs = append(msgs, agent.Message{Role: agent.RoleUser, Content: s})
		}
		return msgs, nil
	case '[':
		var items []responsesInputItem
		if err := json.Unmarshal(raw, &items); err != nil {
			return nil, err
		}
		for _, it := range items {
			switch it.Type {
			case "message", "": // a bare item with a role+content is a message
				if it.Role == "" {
					continue
				}
				msgs = append(msgs, agent.Message{
					Role:    responsesRole(it.Role),
					Content: responsesContentText(it.Content),
				})
			case "function_call":
				// An assistant tool call the client is echoing back into context.
				id := it.CallID
				if id == "" {
					id = it.ID
				}
				msgs = append(msgs, agent.Message{
					Role: agent.RoleAssistant,
					ToolCalls: []agent.ToolCall{{
						ID:       id,
						Type:     "function",
						Function: agent.Func{Name: it.Name, Arguments: it.Arguments},
					}},
				})
			case "function_call_output":
				// A tool RESULT the client executed — the bytes the result-side floor
				// screens (poison quarantine, secret redaction) before the model sees them.
				msgs = append(msgs, agent.Message{
					Role:       agent.RoleTool,
					ToolCallID: it.CallID,
					Content:    it.Output,
				})
			default:
				// Unknown item type (reasoning, image, etc.) — skip rather than 400, so a
				// richer client degrades to a text+tools turn instead of being rejected.
			}
		}
		return msgs, nil
	default:
		return nil, errInvalidInput
	}
}

// errInvalidInput is the 400 cause when `input` is neither a string nor an array.
var errInvalidInput = errInput("input must be a string or an array of input items")

type errInput string

func (e errInput) Error() string { return string(e) }

// responsesContentText flattens a Responses message item's `content` (a bare string
// OR an array of typed parts: input_text / output_text / text) to a single string.
// The chat wire's agent.contentPartText does NOT recognize the Responses-specific
// `input_text`/`output_text` part types, so this wire needs its own part flattener
// or user/assistant content silently drops.
func responsesContentText(raw json.RawMessage) string {
	b := trimLeadingWS(raw)
	if len(b) == 0 {
		return ""
	}
	if b[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			return s
		}
		return ""
	}
	if b[0] != '[' {
		return ""
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}
	out := make([]byte, 0, 64)
	for _, p := range parts {
		// input_text / output_text / text all carry the human-readable text in `text`.
		if p.Text == "" {
			continue
		}
		if len(out) > 0 {
			out = append(out, '\n')
		}
		out = append(out, p.Text...)
	}
	return string(out)
}

// responsesRole maps a Responses message role to the gateway's internal role. The
// Responses wire uses developer/system/user/assistant; `developer` is the Responses
// rename of the system instruction channel, so it folds to RoleSystem.
func responsesRole(role string) string {
	switch role {
	case "system", "developer":
		return agent.RoleSystem
	case "assistant":
		return agent.RoleAssistant
	case "tool":
		return agent.RoleTool
	default:
		return agent.RoleUser
	}
}

// responsesToolsToToolDefs maps the flat Responses function-tool shape onto the
// gateway's agent.ToolDef (the nested chat shape the planner consumes). A
// non-function tool type (web_search, file_search, computer_use, ...) is skipped
// rather than 400'd: fak adjudicates the FUNCTION tool calls; a built-in tool the
// upstream resolves itself is not a kernel-mediated call and carries no args to gate.
func responsesToolsToToolDefs(tools []responsesTool) []agent.ToolDef {
	if len(tools) == 0 {
		return nil
	}
	out := make([]agent.ToolDef, 0, len(tools))
	for _, t := range tools {
		if t.Type != "function" || t.Name == "" {
			continue
		}
		out = append(out, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}
	return out
}

// responsesOutputFromAssistant renders the adjudicated assistant turn into Responses
// output items: a `message` item carrying the assistant prose as an output_text part
// (emitted when there is any content), followed by one `function_call` item per KEPT
// tool call. A TRANSFORM-repaired call's Arguments already carry the kernel's
// canonical bytes (adjudicateProposed rewrote them in place), so the client runs the
// repaired form. call_id is the tool call's id so the client can match its next
// function_call_output to it.
func responsesOutputFromAssistant(asst agent.Message) []responsesOutputItem {
	out := make([]responsesOutputItem, 0, 1+len(asst.ToolCalls))
	if asst.Content != "" {
		out = append(out, responsesOutputItem{
			Type:   "message",
			Role:   agent.RoleAssistant,
			Status: "completed",
			Content: []responsesContentPart{{
				Type: "output_text",
				Text: asst.Content,
			}},
		})
	}
	for _, tc := range asst.ToolCalls {
		args := tc.Function.Arguments
		if args == "" {
			args = "{}"
		}
		out = append(out, responsesOutputItem{
			Type:      "function_call",
			ID:        tc.ID,
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Arguments: args,
			Status:    "completed",
		})
	}
	return out
}

// responsesStatusFor maps the planner finish reason to a Responses status. A turn
// truncated by the output-token ceiling is `incomplete`; everything else (a normal
// stop, an end-of-turn, a tool-call turn) is `completed`.
func responsesStatusFor(finishReason string) string {
	if finishReason == "length" {
		return "incomplete"
	}
	return "completed"
}

// writeResponsesStream synthesizes a well-formed Responses SSE stream from a
// buffered response, matching the request's stream flag. It emits the sequence:
// response.created → response.output_item.added (per item) → response.output_item.done
// (per item) → response.done. This is the synthesized-stream analogue of
// writeChatCompletionStream: the gateway buffers the entire turn, adjudicates the
// proposed tool calls, then re-serializes the adjudicated turn as SSE.
func (s *Server) writeResponsesStream(w http.ResponseWriter, resp responsesResponse) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	// response.created: initial event with response metadata
	type responseCreatedData struct {
		ID        string         `json:"id"`
		Object    string         `json:"object"`
		CreatedAt int64          `json:"created_at"`
		Model     string         `json:"model"`
		Status    string         `json:"status"`
		Usage     responsesUsage `json:"usage"`
	}
	_ = writeSSEEvent(w, "response.created", responseCreatedData{
		ID:        resp.ID,
		Object:    resp.Object,
		CreatedAt: resp.CreatedAt,
		Model:     resp.Model,
		Status:    resp.Status,
		Usage:     resp.Usage,
	})

	// Emit each output item: added → done
	for _, item := range resp.Output {
		// response.output_item.added
		type outputItemAdded struct {
			Index int                 `json:"index"`
			Item  responsesOutputItem `json:"item"`
		}
		_ = writeSSEEvent(w, "response.output_item.added", outputItemAdded{Index: len(resp.Output), Item: item})

		// response.output_item.done
		type outputItemDone struct {
			Index int    `json:"index"`
			Item  string `json:"item"`
		}
		_ = writeSSEEvent(w, "response.output_item.done", outputItemDone{Index: len(resp.Output), Item: "done"})
	}

	// response.done: terminal event with optional fak extension and incomplete details
	type responseDoneData struct {
		ID                string                `json:"id"`
		Object            string                `json:"object"`
		CreatedAt         int64                 `json:"created_at"`
		Model             string                `json:"model"`
		Status            string                `json:"status"`
		Output            []responsesOutputItem `json:"output"`
		OutputText        string                `json:"output_text,omitempty"`
		Usage             responsesUsage        `json:"usage"`
		Fak               *FakExt               `json:"fak,omitempty"`
		IncompleteDetails *responsesIncomplete  `json:"incomplete_details,omitempty"`
	}
	_ = writeSSEEvent(w, "response.done", responseDoneData{
		ID:                resp.ID,
		Object:            resp.Object,
		CreatedAt:         resp.CreatedAt,
		Model:             resp.Model,
		Status:            resp.Status,
		Output:            resp.Output,
		OutputText:        resp.OutputText,
		Usage:             resp.Usage,
		Fak:               resp.Fak,
		IncompleteDetails: resp.IncompleteDetails,
	})

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// writeSSEEvent writes a single SSE event with a typed event name. The Responses
// wire uses named event types (response.created, response.done) rather than the
// generic data: frames the chat wire uses.
func writeSSEEvent(w http.ResponseWriter, event string, data interface{}) error {
	raw, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(w, "event: %s\n", event)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", raw)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

// validateResponsesSampling enforces the Responses sampling-param contract on
// ingress, mirroring validateSampling on the chat wire but naming the Responses
// field (`max_output_tokens`). It returns the first invalid field's 400 message, or
// "" when every present field is in range. As on the chat wire, 0 is NOT rejected
// (an omitted omitempty int and an explicit 0 are indistinguishable and both fall
// through to the planner default); only impossible values (negatives, out-of-band
// floats) are caught.
func validateResponsesSampling(req ResponsesRequest) string {
	if req.MaxOutputTokens < 0 {
		return "max_output_tokens: must be a positive integer"
	}
	return validateSamplingRanges(req.Temperature, req.TopP)
}

// trimLeadingWS returns raw with leading JSON whitespace stripped, so a caller can
// branch on the first significant byte (the string|array union discriminator). It
// mirrors the inline whitespace skip in rawArgs/normalizeStop.
func trimLeadingWS(raw json.RawMessage) []byte {
	b := []byte(raw)
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	return b[i:]
}
