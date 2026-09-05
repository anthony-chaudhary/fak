package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
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
// response.completed). This matches the non-tool-path behavior of the chat wire, where
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
	raw         json.RawMessage
}

func (t *responsesTool) UnmarshalJSON(data []byte) error {
	type wireTool responsesTool
	if err := json.Unmarshal(data, (*wireTool)(t)); err != nil {
		return err
	}
	t.raw = append(t.raw[:0], data...)
	return nil
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
	Namespace string `json:"namespace,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	ID        string `json:"id,omitempty"`
	// function_call_output field:
	Output json.RawMessage `json:"output,omitempty"`
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
	FinishReason      string                `json:"finish_reason,omitempty"`
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
	Namespace string `json:"namespace,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type responsesContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// responsesUsage is the Responses-shaped token accounting projected from the gateway's
// internal agent.Usage. Alongside the input/output/total triple it forwards
// input_tokens_details.cached_tokens so a Codex client records the UPSTREAM provider's
// real prompt-cache reuse instead of the counter being silently dropped to zero (#4776).
// Fields the inbound Responses adapter never parses are DELIBERATELY unforwarded and
// documented at responsesUsageFrom, not silently synthesized here.
type responsesUsage struct {
	InputTokens         int                           `json:"input_tokens"`
	OutputTokens        int                           `json:"output_tokens"`
	TotalTokens         int                           `json:"total_tokens"`
	InputTokensDetails  *responsesInputTokensDetails  `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *responsesOutputTokensDetails `json:"output_tokens_details,omitempty"`
}

// responsesInputTokensDetails is the Responses `usage.input_tokens_details` subobject.
// cached_tokens is the number of input tokens the UPSTREAM provider served from its own
// prompt cache — PROVIDER cache-reuse provenance relayed verbatim from the upstream
// Responses `usage.input_tokens_details.cached_tokens` (which the outbound Responses
// adapter parsed into agent.Usage.PromptTokensDetails). It is deliberately NOT fak's own
// vDSO/served-inline reuse: a fak-served hit is a separate axis and is never relabeled as
// cached_tokens (#4776). cached_tokens carries NO omitempty so a WITNESSED zero (the
// provider reported a counter whose value is 0) renders `"cached_tokens":0`, distinct from
// an OMITTED detail — responsesUsageFrom drops the whole subobject when the provider
// supplied no counter, so a consumer reads "unknown", never a fabricated measured zero.
type responsesInputTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// responsesOutputTokensDetails is the Responses `usage.output_tokens_details` subobject.
// reasoning_tokens is the number of completion tokens generated by a reasoning model
// as chain-of-thought (relayed verbatim from agent.Usage.CompletionTokensDetails).
type responsesOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// responsesUsageFrom projects the gateway's internal agent.Usage onto the client-facing
// Responses usage shape. It forwards input/output/total verbatim and — the #4776 fix —
// preserves input_tokens_details.cached_tokens when (and only when) the upstream provider
// actually supplied that counter.
//
// Omitted-vs-witnessed-zero is carried through the pointer. agent.Usage holds the
// provider's Responses input_tokens_details in PromptTokensDetails (the outbound Responses
// adapter folds input_tokens_details there, falling back to prompt_tokens_details). A nil
// pointer means the provider reported no cache counter — a local/in-kernel turn, or an
// upstream that omitted the field — so the whole input_tokens_details subobject is dropped.
// A non-nil pointer (including CachedTokens==0, a witnessed zero) is forwarded so the zero
// stays distinguishable from silence.
//
// Provenance stays clean: only the provider-relayed counter is forwarded here; fak's own
// vDSO/served-inline reuse is never relabeled as cached_tokens.
//
// OutputTokensDetails forwards the Responses `usage.output_tokens_details.reasoning_tokens`
// subobject when the upstream provider reported reasoning completion tokens (parsed into
// agent.Usage.CompletionTokensDetails).
func responsesUsageFrom(u agent.Usage) responsesUsage {
	ru := responsesUsage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
	}
	// PromptTokensDetails is where the outbound Responses adapter lands the wire's
	// input_tokens_details; InputTokensDetails is the same-shaped fallback. Nil in both
	// means "no provider counter" — leave input_tokens_details omitted.
	if d := u.PromptTokensDetails; d != nil {
		ru.InputTokensDetails = &responsesInputTokensDetails{CachedTokens: d.CachedTokens}
	} else if d := u.InputTokensDetails; d != nil {
		ru.InputTokensDetails = &responsesInputTokensDetails{CachedTokens: d.CachedTokens}
	}
	if u.CompletionTokensDetails != nil && u.CompletionTokensDetails.ReasoningTokens > 0 {
		ru.OutputTokensDetails = &responsesOutputTokensDetails{
			ReasoningTokens: u.CompletionTokensDetails.ReasoningTokens,
		}
	}
	return ru
}

// handleResponses serves POST /v1/responses. Its spine is handleChatCompletions
// step-for-step over the same served-turn core; only the request decode and the
// response render differ (the Responses shape vs the chat shape).
func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	if s.checkWarmupPending(w) {
		return
	}
	// Release the EP follower ranks BEFORE this rank enters the decode, onto THIS wire's
	// own route (#5528). Same placement and same reasoning as the chat, legacy and
	// Anthropic wires: after the method check, before anything reads the body.
	//
	// ONE release covers the whole HTTP turn, including the #5212 denial-recovery sample
	// below. That second completeServed is a second forward pass, but it is not a second
	// inbound request: the follower rank serves the SAME mirrored body through THIS SAME
	// handler, so it reaches the same denial-only verdict and runs its own recovery sample
	// locally. Fanning out again around the recovery would release the ranks into a decode
	// they are already running.
	//
	// Inert on a single-rank serve (FAK_EP_FANOUT_ADDRS unset yields no follower URLs).
	waitEPFanout, ok := s.startEPFanoutFollowers(w, r, epRouteResponses)
	if !ok {
		return
	}
	defer waitEPFanout()
	var req ResponsesRequest
	if !decodeRequestBody(w, r, &req) {
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
	if rejectInvalidSampling(w, validateResponsesSampling(req)) {
		return
	}
	tools := responsesToolsToToolDefs(req.Tools)

	reqModel := strings.TrimSpace(req.Model)
	if reqModel == "" {
		reqModel = s.model
	} else if s.isForceResponsesStream() && isUnsupportedChatGPTModel(reqModel) {
		s.logf("gateway: model %q is not supported by Codex ChatGPT subscription upstream; adapting to configured default %q", reqModel, s.model)
		reqModel = s.model
	}

	ctx, reqTrace, messages, sessionTurn, admitted := s.admitServedRequest(w, r, messages)
	defer sessionTurn.complete()
	if !admitted {
		return
	}
	resultAdmissions, err := s.admitInboundResults(ctx, messages, tools, reqTrace)
	if err != nil {
		writeErr(w, http.StatusBadGateway, "upstream cache invalidation failed")
		return
	}
	if note := lowInfoReceiptFuseNote(resultAdmissions); note != "" {
		resp := responsesResponse{
			ID:        "resp_fak_" + itoa(uint64(time.Now().UnixNano())),
			Object:    "response",
			CreatedAt: time.Now().Unix(),
			Model:     reqModel,
			Status:    "completed",
			Output: responsesOutputFromAssistant(agent.Message{
				Role:    agent.RoleAssistant,
				Content: note,
			}),
			OutputText: note,
			Usage:      responsesUsage{},
			Fak:        fakExtFrom(nil, resultAdmissions),
		}
		if req.Stream {
			s.writeResponsesStream(w, resp)
		} else {
			writeJSON(w, http.StatusOK, resp)
		}
		return
	}

	if shouldYield, estTokens, toolCalls := shouldYieldResponsesSubturn(messages, tools, req.Input); shouldYield {
		if s.logf != nil {
			s.logf("gateway: responses sub-turn yield valve activated: est_tokens=%d tool_calls=%d trace_id=%s", estTokens, toolCalls, reqTrace)
		}
		w.Header().Set(SubturnYieldHeader, "true")
		resp := makeSubturnYieldResponse(reqModel, resultAdmissions)
		if req.Stream {
			s.writeResponsesStream(w, resp)
		} else {
			writeJSON(w, http.StatusOK, resp)
		}
		return
	}

	// Hoisted so the #5212 denial-recovery sample re-samples under the SAME sampling
	// contract as the first call — a recovery that quietly changed model or budget
	// would not be the same turn continuing.
	sampleOpts := []agent.SampleOpt{
		agent.WithModel(reqModel),
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxOutputTokens)),
		agent.WithTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
	}

	began := time.Now()
	comp, err := s.completeServed(ctx, sessionTurn, messages, tools, sampleOpts...)
	if err != nil {
		s.renderTurnDebugError(reqTrace, "openai_responses", err, time.Since(began))
		s.logf("gateway: upstream model error: %v", err)
		s.writeUpstreamErr(w, err)
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

	kept, adjs, dropped, servedText, servedHits, bodyRefused := s.adjudicateProposedTurn(ctx, asst, reqTrace)

	// #5212: a turn whose every proposed call was refused would otherwise reach Codex as
	// a `completed` response carrying ONLY the guard's remediation prose — which Codex
	// reads as the model authoring a final answer, and answers with `task_complete` while
	// the requested work sits untouched. Hand the refusals back to the MODEL as structured
	// tool results and re-sample ONCE, so the turn gets a second actuation opportunity
	// before the client ever sees a response. refusedFirst holds the original refusals
	// (empty when no recovery ran), which stay on the wire extension and in the evidence:
	// they happened, whatever the recovery went on to do.
	//
	var refusedFirst []ToolAdjudication
	if turnIsDenialOnly(kept, dropped, asst.Content, bodyRefused, servedText) {
		if rc, ok := s.recoverDeniedResponsesTurn(ctx, sessionTurn, messages, comp.Message, adjs, tools, sampleOpts...); ok {
			refusedFirst = adjs
			firstUsage := comp.Usage
			comp = rc
			comp.Usage = foldRecoveryUsage(firstUsage, rc.Usage)
			asst = comp.Message
			asst.Role = agent.RoleAssistant
			kept, adjs, dropped, servedText, servedHits, bodyRefused = s.adjudicateProposedTurn(ctx, asst, reqTrace)
		}
	}
	// blocked: the recovery ran and STILL produced neither an allowed actuation nor a
	// model-authored answer. That is a blocked turn, and it is rendered as one below
	// rather than dressed up as a completion.
	blocked := len(refusedFirst) > 0 && turnIsDenialOnly(kept, dropped, asst.Content, bodyRefused, servedText)
	turnAdjs := turnAdjudications(refusedFirst, adjs)

	// #5212: fold this turn's adjudication SHAPE into the same turn-control signal the
	// Anthropic wire already records (messages.go). The Responses wire recorded NOTHING
	// here before, so a Codex session stopping on the same refusal turn after turn was
	// indistinguishable from a run of clean completions — which is why the body's operator
	// had to notice the false `task_complete` by hand. Recorded exactly ONCE per HTTP turn,
	// on the turn's FINAL shape, so a recovered turn resets the streak rather than counting
	// the refusal it successfully routed around.
	//
	// A blocked turn is forced to deny-all even when its refusals were individually tagged
	// RETRYABLE: "retryable" describes a call the model may fix, and by this point the model
	// has already BEEN handed the refusal and re-sampled without producing an allowed call.
	// The retry happened and failed, so what remains is a terminal stop, and counting it as
	// mere feedback would hide precisely the denial→terminal transition this issue is about.
	signal := adjudicationOutcomeForTurn(adjs, len(kept), servedHits)
	if blocked {
		signal = adjudicationOutcomeDenyAll
	}
	denyFP := ""
	if signal == adjudicationOutcomeDenyAll {
		denyFP = denyAllFingerprint(turnAdjs)
	}
	s.recordAdjudicationOutcome(signal, denyFP)

	asst.ToolCalls = kept
	// #3567 output-side shadow: classify the MODEL's own outbound prose (sampled,
	// observe-only) before fak blanks/appends anything -- mirror of the chat path.
	outputNegframeAudit.observe(asst.Content)
	if bodyRefused {
		asst.Content = ""
	}
	// vDSO served-inline (vDSO live in the hot path): fold a fresh cache hit into the
	// assistant text and drop the call, so the client never re-runs the read.
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
	if anyLivelock(adjs) {
		asst.Content = prependAdjudicationContentNote(asst.Content, adjs)
	}

	// finishReason drives both the logged turn classification and the Responses
	// status: a length-truncated turn is `incomplete`, everything else `completed`.
	finish := comp.FinishReason
	if len(kept) > 0 {
		finish = "tool_calls"
	} else if dropped > 0 || servedHits > 0 {
		// Every proposed call was refused OR served inline: give even a fak-unaware
		// client an actionable in-band message instead of an empty turn (mirrors chat).
		finish = "stop"
		if asst.Content == "" {
			asst.Content = denySummary(turnAdjs)
		}
	}
	// #5212: lead the note with the typed blocked state, so neither the model nor a
	// status consumer can read the guard's remediation prose as a finished answer.
	if blocked {
		asst.Content = blockedByGuardNote(turnAdjs) + "\n" + asst.Content
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
		Usage:      responsesUsageFrom(comp.Usage),
	}
	if comp.FinishReason == "length" {
		resp.IncompleteDetails = &responsesIncomplete{Reason: "max_output_tokens"}
	} else if blocked {
		// #5212: the kernel interrupted this turn and recovery could not continue it.
		// `incomplete` + a fak-namespaced reason is the typed BLOCKED state — the wire
		// distinction between "the model is done" and "the guard stopped this turn".
		resp.Status = "incomplete"
		resp.IncompleteDetails = &responsesIncomplete{Reason: deniedGuardIncompleteReason}
	}
	for i := range turnAdjs {
		AttachOperatorRemedyMetadata(&turnAdjs[i])
	}
	SetOperatorRemedyHeaders(w, turnAdjs)
	if len(turnAdjs) > 0 || len(resultAdmissions) > 0 {
		resp.Fak = &FakExt{Adjudications: turnAdjs, ResultAdmissions: resultAdmissions}
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
						Function: agent.Func{Name: it.Name, Namespace: it.Namespace, Arguments: it.Arguments},
					}},
				})
			case "function_call_output":
				// A tool RESULT the client executed — normalize the Responses wire's
				// documented string|input_text union, then send the canonical bytes through
				// the same result-side floor. Representation skew may degrade safely; policy
				// and malformed content still fail closed.
				output, err := decodeResponsesFunctionCallOutput(it.Output)
				if err != nil {
					return nil, err
				}
				msgs = append(msgs, agent.Message{
					Role:       agent.RoleTool,
					ToolCallID: it.CallID,
					Content:    output,
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

const (
	// Codex currently emits a short list of textual MCP result parts. Keep that
	// compatibility path finite even when a hostile client sends tiny fragments.
	maxResponsesFunctionOutputParts = 256
	// The route-level transcript limit is the compatibility ceiling for a single
	// decoded tool result too; no valid HTTP request can exceed it.
	maxResponsesFunctionOutputBytes = maxTranscriptBody
)

type responsesFunctionOutputPart struct {
	Type string  `json:"type"`
	Text *string `json:"text,omitempty"`
}

// decodeResponsesFunctionCallOutput normalizes the Responses output union without
// weakening result admission. Legacy strings retain their exact decoded value.
// Structured output accepts only textual input-content parts; image, file, unknown,
// and malformed parts are refused instead of silently disappearing before the
// result-side policy sees them.
func decodeResponsesFunctionCallOutput(raw json.RawMessage) (string, error) {
	b := trimLeadingWS(raw)
	if len(b) == 0 {
		return "", errInput("function_call_output.output is required")
	}
	switch b[0] {
	case '"':
		var output string
		if err := json.Unmarshal(raw, &output); err != nil {
			return "", errInput("function_call_output.output string is malformed: " + err.Error())
		}
		if len(output) > maxResponsesFunctionOutputBytes {
			return "", errInput(fmt.Sprintf("function_call_output.output exceeds the %d-byte limit", maxResponsesFunctionOutputBytes))
		}
		return output, nil
	case '[':
		var parts []responsesFunctionOutputPart
		if err := json.Unmarshal(raw, &parts); err != nil {
			return "", errInput("function_call_output.output content array is malformed: " + err.Error())
		}
		if len(parts) == 0 {
			return "", errInput("function_call_output.output content array must not be empty")
		}
		if len(parts) > maxResponsesFunctionOutputParts {
			return "", errInput(fmt.Sprintf("function_call_output.output has too many content parts (maximum %d)", maxResponsesFunctionOutputParts))
		}
		var output strings.Builder
		for i, part := range parts {
			if part.Type == "" {
				return "", errInput(fmt.Sprintf("function_call_output.output[%d] content type is required", i))
			}
			if part.Type != "input_text" {
				return "", errInput(fmt.Sprintf("function_call_output.output[%d] has unsupported content type %q", i, part.Type))
			}
			if part.Text == nil {
				return "", errInput(fmt.Sprintf("function_call_output.output[%d].input_text.text is required", i))
			}
			separatorBytes := 0
			if i > 0 {
				separatorBytes = 1
			}
			if len(*part.Text) > maxResponsesFunctionOutputBytes-output.Len()-separatorBytes {
				return "", errInput(fmt.Sprintf("function_call_output.output exceeds the %d-byte limit", maxResponsesFunctionOutputBytes))
			}
			if separatorBytes != 0 {
				output.WriteByte('\n')
			}
			output.WriteString(*part.Text)
		}
		return output.String(), nil
	default:
		return "", errInput("function_call_output.output must be a string or an array of content parts")
	}
}

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

const (
	mcpLegacyPrefix = "mcp__fak__"
	mcpCanonPrefix  = "mcp__fak" + "_guard__"
)

// responsesToolsToToolDefs maps the flat Responses function-tool shape onto the
// gateway's agent.ToolDef (the nested chat shape the planner consumes). A
// non-function tool type (web_search, file_search, computer_use, ...) is skipped
// rather than 400'd: fak adjudicates the FUNCTION tool calls; a built-in tool the
// upstream resolves itself is not a kernel-mediated call and carries no args to gate.
//
// When incoming tools contain duplicate definitions for the same base tool across legacy
// and canonical namespaces, collapse them into the canonical tool definition.
func responsesToolsToToolDefs(tools []responsesTool) []agent.ToolDef {
	if len(tools) == 0 {
		return nil
	}
	hasCanon := make(map[string]bool)
	for _, t := range tools {
		if t.Type == "function" && strings.HasPrefix(t.Name, mcpCanonPrefix) {
			hasCanon[strings.TrimPrefix(t.Name, mcpCanonPrefix)] = true
		}
	}
	out := make([]agent.ToolDef, 0, len(tools))
	seen := make(map[string]bool)
	for _, t := range tools {
		if t.Type != "function" {
			out = append(out, agent.ToolDef{Type: t.Type, ResponsesWire: append(json.RawMessage(nil), t.raw...)})
			continue
		}
		if t.Name == "" {
			continue
		}
		if strings.HasPrefix(t.Name, mcpLegacyPrefix) {
			toolName := strings.TrimPrefix(t.Name, mcpLegacyPrefix)
			if hasCanon[toolName] {
				// Suppress legacy tool when canonical definition is present.
				continue
			}
		}
		if seen[t.Name] {
			continue
		}
		seen[t.Name] = true
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

const reservedGuardBanner = "[fak] BLOCKED_BY_GUARD"

// demoteReservedGuardBanner keeps kernel terminal receipts distinguishable from model
// prose. BLOCKED_BY_GUARD is emitted only by blockedResponsesCompletion after the
// kernel has indexed real denied call IDs. If an upstream model writes the same reserved
// prefix in ordinary assistant content, mark it untrusted before it reaches the client;
// otherwise invented call IDs can masquerade as a kernel receipt and falsely stop the
// managed harness (#6810).
func demoteReservedGuardBanner(text string) string {
	if !strings.Contains(text, reservedGuardBanner) {
		return text
	}
	return strings.ReplaceAll(text, reservedGuardBanner, "[model text; not a fak receipt] BLOCKED_BY_GUARD")
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
		asst.Content = demoteReservedGuardBanner(asst.Content)
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
			Namespace: tc.Function.Namespace,
			Arguments: args,
			Status:    "completed",
		})
	}
	return out
}

func lowInfoReceiptFuseNote(adms []ResultAdmission) string {
	for _, adm := range adms {
		if adm.Livelock == nil || adm.Livelock.Reason != lowInfoReceiptReason {
			continue
		}
		return "[fak] stopped repeated low-information tool-result loop: " +
			adm.Livelock.Event + " repeat=" + itoa(uint64(adm.Livelock.RepeatCount)) +
			" repeated_result=" + resultToolLabel(adm) + "@" + adm.ResultDigest +
			". Stop calling " + resultToolLabel(adm) + " until effect-capable work changes state; continue from the existing state or ask for operator input."
	}
	return ""
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
// (per item) → response.completed. This is the synthesized-stream analogue of
// writeChatCompletionStream: the gateway buffers the entire turn, adjudicates the
// proposed tool calls, then re-serializes the adjudicated turn as SSE.
func (s *Server) writeResponsesStream(w http.ResponseWriter, resp responsesResponse) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	nextSeq := 0
	type responseEvent struct {
		Type           string            `json:"type"`
		SequenceNumber int               `json:"sequence_number"`
		Response       responsesResponse `json:"response"`
	}
	writeResponseEvent := func(event string, response responsesResponse) {
		_ = writeSSEEvent(w, event, responseEvent{
			Type:           event,
			SequenceNumber: nextSeq,
			Response:       response,
		})
		nextSeq++
	}

	// response.created: initial event with response metadata. The generated turn is
	// already buffered, but the stream envelope still mirrors the live Responses API.
	created := resp
	created.Status = "in_progress"
	created.Output = []responsesOutputItem{}
	created.OutputText = ""
	created.Usage = responsesUsage{}
	writeResponseEvent("response.created", created)

	// Emit each output item: added → done
	for i, item := range resp.Output {
		// response.output_item.added
		type outputItemEvent struct {
			Type           string              `json:"type"`
			SequenceNumber int                 `json:"sequence_number"`
			OutputIndex    int                 `json:"output_index"`
			Item           responsesOutputItem `json:"item"`
		}
		_ = writeSSEEvent(w, "response.output_item.added", outputItemEvent{
			Type:           "response.output_item.added",
			SequenceNumber: nextSeq,
			OutputIndex:    i,
			Item:           item,
		})
		nextSeq++

		// response.output_item.done
		_ = writeSSEEvent(w, "response.output_item.done", outputItemEvent{
			Type:           "response.output_item.done",
			SequenceNumber: nextSeq,
			OutputIndex:    i,
			Item:           item,
		})
		nextSeq++
	}

	// response.completed: terminal event with optional fak extension and incomplete details.
	// Codex 0.142.4 treats a stream that closes before this event as incomplete and retries.
	writeResponseEvent("response.completed", resp)

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// writeSSEEvent writes a single SSE event with a typed event name. The Responses
// wire uses named event types (response.created, response.completed) rather than the
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

func (s *Server) isForceResponsesStream() bool {
	if s == nil || s.planner == nil {
		return false
	}
	if hp, ok := s.planner.(*agent.HTTPPlanner); ok {
		return hp.ForceResponsesStream
	}
	if frs, ok := s.planner.(interface{ IsForceResponsesStream() bool }); ok {
		return frs.IsForceResponsesStream()
	}
	return false
}

// isUnsupportedChatGPTModel reports whether model cannot be served by OpenAI's ChatGPT
// backend (such as non-OpenAI model families or unsupported slugs like gpt-5.3-codex).
func isUnsupportedChatGPTModel(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return false
	}
	if strings.HasPrefix(m, "openai/") {
		m = strings.TrimPrefix(m, "openai/")
	}
	for _, prefix := range []string{"gemini", "claude", "qwen", "glm", "llama", "deepseek", "mistral", "gemma"} {
		if strings.HasPrefix(m, prefix) {
			return true
		}
	}
	if m == "gpt-5.3-codex" || m == "gpt-5.3" {
		return true
	}
	return false
}
