package gateway

// gemini.go is the native Gemini generateContent front door (POST
// /v1beta/models/{model}:generateContent and :streamGenerateContent). It is the
// Gemini-CLI / google-genai-facing twin of handleAnthropicMessages (and of
// handleChatCompletions on the OpenAI wire): the SAME planner, the SAME kernel
// adjudication boundary (s.adjudicateProposed on every proposed function call +
// admitInboundResults on every inbound functionResponse), a different downstream
// wire. Point a Gemini-native client at it by repointing its base URL from
// https://generativelanguage.googleapis.com at the fak host, and every function
// call the served model proposes is dropped/repaired by the kernel before the
// client ever sees it.
//
// The upstream planner is non-streaming (it buffers the whole completion), so a
// :streamGenerateContent request SYNTHESIZES a well-formed Gemini SSE sequence from
// the finished, already-adjudicated turn rather than truly streaming tokens — the
// same posture the Anthropic wire takes. A Gemini client parses the data frames
// identically; the round trip — and the functionCall ids it matches results back
// by — is faithful.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// errGeminiToolCallConformance is the fail-closed signal that the upstream
// announced tool calls but none survived parsing (mirrors the OpenAI path's
// ToolCallsDropped guard). The gateway refuses to render an empty turn that would
// silently skip adjudication.
var errGeminiToolCallConformance = errors.New("upstream tool-call format not recognized; refusing to skip adjudication")

// geminiGenerateContentResponse is the buffered (non-streaming) generateContent
// body, and the single SSE data frame a synthesized stream emits.
//
// Fak is a non-standard top-level extension carrying the kernel's per-call
// adjudications — the Gemini-wire twin of ChatResponse.Fak. Gemini's response
// schema is otherwise fixed, but its clients (Gemini CLI, the google-genai SDKs)
// tolerate unknown top-level response keys, so a fak-aware tool can read
// structured verdicts here while the standard parser ignores it. The
// human/agent-readable form of the same information rides as an in-band text part
// (see prependGeminiTextPart) — that text is what a coding agent actually reacts to.
type geminiGenerateContentResponse struct {
	Candidates    []geminiCandidateOut `json:"candidates"`
	UsageMetadata geminiUsageOut       `json:"usageMetadata"`
	ModelVersion  string               `json:"modelVersion,omitempty"`
	Fak           *FakExt              `json:"fak,omitempty"`
}

type geminiCandidateOut struct {
	Content      geminiContentOut `json:"content"`
	FinishReason string           `json:"finishReason"`
	Index        int              `json:"index,omitempty"`
}

type geminiContentOut struct {
	Role  string                `json:"role,omitempty"`
	Parts []agent.GeminiPartOut `json:"parts"`
}

// geminiUsageOut mirrors the generateContent usageMetadata object.
type geminiUsageOut struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type geminiTurn struct {
	candidates       []geminiCandidateOut
	usage            geminiUsageOut
	model            string
	adjs             []ToolAdjudication
	resultAdmissions []ResultAdmission
}

// handleGeminiGenerateContent is the adjudication PROXY on the Gemini wire. It
// decodes the inbound generateContent request into the canonical transcript, arms
// the result-side floor on any inbound functionResponse, forwards the (possibly
// quarantined) transcript to the configured model, runs every PROPOSED
// functionCall through the kernel, and renders the survivors back as a Gemini
// candidate — buffered, or as SSE when the client hit :streamGenerateContent.
func (s *Server) handleGeminiGenerateContent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	model, method, ok := parseGeminiPath(r.URL.Path)
	if !ok {
		writeErr(w, http.StatusNotFound, "unknown Gemini route")
		return
	}
	switch method {
	case "generateContent", "streamGenerateContent":
	default:
		writeErr(w, http.StatusNotFound, "unknown Gemini method "+method)
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
	req, err := agent.DecodeGeminiGenerateContentRequest(raw, model)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	// An empty contents array is a CLIENT error, not an upstream failure — reject
	// it here with a 400 rather than forward a degenerate request and surface the
	// upstream's own 400 as a confusing 502 (parity with the OpenAI/Anthropic paths).
	if len(req.Messages) == 0 {
		writeErr(w, http.StatusBadRequest, "contents: field required")
		return
	}
	stream := req.Stream || method == "streamGenerateContent"

	ctx := r.Context()
	reqTrace := s.useHTTPTrace(w, r, "")
	// Operator control / budget / pace at the served request boundary. With
	// DecideSession wired this mutates the live session table (TurnsLeft debit,
	// budget exhaustion, pace cap); without it the legacy observe-only admission guard
	// still refuses paused/draining/stopped sessions.
	sessionTurn, ok, canceled := s.beginServedSessionTurn(ctx, reqTrace)
	if canceled {
		return
	}
	if !ok {
		writeSessionRefusal(w, sessionTurn.state)
		return
	}

	// Arm the result-side floor BEFORE the planner runs — exact parity with the
	// OpenAI and Anthropic proxy paths. decodeGeminiContent already turned each
	// inbound functionResponse into a canonical RoleTool message, so
	// admitInboundResults routes each through k.AdmitResult keyed on reqTrace.
	resultAdmissions, err := s.admitInboundResults(ctx, req.Messages, reqTrace)
	if err != nil {
		writeErr(w, http.StatusBadGateway, "upstream cache invalidation failed")
		return
	}

	// Forward the Gemini client's per-request sampling (maxOutputTokens,
	// temperature, topP, topK, stopSequences) to the upstream model. Each option is
	// a no-op when its field is absent, so a client that omits them gets the planner
	// default — identical to the pre-seam behavior.
	var temp *float64
	if req.Temperature != 0 {
		temp = &req.Temperature
	}
	opts := []agent.SampleOpt{
		agent.WithModel(model),
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxTokens)),
		agent.WithTemperature(temp),
		agent.WithTopP(req.TopP),
		agent.WithTopK(req.TopK),
		agent.WithStop(req.StopSequences),
	}

	if stream {
		s.streamGeminiPending(w, r, req, opts, reqTrace, model, sessionTurn, resultAdmissions)
		return
	}

	turn, err := s.completeGeminiTurn(ctx, req, opts, reqTrace, model, sessionTurn, resultAdmissions)
	if err != nil {
		s.writeGeminiUpstreamError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, geminiGenerateContentResponse{
		Candidates:    turn.candidates,
		UsageMetadata: turn.usage,
		ModelVersion:  turn.model,
		Fak:           fakExtFrom(turn.adjs, turn.resultAdmissions),
	})
}

// completeGeminiTurn runs the planner, adjudicates every proposed function call,
// and renders the Gemini candidate/usage shape. It is the shared core of the
// buffered and streaming paths.
func (s *Server) completeGeminiTurn(ctx context.Context, req *agent.GeminiGenerateContentRequest, opts []agent.SampleOpt, reqTrace, model string, sessionTurn servedSessionTurn, resultAdmissions []ResultAdmission) (*geminiTurn, error) {
	comp, err := s.completeServed(ctx, sessionTurn, req.Messages, req.Tools, opts...)
	if err != nil {
		return nil, err
	}
	asst := comp.Message
	asst.Role = agent.RoleAssistant

	// Tool-call conformance: the upstream announced function calls but NONE
	// survived parsing + the text-lift fallback. Proceeding would skip
	// adjudication on a call the model intended to make — the exact silent no-op a
	// non-OpenAI-shaped emitter causes. Fail closed (parity with the OpenAI path).
	if comp.ToolCallsDropped && len(asst.ToolCalls) == 0 {
		s.logf("gateway: upstream announced tool_calls but none parsed (conformance fail-closed); model=%s", model)
		return nil, errGeminiToolCallConformance
	}

	kept, adjs, dropped := s.adjudicateProposed(ctx, asst.ToolCalls, reqTrace)
	asst.ToolCalls = kept

	parts := agent.GeminiResponseParts(asst)

	// Make the kernel's decisions LEGIBLE in-band. A Gemini client reads the
	// candidate parts (and feeds them back to its model) but not the `fak`
	// extension, so a dropped or repaired call is otherwise invisible — the agent
	// re-proposes a denied call forever, or proceeds unaware its args were
	// rewritten. Whenever a drop or repair happened, prepend a short text part.
	if dropped > 0 || anyRepaired(adjs) {
		if note := adjudicationNote(adjs); note != "" {
			parts = prependGeminiTextPart(parts, note)
		}
	}
	if note := resultAdmissionNote(resultAdmissions); note != "" {
		parts = prependGeminiTextPart(parts, note)
	}
	// Every proposed call refused and no surviving prose: carry the deny summary as
	// a text part so even a fak-unaware client gets something actionable, rather
	// than an empty candidate (parity with the OpenAI denySummary behavior).
	if len(parts) == 0 && dropped > 0 {
		parts = prependGeminiTextPart(parts, denySummary(adjs))
	}

	// Echo the model the UPSTREAM reported it served; fall back to the client's
	// requested model (or the configured model) when the upstream omitted one.
	modelVersion := comp.Model
	if modelVersion == "" {
		modelVersion = model
	}
	if modelVersion == "" {
		modelVersion = s.model
	}

	total := comp.Usage.TotalTokens
	if total == 0 {
		total = comp.Usage.PromptTokens + comp.Usage.CompletionTokens
	}
	return &geminiTurn{
		candidates: []geminiCandidateOut{{
			Content:      geminiContentOut{Role: "model", Parts: parts},
			FinishReason: agent.GeminiFinishReason(comp.FinishReason),
		}},
		usage: geminiUsageOut{
			PromptTokenCount:     comp.Usage.PromptTokens,
			CandidatesTokenCount: comp.Usage.CompletionTokens,
			TotalTokenCount:      total,
		},
		model:            modelVersion,
		adjs:             adjs,
		resultAdmissions: resultAdmissions,
	}, nil
}

// streamGeminiPending completes the turn, then emits a single Gemini SSE data frame
// carrying the finished, already-adjudicated candidate. There is no Gemini
// "stream start" event to send before the model answers (unlike the Anthropic
// message_start), so nothing is written until the turn is ready — which means an
// upstream failure still maps to an honest HTTP status, not an error frame inside
// an already-opened 200 stream. A Gemini client parses the data frame identically
// to a buffered response.
func (s *Server) streamGeminiPending(w http.ResponseWriter, r *http.Request, req *agent.GeminiGenerateContentRequest, opts []agent.SampleOpt, reqTrace, model string, sessionTurn servedSessionTurn, resultAdmissions []ResultAdmission) {
	turn, err := s.completeGeminiTurn(r.Context(), req, opts, reqTrace, model, sessionTurn, resultAdmissions)
	if err != nil {
		s.writeGeminiUpstreamError(w, err)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	frame := geminiGenerateContentResponse{
		Candidates:    turn.candidates,
		UsageMetadata: turn.usage,
		ModelVersion:  turn.model,
		Fak:           fakExtFrom(turn.adjs, turn.resultAdmissions),
	}
	b, _ := json.Marshal(frame)
	_, _ = w.Write([]byte("data: "))
	_, _ = w.Write(b)
	_, _ = w.Write([]byte("\n\n"))
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// writeGeminiUpstreamError maps a completeGeminiTurn failure to the HTTP status and
// a GENERIC message — the planner error embeds up to 400 bytes of the upstream
// provider's raw body, which must not cross the trust boundary to a (possibly
// unauthenticated) downstream caller (parity with the OpenAI/Anthropic paths).
func (s *Server) writeGeminiUpstreamError(w http.ResponseWriter, err error) {
	if errors.Is(err, errGeminiToolCallConformance) {
		s.logf("gateway: %v", err)
		writeErr(w, http.StatusBadGateway, err.Error())
		return
	}
	s.logf("gateway: upstream model error (gemini): %v", err)
	status, code, msg := s.plannerErrorStatus(err)
	writeErrCode(w, status, code, msg)
}

// prependGeminiTextPart inserts an in-band [fak] note as the FIRST part so a client
// that reads parts top-to-bottom sees the kernel's decision before the surviving
// functionCall parts it is about to run. The note never replaces model prose —
// existing text/functionCall parts follow it untouched.
func prependGeminiTextPart(parts []agent.GeminiPartOut, text string) []agent.GeminiPartOut {
	out := make([]agent.GeminiPartOut, 0, len(parts)+1)
	out = append(out, agent.GeminiPartOut{Text: text})
	return append(out, parts...)
}

// parseGeminiPath splits a /v1beta/models/{model}:{method} route into its model and
// method. The model id never contains ':', so the method is everything after the
// final colon. Both the /v1beta/ and /v1/ version prefixes are accepted so a client
// or proxy that strips/stages the beta prefix still resolves.
func parseGeminiPath(path string) (model, method string, ok bool) {
	for _, pfx := range []string{"/v1beta/", "/v1/"} {
		if strings.HasPrefix(path, pfx) {
			path = strings.TrimPrefix(path, pfx)
			break
		}
	}
	path = strings.TrimPrefix(path, "models/")
	idx := strings.LastIndex(path, ":")
	if idx < 0 {
		return "", "", false
	}
	model, method = path[:idx], path[idx+1:]
	if model == "" || method == "" {
		return "", "", false
	}
	return model, method, true
}
