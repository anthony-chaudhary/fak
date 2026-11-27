package localappcert

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{now: time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

type recordedRequest struct {
	Method string
	Path   string
	Body   map[string]any
}

type sixGateFixture struct {
	mu                   sync.Mutex
	requests             []recordedRequest
	clock                *fakeClock
	models               []string
	modelsStatus         int
	chatFinishReason     string
	chatContent          string
	chatStatus           int
	reasoningField       string
	reasoningValue       string
	reasoningContent     string
	reasoningFinish      string
	reasoningStatus      int
	toolCallName         string
	toolCallCity         string
	toolSkipCall         bool
	toolOmitNonce        bool
	contextDropCount     int
	contextDropEOF       bool
	contextCached        bool
	contextPromptTokens  int
	contextOmitCode      bool
	speedSSE             bool
	speedSource          string
	speedCompletion      int
	speedDraftTokens     int
	speedAcceptedTokens  int
	speedRejectedTokens  int
	speedDuration        time.Duration
	speedChunkCount      int
	speedPerChunkAdvance time.Duration
	speedPredictedMS     float64
	speedFinishReason    string
	speedOmitUsage       bool
}

func newPassingFixture(clock *fakeClock) *sixGateFixture {
	return &sixGateFixture{
		clock:                clock,
		models:               []string{"Qwen/Qwen3.8-27B"},
		modelsStatus:         http.StatusOK,
		chatFinishReason:     "stop",
		chatContent:          "Paris is the capital of France.",
		chatStatus:           http.StatusOK,
		reasoningField:       "reasoning_content",
		reasoningValue:       "17 * 20 = 340 and 17 * 3 = 51, so 340 + 51 = 391.",
		reasoningContent:     "17 * 23 = 391.",
		reasoningFinish:      "stop",
		reasoningStatus:      http.StatusOK,
		toolCallName:         "get_weather",
		toolCallCity:         "Tokyo",
		contextPromptTokens:  850,
		speedSSE:             true,
		speedSource:          SourceTokenizer,
		speedCompletion:      60,
		speedDuration:        2 * time.Second,
		speedChunkCount:      3,
		speedPerChunkAdvance: time.Second,
		speedFinishReason:    "stop",
	}
}

func (f *sixGateFixture) server(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawBody, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		if len(bytes.TrimSpace(rawBody)) > 0 {
			_ = json.Unmarshal(rawBody, &parsed)
		}

		f.mu.Lock()
		f.requests = append(f.requests, recordedRequest{Method: r.Method, Path: r.URL.Path, Body: parsed})
		f.mu.Unlock()

		if r.URL.Path == "/v1/models" {
			f.mu.Lock()
			status := f.modelsStatus
			models := append([]string(nil), f.models...)
			f.mu.Unlock()
			if status != http.StatusOK {
				http.Error(w, "models error", status)
				return
			}
			data := make([]map[string]any, 0, len(models))
			for _, id := range models {
				data = append(data, map[string]any{"id": id, "object": "model"})
			}
			writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
			return
		}

		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}

		// Answers must never be capped in any request.
		if _, has := parsed["max_tokens"]; has {
			http.Error(w, "unexpected max_tokens", http.StatusBadRequest)
			return
		}
		if _, has := parsed["max_completion_tokens"]; has {
			http.Error(w, "unexpected max_completion_tokens", http.StatusBadRequest)
			return
		}

		stream, _ := parsed["stream"].(bool)
		if stream {
			f.handleSpeed(w)
			return
		}

		if _, hasTools := parsed["tools"]; hasTools {
			f.handleTools(w, parsed)
			return
		}

		msgs, _ := parsed["messages"].([]any)
		firstMsg, _ := msgs[0].(map[string]any)
		userPrompt, _ := firstMsg["content"].(string)

		switch {
		case strings.Contains(userPrompt, "Access code: "):
			f.handleContext(w, r, userPrompt)
		case strings.Contains(userPrompt, "17 * 23"):
			f.handleReasoning(w)
		default:
			f.handleChat(w)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (f *sixGateFixture) handleChat(w http.ResponseWriter) {
	f.mu.Lock()
	status := f.chatStatus
	content := f.chatContent
	finish := f.chatFinishReason
	f.mu.Unlock()
	if status != http.StatusOK {
		http.Error(w, "chat failure", status)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"choices": []map[string]any{{
			"index":         0,
			"finish_reason": finish,
			"message": map[string]any{
				"role":    "assistant",
				"content": content,
			},
		}},
	})
}

func (f *sixGateFixture) handleReasoning(w http.ResponseWriter) {
	f.mu.Lock()
	status := f.reasoningStatus
	field := f.reasoningField
	reasoningVal := f.reasoningValue
	content := f.reasoningContent
	finish := f.reasoningFinish
	f.mu.Unlock()
	if status != http.StatusOK {
		http.Error(w, "reasoning failure", status)
		return
	}
	msg := map[string]any{
		"role":    "assistant",
		"content": content,
	}
	if field != "" && reasoningVal != "" {
		msg[field] = reasoningVal
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"choices": []map[string]any{{
			"index":         0,
			"finish_reason": finish,
			"message":       msg,
		}},
	})
}

func (f *sixGateFixture) handleTools(w http.ResponseWriter, parsed map[string]any) {
	msgs, _ := parsed["messages"].([]any)
	f.mu.Lock()
	toolName := f.toolCallName
	city := f.toolCallCity
	skipCall := f.toolSkipCall
	omitNonce := f.toolOmitNonce
	f.mu.Unlock()

	if len(msgs) == 1 {
		if skipCall {
			writeJSON(w, http.StatusOK, map[string]any{
				"choices": []map[string]any{{
					"index":         0,
					"finish_reason": "stop",
					"message": map[string]any{
						"role":    "assistant",
						"content": "The weather in Tokyo is sunny.",
					},
				}},
			})
			return
		}
		argsBytes, _ := json.Marshal(map[string]any{"city": city})
		writeJSON(w, http.StatusOK, map[string]any{
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "tool_calls",
				"message": map[string]any{
					"role": "assistant",
					"tool_calls": []map[string]any{{
						"id":   "call_1",
						"type": "function",
						"function": map[string]any{
							"name":      toolName,
							"arguments": string(argsBytes),
						},
					}},
				},
			}},
		})
		return
	}

	toolMsg, _ := msgs[len(msgs)-1].(map[string]any)
	toolBody, _ := toolMsg["content"].(string)
	var toolResult map[string]any
	_ = json.Unmarshal([]byte(toolBody), &toolResult)
	code, _ := toolResult["verification_code"].(string)
	reply := "Tokyo is clear at 21C."
	if !omitNonce {
		reply += " Verification code: " + code
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"choices": []map[string]any{{
			"index":         0,
			"finish_reason": "stop",
			"message": map[string]any{
				"role":    "assistant",
				"content": reply,
			},
		}},
	})
}

var accessCodePattern = regexp.MustCompile(`Access code:\s*(\S+)`)

func (f *sixGateFixture) handleContext(w http.ResponseWriter, r *http.Request, prompt string) {
	f.mu.Lock()
	shouldDrop := f.contextDropCount > 0
	if shouldDrop {
		f.contextDropCount--
	}
	dropEOF := f.contextDropEOF
	cached := f.contextCached
	promptTokens := f.contextPromptTokens
	omitCode := f.contextOmitCode
	clock := f.clock
	f.mu.Unlock()

	if shouldDrop {
		if clock != nil {
			clock.Advance(250 * time.Millisecond)
		}
		if dropEOF {
			if hj, ok := w.(http.Hijacker); ok {
				conn, _, err := hj.Hijack()
				if err == nil {
					_ = conn.Close()
					return
				}
			}
		}
		http.Error(w, "upstream reset", http.StatusBadGateway)
		return
	}

	if clock != nil {
		clock.Advance(200 * time.Millisecond)
	}
	match := accessCodePattern.FindStringSubmatch(prompt)
	code := ""
	if len(match) == 2 {
		code = match[1]
	}
	reply := "NONE"
	if !omitCode {
		reply = code
	}
	usage := map[string]any{
		"prompt_tokens":     promptTokens,
		"completion_tokens": 4,
	}
	timings := map[string]any{
		"prompt_n": promptTokens,
	}
	if cached {
		usage["prompt_tokens_details"] = map[string]any{"cached_tokens": promptTokens}
		timings["cache_n"] = promptTokens
		timings["prompt_per_second"] = 999999.0
	} else {
		timings["prompt_per_second"] = 4250.0
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"choices": []map[string]any{{
			"index":         0,
			"finish_reason": "stop",
			"message": map[string]any{
				"role":    "assistant",
				"content": reply,
			},
		}},
		"usage":   usage,
		"timings": timings,
	})
}

func (f *sixGateFixture) handleSpeed(w http.ResponseWriter) {
	f.mu.Lock()
	sse := f.speedSSE
	source := f.speedSource
	completion := f.speedCompletion
	draft := f.speedDraftTokens
	accepted := f.speedAcceptedTokens
	rejected := f.speedRejectedTokens
	dur := f.speedDuration
	chunks := f.speedChunkCount
	perChunk := f.speedPerChunkAdvance
	predictedMS := f.speedPredictedMS
	finish := f.speedFinishReason
	omitUsage := f.speedOmitUsage
	clock := f.clock
	f.mu.Unlock()

	usage := map[string]any{}
	if !omitUsage {
		usage["completion_tokens"] = completion
	}
	if source != "" {
		usage["source"] = source
	}
	if draft > 0 || accepted > 0 || rejected > 0 {
		usage["draft_tokens"] = draft
		usage["accepted_tokens"] = accepted
		usage["rejected_tokens"] = rejected
	}
	var timings map[string]any
	if predictedMS > 0 || source != "" {
		timings = map[string]any{}
		if predictedMS > 0 {
			timings["predicted_ms"] = predictedMS
		}
		if source != "" {
			timings["source"] = source
		}
	}

	if !sse {
		if clock != nil && dur > 0 {
			clock.Advance(dur)
		}
		resp := map[string]any{
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": finish,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Memory bandwidth bounds autoregressive token decode.",
				},
			}},
		}
		if len(usage) > 0 {
			resp["usage"] = usage
		}
		if timings != nil {
			resp["timings"] = timings
		}
		writeJSON(w, http.StatusOK, resp)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	if chunks <= 0 {
		chunks = 2
	}
	for i := 0; i < chunks; i++ {
		if i > 0 && clock != nil && perChunk > 0 {
			clock.Advance(perChunk)
		}
		chunk := map[string]any{
			"choices": []map[string]any{{
				"index": 0,
				"delta": map[string]any{"content": fmt.Sprintf("tok%d ", i)},
			}},
		}
		if i == chunks-1 {
			chunk["choices"].([]map[string]any)[0]["finish_reason"] = finish
			if len(usage) > 0 {
				chunk["usage"] = usage
			}
			if timings != nil {
				chunk["timings"] = timings
			}
		}
		b, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func TestSixGatesPassEndToEndWithEvidenceAndMatrixValidation(t *testing.T) {
	clock := newFakeClock()
	fx := newPassingFixture(clock)
	srv := fx.server(t)
	evidenceDir := t.TempDir()

	spec := SixGateSpec{
		Model:           "Qwen/Qwen3.8-27B",
		ContextWindow:   1000,
		Artifact:        "sha256:model",
		RuntimeRevision: "r1",
		Speculation: &SpeculationSpec{
			Enabled:     true,
			Method:      "mtp",
			DraftTokens: 2,
		},
	}

	report, err := RunSixGates(context.Background(), spec, SixGateOptions{
		BaseURL:         srv.URL,
		EvidenceDir:     evidenceDir,
		Now:             clock.Now,
		RuntimeRevision: "r1",
		Artifact:        "sha256:model",
		Receipt: &Receipt{
			Engine:          "fak-native",
			Fallback:        "none",
			Artifact:        "sha256:model",
			RuntimeRevision: "r1",
		},
	})
	if err != nil {
		t.Fatalf("RunSixGates() error = %v", err)
	}
	if !report.Passed || report.AbortedGate != "" {
		t.Fatalf("report = %#v", report)
	}
	if report.Proof.Gates != "load,chat,reasoning,tools,context,speed" {
		t.Fatalf("proof gates = %q", report.Proof.Gates)
	}
	if report.Proof.PrefillTokensPerSecond == nil || *report.Proof.PrefillTokensPerSecond <= 0 {
		t.Fatalf("expected non-cached prefill rate, got %v", report.Proof.PrefillTokensPerSecond)
	}
	if report.Proof.DecodeTokensPerSecond == nil || *report.Proof.DecodeTokensPerSecond != 30.0 {
		t.Fatalf("decode rate = %v, want 30.0", report.Proof.DecodeTokensPerSecond)
	}
	if !strings.HasPrefix(report.Proof.LogSHA256, "sha256:") || report.EvidenceSHA256 == "" {
		t.Fatalf("missing report hash: %#v", report.Proof)
	}
	if len(report.Gates) != len(SixGates) {
		t.Fatalf("got %d gates, want %d", len(report.Gates), len(SixGates))
	}
	for i, wantGate := range SixGates {
		g := report.Gates[i]
		if g.Gate != wantGate || g.Status != StatusPass || g.Evidence == "" || g.EvidenceSHA256 == "" {
			t.Fatalf("gate %d = %#v", i, g)
		}
		if _, err := os.Stat(g.Evidence); err != nil {
			t.Fatalf("missing gate evidence file %q: %v", g.Evidence, err)
		}
	}
	if _, err := os.Stat(filepath.Join(evidenceDir, "six-gate-report.json")); err != nil {
		t.Fatalf("missing six-gate-report.json: %v", err)
	}

	// Verify no chat request ever sent max_tokens or max_completion_tokens.
	for _, req := range fx.requests {
		if req.Path == "/v1/chat/completions" {
			if _, ok := req.Body["max_tokens"]; ok {
				t.Fatalf("request %#v capped with max_tokens", req)
			}
			if _, ok := req.Body["max_completion_tokens"]; ok {
				t.Fatalf("request %#v capped with max_completion_tokens", req)
			}
		}
	}

	// Verify Context gate prompt is sized near 85% of context_window.
	contextGate := report.Gates[4]
	if contextGate.TargetPromptTokens != 850 || contextGate.MinPromptTokens != 600 || contextGate.PromptTokens < 600 {
		t.Fatalf("context gate bounds = %#v", contextGate)
	}

	// Wire passing report into fak.local-app-certification/1 matrix validation.
	m := validMatrix()
	m.Envelopes[0].SixGate = &report
	if err := Validate(m); err != nil {
		t.Fatalf("Validate(matrix with SixGate) error = %v", err)
	}
}

func TestSixGatesOrderedAbortOnFirstNonAnsweringFailureRecordsSpecAndEvidence(t *testing.T) {
	t.Run("load_not_listed_aborts_before_chat", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.models = []string{"other-model"}
		srv := fx.server(t)
		evidenceDir := t.TempDir()

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
			Speculation:   &SpeculationSpec{Enabled: true, Method: "mtp", DraftTokens: 2},
		}, SixGateOptions{BaseURL: srv.URL, EvidenceDir: evidenceDir, Now: clock.Now})
		if err != nil {
			t.Fatalf("RunSixGates() error = %v", err)
		}
		if report.Passed || report.AbortedGate != GateLoad {
			t.Fatalf("report = %#v", report)
		}
		if report.Spec.Speculation == nil || !report.Spec.Speculation.Enabled || report.Spec.Model != "Qwen/Qwen3.8-27B" {
			t.Fatalf("spec metadata not preserved on load failure: %#v", report.Spec)
		}
		if len(report.Gates) != 1 || report.Gates[0].Gate != GateLoad || report.Gates[0].Status != StatusFail || !report.Gates[0].BlewUp {
			t.Fatalf("gates = %#v", report.Gates)
		}
		if report.Gates[0].Reason != ReasonModelNotListed || report.Gates[0].Evidence == "" || report.Gates[0].EvidenceSHA256 == "" {
			t.Fatalf("load failure evidence missing: %#v", report.Gates[0])
		}
		if len(fx.requests) != 1 || fx.requests[0].Path != "/v1/models" {
			t.Fatalf("subsequent gates ran after load blowup: %#v", fx.requests)
		}
		if err := ValidateSixGateReport(report); err != nil {
			t.Fatalf("ValidateSixGateReport(failed) error = %v", err)
		}
	})

	t.Run("reasoning_blew_up_stops_before_tools", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.reasoningStatus = http.StatusInternalServerError
		srv := fx.server(t)
		evidenceDir := t.TempDir()

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, EvidenceDir: evidenceDir, Now: clock.Now})
		if err != nil {
			t.Fatalf("RunSixGates() error = %v", err)
		}
		if report.Passed || report.AbortedGate != GateReasoning || report.Proof.Gates != "load,chat" {
			t.Fatalf("report = %#v", report)
		}
		if len(report.Gates) != 3 {
			t.Fatalf("got %d gates, want 3 (stopped at reasoning)", len(report.Gates))
		}
		if !report.Gates[2].BlewUp || report.Gates[2].Evidence == "" || report.Gates[2].EvidenceSHA256 == "" {
			t.Fatalf("reasoning blowup evidence = %#v", report.Gates[2])
		}
		// Only /v1/models + chat + reasoning should have been hit.
		if len(fx.requests) != 3 {
			t.Fatalf("expected 3 HTTP calls before abort, got %d", len(fx.requests))
		}
	})

	t.Run("answering_failure_continues_to_remaining_gates", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.reasoningContent = "The value is 3910."
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatalf("RunSixGates() error = %v", err)
		}
		if report.Passed || report.AbortedGate != "" {
			t.Fatalf("report = %#v", report)
		}
		if len(report.Gates) != len(SixGates) {
			t.Fatalf("got %d gates, want all 6 when reasoning answered wrongly", len(report.Gates))
		}
		if report.Proof.Gates != "load,chat,tools,context,speed" {
			t.Fatalf("proof gates = %q", report.Proof.Gates)
		}
		if report.Gates[2].Status != StatusFail || report.Gates[2].Reason != ReasonReasoningMismatch || report.Gates[2].BlewUp {
			t.Fatalf("reasoning gate = %#v", report.Gates[2])
		}
	})
}

func TestSixGatesChatGateContract(t *testing.T) {
	t.Run("rejects_capped_spec", func(t *testing.T) {
		evidenceDir := filepath.Join(t.TempDir(), "must-not-exist")
		_, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
			AnswerCapped:  true,
		}, SixGateOptions{BaseURL: "http://127.0.0.1:1234", EvidenceDir: evidenceDir})
		if err == nil || !strings.Contains(err.Error(), "never be capped") {
			t.Fatalf("error = %v, want uncapped refusal", err)
		}
		if _, statErr := os.Stat(evidenceDir); !os.IsNotExist(statErr) {
			t.Fatalf("evidence dir created on preflight refusal: %v", statErr)
		}
	})

	t.Run("rejects_non_stop_finish_reason", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.chatFinishReason = "length"
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		chat := report.Gates[1]
		if chat.Status != StatusFail || chat.Reason != ReasonFinishReasonNotStop || chat.BlewUp {
			t.Fatalf("chat gate = %#v", chat)
		}
	})

	t.Run("empty_answer_blows_up_and_aborts", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.chatContent = "   "
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		if report.AbortedGate != GateChat || len(report.Gates) != 2 || report.Gates[1].Reason != ReasonEmptyAnswer {
			t.Fatalf("report = %#v", report)
		}
	})
}

func TestSixGatesReasoningGateExactIntegerMembershipAndSeparateField(t *testing.T) {
	t.Run("exact_integer_set_membership", func(t *testing.T) {
		accepted := []string{
			"391",
			"17 * 23 = 391.",
			"The exact product is **391**!",
			"Answer: (391)",
			"17, 23, 391",
		}
		for _, text := range accepted {
			if !ExtractExactIntegers(text)[391] {
				t.Fatalf("ExtractExactIntegers(%q) missed 391: %v", text, ExtractExactIntegers(text))
			}
		}

		rejected := []string{
			"3910",
			"the 391st",
			"391st",
			"1391",
			"391.5",
			"391.0",
			".391",
			"-391",
			"1,391",
			"391,000",
			"v391",
			"391_0",
		}
		for _, text := range rejected {
			if ExtractExactIntegers(text)[391] {
				t.Fatalf("ExtractExactIntegers(%q) falsely accepted 391: %v", text, ExtractExactIntegers(text))
			}
		}
	})

	cases := []struct {
		name       string
		field      string
		reasoning  string
		content    string
		wantStatus Status
		wantReason string
	}{
		{
			name:       "rejects_3910",
			field:      "reasoning_content",
			reasoning:  "steps",
			content:    "3910",
			wantStatus: StatusFail,
			wantReason: ReasonReasoningMismatch,
		},
		{
			name:       "rejects_391st",
			field:      "reasoning_content",
			reasoning:  "steps",
			content:    "the 391st",
			wantStatus: StatusFail,
			wantReason: ReasonReasoningMismatch,
		},
		{
			name:       "rejects_inline_thinking_without_separate_field",
			field:      "",
			reasoning:  "",
			content:    "<think>17*23 = 391</think> 391",
			wantStatus: StatusFail,
			wantReason: ReasonReasoningFieldMissing,
		},
		{
			name:       "accepts_reasoning_field_alias",
			field:      "reasoning",
			reasoning:  "17*23 = 391",
			content:    "17 * 23 = 391.",
			wantStatus: StatusPass,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clock := newFakeClock()
			fx := newPassingFixture(clock)
			fx.reasoningField = tc.field
			fx.reasoningValue = tc.reasoning
			fx.reasoningContent = tc.content
			srv := fx.server(t)

			report, err := RunSixGates(context.Background(), SixGateSpec{
				Model:         "Qwen/Qwen3.8-27B",
				ContextWindow: 1000,
			}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
			if err != nil {
				t.Fatal(err)
			}
			got := report.Gates[2]
			if got.Status != tc.wantStatus || got.Reason != tc.wantReason {
				t.Fatalf("reasoning gate = %#v, want status=%s reason=%q", got, tc.wantStatus, tc.wantReason)
			}
		})
	}
}

func TestSixGatesToolsGateFreshHexNonceConsumption(t *testing.T) {
	t.Run("fresh_nonces_across_runs", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		srv := fx.server(t)

		r1, err := RunSixGates(context.Background(), SixGateSpec{Model: "Qwen/Qwen3.8-27B", ContextWindow: 1000}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		r2, err := RunSixGates(context.Background(), SixGateSpec{Model: "Qwen/Qwen3.8-27B", ContextWindow: 1000}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		n1 := r1.Gates[3].ToolNonce
		n2 := r2.Gates[3].ToolNonce
		if n1 == "" || n2 == "" || n1 == n2 {
			t.Fatalf("expected distinct fresh hex nonces, got %q and %q", n1, n2)
		}
	})

	cases := []struct {
		name       string
		mutate     func(*sixGateFixture)
		wantReason string
	}{
		{
			name:       "tool_not_called",
			mutate:     func(fx *sixGateFixture) { fx.toolSkipCall = true },
			wantReason: ReasonToolNotCalled,
		},
		{
			name:       "wrong_tool_name",
			mutate:     func(fx *sixGateFixture) { fx.toolCallName = "other_tool" },
			wantReason: ReasonToolNotCalled,
		},
		{
			name:       "wrong_city_arg",
			mutate:     func(fx *sixGateFixture) { fx.toolCallCity = "London" },
			wantReason: ReasonToolArgMismatch,
		},
		{
			name:       "nonce_not_echoed_in_followup",
			mutate:     func(fx *sixGateFixture) { fx.toolOmitNonce = true },
			wantReason: ReasonToolNonceMissing,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clock := newFakeClock()
			fx := newPassingFixture(clock)
			tc.mutate(fx)
			srv := fx.server(t)

			report, err := RunSixGates(context.Background(), SixGateSpec{
				Model:         "Qwen/Qwen3.8-27B",
				ContextWindow: 1000,
			}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
			if err != nil {
				t.Fatal(err)
			}
			got := report.Gates[3]
			if got.Status != StatusFail || got.Reason != tc.wantReason {
				t.Fatalf("tools gate = %#v, want reason %q", got, tc.wantReason)
			}
		})
	}
}

func TestSixGatesContextGateDepthFloorAndPrefixCachePrefillSuppression(t *testing.T) {
	t.Run("rejects_low_prompt_tokens", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.contextPromptTokens = 599 // < 0.6 * 1000
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		got := report.Gates[4]
		if got.Status != StatusFail || got.Reason != ReasonContextTokensLow {
			t.Fatalf("context gate = %#v, want %s", got, ReasonContextTokensLow)
		}
	})

	t.Run("rejects_missing_planted_code", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.contextOmitCode = true
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		got := report.Gates[4]
		if got.Status != StatusFail || got.Reason != ReasonContextCodeMissing {
			t.Fatalf("context gate = %#v, want %s", got, ReasonContextCodeMissing)
		}
	})

	t.Run("dropped_connection_retry_from_prefix_cache_suppresses_prefill", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.contextDropCount = 2
		fx.contextDropEOF = true
		fx.contextCached = true
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		if !report.Passed {
			t.Fatalf("expected pass on retry, got %#v", report)
		}
		ctxGate := report.Gates[4]
		if len(ctxGate.DropSeconds) != 2 || ctxGate.DropSeconds[0] <= 0 {
			t.Fatalf("drop_seconds = %#v, want 2 recorded drops", ctxGate.DropSeconds)
		}
		if !ctxGate.PrefillSuppressed || !ctxGate.PrefixCached || ctxGate.PrefillTokensPerSecond != nil {
			t.Fatalf("expected prefill suppressed on prefix cache retry: %#v", ctxGate)
		}
		if report.Proof.PrefillTokensPerSecond != nil {
			t.Fatalf("proof must omit cached prefill rate, got %v", *report.Proof.PrefillTokensPerSecond)
		}
		rawProof, err := json.Marshal(report.Proof)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(rawProof), `"prefill"`) {
			t.Fatalf("serialized proof must omit prefill key when suppressed: %s", rawProof)
		}
		if err := ValidateSixGateReport(report); err != nil {
			t.Fatalf("ValidateSixGateReport error = %v", err)
		}
	})
}

func TestSixGatesSpeedGateTokenizerSourceAndSpeculativeDraftExclusion(t *testing.T) {
	t.Run("rejects_char_counted_rate", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.speedSource = SourceChars
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		spd := report.Gates[5]
		if spd.Status != StatusFail || spd.Reason != ReasonSpeedSourceInvalid || report.Passed {
			t.Fatalf("speed gate = %#v", spd)
		}
	})

	t.Run("rejects_omitted_usage_no_char_fallback", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.speedSource = ""
		fx.speedOmitUsage = true
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		spd := report.Gates[5]
		if spd.Status != StatusFail || spd.Reason != ReasonSpeedSourceInvalid || spd.Source != SourceChars {
			t.Fatalf("speed gate = %#v", spd)
		}
	})

	t.Run("rejects_below_floor", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.speedCompletion = 20
		fx.speedChunkCount = 3
		fx.speedPerChunkAdvance = 1 * time.Second // 20 / 2s = 10 tok/s < 15
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		spd := report.Gates[5]
		if spd.Status != StatusFail || spd.Reason != ReasonSpeedBelowFloor || spd.DecodeTokensPerSecond != 10.0 {
			t.Fatalf("speed gate = %#v", spd)
		}
	})

	t.Run("speculative_drafts_not_counted_as_accepted_fails_floor", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.speedCompletion = 40 // server inflated completion_tokens with drafts
		fx.speedDraftTokens = 40
		fx.speedAcceptedTokens = 10
		fx.speedRejectedTokens = 30
		fx.speedChunkCount = 2
		fx.speedPerChunkAdvance = 1 * time.Second // 10 accepted / 1s = 10 < 15
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		spd := report.Gates[5]
		if spd.Status != StatusFail || spd.Reason != ReasonSpeedBelowFloor || spd.CountedTokens != 10 || spd.DecodeTokensPerSecond != 10.0 {
			t.Fatalf("speed gate counted drafts instead of accepted tokens: %#v", spd)
		}
	})

	t.Run("speculative_accepted_tokens_clear_floor_without_drafts", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.speedCompletion = 40
		fx.speedDraftTokens = 40
		fx.speedAcceptedTokens = 20
		fx.speedRejectedTokens = 20
		fx.speedChunkCount = 2
		fx.speedPerChunkAdvance = 1 * time.Second // 20 accepted / 1s = 20 >= 15
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		spd := report.Gates[5]
		if !report.Passed || spd.Status != StatusPass || spd.CountedTokens != 20 || spd.DecodeTokensPerSecond != 20.0 {
			t.Fatalf("speed gate = %#v", spd)
		}
	})

	t.Run("speculative_drafts_without_accepted_count_rejected", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		fx.speedCompletion = 40
		fx.speedDraftTokens = 40
		fx.speedAcceptedTokens = 0
		srv := fx.server(t)

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{BaseURL: srv.URL, Now: clock.Now})
		if err != nil {
			t.Fatal(err)
		}
		spd := report.Gates[5]
		if spd.Status != StatusFail || spd.Reason != ReasonSpeedDraftsCounted {
			t.Fatalf("speed gate = %#v", spd)
		}
	})

	t.Run("stream_stops_at_30s_window", func(t *testing.T) {
		clock := newFakeClock()
		fx := newPassingFixture(clock)
		srv := fx.server(t)

		var chunksRead int
		rt := &steppedSSETransport{
			base:  srv.Client().Transport,
			clock: clock,
			step:  10 * time.Second,
			chunks: []string{
				"data: {\"choices\":[{\"delta\":{\"content\":\"a \"}}],\"usage\":{\"completion_tokens\":100,\"source\":\"tokenizer\"}}\n\n",
				"data: {\"choices\":[{\"delta\":{\"content\":\"b \"}}],\"usage\":{\"completion_tokens\":200,\"source\":\"tokenizer\"}}\n\n",
				"data: {\"choices\":[{\"delta\":{\"content\":\"c \"}}],\"usage\":{\"completion_tokens\":400,\"source\":\"tokenizer\"}}\n\n",
				"data: {\"choices\":[{\"delta\":{\"content\":\"d \"}}],\"usage\":{\"completion_tokens\":600,\"source\":\"tokenizer\"}}\n\n",
				"data: {\"choices\":[{\"delta\":{\"content\":\"e \"}}],\"usage\":{\"completion_tokens\":800,\"source\":\"tokenizer\"}}\n\n",
				"data: {\"choices\":[{\"delta\":{\"content\":\"f \"}}],\"usage\":{\"completion_tokens\":1000,\"source\":\"tokenizer\"}}\n\n",
				"data: [DONE]\n\n",
			},
			onChunk: func() { chunksRead++ },
		}

		report, err := RunSixGates(context.Background(), SixGateSpec{
			Model:         "Qwen/Qwen3.8-27B",
			ContextWindow: 1000,
		}, SixGateOptions{
			BaseURL:     srv.URL,
			Client:      &http.Client{Transport: rt},
			Now:         clock.Now,
			SpeedWindow: 30 * time.Second,
		})
		if err != nil {
			t.Fatal(err)
		}
		spd := report.Gates[5]
		if !report.Passed || spd.Status != StatusPass || spd.DecodeTokensPerSecond != 20.0 || spd.CountedTokens != 600 {
			t.Fatalf("speed gate = %#v", spd)
		}
		if chunksRead != 4 {
			t.Fatalf("chunksRead = %d, want 4 (cutoff at 30s window before chunks 5-6)", chunksRead)
		}
	})
}

type steppedSSETransport struct {
	base    http.RoundTripper
	clock   *fakeClock
	step    time.Duration
	chunks  []string
	onChunk func()
}

func (t *steppedSSETransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Path == "/v1/chat/completions" {
		raw, _ := io.ReadAll(req.Body)
		req.Body = io.NopCloser(bytes.NewReader(raw))
		if bytes.Contains(raw, []byte(`"stream":true`)) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body: &steppedSSEReader{
					clock:   t.clock,
					step:    t.step,
					chunks:  append([]string(nil), t.chunks...),
					onChunk: t.onChunk,
				},
			}, nil
		}
	}
	return t.base.RoundTrip(req)
}

type steppedSSEReader struct {
	clock    *fakeClock
	step     time.Duration
	chunks   []string
	index    int
	advanced int
	onChunk  func()
}

func (r *steppedSSEReader) Read(p []byte) (int, error) {
	if r.index >= len(r.chunks) {
		return 0, io.EOF
	}
	if r.index > 0 && r.advanced < r.index {
		r.clock.Advance(r.step)
		r.advanced = r.index
	}
	n := copy(p, r.chunks[r.index])
	r.chunks[r.index] = r.chunks[r.index][n:]
	if len(r.chunks[r.index]) == 0 {
		r.index++
		if r.onChunk != nil {
			r.onChunk()
		}
	}
	return n, nil
}

func (r *steppedSSEReader) Close() error { return nil }

func TestValidateSixGateReportAndMatrixIntegration(t *testing.T) {
	baseReport := func() SixGateReport {
		tps := 25.0
		prefill := 1200.0
		expected := int64(391)
		return SixGateReport{
			Schema: SixGateSchema,
			Spec: SixGateSpec{
				Model:                     "Qwen/Qwen3.8-27B",
				ContextWindow:             1000,
				RuntimeRevision:           "r1",
				SpeedFloorTokensPerSecond: 15.0,
			},
			Passed: true,
			Proof: SixGateProof{
				Gates:                  "load,chat,reasoning,tools,context,speed",
				DecodeTokensPerSecond:  &tps,
				PrefillTokensPerSecond: &prefill,
			},
			Gates: []GateResult{
				{Gate: GateLoad, Status: StatusPass, EvidenceSHA256: "h1"},
				{Gate: GateChat, Status: StatusPass, FinishReason: "stop", EvidenceSHA256: "h2"},
				{Gate: GateReasoning, Status: StatusPass, SeparateReasoning: true, ExpectedInteger: &expected, EvidenceSHA256: "h3"},
				{Gate: GateTools, Status: StatusPass, ToolNonce: "abc123", NonceMatched: true, EvidenceSHA256: "h4"},
				{Gate: GateContext, Status: StatusPass, PromptTokens: 600, CodeRecovered: true, PrefillTokensPerSecond: &prefill, EvidenceSHA256: "h5"},
				{Gate: GateSpeed, Status: StatusPass, Source: SourceTokenizer, DecodeTokensPerSecond: 25.0, FloorTokensPerSecond: 15.0, CountedTokens: 50, EvidenceSHA256: "h6"},
			},
		}
	}

	if err := ValidateSixGateReport(baseReport()); err != nil {
		t.Fatalf("ValidateSixGateReport(valid) error = %v", err)
	}

	rejections := []struct {
		name   string
		mutate func(*SixGateReport)
		want   string
	}{
		{
			name: "wrong_order",
			mutate: func(r *SixGateReport) {
				r.Gates[1].Gate, r.Gates[2].Gate = r.Gates[2].Gate, r.Gates[1].Gate
			},
			want: "want \"chat\"",
		},
		{
			name: "executed_after_blowup",
			mutate: func(r *SixGateReport) {
				r.Passed = false
				r.Gates[1].Status = StatusFail
				r.Gates[1].BlewUp = true
			},
			want: "after non-answering failure",
		},
		{
			name: "capped_spec",
			mutate: func(r *SixGateReport) {
				r.Spec.AnswerCapped = true
			},
			want: "never cap answers",
		},
		{
			name: "char_counted_speed",
			mutate: func(r *SixGateReport) {
				r.Gates[5].Source = SourceChars
			},
			want: "source \"chars\"",
		},
		{
			name: "cached_prefill_published",
			mutate: func(r *SixGateReport) {
				r.Gates[4].PrefixCached = true
			},
			want: "suppress prefill rate",
		},
		{
			name: "speculative_drafts_counted",
			mutate: func(r *SixGateReport) {
				r.Gates[5].DraftTokens = 40
				r.Gates[5].AcceptedTokens = 20
				r.Gates[5].RejectedTokens = 20
				r.Gates[5].CountedTokens = 40
			},
			want: "speculative drafts",
		},
	}

	for _, tc := range rejections {
		t.Run(tc.name, func(t *testing.T) {
			r := baseReport()
			tc.mutate(&r)
			err := ValidateSixGateReport(r)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidateSixGateReport() error = %v, want containing %q", err, tc.want)
			}
		})
	}

	t.Run("matrix_rejects_failing_six_gate_on_supported_envelope", func(t *testing.T) {
		m := validMatrix()
		r := baseReport()
		r.Passed = false
		r.Gates[5].Status = StatusFail
		r.Proof.Gates = "load,chat,reasoning,tools,context"
		m.Envelopes[0].SixGate = &r
		if err := Validate(m); err == nil || !strings.Contains(err.Error(), "failed six-gate acceptance") {
			t.Fatalf("Validate() error = %v", err)
		}
	})

	t.Run("matrix_rejects_passing_six_gate_on_unsupported_envelope", func(t *testing.T) {
		m := validMatrix()
		r := baseReport()
		m.Envelopes[1].SixGate = &r
		if err := Validate(m); err == nil || !strings.Contains(err.Error(), "must not carry passing six-gate report") {
			t.Fatalf("Validate() error = %v", err)
		}
	})
}
