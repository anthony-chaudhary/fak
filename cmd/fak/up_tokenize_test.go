package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestUpNativeTokenize(t *testing.T) {
	const modelID = "qwen3.8-tokenize"
	plan := macfit.TurnkeyProfile{Tier: macfit.ModelTier{ModelID: modelID}, ContextBudgetTokens: 4096}
	planner := agent.NewInKernelPlannerWithConfig(&model.Model{Cfg: model.Config{MaxPositionEmbeddings: 8192}}, testProbeTokenizer(t), modelID, false, nil, false, agent.InKernelPlannerConfig{ContextTokens: 4096})
	server := &turnkeyServer{plan: plan, planner: planner}
	temperature, topP, frequency, presence := 0.25, 0.8, 0.4, 0.2
	req := gateway.ChatRequest{
		Model:     modelID,
		Messages:  []agent.Message{{Role: agent.RoleUser, Content: "Return JSON using the tool."}},
		Tools:     []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "Read", Description: "read", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"}}}`)}}},
		MaxTokens: 99999, Temperature: &temperature, TopP: &topP,
		ToolChoice:     json.RawMessage(`{"type":"function","function":{"name":"Read"}}`),
		ResponseFormat: json.RawMessage(`{"type":"json_object"}`),
		LogitBias:      map[int]float64{7: -2}, FrequencyPenalty: &frequency, PresencePenalty: &presence,
	}
	maxTokens := turnkeyMaxOutputTokens(plan.ContextBudgetTokens)
	want, err := planner.EncodePrompt(context.Background(), req.Messages, req.Tools,
		agent.WithMaxTokens(maxTokens), agent.WithTemperature(req.Temperature), agent.WithTopP(req.TopP),
		agent.WithToolChoice(req.ToolChoice), agent.WithResponseFormat(req.ResponseFormat), agent.WithLogitBias(req.LogitBias),
		agent.WithFrequencyPenalty(req.FrequencyPenalty), agent.WithPresencePenalty(req.PresencePenalty))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(req)
	recorder := httptest.NewRecorder()
	server.handleTokenize(recorder, httptest.NewRequest(http.MethodPost, "/v1/fak/tokenize", bytes.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("tokenize status %d: %s", recorder.Code, recorder.Body.String())
	}
	var got agent.PromptEncoding
	if err := json.Unmarshal(recorder.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokenize did not use chat request sampling semantics:\n got %+v\nwant %+v", got, want)
	}
	if strings.Contains(recorder.Body.String(), "Return JSON using the tool.") || strings.Contains(strings.ToLower(recorder.Body.String()), "rendered_prompt") {
		t.Fatalf("tokenize response disclosed rendered text: %s", recorder.Body.String())
	}
	if got.PromptTokens != len(got.TokenIDs) || got.ContextWindowTokens != 4096 || got.ReservedOutputTokens != maxTokens || got.RenderedSHA256 == "" || got.RendererID == "" || got.TokenizerID == "" {
		t.Fatalf("incomplete exact tokenization receipt: %+v", got)
	}

	t.Run("request contract", func(t *testing.T) {
		cases := []struct {
			name   string
			srv    *turnkeyServer
			method string
			body   []byte
			status int
			code   string
		}{
			{name: "method", srv: server, method: http.MethodGet, body: body, status: http.StatusMethodNotAllowed},
			{name: "malformed", srv: server, method: http.MethodPost, body: []byte("{"), status: http.StatusBadRequest},
			{name: "oversize", srv: server, method: http.MethodPost, body: bytes.Repeat([]byte("x"), 1<<20+1), status: http.StatusBadRequest},
			{name: "mock", srv: &turnkeyServer{mock: true, plan: plan}, method: http.MethodPost, body: body, status: http.StatusNotImplemented},
			{name: "nil planner", srv: &turnkeyServer{plan: plan}, method: http.MethodPost, body: body, status: http.StatusNotImplemented},
			{name: "non native planner", srv: &turnkeyServer{plan: plan, planner: &turnkeyToolRecordingPlanner{}}, method: http.MethodPost, body: body, status: http.StatusNotImplemented},
			{name: "model mismatch", srv: server, method: http.MethodPost, body: []byte(`{"model":"other","messages":[{"role":"user","content":"x"}]}`), status: http.StatusBadRequest, code: "model_mismatch"},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				r := httptest.NewRecorder()
				tc.srv.handleTokenize(r, httptest.NewRequest(tc.method, "/v1/fak/tokenize", bytes.NewReader(tc.body)))
				if r.Code != tc.status || (tc.code != "" && !strings.Contains(r.Body.String(), tc.code)) {
					t.Fatalf("status/body = %d %q, want %d containing %q", r.Code, r.Body.String(), tc.status, tc.code)
				}
			})
		}
	})

	t.Run("context error uses inference mapping", func(t *testing.T) {
		tinyPlan := macfit.TurnkeyProfile{Tier: macfit.ModelTier{ModelID: modelID}, ContextBudgetTokens: 8}
		tinyPlanner := agent.NewInKernelPlannerWithConfig(&model.Model{Cfg: model.Config{MaxPositionEmbeddings: 8}}, testProbeTokenizer(t), modelID, false, nil, false, agent.InKernelPlannerConfig{ContextTokens: 8})
		r := httptest.NewRecorder()
		(&turnkeyServer{plan: tinyPlan, planner: tinyPlanner}).handleTokenize(r, httptest.NewRequest(http.MethodPost, "/v1/fak/tokenize", strings.NewReader(`{"model":"qwen3.8-tokenize","messages":[{"role":"user","content":"this prompt cannot fit"}],"max_tokens":1}`)))
		if r.Code != http.StatusBadRequest || !strings.Contains(r.Body.String(), "context_length") {
			t.Fatalf("context error mapping = %d %q", r.Code, r.Body.String())
		}
	})
}
