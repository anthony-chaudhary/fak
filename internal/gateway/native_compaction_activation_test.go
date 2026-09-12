package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeCompactionActivationWitnessesSuccessfulPlannerWork(t *testing.T) {
	m := model.NewSynthetic(nativeCompactionGatewayModelConfig())
	m.Quantize()
	p := agent.NewInKernelPlanner(
		m, newByteLevelTokenizer(t),
		"native-compaction-gateway", false, nil, false,
	)
	p.SetPromptShrinkLevers(80, false, false)

	srv := &Server{}
	catalog, err := NewFeatureCatalog([]FeatureStatus{{
		Feature: FeatureCompactHistory, State: FeatureConfiguredActive,
		Provenance: FeatureCLIFlag, Description: "native planner compaction witness",
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := srv.SetFeatureCatalog(catalog); err != nil {
		t.Fatal(err)
	}

	handler := srv.withFeatureActivations(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		messages := []agent.Message{{Role: agent.RoleUser, Content: "short"}}
		if r.URL.Query().Get("compact") == "1" {
			messages = nativeCompactionGatewayMessages()
		}
		if _, err := p.Complete(r.Context(), messages, nil, agent.WithMaxTokens(1)); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_, _ = io.WriteString(w, "ok")
	}))

	active := nativeCompactionGatewayRequest(t, handler, "/v1/chat/completions?compact=1")
	if got := active.Header.Get(HeaderFeaturesEnabled); got != string(FeatureCompactHistory) {
		t.Fatalf("active enabled header = %q, want %q", got, FeatureCompactHistory)
	}
	if got := active.Header.Get(HeaderFeaturesUsed); got != string(FeatureCompactHistory) {
		t.Fatalf("active used header = %q, want one %q", got, FeatureCompactHistory)
	}
	if got := active.Trailer.Get(HeaderFeaturesUsedFinal); got != string(FeatureCompactHistory) {
		t.Fatalf("active final trailer = %q, want one %q", got, FeatureCompactHistory)
	}

	idle := nativeCompactionGatewayRequest(t, handler, "/v1/chat/completions")
	if got := idle.Header.Get(HeaderFeaturesEnabled); got != string(FeatureCompactHistory) {
		t.Fatalf("idle enabled header = %q, want %q", got, FeatureCompactHistory)
	}
	if got := idle.Header.Get(HeaderFeaturesUsed); got != "" {
		t.Fatalf("idle request inherited used header: %q", got)
	}
	if got := idle.Trailer.Get(HeaderFeaturesUsedFinal); got != "" {
		t.Fatalf("idle request inherited final trailer: %q", got)
	}

	earlyHandler := srv.withFeatureActivations(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("feature activation writer dropped the underlying Flusher")
			return
		}
		flusher.Flush()
		if _, err := p.Complete(r.Context(), nativeCompactionGatewayMessages(), nil, agent.WithMaxTokens(1)); err != nil {
			t.Errorf("Complete after early header: %v", err)
			return
		}
		_, _ = io.WriteString(w, "ok")
	}))
	for _, target := range []string{"/v1/messages", "/v1/fak/syscall"} {
		t.Run("early header "+target, func(t *testing.T) {
			early := nativeCompactionGatewayRequest(t, earlyHandler, target)
			if got := early.Header.Get(HeaderFeaturesUsed); got != "" {
				t.Fatalf("early used header = %q, want empty pre-completion snapshot", got)
			}
			if got := early.Trailer.Get(HeaderFeaturesUsedFinal); got != string(FeatureCompactHistory) {
				t.Fatalf("early final trailer = %q, want %q from completed native work", got, FeatureCompactHistory)
			}
		})
	}
}

func nativeCompactionGatewayRequest(t *testing.T, handler http.Handler, target string) *http.Response {
	t.Helper()
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, target, nil))
	response := recorder.Result()
	body, err := io.ReadAll(response.Body)
	_ = response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK || string(body) != "ok" {
		t.Fatalf("response = %d %q, want 200 ok", response.StatusCode, body)
	}
	return response
}

func nativeCompactionGatewayMessages() []agent.Message {
	return []agent.Message{
		{Role: agent.RoleSystem, Content: "System prompt invariant instructions."},
		{Role: agent.RoleUser, Content: "Read the file first."},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r1", Type: "function", Function: agent.Func{Name: "Read", Arguments: `{"path":"large.txt"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r1", Content: strings.Repeat("large file line\n", 30)},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "e1", Type: "function", Function: agent.Func{Name: "Edit", Arguments: `{"path":"large.txt"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "e1", Content: "ok"},
		{Role: agent.RoleAssistant, Content: "File edited."},
		{Role: agent.RoleUser, Content: "Middle turn 1"},
		{Role: agent.RoleAssistant, Content: "Middle reply 1"},
		{Role: agent.RoleUser, Content: "Middle turn 2"},
		{Role: agent.RoleAssistant, Content: "Middle reply 2"},
		{Role: agent.RoleUser, Content: "Latest query: what is final state?"},
	}
}

func nativeCompactionGatewayModelConfig() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 320, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
}
