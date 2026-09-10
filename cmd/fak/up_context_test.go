package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func TestUpModelsAdvertiseKnownNativeContext(t *testing.T) {
	const contextTokens = 2048
	plan := macfit.TurnkeyProfile{
		Tier:                macfit.ModelTier{ModelID: "native-up"},
		ContextBudgetTokens: contextTokens,
	}
	planner := agent.NewInKernelPlannerWithConfig(
		&model.Model{Cfg: model.Config{MaxPositionEmbeddings: 4096}},
		testProbeTokenizer(t),
		plan.Tier.ModelID,
		false,
		nil,
		false,
		agent.InKernelPlannerConfig{ContextTokens: contextTokens},
	)
	server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false, planner)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Shutdown(context.Background())

	got := fetchUpModels(t, server)
	if len(got.Data) != 1 || got.Data[0].ID != plan.Tier.ModelID {
		t.Fatalf("models = %+v, want native model %q", got.Data, plan.Tier.ModelID)
	}
	if got.Data[0].ContextLength == nil || *got.Data[0].ContextLength != contextTokens {
		t.Fatalf("native context_length = %v, want %d", got.Data[0].ContextLength, contextTokens)
	}
}

func TestUpModelsClampNominalPlanToActualNativeContext(t *testing.T) {
	plan := macfit.TurnkeyProfile{
		Tier:                macfit.ModelTier{ModelID: "native-up"},
		ContextBudgetTokens: 4096,
	}
	planner := agent.NewInKernelPlannerWithConfig(
		&model.Model{Cfg: model.Config{MaxPositionEmbeddings: 2048}},
		testProbeTokenizer(t),
		plan.Tier.ModelID,
		false,
		nil,
		false,
		agent.InKernelPlannerConfig{ContextTokens: 4096},
	)
	server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false, planner)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Shutdown(context.Background())

	if got := server.Plan().ContextBudgetTokens; got != 2048 {
		t.Fatalf("server plan context = %d, want actual planner clamp 2048", got)
	}
	got := fetchUpModels(t, server)
	if len(got.Data) != 1 || got.Data[0].ContextLength == nil || *got.Data[0].ContextLength != 2048 {
		t.Fatalf("catalog context = %+v, want actual planner clamp 2048", got.Data)
	}
}

func TestUpModelsOmitUnknownAndMockContext(t *testing.T) {
	tests := []struct {
		name    string
		mock    bool
		planner *agent.InKernelPlanner
	}{
		{name: "mock", mock: true},
		{
			name: "unknown native",
			planner: agent.NewInKernelPlanner(
				&model.Model{}, testProbeTokenizer(t), "unknown-up", false, nil, false,
			),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan := macfit.TurnkeyProfile{Tier: macfit.ModelTier{ModelID: "unknown-up"}}
			server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", tt.mock, tt.planner)
			if err != nil {
				t.Fatal(err)
			}
			defer server.Shutdown(context.Background())
			got := fetchUpModels(t, server)
			if len(got.Data) != 1 {
				t.Fatalf("models = %+v, want one row", got.Data)
			}
			if got.Data[0].ContextLength != nil {
				t.Fatalf("unknown %s context_length = %d, want omitted", tt.name, *got.Data[0].ContextLength)
			}
		})
	}
}

func TestTurnkeyContextTokensCheckedConversion(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, want := range []int{0, 1, maxInt} {
		got, err := turnkeyContextTokens(uint64(want))
		if err != nil {
			t.Fatalf("turnkeyContextTokens(%d): %v", want, err)
		}
		if got != want {
			t.Fatalf("turnkeyContextTokens(%d) = %d, want unchanged", want, got)
		}
	}
	if got, err := turnkeyContextTokens(uint64(maxInt) + 1); err == nil {
		t.Fatalf("turnkeyContextTokens(MaxInt+1) = %d, nil; want overflow refusal", got)
	}
}

func TestTurnkeyPlanContextReachesPlanner(t *testing.T) {
	const contextTokens = 3072
	p := newTurnkeyInKernelPlanner(
		&model.Model{Cfg: model.Config{MaxPositionEmbeddings: 8192}},
		testProbeTokenizer(t),
		"native-up",
		false,
		nil,
		false,
		contextTokens,
	)
	if got := p.ContextWindow(); got != contextTokens {
		t.Fatalf("turnkey planner ContextWindow() = %d, want plan cap %d", got, contextTokens)
	}
	if got := p.RuntimeConfig().ContextTokens; got != contextTokens {
		t.Fatalf("turnkey planner configured context = %d, want %d", got, contextTokens)
	}
}

func TestUpContextBoundaryAndOneOverHTTP(t *testing.T) {
	tok := testProbeTokenizer(t)
	cfg := upContextSyntheticConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	messages := []chatCompletionMessage{{Role: "user", Content: "hi"}}
	ids, err := tok.Encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
	if err != nil {
		t.Fatalf("encode boundary prompt: %v", err)
	}
	contextTokens := len(ids) + 1
	m.Cfg.MaxPositionEmbeddings = contextTokens
	planner := newTurnkeyInKernelPlanner(m, tok, "native-up", false, nil, false, contextTokens)
	plan := macfit.TurnkeyProfile{
		Tier:                macfit.ModelTier{ModelID: "native-up"},
		ContextBudgetTokens: uint64(contextTokens),
	}
	server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false, planner)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Shutdown(context.Background())

	request := chatCompletionRequest{Model: "native-up", Messages: messages, MaxTokens: 1}
	if status, _ := postUpChat(t, server, request); status != http.StatusOK {
		t.Fatalf("exact context boundary status = %d, want 200", status)
	}
	request.MaxTokens = 2
	status, raw := postUpChat(t, server, request)
	if status != http.StatusBadRequest {
		t.Fatalf("one-over context status = %d, want 400; body=%s", status, raw)
	}
	var envelope struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatalf("decode error envelope: %v; body=%s", err, raw)
	}
	if envelope.Error.Code != "context_length_exceeded" {
		t.Fatalf("error code = %q, want context_length_exceeded; body=%s", envelope.Error.Code, raw)
	}
}

func TestUpNonContextInferenceErrorRemainsInternalServerError(t *testing.T) {
	plan := macfit.TurnkeyProfile{
		Tier:                macfit.ModelTier{ModelID: "native-up"},
		ContextBudgetTokens: 4096,
	}
	// The empty tokenizer fails before native execution with an ordinary encoding
	// error. It must not be mislabeled as a client context-length error.
	planner := agent.NewInKernelPlannerWithConfig(
		&model.Model{Cfg: model.Config{MaxPositionEmbeddings: 4096}},
		&tokenizer.Tokenizer{},
		plan.Tier.ModelID,
		false,
		nil,
		false,
		agent.InKernelPlannerConfig{ContextTokens: 4096},
	)
	server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false, planner)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Shutdown(context.Background())

	status, _ := postUpChat(t, server, chatCompletionRequest{
		Model:     plan.Tier.ModelID,
		Messages:  []chatCompletionMessage{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	})
	if status != http.StatusInternalServerError {
		t.Fatalf("ordinary inference error status = %d, want 500", status)
	}
}

type upModelsResponse struct {
	Data []struct {
		ID            string `json:"id"`
		ContextLength *int   `json:"context_length"`
	} `json:"data"`
}

func fetchUpModels(t *testing.T, server *turnkeyServer) upModelsResponse {
	t.Helper()
	resp, err := http.Get("http://" + server.Addr() + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got upModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	return got
}

func postUpChat(t *testing.T, server *turnkeyServer, request chatCompletionRequest) (int, []byte) {
	t.Helper()
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post("http://"+server.Addr()+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, raw
}

func upContextSyntheticConfig() model.Config {
	return model.Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         320,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}
