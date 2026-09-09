package gateway

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

func TestConfiguredInKernelPlannerEnablesSelectedMetalMTP(t *testing.T) {
	t.Setenv("FAK_SPECULATIVE", " mtp ")
	planner := newInKernelChatPlanner(Config{
		InKernelModel: model.NewSynthetic(model.Config{
			ModelType:  "qwen3_5_text",
			LayerTypes: []string{"linear_attention", "full_attention"},
		}),
		Metal: true,
	}, "qwen38-metal-mtp", t.Logf)
	inKernel := planner.(*agent.InKernelPlanner)
	if inKernel.MetalMTPCoordinator() == nil {
		t.Fatal("FAK_SPECULATIVE=mtp did not enable the Metal MTP coordinator")
	}
	t.Cleanup(inKernel.DisableMetalMTP)
}

func TestConfiguredInKernelPlannerLeavesIneligiblePathsUnchanged(t *testing.T) {
	hybrid := func() *model.Model {
		return model.NewSynthetic(model.Config{
			ModelType:  "qwen3_5_text",
			LayerTypes: []string{"linear_attention", "full_attention"},
		})
	}
	tests := []struct {
		name    string
		selectM string
		cfg     Config
	}{
		{name: "unselected", selectM: "", cfg: Config{InKernelModel: hybrid(), Metal: true}},
		{name: "non-metal", selectM: "mtp", cfg: Config{InKernelModel: hybrid()}},
		{name: "device-backend-is-not-metal-session", selectM: "mtp", cfg: Config{InKernelModel: hybrid(), Metal: true, Backend: compute.Default()}},
		{name: "non-hybrid", selectM: "mtp", cfg: Config{InKernelModel: model.NewSynthetic(model.Config{
			ModelType:        "qwen3_5_text",
			LayerTypes:       []string{"full_attention"},
			HiddenSize:       64,
			IntermediateSize: 128,
			VocabSize:        32,
			NumHeads:         4,
			NumKVHeads:       2,
			HeadDim:          16,
			NumLayers:        1,
		}), Metal: true}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("FAK_SPECULATIVE", tt.selectM)
			planner := newInKernelChatPlanner(tt.cfg, "ineligible", t.Logf)
			if got := planner.(*agent.InKernelPlanner).MetalMTPCoordinator(); got != nil {
				t.Fatalf("ineligible path installed Metal MTP coordinator: %#v", got)
			}
		})
	}
}

func TestChatCompletionsMetalMTPSpeculativeHeaderAndDispatch(t *testing.T) {
	abi.RegisterEngine(modelengine.EngineID, modelengine.Default)
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "4")

	cfg := gatewayQwen35MetalReceiptConfig()
	cfg.MTPNumHiddenLayers = 1
	m := model.NewSynthetic(cfg)
	m.Quantize()

	srv, err := New(Config{
		InKernelModel: m,
		Tokenizer:     newByteLevelTokenizer(t),
		Model:         "qwen38-metal",
		Metal:         true,
		MetalMTP:      true,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	coord := srv.MetalMTPCoordinator()
	if coord == nil {
		t.Fatal("expected non-nil MetalMTPCoordinator on server")
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	payload := `{"model":"qwen38-metal","messages":[{"role":"user","content":"HI"}],"temperature":0}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body = %s", resp.StatusCode, string(body))
	}

	if got := resp.Header.Get(HeaderSpeculative); got != SpeculativeMTPMetal {
		t.Fatalf("header %s = %q, want %q", HeaderSpeculative, got, SpeculativeMTPMetal)
	}
	if got := resp.Header.Get("x-fak-speculative"); got != "mtp-metal" {
		t.Fatalf("header x-fak-speculative = %q, want %q", got, "mtp-metal")
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		t.Fatalf("failed to decode response JSON: %v", err)
	}
	if len(chatResp.Choices) == 0 {
		t.Fatal("expected at least one choice in chat response")
	}

	stats := coord.Stats()
	if stats.TotalGenerated == 0 {
		t.Fatal("expected MetalMTPCoordinator TotalGenerated > 0 after completions dispatch")
	}
}

func TestChatCompletionsMetalMTPStreamingHeaderAndDispatch(t *testing.T) {
	abi.RegisterEngine(modelengine.EngineID, modelengine.Default)
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "4")

	cfg := gatewayQwen35MetalReceiptConfig()
	cfg.MTPNumHiddenLayers = 1
	m := model.NewSynthetic(cfg)
	m.Quantize()

	srv, err := New(Config{
		InKernelModel: m,
		Tokenizer:     newByteLevelTokenizer(t),
		Model:         "qwen38-metal",
		Metal:         true,
		MetalMTP:      true,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	payload := `{"model":"qwen38-metal","messages":[{"role":"user","content":"HI"}],"temperature":0,"stream":true}`
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(payload))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions stream: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body = %s", resp.StatusCode, string(body))
	}

	if got := resp.Header.Get(HeaderSpeculative); got != SpeculativeMTPMetal {
		t.Fatalf("stream header %s = %q, want %q", HeaderSpeculative, got, SpeculativeMTPMetal)
	}

	scanner := bufio.NewScanner(resp.Body)
	var chunks []string
	sawDone := false
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if strings.TrimSpace(data) == "[DONE]" {
				sawDone = true
				break
			}
			chunks = append(chunks, data)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("reading stream: %v", err)
	}
	if !sawDone {
		t.Fatal("stream did not end with [DONE]")
	}
	if len(chunks) == 0 {
		t.Fatal("expected stream chunks before [DONE]")
	}
}

func TestChatCompletionsWithoutMetalMTPDoesNotEmitHeader(t *testing.T) {
	abi.RegisterEngine(modelengine.EngineID, modelengine.Default)
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "2")

	cfg := gatewayQwen35MetalReceiptConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()

	srv, err := New(Config{
		InKernelModel:   m,
		Tokenizer:       newByteLevelTokenizer(t),
		Model:           "non-mtp-model",
		Metal:           false,
		DisableMetalMTP: true,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	payload := `{"model":"non-mtp-model","messages":[{"role":"user","content":"hi"}],"temperature":0}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()

	if got := resp.Header.Get(HeaderSpeculative); got != "" {
		t.Fatalf("expected empty header %s, got %q", HeaderSpeculative, got)
	}
}

func TestDualPlannerMetalMTPHeaderRouting(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "4")

	m := model.NewSyntheticQwen38MTP()
	m.Quantize()

	localPlanner := agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "local", false, nil, false)
	coord, err := model.NewMetalMTPCoordinator(nil, model.DefaultMetalMTPConfig())
	if err != nil {
		t.Fatalf("NewMetalMTPCoordinator: %v", err)
	}
	localPlanner.SetMetalMTPCoordinator(coord)

	proxyPlanner := agent.NewMockPlanner("remote-model")
	dual, err := NewDualPlanner(proxyPlanner, localPlanner, "local")
	if err != nil {
		t.Fatalf("NewDualPlanner: %v", err)
	}

	srv := &Server{
		model:   "remote-model",
		planner: dual,
	}

	if !srv.isMetalMTPActive("local") {
		t.Fatal("expected isMetalMTPActive('local') == true")
	}
	if srv.isMetalMTPActive("remote-model") {
		t.Fatal("expected isMetalMTPActive('remote-model') == false")
	}
}
