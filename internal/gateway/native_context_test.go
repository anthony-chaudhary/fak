package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type modelCatalogResponse struct {
	Data []struct {
		ID            string `json:"id"`
		ContextLength *int   `json:"context_length"`
	} `json:"data"`
	Models []struct {
		Slug             string `json:"slug"`
		ContextWindow    *int   `json:"context_window"`
		MaxContextWindow *int   `json:"max_context_window"`
	} `json:"models"`
}

func TestModelsAdvertiseDeclaredNativeContextWindow(t *testing.T) {
	const declaredWindow = 4096
	p := agent.NewInKernelPlanner(
		model.NewSynthetic(model.Config{MaxPositionEmbeddings: declaredWindow}),
		&tokenizer.Tokenizer{},
		"native-qwen",
		false,
		nil,
		false,
	)
	s := &Server{model: "native-qwen", planner: p}

	got := fetchModelCatalog(t, s)
	assertCatalogContext(t, got, "native-qwen", declaredWindow)
}

func TestModelsAdvertiseContextOnlyForKnownNativeRows(t *testing.T) {
	local := agent.NewInKernelPlannerWithConfig(
		model.NewSynthetic(model.Config{MaxPositionEmbeddings: 4096}),
		&tokenizer.Tokenizer{},
		"native-qwen",
		false,
		nil,
		false,
		agent.InKernelPlannerConfig{ContextTokens: 2048},
	)
	dual, err := NewDualPlanner(stubPlanner{}, local, "native-qwen")
	if err != nil {
		t.Fatal(err)
	}
	got := fetchModelCatalog(t, &Server{model: "remote-api", planner: dual})
	assertCatalogContext(t, got, "native-qwen", 2048)
	assertCatalogHasNoContext(t, got, "remote-api")

	unknown := agent.NewInKernelPlanner(&model.Model{}, &tokenizer.Tokenizer{}, "unknown-native", false, nil, false)
	assertCatalogHasNoContext(t, fetchModelCatalog(t, &Server{model: "unknown-native", planner: unknown}), "unknown-native")
}

func TestNativeContextBoundaryAndHTTPErrorLoopback(t *testing.T) {
	tok := newByteLevelTokenizer(t)
	cfg := kvmmuSynthCfg()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := agent.NewInKernelPlanner(m, tok, "native-loopback", false, nil, false)
	messages := []agent.Message{{Role: agent.RoleUser, Content: "hi"}}
	ids, err := tok.Encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
	if err != nil {
		t.Fatalf("encode boundary prompt: %v", err)
	}
	m.Cfg.MaxPositionEmbeddings = len(ids) + 1

	s := newTestServer(t)
	s.model = "native-loopback"
	s.planner = p
	tsURL := httptest.NewServer(s.Handler())
	defer tsURL.Close()

	request := map[string]any{"model": "native-loopback", "messages": messages, "max_tokens": 1}
	if status := postJSON(t, tsURL.URL+"/v1/chat/completions", request, nil); status != http.StatusOK {
		t.Fatalf("exact context boundary HTTP status = %d, want 200", status)
	}

	request["max_tokens"] = 2
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(tsURL.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("one-over context HTTP status = %d, want 400; body=%s", resp.StatusCode, raw)
	}
	var envelope struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatalf("decode context error: %v; body=%s", err, raw)
	}
	if envelope.Error.Code != "context_length_exceeded" {
		t.Fatalf("context error code = %q, want context_length_exceeded; body=%s", envelope.Error.Code, raw)
	}
}

func fetchModelCatalog(t *testing.T, s *Server) modelCatalogResponse {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	s.handleModels(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/models status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var got modelCatalogResponse
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatalf("decode /v1/models: %v", err)
	}
	return got
}

func assertCatalogContext(t *testing.T, got modelCatalogResponse, modelID string, want int) {
	t.Helper()
	for _, row := range got.Data {
		if row.ID == modelID {
			if row.ContextLength == nil || *row.ContextLength != want {
				t.Fatalf("standard catalog %q context_length = %v, want %d", modelID, row.ContextLength, want)
			}
			goto codex
		}
	}
	t.Fatalf("standard catalog missing %q: %+v", modelID, got.Data)

codex:
	for _, row := range got.Models {
		if row.Slug == modelID {
			if row.ContextWindow == nil || *row.ContextWindow != want {
				t.Fatalf("Codex catalog %q context_window = %v, want %d", modelID, row.ContextWindow, want)
			}
			if row.MaxContextWindow == nil || *row.MaxContextWindow != want {
				t.Fatalf("Codex catalog %q max_context_window = %v, want %d", modelID, row.MaxContextWindow, want)
			}
			return
		}
	}
	t.Fatalf("Codex catalog missing %q: %+v", modelID, got.Models)
}

func assertCatalogHasNoContext(t *testing.T, got modelCatalogResponse, modelID string) {
	t.Helper()
	for _, row := range got.Data {
		if row.ID == modelID && row.ContextLength != nil {
			t.Fatalf("standard catalog %q fabricated context_length %d", modelID, *row.ContextLength)
		}
	}
	for _, row := range got.Models {
		if row.Slug == modelID && (row.ContextWindow != nil || row.MaxContextWindow != nil) {
			t.Fatalf("Codex catalog %q fabricated context fields: %+v", modelID, row)
		}
	}
}
