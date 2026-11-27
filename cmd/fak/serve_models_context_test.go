package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type stubInkling2LoopGen struct{}

func (stubInkling2LoopGen) Generate(context.Context, []int, int, map[int]bool, func(int) bool) (agent.Inkling2LoopGeneration, error) {
	return agent.Inkling2LoopGeneration{Generated: 1, Stopped: true}, nil
}

func newTestInkling2Tokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"model":   map[string]any{"type": "BPE", "vocab": map[string]int{"a": 0}, "merges": []string{}},
		"decoder": map[string]any{"type": "ByteLevel"},
		"added_tokens": []map[string]any{
			{"id": 1, "content": agent.Inkling2EndOfTurnMarker, "special": true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.ParseJSON(raw)
	if err != nil {
		t.Fatalf("tokenizer.ParseJSON: %v", err)
	}
	return tok
}

// fak-test:runtime fast est=1s lane=default
func TestServeModelsAdvertisesResolvedContextLengthAndKVDtype(t *testing.T) {
	tok := newTestInkling2Tokenizer(t)

	t.Run("inkling2 planner and serve status advertise context_length and kv_dtype", func(t *testing.T) {
		const wantCtx = 131072
		planner, err := newServeInkling2LoopPlanner(stubInkling2LoopGen{}, tok, "inkling2-mini", wantCtx)
		if err != nil {
			t.Fatalf("newServeInkling2LoopPlanner: %v", err)
		}
		env := ggufload.Inkling2ServeEnvelope{
			MaxCtx:  wantCtx,
			KVDtype: ggufload.Inkling2KVFP8,
		}
		srv, err := gateway.New(gateway.Config{
			Model:               "inkling2-mini",
			NativeChatPlanner:   planner,
			ServeStatusProvider: newInkling2ServeStatusProvider("inkling2-mini", env, nil),
		})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}

		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		req.RemoteAddr = "127.0.0.1:12345"
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /v1/models status = %d, want 200; body=%s", rec.Code, rec.Body.String())
		}

		var body struct {
			Data   []map[string]any `json:"data"`
			Models []map[string]any `json:"models"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("unmarshal /v1/models: %v", err)
		}
		if len(body.Data) != 1 {
			t.Fatalf("data len = %d, want 1; body=%s", len(body.Data), rec.Body.String())
		}
		row := body.Data[0]
		if got, ok := row["context_length"].(float64); !ok || int(got) != wantCtx {
			t.Fatalf("data[0].context_length = %v, want %d", row["context_length"], wantCtx)
		}
		if got, ok := row["kv_dtype"].(string); !ok || got != "fp8" {
			t.Fatalf("data[0].kv_dtype = %v, want %q", row["kv_dtype"], "fp8")
		}
		if len(body.Models) != 1 {
			t.Fatalf("models len = %d, want 1", len(body.Models))
		}
		if got, ok := body.Models[0]["context_window"].(float64); !ok || int(got) != wantCtx {
			t.Fatalf("models[0].context_window = %v, want %d", body.Models[0]["context_window"], wantCtx)
		}
	})

	t.Run("zero context omits context_length and kv_dtype cleanly", func(t *testing.T) {
		planner, err := newServeInkling2LoopPlanner(stubInkling2LoopGen{}, tok, "inkling2-mini", 0)
		if err != nil {
			t.Fatalf("newServeInkling2LoopPlanner: %v", err)
		}
		srv, err := gateway.New(gateway.Config{
			Model:             "inkling2-mini",
			NativeChatPlanner: planner,
		})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}

		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		req.RemoteAddr = "127.0.0.1:12345"
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /v1/models status = %d, want 200; body=%s", rec.Code, rec.Body.String())
		}

		var body struct {
			Data []map[string]any `json:"data"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("unmarshal /v1/models: %v", err)
		}
		if len(body.Data) != 1 {
			t.Fatalf("data len = %d, want 1", len(body.Data))
		}
		row := body.Data[0]
		if _, exists := row["context_length"]; exists {
			t.Fatalf("unexpected context_length present when 0: %v", row["context_length"])
		}
		if _, exists := row["kv_dtype"]; exists {
			t.Fatalf("unexpected kv_dtype present when unset: %v", row["kv_dtype"])
		}
	})
}
