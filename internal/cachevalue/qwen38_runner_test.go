package cachevalue

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestQwen38ColdArmRunnerDerivesRawReceipt(t *testing.T) {
	const (
		model  = "qwen38-native-exact"
		apiEnv = "FAK_TEST_QWEN38_COLD_KEY"
	)
	t.Setenv(apiEnv, "secret-that-must-not-enter-the-receipt")

	var modelCalls, chatCalls atomic.Int32
	var capturedRequest json.RawMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer secret-that-must-not-enter-the-receipt" {
			t.Errorf("Authorization = %q", got)
		}
		switch r.URL.Path {
		case "/healthz":
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "engine": "inkernel", "planner": "inkernel", "model": model})
		case "/v1/models":
			modelCalls.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{map[string]any{"id": model, "owned_by": "fak"}}})
		case "/v1/chat/completions":
			chatCalls.Add(1)
			var err error
			capturedRequest, err = json.RawMessage(strings.Clone(readAllString(t, r))).MarshalJSON()
			if err != nil {
				t.Fatalf("capture request: %v", err)
			}
			// Keep a measurable transport interval without injecting a clock or wall time.
			time.Sleep(5 * time.Millisecond)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"chatcmpl-live","object":"chat.completion","model":"qwen38-native-exact","choices":[{"message":{"role":"assistant","content":"  measured answer  "},"finish_reason":"stop"}],"usage":{"prompt_tokens":41,"completion_tokens":3,"total_tokens":44,"prompt_tokens_details":{"cached_tokens":7}}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	runner := Qwen38ColdArmRunner{Client: srv.Client()}
	receipt, err := runner.Run(context.Background(), Qwen38ColdArmConfig{
		Endpoint:  srv.URL,
		Model:     model,
		APIKeyEnv: apiEnv,
		Trial: Qwen38ColdArmTrial{
			ID: "cold-1", Prompt: "Return a measured answer.", MaxTokens: 32,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if modelCalls.Load() != 1 || chatCalls.Load() != 1 {
		t.Fatalf("requests: models=%d chat=%d, want 1 each", modelCalls.Load(), chatCalls.Load())
	}
	if receipt.Schema != Qwen38ColdArmReceiptSchema || receipt.Evidence.Scope != "partial-cold-arm" || receipt.Evidence.PerformanceClaim {
		t.Fatalf("schema/evidence = %q %+v", receipt.Schema, receipt.Evidence)
	}
	if receipt.Endpoint != srv.URL || receipt.Model != model || receipt.APIKeyEnv != apiEnv || receipt.TrialID != "cold-1" {
		t.Fatalf("receipt identity = %+v", receipt)
	}
	if receipt.Engine != "inkernel" || receipt.Planner != "inkernel" || receipt.ModelOwner != "fak" {
		t.Fatalf("native endpoint identity = engine=%q planner=%q owner=%q", receipt.Engine, receipt.Planner, receipt.ModelOwner)
	}
	if receipt.Command.Method != http.MethodPost || receipt.Command.Path != "/v1/chat/completions" || receipt.HTTPStatus != http.StatusOK {
		t.Fatalf("command/status = %+v HTTP %d", receipt.Command, receipt.HTTPStatus)
	}
	if receipt.WallMS < 5 {
		t.Fatalf("wall_ms = %g, want measured server delay", receipt.WallMS)
	}
	if receipt.Usage.PromptTokens == nil || *receipt.Usage.PromptTokens != 41 ||
		receipt.Usage.CompletionTokens == nil || *receipt.Usage.CompletionTokens != 3 ||
		receipt.Usage.TotalTokens == nil || *receipt.Usage.TotalTokens != 44 ||
		receipt.Usage.CachedTokens == nil || *receipt.Usage.CachedTokens != 7 {
		t.Fatalf("usage = %+v", receipt.Usage)
	}
	sum := sha256.Sum256([]byte("measured answer"))
	if want := "sha256:" + hex.EncodeToString(sum[:]); receipt.OutputSHA256 != want {
		t.Fatalf("output hash = %q, want %q", receipt.OutputSHA256, want)
	}
	if receipt.Memory.Status != "N/A" || receipt.Energy.Status != "N/A" || receipt.Memory.Reason == "" || receipt.Energy.Reason == "" {
		t.Fatalf("typed unavailable resources = memory:%+v energy:%+v", receipt.Memory, receipt.Energy)
	}
	if !json.Valid(receipt.RawRequest) || !json.Valid(receipt.RawResponse) || !reflect.DeepEqual(json.RawMessage(capturedRequest), receipt.RawRequest) {
		t.Fatalf("raw exchange missing or altered: request=%s response=%s", receipt.RawRequest, receipt.RawResponse)
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "secret-that-must-not-enter-the-receipt") {
		t.Fatal("API key leaked into raw receipt")
	}

	// Config owns only declarations and workload input. Measurements are deliberately absent,
	// so a caller cannot smuggle prepared observations into a purported runner receipt.
	for i := 0; i < reflect.TypeOf(Qwen38ColdArmConfig{}).NumField(); i++ {
		name := strings.ToLower(reflect.TypeOf(Qwen38ColdArmConfig{}).Field(i).Name)
		for _, forbidden := range []string{"wall", "usage", "token", "hash", "memory", "energy", "response"} {
			if strings.Contains(name, forbidden) {
				t.Fatalf("config exposes derived observation field %q", name)
			}
		}
	}
}

func TestQwen38ColdArmRunnerLiveNativeMetal(t *testing.T) {
	endpoint := strings.TrimSpace(os.Getenv("FAK_QWEN38_LIVE_ENDPOINT"))
	if endpoint == "" {
		t.Skip("set FAK_QWEN38_LIVE_ENDPOINT to exercise one real fak-native cold arm")
	}
	model := strings.TrimSpace(os.Getenv("FAK_QWEN38_LIVE_MODEL"))
	if model == "" {
		model = Qwen38DefaultAlias
	}
	apiKeyEnv := strings.TrimSpace(os.Getenv("FAK_QWEN38_LIVE_API_KEY_ENV"))
	if apiKeyEnv == "" {
		apiKeyEnv = "FAK_QWEN38_LIVE_API_KEY"
	}
	receipt, err := (Qwen38ColdArmRunner{}).Run(context.Background(), Qwen38ColdArmConfig{
		Endpoint: endpoint, Model: model, APIKeyEnv: apiKeyEnv,
		Trial: Qwen38ColdArmTrial{ID: "live-cold-1", Prompt: "Reply with exactly Q38.", MaxTokens: 8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Engine != "inkernel" || receipt.Planner != "inkernel" || receipt.ModelOwner != "fak" {
		t.Fatalf("live execution identity = engine=%q planner=%q owner=%q", receipt.Engine, receipt.Planner, receipt.ModelOwner)
	}
	value := func(n *int64) any {
		if n == nil {
			return "N/A"
		}
		return *n
	}
	t.Logf("live partial receipt: schema=%s model=%s engine=%s planner=%s status=%d wall_ms=%.3f prompt_tokens=%v completion_tokens=%v total_tokens=%v cached_tokens=%v output_sha256=%s performance_claim=%v",
		receipt.Schema, receipt.Model, receipt.Engine, receipt.Planner, receipt.HTTPStatus,
		receipt.WallMS, value(receipt.Usage.PromptTokens), value(receipt.Usage.CompletionTokens),
		value(receipt.Usage.TotalTokens), value(receipt.Usage.CachedTokens), receipt.OutputSHA256,
		receipt.Evidence.PerformanceClaim)
}

func TestQwen38ColdArmRunnerFailsClosedOnEndpointIdentity(t *testing.T) {
	t.Setenv("FAK_TEST_QWEN38_IDENTITY_KEY", "secret")
	tests := []struct {
		name    string
		health  map[string]any
		models  map[string]any
		wantErr string
	}{
		{
			name:    "wrong advertised model",
			health:  map[string]any{"ok": true, "engine": "inkernel", "planner": "inkernel", "model": "exact"},
			models:  map[string]any{"data": []any{map[string]any{"id": "substitute", "owned_by": "fak"}}},
			wantErr: "exact model",
		},
		{
			name:    "proxy fallback",
			health:  map[string]any{"ok": true, "engine": "inkernel", "planner": "proxy", "model": "exact"},
			models:  map[string]any{"data": []any{map[string]any{"id": "exact", "owned_by": "fak"}}},
			wantErr: "fak-native",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var chatCalls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/healthz":
					_ = json.NewEncoder(w).Encode(tc.health)
				case "/v1/models":
					_ = json.NewEncoder(w).Encode(tc.models)
				case "/v1/chat/completions":
					chatCalls.Add(1)
					_ = json.NewEncoder(w).Encode(map[string]any{})
				}
			}))
			defer srv.Close()
			_, err := (Qwen38ColdArmRunner{Client: srv.Client()}).Run(context.Background(), Qwen38ColdArmConfig{
				Endpoint: srv.URL, Model: "exact", APIKeyEnv: "FAK_TEST_QWEN38_IDENTITY_KEY",
				Trial: Qwen38ColdArmTrial{ID: "cold-1", Prompt: "hello", MaxTokens: 8},
			})
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want %q", err, tc.wantErr)
			}
			if chatCalls.Load() != 0 {
				t.Fatalf("chat requests = %d, want none before exact native identity", chatCalls.Load())
			}
		})
	}
}

func readAllString(t *testing.T, r *http.Request) string {
	t.Helper()
	var body json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	return string(body)
}
