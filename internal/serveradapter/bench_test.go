package serveradapter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

var (
	benchSinkProbeResult ProbeResult
	benchSinkInvocation  Invocation
	benchSinkErr         error
	benchSinkArgv        []string
	benchSinkStatusCode  int
	benchSinkBody        []byte
)

type benchMockDoer struct {
	modelAlias string
}

func (m *benchMockDoer) Do(req *http.Request) (*http.Response, error) {
	var (
		status = http.StatusOK
		body   string
	)
	switch req.URL.Path {
	case "/health":
		body = `{"status":"ok"}`
	case "/v1/models":
		body = `{"object":"list","data":[{"id":"` + m.modelAlias + `","object":"model"}]}`
	case "/v1/chat/completions":
		body = `{"object":"chat.completion","model":"` + m.modelAlias + `","choices":[{"message":{"role":"assistant","content":"OK"}}]}`
	default:
		status = http.StatusNotFound
		body = `{"error":"not found"}`
	}

	resp := &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    req,
	}
	resp.Header.Set("Content-Type", "application/json")
	return resp, nil
}

type benchNotReadyDoer struct{}

func (m *benchNotReadyDoer) Do(req *http.Request) (*http.Response, error) {
	resp := &http.Response{
		StatusCode: http.StatusServiceUnavailable,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"Loading model"}}`)),
		Request:    req,
	}
	resp.Header.Set("Content-Type", "application/json")
	return resp, nil
}

// BenchmarkProbeLlamaServer_MockDoer measures end-to-end probing across all 3
// protocol stages (health, models, chat) using an in-process HTTPDoer.
// It exercises request creation, header configuration, JSON decoding,
// SHA-256 body digests, capability aggregation, and final probe digest hashing.
func BenchmarkProbeLlamaServer_MockDoer(b *testing.B) {
	ctx := context.Background()
	client := &benchMockDoer{modelAlias: "bench-model"}
	target := ProbeTarget{
		BaseURL:    "http://127.0.0.1:8080",
		ModelAlias: "bench-model",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := ProbeLlamaServer(ctx, client, target)
		if err != nil || !res.Ready {
			b.Fatalf("ProbeLlamaServer failed: %v", err)
		}
		benchSinkProbeResult = res
	}
}

// BenchmarkProbeLlamaServer_HTTPLoopback measures end-to-end probing over a live
// loopback HTTP test server, exercising network transport, wire HTTP header parsing,
// and connection pooling.
func BenchmarkProbeLlamaServer_HTTPLoopback(b *testing.B) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"bench-model","object":"model"}]}`))
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"chat.completion","model":"bench-model","choices":[{"message":{"role":"assistant","content":"OK"}}]}`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	ctx := context.Background()
	client := server.Client()
	target := ProbeTarget{
		BaseURL:    server.URL,
		ModelAlias: "bench-model",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := ProbeLlamaServer(ctx, client, target)
		if err != nil || !res.Ready {
			b.Fatalf("ProbeLlamaServer failed: %v", err)
		}
		benchSinkProbeResult = res
	}
}

// BenchmarkProbeLlamaServer_FailureNotReady measures early failure detection when the
// initial health check returns HTTP 503 (model loading), exercising failProbe and ProbeError.
func BenchmarkProbeLlamaServer_FailureNotReady(b *testing.B) {
	ctx := context.Background()
	client := &benchNotReadyDoer{}
	target := ProbeTarget{
		BaseURL:    "http://127.0.0.1:8080",
		ModelAlias: "bench-model",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := ProbeLlamaServer(ctx, client, target)
		if err == nil || res.Ready {
			b.Fatalf("expected probe failure, got ready: %v", err)
		}
		benchSinkProbeResult = res
		benchSinkErr = err
	}
}

// BenchmarkMakeProbeRequest_HeaderAndParsing measures low-level request formation,
// header setting ("Accept" and "Content-Type"), and response body reading with bounded limit.
func BenchmarkMakeProbeRequest_HeaderAndParsing(b *testing.B) {
	ctx := context.Background()
	client := &benchMockDoer{modelAlias: "bench-model"}
	body := []byte(`{"model":"bench-model","messages":[{"role":"user","content":"ping"}],"max_tokens":1}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		code, raw, err := makeProbeRequest(ctx, client, http.MethodPost, "http://127.0.0.1:8080/v1/chat/completions", body)
		if err != nil || code != http.StatusOK {
			b.Fatalf("makeProbeRequest: code=%d, err=%v", code, err)
		}
		benchSinkStatusCode = code
		benchSinkBody = raw
	}
}

// BenchmarkValidateProbeTarget measures URL header/origin parsing and loopback validation.
func BenchmarkValidateProbeTarget(b *testing.B) {
	targets := []ProbeTarget{
		{BaseURL: "http://127.0.0.1:8080", ModelAlias: "llama-3-8b-instruct"},
		{BaseURL: "http://[::1]:9090", ModelAlias: "qwen-2.5-coder"},
		{BaseURL: "http://127.0.0.1:52959", ModelAlias: "deepseek-r1-q4_k_m"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, target := range targets {
			if err := validateProbeTarget(target); err != nil {
				b.Fatalf("validateProbeTarget: %v", err)
			}
		}
	}
}

// BenchmarkDecodeObject measures response body JSON validation and unmarshaling.
func BenchmarkDecodeObject(b *testing.B) {
	healthJSON := []byte(`{"status":"ok"}`)
	modelsJSON := []byte(`{"object":"list","data":[{"id":"bench-model","object":"model"}]}`)
	chatJSON := []byte(`{"object":"chat.completion","model":"bench-model","choices":[{"message":{"role":"assistant","content":"OK"}}]}`)

	var (
		health struct {
			Status string `json:"status"`
		}
		models struct {
			Object string `json:"object"`
			Data   []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		chat struct {
			Object  string `json:"object"`
			Model   string `json:"model"`
			Choices []struct {
				Message *struct {
					Role    string          `json:"role"`
					Content json.RawMessage `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
	)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := decodeObject(healthJSON, &health); err != nil {
			b.Fatalf("decode health: %v", err)
		}
		if err := decodeObject(modelsJSON, &models); err != nil {
			b.Fatalf("decode models: %v", err)
		}
		if err := decodeObject(chatJSON, &chat); err != nil {
			b.Fatalf("decode chat: %v", err)
		}
	}
}

// BenchmarkFinishProbeDigest measures evidence formatting and SHA-256 checksum calculation.
func BenchmarkFinishProbeDigest(b *testing.B) {
	baseResult := ProbeResult{
		Schema:       ProbeSchema,
		BaseURL:      "http://127.0.0.1:8080",
		ModelAlias:   "bench-model",
		Ready:        true,
		Capabilities: []Capability{FeatureHealth, FeatureModelList, FeatureChat},
		Observations: []ProbeObservation{
			observation(ProbeHealth, http.MethodGet, "/health", http.StatusOK, []byte(`{"status":"ok"}`)),
			observation(ProbeModels, http.MethodGet, "/v1/models", http.StatusOK, []byte(`{"object":"list","data":[{"id":"bench-model"}]}`)),
			observation(ProbeChat, http.MethodPost, "/v1/chat/completions", http.StatusOK, []byte(`{"object":"chat.completion"}`)),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := finishProbe(baseResult, "")
		if res.ProbeDigest == "" {
			b.Fatal("empty probe digest")
		}
		benchSinkProbeResult = res
	}
}

// BenchmarkNewLlamaInvocation measures command contract validation and CLI flag rendering.
func BenchmarkNewLlamaInvocation(b *testing.B) {
	modelPath := filepath.Clean(filepath.Join(b.TempDir(), "bench-model.gguf"))
	execPath := filepath.Clean(filepath.Join(b.TempDir(), "llama-server"))

	identity := ExecutableIdentity{
		Adapter:       AdapterLlamaServer,
		Path:          execPath,
		Version:       "version: 1.0.0",
		VersionDigest: "sha256:" + strings.Repeat("a", 64),
	}
	spec := InvocationSpec{
		ModelPath:   modelPath,
		ModelAlias:  "bench-model",
		Port:        8080,
		TokenWindow: 8192,
		Threads:     8,
		GPULayers:   33,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inv, err := NewLlamaInvocation(identity, spec)
		if err != nil {
			b.Fatalf("NewLlamaInvocation: %v", err)
		}
		benchSinkInvocation = inv
	}
}

// BenchmarkInvocationArgv measures defensive copying of direct-process argument vectors.
func BenchmarkInvocationArgv(b *testing.B) {
	inv := Invocation{
		Executable: "llama-server",
		Args: []string{
			"--model", "/opt/models/model.gguf",
			"--alias", "bench-model",
			"--host", "127.0.0.1",
			"--port", "8080",
			"--ctx-size", "8192",
			"--threads", "8",
			"--n-gpu-layers", "33",
		},
		BaseURL:    "http://127.0.0.1:8080",
		ModelAlias: "bench-model",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		argv := inv.Argv()
		if len(argv) != len(inv.Args)+1 {
			b.Fatal("invalid argv length")
		}
		benchSinkArgv = argv
	}
}
