package serveradapter_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/serveradapter"
)

func TestExternalAdapterContract(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fake, capture := buildFakeLlamaServer(t, ctx)
	identity, err := serveradapter.InspectExecutable(ctx, fake)
	if err != nil {
		t.Fatalf("inspect executable: %v", err)
	}
	if identity.Adapter != serveradapter.AdapterLlamaServer || identity.Path != fake || identity.Version == "" || !strings.HasPrefix(identity.VersionDigest, "sha256:") {
		t.Fatalf("unexpected executable identity: %+v", identity)
	}

	requests := make([]string, 0, 3)
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, r.Method+" "+r.URL.Path)
		writeJSON(t, w, http.StatusOK, `{"status":"ok"}`)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, r.Method+" "+r.URL.Path)
		writeJSON(t, w, http.StatusOK, `{"object":"list","data":[{"id":"code-model","object":"model"}]}`)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, r.Method+" "+r.URL.Path)
		var request struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
			MaxTokens int  `json:"max_tokens"`
			Stream    bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode chat request: %v", err)
		}
		if request.Model != "code-model" || len(request.Messages) != 1 || request.Messages[0].Role != "user" || request.MaxTokens != 1 || request.Stream {
			t.Errorf("unexpected chat probe: %+v", request)
		}
		writeJSON(t, w, http.StatusOK, `{"object":"chat.completion","model":"code-model","choices":[{"message":{"role":"assistant","content":"OK"}}]}`)
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	port := serverPort(t, server.URL)

	modelPath := filepath.Clean(filepath.Join(t.TempDir(), "model $(touch owned); name.gguf"))
	invocation, err := serveradapter.NewLlamaInvocation(identity, serveradapter.InvocationSpec{
		ModelPath:   modelPath,
		ModelAlias:  "code-model",
		Port:        port,
		TokenWindow: 8192,
		Threads:     8,
		GPULayers:   64,
	})
	if err != nil {
		t.Fatalf("render invocation: %v", err)
	}
	wantArgs := []string{
		"--model", modelPath,
		"--alias", "code-model",
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--ctx-size", "8192",
		"--threads", "8",
		"--n-gpu-layers", "64",
	}
	if invocation.Executable != fake || !reflect.DeepEqual(invocation.Args, wantArgs) {
		t.Fatalf("rendered invocation = %#v, want executable %q args %#v", invocation, fake, wantArgs)
	}
	if strings.Contains(strings.Join(invocation.Args, " "), "0.0.0.0") || invocation.BaseURL != server.URL {
		t.Fatalf("invocation is not bound to the fixture loopback origin: %+v", invocation)
	}

	cmd := exec.CommandContext(ctx, invocation.Executable, invocation.Args...)
	cmd.Env = append(os.Environ(), "ARGV_CAPTURE="+capture)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("execute external argv fixture: %v (%s)", err, output)
	}
	var captured []string
	readJSON(t, capture, &captured)
	if !reflect.DeepEqual(captured, wantArgs) {
		t.Fatalf("fake executable captured %#v, want %#v", captured, wantArgs)
	}

	result, err := serveradapter.ProbeLlamaServer(ctx, server.Client(), serveradapter.ProbeTarget{
		BaseURL:    invocation.BaseURL,
		ModelAlias: invocation.ModelAlias,
	})
	if err != nil {
		t.Fatalf("probe fixture server: %v", err)
	}
	wantCapabilities := []serveradapter.Capability{
		serveradapter.FeatureHealth,
		serveradapter.FeatureModelList,
		serveradapter.FeatureChat,
	}
	if !result.Ready || !reflect.DeepEqual(result.Capabilities, wantCapabilities) || len(result.Observations) != 3 || !strings.HasPrefix(result.ProbeDigest, "sha256:") {
		t.Fatalf("unexpected probe result: %+v", result)
	}
	wantRequests := []string{"GET /health", "GET /v1/models", "POST /v1/chat/completions"}
	if !reflect.DeepEqual(requests, wantRequests) {
		t.Fatalf("probe order = %#v, want %#v", requests, wantRequests)
	}

	witnessPath := filepath.Join(t.TempDir(), "result.json")
	witness, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatalf("marshal witness: %v", err)
	}
	if err := os.WriteFile(witnessPath, append(witness, '\n'), 0o600); err != nil {
		t.Fatalf("write witness: %v", err)
	}
	var readBack serveradapter.ProbeResult
	readJSON(t, witnessPath, &readBack)
	if !reflect.DeepEqual(readBack, result) {
		t.Fatalf("witness read-back differs: got %+v want %+v", readBack, result)
	}
	t.Logf("independent witness: %s", witness)
}

func TestProbeFailureClassesAndReadinessOrdering(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		healthStatus     int
		healthBody       string
		modelsStatus     int
		modelsBody       string
		chatStatus       int
		chatBody         string
		wantKind         serveradapter.FailureKind
		wantProbe        serveradapter.ProbeKind
		wantRequests     int
		wantCapabilities []serveradapter.Capability
	}{
		{
			name:         "not ready",
			healthStatus: http.StatusServiceUnavailable,
			healthBody:   `{"error":{"message":"Loading model"}}`,
			wantKind:     serveradapter.FailureNotReady,
			wantProbe:    serveradapter.ProbeHealth,
			wantRequests: 1,
		},
		{
			name:             "wrong model",
			healthStatus:     http.StatusOK,
			healthBody:       `{"status":"ok"}`,
			modelsStatus:     http.StatusOK,
			modelsBody:       `{"object":"list","data":[{"id":"other-model"}]}`,
			wantKind:         serveradapter.FailureWrongModel,
			wantProbe:        serveradapter.ProbeModels,
			wantRequests:     2,
			wantCapabilities: []serveradapter.Capability{serveradapter.FeatureHealth, serveradapter.FeatureModelList},
		},
		{
			name:             "unsupported chat",
			healthStatus:     http.StatusOK,
			healthBody:       `{"status":"ok"}`,
			modelsStatus:     http.StatusOK,
			modelsBody:       `{"object":"list","data":[{"id":"code-model"}]}`,
			chatStatus:       http.StatusNotFound,
			chatBody:         `{"error":{"type":"not_supported_error"}}`,
			wantKind:         serveradapter.FailureChatUnavailable,
			wantProbe:        serveradapter.ProbeChat,
			wantRequests:     3,
			wantCapabilities: []serveradapter.Capability{serveradapter.FeatureHealth, serveradapter.FeatureModelList},
		},
		{
			name:             "malformed response",
			healthStatus:     http.StatusOK,
			healthBody:       `{"status":"ok"}`,
			modelsStatus:     http.StatusOK,
			modelsBody:       `not-json`,
			wantKind:         serveradapter.FailureMalformedResponse,
			wantProbe:        serveradapter.ProbeModels,
			wantRequests:     2,
			wantCapabilities: []serveradapter.Capability{serveradapter.FeatureHealth},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			requests := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests++
				switch r.URL.Path {
				case "/health":
					writeJSON(t, w, tt.healthStatus, tt.healthBody)
				case "/v1/models":
					writeJSON(t, w, tt.modelsStatus, tt.modelsBody)
				case "/v1/chat/completions":
					writeJSON(t, w, tt.chatStatus, tt.chatBody)
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			result, err := serveradapter.ProbeLlamaServer(context.Background(), server.Client(), serveradapter.ProbeTarget{BaseURL: server.URL, ModelAlias: "code-model"})
			var probeErr *serveradapter.ProbeError
			if !errors.As(err, &probeErr) || probeErr.Kind != tt.wantKind || probeErr.Probe != tt.wantProbe {
				t.Fatalf("error = %#v, want kind %q probe %q", err, tt.wantKind, tt.wantProbe)
			}
			if result.Ready || requests != tt.wantRequests || !reflect.DeepEqual(result.Capabilities, nonNilCapabilities(tt.wantCapabilities)) || !strings.HasPrefix(result.ProbeDigest, "sha256:") {
				t.Fatalf("failure result = %+v requests=%d", result, requests)
			}
		})
	}
}

func TestProbeClassifiesNotListening(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	baseURL := server.URL
	client := server.Client()
	server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	result, err := serveradapter.ProbeLlamaServer(ctx, client, serveradapter.ProbeTarget{BaseURL: baseURL, ModelAlias: "code-model"})
	var probeErr *serveradapter.ProbeError
	if !errors.As(err, &probeErr) || probeErr.Kind != serveradapter.FailureNotListening || probeErr.Probe != serveradapter.ProbeHealth {
		t.Fatalf("error = %#v, want not-listening health error", err)
	}
	if result.Ready || len(result.Capabilities) != 0 || len(result.Observations) != 1 || result.ProbeDigest == "" {
		t.Fatalf("not-listening result = %+v", result)
	}
}

func TestRenderRejectsOutOfContractOptions(t *testing.T) {
	t.Parallel()
	identity := serveradapter.ExecutableIdentity{
		Adapter:       serveradapter.AdapterLlamaServer,
		Path:          filepath.Join(t.TempDir(), executableName()),
		Version:       "version: 1",
		VersionDigest: "sha256:" + strings.Repeat("a", 64),
	}
	valid := serveradapter.InvocationSpec{
		ModelPath:   filepath.Join(t.TempDir(), "model.gguf"),
		ModelAlias:  "code-model",
		Port:        8080,
		TokenWindow: 4096,
		Threads:     4,
		GPULayers:   0,
	}
	tests := []struct {
		name   string
		mutate func(*serveradapter.InvocationSpec)
	}{
		{"relative model", func(s *serveradapter.InvocationSpec) { s.ModelPath = "model.gguf" }},
		{"wrong extension", func(s *serveradapter.InvocationSpec) { s.ModelPath = filepath.Join(t.TempDir(), "model.bin") }},
		{"unsafe alias", func(s *serveradapter.InvocationSpec) { s.ModelAlias = "--alias,other" }},
		{"zero port", func(s *serveradapter.InvocationSpec) { s.Port = 0 }},
		{"wild port", func(s *serveradapter.InvocationSpec) { s.Port = 65536 }},
		{"small context", func(s *serveradapter.InvocationSpec) { s.TokenWindow = serveradapter.MinTokenWindow - 1 }},
		{"large context", func(s *serveradapter.InvocationSpec) { s.TokenWindow = serveradapter.MaxTokenWindow + 1 }},
		{"zero threads", func(s *serveradapter.InvocationSpec) { s.Threads = 0 }},
		{"too many threads", func(s *serveradapter.InvocationSpec) { s.Threads = serveradapter.MaxThreads + 1 }},
		{"negative GPU layers", func(s *serveradapter.InvocationSpec) { s.GPULayers = -1 }},
		{"too many GPU layers", func(s *serveradapter.InvocationSpec) { s.GPULayers = serveradapter.MaxGPULayers + 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := valid
			tt.mutate(&spec)
			if _, err := serveradapter.NewLlamaInvocation(identity, spec); err == nil {
				t.Fatal("NewLlamaInvocation accepted out-of-contract options")
			}
		})
	}
}

func TestProbeRejectsWildcardOrigin(t *testing.T) {
	t.Parallel()
	result, err := serveradapter.ProbeLlamaServer(context.Background(), http.DefaultClient, serveradapter.ProbeTarget{
		BaseURL:    "http://0.0.0.0:8080",
		ModelAlias: "code-model",
	})
	if err == nil || result.Ready || len(result.Observations) != 0 {
		t.Fatalf("wildcard probe result = %+v err=%v", result, err)
	}
}

func buildFakeLlamaServer(t *testing.T, ctx context.Context) (string, string) {
	t.Helper()
	dir := t.TempDir()
	source := filepath.Join(dir, "main.go")
	program := `package main
import (
    "encoding/json"
    "fmt"
    "os"
)
func main() {
    if len(os.Args) == 2 && os.Args[1] == "--version" {
        fmt.Println("llama.cpp version: b1234 (deadbeef)")
        return
    }
    raw, err := json.Marshal(os.Args[1:])
    if err != nil { panic(err) }
    if err := os.WriteFile(os.Getenv("ARGV_CAPTURE"), raw, 0600); err != nil { panic(err) }
}
`
	if err := os.WriteFile(source, []byte(program), 0o600); err != nil {
		t.Fatalf("write fake source: %v", err)
	}
	executable := filepath.Join(dir, executableName())
	cmd := exec.CommandContext(ctx, "go", "build", "-o", executable, source)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build fake executable: %v (%s)", err, output)
	}
	return executable, filepath.Join(dir, "argv.json")
}

func executableName() string {
	if runtime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

func serverPort(t *testing.T, rawURL string) int {
	t.Helper()
	parsed, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	return port
}

func writeJSON(t *testing.T, w http.ResponseWriter, status int, body string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := fmt.Fprint(w, body); err != nil {
		t.Errorf("write response: %v", err)
	}
}

func readJSON(t *testing.T, path string, out any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

func nonNilCapabilities(in []serveradapter.Capability) []serveradapter.Capability {
	if in == nil {
		return []serveradapter.Capability{}
	}
	return in
}
