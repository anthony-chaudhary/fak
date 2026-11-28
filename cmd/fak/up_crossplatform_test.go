package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type stubNamedBackend struct {
	compute.Backend
	name string
}

func (b *stubNamedBackend) Name() string { return b.name }

func TestUpCrossPlatformDryRunPlan(t *testing.T) {
	var stdout, stderr bytes.Buffer
	runTurnkeyUp(strings.NewReader(""), &stdout, &stderr, []string{"--dry-run", "--memory-gib", "16"})
	if stderr.Len() != 0 {
		t.Fatalf("unexpected dry-run stderr: %s", stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak up — Turnkey Execution Plan", "System Memory", "7B"} {
		if !strings.Contains(out, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "Apple Silicon") {
		t.Fatalf("dry-run output should be platform-neutral, got:\n%s", out)
	}

	stdout.Reset()
	stderr.Reset()
	runTurnkeyUp(strings.NewReader(""), &stdout, &stderr, []string{"--dry-run", "--json", "--memory-gib", "16"})
	if stderr.Len() != 0 {
		t.Fatalf("unexpected json dry-run stderr: %s", stderr.String())
	}
	var plan macfit.TurnkeyProfile
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("unmarshal dry-run json: %v\n%s", err, stdout.String())
	}
	if plan.Tier.Name != "7B" || plan.ContextBudgetTokens == 0 {
		t.Fatalf("unexpected dry-run plan: %+v", plan)
	}
}

func TestUpCrossPlatformBackendStamp(t *testing.T) {
	cases := []struct {
		backend string
		want    string
	}{
		{backend: "cuda", want: "fak up: in-kernel chat forward: CUDA GPU (backend=cuda)"},
		{backend: "vulkan", want: "fak up: in-kernel chat forward: Vulkan GPU (backend=vulkan)"},
		{backend: "cpu", want: "fak up: in-kernel chat forward: CPU (backend=cpu)"},
		{backend: "metal", want: ""},
		{backend: "", want: ""},
	}
	for _, tc := range cases {
		var buf bytes.Buffer
		printTurnkeyComputeBackendStamp(&buf, tc.backend)
		got := strings.TrimSpace(buf.String())
		if got != tc.want {
			t.Fatalf("printTurnkeyComputeBackendStamp(%q) = %q, want %q", tc.backend, got, tc.want)
		}
	}

	for _, name := range []string{"cuda", "vulkan", "cpu"} {
		t.Run("native-startup-"+name, func(t *testing.T) {
			deps := turnkeyNativeLoadDeps{
				resolveBackend: func() (compute.Backend, error) {
					if name == "cpu" {
						return nil, nil
					}
					return &stubNamedBackend{Backend: compute.Default(), name: name}, nil
				},
				resolveMetal: func() (serveMetalDecision, error) {
					return serveMetalDecision{skippedBecause: skipReasonNotCompiled}, nil
				},
				refusePeak: func(string) error { return nil },
				admitAndLoad: func(_ bool, _ string, load func(), _ *serveFitBudget) (func(), error) {
					load()
					return func() {}, nil
				},
				loadModel: func(string, compute.Backend, int, *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile) {
					return &fakmodel.Model{}, false, nil
				},
				loadTokenizer: func(string) (*tokenizer.Tokenizer, bool) {
					return &tokenizer.Tokenizer{}, true
				},
				newPlanner: func(*fakmodel.Model, *tokenizer.Tokenizer, string, bool, compute.Backend, bool, int) *agent.InKernelPlanner {
					return nil
				},
			}
			res, err := loadTurnkeyNativeResourcesWith(context.Background(), t.TempDir(), name, 2048, deps)
			if err != nil {
				t.Fatalf("loadTurnkeyNativeResourcesWith(%s): %v", name, err)
			}
			defer res.Close()
			if res.Startup.BackendName != name {
				t.Fatalf("BackendName = %q, want %q", res.Startup.BackendName, name)
			}
			if runtime.GOOS != "darwin" && res.Startup.MetalDecision.skippedBecause != "" {
				t.Fatalf("non-darwin MetalDecision.skippedBecause = %q, want empty", res.Startup.MetalDecision.skippedBecause)
			}
		})
	}
}

func TestUpCrossPlatformInkling2BackendAdmitsCUDA(t *testing.T) {
	origGate := turnkeyInkling2Servable
	t.Cleanup(func() { turnkeyInkling2Servable = origGate })
	turnkeyInkling2Servable = func() error { return errors.New("metal required") }

	got, err := resolveTurnkeyInkling2Backend("linux", "cuda", nil)
	if err != nil || got != "cuda" {
		t.Fatalf("resolveTurnkeyInkling2Backend(linux, cuda) = (%q, %v), want (\"cuda\", nil)", got, err)
	}
	if _, err := resolveTurnkeyInkling2Backend("linux", "vulkan", nil); err == nil {
		t.Fatal("resolveTurnkeyInkling2Backend(linux, vulkan) expected refusal via gate, got nil")
	}
	if _, err := resolveTurnkeyInkling2Backend("darwin", "cuda", nil); err == nil {
		t.Fatal("resolveTurnkeyInkling2Backend(darwin, cuda) expected refusal via gate, got nil")
	}
}

func TestUpCrossPlatformMockServer(t *testing.T) {
	plan, err := macfit.ConfigureTurnkeyWithOptions(16*macfit.GiB, macfit.TurnkeyOptions{StaticOnly: true})
	if err != nil {
		t.Fatalf("ConfigureTurnkeyWithOptions: %v", err)
	}
	srv, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", true)
	if err != nil {
		t.Fatalf("startTurnkeyServer(mock): %v", err)
	}
	defer srv.Close()

	base := "http://" + srv.Addr()
	for _, path := range []string{"/healthz", "/readyz", "/v1/models"} {
		resp, err := http.Get(base + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("GET %s status = %d, want 200", path, resp.StatusCode)
		}
	}

	reqBody := `{"model":"` + plan.Tier.ModelID + `","messages":[{"role":"user","content":"hello"}],"stream":false}`
	resp, err := http.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions status = %d: %s", resp.StatusCode, raw)
	}
	if strings.Contains(string(raw), "Apple Silicon") {
		t.Fatalf("mock chat response should be platform-neutral, got: %s", raw)
	}
}

func TestUpCrossPlatformNonTierAliasOverride(t *testing.T) {
	var stdout, stderr bytes.Buffer
	runTurnkeyUp(strings.NewReader(""), &stdout, &stderr, []string{"--dry-run", "--model", "smollm2"})
	if stderr.Len() != 0 {
		t.Fatalf("unexpected dry-run stderr for --model smollm2: %s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "smollm2") {
		t.Fatalf("dry-run output missing smollm2:\n%s", stdout.String())
	}
}
