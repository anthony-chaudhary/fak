package gateway

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type vulkanMTPGatewayBackend struct {
	compute.Backend
	name, allLogitsPath, rawHiddenPath string
}

func (b vulkanMTPGatewayBackend) Name() string { return b.name }

func (b vulkanMTPGatewayBackend) Qwen35SequenceAllLogitsPath() string {
	return b.allLogitsPath
}

func (vulkanMTPGatewayBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func (vulkanMTPGatewayBackend) Qwen35SequencePrefill(compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	return compute.Qwen35SequencePrefillResult{}, errors.New("selector-only backend must not execute")
}

func (b vulkanMTPGatewayBackend) Qwen35SequenceRawHiddenPath() string {
	return b.rawHiddenPath
}

func (vulkanMTPGatewayBackend) Qwen35MTPDraftPath() string {
	return compute.Qwen35MTPDraftPath
}

func (b vulkanMTPGatewayBackend) Qwen35MTPFuse(compute.Qwen35MTPFuseRequest) (compute.Tensor, error) {
	return compute.Tensor{}, errors.New("selector-only backend must not execute")
}

func (vulkanMTPGatewayBackend) Qwen35SequencePrefixReplayPath() string {
	return compute.Qwen35SequencePrefixReplayPath
}

func admittedVulkanMTPConfig(t *testing.T) Config {
	t.Helper()
	t.Setenv("FAK_SPECULATIVE", "mtp")
	t.Setenv("SPECULATIVE", "")
	return Config{
		InKernelModel: model.NewSyntheticQwen38MTP(),
		InKernelQ4K:   true,
		Backend: vulkanMTPGatewayBackend{
			Backend:       compute.Default(),
			name:          "vulkan",
			allLogitsPath: compute.Qwen35SequenceAllLogitsPath,
			rawHiddenPath: compute.Qwen35SequenceRawHiddenPath,
		},
	}
}

type vulkanMTPGatewayPlanner struct {
	receipt *agent.VulkanMTPExecution
	err     error
	got     agent.SampleParams
}

type vulkanMTPConcurrentPlanner struct {
	entered chan string
	release <-chan struct{}
}

func (*vulkanMTPConcurrentPlanner) Model() string          { return "qwen38-vulkan" }
func (*vulkanMTPConcurrentPlanner) VulkanMTPEnabled() bool { return true }
func (*vulkanMTPConcurrentPlanner) StreamingSupported() bool {
	return true
}

func (p *vulkanMTPConcurrentPlanner) execute(ctx context.Context, messages []agent.Message) (*agent.Completion, error) {
	key := messages[0].Content
	p.entered <- key
	select {
	case <-p.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	receipt := &agent.VulkanMTPExecution{
		Engine:         SpeculativeMTPVulkan,
		Backend:        "vulkan",
		RequestedDepth: 2,
	}
	if key == "success" {
		receipt.Used = true
		receipt.EffectiveDepth = 2
		receipt.ProposalRounds = 1
		receipt.ProposedTokens = 2
		receipt.AcceptedTokens = 1
		receipt.RejectedTokens = 1
		receipt.RollbackTokens = 1
		receipt.TargetOperations = 1
		receipt.RecurrentRepairTokens = 1
	} else {
		receipt.DowngradeReason = "constructor-refused"
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, VulkanMTP: receipt}, nil
}

func (p *vulkanMTPConcurrentPlanner) Complete(ctx context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.execute(ctx, messages)
}

func (p *vulkanMTPConcurrentPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	comp, err := p.execute(ctx, messages)
	if err == nil && sink != nil {
		err = sink("ok")
	}
	return comp, err
}

func (p *vulkanMTPGatewayPlanner) Model() string { return "qwen38-vulkan" }

func (p *vulkanMTPGatewayPlanner) VulkanMTPEnabled() bool { return true }

func (p *vulkanMTPGatewayPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	for _, opt := range opts {
		opt(&p.got)
	}
	return &agent.Completion{
		Message:   agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		VulkanMTP: p.receipt,
	}, p.err
}

func (p *vulkanMTPGatewayPlanner) StreamingSupported() bool { return true }

func (p *vulkanMTPGatewayPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	for _, opt := range opts {
		opt(&p.got)
	}
	if sink != nil {
		_ = sink("ok") // commits the SSE headers before the execution receipt exists
	}
	return &agent.Completion{
		Message:   agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		VulkanMTP: p.receipt,
	}, p.err
}

func TestGatewaySelectsVulkanMTPOnlyForAdmittedEnvelope(t *testing.T) {
	t.Run("configuration admission", func(t *testing.T) {
		base := admittedVulkanMTPConfig(t)
		if !shouldEnableVulkanMTP(base) {
			t.Fatal("eligible retained-head Q4_K Vulkan route was not admitted")
		}
		planner := newInKernelChatPlanner(base, "qwen38-vulkan", t.Logf).(*agent.InKernelPlanner)
		if !planner.VulkanMTPEnabled() {
			t.Fatal("admitted route did not configure request-bound Vulkan MTP")
		}

		for _, tc := range []struct {
			name string
			env  string
			edit func(*Config)
		}{
			{name: "unselected"},
			{name: "non Q4K target", env: "mtp", edit: func(c *Config) { c.InKernelQ4K = false }},
			{name: "missing retained head", env: "mtp", edit: func(c *Config) { c.InKernelModel = gatewayNGramTestModel() }},
			{name: "non Vulkan backend", env: "mtp", edit: func(c *Config) {
				c.Backend = vulkanMTPGatewayBackend{Backend: compute.Default(), name: "cpu-ref", allLogitsPath: compute.Qwen35SequenceAllLogitsPath, rawHiddenPath: compute.Qwen35SequenceRawHiddenPath}
			}},
			{name: "missing all-logits capability", env: "mtp", edit: func(c *Config) { c.Backend = compute.Default() }},
			{name: "wrong raw-hidden capability", env: "mtp", edit: func(c *Config) {
				c.Backend = vulkanMTPGatewayBackend{Backend: compute.Default(), name: "vulkan", allLogitsPath: compute.Qwen35SequenceAllLogitsPath, rawHiddenPath: "stale-contract"}
			}},
			{name: "Metal remains separate", env: "mtp", edit: func(c *Config) { c.Backend = nil; c.InKernelQ4K = false; c.Metal = true }},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Setenv("FAK_SPECULATIVE", tc.env)
				cfg := base
				if tc.edit != nil {
					tc.edit(&cfg)
				}
				if shouldEnableVulkanMTP(cfg) {
					t.Fatal("unsupported route admitted Vulkan MTP")
				}
			})
		}

		t.Setenv("FAK_SPECULATIVE", "ngram")
		if !shouldEnableNGramSpeculative(base) || shouldEnableVulkanMTP(base) {
			t.Fatal("ngram selection did not remain on its existing Vulkan route")
		}
		t.Setenv("FAK_SPECULATIVE", "mtp")
		metal := base
		metal.Backend, metal.InKernelQ4K, metal.Metal = nil, false, true
		if !shouldEnableMetalMTP(metal) || shouldEnableVulkanMTP(metal) {
			t.Fatal("Metal MTP selection did not remain separate from Vulkan MTP")
		}

		t.Setenv("FAK_SPECULATIVE", "mtp")
		t.Setenv("SPECULATIVE", "ngram")
		if shouldEnableVulkanMTP(base) {
			t.Fatal("conflicting selectors admitted the new Vulkan MTP route")
		}
		if !shouldEnableNGramSpeculative(base) {
			t.Fatal("conflicting selectors changed the existing ngram OR semantics")
		}
		if !shouldEnableMetalMTP(metal) {
			t.Fatal("conflicting selectors changed the existing Metal MTP OR semantics")
		}
	})
	abi.RegisterEngine("test", echoEngine{})

	for _, tc := range []struct {
		name          string
		receipt       *agent.VulkanMTPExecution
		temperature   string
		wantEngine    string
		wantDowngrade string
	}{
		{
			name:       "actual execution",
			receipt:    &agent.VulkanMTPExecution{Engine: SpeculativeMTPVulkan, Used: true, ProposalRounds: 1, TargetOperations: 1},
			wantEngine: SpeculativeMTPVulkan,
		},
		{
			name:          "runtime downgrade",
			receipt:       &agent.VulkanMTPExecution{Engine: SpeculativeMTPVulkan, DowngradeReason: "proposal-error"},
			wantDowngrade: "proposal-error",
		},
		{name: "sampling or unsupported bypass", temperature: "0.7"},
	} {
		t.Run(tc.name+" headers", func(t *testing.T) {
			srv, err := New(Config{EngineID: "test", Model: "qwen38-vulkan"})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			srv.planner = &vulkanMTPGatewayPlanner{receipt: tc.receipt}
			temperature := tc.temperature
			if temperature == "" {
				temperature = "0"
			}
			payload := `{"model":"qwen38-vulkan","messages":[{"role":"user","content":"hi"}],"temperature":` + temperature + `}`
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(payload))
			req.Header.Set("content-type", "application/json")
			w := httptest.NewRecorder()
			srv.Handler().ServeHTTP(w, req)
			resp := w.Result()
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, body)
			}
			if got := resp.Header.Get(HeaderSpeculative); got != tc.wantEngine {
				t.Fatalf("%s = %q, want %q", HeaderSpeculative, got, tc.wantEngine)
			}
			if got := resp.Header.Get(HeaderSpeculativeDowngrade); got != tc.wantDowngrade {
				t.Fatalf("%s = %q, want %q", HeaderSpeculativeDowngrade, got, tc.wantDowngrade)
			}
		})
	}

	for _, tc := range []struct {
		name          string
		receipt       *agent.VulkanMTPExecution
		plannerErr    error
		penalties     bool
		wantEngine    string
		wantDowngrade string
	}{
		{
			name:       "actual execution after first SSE byte",
			receipt:    &agent.VulkanMTPExecution{Engine: SpeculativeMTPVulkan, Used: true, ProposalRounds: 1, TargetOperations: 1},
			wantEngine: SpeculativeMTPVulkan,
		},
		{
			name:          "runtime downgrade after first SSE byte",
			receipt:       &agent.VulkanMTPExecution{Engine: SpeculativeMTPVulkan, DowngradeReason: "proposal-error"},
			wantDowngrade: "proposal-error",
		},
		{
			name:          "error receipt after first SSE byte",
			receipt:       &agent.VulkanMTPExecution{Engine: SpeculativeMTPVulkan, DowngradeReason: "cancelled", Cancelled: true},
			plannerErr:    context.Canceled,
			wantDowngrade: "cancelled",
		},
		{name: "scored sampling bypass after first SSE byte", penalties: true},
	} {
		t.Run(tc.name+" trailers", func(t *testing.T) {
			srv, err := New(Config{EngineID: "test", Model: "qwen38-vulkan"})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			planner := &vulkanMTPGatewayPlanner{receipt: tc.receipt, err: tc.plannerErr}
			srv.planner = planner
			payload := `{"model":"qwen38-vulkan","messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true}`
			if tc.penalties {
				payload = `{"model":"qwen38-vulkan","messages":[{"role":"user","content":"hi"}],"temperature":0,"frequency_penalty":0.5,"presence_penalty":0.25,"stream":true}`
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(payload))
			req.Header.Set("content-type", "application/json")
			w := httptest.NewRecorder()
			srv.Handler().ServeHTTP(w, req)
			resp := w.Result()
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("status = %d, want committed SSE 200; body=%s", resp.StatusCode, body)
			}
			_, _ = io.ReadAll(resp.Body)
			if got := resp.Trailer.Get(HeaderSpeculative); got != tc.wantEngine {
				t.Fatalf("trailer %s = %q, want %q", HeaderSpeculative, got, tc.wantEngine)
			}
			if got := resp.Trailer.Get(HeaderSpeculativeDowngrade); got != tc.wantDowngrade {
				t.Fatalf("trailer %s = %q, want %q", HeaderSpeculativeDowngrade, got, tc.wantDowngrade)
			}
			if tc.penalties && (planner.got.FrequencyPenalty == nil || *planner.got.FrequencyPenalty != 0.5 || planner.got.PresencePenalty == nil || *planner.got.PresencePenalty != 0.25) {
				t.Fatalf("streamed penalties = frequency %v presence %v, want 0.5/0.25", planner.got.FrequencyPenalty, planner.got.PresencePenalty)
			}
		})
	}
}

func TestGatewayVulkanMTPConcurrentReceiptsRemainRequestLocal(t *testing.T) {
	abi.RegisterEngine("test", echoEngine{})
	srv, err := New(Config{EngineID: "test", Model: "qwen38-vulkan"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	release := make(chan struct{})
	released := false
	defer func() {
		if !released {
			close(release)
		}
	}()
	planner := &vulkanMTPConcurrentPlanner{entered: make(chan string, 2), release: release}
	srv.planner = planner

	type response struct {
		name string
		resp *http.Response
	}
	responses := make(chan response, 2)
	run := func(name string, stream bool) {
		payload := `{"model":"qwen38-vulkan","messages":[{"role":"user","content":"` + name + `"}],"temperature":0}`
		if stream {
			payload = strings.TrimSuffix(payload, "}") + `,"stream":true}`
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(payload))
		req.Header.Set("content-type", "application/json")
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)
		resp := w.Result()
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		responses <- response{name: name, resp: resp}
	}
	go run("success", false)
	go run("downgrade", true)
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	seen := make(map[string]bool, 2)
	for range 2 {
		select {
		case name := <-planner.entered:
			seen[name] = true
		case <-deadline.C:
			t.Fatal("concurrent requests did not both reach the planner")
		}
	}
	if !seen["success"] || !seen["downgrade"] {
		t.Fatalf("concurrent planner entries = %v", seen)
	}
	close(release)
	released = true

	for range 2 {
		var result response
		select {
		case result = <-responses:
		case <-deadline.C:
			t.Fatal("concurrent request did not return after planner release")
		}
		if result.resp.StatusCode != http.StatusOK {
			t.Fatalf("%s status = %d, want 200", result.name, result.resp.StatusCode)
		}
		switch result.name {
		case "success":
			if got := result.resp.Header.Get(HeaderSpeculative); got != SpeculativeMTPVulkan {
				t.Fatalf("success %s = %q, want %q", HeaderSpeculative, got, SpeculativeMTPVulkan)
			}
			if got := result.resp.Header.Get(HeaderSpeculativeDowngrade); got != "" {
				t.Fatalf("success inherited downgrade %q", got)
			}
		case "downgrade":
			if got := result.resp.Trailer.Get(HeaderSpeculative); got != "" {
				t.Fatalf("SSE downgrade inherited used engine %q", got)
			}
			if got := result.resp.Trailer.Get(HeaderSpeculativeDowngrade); got != "constructor-refused" {
				t.Fatalf("SSE downgrade trailer = %q, want constructor-refused", got)
			}
		default:
			t.Fatalf("unexpected response %q", result.name)
		}
	}
}
