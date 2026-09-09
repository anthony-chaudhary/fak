package agent

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelContextWindow(t *testing.T) {
	tests := []struct {
		name       string
		declared   int
		configured int
		want       int
	}{
		{name: "model declaration", declared: 4096, want: 4096},
		{name: "configured narrower", declared: 4096, configured: 2048, want: 2048},
		{name: "model narrower", declared: 4096, configured: 8192, want: 4096},
		{name: "configuration supplies missing metadata", configured: 2048, want: 2048},
		{name: "unknown stays unknown", want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewInKernelPlannerWithConfig(
				&model.Model{Cfg: model.Config{MaxPositionEmbeddings: tt.declared}},
				nil,
				"native-test",
				false,
				nil,
				false,
				InKernelPlannerConfig{ContextTokens: tt.configured},
			)
			if got := p.ContextWindow(); got != tt.want {
				t.Fatalf("ContextWindow() = %d, want %d", got, tt.want)
			}
			if got := p.RuntimeConfig().ContextTokens; got != tt.configured {
				t.Fatalf("RuntimeConfig().ContextTokens = %d, want configured value %d", got, tt.configured)
			}
		})
	}
}

func TestInKernelContextBoundaryExecutes(t *testing.T) {
	tok := loadProbeTok(t)
	cfg := tinyConcurrencyConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, tok, "native-boundary", false, nil, false)
	messages := []Message{{Role: RoleUser, Content: "hi"}}
	promptTokens := encodedInKernelPromptTokens(t, tok, messages, m.Cfg)
	m.Cfg.MaxPositionEmbeddings = promptTokens + 1

	comp, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(1))
	if err != nil {
		t.Fatalf("exact prompt+max_tokens boundary rejected: %v", err)
	}
	if comp == nil {
		t.Fatal("exact prompt+max_tokens boundary returned nil completion")
	}
}

func TestInKernelContextOverflowRejectedBeforeExecutionOnEveryNativePath(t *testing.T) {
	tok := loadProbeTok(t)
	messages := []Message{{Role: RoleUser, Content: "oversize"}}
	cfg := tinyConcurrencyConfig()
	promptTokens := encodedInKernelPromptTokens(t, tok, messages, cfg)
	cfg.MaxPositionEmbeddings = promptTokens

	device, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend is not registered")
	}
	tests := []struct {
		name    string
		backend compute.Backend
		metal   bool
		maxNew  int
	}{
		{name: "CPU", maxNew: 1},
		{name: "Metal", metal: true, maxNew: 1},
		{name: "device", backend: device, maxNew: 1},
		{name: "integer overflow", maxNew: math.MaxInt},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// An intentionally uninitialized model makes any attempt to allocate a
			// session or execute a forward fail. Receiving the typed context error
			// therefore proves validation happened first on this backend path.
			p := NewInKernelPlanner(&model.Model{Cfg: cfg}, tok, "native-oversize", false, tt.backend, false)
			// Set the path bit after construction: on a real Darwin/Metal build the
			// constructor prepares resident weights, which this deliberately empty
			// pre-execution fixture does not contain.
			p.metal = tt.metal
			comp, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(tt.maxNew))
			if comp != nil {
				t.Fatalf("oversize request returned completion: %+v", comp)
			}
			var contextErr *InKernelContextLengthError
			if !errors.As(err, &contextErr) {
				t.Fatalf("oversize error = %T (%v), want *InKernelContextLengthError", err, err)
			}
			if contextErr.PromptTokens != promptTokens || contextErr.MaxNewTokens != tt.maxNew || contextErr.MaxContext != promptTokens {
				t.Fatalf("context error = %+v, want prompt=%d max_new=%d max_context=%d", contextErr, promptTokens, tt.maxNew, promptTokens)
			}
		})
	}
}

func encodedInKernelPromptTokens(t *testing.T, tok interface {
	Encode(string) ([]int, error)
}, messages []Message, cfg model.Config) int {
	t.Helper()
	ids, err := tok.Encode(renderInKernelChatMLRequest(messages, nil, cfg, nil, nil))
	if err != nil {
		t.Fatalf("encode native prompt: %v", err)
	}
	return len(ids)
}
