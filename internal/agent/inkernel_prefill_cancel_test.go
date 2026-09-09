package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type cancelPrefillSession struct {
	calls       []recordedPrefillCall
	cancel      context.CancelFunc
	cancelAfter int
}

func (s *cancelPrefillSession) PrefillNoLogits(ids []int) {
	s.record("no-logits", ids)
}

func (s *cancelPrefillSession) Prefill(ids []int) []float32 {
	s.record("logits", ids)
	return []float32{float32(len(ids))}
}

func (s *cancelPrefillSession) record(kind string, ids []int) {
	s.calls = append(s.calls, recordedPrefillCall{kind: kind, ids: append([]int(nil), ids...)})
	if s.cancel != nil && len(s.calls) == s.cancelAfter {
		s.cancel()
	}
}

type namedPrefillBackend struct {
	compute.Backend
	name string
}

func (b namedPrefillBackend) Name() string { return b.name }

func (b namedPrefillBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func (b namedPrefillBackend) Qwen35SequenceEmbeddingRowsPath() string {
	return compute.Qwen35SequenceEmbeddingRowsPath
}

func (b namedPrefillBackend) Qwen35SequencePrefill(compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	panic("test capability marker must not execute")
}

type genericPrefillBackend struct {
	compute.Backend
	name string
}

func (b genericPrefillBackend) Name() string { return b.name }

type markerOnlyPrefillBackend struct{ genericPrefillBackend }

func (markerOnlyPrefillBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func (markerOnlyPrefillBackend) Qwen35SequenceEmbeddingRowsPath() string {
	return compute.Qwen35SequenceEmbeddingRowsPath
}

func TestInKernelPrefillCancellationBoundaries(t *testing.T) {
	const width = 128
	newTarget := func() *InKernelPlanner {
		p := qwenQ4KPrefillPlanner(namedPrefillBackend{Backend: compute.Default(), name: "vulkan"})
		p.qwenQ4KPrefillChunkTokens = width
		return p
	}

	t.Run("before work", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		s := &cancelPrefillSession{}
		logits, err := newTarget().prefillDivergentSuffix(ctx, s, make([]int, width+1))
		if !errors.Is(err, context.Canceled) || logits != nil {
			t.Fatalf("result = (%v, %v), want (nil, context.Canceled)", logits, err)
		}
		if len(s.calls) != 0 {
			t.Fatalf("calls = %#v, want no model work", s.calls)
		}
	})

	t.Run("between chunks", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		s := &cancelPrefillSession{cancel: cancel, cancelAfter: 1}
		logits, err := newTarget().prefillDivergentSuffix(ctx, s, make([]int, 2*width+1))
		if !errors.Is(err, context.Canceled) || logits != nil {
			t.Fatalf("result = (%v, %v), want (nil, context.Canceled)", logits, err)
		}
		want := []recordedPrefillCall{{kind: "no-logits", ids: make([]int, width)}}
		if !reflect.DeepEqual(s.calls, want) {
			t.Fatalf("calls = %#v, want exactly the first intermediate chunk", s.calls)
		}
	})

	for _, tc := range []struct {
		name string
		ids  int
		at   int
	}{
		{name: "short final call", ids: width, at: 1},
		{name: "chunked final call", ids: width + 1, at: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			s := &cancelPrefillSession{cancel: cancel, cancelAfter: tc.at}
			logits, err := newTarget().prefillDivergentSuffix(ctx, s, make([]int, tc.ids))
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v, want context.Canceled", err)
			}
			if logits != nil {
				t.Fatalf("logits = %v, want cancellation to suppress final logits", logits)
			}
			if got := s.calls[len(s.calls)-1].kind; got != "logits" {
				t.Fatalf("final call kind = %q, want logits call before cancellation is observed", got)
			}
		})
	}
}

func TestInKernelPrefillChunkRouting(t *testing.T) {
	const configured = 256
	ids := make([]int, 2*configured+1)
	tests := []struct {
		name       string
		planner    *InKernelPlanner
		wantWidths []int
		wantKinds  []string
	}{
		{
			name:       "supported Vulkan Qwen hybrid configured width",
			planner:    qwenQ4KPrefillPlanner(namedPrefillBackend{Backend: compute.Default(), name: "vulkan"}),
			wantWidths: []int{configured, configured, 1},
			wantKinds:  []string{"no-logits", "no-logits", "logits"},
		},
		{
			name:       "generic backend remains monolithic",
			planner:    qwenQ4KPrefillPlanner(genericPrefillBackend{Backend: compute.Default(), name: "generic-device"}),
			wantWidths: []int{len(ids)},
			wantKinds:  []string{"logits"},
		},
		{
			name: "path markers without sequence operation remain monolithic",
			planner: qwenQ4KPrefillPlanner(markerOnlyPrefillBackend{genericPrefillBackend{
				Backend: compute.Default(), name: "vulkan-marker-only",
			}}),
			wantWidths: []int{len(ids)},
			wantKinds:  []string{"logits"},
		},
	}
	tests[0].planner.qwenQ4KPrefillChunkTokens = configured
	tests[1].planner.qwenQ4KPrefillChunkTokens = configured
	tests[2].planner.qwenQ4KPrefillChunkTokens = configured

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := &cancelPrefillSession{}
			logits, err := tc.planner.prefillDivergentSuffix(context.Background(), s, ids)
			if err != nil || len(logits) == 0 {
				t.Fatalf("result = (%v, %v), want final logits", logits, err)
			}
			if len(s.calls) != len(tc.wantWidths) {
				t.Fatalf("calls = %#v, want widths %v", s.calls, tc.wantWidths)
			}
			for i, call := range s.calls {
				if call.kind != tc.wantKinds[i] || len(call.ids) != tc.wantWidths[i] {
					t.Fatalf("call %d = %s/%d, want %s/%d", i, call.kind, len(call.ids), tc.wantKinds[i], tc.wantWidths[i])
				}
			}
		})
	}

	defaultPlanner := qwenQ4KPrefillPlanner(namedPrefillBackend{Backend: compute.Default(), name: "vulkan"})
	defaultIDs := make([]int, inKernelQwenQ4KPrefillChunkTokens+1)
	defaultSession := &cancelPrefillSession{}
	if _, err := defaultPlanner.prefillDivergentSuffix(context.Background(), defaultSession, defaultIDs); err != nil {
		t.Fatal(err)
	}
	if len(defaultSession.calls) != 2 || len(defaultSession.calls[0].ids) != inKernelQwenQ4KPrefillChunkTokens || defaultSession.calls[0].kind != "no-logits" || defaultSession.calls[1].kind != "logits" {
		t.Fatalf("default Vulkan routing calls = %#v, want %d-token intermediate chunk then final logits", defaultSession.calls, inKernelQwenQ4KPrefillChunkTokens)
	}
}
