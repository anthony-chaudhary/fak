package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type recordedPrefillCall struct {
	kind string
	ids  []int
}

type recordingPrefillSession struct {
	calls  []recordedPrefillCall
	state  []int
	cancel context.CancelFunc
}

func (s *recordingPrefillSession) PrefillNoLogits(ids []int) {
	s.record("no-logits", ids)
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil
	}
}

func (s *recordingPrefillSession) Prefill(ids []int) []float32 {
	s.record("logits", ids)
	var sum int
	for _, id := range s.state {
		sum += id
	}
	return []float32{float32(len(s.state)), float32(sum)}
}

func (s *recordingPrefillSession) record(kind string, ids []int) {
	s.calls = append(s.calls, recordedPrefillCall{kind: kind, ids: append([]int(nil), ids...)})
	s.state = append(s.state, ids...)
}

func TestInKernelQwenQ4KBoundedPrefill(t *testing.T) {
	ids := make([]int, 2*inKernelQwenQ4KPrefillChunkTokens+1)
	for i := range ids {
		ids[i] = i + 1
	}
	p := qwenQ4KPrefillPlanner(nil)

	monolithic := &recordingPrefillSession{}
	wantLogits := monolithic.Prefill(ids)
	chunked := &recordingPrefillSession{}
	gotLogits, err := p.prefillDivergentSuffix(context.Background(), chunked, ids)
	if err != nil {
		t.Fatal(err)
	}

	wantCalls := []recordedPrefillCall{
		{kind: "no-logits", ids: ids[:inKernelQwenQ4KPrefillChunkTokens]},
		{kind: "no-logits", ids: ids[inKernelQwenQ4KPrefillChunkTokens : 2*inKernelQwenQ4KPrefillChunkTokens]},
		{kind: "logits", ids: ids[2*inKernelQwenQ4KPrefillChunkTokens:]},
	}
	if !reflect.DeepEqual(chunked.calls, wantCalls) {
		t.Fatalf("prefill calls = %#v, want %#v", chunked.calls, wantCalls)
	}
	if !reflect.DeepEqual(chunked.state, monolithic.state) {
		t.Fatalf("chunked state differs from monolithic: got %d tokens, want %d", len(chunked.state), len(monolithic.state))
	}
	if !reflect.DeepEqual(gotLogits, wantLogits) {
		t.Fatalf("final logits = %v, want monolithic %v", gotLogits, wantLogits)
	}
	for i, call := range chunked.calls {
		if len(call.ids) > inKernelQwenQ4KPrefillChunkTokens {
			t.Fatalf("call %d width = %d, want <= %d", i, len(call.ids), inKernelQwenQ4KPrefillChunkTokens)
		}
	}
}

func TestInKernelQwenQ4KBoundedPrefillCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	s := &recordingPrefillSession{cancel: cancel}
	ids := make([]int, inKernelQwenQ4KPrefillChunkTokens+2)

	logits, err := qwenQ4KPrefillPlanner(nil).prefillDivergentSuffix(ctx, s, ids)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if logits != nil {
		t.Fatalf("logits = %v, want nil after cancellation", logits)
	}
	if len(s.calls) != 1 || s.calls[0].kind != "no-logits" || len(s.calls[0].ids) != inKernelQwenQ4KPrefillChunkTokens {
		t.Fatalf("calls after cancellation = %#v, want one %d-token no-logits call", s.calls, inKernelQwenQ4KPrefillChunkTokens)
	}
}

func TestInKernelQwenQ4KBoundedPrefillLeavesOtherPathsSingleCall(t *testing.T) {
	long := make([]int, inKernelQwenQ4KPrefillChunkTokens+1)
	tests := []struct {
		name string
		p    *InKernelPlanner
		ids  []int
	}{
		{name: "target-small", p: qwenQ4KPrefillPlanner(nil), ids: long[:inKernelQwenQ4KPrefillChunkTokens]},
		{name: "non-qwen", p: &InKernelPlanner{m: &model.Model{}, q4k: true}, ids: long},
		{name: "non-q4k", p: &InKernelPlanner{m: qwenHybridPrefillModel()}, ids: long},
		{name: "backend", p: qwenQ4KPrefillPlanner(compute.Default()), ids: long},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &recordingPrefillSession{}
			logits, err := tt.p.prefillDivergentSuffix(context.Background(), s, tt.ids)
			if err != nil {
				t.Fatal(err)
			}
			if len(s.calls) != 1 || s.calls[0].kind != "logits" || !reflect.DeepEqual(s.calls[0].ids, tt.ids) {
				t.Fatalf("calls = %#v, want one unchanged logits prefill", s.calls)
			}
			if len(logits) == 0 {
				t.Fatal("single-call path discarded final logits")
			}
		})
	}
}

func qwenQ4KPrefillPlanner(backend compute.Backend) *InKernelPlanner {
	return &InKernelPlanner{m: qwenHybridPrefillModel(), q4k: true, backend: backend}
}

func qwenHybridPrefillModel() *model.Model {
	return &model.Model{Cfg: model.Config{LayerTypes: []string{"linear_attention"}}}
}
