package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelQwenQ4KPrefillChunkConfig(t *testing.T) {
	accepted := []int{128, 512, 768, 1024, 2048, 4096, 8192}
	for _, want := range accepted {
		got, err := resolveInKernelQwenQ4KPrefillChunkTokens(want)
		if err != nil || got != want {
			t.Errorf("resolve(%d) = (%d, %v), want (%d, nil)", want, got, err, want)
		}
	}
	if got, err := resolveInKernelQwenQ4KPrefillChunkTokens(0); err != nil || got != inKernelQwenQ4KPrefillChunkTokens {
		t.Fatalf("unset resolve = (%d, %v), want default (%d, nil)", got, err, inKernelQwenQ4KPrefillChunkTokens)
	}
	for _, raw := range []int{-512, 127, 8193, 16384} {
		got, err := resolveInKernelQwenQ4KPrefillChunkTokens(raw)
		var typed *model.InKernelQwenQ4KPrefillChunkConfigError
		if got != 0 || !errors.As(err, &typed) || typed.Value != fmt.Sprint(raw) {
			t.Errorf("resolve(%d) = (%d, %T %v), want (0, typed error retaining value)", raw, got, err, err)
		}
	}

	t.Setenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS", "4096")
	p := NewInKernelPlannerWithConfig(qwenHybridPrefillModel(), nil, "qwen-config-once", true, nil, false, InKernelPlannerConfig{QwenQ4KPrefillChunkTokens: 2048})
	if got := p.effectiveQwenQ4KPrefillChunkTokens(); got != 2048 {
		t.Fatalf("planner reread env after construction: width = %d, want 2048", got)
	}
}

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

func TestInKernelQwenQ4KPrefillChunkConfigPartitions(t *testing.T) {
	for _, width := range []int{512, 1024, 2048, 4096, 8192} {
		t.Run(fmt.Sprint(width), func(t *testing.T) {
			ids := make([]int, 2*width+1)
			for i := range ids {
				ids[i] = i + 1
			}
			p := qwenQ4KPrefillPlanner(nil)
			p.qwenQ4KPrefillChunkTokens = width

			monolithic := &recordingPrefillSession{}
			wantLogits := monolithic.Prefill(ids)
			chunked := &recordingPrefillSession{}
			gotLogits, err := p.prefillDivergentSuffix(context.Background(), chunked, ids)
			if err != nil {
				t.Fatal(err)
			}
			wantCalls := []recordedPrefillCall{
				{kind: "no-logits", ids: ids[:width]},
				{kind: "no-logits", ids: ids[width : 2*width]},
				{kind: "logits", ids: ids[2*width:]},
			}
			if !reflect.DeepEqual(chunked.calls, wantCalls) {
				t.Fatalf("prefill calls = %#v, want %#v", chunked.calls, wantCalls)
			}
			if !reflect.DeepEqual(chunked.state, monolithic.state) || !reflect.DeepEqual(gotLogits, wantLogits) {
				t.Fatalf("configured width %d changed state/logits parity", width)
			}
			for i, call := range chunked.calls {
				if len(call.ids) > width {
					t.Fatalf("call %d width = %d, want <= %d", i, len(call.ids), width)
				}
			}
		})
	}
}

func TestInKernelQwenQ4KBoundedPrefillCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	s := &recordingPrefillSession{cancel: cancel}
	const width = 1024
	ids := make([]int, width+2)
	p := qwenQ4KPrefillPlanner(nil)
	p.qwenQ4KPrefillChunkTokens = width

	logits, err := p.prefillDivergentSuffix(ctx, s, ids)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if logits != nil {
		t.Fatalf("logits = %v, want nil after cancellation", logits)
	}
	if len(s.calls) != 1 || s.calls[0].kind != "no-logits" || len(s.calls[0].ids) != width {
		t.Fatalf("calls after cancellation = %#v, want one %d-token no-logits call", s.calls, width)
	}
}

func TestInKernelQwenQ4KPrefillChunkInvalidRefusesBeforeModelWork(t *testing.T) {
	typed := &model.InKernelQwenQ4KPrefillChunkConfigError{Value: "768"}
	p := qwenQ4KPrefillPlanner(nil)
	p.qwenQ4KPrefillChunkConfigErr = typed

	// The target planner deliberately has no tokenizer. Reaching tokenization or
	// model execution would panic; the typed error must return first.
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "must not run"}}, nil)
	var got *model.InKernelQwenQ4KPrefillChunkConfigError
	if !errors.As(err, &got) || got != typed {
		t.Fatalf("Complete error = %T %v, want retained typed config error", err, err)
	}
}

func TestInKernelQwenQ4KPrefillChunkReceiptReadback(t *testing.T) {
	p := qwenQ4KPrefillPlanner(nil)
	p.qwenQ4KPrefillChunkTokens = 4096
	receipt := p.buildNativeInferenceReceipt(&nativeInferenceMeasurement{tokenIDs: []int{7}, logprobs: []float64{-0.25}}, 1.25, 0.5)
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var got model.NativeInferenceReceipt
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.PrefillChunkTokens != 4096 {
		t.Fatalf("receipt prefill_chunk_tokens = %d, want 4096; json=%s", got.PrefillChunkTokens, raw)
	}
	nonTarget := &InKernelPlanner{m: &model.Model{}, q4k: true, qwenQ4KPrefillChunkTokens: 4096}
	if got := nonTarget.nativeInferencePrefillChunkTokens(); got != 0 {
		t.Fatalf("non-target receipt prefill_chunk_tokens = %d, want 0 (not applicable)", got)
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
		{name: "non-qwen", p: &InKernelPlanner{m: &model.Model{}, q4k: true, qwenQ4KPrefillChunkConfigErr: &model.InKernelQwenQ4KPrefillChunkConfigError{Value: "invalid"}}, ids: long},
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
