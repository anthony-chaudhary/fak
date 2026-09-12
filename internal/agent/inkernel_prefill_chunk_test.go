package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
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

type panelCapPrefillBackend struct {
	compute.Backend
	bufferBytes int64
}

func (b panelCapPrefillBackend) MaxWeightBufferBytes() int64 { return b.bufferBytes }

func (panelCapPrefillBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func (panelCapPrefillBackend) Qwen35SequenceEmbeddingRowsPath() string {
	return compute.Qwen35SequenceEmbeddingRowsPath
}

func (panelCapPrefillBackend) Qwen35SequencePrefill(compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	panic("test capability marker must not execute")
}

func TestQwenPrefillMaxTokenPanelWidthUsesWidestPanel(t *testing.T) {
	tests := []struct {
		name string
		cfg  model.Config
		want int64
	}{
		{
			name: "gdn gate wider than intermediate",
			cfg: model.Config{
				HiddenSize: 64, IntermediateSize: 100,
				NumHeads: 2, NumKVHeads: 1, HeadDim: 8,
				LinearNumKeyHeads: 3, LinearKeyHeadDim: 20,
				LinearNumValueHeads: 2, LinearValueHeadDim: 20,
			},
			want: 160, // 2*(3*20) + 2*20
		},
		{
			name: "doubled q gate wider than intermediate",
			cfg: model.Config{
				HiddenSize: 64, IntermediateSize: 100,
				NumHeads: 9, NumKVHeads: 1, HeadDim: 8,
				LinearNumKeyHeads: 1, LinearKeyHeadDim: 8,
				LinearNumValueHeads: 1, LinearValueHeadDim: 8,
			},
			want: 144, // 2*(9*8)
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := qwenPrefillMaxTokenPanelWidth(tt.cfg)
			if !ok || got != tt.want {
				t.Fatalf("qwenPrefillMaxTokenPanelWidth() = (%d, %v), want (%d, true)", got, ok, tt.want)
			}
		})
	}
}

func TestInKernelQwenQ4KPrefillChunkDeviceCap(t *testing.T) {
	cfg := model.Config{
		LayerTypes: []string{"linear_attention"},
		HiddenSize: 64, IntermediateSize: 100,
		NumHeads: 2, NumKVHeads: 1, HeadDim: 8,
		LinearNumKeyHeads: 3, LinearKeyHeadDim: 20,
		LinearNumValueHeads: 2, LinearValueHeadDim: 20,
	}
	const widest = int64(160)
	const rowBytes = widest * 4
	newPlanner := func(configured int, capBytes int64) *InKernelPlanner {
		backend := panelCapPrefillBackend{Backend: compute.Default(), bufferBytes: capBytes}
		return NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "qwen-panel-cap", true, backend, false, InKernelPlannerConfig{QwenQ4KPrefillChunkTokens: configured})
	}

	t.Run("unset preserves historical default", func(t *testing.T) {
		p := newPlanner(0, rowBytes*64)
		if got := p.effectiveQwenQ4KPrefillChunkTokens(); got != inKernelQwenQ4KPrefillChunkTokens {
			t.Fatalf("effective chunk = %d, want unchanged default %d", got, inKernelQwenQ4KPrefillChunkTokens)
		}
	})
	t.Run("explicit smaller request is unchanged", func(t *testing.T) {
		p := newPlanner(128, rowBytes*256)
		if got := p.effectiveQwenQ4KPrefillChunkTokens(); got != 128 {
			t.Fatalf("effective chunk = %d, want configured 128", got)
		}
	})
	t.Run("explicit large request uses exact device cap", func(t *testing.T) {
		p := newPlanner(8192, rowBytes*777+rowBytes-1)
		if got := p.effectiveQwenQ4KPrefillChunkTokens(); got != 777 {
			t.Fatalf("effective chunk = %d, want floor(buffer/row) 777", got)
		}
	})
	t.Run("device cap below configured minimum is exact", func(t *testing.T) {
		p := newPlanner(8192, rowBytes*64+rowBytes-1)
		if got := p.effectiveQwenQ4KPrefillChunkTokens(); got != 64 {
			t.Fatalf("effective chunk = %d, want floor(buffer/row) 64", got)
		}
	})
}

func TestInKernelQwenQ4KPrefillPanelCannotFitRefusesBeforeTokenization(t *testing.T) {
	cfg := model.Config{
		LayerTypes: []string{"linear_attention"},
		HiddenSize: 64, IntermediateSize: 100,
		NumHeads: 2, NumKVHeads: 1, HeadDim: 8,
		LinearNumKeyHeads: 3, LinearKeyHeadDim: 20,
		LinearNumValueHeads: 2, LinearValueHeadDim: 20,
	}
	const rowBytes = int64(160 * 4)
	backend := panelCapPrefillBackend{Backend: compute.Default(), bufferBytes: rowBytes - 1}
	p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "qwen-panel-refusal", true, backend, false, InKernelPlannerConfig{QwenQ4KPrefillChunkTokens: 8192})

	// The planner deliberately has no tokenizer. Reaching tokenization would panic;
	// capacity admission must return the typed refusal first.
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "must not tokenize"}}, nil)
	var capacityErr *qwenQ4KPrefillPanelCapacityError
	if !errors.As(err, &capacityErr) {
		t.Fatalf("Complete error = %T %v, want *qwenQ4KPrefillPanelCapacityError", err, err)
	}
	if capacityErr.BufferBytes != rowBytes-1 || capacityErr.RowBytes != rowBytes {
		t.Fatalf("capacity error = %+v, want buffer=%d row=%d", capacityErr, rowBytes-1, rowBytes)
	}
}

func TestQwenPrefillMaxTokenPanelWidthRejectsInvalidDimensions(t *testing.T) {
	valid := model.Config{
		LayerTypes: []string{"linear_attention"},
		HiddenSize: 64, IntermediateSize: 100,
		NumHeads: 2, NumKVHeads: 1, HeadDim: 8,
		LinearNumKeyHeads: 3, LinearKeyHeadDim: 20,
		LinearNumValueHeads: 2, LinearValueHeadDim: 20,
	}
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name       string
		requires64 bool
		edit       func(*model.Config)
	}{
		{name: "missing hidden", edit: func(cfg *model.Config) { cfg.HiddenSize = 0 }},
		{name: "negative intermediate", edit: func(cfg *model.Config) { cfg.IntermediateSize = -1 }},
		{name: "q projection overflow", requires64: true, edit: func(cfg *model.Config) { cfg.NumHeads, cfg.HeadDim = maxInt, 2 }},
		{name: "gdn composite overflow", requires64: true, edit: func(cfg *model.Config) {
			cfg.LinearNumKeyHeads, cfg.LinearKeyHeadDim = maxInt, 1
			cfg.LinearNumValueHeads, cfg.LinearValueHeadDim = 1, 1
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.requires64 && strconv.IntSize != 64 {
				t.Skip("overflow fixture requires a 64-bit int")
			}
			cfg := valid
			tt.edit(&cfg)
			if got, ok := qwenPrefillMaxTokenPanelWidth(cfg); ok || got != 0 {
				t.Fatalf("qwenPrefillMaxTokenPanelWidth() = (%d, %v), want (0, false)", got, ok)
			}
			backend := panelCapPrefillBackend{Backend: compute.Default(), bufferBytes: 1 << 20}
			p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "qwen-invalid-panel", true, backend, false, InKernelPlannerConfig{QwenQ4KPrefillChunkTokens: 8192})
			bounded, capacityErr := p.deviceBoundedQwenQ4KPrefillChunkTokens()
			if bounded != 0 || capacityErr == nil || capacityErr.Reason != "model panel dimensions are invalid or overflow int64" {
				t.Fatalf("device bound = (%d, %+v), want typed invalid-dimension refusal", bounded, capacityErr)
			}
		})
	}

	t.Run("float32 row byte overflow", func(t *testing.T) {
		if strconv.IntSize != 64 {
			t.Skip("overflow fixture requires a 64-bit int")
		}
		cfg := valid
		cfg.HiddenSize = maxInt / 2
		widest, ok := qwenPrefillMaxTokenPanelWidth(cfg)
		if !ok || widest != int64(maxInt/2) {
			t.Fatalf("panel width = (%d, %v), want (%d, true)", widest, ok, maxInt/2)
		}
		backend := panelCapPrefillBackend{Backend: compute.Default(), bufferBytes: 1 << 20}
		p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "qwen-overflow-panel", true, backend, false, InKernelPlannerConfig{QwenQ4KPrefillChunkTokens: 8192})
		bounded, capacityErr := p.deviceBoundedQwenQ4KPrefillChunkTokens()
		if bounded != 0 || capacityErr == nil || capacityErr.Reason != "widest float32 token row overflows int64 bytes" {
			t.Fatalf("device bound = (%d, %+v), want typed float32-row overflow refusal", bounded, capacityErr)
		}
	})
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
