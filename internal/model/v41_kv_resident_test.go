package model

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41ResidentKVTestBackend adds a CPU-storage row log to the shared-attention
// recorder. It tests mirror routing, transfer volume and failure typing only;
// physical device behavior is TestV41DecodeResidentH2DVulkan.
type v41ResidentKVTestBackend struct {
	*v41SharedAttentionTestBackend
	gathers  int
	rowFault string
}

func newV41ResidentKVTestBackend() *v41ResidentKVTestBackend {
	return &v41ResidentKVTestBackend{v41SharedAttentionTestBackend: newV41SharedAttentionTestBackend()}
}

func (b *v41ResidentKVTestBackend) own(shape []int, values []float32) compute.Tensor {
	out := compute.NewF32(b, append([]int(nil), shape...), values)
	b.allocations[out.Buf()] = true
	b.managed[out.Buf()] = true
	return out
}

func (b *v41ResidentKVTestBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 resident ") && !strings.HasPrefix(site, "V4.1 shared attention ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 {
		panic("unexpected resident upload dtype")
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	b.uploads++
	b.uploadBytes += 4 * len(values)
	return b.own(x.Shape, values)
}

// V41SharedAttention repeats the recorder's rounded control over this
// wrapper's own tensors.
func (b *v41ResidentKVTestBackend) V41SharedAttention(q, kv, sink compute.Tensor, rows, heads, dim int, scale float32, mode compute.V41SharedAttentionMode, hasSink bool) (compute.Tensor, error) {
	sinkWidth := 1
	if hasSink {
		sinkWidth = heads
	}
	for _, input := range []struct {
		x     compute.Tensor
		shape []int
	}{{q, []int{heads, dim}}, {kv, []int{rows, dim}}, {sink, []int{sinkWidth}}} {
		if input.x.Backend() != b || !b.allocations[input.x.Buf()] || !reflect.DeepEqual(input.x.Shape, input.shape) {
			return compute.Tensor{}, errors.New("invalid tensor or dead binding")
		}
	}
	call := v41SharedAttentionTestCall{
		q: append([]float32(nil), b.Backend.Read(q)...), kv: append([]float32(nil), b.Backend.Read(kv)...), sink: append([]float32(nil), b.Backend.Read(sink)...),
		rows: rows, heads: heads, dim: dim, scale: scale, mode: mode, hasSink: hasSink,
	}
	b.calls = append(b.calls, call)
	values, err := v41SharedAttentionRoundedControl(call)
	if err != nil {
		return compute.Tensor{}, err
	}
	return b.own([]int{heads, dim}, values), nil
}

func (b *v41ResidentKVTestBackend) V41RowLogAvailable() bool { return true }

func (b *v41ResidentKVTestBackend) fault(op string) error {
	if b.rowFault == op {
		return &compute.BackendError{Backend: "cpu-ref-row-log", Site: op, Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
	}
	return nil
}

func (b *v41ResidentKVTestBackend) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	if err := compute.ValidateV41RowLogAlloc(rows, width); err != nil {
		return compute.Tensor{}, err
	}
	if err := b.fault("alloc"); err != nil {
		return compute.Tensor{}, err
	}
	return b.own([]int{rows, width}, make([]float32, rows*width)), nil
}

func (b *v41ResidentKVTestBackend) V41RowLogCopy(dst compute.Tensor, dstRow int, src compute.Tensor, srcRow, rows int) error {
	rowBytes, err := compute.ValidateV41RowLogCopy(dst, dstRow, src, srcRow, rows)
	if err != nil {
		return err
	}
	if !b.allocations[dst.Buf()] || !b.allocations[src.Buf()] {
		return errors.New("row-log copy on a dead buffer")
	}
	if err := b.fault("copy"); err != nil {
		return err
	}
	w := rowBytes / 4
	copy(b.Backend.Read(dst)[dstRow*w:(dstRow+rows)*w], b.Backend.Read(src)[srcRow*w:(srcRow+rows)*w])
	return nil
}

func (b *v41ResidentKVTestBackend) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	total, err := compute.ValidateV41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	if err := b.fault("gather"); err != nil {
		return compute.Tensor{}, err
	}
	out := make([]float32, 0, total*width)
	for _, span := range spans {
		if !b.allocations[span.Src.Buf()] {
			return compute.Tensor{}, errors.New("row-log gather from a dead buffer")
		}
		out = append(out, b.Backend.Read(span.Src)[span.Row*width:(span.Row+span.Rows)*width]...)
	}
	b.gathers++
	return b.own([]int{total, width}, out), nil
}

type v41ResidentKVRun struct {
	logits  [][]float32
	h2d     []int // attention upload bytes per decode step
	gathers int
}

func v41ResidentKVDecode(t *testing.T, build func(*testing.T) *Model, ctx int, disabled bool) v41ResidentKVRun {
	t.Helper()
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = disabled
	defer func() { v41ResidentKVDisabled = prior }()
	m := build(t)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	b := newV41ResidentKVTestBackend()
	s := v41DenseTestSession(t, m, b)
	v41SharedAttentionOnly(s)
	if s.v41State().sharedAttention == nil {
		t.Fatal("shared attention was not bound")
	}
	var run v41ResidentKVRun
	logits := append([]float32(nil), s.Prefill(v41DecodeSubmitsPrompt(ctx, m.Cfg.VocabSize))...)
	run.logits = append(run.logits, logits)
	for i := 0; i < v41DecodeSubmitsSteps; i++ {
		before := b.uploadBytes
		logits = append([]float32(nil), s.Step(argmaxF32(logits))...)
		run.logits = append(run.logits, logits)
		run.h2d = append(run.h2d, b.uploadBytes-before)
	}
	run.gathers = b.gathers
	return run
}

// TestV41DecodeResidentH2D is the cpu-ref half of the #13821 witness: with the
// session KV mirror on, steady per-step attention upload volume no longer
// depends on context length, and logits equal the per-step upload path bit for
// bit. Each fail-closed refusal keeps its sentinel.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=10s lane=default
func TestV41DecodeResidentH2D(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("compressed_%t", compressed), func(t *testing.T) {
			build := func(t *testing.T) *Model { return v41DecodeSubmitsLayeredFixture(t, 1, compressed) }
			runs := map[int]v41ResidentKVRun{}
			for _, ctx := range []int{4, 64} {
				on, off := v41ResidentKVDecode(t, build, ctx, false), v41ResidentKVDecode(t, build, ctx, true)
				if on.gathers == 0 || off.gathers != 0 {
					t.Fatalf("ctx=%d mirror gathers on=%d off=%d", ctx, on.gathers, off.gathers)
				}
				if !reflect.DeepEqual(on.logits, off.logits) {
					t.Fatalf("ctx=%d logits differ between the mirror and the per-step upload path", ctx)
				}
				t.Logf("ctx=%d attention h2d bytes per step: mirror=%v upload=%v", ctx, on.h2d, off.h2d)
				runs[ctx] = on
			}
			// Step 0 seeds the mirror; later steps carry only the query and new rows.
			if a, b := runs[4].h2d[1:], runs[64].h2d[1:]; !reflect.DeepEqual(a, b) {
				t.Fatalf("steady mirror upload bytes depend on context: ctx4=%v ctx64=%v", a, b)
			}
		})
	}

	t.Run("sentinels", func(t *testing.T) {
		const dim = 4
		request := func(owner *V41AttentionState, pos, ringRows, windowRows, compressedRows int) v41SharedAttentionRequest {
			rows := windowRows + compressedRows
			idx := make([]int32, rows+1)
			for i := range idx {
				idx[i] = int32(i)
			}
			idx[rows] = -1
			kv := make([]float32, rows*dim)
			for i := range kv {
				kv[i] = float32(i%7) / 8
			}
			view := &v41ResidentKVView{owner: owner, pos: pos, ringRows: ringRows, firstKey: pos - windowRows + 1, windowRows: windowRows, compressedRows: compressedRows}
			return v41SharedAttentionRequest{
				mode: compute.V41SharedAttentionPlain, q: []float32{1, 0.5, -0.25, 0.125}, kv: kv, idx: idx,
				plain:    V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: dim, N: rows, TopK: rows + 1, Softmax: 0.5},
				resident: view,
			}
		}
		for _, tc := range []struct {
			name     string
			rowFault string
			calls    func(*V41AttentionState) []v41SharedAttentionRequest
			want     error
		}{
			{"capacity", "", func(o *V41AttentionState) []v41SharedAttentionRequest {
				return []v41SharedAttentionRequest{request(o, 5, 2, 4, 0)}
			}, errV41ResidentKVCapacity},
			{"desync", "", func(o *V41AttentionState) []v41SharedAttentionRequest {
				return []v41SharedAttentionRequest{request(o, 0, 4, 1, 2), request(o, 1, 4, 2, 1)}
			}, errV41ResidentKVDesync},
			{"backend", "gather", func(o *V41AttentionState) []v41SharedAttentionRequest {
				return []v41SharedAttentionRequest{request(o, 0, 4, 1, 0)}
			}, errV41ResidentKVBackend},
		} {
			t.Run(tc.name, func(t *testing.T) {
				b := newV41ResidentKVTestBackend()
				b.rowFault = tc.rowFault
				s := v41DenseTestSession(t, v41IncrementalPlainModel(t, 1), b)
				attend := s.v41SharedAttentionFunc()
				if attend == nil {
					t.Fatal("shared attention was not bound")
				}
				var err error
				calls := tc.calls(&V41AttentionState{})
				for i, call := range calls {
					if _, err = attend(0, call); err != nil && i < len(calls)-1 {
						t.Fatalf("setup call %d: %v", i, err)
					}
				}
				var closed *BackendForwardOperationError
				if !errors.Is(err, tc.want) || !errors.As(err, &closed) {
					t.Fatalf("err=%v, want %v through the forward operation error", err, tc.want)
				}
			})
		}
	})
}
