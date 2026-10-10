package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41SharedAttentionTestCall struct {
	q, kv, sink      []float32
	rows, heads, dim int
	scale            float32
	mode             compute.V41SharedAttentionMode
	hasSink          bool
}

// This recorder uses CPU storage to test adapter contracts. It is not device
// qualification, and neither numerical control calls production attention.
type v41SharedAttentionTestBackend struct {
	*v41DenseTestBackend
	supported                   bool
	calls                       []v41SharedAttentionTestCall
	allocations                 map[compute.Buffer]bool
	managed                     map[compute.Buffer]bool
	output                      compute.Buffer
	readScratch                 []float32
	uploads, uploadBytes, reads int
	fault                       string
	failCall                    int // zero fails every call; otherwise one-based dispatch
	cause                       error
	panicValue                  any
	cleanupPanic                any
	freeAttempts                int
}

func newV41SharedAttentionTestBackend() *v41SharedAttentionTestBackend {
	return &v41SharedAttentionTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), supported: true, allocations: map[compute.Buffer]bool{}, managed: map[compute.Buffer]bool{}}
}

func (b *v41SharedAttentionTestBackend) SupportsV41SharedAttention() bool { return b.supported }

func (b *v41SharedAttentionTestBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 shared attention ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("unexpected shared attention upload")
	}
	if strings.TrimPrefix(site, "V4.1 shared attention ")+" upload" == b.fault && (b.failCall == 0 || b.failCall == len(b.calls)+1) {
		panic(b.cause)
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	out := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.allocations[out.Buf()] = true
	b.managed[out.Buf()] = true
	b.uploads++
	b.uploadBytes += 4 * len(values)
	return out
}

func (b *v41SharedAttentionTestBackend) V41SharedAttention(q, kv, sink compute.Tensor, rows, heads, dim int, scale float32, mode compute.V41SharedAttentionMode, hasSink bool) (compute.Tensor, error) {
	sinkWidth := 1
	if hasSink {
		sinkWidth = heads
	}
	for _, input := range []struct {
		x     compute.Tensor
		shape []int
	}{{q, []int{heads, dim}}, {kv, []int{rows, dim}}, {sink, []int{sinkWidth}}} {
		if input.x.Backend() != b || !b.allocations[input.x.Buf()] || input.x.Dtype != compute.F32 || !reflect.DeepEqual(input.x.Shape, input.shape) {
			return compute.Tensor{}, errors.New("invalid tensor or dead sink binding")
		}
	}
	if q.Buf() == kv.Buf() || q.Buf() == sink.Buf() || kv.Buf() == sink.Buf() {
		return compute.Tensor{}, errors.New("aliased input buffers")
	}
	call := v41SharedAttentionTestCall{
		q: append([]float32(nil), b.Backend.Read(q)...), kv: append([]float32(nil), b.Backend.Read(kv)...), sink: append([]float32(nil), b.Backend.Read(sink)...),
		rows: rows, heads: heads, dim: dim, scale: scale, mode: mode, hasSink: hasSink,
	}
	if !hasSink && (len(call.sink) != 1 || math.Float32bits(call.sink[0]) != 0) {
		return compute.Tensor{}, errors.New("absent sink requires a live zero dummy")
	}
	b.calls = append(b.calls, call)
	fail := b.failCall == 0 || b.failCall == len(b.calls)
	if fail && b.fault == "dispatch" {
		return compute.Tensor{}, b.cause
	}
	if fail && b.fault == "panic" {
		if b.panicValue != nil {
			panic(b.panicValue)
		}
		panic(b.cause)
	}
	if fail && b.fault == "alias" {
		return q, nil
	}
	values, err := v41SharedAttentionRoundedControl(call)
	if err != nil {
		return compute.Tensor{}, err
	}
	out := compute.NewF32(b, []int{heads, dim}, values)
	b.allocations[out.Buf()], b.output = true, out.Buf()
	b.managed[out.Buf()] = true
	if fail && b.fault == "partial-output" {
		return out, b.cause
	}
	if fail && b.fault == "shape" {
		out.Shape = []int{heads * dim}
	}
	if fail && b.fault == "dtype" {
		out.Dtype = compute.F16
	}
	return out, nil
}

func (b *v41SharedAttentionTestBackend) Read(x compute.Tensor) []float32 {
	if x.Buf() != b.output {
		return b.v41DenseTestBackend.Read(x)
	}
	b.reads++
	fail := b.failCall == 0 || b.failCall == len(b.calls)
	if fail && b.fault == "read" {
		panic(b.cause)
	}
	values := b.Backend.Read(x)
	b.readScratch = append(b.readScratch[:0], values...)
	if fail && b.fault == "short-read" {
		return b.readScratch[:len(b.readScratch)-1]
	}
	if fail && b.fault == "nonfinite" {
		b.readScratch[len(b.readScratch)-1] = float32(math.NaN())
	}
	return b.readScratch
}

func (b *v41SharedAttentionTestBackend) Free(x compute.Tensor) {
	if !b.managed[x.Buf()] {
		b.v41DenseTestBackend.Free(x)
		return
	}
	if !b.allocations[x.Buf()] {
		panic("shared attention double free or nonowned tensor")
	}
	isOutput := x.Buf() == b.output
	if isOutput {
		for i := range b.readScratch {
			b.readScratch[i] = float32(math.NaN())
		}
		b.output = nil
	}
	for i := range b.Backend.Read(x) {
		b.Backend.Read(x)[i] = float32(math.NaN())
	}
	delete(b.allocations, x.Buf())
	b.Backend.Free(x)
	b.freeAttempts++
	if b.cleanupPanic != nil {
		panic(b.cleanupPanic)
	}
	if isOutput && b.fault == "free" && (b.failCall == 0 || b.failCall == len(b.calls)) {
		panic(b.cause)
	}
}

// Explicit rounding through float64 avoids relying on compiler FMA choices.
// This is the sequential F32 control, separate from the real-number oracle.
func v41SharedAttentionRoundedControl(c v41SharedAttentionTestCall) ([]float32, error) {
	f32add := func(a, b float32) float32 { return float32(float64(a) + float64(b)) }
	f32mul := func(a, b float32) float32 { return float32(float64(a) * float64(b)) }
	f32exp := func(a, b float32) float32 { return float32(math.Exp(float64(float32(float64(a) - float64(b))))) }
	finite := func(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }
	errAt := func(stage compute.V41SharedAttentionStage, h, s, d int, value float32) error {
		return &compute.V41SharedAttentionArithmeticError{Stage: stage, Head: h, SelectedSlot: s, Element: d, ValueBits: math.Float32bits(value)}
	}
	out := make([]float32, c.heads*c.dim)
	for h := 0; h < c.heads; h++ {
		score := func(slot int) float32 {
			var dot float32
			for d := 0; d < c.dim; d++ {
				dot = f32add(dot, f32mul(c.q[h*c.dim+d], c.kv[slot*c.dim+d]))
			}
			return f32mul(dot, c.scale)
		}
		maximum := float32(math.Inf(-1))
		if c.mode == compute.V41SharedAttentionCompressed {
			maximum = -math.MaxFloat32
		}
		if c.hasSink {
			maximum = c.sink[h]
		}
		for slot := 0; slot < c.rows; slot++ {
			s := score(slot)
			if !finite(s) {
				return nil, errAt(compute.V41SharedAttentionStageScore, h, slot, -1, s)
			}
			if s > maximum {
				maximum = s
			}
		}
		if c.mode == compute.V41SharedAttentionCompressed && maximum == -math.MaxFloat32 {
			continue
		}
		var denominator float32
		if c.hasSink {
			denominator = f32exp(c.sink[h], maximum)
			if !finite(denominator) {
				return nil, errAt(compute.V41SharedAttentionStageExp, h, -1, -1, denominator)
			}
		}
		for slot := 0; slot < c.rows; slot++ {
			term := f32exp(score(slot), maximum)
			if !finite(term) {
				return nil, errAt(compute.V41SharedAttentionStageExp, h, slot, -1, term)
			}
			denominator = f32add(denominator, term)
		}
		if !finite(denominator) {
			return nil, errAt(compute.V41SharedAttentionStageDenominator, h, -1, -1, denominator)
		}
		if denominator == 0 {
			continue
		}
		for slot := 0; slot < c.rows; slot++ {
			weight := float32(float64(f32exp(score(slot), maximum)) / float64(denominator))
			if !finite(weight) {
				return nil, errAt(compute.V41SharedAttentionStageWeight, h, slot, -1, weight)
			}
			for d := 0; d < c.dim; d++ {
				i := h*c.dim + d
				out[i] = f32add(out[i], f32mul(weight, c.kv[slot*c.dim+d]))
				if !finite(out[i]) {
					return nil, errAt(compute.V41SharedAttentionStageValue, h, slot, d, out[i])
				}
			}
		}
	}
	return out, nil
}

func v41SharedAttentionRealOracle(q []float32, rows [][]float32, sink []float32, heads, dim int, scale float32) []float64 {
	out := make([]float64, heads*dim)
	if len(rows) == 0 {
		return out
	}
	for h := 0; h < heads; h++ {
		scores := make([]float64, len(rows))
		maximum := math.Inf(-1)
		if sink != nil {
			maximum = float64(sink[h])
		}
		for i, row := range rows {
			for d, v := range row {
				scores[i] += float64(q[h*dim+d]) * float64(v)
			}
			scores[i] *= float64(scale)
			maximum = math.Max(maximum, scores[i])
		}
		denominator := float64(0)
		if sink != nil {
			denominator = math.Exp(float64(sink[h]) - maximum)
		}
		for i := range scores {
			scores[i] = math.Exp(scores[i] - maximum)
			denominator += scores[i]
		}
		for i, row := range rows {
			for d, v := range row {
				out[h*dim+d] += scores[i] / denominator * float64(v)
			}
		}
	}
	return out
}

func v41SharedAttentionNear(t *testing.T, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length got=%d want=%d", len(got), len(want))
	}
	var maxAbs, maxRel float64
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(want[i]) || math.IsInf(want[i], 0) {
			t.Fatalf("non-finite comparison at %d: %v / %v", i, v, want[i])
		}
		delta := math.Abs(float64(v) - want[i])
		rel := delta / math.Max(math.Abs(want[i]), 1e-30)
		maxAbs = math.Max(maxAbs, delta)
		maxRel = math.Max(maxRel, rel)
		// Frozen before device qualification; not measured hardware tolerance.
		if delta > 2e-5+2e-5*math.Abs(want[i]) {
			t.Fatalf("element %d got=%g want=%g abs=%g rel=%g", i, v, want[i], delta, rel)
		}
	}
	t.Logf("software control max absolute=%g max relative=%g", maxAbs, maxRel)
}

// Runtime estimate only, unmeasured in this source-only packet. A single compact
// corpus covers routing, independent arithmetic, ownership and error mapping.
// fak-test:runtime fast est=100ms lane=default
func TestV41SharedAttentionDeviceAdapter(t *testing.T) {
	newSession := func(t *testing.T) (*Session, *v41SharedAttentionTestBackend) {
		t.Helper()
		b := newV41SharedAttentionTestBackend()
		s := &Session{M: &Model{}, Backend: b}
		t.Cleanup(s.Close)
		return s, b
	}
	plainOpt := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 2, HeadDim: 2, N: 2, TopK: 4, Softmax: 0.7}
	q, kv, idx := []float32{0, 0, 0, 0}, []float32{2, -4, 6, 8}, []int32{1, 1, 0, -1}

	t.Run("analytic-sinks-duplicates-mask-and-ownership", func(t *testing.T) {
		for _, tc := range []struct {
			name     string
			sink     []float32
			length   []int32
			want     []float64
			selected []float32
		}{
			{"nil", nil, nil, []float64{14.0 / 3, 4, 14.0 / 3, 4}, []float32{6, 8, 6, 8, 2, -4}},
			{"unequal-sinks", []float32{0, float32(math.Ln2)}, nil, []float64{3.5, 3, 2.8, 2.4}, []float32{6, 8, 6, 8, 2, -4}},
			{"length-counts-slots", []float32{0, float32(math.Ln2)}, []int32{2}, []float64{4, 16.0 / 3, 3, 4}, []float32{6, 8, 6, 8}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				s, b := newSession(t)
				o := plainOpt
				o.TopKLength = tc.length
				beforeQ, beforeKV, beforeIdx := append([]float32(nil), q...), append([]float32(nil), kv...), append([]int32(nil), idx...)
				beforeSink := append([]float32(nil), tc.sink...)
				out, err := v41SparseAttentionSinkWithDevice(4, q, kv, tc.sink, idx, o, s.v41SharedAttentionFunc())
				if err != nil {
					t.Fatal(err)
				}
				v41SharedAttentionNear(t, out, tc.want)
				if len(b.calls) != 1 || !reflect.DeepEqual(b.calls[0].kv, tc.selected) || len(b.allocations) != 0 || b.uploads != 3 || b.reads != 1 {
					t.Fatalf("dispatch/gather/lifetime mismatch: %+v", b.calls)
				}
				sinkWidth := len(tc.sink)
				if tc.sink == nil {
					sinkWidth = 1
				}
				if b.uploadBytes != 4*(len(q)+len(tc.selected)+sinkWidth) {
					t.Fatal("sink/activation upload accounting mismatch")
				}
				if !reflect.DeepEqual(q, beforeQ) || !reflect.DeepEqual(kv, beforeKV) || !reflect.DeepEqual(idx, beforeIdx) || !reflect.DeepEqual(tc.sink, beforeSink) {
					t.Fatal("caller input mutated")
				}
				// A second call and poisoned readback must not alter a prior result.
				_, err = v41SparseAttentionSinkWithDevice(4, q, kv, tc.sink, idx, o, s.v41SharedAttentionFunc())
				if err != nil {
					t.Fatal(err)
				}
				v41SharedAttentionNear(t, out, tc.want)
			})
		}
	})

	t.Run("causal-groups-source-ratio-and-empty", func(t *testing.T) {
		values := [][]float32{{2, -4}, {6, 8}}
		for offset, wantRows := range [][]int{{}, {0}, {0}, {1, 0, 1}} {
			s, b := newSession(t)
			o := V41AttentionSharedKVOptions{Layer: 3, Ratio: 2, QueryOffset: offset, Groups: 2, Heads: 2, HeadDim: 2, Softmax: 1, Sink: []float32{0, 0}, Idx: []int32{1, 0, 1, -1}, IndexTopK: 4}
			req := v41SharedAttentionRequest{mode: compute.V41SharedAttentionCompressed, q: q, values: values, sink: o.Sink, idx: o.Idx, compressed: o}
			p, err := v41PrepareSharedAttention(req)
			if err != nil {
				t.Fatal(err)
			}
			if len(p.sourceRows) != len(wantRows) {
				t.Fatalf("offset %d rows=%v want=%v", offset, p.sourceRows, wantRows)
			}
			var selected [][]float32
			for i, r := range wantRows {
				if p.sourceRows[i] != r {
					t.Fatalf("slot %d got %d want %d", i, p.sourceRows[i], r)
				}
				selected = append(selected, values[r])
			}
			out, err := v41AttentionCompressedForwardWithDevice(q, values, o, s.v41SharedAttentionFunc())
			if err != nil {
				t.Fatal(err)
			}
			v41SharedAttentionNear(t, out, v41SharedAttentionRealOracle(q, selected, o.Sink, 2, 2, 1))
			if len(wantRows) == 0 {
				if b.uploads != 0 || len(b.calls) != 0 || b.reads != 0 {
					t.Fatal("empty selection touched the backend")
				}
				for _, v := range out {
					if math.Float32bits(v) != 0 {
						t.Fatal("empty result is not exact positive zero")
					}
				}
				out[0] = 99
				again, err := v41AttentionCompressedForwardWithDevice(q, values, o, s.v41SharedAttentionFunc())
				if err != nil || again[0] != 0 || q[0] != 0 {
					t.Fatal("empty result borrowed storage")
				}
			} else if len(b.calls) != 1 || b.calls[0].mode != compute.V41SharedAttentionCompressed {
				t.Fatal("compressed dispatch missing")
			}
		}
		// No index list means all complete groups in ascending source order.
		o := V41AttentionSharedKVOptions{Ratio: 2, QueryOffset: 3, Groups: 2, Heads: 2, HeadDim: 2, Softmax: 1}
		p, err := v41PrepareSharedAttention(v41SharedAttentionRequest{mode: compute.V41SharedAttentionCompressed, q: q, values: values, compressed: o})
		if err != nil || !reflect.DeepEqual(p.sourceRows, []int{0, 1}) {
			t.Fatalf("unindexed causal selection: %v, %v", p.sourceRows, err)
		}
	})

	t.Run("empty-plain-and-interior-padding", func(t *testing.T) {
		for _, sink := range [][]float32{nil, {0, 0}} {
			s, b := newSession(t)
			o := plainOpt
			o.TopKLength = []int32{0}
			out, err := v41SparseAttentionSinkWithDevice(0, q, kv, sink, idx, o, s.v41SharedAttentionFunc())
			if err != nil || b.uploads != 0 || len(b.calls) != 0 || b.reads != 0 {
				t.Fatalf("empty plain dispatch: %v", err)
			}
			for _, value := range out {
				if math.Float32bits(value) != 0 {
					t.Fatal("empty plain output is not exact zero")
				}
			}
		}
		s, b := newSession(t)
		o := plainOpt
		o.TopKLength = []int32{2}
		out, err := v41SparseAttentionSinkWithDevice(0, q, kv, []float32{0, 0}, []int32{1, -1, 0, 1}, o, s.v41SharedAttentionFunc())
		if err != nil {
			t.Fatal(err)
		}
		v41SharedAttentionNear(t, out, []float64{3, 4, 3, 4})
		if len(b.calls) != 1 || !reflect.DeepEqual(b.calls[0].kv, []float32{6, 8}) {
			t.Fatal("TopKLength counted selected rows instead of original slots")
		}
	})

	t.Run("validation-precedes-every-mask", func(t *testing.T) {
		for _, name := range []string{"padding-index", "unused-KV", "empty-query", "empty-sink", "empty-negative-index", "geometry-overflow", "compressed-future-row", "compressed-future-index", "compressed-width", "compressed-source-length", "inverse"} {
			t.Run(name, func(t *testing.T) {
				s, b := newSession(t)
				x, y, z, indices := append([]float32(nil), q...), append([]float32(nil), kv...), []float32{0, 0}, append([]int32(nil), idx...)
				o := plainOpt
				o.TopKLength = []int32{0}
				var err error
				switch name {
				case "padding-index":
					indices[3] = 2
				case "unused-KV":
					y[0] = float32(math.Inf(1))
				case "empty-query":
					x[0] = float32(math.NaN())
				case "empty-sink":
					z[1] = float32(math.NaN())
				case "empty-negative-index":
					indices[3] = -2
				case "geometry-overflow":
					o.Heads = math.MaxInt32
					o.HeadDim = 2
				case "inverse":
					o.Inverse = func(int, []float32) error { return nil }
					o.RopeDim = 2
				default:
					rows := [][]float32{{2, -4}, {6, 8}}
					co := V41AttentionSharedKVOptions{Layer: 2, Ratio: 2, QueryOffset: 0, Groups: 2, Heads: 2, HeadDim: 2, Softmax: 1, Idx: []int32{1}, IndexTopK: 1}
					if name == "compressed-future-row" {
						rows[1][0] = float32(math.NaN())
					}
					if name == "compressed-future-index" {
						co.Idx[0] = 2
					}
					if name == "compressed-width" {
						rows[1] = []float32{6}
					}
					if name == "compressed-source-length" {
						co.SourceRows = rows[:1]
					}
					_, err = v41AttentionCompressedForwardWithDevice(x, rows, co, s.v41SharedAttentionFunc())
				}
				if !strings.HasPrefix(name, "compressed") {
					_, err = v41SparseAttentionSinkWithDevice(2, x, y, z, indices, o, s.v41SharedAttentionFunc())
				}
				var selected *V41SharedAttentionOperationError
				var closed *BackendForwardOperationError
				if !errors.As(err, &selected) || !errors.As(err, &closed) || !errors.Is(err, ErrV41ForwardStage) || !s.BackendSessionClosed() || b.uploads != 0 || len(b.calls) != 0 {
					t.Fatalf("invalid input escaped validation/closure: %v", err)
				}
			})
		}
	})

	t.Run("selected-faults-close-without-replay", func(t *testing.T) {
		for _, fault := range []string{"query upload", "KV upload", "sink upload", "dispatch", "partial-output", "panic", "alias", "shape", "dtype", "read", "short-read", "nonfinite", "free"} {
			t.Run(fault, func(t *testing.T) {
				s, b := newSession(t)
				b.fault = fault
				b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
				attend := s.v41SharedAttentionFunc()
				out, err := v41SparseAttentionSinkWithDevice(7, q, kv, nil, idx, plainOpt, attend)
				var selected *V41SharedAttentionOperationError
				var closed *BackendForwardOperationError
				if out != nil || !errors.As(err, &selected) || !errors.As(err, &closed) || closed.Path != "v41-shared-attention" || closed.Layer != 7 || !s.BackendSessionClosed() || len(b.allocations) != 0 {
					t.Fatalf("selected fault escaped or leaked: out=%v err=%v live=%d", out, err, len(b.allocations))
				}
				if (strings.HasSuffix(fault, "upload") || fault == "dispatch" || fault == "partial-output" || fault == "panic" || fault == "read" || fault == "free") && !errors.Is(err, b.cause) {
					t.Fatal("typed backend cause lost")
				}
				calls := len(b.calls)
				var retry any
				func() {
					defer func() { retry = recover() }()
					_, _ = v41SparseAttentionSinkWithDevice(7, q, kv, nil, idx, plainOpt, attend)
				}()
				if retry != s.halFailure || len(b.calls) != calls {
					t.Fatal("closed session performed another dispatch")
				}
			})
		}
		for _, value := range []any{errors.New("unclassified selected panic"), "plain panic"} {
			s, b := newSession(t)
			b.fault = "panic"
			b.panicValue = value
			var got any
			func() {
				defer func() { got = recover() }()
				_, _ = v41SparseAttentionSinkWithDevice(7, q, kv, nil, idx, plainOpt, s.v41SharedAttentionFunc())
			}()
			var closed *BackendForwardOperationError
			if got != value || !errors.As(s.halFailure, &closed) || !s.BackendSessionClosed() || len(b.allocations) != 0 || len(b.calls) != 1 {
				t.Fatal("unclassified panic identity/cleanup lost")
			}
		}
	})

	t.Run("producer-attribution-keeps-duplicate-slot-and-cause", func(t *testing.T) {
		for _, tc := range []struct {
			stage         compute.V41SharedAttentionStage
			slot, element int
			producer      string
		}{
			{compute.V41SharedAttentionStageScore, 1, -1, "score accumulate"},
			{compute.V41SharedAttentionStageExp, 1, -1, "softmax denominator"},
			{compute.V41SharedAttentionStageExp, -1, -1, "softmax denominator"},
			{compute.V41SharedAttentionStageDenominator, -1, -1, "softmax denominator"},
			{compute.V41SharedAttentionStageWeight, 1, -1, "weighted value accumulate"},
			{compute.V41SharedAttentionStageValue, 1, 1, "weighted value accumulate"},
			{compute.V41SharedAttentionStageOutput, -1, 1, ""},
		} {
			t.Run(fmt.Sprintf("stage-%d-slot-%d", tc.stage, tc.slot), func(t *testing.T) {
				s, b := newSession(t)
				b.fault = "dispatch"
				a := &compute.V41SharedAttentionArithmeticError{Stage: tc.stage, Head: 1, SelectedSlot: tc.slot, Element: tc.element, ValueBits: 0x7f800000}
				b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage, Message: a.Error(), Recovered: a}
				o := V41AttentionSharedKVOptions{Layer: 9, Ratio: 2, QueryOffset: 5, Groups: 4, Heads: 2, HeadDim: 2, Softmax: 1, Sink: []float32{0, 0}, Idx: []int32{-1, 2, 2, 0, 3}, IndexTopK: 5}
				_, err := v41AttentionCompressedForwardWithDevice(q, [][]float32{{2, 3}, {4, 5}, {6, 7}, {8, 9}}, o, s.v41SharedAttentionFunc())
				var retained *compute.V41SharedAttentionArithmeticError
				if !errors.As(err, &retained) || retained != a || retained.ValueBits != 0x7f800000 || !errors.Is(err, b.cause) || len(b.allocations) != 0 {
					t.Fatalf("compute cause lost: %v", err)
				}
				if tc.producer != "" {
					if !strings.Contains(err.Error(), tc.producer) || !strings.Contains(err.Error(), "t=0 h=1") || !errors.Is(err, ErrV41ForwardStage) {
						t.Fatalf("producer attribution lost: %v", err)
					}
					if tc.slot >= 0 && !strings.Contains(err.Error(), "group=2") {
						t.Fatalf("duplicate slot mapped to wrong source group: %v", err)
					}
				} else if strings.Contains(err.Error(), "compressed attention") {
					t.Fatalf("output fault invented producer: %v", err)
				}
			})
		}
		s, b := newSession(t)
		b.fault = "dispatch"
		b.cause = &compute.V41SharedAttentionArithmeticError{Stage: compute.V41SharedAttentionStageScore, Head: 0, SelectedSlot: 99, Element: -1, ValueBits: 0x7f800000}
		_, err := v41SparseAttentionSinkWithDevice(0, q, kv, nil, idx, plainOpt, s.v41SharedAttentionFunc())
		var protocol *compute.V41SharedAttentionProtocolError
		if !errors.As(err, &protocol) || !errors.Is(err, b.cause) || errors.Is(err, ErrV41SparseSinkNonFinite) {
			t.Fatalf("malformed status not kept as protocol failure: %v", err)
		}
	})

	t.Run("plain-arithmetic-retains-cause-and-host-sentinel", func(t *testing.T) {
		for _, tc := range []struct {
			stage         compute.V41SharedAttentionStage
			slot, element int
		}{
			{compute.V41SharedAttentionStageScore, 0, -1},
			{compute.V41SharedAttentionStageExp, 0, -1},
			{compute.V41SharedAttentionStageDenominator, -1, -1},
			{compute.V41SharedAttentionStageWeight, 0, -1},
			{compute.V41SharedAttentionStageValue, 0, 0},
			{compute.V41SharedAttentionStageOutput, -1, 0},
		} {
			t.Run(fmt.Sprintf("stage-%d", tc.stage), func(t *testing.T) {
				s, b := newSession(t)
				arithmetic := &compute.V41SharedAttentionArithmeticError{Stage: tc.stage, Head: 0, SelectedSlot: tc.slot, Element: tc.element, ValueBits: math.Float32bits(float32(math.Inf(1)))}
				b.fault, b.cause = "dispatch", arithmetic
				out, err := v41SparseAttentionSinkWithDevice(0, q, kv, nil, idx, plainOpt, s.v41SharedAttentionFunc())
				var retained *compute.V41SharedAttentionArithmeticError
				if out != nil || !errors.Is(err, ErrV41SparseSinkNonFinite) || !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &retained) || retained != arithmetic || !errors.Is(err, arithmetic) || len(b.allocations) != 0 {
					t.Fatalf("plain arithmetic refusal lost: out=%v err=%v retained=%v", out, err, retained)
				}
			})
		}
	})

	t.Run("unsupported-host-and-selected-sentinel-overflow", func(t *testing.T) {
		one := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 1, N: 1, TopK: 1, Softmax: 1}
		for _, sign := range []float32{1, -1} {
			s, b := newSession(t)
			b.supported = false
			if s.v41SharedAttentionFunc() != nil {
				t.Fatal("unsupported backend selected")
			}
			out, err := v41SparseAttentionSinkWithDevice(0, []float32{sign * math.MaxFloat32}, []float32{2}, nil, []int32{0}, one, s.v41SharedAttentionFunc())
			// The host guard introduced by 1882b2f8849 refuses both signs of
			// arithmetic overflow rather than publishing NaN or a zero result.
			if out != nil || !errors.Is(err, ErrV41SparseSinkNonFinite) || len(b.calls) != 0 {
				t.Fatalf("host sparse-sink refusal lost: out=%v err=%v", out, err)
			}
			s, b = newSession(t)
			out, err = v41SparseAttentionSinkWithDevice(0, []float32{sign * math.MaxFloat32}, []float32{2}, nil, []int32{0}, one, s.v41SharedAttentionFunc())
			var arithmetic *compute.V41SharedAttentionArithmeticError
			if out != nil || !errors.Is(err, ErrV41SparseSinkNonFinite) || !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &arithmetic) || arithmetic.Stage != compute.V41SharedAttentionStageScore || arithmetic.SelectedSlot != 0 || arithmetic.ValueBits != math.Float32bits(sign*float32(math.Inf(1))) {
				t.Fatalf("selected score tightening lost: %v", err)
			}
		}
		for _, sink := range [][]float32{nil, {-math.MaxFloat32}} {
			s, _ := newSession(t)
			out, err := v41SparseAttentionSinkWithDevice(0, []float32{-math.MaxFloat32}, []float32{1}, sink, []int32{0}, one, s.v41SharedAttentionFunc())
			want := float32(1)
			if sink != nil {
				want = 0.5
			}
			if err != nil || out[0] != want {
				t.Fatalf("plain finite sentinel changed: %v %v", out, err)
			}
			s, _ = newSession(t)
			co := V41AttentionSharedKVOptions{Ratio: 1, Groups: 1, Heads: 1, HeadDim: 1, Softmax: 1, Sink: sink}
			out, err = v41AttentionCompressedForwardWithDevice([]float32{-math.MaxFloat32}, [][]float32{{1}}, co, s.v41SharedAttentionFunc())
			if err != nil || math.Float32bits(out[0]) != 0 {
				t.Fatalf("compressed finite sentinel changed: %v %v", out, err)
			}
		}
		for _, sink := range [][]float32{nil, {0}} {
			s, _ := newSession(t)
			out, err := v41SparseAttentionSinkWithDevice(0, []float32{1e30}, []float32{1}, sink, []int32{0}, one, s.v41SharedAttentionFunc())
			if err != nil || out[0] != 1 {
				t.Fatalf("finite huge one-row result changed: %v %v", out, err)
			}
		}
		s, _ := newSession(t)
		two := one
		two.N = 2
		two.TopK = 2
		out, err := v41SparseAttentionSinkWithDevice(0, []float32{math.MaxFloat32}, []float32{1, -1}, nil, []int32{0, 1}, two, s.v41SharedAttentionFunc())
		if err != nil || out[0] != 1 {
			t.Fatalf("finite score difference -Inf must exponentiate to zero: %v %v", out, err)
		}
		if (&Session{M: &Model{}, Backend: compute.Default()}).v41SharedAttentionFunc() != nil || (*Session)(nil).v41SharedAttentionFunc() != nil {
			t.Fatal("non-device/nil session selected")
		}
	})

	t.Run("independent-BF16-widened-512-column-control", func(t *testing.T) {
		const heads, dim = 2, 512
		x := make([]float32, heads*dim)
		rows := make([][]float32, 3)
		literals := []uint32{0x3f810000, 0xbea00000, 0x00000000, 0x3e800000, 0xbf020000}
		for i := range x {
			x[i] = float32(math.Sin(float64(i+3))*0.07 + 0.000013*float64(i%7))
		}
		var flat []float32
		for r := range rows {
			rows[r] = make([]float32, dim)
			for d := range rows[r] {
				rows[r][d] = math.Float32frombits(literals[(d+r)%len(literals)])
			}
			flat = append(flat, rows[r]...)
		}
		sink := []float32{0.2, -0.7}
		s, b := newSession(t)
		o := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: heads, HeadDim: dim, N: 3, TopK: 4, Softmax: -0.03125}
		out, err := v41SparseAttentionSinkWithDevice(0, x, flat, sink, []int32{2, 0, 2, 1}, o, s.v41SharedAttentionFunc())
		if err != nil {
			t.Fatal(err)
		}
		v41SharedAttentionNear(t, out, v41SharedAttentionRealOracle(x, [][]float32{rows[2], rows[0], rows[2], rows[1]}, sink, heads, dim, o.Softmax))
		if !reflect.DeepEqual(b.calls[0].q, x) || !reflect.DeepEqual(b.calls[0].kv[:dim], rows[2]) {
			t.Fatal("Q or widened KV was recast")
		}
	})
}

// fak-test:runtime fast est=5ms lane=default
// Estimate is unmeasured. CPU recorder checks the callback contract only; no
// current native-backend double fault or physical qualification is claimed.
func TestV41SharedAttentionPrimaryFailureSurvivesCleanup(t *testing.T) {
	for _, cleanup := range []any{&compute.BackendError{Backend: "cleanup", Class: compute.VulkanClassExecutionFailed, Err: errors.New("release failure")}, "unclassified release panic"} {
		for _, tc := range []struct {
			name, fault string
			panicValue  any
			owned       int
			cleanupOnly bool
		}{
			{name: "returned dispatch", fault: "dispatch", owned: 3},
			{name: "returned partial", fault: "partial-output", owned: 4},
			{name: "typed dispatch panic", fault: "panic", owned: 3},
			{name: "typed read panic", fault: "read", owned: 4},
			{name: "unknown pointer panic", fault: "panic", panicValue: &struct{ marker int }{17}, owned: 3},
			{name: "unknown error panic", fault: "panic", panicValue: errors.New("original unknown panic"), owned: 3},
			{name: "unknown string panic", fault: "panic", panicValue: "original panic", owned: 3},
			{name: "cleanup only", owned: 4, cleanupOnly: true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				b := newV41SharedAttentionTestBackend()
				s := &Session{M: &Model{}, Backend: b, halW: map[string]compute.Tensor{}}
				t.Cleanup(s.Close)
				primary := &compute.BackendError{Backend: "primary", Class: compute.VulkanClassExecutionFailed, Err: errors.New("selected operation failure")}
				b.fault, b.cause, b.panicValue, b.cleanupPanic = tc.fault, primary, tc.panicValue, cleanup
				var out []float32
				var err error
				recovered := v41IndexerTestRecover(func() {
					out, err = v41SparseAttentionSinkWithDevice(4, []float32{1}, []float32{2}, nil, []int32{0}, V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 1, N: 1, TopK: 1, Softmax: 1}, s.v41SharedAttentionFunc())
				})
				if tc.panicValue != nil {
					if recovered != tc.panicValue {
						t.Fatalf("primary panic identity lost: got %v want %v", recovered, tc.panicValue)
					}
				} else if tc.cleanupOnly {
					if cleanupErr, ok := cleanup.(error); ok {
						if recovered != nil || !errors.Is(err, cleanupErr) {
							t.Fatalf("cleanup-only typed failure changed: panic=%v err=%v", recovered, err)
						}
					} else if recovered != cleanup {
						t.Fatalf("cleanup-only panic changed: %v", recovered)
					}
				} else if recovered != nil || !errors.Is(err, primary) {
					t.Fatalf("primary selected cause lost: panic=%v err=%v", recovered, err)
				}
				var closed *BackendForwardOperationError
				if out != nil || !s.BackendSessionClosed() || !errors.As(s.halFailure, &closed) || closed.Layer != 4 || len(b.allocations) != 0 || b.freeAttempts != tc.owned || len(b.calls) != 1 {
					t.Fatalf("failure did not close and attempt every owned release: out=%v closed=%v live=%d free=%d calls=%d", out, s.halFailure, len(b.allocations), b.freeAttempts, len(b.calls))
				}
				if !tc.cleanupOnly && tc.panicValue == nil && !errors.Is(s.halFailure, primary) {
					t.Fatal("latched primary cause changed")
				}
				releases := b.freeAttempts
				retry := v41IndexerTestRecover(func() {
					_, _ = v41SparseAttentionSinkWithDevice(4, []float32{1}, []float32{2}, nil, []int32{0}, V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 1, N: 1, TopK: 1, Softmax: 1}, s.v41SharedAttentionFunc())
				})
				if retry != s.halFailure || len(b.calls) != 1 || b.freeAttempts != releases {
					t.Fatal("closed callback retried work or cleanup")
				}
			})
		}
	}
}
