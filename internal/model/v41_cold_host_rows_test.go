package model

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"runtime"
	"runtime/pprof"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

// Frozen public24344f57 oracle. The combined body only renames its function and
// redirects its original sparse dispatch to the frozen primitive below.
func v41ColdAttentionCombinedOracle(layer, pos, sourceRatio, heads, headDim int, q, sink []float32, window, compressed [][]float32, compressedIDs []int32, scale float32, resident *v41ResidentKVView, attend v41SharedAttentionFunc, continuation *v41SharedAttentionContinuation) ([]float32, error) {
	fail := func(message string) ([]float32, error) {
		return nil, v41StageErr(v41StageAttention, layer, fmt.Errorf("%w: combined attention %s", ErrV41ForwardStage, message))
	}
	if layer < 0 || pos < 0 || sourceRatio <= 0 || heads <= 0 || headDim <= 0 {
		return fail("has invalid position or geometry")
	}
	maxInt := int(^uint(0) >> 1)
	if len(window) > math.MaxInt32 || len(compressed) > math.MaxInt32-len(window) || len(compressedIDs) > maxInt-len(window) {
		return fail("row or index count overflows")
	}
	rows := len(window) + len(compressed)
	slots := len(window) + len(compressedIDs)
	qCount, qOK := checkedMulInt(heads, headDim)
	kvCount, kvOK := checkedMulInt(rows, headDim)
	_, qBytesOK := checkedMulInt(qCount, 4)
	_, kvBytesOK := checkedMulInt(kvCount, 4)
	_, idxBytesOK := checkedMulInt(slots, 4)
	if !qOK || !kvOK || !qBytesOK || !kvBytesOK || !idxBytesOK || (attend != nil && slots > math.MaxInt32) ||
		len(q) != qCount || (sink != nil && len(sink) != heads) {
		return fail("input shape or byte count is invalid")
	}
	if !finite32(scale) || scale == 0 {
		return fail("scale must be finite and nonzero")
	}
	for _, values := range [][]float32{q, sink} {
		if err := finiteRow32(values, "combined attention input"); err != nil {
			return nil, v41StageErr(v41StageAttention, layer, err)
		}
	}
	// Validate every original row and ID before causal masking. An invalid future
	// row or padding ID cannot disappear merely because this query cannot see it.
	for _, stream := range [][][]float32{window, compressed} {
		for _, row := range stream {
			if len(row) != headDim {
				return fail("row width differs from head width")
			}
			if err := finiteRow32(row, "combined attention row"); err != nil {
				return nil, v41StageErr(v41StageAttention, layer, err)
			}
		}
	}
	for _, id := range compressedIDs {
		if id < -1 || int(id) >= len(compressed) {
			return fail("canonical compressed row ID is out of range")
		}
	}
	idx := make([]int32, 0, slots)
	for i := range window {
		idx = append(idx, int32(i))
	}
	for _, id := range compressedIDs {
		// Avoid (pos+1)/ratio and (id+1)*ratio overflow at extreme positions.
		if id < 0 || pos < sourceRatio-1 || int(id) > (pos-(sourceRatio-1))/sourceRatio {
			idx = append(idx, -1)
		} else {
			idx = append(idx, int32(len(window)+int(id)))
		}
	}
	opt := V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: heads, HeadDim: headDim, N: rows, TopK: len(idx), Softmax: scale,
	}
	var out []float32
	var ownedOutput bool
	var err error
	if attend != nil && resident != nil && !v41ResidentKVDisabled {
		request := v41SharedAttentionRequest{
			mode: compute.V41SharedAttentionPlain, q: q, sink: sink, idx: idx, plain: opt,
			rowKV:    &v41SharedAttentionRows{window: window, compressed: compressed},
			resident: resident, sinkKey: resident.sinkKey,
		}
		c := continuation
		if c != nil && !c.used && c.layer == layer && c.pos == pos && c.sourceRatio == sourceRatio &&
			c.heads == heads && c.headDim == headDim && c.scale == scale && c.live() {
			// All original rows were checked above. Nothing can call user code
			// between sealing this local request and the private preparation.
			validation := &v41SharedAttentionValidation{continuation: c, request: &request, rows: request.rowKV,
				window: window, compressed: compressed}
			var result v41SharedAttentionResult
			result, err = c.binding.native.core.validated(layer, &request, validation)
			out, ownedOutput = result.values, err == nil
			if err != nil {
				err = &V41SharedAttentionOperationError{Layer: layer, Cause: v41StageErr(v41StageAttention, layer, err)}
			}
		} else {
			out, err = v41SelectedSharedAttention(layer, request, attend)
		}
	} else {
		flat := make([]float32, 0, kvCount)
		for _, stream := range [][][]float32{window, compressed} {
			for _, row := range stream {
				flat = append(flat, row...)
			}
		}
		out, err = v41ColdAttentionResidentOracle(layer, q, flat, sink, idx, opt, resident, attend)
	}
	if err != nil {
		return nil, err
	}
	if len(out) != qCount {
		_, err = fail("returned an invalid output width")
	} else if finiteErr := finiteRow32(out, "combined attention output"); finiteErr != nil {
		err = v41StageErr(v41StageAttention, layer, finiteErr)
	}
	if err != nil {
		if attend != nil {
			return nil, &V41SharedAttentionOperationError{Layer: layer, Cause: err}
		}
		return nil, err
	}
	// The nil callback uses the host primitive, which owns its fresh result.
	// Ordinary callbacks can return borrowed storage and still need a copy.
	if ownedOutput || attend == nil {
		return out, nil
	}
	return append([]float32(nil), out...), nil
}

func v41ColdAttentionSparseOracle(q, kv []float32, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions) ([]float32, error) {
	if opt.B <= 0 || opt.M <= 0 || opt.Heads <= 0 || opt.HeadDim <= 0 || opt.TopK < 0 || opt.N < 0 {
		return nil, fmt.Errorf("model: invalid V4.1 sparse sink geometry b=%d m=%d heads=%d dim=%d topk=%d n=%d", opt.B, opt.M, opt.Heads, opt.HeadDim, opt.TopK, opt.N)
	}
	if !finite32(opt.Softmax) || opt.Softmax == 0 {
		return nil, fmt.Errorf("model: V4.1 sparse sink softmax scale must be finite and non-zero, got %g", opt.Softmax)
	}
	if opt.Inverse != nil && (opt.RopeDim <= 0 || opt.RopeDim > opt.HeadDim || opt.RopeDim%2 != 0) {
		return nil, fmt.Errorf("model: V4.1 sparse sink inverse rotation needs an even rope dim in (0,%d], got %d", opt.HeadDim, opt.RopeDim)
	}
	if len(q) != opt.B*opt.M*opt.Heads*opt.HeadDim {
		return nil, fmt.Errorf("model: V4.1 sparse sink query length %d, want %d", len(q), opt.B*opt.M*opt.Heads*opt.HeadDim)
	}
	if len(kv) != opt.B*opt.N*opt.HeadDim {
		return nil, fmt.Errorf("model: V4.1 sparse sink kv length %d, want %d", len(kv), opt.B*opt.N*opt.HeadDim)
	}
	if len(idx) != opt.B*opt.M*opt.TopK {
		return nil, fmt.Errorf("model: V4.1 sparse sink index length %d, want %d", len(idx), opt.B*opt.M*opt.TopK)
	}
	if sink != nil && len(sink) != opt.Heads {
		return nil, fmt.Errorf("model: V4.1 sparse sink length %d, want %d", len(sink), opt.Heads)
	}
	if opt.TopKLength != nil && len(opt.TopKLength) != opt.B*opt.M {
		return nil, fmt.Errorf("model: V4.1 sparse sink topk length %d, want %d", len(opt.TopKLength), opt.B*opt.M)
	}
	if opt.TopKLength != nil {
		for i, v := range opt.TopKLength {
			if v < 0 || int(v) > opt.TopK {
				return nil, fmt.Errorf("model: V4.1 sparse sink topk length %d out of range at %d", v, i)
			}
		}
	}
	for i, v := range q {
		if !finite32(v) {
			return nil, fmt.Errorf("model: V4.1 sparse sink non-finite query element %d", i)
		}
	}
	for i, v := range kv {
		if !finite32(v) {
			return nil, fmt.Errorf("model: V4.1 sparse sink non-finite kv element %d", i)
		}
	}
	if sink != nil {
		for i, v := range sink {
			if !finite32(v) {
				return nil, fmt.Errorf("model: V4.1 sparse sink non-finite sink value %d", i)
			}
		}
	}
	for i, v := range idx {
		if v < -1 || int(v) >= opt.N {
			return nil, fmt.Errorf("model: V4.1 sparse sink row index %d out of range at %d", v, i)
		}
	}

	negInf := float32(math.Inf(-1))
	o := make([]float32, opt.B*opt.M*opt.Heads*opt.HeadDim)
	for b := 0; b < opt.B; b++ {
		for m := 0; m < opt.M; m++ {
			idxBase := (b*opt.M + m) * opt.TopK
			// validCount is the number of leading slots that count; slots at
			// index >= validCount are padding and contribute nothing.
			validCount := opt.TopK
			if opt.TopKLength != nil {
				validCount = int(opt.TopKLength[b*opt.M+m])
			}
			for h := 0; h < opt.Heads; h++ {
				qBase := ((b*opt.M+m)*opt.Heads + h) * opt.HeadDim
				// Stage 1: max over valid scaled scores, seeded with the sink.
				maxScore := negInf
				if sink != nil {
					maxScore = sink[h]
				}
				for i := 0; i < validCount; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					kvBase := (b*opt.N + row) * opt.HeadDim
					var dot float32
					for d := 0; d < opt.HeadDim; d++ {
						dot += float32(q[qBase+d] * kv[kvBase+d])
					}
					dot *= opt.Softmax
					if !finite32(dot) {
						return nil, sparseSinkNonFinite(v41SinkStageScore, b, m, h, row, -1, dot)
					}
					if dot > maxScore {
						maxScore = dot
					}
				}
				if maxScore == negInf {
					// No valid row and no sink: the reference's finite bound
					// yields an all-zero output row.
					continue
				}
				// Stage 2: softmax denominator with the sink included.
				var sum float32
				if sink != nil {
					sum = exp32(sink[h] - maxScore)
				}
				for i := 0; i < validCount; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					kvBase := (b*opt.N + row) * opt.HeadDim
					var dot float32
					for d := 0; d < opt.HeadDim; d++ {
						dot += float32(q[qBase+d] * kv[kvBase+d])
					}
					// Match the rounded float32 score used by the maximum pass.
					// A finite score gap may round to -Inf; exp(-Inf)=0 is valid.
					term := exp32(float32(dot*opt.Softmax) - maxScore)
					if !finite32(term) {
						return nil, sparseSinkNonFinite(v41SinkStageSoftmax, b, m, h, row, -1, term)
					}
					sum += term
				}
				if !finite32(sum) {
					return nil, sparseSinkNonFinite(v41SinkStageSoftmax, b, m, h, -1, -1, sum)
				}
				if sum == 0 {
					continue
				}
				// Stage 3: weighted value sum, normalized by the denominator.
				oBase := qBase
				for i := 0; i < validCount; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					kvBase := (b*opt.N + row) * opt.HeadDim
					var dot float32
					for d := 0; d < opt.HeadDim; d++ {
						dot += float32(q[qBase+d] * kv[kvBase+d])
					}
					weight := exp32(float32(dot*opt.Softmax)-maxScore) / sum
					if !finite32(weight) {
						return nil, sparseSinkNonFinite(v41SinkStageValue, b, m, h, row, -1, weight)
					}
					for d := 0; d < opt.HeadDim; d++ {
						o[oBase+d] += float32(weight * kv[kvBase+d])
						if !finite32(o[oBase+d]) {
							return nil, sparseSinkNonFinite(v41SinkStageValue, b, m, h, row, d, o[oBase+d])
						}
					}
				}
			}
		}
	}
	if opt.Inverse != nil {
		for b := 0; b < opt.B; b++ {
			for m := 0; m < opt.M; m++ {
				for h := 0; h < opt.Heads; h++ {
					headBase := ((b*opt.M+m)*opt.Heads + h) * opt.HeadDim
					tail := o[headBase+opt.HeadDim-opt.RopeDim : headBase+opt.HeadDim]
					if err := opt.Inverse(opt.RopeDim, tail); err != nil {
						return nil, err
					}
				}
			}
		}
		// A callback can retain a previously supplied tail. Validate the final
		// tensor after every callback has returned, not only its latest tail.
		for i, value := range o {
			if !finite32(value) {
				head := i / opt.HeadDim
				return nil, sparseSinkNonFinite(v41SinkStageInverse, head/(opt.M*opt.Heads), head/opt.Heads%opt.M, head%opt.Heads, -1, i%opt.HeadDim, value)
			}
		}
	}
	return o, nil
}

func v41ColdAttentionResidentOracle(layer int, q, kv, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions, resident *v41ResidentKVView, attend v41SharedAttentionFunc) ([]float32, error) {
	if attend == nil {
		return v41ColdAttentionSparseOracle(q, kv, sink, idx, opt)
	}
	return v41SparseAttentionSinkResident(layer, q, kv, sink, idx, opt, resident, attend)
}

func v41ColdAttentionEqual(t *testing.T, got, want []float32, ge, we error) {
	t.Helper()
	if fmt.Sprint(ge) != fmt.Sprint(we) || (got == nil) != (want == nil) || len(got) != len(want) {
		t.Fatalf("result mismatch: %v/%v errors %v/%v", got, want, ge, we)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("bit mismatch at%d: %x/%x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
	for _, sentinel := range []error{ErrV41SparseSinkNonFinite, ErrV41ForwardStage} {
		if errors.Is(ge, sentinel) != errors.Is(we, sentinel) {
			t.Fatal("error identity mismatch")
		}
	}
}

func TestV41ColdAttentionRows(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*v41CombinedHostFixture, *int, *int, *float32)
	}{
		{"duplicates_padding_causal_sink", func(*v41CombinedHostFixture, *int, *int, *float32) {}},
		{"empty", func(f *v41CombinedHostFixture, _, _ *int, _ *float32) {
			f.window = nil
			f.compressed = nil
			f.ids = nil
			f.sink = nil
		}},
		{"all_masked_sink", func(f *v41CombinedHostFixture, p, _ *int, _ *float32) { f.window = nil; *p = 0 }},
		{"all_masked_no_sink", func(f *v41CombinedHostFixture, p, _ *int, _ *float32) { f.window = nil; f.sink = nil; *p = 0 }},
		{"shared_rows_and_query", func(f *v41CombinedHostFixture, _, _ *int, _ *float32) {
			f.window[0] = f.q[:f.dim]
			f.compressed[0] = f.window[0]
			f.window[1] = f.window[0]
		}},
		{"signed_zero", func(f *v41CombinedHostFixture, _, _ *int, _ *float32) {
			for i := range f.q {
				f.q[i] = math.Float32frombits(0x80000000)
			}
		}},
		{"finite_score_overflow", func(f *v41CombinedHostFixture, _, _ *int, _ *float32) {
			f.q[0] = math.MaxFloat32
			f.window[0][0] = math.MaxFloat32
		}},
		{"future_row_nonfinite", func(f *v41CombinedHostFixture, p, _ *int, _ *float32) {
			*p = 0
			f.compressed[3][0] = float32(math.NaN())
		}},
		{"future_row_width", func(f *v41CombinedHostFixture, p, _ *int, _ *float32) { *p = 0; f.compressed[3] = nil }},
		{"shape_before_scale", func(f *v41CombinedHostFixture, _, _ *int, s *float32) { f.q = nil; *s = 0 }},
		{"scale_before_rows", func(f *v41CombinedHostFixture, _, _ *int, s *float32) { *s = 0; f.window[0] = nil }},
		{"query_before_sink", func(f *v41CombinedHostFixture, _, _ *int, _ *float32) {
			f.q[0] = float32(math.NaN())
			f.sink[0] = float32(math.Inf(1))
		}},
		{"rows_before_ids", func(f *v41CombinedHostFixture, _, _ *int, _ *float32) { f.window[0] = nil; f.ids[0] = -2 }},
		{"invalid_padding_id", func(f *v41CombinedHostFixture, p, _ *int, _ *float32) { *p = 0; f.ids[0] = -2 }},
		{"geometry", func(_ *v41CombinedHostFixture, _, r *int, _ *float32) { *r = 0 }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newV41CombinedHostFixture()
			pos, ratio, scale := 5, 2, float32(.125)
			c.mutate(&f, &pos, &ratio, &scale)
			want, we := v41ColdAttentionCombinedOracle(0, pos, ratio, f.heads, f.dim, f.q, f.sink, f.window, f.compressed, f.ids, scale, nil, nil, nil)
			got, ge := v41CombinedAttentionWithContinuation(0, pos, ratio, f.heads, f.dim, f.q, f.sink, f.window, f.compressed, f.ids, scale, nil, nil, nil)
			v41ColdAttentionEqual(t, got, want, ge, we)
			if ge == nil && len(got) > 0 {
				got[0] = 123
				again, ae := v41CombinedAttentionWithContinuation(0, pos, ratio, f.heads, f.dim, f.q, f.sink, f.window, f.compressed, f.ids, scale, nil, nil, nil)
				v41ColdAttentionEqual(t, again, want, ae, we)
			}
		})
	}
	t.Run("selected_failure_no_replay", func(t *testing.T) {
		f := newV41CombinedHostFixture()
		sentinel := errors.New("selected failure")
		calls := 0
		cb := func(_ int, r v41SharedAttentionRequest) ([]float32, error) {
			calls++
			if !reflect.DeepEqual(r.idx, f.wantIdx) {
				t.Fatal("changed selected index order")
			}
			return nil, sentinel
		}
		want, we := v41ColdAttentionCombinedOracle(0, 5, 2, f.heads, f.dim, f.q, f.sink, f.window, f.compressed, f.ids, .125, nil, cb, nil)
		got, ge := v41CombinedAttentionWithContinuation(0, 5, 2, f.heads, f.dim, f.q, f.sink, f.window, f.compressed, f.ids, .125, nil, cb, nil)
		v41ColdAttentionEqual(t, got, want, ge, we)
		if calls != 2 || !errors.Is(ge, sentinel) {
			t.Fatal("selected failure lost or replayed")
		}
	})
	t.Run("allocation", func(t *testing.T) {
		f := newV41CombinedHostFixture()
		var out []float32
		var err error
		old := testing.AllocsPerRun(100, func() {
			out, err = v41ColdAttentionCombinedOracle(0, 5, 2, f.heads, f.dim, f.q, f.sink, f.window, f.compressed, f.ids, .125, nil, nil, nil)
		})
		current := testing.AllocsPerRun(100, func() { out, err = f.compose(nil) })
		if err != nil || len(out) != len(f.q) {
			t.Fatal(err)
		}
		t.Logf("combined allocations old=%g current=%g", old, current)
		if current >= old {
			t.Fatalf("row-backed contraction did not remove an allocation: %g >= %g", current, old)
		}
	})
}

func TestV41ColdAttentionPublicSparse(t *testing.T) {
	for _, mode := range []string{"batch", "masked_bad_id", "masked_nan", "length", "inverse_error", "inverse_nonfinite"} {
		t.Run(mode, func(t *testing.T) {
			opt := V41SparseAttentionSinkOptions{B: 2, M: 2, Heads: 2, HeadDim: 4, N: 3, TopK: 3, Softmax: .25, TopKLength: []int32{2, 0, 3, 1}}
			q := make([]float32, 32)
			kv := make([]float32, 24)
			for i := range q {
				q[i] = float32(i%7-3) / 8
			}
			for i := range kv {
				kv[i] = float32(i%5-2) / 8
			}
			ids := []int32{0, 0, -1, 1, 2, -1, 2, 1, 0, -1, 0, 1}
			sink := []float32{0, -1}
			calls := 0
			sentinel := errors.New("inverse stop")
			switch mode {
			case "masked_bad_id":
				ids[5] = 9
			case "masked_nan":
				kv[0] = float32(math.NaN())
			case "length":
				opt.TopKLength[0] = 4
			case "inverse_nonfinite":
				opt.RopeDim = 2
				opt.Inverse = func(_ int, tail []float32) error {
					calls++
					if calls == 3 {
						tail[0] = float32(math.Inf(1))
					}
					return nil
				}
			case "inverse_error":
				opt.RopeDim = 2
				opt.Inverse = func(_ int, _ []float32) error {
					calls++
					if calls == 3 {
						return sentinel
					}
					return nil
				}
			}
			want, we := v41ColdAttentionSparseOracle(q, kv, sink, ids, opt)
			wantCalls := calls
			calls = 0
			got, ge := V41SparseAttentionSink(q, kv, sink, ids, opt)
			v41ColdAttentionEqual(t, got, want, ge, we)
			if calls != wantCalls || errors.Is(ge, sentinel) != errors.Is(we, sentinel) {
				t.Fatal("inverse callback contract changed")
			}
		})
	}
}

func BenchmarkV41ColdAttentionRows(b *testing.B) {
	for _, n := range []int{16, 128, 512} {
		for _, arm := range []string{"oracle", "candidate"} {
			b.Run(fmt.Sprintf("rows%d/%s", n, arm), func(b *testing.B) {
				f := newV41CombinedHostFixture()
				f.dim = 512
				f.heads = 8
				f.q = make([]float32, 4096)
				f.sink = make([]float32, 8)
				f.window = make([][]float32, 128)
				f.compressed = make([][]float32, n)
				f.ids = []int32{0, int32(n - 1), 0, -1}
				for _, rows := range [][][]float32{f.window, f.compressed} {
					for i := range rows {
						rows[i] = make([]float32, 512)
						for j := range rows[i] {
							rows[i][j] = float32((i+j)%13-6) / 32
						}
					}
				}
				for i := range f.q {
					f.q[i] = float32(i%7-3) / 32
				}
				run := v41CombinedAttentionWithContinuation
				if arm == "oracle" {
					run = v41ColdAttentionCombinedOracle
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					out, err := run(0, 2*n, 2, f.heads, f.dim, f.q, f.sink, f.window, f.compressed, f.ids, .125, nil, nil, nil)
					if err != nil || len(out) != len(f.q) {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// fak-test:justify why=cost when=explicit:FAK_TEST_COLD_ATTENTION_ROWS
// fak-test:runtime slow est=120s lane=optin
func TestV41ColdAttentionSessionScaling(t *testing.T) {
	raw := os.Getenv("FAK_TEST_COLD_ATTENTION_ROWS")
	if raw == "" {
		t.Skip("set FAK_TEST_COLD_ATTENTION_ROWS to16,128,1024")
	}
	n, err := strconv.Atoi(raw)
	if err != nil || (n != 16 && n != 128 && n != 1024) {
		t.Fatal("expected16,128,1024")
	}
	m := v41IndexerTestFixture(t)
	m.Cfg.Window = []int{128, 128}
	if compute.Default().Name() != "cpu-ref" {
		t.Fatal("requires cpu-ref")
	}
	stepobs.Attach()
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	s, err := m.NewBackendSessionChecked(compute.Default())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	ids := make([]int, n)
	for i := range ids {
		ids[i] = i % m.Cfg.VocabSize
	}
	before := v41IndexerTestPhase(t, m, "prefill")
	var profile *os.File
	stopProfile := func() {}
	if path := os.Getenv("FAK_TEST_COLD_ATTENTION_PROFILE"); path != "" {
		profile, err = os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if err = pprof.StartCPUProfile(profile); err != nil {
			profile.Close()
			t.Fatal(err)
		}
		active := true
		stopProfile = func() {
			if !active {
				return
			}
			active = false
			pprof.StopCPUProfile()
			if closeErr := profile.Close(); closeErr != nil {
				t.Errorf("close CPU profile: %v", closeErr)
			}
		}
		defer stopProfile()
	}
	var pre, post runtime.MemStats
	runtime.ReadMemStats(&pre)
	started := time.Now()
	logits := s.Prefill(ids)
	elapsed := time.Since(started)
	runtime.ReadMemStats(&post)
	stopProfile()
	if len(logits) == 0 {
		t.Fatal("no logits")
	}
	for _, v := range logits {
		if !finite32(v) {
			t.Fatal("nonfinite logits")
		}
	}
	rows, ratio, ok := s.v41Forward.attn.TopK()
	if !ok || ratio != 2 || len(rows) != n {
		t.Fatal("missing TopK publication")
	}
	for pos, row := range rows {
		v41IndexerTestSelectionCount(t, row, (pos+1)/2, m.Cfg.IndexTopK)
	}
	encodedRows, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	delta := v41DenseTestDelta(v41IndexerTestPhase(t, m, "prefill"), before)
	if delta["indexer_projection_host_rows"] != float64(2*n+n/2) || delta["indexer_projection_device_calls"] != 0 {
		t.Fatal("unexpected projection route", delta)
	}
	record := struct {
		Rows                    int   `json:"rows"`
		ElapsedNS               int64 `json:"elapsed_ns"`
		Mallocs, AllocatedBytes uint64
		TopKSHA                 string
		Logits                  []float32
		Counters                map[string]float64
	}{n, elapsed.Nanoseconds(), post.Mallocs - pre.Mallocs, post.TotalAlloc - pre.TotalAlloc, fmt.Sprintf("%x", sha256.Sum256(encodedRows)), logits, delta}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("COLD_ATTENTION_SCALING %s\n", encoded)
}
