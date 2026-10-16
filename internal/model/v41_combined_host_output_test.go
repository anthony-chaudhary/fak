package model

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41CombinedHostFixture struct {
	heads, dim         int
	q, sink            []float32
	window, compressed [][]float32
	ids, wantIdx       []int32
}

func newV41CombinedHostFixture() v41CombinedHostFixture {
	f := v41CombinedHostFixture{heads: 4, dim: 32,
		ids: []int32{1, 0, 1, -1, 3}, wantIdx: []int32{0, 1, 3, 2, 3, -1, -1}}
	f.q, f.sink = make([]float32, f.heads*f.dim), make([]float32, f.heads)
	for i := range f.q {
		f.q[i] = float32(i%11-5) / 16
	}
	for i := range f.sink {
		f.sink[i] = float32(i-2) / 8
	}
	f.window, f.compressed = make([][]float32, 2), make([][]float32, 4)
	for stream, rows := range [][][]float32{f.window, f.compressed} {
		for i := range rows {
			rows[i] = make([]float32, f.dim)
			for j := range rows[i] {
				rows[i][j] = float32((stream*13+i*7+j)%19-9) / 16
			}
		}
	}
	return f
}

func v41CombinedHostPack(window, compressed [][]float32, dim int) []float32 {
	out := make([]float32, 0, (len(window)+len(compressed))*dim)
	for _, rows := range [][][]float32{window, compressed} {
		for _, row := range rows {
			out = append(out, row...)
		}
	}
	return out
}

func (f v41CombinedHostFixture) compose(attend v41SharedAttentionFunc) ([]float32, error) {
	return v41CombinedAttention(0, 5, 2, f.heads, f.dim, f.q, f.sink,
		f.window, f.compressed, f.ids, .125, attend)
}

// Independent fixture packing includes the same necessary index/row ownership
// as the composer. The literal index schedule covers duplicate, padding and
// future groups; this does not call the composer or add a synthetic clone.
func (f v41CombinedHostFixture) control() ([]float32, error) {
	idx := make([]int32, len(f.wantIdx))
	copy(idx, f.wantIdx)
	flat := v41CombinedHostPack(f.window, f.compressed, f.dim)
	return V41SparseAttentionSink(f.q, flat, f.sink, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: f.heads, HeadDim: f.dim,
		N: len(f.window) + len(f.compressed), TopK: len(idx), Softmax: .125,
	})
}

func v41CombinedHostBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) == 0 || len(got) != len(want) {
		t.Fatalf("output widths=%d/%d", len(got), len(want))
	}
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) ||
			math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("output[%d]=%g, want exact %g", i, got[i], want[i])
		}
	}
}

// Host allocation and ordinary-callback ownership controls, followed by actual
// source/reader Session Prefill and Step. Runtime is an unmeasured estimate.
// No parallel execution: the incremental route observer is package-global.
// fak-test:runtime medium est=10s lane=default
func TestV41CombinedHostOutput(t *testing.T) {
	t.Run("ownership", func(t *testing.T) {
		f := newV41CombinedHostFixture()
		inputQ, inputSink := append([]float32(nil), f.q...), append([]float32(nil), f.sink...)
		inputKV, inputIDs := v41CombinedHostPack(f.window, f.compressed, f.dim), append([]int32(nil), f.ids...)
		want, err := f.control()
		if err != nil {
			t.Fatal(err)
		}
		got, err := f.compose(nil)
		if err != nil {
			t.Fatal(err)
		}
		v41CombinedHostBits(t, got, want)
		got[0] = 37
		again, err := f.compose(nil)
		if err != nil {
			t.Fatal(err)
		}
		v41CombinedHostBits(t, again, want)
		borrowed := append([]float32(nil), want...)
		calls := 0
		copied, err := f.compose(func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
			calls++
			if layer != 0 || r.mode != compute.V41SharedAttentionPlain ||
				!reflect.DeepEqual(r.idx, f.wantIdx) || !reflect.DeepEqual(r.kv, inputKV) {
				t.Fatal("ordinary callback received a different logical contraction")
			}
			return borrowed, nil
		})
		if err != nil || calls != 1 {
			t.Fatalf("ordinary callback calls=%d err=%v", calls, err)
		}
		v41CombinedHostBits(t, copied, want)
		copied[0] = 41
		if borrowed[0] != want[0] {
			t.Fatal("composer output aliases borrowed callback storage")
		}
		borrowed[1] = 43
		if copied[1] != want[1] {
			t.Fatal("callback storage aliases composer output")
		}
		if !reflect.DeepEqual(inputQ, f.q) || !reflect.DeepEqual(inputSink, f.sink) ||
			!reflect.DeepEqual(inputKV, v41CombinedHostPack(f.window, f.compressed, f.dim)) ||
			!reflect.DeepEqual(inputIDs, f.ids) {
			t.Fatal("host output aliases or mutates logical inputs")
		}
	})

	t.Run("allocation", func(t *testing.T) {
		f := newV41CombinedHostFixture()
		inputQ, inputSink := append([]float32(nil), f.q...), append([]float32(nil), f.sink...)
		inputKV, inputIDs := v41CombinedHostPack(f.window, f.compressed, f.dim), append([]int32(nil), f.ids...)
		got, gotErr := f.compose(nil)
		want, wantErr := f.control()
		if gotErr != nil || wantErr != nil {
			t.Fatalf("composer/control errors=%v/%v", gotErr, wantErr)
		}
		v41CombinedHostBits(t, got, want)
		composeAllocs := testing.AllocsPerRun(100, func() { got, gotErr = f.compose(nil) })
		controlAllocs := testing.AllocsPerRun(100, func() { want, wantErr = f.control() })
		if gotErr != nil || wantErr != nil {
			t.Fatalf("measured composer/control errors=%v/%v", gotErr, wantErr)
		}
		v41CombinedHostBits(t, got, want)
		if !reflect.DeepEqual(inputQ, f.q) || !reflect.DeepEqual(inputSink, f.sink) ||
			!reflect.DeepEqual(inputKV, v41CombinedHostPack(f.window, f.compressed, f.dim)) ||
			!reflect.DeepEqual(inputIDs, f.ids) {
			t.Fatal("allocation sample mutated inputs")
		}
		if composeAllocs > controlAllocs {
			t.Fatalf("redundant combined host output copy: composer=%g packed_host=%g parity=true inputs_unchanged=true output_width=%d",
				composeAllocs, controlAllocs, len(got))
		}
	})

	t.Run("session", func(t *testing.T) {
		m := v41ReaderWidthFixture(t)
		t.Cleanup(func() {
			if err := m.CloseWeights(); err != nil {
				t.Errorf("close weights: %v", err)
			}
		})
		host, callback := m.NewSession(), m.NewSession()
		t.Cleanup(host.Close)
		t.Cleanup(callback.Close)
		if host.Backend != nil || callback.Backend != nil || host.v41State().sharedAttentionFunc() != nil {
			t.Fatal("fixture must select ordinary host attention")
		}
		calls := make([]int, m.Cfg.NumLayers)
		callback.v41State().setSharedAttention(func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
			if layer < 0 || layer >= len(calls) || r.mode != compute.V41SharedAttentionPlain {
				t.Fatal("unexpected source/reader contraction")
			}
			calls[layer]++
			flat := r.kv
			if r.rowKV != nil {
				flat = v41CombinedHostPack(r.rowKV.window, r.rowKV.compressed, r.plain.HeadDim)
			}
			return V41SparseAttentionSink(r.q, flat, r.sink, r.idx, r.plain)
		})
		prefix := []int{1, 2, 3, 4}
		v41CombinedHostBits(t, host.Prefill(prefix), callback.Prefill(prefix))
		if !host.v41IncrementalEligible() || !callback.v41IncrementalEligible() {
			t.Fatal("source/reader Prefill did not seed incremental state")
		}
		for _, count := range calls {
			if count != len(prefix) {
				t.Fatalf("Prefill callback calls=%v, want one per input and layer", calls)
			}
		}
		hostRegistry, callbackRegistry := host.v41Forward.attn, callback.v41Forward.attn
		readSteps, restore := v41SessionIncrementalCalls(t)
		defer restore()
		continuation := []int{5}
		for _, token := range continuation {
			v41CombinedHostBits(t, host.Step(token), callback.Step(token))
		}
		history := append(append([]int(nil), prefix...), continuation...)
		sessions := []*Session{host, callback}
		if readSteps() != len(sessions)*len(continuation) ||
			host.v41Forward.attn != hostRegistry || callback.v41Forward.attn != callbackRegistry {
			t.Fatal("Step did not retain and advance the actual incremental source/reader state")
		}
		for _, count := range calls {
			if count != len(history) {
				t.Fatalf("callback calls=%v, want one per input and layer", calls)
			}
		}
		for _, s := range sessions {
			if !reflect.DeepEqual(s.v41Forward.history, history) {
				t.Fatal("session history differs from requested inputs")
			}
			rows, ok := s.v41Forward.attn.KVSourceRows(0)
			if !ok || len(rows) != len(history)/m.Cfg.DeepSeekV41.CompressRatios[0] {
				t.Fatal("source registry differs from complete requested groups")
			}
		}
	})
}
