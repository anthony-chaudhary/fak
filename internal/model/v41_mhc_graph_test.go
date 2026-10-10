package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// Source-only authored witnesses; runtime estimates are unmeasured. This pins
// the graph equations, not whole-checkpoint or physical-device parity.
// fak-test:runtime medium est=3s lane=default
func TestV41FullMHCGraph(t *testing.T) {
	t.Parallel()
	t.Run("two layers and independent negative controls", func(t *testing.T) {
		m := v41MHCProjVariant(t, "F32", false, false, 2)
		ids := []int{1, 2}
		got := m.Forward(ids)
		want, hidden := v41OracleForwardLatentNormHidden(t, m, ids, v41LatentNormOpts{})
		for pos := range ids {
			v41LogitsClose(t, "two-phase graph", got.Logits[pos], want[pos])
			assertV41RowsClose(t, "final carried hidden", got.Hidden[len(got.Hidden)-1][pos*m.Cfg.HiddenSize:(pos+1)*m.Cfg.HiddenSize], hidden[pos], cpuOracleTol)
		}
		for name, opts := range map[string]v41LatentNormOpts{
			"same-layer attention pre":             {graphCurrentPre: true},
			"reuse attention coefficients for FFN": {graphReuseAttention: true},
			"single combined post":                 {graphCombinedPost: true},
			"stream-zero head":                     {graphStreamZero: true},
		} {
			_, wrong := v41OracleForwardLatentNormHidden(t, m, ids, opts)
			if !v41LogitsDiverge(sysFlatten(hidden), sysFlatten(wrong), cpuOracleTol) {
				t.Fatalf("graph negative control %q is vacuous", name)
			}
		}
		s := m.NewSession()
		defer s.Close()
		v41LogitsClose(t, "prefill", s.Prefill(ids[:1]), lastLogits(m.Forward(ids[:1])))
		v41LogitsClose(t, "cross-layer decode carry", s.Step(ids[1]), got.Logits[1])
	})
	t.Run("owned initialization and collapse", func(t *testing.T) {
		input := []float32{1.01171875, -2.03125}
		original := append([]float32(nil), input...)
		streams, err := v41FullInitialStreams(input)
		if err != nil {
			t.Fatal(err)
		}
		want := v41LatentNormOracleBF16(input)
		for _, row := range streams {
			if !reflect.DeepEqual(row, want) {
				t.Fatal("initial streams do not repeat BF16 embedding")
			}
		}
		streams[0][0] = 8
		if !reflect.DeepEqual(input, original) || !reflect.DeepEqual(streams[1], want) {
			t.Fatal("initial streams alias caller or each other")
		}
		before := sysFlatten(streams)
		pre := []float32{.125, .25, .5, .75}
		out, err := v41MHCPreBF16(0, streams, pre)
		if err != nil || !reflect.DeepEqual(out, v41GraphOracleCollapse(streams, pre)) {
			t.Fatalf("collapse: %v", err)
		}
		out[0] = -99
		if !reflect.DeepEqual(before, sysFlatten(streams)) {
			t.Fatal("collapse overwrote source stream zero")
		}
	})
	t.Run("full admission requires every FFN leaf", func(t *testing.T) {
		for _, leaf := range []string{"mhc.ffn_mixes.weight", "mhc.ffn_base", "mhc.ffn_scale"} {
			m := v41FullStepModel(t)
			delete(m.manifest, layerName(0, leaf))
			if err := m.v41AdmitMHC(0); err == nil {
				t.Fatalf("missing %s admitted", leaf)
			}
		}
	})
	t.Run("carry rejects missing malformed and wrong layer", func(t *testing.T) {
		m := v41FullStepModel(t)
		for name, carry := range map[string]*v41MHCCarry{
			"missing":         nil,
			"wrong row count": newV41MHCCarry(2),
			"wrong width":     {pre: [][]float32{{1, 0, 0}}},
			"nonfinite":       {pre: [][]float32{{1, float32(math.NaN()), 0, 0}}},
			"out of order":    {pre: [][]float32{{1, 0, 0, 0}}, nextLayer: 1},
		} {
			x, streams := v41LayerStepInputs(t, m, 1, true)
			before := sysFlatten(streams)
			state := v41FullStepSeed(t, m, 0, nil)
			err := m.v41LayerStep(0, x, streams, 0, state, &v41ProjScratch{mhcCarry: carry})
			if err == nil || !reflect.DeepEqual(before, sysFlatten(streams)) || state.nextWindowPos != 0 {
				t.Fatalf("%s carry escaped or mutated state: %v", name, err)
			}
		}
	})
	t.Run("late prefill FFN failure publishes no streams or carry", func(t *testing.T) {
		m := v41FullStepModel(t)
		cfg := m.Cfg
		x0, s0 := v41LayerStepInputs(t, m, 1, true)
		x1, s1 := v41LayerStepInputs(t, m, 2, true)
		x, streams := [][]float32{x0, x1}, [][][]float32{s0, s1}
		before := append(sysFlatten(s0), sysFlatten(s1)...)
		carry := newV41MHCCarry(2)
		stop := errors.New("selected last-token FFN mix failure")
		calls := 0
		st := &v41ForwardState{mhcFFNProjection: func(int, []float32, int, float32, bool, bool) ([]float32, v41DenseProjectionOutcome, error) {
			calls++
			if calls == 2 {
				return nil, v41ProjectionError, stop
			}
			return make([]float32, v41MHCMixWidth), v41ProjectionHandled, nil
		}}
		route, err := v41RouterConfigFullGeometry(cfg)
		if err != nil {
			t.Fatal(err)
		}
		err = m.v41Layer(0, []int{1, 2}, x, streams, true, cfg.HeadDim, cfg.NumHeads, cfg.HiddenSize, float32(cfg.RMSNormEps), hcItersOrDefault(cfg), hcEpsOrDefault(cfg), route, st, &v41ProjScratch{mhcCarry: carry}, nil)
		var selected *V41ProjectionOperationError
		if !errors.Is(err, stop) || !errors.As(err, &selected) || selected.Leaf != "mhc.ffn_mixes.weight" || calls != 2 {
			t.Fatalf("selected FFN failure replayed or lost identity: calls=%d err=%v", calls, err)
		}
		after := append(sysFlatten(s0), sysFlatten(s1)...)
		if !reflect.DeepEqual(before, after) || !reflect.DeepEqual(carry, newV41MHCCarry(2)) {
			t.Fatal("failed token panel partially published streams/carry")
		}
	})
}
