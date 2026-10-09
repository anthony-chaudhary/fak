package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// Independent transcription of hc_pre's dtype copyback followed by RMSNorm.
// No production normalization, BF16 helper, or projection is used here.
func v41AttentionInputNormOracle(collapsed, gain []float32, eps float32) []float32 {
	x := make([]float32, len(collapsed))
	var squares float32
	for i, value := range collapsed {
		x[i] = float32(v41RefBF16(float64(value)))
		squares = float32(squares + float32(x[i]*x[i]))
	}
	variance := float32(squares / float32(len(x)))
	inv := float32(1 / math.Sqrt(float64(float32(variance+eps))))
	for i := range x {
		normalized := float32(x[i] * inv)
		x[i] = float32(v41RefBF16(float64(float32(gain[i] * normalized))))
	}
	return x
}

// Pins the missing post-collapse operation, not full checkpoint/GPU parity.
// Runtime estimate is unmeasured.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41AttentionInputNormAfterCollapse(t *testing.T) {
	t.Parallel()
	t.Run("dtype gain and ownership", func(t *testing.T) {
		manifest, raw := synthBuildRaw([]synthTensor{{layerName(0, "attn_norm.weight"), []int{4}}},
			func(string, func() float32) float32 { return 1 })
		m := &Model{Cfg: Config{HiddenSize: 4}, manifest: manifest, raw: raw}
		gain := []float32{1, 2, -1, -2}
		v41WriteTensorF32(t, m, layerName(0, "attn_norm.weight"), gain)
		// This exact odd tie rounds to 1.015625 before normalization. Skipping
		// that cast changes the final BF16 result (0.3359375 vs 0.337890625).
		input := []float32{1.01171875, 1.01171875, 1.01171875, 1.01171875}
		original := append([]float32(nil), input...)
		got, err := m.v41AttentionInputNorm(0, input, 8)
		want := []float32{0.337890625, 0.67578125, -0.337890625, -0.67578125}
		if err != nil || !reflect.DeepEqual(got, want) || !reflect.DeepEqual(input, original) {
			t.Fatalf("BF16 -> learned RMSNorm -> BF16: got=%v want=%v input=%v err=%v", got, want, input, err)
		}
		if !reflect.DeepEqual(got, v41AttentionInputNormOracle(input, gain, 8)) {
			t.Fatal("fixed expected values and independent equation disagree")
		}
		got[0] = 7
		if !reflect.DeepEqual(input, original) {
			t.Fatal("normalization output aliases the raw collapse")
		}
		for _, tc := range []struct {
			name              string
			inputWidth, gains int
			eps               float32
		}{
			{"input width", 3, 4, 8}, {"gain width", 4, 3, 8},
			{"zero epsilon", 4, 4, 0}, {"negative epsilon", 4, 4, -1},
			{"NaN epsilon", 4, 4, float32(math.NaN())}, {"infinite epsilon", 4, 4, float32(math.Inf(1))},
		} {
			manifest, raw := synthBuildRaw([]synthTensor{{layerName(0, "attn_norm.weight"), []int{tc.gains}}},
				func(string, func() float32) float32 { return 1 })
			badModel := &Model{Cfg: Config{HiddenSize: 4}, manifest: manifest, raw: raw}
			if got, err := badModel.v41AttentionInputNorm(0, make([]float32, tc.inputWidth), tc.eps); err == nil || got != nil {
				t.Fatalf("%s escaped: got=%v err=%v", tc.name, got, err)
			}
		}
		for _, value := range []float32{math.MaxFloat32, float32(math.NaN())} {
			bad := append([]float32(nil), input...)
			bad[0] = value
			before := math.Float32bits(bad[0])
			_, err := m.v41AttentionInputNorm(0, bad, 8)
			var attributed *V41ProjectionOperationError
			if !errors.As(err, &attributed) || attributed.Leaf != "attn_norm.weight" ||
				attributed.Stage != string(v41StageAttention) || math.Float32bits(bad[0]) != before {
				t.Fatalf("invalid BF16 input lost attribution or mutated caller: %v", err)
			}
		}
		for _, value := range []float32{math.MaxFloat32, float32(math.NaN())} {
			v41WriteTensorF32(t, m, layerName(0, "attn_norm.weight"), []float32{value, 1, 1, 1})
			// With unit input and epsilon below F32 precision, MaxFloat32 is
			// finite after learned scaling but overflows the BF16 copyback.
			if got, err := m.v41AttentionInputNorm(0, []float32{1, 1, 1, 1}, 1e-20); err == nil || got != nil {
				t.Fatalf("nonfinite gain/BF16 output escaped: got=%v err=%v", got, err)
			}
		}
	})
	for _, route := range []string{"prefill", "plain step", "role step"} {
		t.Run(route, func(t *testing.T) {
			m := v41FullStepModel(t)
			cfg := m.Cfg
			H, eps := cfg.HiddenSize, float32(cfg.RMSNormEps)
			gain := make([]float32, H)
			streams := make([][]float32, 4)
			for h := range streams {
				streams[h] = make([]float32, H)
				for i := range streams[h] {
					streams[h][i] = float32((i+3*h)%11-5) + float32(h)/8
				}
			}
			for i := range gain {
				gain[i] = float32(i%7+1) / 4
				if i%3 == 0 {
					gain[i] = -gain[i]
				}
			}
			v41WriteTensorF32(t, m, layerName(0, "attn_norm.weight"), gain)
			// A zero mix projection and zero base give sigmoid(0)+hc_eps.
			// Keep this collapse calculation separate from v41MHCPre.
			pre := float32(0.5) + hcEpsOrDefault(cfg)
			collapsed := make([]float32, H)
			for _, stream := range streams {
				for i, value := range stream {
					collapsed[i] = float32(collapsed[i] + float32(pre*value))
				}
			}
			want := v41AttentionInputNormOracle(collapsed, gain, eps)
			if reflect.DeepEqual(want, collapsed) {
				t.Fatal("fixture cannot distinguish the missing norm")
			}
			before := sysFlatten(streams)
			stop := errors.New("stop after normalized attention inputs")
			calls := map[string]int{}
			project := func(_ int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
				calls[leaf]++
				if leaf == "attn.wq_a.weight" || leaf == "attn.wkv.weight" {
					for r := 0; r < rows; r++ {
						if !reflect.DeepEqual(panel[r*in:(r+1)*in], want) {
							t.Fatalf("%s row %d did not consume normalized BF16 collapse", leaf, r)
						}
					}
				}
				if leaf == "attn.wkv.weight" {
					return nil, v41ProjectionError, stop
				}
				return make([]float32, out*rows), v41ProjectionHandled, nil
			}
			mix := func(int, []float32, int, float32, bool, bool) ([]float32, v41DenseProjectionOutcome, error) {
				return make([]float32, v41MHCMixWidth), v41ProjectionHandled, nil
			}
			scratch := &v41ProjScratch{denseProjection: project, mhcProjection: mix}
			state, err := NewV41AttentionState(cfg.HeadDim, 8)
			if err != nil {
				t.Fatal(err)
			}
			x := streams[0]
			switch route {
			case "prefill":
				st := &v41ForwardState{denseProjection: project, mhcProjection: mix}
				err = m.v41Layer(0, []int{0, 0}, [][]float32{x, x}, [][][]float32{streams, streams}, true,
					cfg.HeadDim, cfg.NumHeads, H, eps, hcItersOrDefault(cfg), hcEpsOrDefault(cfg), v41RouterConfig{}, st, scratch, nil)
			case "plain step":
				err = m.v41LayerStep(0, x, streams, 0, state, scratch)
			case "role step":
				plan := V41AttentionPlan{Role: V41AttentionRoleKVSource, Ratio: 4}
				err = m.v41LayerStepRole(0, plan, x, streams, 0, state, state, scratch)
			}
			if !errors.Is(err, stop) || calls["attn.wq_a.weight"] != 1 || calls["attn.wkv.weight"] != 1 {
				t.Fatalf("actual attention route did not reach both operands: calls=%v err=%v", calls, err)
			}
			if !reflect.DeepEqual(before, sysFlatten(streams)) || state.nextWindowPos != 0 {
				t.Fatal("attention input preparation mutated residual streams or retained state")
			}
		})
	}
}
