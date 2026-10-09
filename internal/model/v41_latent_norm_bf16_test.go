package model

import (
	"errors"
	"math"
	"testing"
)

// Keep projection casts in the independent full oracle even when its negative
// control omits normalization. This never calls the production BF16 helper.
func v41LatentNormOracleBF16(values []float32) []float32 {
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = float32(v41RefBF16(float64(value)))
	}
	return out
}

// Owner boundaries only: callback arithmetic/ABI and GPU qualification are not
// changed. Runtime estimate is unmeasured.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41LatentNormBF16Boundaries(t *testing.T) {
	t.Parallel()
	fill := func(n int, pattern ...float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = pattern[i%len(pattern)]
		}
		return out
	}
	check := func(t *testing.T, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("width %d, want %d", len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("element %d = %g (%08x), want %g (%08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
	for _, tc := range []struct {
		name, leaf string
		width      int
		call       func(*Model, []float32, func(int, []float32) ([]float32, error)) error
	}{
		{"query", "attn.wq_a_norm.weight", 8, func(m *Model, x []float32, f func(int, []float32) ([]float32, error)) error {
			return m.v41QueryNormInPlace(0, x, 8, f)
		}},
		{"KV", "attn.kv_norm.weight", v41KVLoraRank, func(m *Model, x []float32, f func(int, []float32) ([]float32, error)) error {
			return m.v41KVNormInPlace(0, x, 8, f)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			manifest, raw := synthBuildRaw([]synthTensor{{layerName(0, tc.leaf), []int{tc.width}}},
				func(string, func() float32) float32 { return 1 })
			m := &Model{Cfg: Config{QLoraRank: 8}, manifest: manifest, raw: raw}
			v41WriteTensorF32(t, m, layerName(0, tc.leaf), fill(tc.width, 1, 2, -1, -2))
			original := fill(tc.width, 1.01171875)
			input := append([]float32(nil), original...)
			if err := tc.call(m, input, nil); err != nil {
				t.Fatal(err)
			}
			// Odd midpoint projection rounds to 1.015625, then the signed gains
			// multiply the F32 normalized value before its BF16 copyback.
			check(t, input, fill(tc.width, 0.337890625, 0.67578125, -0.337890625, -0.67578125))

			negativeZero := math.Float32frombits(0x80000000)
			shared := fill(tc.width, 1.00390625, 1.01171875, -1.00390625, -1.01171875, 0, negativeZero)
			sharedBefore := append([]float32(nil), shared...)
			input = append([]float32(nil), original...)
			calls := 0
			if err := tc.call(m, input, func(_ int, staged []float32) ([]float32, error) {
				calls++
				check(t, staged, fill(tc.width, 1.015625))
				check(t, input, original)
				return shared, nil
			}); err != nil || calls != 1 {
				t.Fatalf("callback calls=%d err=%v", calls, err)
			}
			check(t, input, fill(tc.width, 1, 1.015625, -1, -1.015625, 0, negativeZero))
			check(t, shared, sharedBefore)

			input = append([]float32(nil), original...)
			var borrowed []float32
			if err := tc.call(m, input, func(_ int, staged []float32) ([]float32, error) {
				staged[0] = 1.00390625
				borrowed = staged
				return staged, nil
			}); err != nil {
				t.Fatal(err)
			}
			if input[0] != 1 || borrowed[0] != 1.00390625 {
				t.Fatal("aliased callback result was rounded in place")
			}

			stop := errors.New("selected norm failure")
			for _, fault := range []string{"error", "short", "NaN", "Inf", "BF16 overflow"} {
				input = append([]float32(nil), original...)
				calls = 0
				err := tc.call(m, input, func(_ int, staged []float32) ([]float32, error) {
					calls++
					staged[0] = 9 // caller must remain untouched even on a selected failure
					values := fill(tc.width, 1)
					switch fault {
					case "error":
						return nil, stop
					case "short":
						return values[:len(values)-1], nil
					case "NaN":
						values[0] = float32(math.NaN())
					case "Inf":
						values[0] = float32(math.Inf(1))
					case "BF16 overflow":
						values[0] = math.MaxFloat32
					}
					return values, nil
				})
				var attributed *V41ProjectionOperationError
				if !errors.As(err, &attributed) || attributed.Leaf != tc.leaf || attributed.Stage != string(v41StageAttention) || calls != 1 ||
					(fault == "error" && !errors.Is(err, stop)) {
					t.Fatalf("%s: calls=%d lost failure identity: %v", fault, calls, err)
				}
				check(t, input, original)
			}
			for _, bad := range [][]float32{fill(tc.width-1, 1), fill(tc.width, math.MaxFloat32), fill(tc.width, float32(math.NaN()))} {
				before := append([]float32(nil), bad...)
				calls = 0
				err := tc.call(m, bad, func(int, []float32) ([]float32, error) {
					calls++
					return nil, stop
				})
				if err == nil || calls != 0 {
					t.Fatalf("bad projection reached normalization: calls=%d err=%v", calls, err)
				}
				check(t, bad, before)
			}
		})
	}
	for _, route := range []string{"prefill", "plain step", "role step"} {
		t.Run(route, func(t *testing.T) {
			m := v41FullStepModel(t)
			cfg := m.Cfg
			x := fill(cfg.HiddenSize, 1)
			streams := [][]float32{x, fill(cfg.HiddenSize, 0), fill(cfg.HiddenSize, 0), fill(cfg.HiddenSize, 0)}
			qCalls, kvCalls, rotaryCalls := 0, 0, 0
			normalize := func(width int, calls *int) func(int, []float32) ([]float32, error) {
				return func(_ int, staged []float32) ([]float32, error) {
					*calls = *calls + 1
					check(t, staged, fill(width, 1.015625))
					return fill(width, 1.00390625, 1.01171875), nil
				}
			}
			project := func(_ int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
				if leaf == "attn.wq_b.weight" {
					check(t, panel, fill(in*rows, 1, 1.015625))
					return fill(out*rows, 0), v41ProjectionHandled, nil
				}
				return fill(out*rows, 1.01171875), v41ProjectionHandled, nil
			}
			stop := errors.New("stop before rotary")
			rotate := func(_ int, _, kv, _, _ []float32, _, _, _ int) ([]float32, []float32, error) {
				rotaryCalls++
				check(t, kv, fill(cfg.HeadDim, 1, 1.015625))
				return nil, nil, stop
			}
			qNorm, kvNorm := normalize(cfg.QLoraRank, &qCalls), normalize(v41KVLoraRank, &kvCalls)
			scratch := &v41ProjScratch{denseProjection: project, queryNorm: qNorm, kvNorm: kvNorm, tailRoPE: rotate}
			state, err := NewV41AttentionState(cfg.HeadDim, 8)
			if err != nil {
				t.Fatal(err)
			}
			switch route {
			case "prefill":
				st := &v41ForwardState{denseProjection: project, queryNorm: qNorm, kvNorm: kvNorm, tailRoPE: rotate}
				err = m.v41Layer(0, []int{0, 0}, [][]float32{x, x}, [][][]float32{streams, streams}, true,
					cfg.HeadDim, cfg.NumHeads, cfg.HiddenSize, float32(cfg.RMSNormEps), hcItersOrDefault(cfg), hcEpsOrDefault(cfg), v41RouterConfig{}, st, scratch, nil)
			case "plain step":
				err = m.v41LayerStep(0, x, streams, 0, state, scratch)
			case "role step":
				err = m.v41LayerStepRole(0, V41AttentionPlan{Role: V41AttentionRoleKVSource, Ratio: 4}, x, streams, 0, state, state, scratch)
			}
			wantQCalls := 1
			if route == "prefill" {
				wantQCalls = 2
			}
			if !errors.Is(err, stop) || qCalls != wantQCalls || kvCalls != 1 || rotaryCalls != 1 || state.nextWindowPos != 0 {
				t.Fatalf("live route calls q=%d kv=%d rotary=%d err=%v", qCalls, kvCalls, rotaryCalls, err)
			}
		})
	}
}
