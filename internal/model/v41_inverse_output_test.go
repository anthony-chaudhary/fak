package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The fixture disables YaRN so the oracle needs only the reference's direct
// complex conjugation. It does not call a production table or rotation helper.
func v41InverseOutputOracle(input []float32, heads, headDim, rotaryDim, pos int, theta float64) []float32 {
	out := append([]float32(nil), input...)
	for h := 0; h < heads; h++ {
		for pair := 0; pair < rotaryDim/2; pair++ {
			i := h*headDim + headDim - rotaryDim + 2*pair
			angle := float64(pos) / math.Pow(theta, float64(2*pair)/float64(rotaryDim))
			c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
			a, b := input[i], input[i+1]
			out[i] = float32(a*c) + float32(b*s)
			out[i+1] = float32(b*c) - float32(a*s)
		}
	}
	return out
}

// This is a software wiring witness, not a hardware or full-model parity claim.
// It exercises all four post-contraction callers, including suffix and restored
// continuation, and observes the exact operand handed to grouped projection.
// The runtime estimate is provisional until executable qualification.
// fak-test:runtime medium est=10s lane=default
func TestV41InverseAttentionOutput(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("inverse-output software witness requires cpu-ref")
	}
	for _, role := range []bool{false, true} {
		name := "plain"
		if role {
			name = "compressed-source-reader"
		}
		t.Run(name, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			if role {
				m = v41ReaderWidthFixture(t)
			}
			m.Cfg.RopeTheta = 256
			m.Cfg.RopeThetaPerLayer = nil
			m.Cfg.RopeScaling, m.Cfg.LongRope = "", nil
			m.Cfg.RopeFactor, m.Cfg.RopeOrigContext = 0, 0
			m.Cfg.DeepSeekV41.CompressRopeTheta = 4096
			b := newV41SharedAttentionTestBackend()
			s := v41DenseTestSession(t, m, b)
			requests, projections := 0, 0
			distinct := false
			var expected, unrotated []float32
			bind := func(session *Session) {
				v41SharedAttentionOnly(session)
				st := session.v41State()
				selected := st.sharedAttention
				if selected == nil {
					t.Fatal("shared-attention operation was not selected")
				}
				st.sharedAttention = func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
					wantLayer, pos := requests/2, requests%2
					if requests >= 4 {
						wantLayer, pos = requests%2, requests/2
					}
					if layer != wantLayer || projections != requests {
						t.Fatalf("contraction order: layer=%d want=%d requests=%d projections=%d", layer, wantLayer, requests, projections)
					}
					if request.plain.Inverse != nil || request.compressed.Inverse != nil {
						t.Fatal("inverse was sent through the unsupported device options")
					}
					out, err := selected(layer, request)
					if err != nil {
						return nil, err
					}
					unrotated = append([]float32(nil), out...)
					theta := float64(256)
					if role {
						theta = 4096
					}
					expected = v41InverseOutputOracle(out, m.Cfg.NumHeads, m.Cfg.HeadDim, m.Cfg.QKRopeHeadDim, pos, theta)
					requests++
					return out, nil
				}
				st.groupedOutput = func(layer int, out []float32, heads, headDim, groups, rank, dim int) ([]float32, v41DenseProjectionOutcome, error) {
					if requests != projections+1 || len(out) != len(expected) {
						t.Fatal("grouped projection did not immediately consume the contraction output")
					}
					for i, want := range expected {
						if i%headDim < headDim-m.Cfg.QKRopeHeadDim {
							if math.Float32bits(out[i]) != math.Float32bits(unrotated[i]) {
								t.Fatalf("output prefix changed at %d", i)
							}
						}
						if delta := math.Abs(float64(out[i]) - float64(want)); math.IsNaN(delta) || delta > 1e-6+1e-6*math.Abs(float64(want)) {
							t.Fatalf("projection %d layer %d input[%d]=%g want inverse=%g", projections, layer, i, out[i], want)
						}
						if math.Abs(float64(want)-float64(unrotated[i])) > 1e-5 {
							distinct = true
						}
					}
					projections++
					return make([]float32, dim), v41ProjectionHandled, nil
				}
			}
			bind(s)
			s.Prefill([]int{1, 2})
			if !s.v41IncrementalEligible() || requests != 4 {
				t.Fatal("cold prefill did not seed both layer paths")
			}
			s.Step(3)
			s.Prefill([]int{4})
			if requests != 8 {
				t.Fatal("Step or suffix replayed the prefix")
			}
			snapshot, err := s.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer snapshot.Close()
			target := v41DenseTestSession(t, m, b)
			if err := snapshot.Restore(target); err != nil {
				t.Fatal(err)
			}
			bind(target)
			s.Close()
			target.Prefill([]int{5})
			if requests != 10 || projections != requests || !distinct || target.v41Forward.callbackOwner != target {
				t.Fatalf("continuation witness requests=%d projections=%d changed=%v", requests, projections, distinct)
			}
		})
	}
}
