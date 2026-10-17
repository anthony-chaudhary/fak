package model

import "testing"

// Clock advances model callback work, including down and final readback. This
// witnesses the real cold MoE dispatch seam, not GPU execution or throughput.
// fak-test:runtime fast est=10ms lane=default
func TestV41ColdScalarDispatchEnvelope(t *testing.T) {
	for _, rows := range []int{1, 2} {
		for _, split := range []bool{false, true} {
			for _, tick := range []int64{0, 1} {
				cfg := Config{HiddenSize: 1, MoEIntermediateSize: 1, HeadDim: v41KVLoraRank}
				m := &Model{Cfg: cfg}
				now := int64(100)
				m.v41SetExpertTimeClock(func() int64 { return now })
				m.v41SetExpertFaultPhase(V41PhasePrefill)
				st := &v41ForwardState{}
				st.routedExpert = func(int, string, []float32, float32) ([]float32, v41RoutedExpertOutcome, error) {
					if split {
						now += 1000 * tick // A declined admission is not a dispatched contraction.
						return nil, v41RoutedExpertDeclined, nil
					}
					now += 7 * tick  // gate/up and activation
					now += 11 * tick // down
					now += 13 * tick // final readback
					return []float32{1}, v41RoutedExpertHandled, nil
				}
				st.expertGateUp = func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
					now += 7 * tick
					return []float32{1}, v41GateUpHandled, nil
				}
				st.expertDown = func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
					now += (11 + 13) * tick
					return []float32{1}, v41DownHandled, nil
				}
				inputs, picks := make([][]float32, rows), make([][]routePick, rows)
				for r := range inputs {
					inputs[r], picks[r] = []float32{1}, []routePick{{expert: 0, weight: 1}}
				}
				in := v41LayerMoEInput{full: true, H: 1, x: inputs, st: st, scratch: &v41ProjScratch{}}
				// Reuse the original state across layers: wrappers must not stack.
				for layer := 0; layer < 2; layer++ {
					got, err := m.v41LayerRoutedContraction(in, inputs, picks)
					if err != nil || len(got) != rows || got[0][0] != 1 {
						t.Fatalf("rows=%d split=%v output=%v err=%v", rows, split, got, err)
					}
				}
				got := m.V41ExpertFaultAttribution().Prefill
				count := 2 * rows
				samples := count
				if split {
					samples *= 2
				}
				if got.IncrementalDeviceGateUpCalls != count || got.IncrementalDeviceDownCalls != count || got.IncrementalDeviceDispatchNanos != int64(count)*31*tick || got.Coverage.DeviceDispatchTime.MeasuredSamples != samples || got.Coverage.DeviceDispatchTime.Status != "measured" {
					t.Fatalf("rows=%d split=%v tick=%d dispatch=%+v", rows, split, tick, got)
				}
				if got.ContractionNanos != 0 || got.Coverage.ContractionTime.MeasuredSamples != 0 || got.Coverage.ContractionTime.Status != "unavailable" || got.Faults != 0 {
					t.Fatalf("dispatch fabricated host contraction or faults: %+v", got)
				}
				m.v41SetExpertFaultPhase(V41PhaseUnknown)
			}
		}
	}
}
