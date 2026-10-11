package model

import "testing"

// The Halo F32 witness (TestV41MHCProjectionVulkan) needs an independent host
// control. Since 6a54495ecd4 a transposed F32 mix is device eligible, so the
// control must decline mHC explicitly; orientation alone no longer keeps it on
// host. This pins both halves on the cpu-ref recorder.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=4s lane=default
func TestV41MHCTransposedF32HostControlRoute(t *testing.T) {
	t.Parallel()
	for _, scenario := range []struct {
		name    string
		decline bool
	}{{"orientation-only", false}, {"explicit-decline", true}} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			control := v41MHCProjVariant(t, "F32", true, false, 2)
			b := newV41MHCProjBackend(4*control.Cfg.HiddenSize, false)
			s := v41EngProjSession(t, control, b)
			if scenario.decline {
				decline := func(int, []float32, int, float32, bool, bool) ([]float32, v41DenseProjectionOutcome, error) {
					return nil, v41ProjectionDeclined, nil
				}
				s.v41State().mhcProjection = decline
				s.v41State().mhcFFNProjection = decline
			}
			for index, ids := range [][]int{{1, 2, 3}, {4}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				before := v41MHCProjPhase(t, control, phase)
				if index == 1 {
					s.Step(ids[0])
				} else {
					s.Prefill(ids)
				}
				delta := v41DenseTestDelta(v41MHCProjPhase(t, control, phase), before)
				rows := float64(2 * 2 * len(ids))
				if scenario.decline {
					if delta["mhc_projection_device_calls"] != 0 || delta["mhc_projection_matmul_calls"] != 0 || delta["mhc_projection_host_calls"] != rows || delta["mhc_projection_host_rows"] != rows {
						t.Errorf("declined control left host route=%v", delta)
					}
				} else if delta["mhc_projection_device_calls"] != rows || delta["mhc_projection_host_calls"] != 0 {
					t.Errorf("transposed F32 control not device eligible=%v", delta)
				}
			}
			wantStages := 4
			if scenario.decline {
				wantStages = 0
			}
			if b.stages != wantStages {
				t.Errorf("immutable mHC staging=%d decline=%t", b.stages, scenario.decline)
			}
		})
	}
}
