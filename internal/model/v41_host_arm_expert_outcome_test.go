package model

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41HostArmExpertOutcome(t *testing.T) {
	const hostHandled v41RoutedExpertOutcome = 3
	for _, outcome := range []v41RoutedExpertOutcome{hostHandled, v41RoutedExpertHandled} {
		m := &Model{}
		m.v41SetExpertFaultPhase(V41PhasePrefill)
		t.Cleanup(func() { m.CloseWeights() })
		cfg := Config{HiddenSize: 2, MoEIntermediateSize: 2}
		calls := 0
		routed := func(_ int, _ string, _ []float32, _ float32) ([]float32, v41RoutedExpertOutcome, error) {
			calls++
			return []float32{1.003, -2.007}, outcome, nil
		}
		out, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", []float32{.25, .5}, 1, cfg, routed, nil, nil, nil, nil, false)
		if err != nil || calls != 1 {
			t.Fatalf("closed handled outcome=%d calls=%d err=%v", outcome, calls, err)
		}
		v41TailRoPECheckBits(t, "host outcome publication", out, []float32{v41RoundBF16(1.003), v41RoundBF16(-2.007)})
		wantDevice := 0
		if outcome == v41RoutedExpertHandled {
			wantDevice = 1
		}
		if got := m.V41ExpertFaultAttribution().Prefill.Coverage.DeviceBypassContractions; got != wantDevice {
			t.Fatalf("outcome=%d device contractions=%d want=%d", outcome, got, wantDevice)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41HostArmExpertOutcomePreservesErrors(t *testing.T) {
	const hostHandled v41RoutedExpertOutcome = 3
	failure := errors.New("selected host outcome witness")
	for _, tc := range []struct {
		outcome v41RoutedExpertOutcome
		values  []float32
		cause   error
		want    error
	}{{hostHandled, nil, failure, failure}, {hostHandled, []float32{1}, nil, errV41ExpertResult}, {v41RoutedExpertOutcome(255), []float32{1, 2}, nil, errV41ExpertResult}} {
		m := &Model{}
		routed := func(_ int, _ string, _ []float32, _ float32) ([]float32, v41RoutedExpertOutcome, error) {
			return tc.values, tc.outcome, tc.cause
		}
		_, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", []float32{1, 2}, 1, Config{HiddenSize: 2, MoEIntermediateSize: 2}, routed, nil, nil, nil, nil, false)
		if !errors.Is(err, tc.want) {
			t.Fatal("handled outcome changed closed failure", err)
		}
	}
}
