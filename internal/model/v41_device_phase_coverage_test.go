package model

import (
	"errors"
	"reflect"
	"testing"
)

// This is a production routed-function wiring witness with a deterministic
// clock, not device performance qualification.
// fak-test:runtime fast est=10ms lane=default
func TestV41DevicePhaseCoverage(t *testing.T) {
	for _, phase := range []V41ExpertPhase{V41PhasePrefill, V41PhaseDecode} {
		for _, path := range []string{"device", "declined_host", "host", "error"} {
			t.Run(phase.String()+"/"+path, func(t *testing.T) {
				m := &Model{}
				m.v41SetExpertTimeClock(func() int64 { return 100 })
				m.v41SetExpertFaultPhase(phase)
				if path == "error" {
					m.v41NoteExpertFaultToken(1)
				}
				cfg := Config{HiddenSize: 1, MoEIntermediateSize: 1}
				failure := errors.New("selected device failure")
				var routed v41RoutedExpertFunc
				if path != "host" {
					routed = func(int, string, []float32, float32) ([]float32, v41RoutedExpertOutcome, error) {
						switch path {
						case "device":
							return []float32{1}, v41RoutedExpertHandled, nil
						case "error":
							return nil, v41RoutedExpertError, failure
						default:
							return nil, v41RoutedExpertDeclined, nil
						}
					}
				}
				triple := func() ([]float32, []float32, []float32, error) {
					m.v41NoteExpertResidentHit()
					return []float32{1}, []float32{1}, []float32{1}, nil
				}
				_, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", []float32{1}, 1, cfg, routed, nil, nil, triple, nil, false)
				at := m.V41ExpertFaultAttribution()
				got := at.Prefill
				if phase == V41PhaseDecode {
					got = at.Decode
				}
				if path == "error" {
					if !errors.Is(err, failure) || got.Contractions != 0 || got.ContractionNanos != 0 || got.Coverage.ContractionTime.MeasuredSamples != 0 || got.Coverage.ContractionTime.Status != "unavailable" {
						t.Fatalf("error fabricated successful timing: %+v, %v", got, err)
					}
					return
				}
				if err != nil || got.Contractions != 1 || got.ContractionNanos != 0 {
					t.Fatalf("routed result: %+v, %v", got, err)
				}
				if got.Coverage.RequestIsolation != "unsupported" || got.Coverage.Dependency == "" {
					t.Fatal("missing shared phase limitation")
				}
				if path == "device" {
					if got.Coverage.ContractionTime.Status != "unavailable" || got.Coverage.ContractionTime.UnavailableSamples != 1 || got.Coverage.HostMaterialization.Status != "not_applicable" {
						t.Fatalf("device unknown elapsed or host bypass mislabeled: %+v", got.Coverage)
					}
				} else if got.Coverage.ContractionTime.Status != "measured" || got.Coverage.ContractionTime.MeasuredSamples != 1 || got.Coverage.HostMaterialization.Status != "measured" {
					t.Fatalf("host timed zero was lost: %+v", got.Coverage)
				}
				if got.Faults != 0 || got.FaultedBytes != 0 || got.DequantBytes != 0 {
					t.Fatal("device upload or resident hit fabricated host faults/dequant")
				}
			})
		}
	}
	m := &Model{}
	m.v41SetExpertFaultPhase(V41PhasePrefill)
	m.v41SetExpertTimeClock(func() int64 { return 100 })
	m.v41NoteExpertFaultDoorNanos(0)
	m.v41NoteExpertDequantNanos(0)
	m.v41NoteIncrementalDeviceDispatch(true, 100)
	m.v41NoteExpertActivation(1, 0, 16, 100)
	got := m.V41ExpertFaultAttribution()
	for _, field := range []V41PhaseFieldCoverage{got.Prefill.Coverage.FaultTime, got.Prefill.Coverage.DequantTime, got.Prefill.Coverage.DeviceDispatchTime, got.Prefill.Coverage.DeviceActivationTime} {
		if field.Status != "measured" || field.MeasuredSamples != 1 {
			t.Fatalf("measured zero clock pair lost: %+v", field)
		}
	}
	if got.Prefill.Coverage.DeviceDispatchTime.Boundary != "host_observed_attempt_submission_through_return_or_failure_completion_not_guaranteed" {
		t.Fatal("dispatch timing incorrectly guarantees device completion")
	}
	if got.Decode != (V41ExpertFaultAttributionPhase{}) {
		t.Fatal("raw untouched phase changed compatibility zero")
	}
	l := v41ExpertFaultLedgerOf(m, false)
	l.mu.Lock()
	busy := m.V41ExpertFaultAttribution()
	l.mu.Unlock()
	if busy.Prefill.Coverage.RequestIsolationReason != "ledger_busy" || busy.Prefill.Coverage.ContractionTime.Status != "unavailable" {
		t.Fatal("busy read fabricated measured zero")
	}
}

// Uses the existing CPU backend fixture through actual Prefill and Step entry
// points. It proves instrumentation boundaries only, without hardware claims.
// fak-test:runtime medium est=10s lane=default
func TestV41DevicePhaseCoverageEntryPoints(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "1")
	m := v41PinnedSlateFixture(t, false)
	b := &v41EntryPointCohortWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, now: 100}
	b.ordinary = newV41DeferredPublishRecorder()
	b.ordinary.self = b
	b.advance = func(n int64) { b.now += n }
	m.v41SetExpertTimeClock(func() int64 { return b.now })
	defer m.v41SetExpertTimeClock(nil)
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if len(s.Prefill([]int{1, 2, 3})) == 0 {
		t.Fatal("prefill returned no logits")
	}
	prefill := m.V41ExpertFaultAttribution()
	if prefill.Prefill.Coverage.DeviceDispatchTime.Status != "measured" || prefill.Prefill.IncrementalDeviceDispatchNanos <= 0 {
		t.Fatal("prefill completion envelope not covered")
	}
	if len(s.Step(4)) == 0 {
		t.Fatal("decode returned no logits")
	}
	after := m.V41ExpertFaultAttribution()
	if !reflect.DeepEqual(after.Prefill, prefill.Prefill) || after.Decode.Tokens != 1 || after.Decode.Coverage.DeviceDispatchTime.Status != "measured" {
		t.Fatal("decode changed prefill or lost independent envelope")
	}
	if after.Decode.Coverage.DeviceDispatchTime.Overlap == "" || after.Decode.Coverage.DeviceActivationTime.Overlap == "" {
		t.Fatal("overlapping envelopes lack labels")
	}
}
