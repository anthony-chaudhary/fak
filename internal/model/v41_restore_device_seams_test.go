package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// These are software binding contracts over cpu-ref arithmetic. They do not
// qualify Vulkan prefix reuse or physical DeepSeek V4.1 Flash execution.

// fak-test:runtime fast est=1s lane=default
func TestV41RestoredSessionRebindsDeviceSeams(t *testing.T) {
	priorSDOTForce := q4kSDOTForce
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { q4kSDOTForce = priorSDOTForce })
	m := v41MixedQuantExpertModel(t)
	// A zero limit selects the fused SwiGLU path counted below.
	m.Cfg.SwigluLimit = 0
	be := &v41HalSeamBackend{Backend: compute.Default()}
	source := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
	defer source.Close()
	sourceState := source.v41State()
	sourceState.history = []int{1, 3, 5}
	// Use the existing continuation fixture to carry window/compressor and
	// publication state through the callback-only repair.
	continuation := v41BackendSnapshotSession(t, 2, []int{1, 2}, []int{1, 3, 5})
	defer continuation.Close()
	sourceState.attn = continuation.v41Forward.attn.clone()
	sourceState.layers = captureV41ForwardSnapshot(continuation.v41Forward).layers
	want := captureV41ForwardSnapshot(sourceState)
	restored := captureV41ForwardSnapshot(sourceState).restore()
	if restored.expertGateUp != nil || restored.expertDown != nil {
		t.Fatal("snapshot copied source session callbacks")
	}
	target := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}, v41Forward: restored}
	defer target.Close()
	for i := 0; i < 2; i++ {
		if target.v41State() != restored {
			t.Fatal("binding replaced the restored state pointer")
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(restored), want) {
			t.Fatal("binding changed continuation data")
		}
	}
	if restored.expertGateUp == nil || restored.expertDown == nil {
		t.Fatal("restored device session did not bind both callbacks")
	}
	// Q2_K fixture scales are 1.0; bound the fused intermediate
	// so reduction-order cancellation does not dominate output-relative parity.
	x := make([]float32, m.Cfg.HiddenSize)
	for i := range x {
		x[i] = float32((i%11)-5) / 1024
	}
	fused, outcome, err := restored.expertGateUp(0, "ffn.experts.0", x)
	if err != nil || outcome != v41GateUpHandled {
		t.Fatalf("restored gate/up outcome=%v err=%v", outcome, err)
	}
	nonzero := false
	for i, value := range fused {
		if !(value >= -2 && value <= 2) {
			t.Fatalf("fixture fused[%d]=%g exceeds finite [-2,2] bound", i, value)
		}
		nonzero = nonzero || value != 0
	}
	if !nonzero {
		t.Fatal("fixture produced a zero fused intermediate")
	}
	got, downOutcome, err := restored.expertDown(0, "ffn.experts.0", fused)
	if err != nil || downOutcome != v41DownHandled {
		t.Fatalf("restored down outcome=%v err=%v", downOutcome, err)
	}
	if be.matmuls != 3 || be.swiglu != 1 {
		t.Fatalf("restored expert ops matmul=%d swiglu=%d, want 3/1", be.matmuls, be.swiglu)
	}
	for _, suffix := range []string{"w1", "w3", "w2"} {
		key := "kquant-raw:" + layerName(0, "ffn.experts.0."+suffix+".weight")
		_, targetOwns := target.halW[key]
		_, sourceOwns := source.halW[key]
		if !targetOwns || sourceOwns {
			t.Fatalf("%s was not staged exclusively by the destination", key)
		}
	}
	wantDown := make([]float32, m.Cfg.HiddenSize)
	kQuantMatRowsRange(m.kqw[layerName(0, "ffn.experts.0.w2.weight")], fused, wantDown, 0, m.Cfg.HiddenSize)
	assertV41LogitsClose(t, got, wantDown, "restored down vs host contraction")
	// Compare identical operands on the unsnapshotted device binding as well.
	// The preceding ownership checks must run before the source is used here.
	freshDown, freshOutcome, err := source.v41State().expertDown(0, "ffn.experts.0", fused)
	if err != nil || freshOutcome != v41DownHandled || !reflect.DeepEqual(got, freshDown) {
		t.Fatalf("restored down differs from fresh device binding: outcome=%v err=%v", freshOutcome, err)
	}
	if be.matmuls != 4 || be.swiglu != 1 {
		t.Fatalf("fresh down control ops matmul=%d swiglu=%d, want 4/1 total", be.matmuls, be.swiglu)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41RestoreSeamsPreserveExistingCallbacks(t *testing.T) {
	m := v41MixedQuantExpertModel(t)
	t.Run("fresh-state", func(t *testing.T) {
		s := &Session{M: m, Backend: &v41HalSeamBackend{Backend: compute.Default()}}
		state := s.v41State()
		if state == nil || state.expertGateUp == nil || state.expertDown == nil || s.v41State() != state {
			t.Fatal("fresh state did not bind both callbacks idempotently")
		}
	})
	for _, mask := range []int{1, 2, 3} {
		t.Run(itoa(mask), func(t *testing.T) {
			gateCalls, downCalls := 0, 0
			state := &v41ForwardState{history: []int{1, 3}}
			if mask&1 != 0 {
				state.expertGateUp = func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
					gateCalls++
					return nil, v41GateUpDeclined, nil
				}
			}
			if mask&2 != 0 {
				state.expertDown = func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
					downCalls++
					return nil, v41DownDeclined, nil
				}
			}
			s := &Session{M: m, Backend: &v41HalSeamBackend{Backend: compute.Default()}, v41Forward: state}
			state.callbackOwner = s
			if state.expertGateUp == nil {
				state.expertGateUp = s.v41ExpertGateUpFunc()
			}
			if state.expertDown == nil {
				state.expertDown = s.v41ExpertDownFunc()
			}
			for i := 0; i < 2; i++ {
				if s.v41State() != state || state.expertGateUp == nil || state.expertDown == nil {
					t.Fatal("partial binding replaced state or failed to fill a missing callback")
				}
				if mask&1 != 0 {
					state.expertGateUp(0, "", nil)
				}
				if mask&2 != 0 {
					state.expertDown(0, "", nil)
				}
			}
			if gateCalls != 2*(mask&1) || downCalls != 2*((mask>>1)&1) {
				t.Fatal("repeated binding replaced an existing callback")
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41RestoreSeamsEligibility(t *testing.T) {
	m := v41MixedQuantExpertModel(t)
	for name, s := range map[string]*Session{
		"nil-model":   {Backend: &v41HalSeamBackend{Backend: compute.Default()}},
		"nil-backend": {M: m},
		"non-device":  {M: m, Backend: compute.Pick("cpu-ref")},
	} {
		t.Run(name, func(t *testing.T) {
			s.v41Forward = captureV41ForwardSnapshot(&v41ForwardState{history: []int{1, 3}}).restore()
			want := captureV41ForwardSnapshot(s.v41Forward)
			state := s.v41State()
			if state.expertGateUp != nil || state.expertDown != nil || !reflect.DeepEqual(captureV41ForwardSnapshot(state), want) {
				t.Fatal("ineligible session bound callbacks or changed continuation")
			}
		})
	}
	var nilSession *Session
	if nilSession.v41ExpertGateUpFunc() != nil || nilSession.v41ExpertDownFunc() != nil {
		t.Fatal("nil session callback factory was eligible")
	}
	rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: &noQ3KSeamBackend{expertHALRecordingBackend: rec}, halW: map[string]compute.Tensor{},
		v41Forward: captureV41ForwardSnapshot(&v41ForwardState{history: []int{1}}).restore()}
	defer s.Close()
	state := s.v41State()
	if state.expertGateUp == nil || state.expertDown == nil {
		t.Fatal("DeviceMemory callback eligibility was confused with per-expert dtype admission")
	}
	if _, outcome, err := state.expertDown(0, "ffn.experts.0", make([]float32, m.Cfg.MoEIntermediateSize)); err != nil || outcome != v41DownDeclined {
		t.Fatalf("missing Q3_K outcome=%v err=%v, want decline", outcome, err)
	}
	if len(s.halW) != 0 || len(rec.uploads) != 0 || rec.matmuls != 0 {
		t.Fatal("missing Q3_K down callback staged weights or ran MatMul")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41RestoreEagerDeviceSeams(t *testing.T) {
	source := v41BackendSnapshotSession(t, 2, []int{1, 2}, []int{1, 3, 5})
	defer source.Close()
	v41BackendSeedHALKV(t, source, []int{3, 4})
	// The backend remains the already-qualified cpu-ref identity. Presenting
	// DeviceMemory here exercises callback binding without widening qualification.
	be := &v41HalSeamBackend{Backend: source.Backend}
	source.Backend = be
	for _, hadState := range []bool{true, false} {
		name := "absent-state"
		if hadState {
			name = "present-state"
		}
		t.Run(name, func(t *testing.T) {
			snap, err := source.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer snap.Close()
			want := snap.v41.clone()
			if !hadState {
				snap.v41 = &v41ForwardSnapshot{hadState: false}
			}
			target, err := source.M.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer target.Close()
			if !hadState {
				target.v41Forward = &v41ForwardState{}
			}
			if err := snap.Restore(target); err != nil {
				t.Fatal(err)
			}
			if snap.v41 != nil || snap.Cache != nil || snap.halKV != nil {
				t.Fatal("restore failed to transfer snapshot ownership")
			}
			if !hadState {
				if target.v41Forward != nil {
					t.Fatal("absent-state restore allocated continuation")
				}
				return
			}
			state := target.v41Forward
			if state == nil || state.expertGateUp == nil || state.expertDown == nil {
				t.Fatal("Restore did not eagerly bind device callbacks")
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(state), want) || target.halKV.Len() != source.halKV.Len() || !reflect.DeepEqual(target.halLineage, source.halLineage) {
				t.Fatal("eager binding changed continuation, HAL KV, or lineage")
			}
			assertV41LogitsClose(t, target.Step(6), source.Step(6), "eager restore continuation")
		})
	}
}
