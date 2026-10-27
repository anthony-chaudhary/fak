package model

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmLatentPublicationHost(t *testing.T) {
	for _, tc := range []struct {
		name     string
		values   []float32
		rounds   int64
		observed int
		fail     bool
	}{
		{"empty", nil, 0, 0, false},
		{"widened", []float32{1, -2}, 0, 0, false},
		{"nonwidened", []float32{1.003, -2.011}, 2, 1, false},
		{"mixed", []float32{1, -2.011}, 2, 1, false},
		{"invalid-first", []float32{float32(math.NaN()), 1.003}, 0, 0, true},
		{"invalid-later", []float32{1.003, float32(math.Inf(1))}, 1, 1, true},
		{"overflow-first", []float32{math.MaxFloat32, 1.003}, 1, 1, true},
		{"overflow-later", []float32{1.003, math.MaxFloat32}, 2, 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			original := append([]float32(nil), tc.values...)
			reference, referenceErr := v41LatentNormBF16Publish(0, "attn.wq_a_norm.weight", "projection", tc.values, nil)
			stop := enableV41BF16Witness()
			defer stop()
			var observed int
			var atObservation int64
			result, err := v41LatentNormBF16PublishObserved(0, "attn.wq_a_norm.weight", "projection", tc.values, nil, func(stage string) {
				if stage != "bf16-boundary" {
					t.Fatalf("stage=%s", stage)
				}
				observed++
				atObservation = v41HostBF16Rounds.Load()
			})
			if (err != nil) != tc.fail || (referenceErr != nil) != tc.fail || (tc.fail && !errors.Is(err, ErrV41ForwardStage)) {
				t.Fatalf("error=%v reference=%v", err, referenceErr)
			}
			if observed != tc.observed || v41HostBF16Rounds.Load() != tc.rounds || (observed > 0 && atObservation != tc.rounds) {
				t.Fatalf("observed=%d at=%d rounds=%d want=%d/%d", observed, atObservation, v41HostBF16Rounds.Load(), tc.observed, tc.rounds)
			}
			v41TailRoPECheckBits(t, "input unchanged", tc.values, original)
			v41TailRoPECheckBits(t, "legacy parity", result, reference)
			stop()
			withoutObserver, nilErr := v41LatentNormBF16PublishObserved(0, "attn.wq_a_norm.weight", "projection", tc.values, nil, nil)
			if (nilErr != nil) != tc.fail {
				t.Fatalf("nil observer error=%v", nilErr)
			}
			v41TailRoPECheckBits(t, "nil observer parity", withoutObserver, reference)
			if len(result) > 0 {
				result[0] = 9
				v41TailRoPECheckBits(t, "owned result", tc.values, original)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmLatentPublicationSelected(t *testing.T) {
	for _, mode := range []string{"synchronous", "deferred", "publication-error", "deferred-error", "widened", "empty"} {
		t.Run(mode, func(t *testing.T) {
			b := newV41BF16PublishRecorder()
			if mode == "deferred" || mode == "deferred-error" {
				b = newV41DeferredPublishRecorder()
			}
			fault := errors.New("selected publication failure")
			if mode == "publication-error" {
				b.faultAt = 1
				b.fault = fault
			}
			if mode == "deferred-error" {
				b.deferredReadErr = fault
			}
			s := &Session{M: &Model{Cfg: Config{NumLayers: 1, HiddenSize: 2}}, Backend: b}
			t.Cleanup(s.Close)
			input := []float32{1.003, -2.011}
			if mode == "widened" {
				input = []float32{1, -2}
			}
			if mode == "empty" {
				input = nil
			}
			original := append([]float32(nil), input...)
			reference, err := v41LatentNormBF16Publish(0, "attn.wq_a_norm.weight", "projection", input, nil)
			if err != nil {
				t.Fatal(err)
			}
			var observed int
			stop := enableV41BF16Witness()
			defer stop()
			result, err := v41LatentNormBF16PublishObserved(0, "attn.wq_a_norm.weight", "projection", input, s.v41BF16PublishFunc(), func(string) { observed++ })
			fail := mode == "publication-error" || mode == "deferred-error"
			if fail {
				if !errors.Is(err, fault) {
					t.Fatalf("lost selected failure: %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				v41TailRoPECheckBits(t, "device parity", result, reference)
			}
			if observed != 0 || v41HostBF16Rounds.Load() != 0 {
				t.Fatalf("selected/fast path replayed on host: observed=%d rounds=%d", observed, v41HostBF16Rounds.Load())
			}
			calls := b.calls
			if b.v41DeferredLedger != nil {
				calls += b.deferredPublishes
			}
			wantCalls := 1
			if mode == "widened" || mode == "empty" {
				wantCalls = 0
			}
			if calls != wantCalls {
				t.Fatalf("publication calls=%d want=%d", calls, wantCalls)
			}
			v41TailRoPECheckBits(t, "selected input unchanged", input, original)
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41HostArmLatentPublicationOwnedCopyFastPath(t *testing.T) {
	for _, owned := range []bool{false, true} {
		input := []float32{1, -2}
		var calls int
		result, err := v41PublishBF16Input(0, "attn.wq_a_norm.weight projection", input, nil, func(values []float32) ([]float32, error) {
			calls++
			return v41LatentNormBF16Copy(0, "attn.wq_a_norm.weight", "projection", values)
		}, owned)
		if err != nil || calls != 0 {
			t.Fatalf("owned=%v host calls=%d error=%v", owned, calls, err)
		}
		if (&result[0] == &input[0]) != owned {
			t.Fatalf("owned=%v allocation contract changed", owned)
		}
	}
}
