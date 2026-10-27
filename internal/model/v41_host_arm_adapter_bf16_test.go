package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmAdapterBF16Norm(t *testing.T) {
	for _, tc := range []struct {
		name, fault string
		input       []float32
		want        int
		fail        bool
	}{
		{"host", "", []float32{1.003, -2.011}, 2, false},
		{"invalid-first", "", []float32{float32(math.NaN()), 1}, 0, true},
		{"invalid-later", "", []float32{1, float32(math.Inf(1))}, 0, true},
		{"input-overflow", "", []float32{math.MaxFloat32, 1}, 1, true},
		{"rms-failure", "rmsnorm", []float32{1, 2}, 1, true},
		{"nonfinite-output", "nonfinite readback", []float32{1, 2}, 1, true},
		{"output-overflow", "BF16 output", []float32{1, 2}, 2, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gain := []float32{0.75, -1.25}
			s, b := v41CompressorNormTestSession(t, v41CompressorNormTinyModel(t, gain, 1e-5))
			var stages []string
			s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
			if tc.fault != "" {
				b.faultLayer = 0
				b.faultSite = tc.fault
				b.fault = &compute.BackendError{Backend: "norm-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			}
			got, err := s.v41CompressorNormFunc()(0, tc.input, gain, 1e-5)
			if (err != nil) != tc.fail {
				t.Fatalf("failure=%v want=%v", err, tc.fail)
			}
			want := make([]string, tc.want)
			for i := range want {
				want[i] = "bf16-boundary"
			}
			if len(stages) != tc.want || (tc.want > 0 && !reflect.DeepEqual(stages, want)) {
				t.Fatalf("stages=%v want=%v", stages, want)
			}
			if !tc.fail {
				v41TailRoPECheckBits(t, "norm", got, v41CompressorNormRefTail(tc.input, gain, 1e-5, ""))
				if len(b.records) != 1 || b.readbacks != 1 {
					t.Fatal("normalization replayed")
				}
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmAdapterBF16Publication(t *testing.T) {
	for _, tc := range []struct {
		name, fault    string
		rows, want     int
		resident, fail bool
	}{
		{"latent", "", 1, 1, false, false}, {"index-latent", "", 2, 2, false, false},
		{"resident", "", 2, 0, true, false}, {"backend-error", "error", 2, 0, false, true},
		{"read-length", "read-length", 2, 0, false, true},
		{"invalid-first-index", "index-nonfinite", 2, 0, false, true},
		{"index-overflow", "index-overflow", 2, 1, false, true},
		{"invalid-later-latent", "latent-nonfinite", 2, 2, false, true},
		{"invalid-later-single-tail", "latent-nonfinite", 1, 1, false, true},
		{"invalid-first-single-tail", "", 1, 0, false, true},
		{"invalid-later-index", "", 2, 1, false, true},
		{"latent-overflow", "latent-overflow", 2, 2, false, true},
		{"resident-publication-error", "index-overflow", 2, 0, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newV41PublicationRoPERecorder()
			b.publishes = tc.resident
			b.fault = tc.fault
			b.failCall = 1
			b.cause = errors.New("selected tail failure")
			s := &Session{M: &Model{}, Backend: b}
			t.Cleanup(s.Close)
			var stages []string
			s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
			input := []float32{1.003, -2.011, 3.007, -4.013}[:tc.rows*2]
			cos := float32(1)
			if tc.name == "invalid-first-single-tail" {
				input[0] = math.MaxFloat32
				cos = 2
			}
			if tc.name == "invalid-later-index" {
				input[1] = math.MaxFloat32
				cos = 2
			}
			got, err := s.v41CompressedPublicationRoPEFunc()(0, input, []float32{cos}, []float32{0}, tc.rows, 2)
			if (err != nil) != tc.fail {
				t.Fatalf("failure=%v want=%v", err, tc.fail)
			}
			if len(stages) != tc.want {
				t.Fatalf("stages=%v want=%d", stages, tc.want)
			}
			for _, stage := range stages {
				if stage != "bf16-boundary" {
					t.Fatalf("device rotation counted: %v", stages)
				}
			}
			if !tc.fail {
				want := append([]float32(nil), input...)
				for i, v := range want {
					want[i] = v41CompressorNormRefCast(v)
				}
				v41TailRoPECheckBits(t, "tail", got, want)
			}
			if len(b.calls) != 1 {
				t.Fatal("selected rotation replayed")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmAdapterBF16SessionOwnership(t *testing.T) {
	gain := []float32{1, 1}
	m := v41CompressorNormTinyModel(t, gain, 1e-5)
	source, b := v41CompressorNormTestSession(t, m)
	if _, err := source.v41CompressorNormFunc()(0, []float32{1, 2}, gain, 1e-5); err != nil {
		t.Fatal(err)
	}
	target := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
	t.Cleanup(target.Close)
	var old, current int
	source.hostStageObserver = func(string) { old++ }
	target.hostStageObserver = func(string) { current++ }
	b.owner = target
	if _, err := target.v41CompressorNormFunc()(0, []float32{1, 2}, gain, 1e-5); err != nil {
		t.Fatal(err)
	}
	if old != 0 || current != 2 {
		t.Fatalf("session observer ownership old=%d current=%d", old, current)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmAdapterBF16ResidentDeferred(t *testing.T) {
	for _, mode := range []string{"deferred", "deferred-failure", "device-only"} {
		t.Run(mode, func(t *testing.T) {
			b := newV41PublicationRoPERecorder()
			s := &Session{M: &Model{}, Backend: b}
			t.Cleanup(s.Close)
			if mode != "device-only" {
				b.publishes = true
				b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
				if mode == "deferred-failure" {
					b.deferredFaultAt = 1
					b.deferredFault = &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishTailRoPEOutput, Index: 0}
				}
			} else {
				s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			}
			var count int
			s.hostStageObserver = func(string) { count++ }
			_, err := s.v41CompressedPublicationRoPEFunc()(0, []float32{1.003, -2.011}, []float32{1}, []float32{0}, 1, 2)
			if (err != nil) != (mode != "deferred") {
				t.Fatalf("mode=%s failure=%v", mode, err)
			}
			if count != 0 {
				t.Fatalf("device path counted host arms: %d", count)
			}
			if mode != "device-only" && (b.deferredPublishes != 1 || b.deferredFences != 1) {
				t.Fatal("deferred publication replayed or unfenced")
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41HostArmAdapterBF16RestoredSession(t *testing.T) {
	m := v41CompressorTestFixture(t)
	source, b := v41CompressorNormTestSession(t, m)
	source.Prefill([]int{1, 2, 3})
	snap, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	target := v41DenseTestSession(t, m, b)
	if err := snap.Restore(target); err != nil {
		t.Fatal(err)
	}
	var old, current int
	source.hostStageObserver = func(string) { old++ }
	target.hostStageObserver = func(string) { current++ }
	b.owner = target
	gain := m.tensor(layerName(0, v41CompressorNormTestLeaf))
	input := make([]float32, len(gain))
	for i := range input {
		input[i] = 1.003
	}
	if _, err := target.v41State().compressorNorm(0, input, gain, float32(m.Cfg.RMSNormEps)); err != nil {
		t.Fatal(err)
	}
	if old != 0 || current != 2 {
		t.Fatalf("restored observer ownership old=%d current=%d", old, current)
	}
}

type v41HostArmAdapterBF16NormPublisher struct{ *v41BF16PublishRecorder }

func (b *v41HostArmAdapterBF16NormPublisher) SupportsV41CompressorNorm(width int, eps float32) bool {
	return width == 2 && math.Float32bits(eps) == math.Float32bits(1e-5)
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmAdapterBF16NormPublication(t *testing.T) {
	for _, mode := range []string{"resident", "deferred", "publication-error"} {
		t.Run(mode, func(t *testing.T) {
			gain := []float32{0.75, -1.25}
			m := v41CompressorNormTinyModel(t, gain, 1e-5)
			t.Cleanup(func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
			b := &v41HostArmAdapterBF16NormPublisher{newV41BF16PublishRecorder()}
			b.self = b
			if mode == "deferred" {
				b.v41DeferredLedger = newV41DeferredLedger(b.owner)
			}
			fault := errors.New("selected publication failure")
			if mode == "publication-error" {
				b.faultAt = 1
				b.fault = fault
			}
			s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
			t.Cleanup(s.Close)
			var count int
			s.hostStageObserver = func(string) { count++ }
			input := []float32{1.003, -2.011}
			got, err := s.v41CompressorNormFunc()(0, input, gain, 1e-5)
			if mode == "publication-error" {
				if !errors.Is(err, fault) {
					t.Fatalf("lost publication failure: %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				v41TailRoPECheckBits(t, "resident norm", got, v41CompressorNormRefTail(input, gain, 1e-5, ""))
			}
			if count != 0 {
				t.Fatalf("device publication counted host conversion: %d", count)
			}
		})
	}
}
