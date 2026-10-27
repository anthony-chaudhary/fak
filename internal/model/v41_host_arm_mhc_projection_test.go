package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41HostArmMHCProjectionBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("output lengths %d != %d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("output bit mismatch at %d: %08x != %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// fak-test:runtime fast est=0.1s lane=default
func TestV41HostArmMHCProjectionScalar(t *testing.T) {
	for _, tc := range []struct {
		name             string
		full, transposed bool
	}{
		{"full", true, false}, {"transposed", true, true}, {"reduced", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41MHCProjVariant(t, "F32", tc.transposed, !tc.full, 1)
			H := m.Cfg.HiddenSize
			width := H
			if tc.full {
				width *= 4
			}
			input := make([]float32, width)
			for i := range input {
				input[i] = float32(i%7-3) * 0.125
			}
			input[0] = float32(math.Copysign(0, -1))
			var events []string
			observe := func(stage string) { events = append(events, stage) }
			project := m.v41MHCProjectorObserved(0, H, 1e-6, tc.full, tc.transposed, &v41ProjScratch{}, observe)
			got, err := project(input)
			if err != nil {
				t.Fatal(err)
			}
			legacy := m.v41MHCProjector(0, H, 1e-6, tc.full, tc.transposed, &v41ProjScratch{})
			want, err := legacy(input)
			if err != nil {
				t.Fatal(err)
			}
			v41HostArmMHCProjectionBits(t, got, want)
			weight, ok := m.residentF32Mat(layerName(0, "mhc.mixes.weight"))
			if !ok {
				t.Fatal("fixture has no F32 projection weight")
			}
			if tc.full {
				streams := [][]float32{input[:H], input[H : 2*H], input[2*H : 3*H], input[3*H:]}
				plain, e := v41MHCProjectFull(weight, streams, H, 1e-6, tc.transposed)
				if e != nil {
					t.Fatal(e)
				}
				v41HostArmMHCProjectionBits(t, got, plain)
				before := append([]float32(nil), input...)
				for _, run := range []func(func(string)) ([]float32, error){
					func(o func(string)) ([]float32, error) {
						return v41MHCProjectFlatObserved(weight, input, 1e-6, tc.transposed, o)
					},
					func(o func(string)) ([]float32, error) {
						return v41MHCProjectFullObserved(weight, streams, H, 1e-6, tc.transposed, o)
					},
				} {
					var helperEvents []string
					values, e := run(func(stage string) { helperEvents = append(helperEvents, stage) })
					if e != nil {
						t.Fatal(e)
					}
					v41HostArmMHCProjectionBits(t, values, plain)
					v41HostArmMHCProjectionBits(t, input, before)
					if !reflect.DeepEqual(helperEvents, []string{"rms-norm", "mhc-projection"}) {
						t.Fatalf("helper events %v", helperEvents)
					}
				}
			} else {
				v41HostArmMHCProjectionBits(t, got, matRows(weight, input, v41MHCMixWidth, H))
			}
			wantEvents := []string{"mhc-projection"}
			if tc.full {
				wantEvents = []string{"rms-norm", "mhc-projection"}
			}
			if !reflect.DeepEqual(events, wantEvents) {
				t.Fatalf("events %v != %v", events, wantEvents)
			}
			// Cached weight reuse still executes and observes the contraction.
			events = nil
			again, err := project(input)
			if err != nil {
				t.Fatal(err)
			}
			v41HostArmMHCProjectionBits(t, again, got)
			if !reflect.DeepEqual(events, wantEvents) {
				t.Fatalf("repeat events %v", events)
			}
		})
	}
}

// fak-test:runtime fast est=0.1s lane=default
func TestV41HostArmMHCProjectionAdmission(t *testing.T) {
	var events []string
	o := func(stage string) { events = append(events, stage) }
	if _, err := v41MHCProjectFullObserved(nil, nil, 0, 1e-6, false, o); err == nil {
		t.Fatal("invalid full geometry succeeded")
	}
	if _, err := v41MHCProjectFullObserved(nil, [][]float32{{1}, {2}, {3}, {4}}, 1, 1e-6, false, o); err == nil {
		t.Fatal("invalid weight width succeeded")
	}
	if _, err := v41MHCProjectFlatObserved(nil, []float32{1, 2, 3, 4}, 1e-6, false, o); err == nil {
		t.Fatal("invalid flat weight width succeeded")
	}
	m := v41MHCProjVariant(t, "F32", false, true, 1)
	H := m.Cfg.HiddenSize
	for _, scratch := range []*v41ProjScratch{nil, {}} {
		if _, err := m.v41MHCProjectorNamedObserved(0, "mhc.mixes.weight", H, 1e-6, false, false, scratch, o)(nil); err == nil {
			t.Fatal("invalid model input succeeded")
		}
	}
	cause := errors.New("projection fixture failure")
	scratch := &v41ProjScratch{mhcProjection: func(int, []float32, int, float32, bool, bool) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionError, cause
	}}
	_, err := m.v41MHCProjectorObserved(0, H, 1e-6, false, false, scratch, o)(make([]float32, H))
	if !errors.Is(err, cause) {
		t.Fatalf("callback failure = %v", err)
	}
	if len(events) != 0 {
		t.Fatalf("pre-arithmetic refusals emitted %v", events)
	}
	// A selected decline followed by actual scalar fallback must be counted.
	scratch.mhcProjection = func(int, []float32, int, float32, bool, bool) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionDeclined, nil
	}
	if _, err := m.v41MHCProjectorObserved(0, H, 1e-6, false, false, scratch, o)(make([]float32, H)); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"mhc-projection"}) {
		t.Fatalf("fallback events %v", events)
	}
}

type v41HostArmMHCProjectionBackend struct {
	*v41MHCProjBackend
	readValues []float32
}

func (b *v41HostArmMHCProjectionBackend) Read(x compute.Tensor) []float32 {
	values := b.v41MHCProjBackend.Read(x)
	if _, selected := b.outputs[x.Buf()]; selected && b.readValues != nil {
		return append([]float32(nil), b.readValues...)
	}
	return values
}

// fak-test:runtime fast est=0.2s lane=default
func TestV41HostArmMHCProjectionSelected(t *testing.T) {
	for _, tc := range []struct {
		name  string
		full  bool
		fault string
		want  []string
	}{
		{"full", true, "", []string{"rms-norm"}},
		{"reduced", false, "", nil},
		{"declined", true, "decline", nil},
		{"read-shape", true, "shape", nil},
		{"read-nonfinite", true, "nonfinite", nil},
		{"rms-overflow", true, "overflow", []string{"rms-norm"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41MHCProjVariant(t, "F32", false, !tc.full, 1)
			H := m.Cfg.HiddenSize
			width := H
			if tc.full {
				width *= 4
			}
			b := &v41HostArmMHCProjectionBackend{v41MHCProjBackend: newV41MHCProjBackend(width, false)}
			s := v41EngProjSession(t, m, b)
			var events []string
			s.hostStageObserver = func(stage string) { events = append(events, stage) }
			input := make([]float32, width)
			for i := range input {
				input[i] = 1e-5
			}
			switch tc.fault {
			case "decline":
				input = input[:len(input)-1]
			case "shape":
				b.readValues = []float32{1}
			case "nonfinite":
				b.readValues = make([]float32, 24)
				b.readValues[0] = float32(math.NaN())
			case "overflow":
				b.readValues = make([]float32, 24)
				b.readValues[0] = math.MaxFloat32
			}
			got, outcome, err := s.v41MHCProjectionFuncNamed("mhc.mixes.weight")(0, input, H, 1e-8, tc.full, false)
			if tc.fault == "decline" {
				if err != nil || outcome != v41ProjectionDeclined || b.attempts != 0 {
					t.Fatalf("decline outcome %v, %v; attempts %d", outcome, err, b.attempts)
				}
			} else if tc.fault != "" {
				var selected *BackendForwardOperationError
				if outcome != v41ProjectionError || !errors.As(err, &selected) || !s.BackendSessionClosed() {
					t.Fatalf("selected failure outcome %v: %v", outcome, err)
				}
			} else {
				if outcome != v41ProjectionHandled || err != nil || b.attempts != 1 {
					t.Fatalf("selected outcome %v: %v; attempts %d", outcome, err, b.attempts)
				}
				// The observer changes no bits in the existing selected callback.
				s.hostStageObserver = nil
				want, again, e := s.v41MHCProjectionFunc()(0, input, H, 1e-8, tc.full, false)
				if e != nil || again != v41ProjectionHandled {
					t.Fatalf("unobserved selected outcome %v: %v", again, e)
				}
				v41HostArmMHCProjectionBits(t, got, want)
			}
			if !reflect.DeepEqual(events, tc.want) {
				t.Fatalf("events %v != %v", events, tc.want)
			}
		})
	}
}
