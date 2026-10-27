package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=0.1s lane=default
func TestV41HostArmMHCScalarSequence(t *testing.T) {
	var events []string
	observe := func(stage string) { events = append(events, stage) }
	x := []float32{1.125, -2.25}
	streams, err := v41FullInitialStreamsOnObserved(nil, x, observe)
	if err != nil {
		t.Fatal(err)
	}
	wantStreams, err := v41FullInitialStreams(x)
	if err != nil || !reflect.DeepEqual(streams, wantStreams) {
		t.Fatalf("initial streams differ: %v", err)
	}
	mixes, scale, base := make([]float32, 24), []float32{1, 1, 1}, make([]float32, 24)
	mix, err := v41MHCSplitOnObserved(nil, 0, mixes, scale, base, 4, 2, 1e-6, observe)
	if err != nil {
		t.Fatal(err)
	}
	wantMix, err := v41MHCSplit(mixes, scale, base, 4, 2, 1e-6)
	if err != nil || !reflect.DeepEqual(mix, wantMix) {
		t.Fatalf("split differs: %v", err)
	}
	pre, err := v41MHCPreBF16OnObserved(nil, 0, streams, mix.pre, observe)
	if err != nil {
		t.Fatal(err)
	}
	wantPre, err := v41MHCPreBF16(0, streams, mix.pre)
	if err != nil || !reflect.DeepEqual(pre, wantPre) {
		t.Fatalf("pre differs: %v", err)
	}
	next, err := v41MHCPostBF16OnObserved(nil, 0, x, streams, mix, observe)
	if err != nil {
		t.Fatal(err)
	}
	wantNext, err := v41MHCPostBF16(0, x, streams, mix)
	if err != nil || !reflect.DeepEqual(next, wantNext) {
		t.Fatalf("post differs: %v", err)
	}
	want := []string{"bf16-boundary", "mhc-streams", "mhc-streams", "mhc-streams", "bf16-boundary", "bf16-boundary", "mhc-streams", "bf16-boundary", "bf16-boundary", "bf16-boundary", "bf16-boundary"}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
	// The observed pre/post execute their actual scalar multiply/add loops.
	plainPre, err := v41MHCPreObserved(streams, []float32{1, 0, 0, 0}, nil)
	if err != nil || !reflect.DeepEqual(plainPre, x) {
		t.Fatalf("scalar pre = %v, %v", plainPre, err)
	}
	identity := make([]float32, 16)
	for i := 0; i < 4; i++ {
		identity[i*4+i] = 1
	}
	plainPost, err := v41MHCPostObserved(x, streams, make([]float32, 4), identity, nil)
	if err != nil || !reflect.DeepEqual(plainPost, streams) {
		t.Fatalf("scalar post = %v, %v", plainPost, err)
	}
}

// fak-test:runtime fast est=0.1s lane=default
func TestV41HostArmMHCAdmissionAndPartialFailure(t *testing.T) {
	for _, tc := range []struct {
		name string
		run  func(func(string)) error
		want []string
	}{
		{"split-geometry", func(o func(string)) error { _, e := v41MHCSplitObserved(nil, nil, nil, 3, 1, 1e-6, o); return e }, nil},
		{"pre-geometry", func(o func(string)) error { _, e := v41MHCPreObserved(nil, nil, o); return e }, nil},
		{"post-geometry", func(o func(string)) error { _, e := v41MHCPostObserved(nil, nil, nil, nil, o); return e }, nil},
		{"pre-first-width", func(o func(string)) error {
			_, e := v41MHCPreObserved([][]float32{{1}, {2}, {3}, {4}}, nil, o)
			return e
		}, nil},
		{"pre-later-width", func(o func(string)) error {
			_, e := v41MHCPreObserved([][]float32{{1}, nil, {3}, {4}}, []float32{1, 1, 1, 1}, o)
			return e
		}, []string{"mhc-streams"}},
		{"post-first-width", func(o func(string)) error {
			_, e := v41MHCPostObserved([]float32{1}, [][]float32{nil, {2}, {3}, {4}}, make([]float32, 4), make([]float32, 16), o)
			return e
		}, nil},
		{"post-later-width", func(o func(string)) error {
			_, e := v41MHCPostObserved([]float32{1}, [][]float32{{1}, nil, {3}, {4}}, make([]float32, 4), make([]float32, 16), o)
			return e
		}, []string{"mhc-streams"}},
		{"bf16-later-fault", func(o func(string)) error {
			_, e := v41MHCBF16Observed(0, []float32{1, float32(math.NaN())}, o)
			return e
		}, []string{"bf16-boundary"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var events []string
			if err := tc.run(func(stage string) { events = append(events, stage) }); err == nil {
				t.Fatal("invalid input succeeded")
			}
			if !reflect.DeepEqual(events, tc.want) {
				t.Fatalf("events = %v, want %v", events, tc.want)
			}
		})
	}
	var events []string
	observe := func(stage string) { events = append(events, stage) }
	if _, err := v41FullInitialStreamsObserved(nil, observe); err != nil {
		t.Fatal(err)
	}
	if len(events) != 0 {
		t.Fatalf("empty arithmetic emitted %v", events)
	}
	values := []float32{1.003, -0.0, -2.017}
	want, err := v41MHCBF16(0, values)
	if err != nil {
		t.Fatal(err)
	}
	got, err := v41MHCBF16IntoObserved(0, values, values, observe)
	if err != nil || !reflect.DeepEqual(got, want) || !reflect.DeepEqual(events, []string{"bf16-boundary"}) {
		t.Fatalf("in-place BF16 = %v, %v; events %v", got, err, events)
	}
}

// fak-test:runtime fast est=0.2s lane=default
func TestV41HostArmMHCSelectedDispatch(t *testing.T) {
	streams := [][]float32{{1, 2}, {1, 2}, {1, 2}, {1, 2}}
	mix := v41MHCMix{pre: []float32{1, 0, 0, 0}, post: make([]float32, 4), comb: make([]float32, 16)}
	for _, op := range []string{"initial", "split", "pre", "post"} {
		for _, fail := range []bool{false, true} {
			t.Run(op+map[bool]string{false: "/success", true: "/failure"}[fail], func(t *testing.T) {
				m := v41FullStepPatchedNorms(t)
				b := newV41MHCTestBackend(true)
				s := v41DenseTestSession(t, m, b)
				d := s.v41MHCDeviceSeam()
				if d == nil {
					t.Fatal("selected seam absent")
				}
				if fail {
					b.failOp, b.failCall = op, 1
					b.cause = &compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
				}
				var events []string
				o := func(stage string) { events = append(events, stage) }
				var err error
				switch op {
				case "initial":
					_, err = v41FullInitialStreamsOnObserved(d, []float32{1, 2}, o)
				case "split":
					_, err = v41MHCSplitOnObserved(d, 0, make([]float32, 24), []float32{1, 1, 1}, make([]float32, 24), 4, 2, 1e-6, o)
				case "pre":
					_, err = v41MHCPreBF16OnObserved(d, 0, streams, mix.pre, o)
				case "post":
					_, err = v41MHCPostBF16OnObserved(d, 0, []float32{1, 2}, streams, mix, o)
				}
				if len(events) != 0 || b.calls[op] != 1 {
					t.Fatalf("selected %s emitted %v; calls %d", op, events, b.calls[op])
				}
				if fail {
					var selected *V41MHCOperationError
					if !errors.As(err, &selected) || !errors.Is(err, compute.ErrVulkanExecutionFailed) || !s.BackendSessionClosed() {
						t.Fatalf("selected failure = %v", err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

// fak-test:runtime fast est=0.1s lane=default
func TestV41HostArmMHCSelectedDiagnosticsSilent(t *testing.T) {
	d := &v41MHCDevice{}
	var events []string
	o := func(stage string) { events = append(events, stage) }
	_, geometryErr := v41MHCSplitOnObserved(d, 0, nil, nil, nil, 3, 1, 1e-6, o)
	mixes := make([]float32, 24)
	mixes[7] = float32(math.NaN())
	_, finiteErr := v41MHCSplitOnObserved(d, 0, mixes, []float32{1, 1, 1}, make([]float32, 24), 4, 1, 1e-6, o)
	_, initialErr := v41FullInitialStreamsOnObserved(d, []float32{float32(math.NaN())}, o)
	_, preErr := v41MHCPreBF16OnObserved(d, 0, nil, nil, o)
	_, postErr := v41MHCPostBF16OnObserved(d, 0, []float32{float32(math.NaN())}, nil, v41MHCMix{}, o)
	for _, err := range []error{geometryErr, finiteErr, initialErr, preErr, postErr} {
		var stage *V41ForwardError
		if !errors.As(err, &stage) || stage.Stage != v41StageMHC {
			t.Fatalf("diagnostic error = %v", err)
		}
	}
	if len(events) != 0 {
		t.Fatalf("diagnostics emitted host replay: %v", events)
	}
}
