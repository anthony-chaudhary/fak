package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// These recorders exercise the production binding functions, not CUDA math.
// Method presence deliberately advertises deferred support in every case.
type v41NamedDeferredQualificationRecorder struct {
	*v41RoutedIDTestBackend
	name   string
	probes int
}

func (b *v41NamedDeferredQualificationRecorder) Name() string { return b.name }
func (b *v41NamedDeferredQualificationRecorder) V41RoutedIDAvailable() bool {
	b.probes++
	return true
}

type v41ExplicitDeferredQualificationRecorder struct {
	*v41NamedDeferredQualificationRecorder
	qualified bool
}

func (b *v41ExplicitDeferredQualificationRecorder) V41RoutedLayerQualified() bool {
	return b.qualified
}

// fak-test:runtime fast est=1s lane=default
func TestV41RoutedLayerQualificationAdmission(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41PrefillExpertPanelEnv, "1")
	m := v41IncrementalExpertFixture(t, true, false)
	for _, tc := range []struct {
		name        string
		backendName string
		explicit    bool
		qualified   bool
		want        bool
	}{
		{name: "CUDA methods alone", backendName: "cuda"},
		{name: "CUDA explicit refusal", backendName: "cuda", explicit: true},
		{name: "CUDA future qualified", backendName: "cuda", explicit: true, qualified: true, want: true},
		{name: "legacy Vulkan", backendName: "vulkan", want: true},
		{name: "explicit backend refusal", backendName: "vulkan", explicit: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := newV41RoutedIDTestBackend()
			named := &v41NamedDeferredQualificationRecorder{v41RoutedIDTestBackend: recorder, name: tc.backendName}
			var backend compute.Backend = named
			if tc.explicit {
				backend = &v41ExplicitDeferredQualificationRecorder{v41NamedDeferredQualificationRecorder: named, qualified: tc.qualified}
			}
			s := &Session{M: m, Backend: backend}
			if s.v41RoutedExpertFunc() == nil {
				t.Fatal("fixture did not bind the prerequisite per-pick seam")
			}
			if got := s.v41RoutedLayerFunc() != nil; got != tc.want {
				t.Fatalf("batched routed binding=%v want %v", got, tc.want)
			}
			if !tc.want {
				_, outcome, err := s.v41RoutedExpertPanel(0, "unused", nil, nil, nil)
				if outcome != v41RoutedExpertDeclined || err != nil || named.probes != 0 {
					t.Fatalf("unqualified expert panel outcome=%v err=%v availability probes=%d", outcome, err, named.probes)
				}
			}
			if recorder.calls != 0 || recorder.deferredReads != 0 || recorder.publishes != 0 || len(recorder.pending) != 0 {
				t.Fatal("admission performed work or acquired status ownership")
			}
		})
	}
}
