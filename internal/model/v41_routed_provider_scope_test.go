package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ScopedRoutedProviderRecorder struct {
	*v41RoutedIDTestBackend
	routed    compute.V41DeferredStatusBackend
	routedOK  bool
	generic   compute.V41DeferredStatusBackend
	genericOK bool
}

func (b *v41ScopedRoutedProviderRecorder) V41RoutedDeferredStatusForModel() (compute.V41DeferredStatusBackend, bool) {
	return b.routed, b.routedOK
}
func (b *v41ScopedRoutedProviderRecorder) V41DeferredStatusForModel() (compute.V41DeferredStatusBackend, bool) {
	return b.generic, b.genericOK
}

// fak-test:runtime fast est=1s lane=default
func TestV41RoutedProviderScope(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	m := v41IncrementalExpertFixture(t, true, false)
	for _, tc := range []struct {
		name            string
		routed, generic bool
		want            bool
	}{
		{"routed-only", true, false, true},
		{"routed-refusal-authoritative", false, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := newV41RoutedIDTestBackend()
			b := &v41ScopedRoutedProviderRecorder{v41RoutedIDTestBackend: base, routed: base, routedOK: tc.routed, generic: base, genericOK: tc.generic}
			s := &Session{M: m, Backend: b}
			if got := s.v41RoutedLayerFunc() != nil; got != tc.want {
				t.Fatalf("routed binding=%v want %v", got, tc.want)
			}
			if _, ok := compute.ResolveV41DeferredStatus(b); ok != tc.generic {
				t.Fatal("routed provider widened generic consumers")
			}
			if base.calls != 0 || base.deferredReads != 0 {
				t.Fatal("binding issued device work")
			}
		})
	}
	legacy := newV41RoutedLayerTestBackend()
	if (&Session{M: m, Backend: legacy}).v41RoutedLayerFunc() == nil {
		t.Fatal("legacy direct binding changed")
	}
}
