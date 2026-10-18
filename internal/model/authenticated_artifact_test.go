package model

import (
	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41AuthenticatedArtifactTraceSeparation(t *testing.T) {
	m := v41ReducedModel(t)
	capture := func() computetrace.Artifact {
		t.Helper()
		rec, stop := computetrace.Enable(4096, "artifact", "artifact")
		defer stop()
		m.Forward([]int{1, 3})
		a := rec.Artifact()
		if len(a.Events) == 0 {
			t.Fatal("forward emitted no trace")
		}
		return a
	}
	unknown := capture()
	if _, ok := m.AuthenticatedArtifactIdentity(); ok {
		t.Fatal("fixture claims checkpoint membership")
	}
	id := AuthenticatedArtifactIdentity{ManifestSHA256: strings.Repeat("a", 64), ShardSetSHA256: strings.Repeat("b", 64)}
	// Internal fixture injection stands in for the separately tested loader.
	// No published checkpoint membership or serving qualification is asserted.
	m.authenticatedArtifact = &id
	old := capture()
	changed := id
	changed.ManifestSHA256 = strings.Repeat("c", 64)
	changed.ShardSetSHA256 = strings.Repeat("d", 64)
	m.authenticatedArtifact = &changed
	newer := capture()
	for _, other := range []computetrace.Artifact{unknown, old} {
		if other.Events[0].WeightDigest == newer.Events[0].WeightDigest || other.Events[0].ProvenanceDigest == newer.Events[0].ProvenanceDigest {
			t.Fatal("trace alias across artifact identities")
		}
		if got := computetrace.CompareActivationTraces(other, newer, computetrace.ActivationTolerance{}); got.Verdict != computetrace.TraceIncomparable {
			t.Fatalf("cross-artifact verdict: %+v", got)
		}
	}
	// A new local path does not change authenticated byte identity.
	changed.ManifestSHA256 = strings.Repeat("e", 64)
	relocated := capture()
	if got := computetrace.CompareActivationTraces(newer, relocated, computetrace.ActivationTolerance{}); got.Verdict != computetrace.TraceEqual {
		t.Fatalf("relocated verdict: %+v", got)
	}
	b := NewQuantBuilder(Config{}, false)
	if err := b.BindAuthenticatedArtifactIdentity(id); err != nil {
		t.Fatal(err)
	}
	if err := b.BindAuthenticatedArtifactIdentity(changed); err == nil {
		t.Fatal("builder rebound identity")
	}
	b2 := NewQuantBuilder(Config{}, false)
	b2.built = true
	if err := b2.BindAuthenticatedArtifactIdentity(id); err == nil {
		t.Fatal("published builder bound identity")
	}
}
