package model

import (
	"errors"
	"reflect"
	"testing"
)

// State-only prefill has real continuation state but no logits coverage receipt.
// Artifact separation cannot depend on that optional receipt (#13893). The
// reduced fixtures witness ownership, not published checkpoint or device parity.
// fak-test:runtime fast est=500ms lane=default
func TestV41StateOnlyPrefixModelIdentity(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	foreign := v41IncrementalPlainModel(t, 2)
	if !reflect.DeepEqual(m.Cfg, foreign.Cfg) {
		t.Fatal("fixture models must have identical configuration")
	}
	prefix, next := v41IncrementalPrefix(m.Cfg, 4)
	source := m.NewSession()
	t.Cleanup(source.Close)
	source.PrefillNoLogits(prefix)
	if source.v41Forward == nil || source.v41Forward.forwardCoverage != nil ||
		!reflect.DeepEqual(source.v41Forward.history, prefix) {
		t.Fatal("real state-only prefill must retain history without logits coverage")
	}
	cached, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cached.Close)
	if cached.coverage != nil || cached.v41 == nil || !cached.v41.hadState {
		t.Fatal("snapshot lost the state-only prefill premise")
	}
	cached.PrepareForCache()
	fork, err := cached.CloneForRestore()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(fork.Close)

	// Keep the foreign target at the same position so token-count admission
	// cannot mask a missing model-identity check.
	target := foreign.NewSession()
	t.Cleanup(target.Close)
	target.PrefillNoLogits(prefix)
	beforeCache, beforeState := target.Cache, captureV41ForwardSnapshot(target.v41Forward)
	beforeSnapshot := fork.v41.clone()
	if err := fork.Restore(target); !errors.Is(err, ErrNativeForwardCoverageInvalid) {
		t.Fatalf("cross-model state-only restore = %v, want identity refusal", err)
	}
	if target.Cache != beforeCache ||
		!reflect.DeepEqual(captureV41ForwardSnapshot(target.v41Forward), beforeState) {
		t.Fatal("refused restore changed the foreign target")
	}
	if fork.Cache == nil || fork.v41 == nil || !reflect.DeepEqual(fork.v41.clone(), beforeSnapshot) {
		t.Fatal("refused restore consumed or changed the snapshot")
	}

	// A refused foreign restore must leave the same-model cache hit usable.
	// Both ordinary clones and immutable cache forks retain that legal reuse.
	ordinary, err := cached.Clone()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(ordinary.Close)
	want := append([]float32(nil), source.Step(next)...)
	for _, snapshot := range []*PrefixSnapshot{fork, ordinary} {
		branch := m.NewSession()
		t.Cleanup(branch.Close)
		if err := snapshot.Restore(branch); err != nil {
			t.Fatalf("same-model state-only restore: %v", err)
		}
		if snapshot.Cache != nil || snapshot.v41 != nil {
			t.Fatal("successful restore retained transferred ownership")
		}
		assertV41LogitsClose(t, branch.Step(next), want, "state-only same-model reuse")
	}
	if cached.v41 == nil || !reflect.DeepEqual(cached.v41.history, prefix) {
		t.Fatal("restored branches changed the cached prefix")
	}
}
