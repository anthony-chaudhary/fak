package vdso

import "testing"

// witness_tracker_test.go — witnesses for the §A4 re-check revocation source.

func TestWitnessTrackerRevokesStaleWitnessOnReObservation(t *testing.T) {
	v := New(16)
	tr := NewWitnessTracker(v)

	// First observation of a resource: nothing revoked.
	if tr.Observe("file:foo.go", "git_sha:foo.go=v1") {
		t.Fatalf("first observation should not revoke")
	}
	if v.Revoked("git_sha:foo.go=v1") {
		t.Fatalf("v1 should not be revoked after a first observation")
	}

	// Re-observe under the SAME witness: still nothing revoked.
	if tr.Observe("file:foo.go", "git_sha:foo.go=v1") {
		t.Fatalf("re-observation under the same witness should not revoke")
	}
	if v.Revoked("git_sha:foo.go=v1") {
		t.Fatalf("v1 still must not be revoked")
	}

	// Re-observe under a DIFFERENT witness (the world moved): the OLD witness is revoked.
	if !tr.Observe("file:foo.go", "git_sha:foo.go=v2") {
		t.Fatalf("a changed witness must report changed=true")
	}
	if !v.Revoked("git_sha:foo.go=v1") {
		t.Fatalf("the stale witness v1 must now be revoked")
	}
	if v.Revoked("git_sha:foo.go=v2") {
		t.Fatalf("the current witness v2 must NOT be revoked")
	}
	if got := tr.Witness("file:foo.go"); got != "git_sha:foo.go=v2" {
		t.Fatalf("tracked witness = %q, want v2", got)
	}
}

func TestWitnessTrackerIndependentResources(t *testing.T) {
	v := New(16)
	tr := NewWitnessTracker(v)
	tr.Observe("a", "wa1")
	tr.Observe("b", "wb1")
	// changing a does not touch b's witness.
	tr.Observe("a", "wa2")
	if !v.Revoked("wa1") {
		t.Fatalf("wa1 should be revoked")
	}
	if v.Revoked("wb1") {
		t.Fatalf("wb1 (untouched resource) must not be revoked")
	}
}

func TestWitnessTrackerEmptyWitnessClearsWithoutRevoking(t *testing.T) {
	v := New(16)
	tr := NewWitnessTracker(v)
	tr.Observe("r", "w1")
	// clearing (empty witness) un-tracks without revoking.
	if tr.Observe("r", "") {
		t.Fatalf("empty witness should not report a change")
	}
	if v.Revoked("w1") {
		t.Fatalf("clearing must not revoke the prior witness")
	}
	if tr.Witness("r") != "" {
		t.Fatalf("resource should be untracked after clear")
	}
}
