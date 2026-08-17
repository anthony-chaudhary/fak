package trajectory

import (
	"reflect"
	"testing"
	"time"
)

func TestContinuationBundlePreservesResumeAndForkLineage(t *testing.T) {
	at := time.Date(2026, 8, 17, 20, 0, 0, 0, time.UTC)
	events := []Event{derivedFixture("checkpoint", EventCheckpoint, "saved", at, 1, `{"state":"ready"}`)}
	checkpoint := []byte(`{"cursor":7}`)
	artifacts := []PortableArtifact{{Name: "report", Digest: digestBytes([]byte("report"))}}
	bundle, err := BuildContinuationBundle("s1", "p0", events, checkpoint, artifacts, map[string]string{"codex-jsonl": "1"}, []string{"provider_kv_not_portable"}, at)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := BuildContinuationBundle("s1", "p0", events, checkpoint, artifacts, map[string]string{"codex-jsonl": "1"}, []string{"provider_kv_not_portable"}, at)
	if !reflect.DeepEqual(bundle, again) {
		t.Fatal("nondeterministic bundle")
	}
	resume, err := VerifyContinuation(bundle, "resume", "s1", events, checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	fork, err := VerifyContinuation(bundle, "fork", "s2", events, checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	if resume.ManifestAddress != bundle.ContentAddress || fork.InputIdentity != "s1" || fork.OutputIdentity != "s2" {
		t.Fatalf("lost lineage: %#v %#v", resume, fork)
	}
}
func TestContinuationBundleRejectsTamperingAndMissingState(t *testing.T) {
	at := time.Date(2026, 8, 17, 20, 0, 0, 0, time.UTC)
	events := []Event{derivedFixture("state", EventState, "snapshot", at, 1, `{"state":"ready"}`)}
	checkpoint := []byte(`{"state":"ready"}`)
	bundle, err := BuildContinuationBundle("s1", "", events, checkpoint, nil, nil, nil, at)
	if err != nil {
		t.Fatal(err)
	}
	tampered := bundle
	tampered.Loss = []string{"invented"}
	if tampered.Verify() == nil {
		t.Fatal("tamper accepted")
	}
	if _, err := VerifyContinuation(bundle, "resume", "s1", events, []byte(`{"other":true}`)); err == nil {
		t.Fatal("state mismatch accepted")
	}
	if _, err := VerifyContinuation(bundle, "fork", "s1", events, checkpoint); err == nil {
		t.Fatal("fork reused identity")
	}
	if _, err := BuildContinuationBundle("s1", "", events, nil, nil, nil, nil, at); err == nil {
		t.Fatal("missing checkpoint accepted")
	}
}
