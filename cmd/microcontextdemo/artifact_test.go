package main

import "testing"

func TestCheckedInS1Artifact(t *testing.T) {
	if err := verifyArtifact("../../experiments/microcontext/s1-gcp-realendpoint-workers4-pass-2026-08-06.json"); err != nil {
		t.Fatal(err)
	}
}
