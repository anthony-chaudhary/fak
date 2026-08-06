package main

import "testing"

func TestCheckedInS1Artifact(t *testing.T) {
	if err := verifyArtifact("../../experiments/microcontext/s1-gcp-realendpoint-workers4-pass-2026-08-06.json"); err != nil {
		t.Fatal(err)
	}
}

func TestCheckedInS3Artifact(t *testing.T) {
	if err := verifyS3Artifact("../../experiments/microcontext/s3-local-hibernate-restart-2026-08-06.json"); err != nil {
		t.Fatal(err)
	}
}
