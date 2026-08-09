package main

import "testing"

func TestVerifyAPIOnlyArtifact(t *testing.T) {
	if err := verifyAPIOnlyArtifact("../../experiments/microcontext/s6-groq-api-only-4-pass-2026-08-06.json"); err != nil {
		t.Fatal(err)
	}
}
