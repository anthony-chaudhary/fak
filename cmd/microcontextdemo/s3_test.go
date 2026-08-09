package main

import (
	"context"
	"path/filepath"
	"testing"
)

func TestS3ThousandContextsRestartExactlyOnce(t *testing.T) {
	r, err := runS3(context.Background(), s3Config{
		Contexts: 1000, Workers: 16, ResidentHigh: 16, ResidentLow: 8,
		WarmCap: 4, Turns: 2, MemoryBytes: 64 << 20, Dir: filepath.Join(t.TempDir(), "run"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Completed != 1000 || r.DuplicateEffects != 0 || r.AtRestart.Hibernated != 1000 {
		t.Fatalf("bad report: %+v", r)
	}
}

func TestVerifyS3RejectsOverResidentArtifact(t *testing.T) {
	r := s3Report{Schema: s3Schema, Verdict: "PASS", Contexts: 1000, Completed: 1000, UniqueRetirements: 1000,
		ForcedRestarts: 1, AtRestart: s3StateCounts{Hibernated: 1000}, PeakResident: 17, ResidentLimit: 16,
		FrozenBytes: 1, RestoredTurns: 1000, ReplayTurnsAvoided: 1000, MemoryEnvelopeBytes: 1}
	if err := verifyS3Report(r); err == nil {
		t.Fatal("expected residency failure")
	}
}
