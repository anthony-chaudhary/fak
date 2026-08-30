package selfinstall

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVerifyAndStoreSignedTargetPreservesPriorSlot(t *testing.T) {
	dir := t.TempDir()
	candidate := filepath.Join(dir, "candidate")
	body := []byte("verified executable bytes")
	if err := os.WriteFile(candidate, body, 0o755); err != nil {
		t.Fatal(err)
	}
	target := VerifiedTarget{
		MetadataGeneration: 7, SourceCommit: cacheTestCommitA,
		ArtifactDigest: digestBytes(body), ArtifactSize: int64(len(body)), AppVersion: "2.1.0",
	}
	run := func(context.Context, string, string, ...string) (string, bool) {
		out, _ := json.Marshal(candidateVersionIdentity{AppVersion: target.AppVersion, Commit: target.SourceCommit, Stamped: true})
		return string(out), true
	}
	if err := VerifyTarget(context.Background(), run, candidate, dir, target); err != nil {
		t.Fatal(err)
	}
	slot, err := StoreVerifiedSlot(filepath.Join(dir, "fak"), candidate, target)
	if err != nil {
		t.Fatal(err)
	}
	next := target
	next.MetadataGeneration = 8
	next.SourceCommit = cacheTestCommitB
	next.AppVersion = "2.2.0"
	nextBody := []byte("next verified executable")
	next.ArtifactDigest, next.ArtifactSize = digestBytes(nextBody), int64(len(nextBody))
	nextCandidate := filepath.Join(dir, "next")
	if err := os.WriteFile(nextCandidate, nextBody, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := StoreVerifiedSlot(filepath.Join(dir, "fak"), nextCandidate, next); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(slot); err != nil || string(got) != string(body) {
		t.Fatalf("prior verified slot was not recoverable: %q, %v", got, err)
	}
}

func TestVerifyTargetFailsClosed(t *testing.T) {
	dir := t.TempDir()
	candidate := filepath.Join(dir, "candidate")
	body := []byte("artifact")
	if err := os.WriteFile(candidate, body, 0o755); err != nil {
		t.Fatal(err)
	}
	base := VerifiedTarget{MetadataGeneration: 1, SourceCommit: cacheTestCommitA, ArtifactDigest: digestBytes(body), ArtifactSize: int64(len(body)), AppVersion: "1.0.0"}
	run := func(context.Context, string, string, ...string) (string, bool) {
		out, _ := json.Marshal(candidateVersionIdentity{AppVersion: base.AppVersion, Commit: base.SourceCommit, Stamped: true})
		return string(out), true
	}
	for _, tc := range []struct {
		name string
		edit func(*VerifiedTarget)
	}{
		{"digest", func(v *VerifiedTarget) { v.ArtifactDigest = strings.Repeat("0", 64) }},
		{"size", func(v *VerifiedTarget) { v.ArtifactSize++ }},
		{"version", func(v *VerifiedTarget) { v.AppVersion = "2.0.0" }},
		{"commit", func(v *VerifiedTarget) { v.SourceCommit = cacheTestCommitB }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target := base
			tc.edit(&target)
			if err := VerifyTarget(context.Background(), run, candidate, dir, target); err == nil {
				t.Fatal("mismatched target accepted")
			}
		})
	}
}
