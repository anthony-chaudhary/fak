package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

func TestWipPayloadReadingParentlessRefIsNotEmpty(t *testing.T) {
	repo := t.TempDir()
	git := func(input string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Stdin = strings.NewReader(input)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("", "init", "-q")
	git("", "config", "user.name", "test")
	git("", "config", "user.email", "test@example.invalid")
	if err := os.WriteFile(filepath.Join(repo, "payload.txt"), []byte("checkpoint\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("", "add", "payload.txt")
	obj := git("parentless checkpoint\n", "commit-tree", git("", "write-tree"))
	reading := wipPayloadReading(context.Background(), repo, wipref.RefRecord{Object: obj, Stamp: wipref.Stamp{Scope: []string{"payload.txt"}}})
	got := wipref.BuildPayloadCensus(reading)
	if !got.Read || got.Files != 1 || got.StateOf("payload.txt") != wipref.PayloadAbsent {
		t.Fatalf("parentless payload got %+v; want one measured A file", got)
	}
}

func TestWipPayloadReadingChecksPlumbingStatus(t *testing.T) {
	got := wipPayloadReading(context.Background(), t.TempDir(), wipref.RefRecord{Object: "not-an-object", Stamp: wipref.Stamp{Scope: []string{"payload.txt"}}})
	if got.Read || got.Unreadable == "" {
		t.Fatalf("got %+v; failed plumbing must be unreadable, never empty", got)
	}
}
