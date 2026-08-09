package wipref

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPayloadCensusStrictlyLargerHeadIsDiverged(t *testing.T) {
	repo := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	git("config", "user.name", "wipref-test")
	git("config", "user.email", "wipref-test@example.invalid")
	path := filepath.Join(repo, "payload.txt")
	if err := os.WriteFile(path, []byte("checkpoint\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "payload.txt")
	git("commit", "-qm", "checkpoint")
	checkpoint := git("rev-parse", "HEAD")
	if err := os.WriteFile(path, []byte("checkpoint\nlater strictly larger form\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("commit", "-qam", "larger head")

	// Trap B: exact blob identity says the checkpoint blob is absent/unlanded.
	headBlob := git("rev-parse", "HEAD:payload.txt")
	checkpointBlob := git("rev-parse", checkpoint+":payload.txt")
	if headBlob == checkpointBlob {
		t.Fatal("fixture does not demonstrate naive unlanded result")
	}

	cmd := exec.CommandContext(context.Background(), "git", "-C", repo, "diff", "--name-status", "--no-renames", "-z", "HEAD", checkpoint, "--", "payload.txt")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("two-dot name-status: %v", err)
	}
	got := BuildPayloadCensus(PayloadReading{Read: true, Paths: []string{"payload.txt"}, NameStatus: string(out)})
	if state := got.StateOf("payload.txt"); state != PayloadDiverged {
		t.Fatalf("state=%q, want %q; raw=%q", state, PayloadDiverged, out)
	}
	if !RefusePayloadCheckout(got.StateOf("payload.txt")) {
		t.Fatal("DIVERGED checkout was not refused")
	}
	if remedy, refused := PayloadRemedy(PayloadDiverged, checkpoint, "payload.txt"); !refused || !strings.Contains(remedy, "git diff HEAD:") {
		t.Fatalf("remedy=%q refused=%v, want three-way refusal", remedy, refused)
	}
}

func TestBuildPayloadCensusUnreadableIsNotEmpty(t *testing.T) {
	got := BuildPayloadCensus(Unread("git diff exited 128"))
	if got.Read || got.Unreadable == "" {
		t.Fatalf("got %+v, want unreadable not empty", got)
	}
}
