package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wiprecon"
)

type recoveryMatrixRow struct {
	ID         int    `json:"id"`
	Boundary   string `json:"boundary"`
	Status     string `json:"status"`
	NextAction string `json:"next_action"`
}

// TestRecoveryCrashMatrixCatalog keeps the checked-in operator artifact typed,
// complete, and actionable. The live tests below provide the Git witnesses for
// its rows; this catalog makes those results consumable without scraping test logs.
func TestRecoveryCrashMatrixCatalog(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "docs", "proofs", "recovery-crash-matrix.json"))
	if err != nil {
		t.Fatal(err)
	}
	var rows []recoveryMatrixRow
	if err := json.Unmarshal(b, &rows); err != nil {
		t.Fatal(err)
	}
	if len(rows) != 10 {
		t.Fatalf("matrix rows=%d want 10", len(rows))
	}
	for i, row := range rows {
		if row.ID != i+1 || row.Boundary == "" || row.Status == "" || row.NextAction == "" {
			t.Fatalf("row %d is not typed/actionable: %+v", i+1, row)
		}
	}
}

// TestRecoveryCrashMatrixCheckpointBoundaries is the real-Git acceptance row for
// #5999(1-2): source/index bytes are unchanged around both injected stops; only
// the post-CAS stop has a reachable checkpoint ref with the exact delta.
func TestRecoveryCrashMatrixCheckpointBoundaries(t *testing.T) {
	requireRecoveryGit(t)
	for _, tc := range []struct {
		point   string
		wantRef bool
	}{{"before-ref-update", false}, {"after-ref-update", true}} {
		t.Run(tc.point, func(t *testing.T) {
			repo := initRecoveryRepo(t)
			path := filepath.Join(repo, "a.txt")
			if err := os.WriteFile(path, []byte("two\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			before, _ := os.ReadFile(path)
			indexBefore := runRecoveryGit(t, repo, "write-tree")
			old := wipCheckpointFault
			wipCheckpointFault = func(point string) error {
				if point == tc.point {
					return errors.New("injected crash")
				}
				return nil
			}
			defer func() { wipCheckpointFault = old }()
			_, err := wipCheckpoint(context.Background(), repo, "matrix", true, 1234)
			if err == nil || !strings.Contains(err.Error(), "injected crash") {
				t.Fatalf("checkpoint err=%v", err)
			}
			after, _ := os.ReadFile(path)
			if string(after) != string(before) {
				t.Fatalf("source mutated %q -> %q", before, after)
			}
			if got := runRecoveryGit(t, repo, "write-tree"); got != indexBefore {
				t.Fatalf("index mutated %s -> %s", indexBefore, got)
			}
			ref := "refs/fak/wip/matrix"
			exists := recoveryGitOK(repo, "show-ref", "--verify", "--quiet", ref)
			if exists != tc.wantRef {
				t.Fatalf("ref exists=%v want %v", exists, tc.wantRef)
			}
			if tc.wantRef {
				if got := strings.TrimSpace(runRecoveryGit(t, repo, "show", ref+":a.txt")); got != "two" {
					t.Fatalf("checkpoint bytes=%q", got)
				}
				res, err := wipReconcile(context.Background(), repo)
				if err != nil || len(res.Decisions) == 0 {
					t.Fatalf("reconcile result=%+v err=%v", res, err)
				}
			}
		})
	}
}

// Row 3 proves the checkpoint mechanism itself, not just worker candidates,
// survives loss of the source clone after an independently read-back mirror.
func TestRecoveryCrashMatrixCheckpointRemoteHostLoss(t *testing.T) {
	requireRecoveryGit(t)
	ctx := context.Background()
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	source := filepath.Join(root, "source")
	fresh := filepath.Join(root, "fresh")
	runRecoveryGit(t, root, "init", "--bare", "-q", remote)
	if err := os.Mkdir(source, 0o755); err != nil {
		t.Fatal(err)
	}
	runRecoveryGit(t, source, "init", "-q", "-b", "main")
	runRecoveryGit(t, source, "config", "user.name", "Test")
	runRecoveryGit(t, source, "config", "user.email", "test-at-example.invalid")
	if err := os.WriteFile(filepath.Join(source, "a.txt"), []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runRecoveryGit(t, source, "add", "a.txt")
	runRecoveryGit(t, source, "commit", "-q", "-m", "base")
	runRecoveryGit(t, source, "remote", "add", "origin", remote)
	runRecoveryGit(t, source, "push", "-q", "-u", "origin", "main")
	if err := os.WriteFile(filepath.Join(source, "a.txt"), []byte("checkpoint\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, source, "remote-matrix", true, 4242); err != nil {
		t.Fatal(err)
	}
	push, err := wipSync(ctx, source, "origin", true, false)
	if err != nil || !push.Pushed || push.Replicated < 1 {
		t.Fatalf("push=%+v err=%v", push, err)
	}
	runRecoveryGit(t, root, "clone", "-q", remote, fresh)
	fetch, err := wipSync(ctx, fresh, "origin", false, true)
	if err != nil || !fetch.Fetched || fetch.Mirrored < 1 {
		t.Fatalf("fetch=%+v err=%v", fetch, err)
	}
	ref := "refs/fak/remotewip/origin/remote-matrix"
	if !recoveryGitOK(fresh, "show-ref", "--verify", "--quiet", ref) {
		t.Fatalf("fresh clone missing %s", ref)
	}
	if got := strings.TrimSpace(runRecoveryGit(t, fresh, "show", ref+":a.txt")); got != "checkpoint" {
		t.Fatalf("fresh checkpoint bytes=%q", got)
	}
	// A fresh clone stores fetched checkpoints in its per-remote mirror. The
	// exact next action is to restore/inspect that mirror entry before reconcile.
	if fetch.SyncedAt == 0 || fetch.Source == "" {
		t.Fatalf("fresh mirror has no freshness receipt: %+v", fetch)
	}
}

// Row 7 pins the pure CAS ownership decision used by the production adopt
// command: one successor wins, a concurrent successor is refused, and takeover
// is available only after both expiry and independent holder-death evidence.
func TestRecoveryCrashMatrixAdoptionOwnership(t *testing.T) {
	now := int64(1000)
	reqA := wiprecon.AdoptRequest{
		Session: "crashed", Action: wiprecon.ActReclaim,
		CheckpointRef: "refs/fak/wip/crashed", CheckpointSHA: strings.Repeat("a", 40),
		DeltaDigest: strings.Repeat("d", 64), Successor: "resume-a", Now: now, TTLSeconds: 60,
	}
	first := wiprecon.DecideAdopt(nil, reqA)
	if !first.Verdict.Granted() {
		t.Fatalf("first decision=%+v", first)
	}
	receipt, changed := wiprecon.ApplyAdopt(nil, reqA, first.Verdict)
	if !changed {
		t.Fatal("first ownership claim did not change receipt")
	}
	reqB := reqA
	reqB.Successor = "resume-b"
	if got := wiprecon.DecideAdopt(&receipt, reqB); got.Verdict.Granted() {
		t.Fatalf("concurrent successor won: %+v", got)
	}
	reqB.Now = receipt.ExpiresAt() + 1
	reqB.IncumbentLive = true
	if got := wiprecon.DecideAdopt(&receipt, reqB); got.Verdict.Granted() {
		t.Fatalf("stale-time alone authorized takeover: %+v", got)
	}
	reqB.IncumbentLive = false
	takeover := wiprecon.DecideAdopt(&receipt, reqB)
	if !takeover.Verdict.Granted() {
		t.Fatalf("expired dead-holder takeover=%+v", takeover)
	}
	moved := reqA
	moved.CheckpointSHA = strings.Repeat("b", 40)
	if got := wiprecon.DecideAdopt(&receipt, moved); got.Verdict.Granted() {
		t.Fatalf("moved checkpoint retained ownership: %+v", got)
	}
}

func initRecoveryRepo(t *testing.T) string {
	t.Helper()
	repo := filepath.Join(t.TempDir(), "repo")
	if err := os.Mkdir(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	runRecoveryGit(t, repo, "init", "-q", "-b", "main")
	runRecoveryGit(t, repo, "config", "user.name", "Test")
	runRecoveryGit(t, repo, "config", "user.email", "test-at-example.invalid")
	runRecoveryGit(t, repo, "config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(repo, "a.txt"), []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runRecoveryGit(t, repo, "add", "a.txt")
	runRecoveryGit(t, repo, "commit", "-q", "-m", "base")
	return repo
}

func requireRecoveryGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
}
func runRecoveryGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	out, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}
func recoveryGitOK(dir string, args ...string) bool {
	c := exec.Command("git", args...)
	c.Dir = dir
	return c.Run() == nil
}
