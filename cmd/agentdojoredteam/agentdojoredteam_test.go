package main

// Unit tests for the pure, deterministic helpers in agentdojoredteam:
//   - ratio: the ASR fraction n/d with a divide-by-zero guard.
//   - verdictLabel: the attack-succeeded boolean rendered as a stream label.
// Both are self-contained (stdlib only, no model/network/fixture), so the
// expected values below are computed by hand and the tests fail on regression.

import (
	"context"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentdojo"
)

func TestRatio(t *testing.T) {
	const eps = 1e-9
	tests := []struct {
		name string
		n    int
		d    int
		want float64
	}{
		{"zero numerator", 0, 10, 0.0},
		{"three tenths", 3, 10, 0.3},
		{"full", 10, 10, 1.0},
		{"quarter", 1, 4, 0.25},
		{"one third", 1, 3, 1.0 / 3.0},
		{"greater than one", 7, 2, 3.5},
		{"zero denominator guard", 5, 0, 0.0},
		{"zero over zero guard", 0, 0, 0.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ratio(tc.n, tc.d)
			if math.Abs(got-tc.want) > eps {
				t.Errorf("ratio(%d, %d) = %v, want %v", tc.n, tc.d, got, tc.want)
			}
		})
	}
}

func TestVerdictLabel(t *testing.T) {
	tests := []struct {
		name            string
		attackSucceeded bool
		want            string
	}{
		{"attack landed", true, "MISSED"},
		{"attack blocked", false, "caught"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := verdictLabel(tc.attackSucceeded); got != tc.want {
				t.Errorf("verdictLabel(%v) = %q, want %q", tc.attackSucceeded, got, tc.want)
			}
		})
	}
}

func TestReproduceCommand(t *testing.T) {
	if got := reproduceCommand(false, 0); got != "go run ./cmd/agentdojoredteam -json" {
		t.Fatalf("default command = %q", got)
	}
	if got := reproduceCommand(true, 7); got != "go run ./cmd/agentdojoredteam -json -seeds -seed 7" {
		t.Fatalf("seeded command = %q", got)
	}
}

func TestCorpusHashStableAcrossPresentationOrder(t *testing.T) {
	ordered := agentdojo.Matrix()
	shuffled := append([]agentdojo.Attack(nil), ordered...)
	for i, j := 0, len(shuffled)-1; i < j; i, j = i+1, j-1 {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	}
	if got, want := corpusHash(shuffled), corpusHash(ordered); got != want {
		t.Fatalf("corpusHash depends on order: got %s want %s", got, want)
	}
	if got := corpusHash(ordered); !strings.HasPrefix(got, "sha256:") {
		t.Fatalf("corpusHash = %q, want sha256 prefix", got)
	}
}

func TestConfigMatrixRowsMarksProductionAndDiagnosticBrackets(t *testing.T) {
	rows := configMatrixRows([]agentdojo.ConfigReport{
		{Config: agentdojo.DefenseConfig{Name: "prod"}, Report: agentdojo.Report{Total: 10, Succeeded: 0}},
		{Config: agentdojo.DefenseConfig{Name: "loose"}, Report: agentdojo.Report{Total: 10, Succeeded: 2, ASR: 0.2}},
	})
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(rows))
	}
	if rows[0].Name != "prod" || rows[0].Role != "production" || rows[0].Gate != "PASS" {
		t.Fatalf("production row = %+v", rows[0])
	}
	if rows[1].Name != "loose" || rows[1].Role != "bracket" || rows[1].Gate != "diagnostic" || rows[1].ASR != 0.2 {
		t.Fatalf("bracket row = %+v", rows[1])
	}
}

func TestGitTreeModifiedIncludesUntrackedFiles(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	runGit(t, dir, "init")
	runGit(t, dir, "config", "user.email", "agentdojo@example.invalid")
	runGit(t, dir, "config", "user.name", "AgentDojo Test")
	if err := os.WriteFile(filepath.Join(dir, "tracked.txt"), []byte("tracked\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, dir, "add", "tracked.txt")
	runGit(t, dir, "commit", "-m", "initial")

	if got := gitTreeModifiedIn(dir); got != "false" {
		t.Fatalf("clean tree modified = %q, want false", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "untracked.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := gitTreeModifiedIn(dir); got != "true" {
		t.Fatalf("tree with untracked file modified = %q, want true", got)
	}
}

func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func TestBenignControlsComplete(t *testing.T) {
	rows := runBenignControls(context.Background())
	if len(rows) == 0 {
		t.Fatal("expected at least one benign control")
	}
	if got := completedBenign(rows); got != len(rows) {
		t.Fatalf("completed benign controls = %d/%d, rows=%+v", got, len(rows), rows)
	}
}
