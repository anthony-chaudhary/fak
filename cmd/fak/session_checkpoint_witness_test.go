package main

// session_checkpoint_witness_test.go — the end-to-end half of #2425's witness: the CLI
// verb against a REAL git fixture repo. The library half (internal/sessionledger,
// TestCheckpointWitnessBinds) proves the two-axis binding hermetically; this proves the
// verb reads a real workspace, prints BOTH hashes, and that a verify against a mutated
// tracked file / a rewritten ledger fails naming the axis that actually moved.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// runCheckpointWitness drives the verb and returns its streams and exit code.
func runCheckpointWitness(t *testing.T, argv ...string) (stdout, stderr string, code int) {
	t.Helper()
	var out, errb bytes.Buffer
	code = runSessionCheckpointWitness(&out, &errb, argv)
	return out.String(), errb.String(), code
}

func TestSessionCheckpointWitnessPrintsBothHashes(t *testing.T) {
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "seed", map[string]string{"a.txt": "one\n"})
	head, err := git("rev-parse", "HEAD")
	if err != nil {
		t.Fatalf("rev-parse: %s", head)
	}
	headSHA := strings.TrimSpace(head)
	ledgerDir := t.TempDir()

	stdout, stderr, code := runCheckpointWitness(t, "trace-cli", "--repo", repo, "--ledger-dir", ledgerDir)
	if code != 0 {
		t.Fatalf("mint exit %d, stderr: %s", code, stderr)
	}
	// Acceptance: `fak session checkpoint` prints BOTH hashes — the transcript head and
	// the git tree witness.
	if !strings.Contains(stdout, headSHA) {
		t.Fatalf("output does not print the git HEAD SHA %s:\n%s", headSHA, stdout)
	}
	for _, want := range []string{"checkpoint  ", "transcript: ledger head", "tree:"} {
		if !strings.Contains(stdout, want) {
			t.Fatalf("output missing %q:\n%s", want, stdout)
		}
	}

	// The receipt the ledger kept is the one the operator was shown.
	l, err := sessionledger.Open(ledgerDir)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	cp, err := l.LatestCheckpoint("trace-cli")
	if err != nil {
		t.Fatalf("latest checkpoint: %v", err)
	}
	if cp.Tree.HeadSHA != headSHA {
		t.Fatalf("checkpoint HEAD %s, want %s", cp.Tree.HeadSHA, headSHA)
	}
	if !strings.Contains(stdout, string(cp.Hash)) {
		t.Fatalf("output does not print the checkpoint id %s:\n%s", cp.Hash, stdout)
	}
}

func TestSessionCheckpointWitnessVerifyNamesTheTreeAxis(t *testing.T) {
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "seed", map[string]string{"a.txt": "one\n"})
	ledgerDir := t.TempDir()
	args := []string{"trace-tree", "--repo", repo, "--ledger-dir", ledgerDir}

	if _, stderr, code := runCheckpointWitness(t, args...); code != 0 {
		t.Fatalf("mint exit %d: %s", code, stderr)
	}
	stdout, stderr, code := runCheckpointWitness(t, append(append([]string{}, args...), "--verify")...)
	if code != 0 {
		t.Fatalf("checkpoint then verify should pass, exit %d: %s", code, stderr)
	}
	if !strings.Contains(stdout, "verified") {
		t.Fatalf("verify output does not say verified:\n%s", stdout)
	}

	// Mutate one TRACKED file: the tree axis must catch it, and say so.
	if err := os.WriteFile(filepath.Join(repo, "a.txt"), []byte("two\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, stderr, code = runCheckpointWitness(t, append(append([]string{}, args...), "--verify")...)
	if code != 1 {
		t.Fatalf("verify after a tracked-file edit should fail, exit %d", code)
	}
	if !strings.Contains(stderr, "tree axis") {
		t.Fatalf("failure does not name the tree axis: %s", stderr)
	}
	if strings.Contains(stderr, "transcript") {
		t.Fatalf("a tree-only mutation must not accuse the transcript axis: %s", stderr)
	}

	// The witness is over CONTENT, not mtime: restoring the bytes restores the binding.
	if err := os.WriteFile(filepath.Join(repo, "a.txt"), []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, stderr, code = runCheckpointWitness(t, append(append([]string{}, args...), "--verify")...); code != 0 {
		t.Fatalf("restoring the file's bytes should restore the binding, exit %d: %s", code, stderr)
	}

	// A commit MOVES HEAD, which is the other half of the tree axis.
	commitFiles(t, repo, git, "second", map[string]string{"b.txt": "b\n"})
	_, stderr, code = runCheckpointWitness(t, append(append([]string{}, args...), "--verify")...)
	if code != 1 {
		t.Fatalf("verify after HEAD moved should fail, exit %d", code)
	}
	if !strings.Contains(stderr, "tree axis") || !strings.Contains(stderr, "HEAD moved") {
		t.Fatalf("failure does not name the moved HEAD on the tree axis: %s", stderr)
	}
}

func TestSessionCheckpointWitnessVerifyNamesTheTranscriptAxis(t *testing.T) {
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "seed", map[string]string{"a.txt": "one\n"})
	ledgerDir := t.TempDir()
	const trace = "trace-transcript"

	// A turn ahead of the checkpoint gives the rewrite something to corrupt that is NOT
	// the checkpoint record itself — the binding must catch a rewrite anywhere in the
	// history it sits on.
	l, err := sessionledger.Open(ledgerDir)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	if _, err := l.Append(trace, "turn", []byte(`{"n":1}`)); err != nil {
		t.Fatalf("seed turn: %v", err)
	}
	args := []string{trace, "--repo", repo, "--ledger-dir", ledgerDir}
	if _, stderr, code := runCheckpointWitness(t, args...); code != 0 {
		t.Fatalf("mint exit %d: %s", code, stderr)
	}

	// Rewrite the append-only log under the checkpoint. The workspace is untouched, so
	// only the transcript axis may fire.
	log := filepath.Join(ledgerDir, sessionledger.LogName)
	raw, err := os.ReadFile(log)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	tampered := bytes.Replace(raw, []byte(`{"n":1}`), []byte(`{"n":9}`), 1)
	if bytes.Equal(tampered, raw) {
		t.Fatal("tamper matched nothing — the ledger did not record the seeded turn")
	}
	if err := os.WriteFile(log, tampered, 0o600); err != nil {
		t.Fatalf("write tampered ledger: %v", err)
	}

	_, stderr, code := runCheckpointWitness(t, append(append([]string{}, args...), "--verify")...)
	if code != 1 {
		t.Fatalf("verify against a rewritten ledger should fail, exit %d", code)
	}
	if !strings.Contains(stderr, "transcript axis") {
		t.Fatalf("failure does not name the transcript axis: %s", stderr)
	}

	// --json carries the same verdict as data: verified=false plus the axis, so a caller
	// branches without parsing prose.
	stdout, _, code := runCheckpointWitness(t, append(append([]string{}, args...), "--verify", "--json")...)
	if code != 1 {
		t.Fatalf("--json verify should keep the failing exit code, got %d", code)
	}
	var rep checkpointWitnessReport
	if err := json.Unmarshal([]byte(stdout), &rep); err != nil {
		t.Fatalf("decode --json report %q: %v", stdout, err)
	}
	if rep.Verified == nil || *rep.Verified {
		t.Fatalf("report should say verified=false: %+v", rep)
	}
	if rep.Axis != sessionledger.AxisTranscript {
		t.Fatalf("report axis = %q, want %q", rep.Axis, sessionledger.AxisTranscript)
	}
}

func TestSessionCheckpointWitnessUsage(t *testing.T) {
	if _, stderr, code := runCheckpointWitness(t); code != 2 {
		t.Fatalf("a bare verb is a usage error, exit %d: %s", code, stderr)
	}
	if _, _, code := runCheckpointWitness(t, "t", "--untracked", "sometimes"); code != 1 && code != 2 {
		t.Fatalf("an unknown --untracked mode must be refused, exit %d", code)
	}
	if _, stderr, code := runCheckpointWitness(t, "t", "--repo", filepath.Join(t.TempDir(), "nope")); code != 1 {
		t.Fatalf("a non-repo must fail closed, exit %d: %s", code, stderr)
	}
}
