package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The counting fold is the part of `fak bench gitspawn` that can be wrong SILENTLY. A
// miscount does not crash, does not fail a build, and produces a plausible-looking smaller
// number -- which is exactly the failure this verb exists to avoid, since the poll sampler
// it replaces was wrong by 20x while looking perfectly healthy. So the fold is pinned here
// against hand-written Trace2 records rather than only through a live git run.

func writeTrace2(t *testing.T, dir, name string, lines ...string) {
	t.Helper()
	var b []byte
	for _, l := range lines {
		b = append(b, l...)
		b = append(b, '\n')
	}
	if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func gitSpawnStartLine(t *testing.T, sid string, argv ...string) string {
	t.Helper()
	b, err := json.Marshal(gitSpawnStartEvent{Event: "start", Sid: sid, Argv: argv})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

func TestReadGitSpawnEventsCountsOnePerProcess(t *testing.T) {
	dir := t.TempDir()
	// Two top-level processes in separate files -- the per-file layout is what makes the
	// count immune to concurrent-append loss, so the test uses it the way git does.
	writeTrace2(t, dir, "ev-1", gitSpawnStartLine(t, "sid-a", "git", "rev-parse", "HEAD"))
	writeTrace2(t, dir, "ev-2", gitSpawnStartLine(t, "sid-b", "git", "-C", "/repo", "rev-parse", "--short", "HEAD"))
	// A git process spawned BY git carries a hierarchical sid. Counted separately and kept
	// out of the headline, because the hot path did not pay a CreateProcess for it directly.
	writeTrace2(t, dir, "ev-3", gitSpawnStartLine(t, "sid-b/sid-c", "git", "gc", "--auto"))
	// Non-start records, blank lines and garbage must not contribute.
	writeTrace2(t, dir, "ev-4",
		gitSpawnStartLine(t, "sid-d", "git", "status", "--porcelain"),
		`{"event":"exit","sid":"sid-d","code":0}`,
		"",
		"not json at all",
	)

	got, err := readGitSpawnEvents(dir)
	if err != nil {
		t.Fatalf("readGitSpawnEvents: %v", err)
	}
	if got.Top != 3 {
		t.Errorf("Top = %d, want 3 (three depth-0 git processes)", got.Top)
	}
	if got.Nested != 1 {
		t.Errorf("Nested = %d, want 1 (the git-spawned-by-git child)", got.Nested)
	}
	if got.Commands["rev-parse"] != 2 {
		t.Errorf("Commands[rev-parse] = %d, want 2 -- `git -C <dir> rev-parse` must group with plain `git rev-parse`", got.Commands["rev-parse"])
	}
	if got.Commands["status"] != 1 {
		t.Errorf("Commands[status] = %d, want 1", got.Commands["status"])
	}
	if got.Discarded {
		t.Error("Discarded = true with no sentinel present")
	}
}

// A run that trips trace2.maxFiles undercounts, and an undercount reported as a small
// number is worse than one reported as invalid: it reads as an improvement.
func TestReadGitSpawnEventsFlagsDiscardSentinel(t *testing.T) {
	dir := t.TempDir()
	writeTrace2(t, dir, "ev-1", gitSpawnStartLine(t, "sid-a", "git", "rev-parse", "HEAD"))
	writeTrace2(t, dir, gitSpawnDiscardSentinel, "")

	got, err := readGitSpawnEvents(dir)
	if err != nil {
		t.Fatalf("readGitSpawnEvents: %v", err)
	}
	if !got.Discarded {
		t.Fatal("Discarded = false, want true -- a run that dropped files must not report a partial count as if it were complete")
	}
	if got.Top != 1 {
		t.Errorf("Top = %d, want 1 (the sentinel itself is not a process)", got.Top)
	}
}

// An absent directory means the path spawned no git at all. That is a real answer (0), not
// an error -- a later rung that eliminates every spawn on a path must not fail here.
func TestReadGitSpawnEventsMissingDirIsZeroNotError(t *testing.T) {
	got, err := readGitSpawnEvents(filepath.Join(t.TempDir(), "never-created"))
	if err != nil {
		t.Fatalf("missing dir should be zero, got error: %v", err)
	}
	if got.Top != 0 || got.Nested != 0 {
		t.Errorf("got Top=%d Nested=%d, want 0/0", got.Top, got.Nested)
	}
}

func TestGitSpawnSubcommand(t *testing.T) {
	for _, tc := range []struct {
		name string
		argv []string
		want string
	}{
		{"plain", []string{"git", "rev-parse", "HEAD"}, "rev-parse"},
		{"dash_C_value_skipped", []string{"git", "-C", "/repo", "status"}, "status"},
		{"dash_c_config_skipped", []string{"git", "-c", "core.hooksPath=", "commit-tree"}, "commit-tree"},
		{"git_dir_skipped", []string{"git", "--git-dir", "/r/.git", "cat-file"}, "cat-file"},
		{"work_tree_skipped", []string{"git", "--work-tree", "/r", "add"}, "add"},
		// A valueless flag must NOT swallow the subcommand that follows it.
		{"valueless_flag", []string{"git", "--no-pager", "diff"}, "diff"},
		{"no_subcommand", []string{"git", "--version"}, "git"},
		{"empty", []string{}, "git"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := gitSpawnSubcommand(tc.argv); got != tc.want {
				t.Errorf("gitSpawnSubcommand(%q) = %q, want %q", tc.argv, got, tc.want)
			}
		})
	}
}

// The per-command table is what tells a later rung WHICH call is worth batching, so its
// order has to be deterministic: a table that reshuffles between runs cannot be diffed
// against a baseline fixture.
func TestTopCommandsIsStableAndTruncates(t *testing.T) {
	c := gitSpawnCount{Commands: map[string]int{
		"rev-parse": 5, "diff": 1, "add": 1, "commit-tree": 1, "read-tree": 1, "status": 3,
	}}
	got := c.topCommands(3)
	want := []gitSpawnCmdCount{{Command: "rev-parse", Spawns: 5}, {Command: "status", Spawns: 3}, {Command: "add", Spawns: 1}}
	if len(got) != len(want) {
		t.Fatalf("topCommands(3) returned %d entries, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("entry %d = %+v, want %+v (descending by count, then name)", i, got[i], want[i])
		}
	}
	// Ties break by name, so two hosts with the same counts print the same table.
	all := c.topCommands(99)
	if all[2].Command != "add" || all[3].Command != "commit-tree" {
		t.Errorf("tie order = %q,%q, want add,commit-tree", all[2].Command, all[3].Command)
	}
}
