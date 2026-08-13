package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/treedoctor"
)

func TestRenderTreeDoctorJSONIncludesActions(t *testing.T) {
	rep := treedoctor.Report{
		Lock: treedoctor.LockState{
			Path:      ".git/fak-commit.lock",
			Present:   true,
			HolderPID: 1234,
			Stale:     true,
		},
		Worktrees: []treedoctor.WorktreeState{
			{Path: "/repo", IsMain: true, Keep: "main checkout"},
			{Path: "/tmp/wt", Head: "abc", Merged: true, Prunable: true},
			{Path: "/tmp/live", Head: "def", Merged: true, Live: true, Keep: "live (touched within window)"},
		},
	}
	var out bytes.Buffer
	err := renderTreeDoctorJSON(&out, treeDoctorJSON{
		Schema:      "fak-tree-doctor/1",
		Apply:       false,
		RepoRoot:    "/repo",
		Trunk:       "origin/main",
		NeedsAction: treeDoctorNeedsAction(rep),
		Report:      rep,
		Actions:     []string{"would reap stale commit lock (dead PID 1234)", "would prune merged worktree /tmp/wt"},
	})
	if err != nil {
		t.Fatalf("renderTreeDoctorJSON: %v", err)
	}

	var got struct {
		Schema      string   `json:"schema"`
		Apply       bool     `json:"apply"`
		RepoRoot    string   `json:"repo_root"`
		Trunk       string   `json:"trunk"`
		NeedsAction bool     `json:"needs_action"`
		Actions     []string `json:"actions"`
		Report      struct {
			Lock struct {
				Path      string `json:"path"`
				Present   bool   `json:"present"`
				HolderPID int    `json:"holder_pid"`
				Stale     bool   `json:"stale"`
			} `json:"lock"`
			Worktrees []struct {
				Path     string `json:"path"`
				Prunable bool   `json:"prunable"`
				Keep     string `json:"keep"`
			} `json:"worktrees"`
		} `json:"report"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("json output did not decode: %v\n%s", err, out.String())
	}
	if got.Schema != "fak-tree-doctor/1" || got.Apply || got.RepoRoot != "/repo" || got.Trunk != "origin/main" {
		t.Fatalf("top-level payload mismatch: %+v", got)
	}
	if !got.NeedsAction || len(got.Actions) != 2 {
		t.Fatalf("action summary mismatch: needs=%v actions=%v", got.NeedsAction, got.Actions)
	}
	if !got.Report.Lock.Stale || got.Report.Lock.HolderPID != 1234 {
		t.Fatalf("lock payload mismatch: %+v", got.Report.Lock)
	}
	if len(got.Report.Worktrees) != 3 || !got.Report.Worktrees[1].Prunable || got.Report.Worktrees[2].Keep == "" {
		t.Fatalf("worktree payload mismatch: %+v", got.Report.Worktrees)
	}
}

func TestTreeDoctorTextStillNamesPlannedApply(t *testing.T) {
	rep := treedoctor.Report{
		Lock: treedoctor.LockState{},
		Worktrees: []treedoctor.WorktreeState{
			{Path: "/repo", IsMain: true, Keep: "main checkout"},
			{Path: "/tmp/wt", Merged: true, Prunable: true},
		},
	}
	var out bytes.Buffer
	renderTreeDoctorText(&out, rep, []string{"would prune merged worktree /tmp/wt"}, false)
	got := out.String()
	for _, want := range []string{"commit lock: none", "worktree PRUNABLE: /tmp/wt", "planned actions (1)", "run with --apply"} {
		if !strings.Contains(got, want) {
			t.Fatalf("text output missing %q:\n%s", want, got)
		}
	}
}

func TestTreeDoctorTrunkDefault(t *testing.T) {
	if got := treeDoctorTrunk(" "); got != treedoctor.DefaultTrunk {
		t.Fatalf("default trunk = %q, want %q", got, treedoctor.DefaultTrunk)
	}
	if got := treeDoctorTrunk("origin/dev"); got != "origin/dev" {
		t.Fatalf("explicit trunk = %q", got)
	}
}

func TestCmdTreeDoctorScratchPathCreatesNamespacedParent(t *testing.T) {
	repo := t.TempDir()
	path, err := treedoctor.PrepareScratchPath(repo, "fleet-loop/tick.json")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(filepath.ToSlash(path), "/_scratch/fleet-loop/tick.json") {
		t.Fatalf("path = %q", path)
	}
}
