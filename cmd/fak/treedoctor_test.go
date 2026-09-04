package main

import (
	"bytes"
	"encoding/json"
	"flag"
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

func TestScratchProducerReceiptHumanAndJSON(t *testing.T) {
	receipt := treedoctor.ScratchProducerReceipt{
		Schema:         treedoctor.ScratchProducerReceiptSchema,
		Producer:       "selected",
		ResolvedTarget: filepath.Join("repo", "_scratch", "selected"),
		Verdict:        treedoctor.ScratchProducerReaped,
		RemovedCount:   3,
	}
	var human bytes.Buffer
	writeScratchProducerReceipt(&human, receipt, false)
	for _, want := range []string{"reaped 3 entries", receipt.ResolvedTarget, `producer "selected"`} {
		if !strings.Contains(human.String(), want) {
			t.Fatalf("human receipt missing %q:\n%s", want, human.String())
		}
	}

	var machine bytes.Buffer
	writeScratchProducerReceipt(&machine, receipt, true)
	var got treedoctor.ScratchProducerReceipt
	if err := json.Unmarshal(machine.Bytes(), &got); err != nil {
		t.Fatalf("decode JSON receipt: %v\n%s", err, machine.String())
	}
	if got.Schema != treedoctor.ScratchProducerReceiptSchema || got.ResolvedTarget != receipt.ResolvedTarget || got.Verdict != treedoctor.ScratchProducerReaped || got.RemovedCount != 3 {
		t.Fatalf("JSON receipt = %+v", got)
	}
}

func TestReapScratchRejectsOtherModesAndExtraTargets(t *testing.T) {
	fs := flag.NewFlagSet("tree-doctor-test", flag.ContinueOnError)
	fs.Bool("json", false, "")
	fs.Bool("apply", false, "")
	fs.String("reap-scratch", "", "")
	if err := fs.Parse([]string{"--reap-scratch", "selected", "--apply", "peer"}); err != nil {
		t.Fatal(err)
	}
	got := disallowedTreeDoctorFlags(fs, "reap-scratch", "root", "json")
	if len(got) != 1 || got[0] != "--apply" {
		t.Fatalf("disallowed flags = %v, want [--apply]", got)
	}
	if args := fs.Args(); len(args) != 1 || args[0] != "peer" {
		t.Fatalf("extra targets = %v, want [peer]", args)
	}
}

func TestRenderWIPTextShowsTypedDurableActions(t *testing.T) {
	var out bytes.Buffer
	renderWIPText(&out, []treedoctor.WIPFile{
		{Path: ".claude/goal-prompts/resfleet-6557.md", Kind: "claude-control", Action: "park-or-delete", Class: "abandoned", LandOrPark: true, AgeSeconds: 7200},
		{Path: "internal/x/testdata/case.json", Kind: "test-fixture", Action: "land-or-delete", Class: "abandoned", LandOrPark: true, AgeSeconds: 7200},
	})
	got := out.String()
	for _, want := range []string{"untracked durable WIP", "claude-control", "park-or-delete", "test-fixture", "land-or-delete"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render missing %q:\n%s", want, got)
		}
	}
}

func TestTreeDoctorRenderScratchHygieneWarning(t *testing.T) {
	rep := treedoctor.Report{
		ScratchHygiene: treedoctor.ScratchHygieneReport{
			ScratchUntrackedGoFiles: 10001,
			Threshold:               10000,
			Exceeded:                true,
			Warning:                 "_scratch contains >10,000 untracked .go files (10001) without quarantine; isolate workspace scope or reap scratch to prevent LSP/gopls memory explosion",
		},
	}
	var out bytes.Buffer
	renderTreeDoctorText(&out, rep, nil, false)
	got := out.String()
	if !strings.Contains(got, "warning: _scratch contains >10,000 untracked .go files") {
		t.Fatalf("output missing scratch hygiene warning:\n%s", got)
	}
	if !strings.Contains(got, "LSP/gopls memory explosion") {
		t.Fatalf("output missing memory explosion mention:\n%s", got)
	}
}
