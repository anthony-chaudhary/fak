package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func TestDecideGuardResourceTreeLimitSelectsLargestOffender(t *testing.T) {
	p := guardResourcePolicy{PollInterval: time.Second, MaxTreeCommit: 100, MinSystemHeadroom: 10}
	s := procguard.CommitSnapshot{TreeCommitBytes: 101, SystemCommitBytes: 50, SystemCommitLimit: 1000, Processes: []procguard.CommitProcess{{PID: 2, CommitBytes: 60}, {PID: 3, CommitBytes: 40}}}
	d := decideGuardResource(p, s)
	if !d.Stop || d.Reason != "CHILD_TREE_COMMIT_LIMIT" || d.Offender.PID != 2 {
		t.Fatalf("decision=%+v", d)
	}
}

func TestDecideGuardResourceSystemHeadroom(t *testing.T) {
	p := guardResourcePolicy{MaxTreeCommit: 1000, MinSystemHeadroom: 100}
	d := decideGuardResource(p, procguard.CommitSnapshot{TreeCommitBytes: 10, SystemCommitBytes: 950, SystemCommitLimit: 1000})
	if !d.Stop || d.Reason != "SYSTEM_COMMIT_HEADROOM" || d.HeadroomBytes != 50 {
		t.Fatalf("decision=%+v", d)
	}
}

func TestDecideGuardResourceAllowsHealthyTree(t *testing.T) {
	p := guardResourcePolicy{MaxTreeCommit: 100, MinSystemHeadroom: 10}
	d := decideGuardResource(p, procguard.CommitSnapshot{TreeCommitBytes: 99, SystemCommitBytes: 500, SystemCommitLimit: 1000})
	if d.Stop {
		t.Fatalf("decision=%+v", d)
	}
}

func TestGuardResourcePolicyFromEnv(t *testing.T) {
	t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "123")
	t.Setenv("FAK_SYSTEM_COMMIT_HEADROOM_MB", "456")
	t.Setenv("FAK_CHILD_RESOURCE_POLL", "250ms")
	p := guardResourcePolicyFromEnv()
	if p.MaxTreeCommit != 123<<20 || p.MinSystemHeadroom != 456<<20 || p.PollInterval != 250*time.Millisecond {
		t.Fatalf("policy=%+v", p)
	}
}

func TestAppendGuardResourceReceipt(t *testing.T) {
	path := t.TempDir() + "/receipt.jsonl"
	if err := appendGuardResourceReceipt(path, guardResourceReceipt{Schema: "fak.guard.child-resource.v1", RootPID: 42, DescendantsSurvive: false}); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"root_pid":42`) || !strings.Contains(string(b), `"descendants_survive":false`) {
		t.Fatalf("receipt=%s", b)
	}
}

func TestGuardResourceReceiptPathDefaultsDurably(t *testing.T) {
	t.Setenv("FAK_CHILD_RESOURCE_JOURNAL", "")
	base := t.TempDir()
	t.Setenv("APPDATA", base)
	got := guardResourceReceiptPath()
	want := filepath.Join(base, "fak", "guard", "child-resource.jsonl")
	if got != want {
		t.Fatalf("default receipt path=%q want=%q", got, want)
	}
}
