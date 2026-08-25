package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func TestDecideGuardResourceTreeLimitSelectsLargestOffender(t *testing.T) {
	p := guardResourcePolicy{PollInterval: time.Second, Metric: procguard.MemoryMetricCommit, MaxTreeBytes: 100, MinSystemHeadroom: 10}
	s := procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, TreeBytes: 101, SystemBytes: 50, SystemLimit: 1000, Processes: []procguard.MemoryProcess{{PID: 2, Bytes: 60}, {PID: 3, Bytes: 40}}}
	d := decideGuardResource(p, s)
	if !d.Stop || d.Reason != "CHILD_TREE_COMMIT_LIMIT" || d.Metric != procguard.MemoryMetricCommit || d.Offender.PID != 2 {
		t.Fatalf("decision=%+v", d)
	}
}

func TestDecideGuardResourceRSSIsMetricHonest(t *testing.T) {
	p := guardResourcePolicy{Metric: procguard.MemoryMetricRSS, MaxTreeBytes: 100, MinSystemHeadroom: 100}
	d := decideGuardResource(p, procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS, TreeBytes: 101, SystemBytes: 999, SystemLimit: 1000, Processes: []procguard.MemoryProcess{{PID: 2, Bytes: 101}}})
	if !d.Stop || d.Reason != "CHILD_TREE_RSS_LIMIT" || d.Metric != procguard.MemoryMetricRSS {
		t.Fatalf("decision=%+v", d)
	}
	// Darwin has no system-wide RSS pressure sample. A commit-headroom override
	// must not reinterpret physical capacity as current RSS usage.
	d = decideGuardResource(guardResourcePolicy{Metric: procguard.MemoryMetricRSS, MaxTreeBytes: 1000, MinSystemHeadroom: 100}, procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS, TreeBytes: 10, SystemBytes: 999, SystemLimit: 1000})
	if d.Stop {
		t.Fatalf("commit headroom applied to RSS: %+v", d)
	}
}

func TestDecideGuardResourceSystemHeadroom(t *testing.T) {
	p := guardResourcePolicy{Metric: procguard.MemoryMetricCommit, MaxTreeBytes: 1000, MinSystemHeadroom: 100}
	d := decideGuardResource(p, procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, TreeBytes: 10, SystemBytes: 950, SystemLimit: 1000})
	if !d.Stop || d.Reason != "SYSTEM_COMMIT_HEADROOM" || d.HeadroomBytes != 50 {
		t.Fatalf("decision=%+v", d)
	}
}

func TestDecideGuardResourceAllowsHealthyTree(t *testing.T) {
	p := guardResourcePolicy{Metric: procguard.MemoryMetricCommit, MaxTreeBytes: 100, MinSystemHeadroom: 10}
	d := decideGuardResource(p, procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, TreeBytes: 99, SystemBytes: 500, SystemLimit: 1000})
	if d.Stop {
		t.Fatalf("decision=%+v", d)
	}
}

func TestGuardResourcePolicyFromEnv(t *testing.T) {
	t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "123")
	t.Setenv("FAK_SYSTEM_COMMIT_HEADROOM_MB", "456")
	t.Setenv("FAK_CHILD_RESOURCE_POLL", "250ms")
	p := guardResourcePolicyFromEnv()
	if p.MaxTreeBytes != 123<<20 || p.MinSystemHeadroom != 456<<20 || p.PollInterval != 250*time.Millisecond {
		t.Fatalf("policy=%+v", p)
	}
	if runtime.GOOS == "darwin" && p.Metric != procguard.MemoryMetricRSS {
		t.Fatalf("darwin policy metric=%q want rss", p.Metric)
	}
}

func TestGuardResourceGenericOverrideTakesPrecedence(t *testing.T) {
	t.Setenv("FAK_CHILD_MAX_MEMORY_MB", "77")
	t.Setenv("FAK_CHILD_MAX_RSS_MB", "88")
	t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "99")
	if got := guardResourcePolicyFromEnv().MaxTreeBytes; got != 77<<20 {
		t.Fatalf("generic override=%d want %d", got, uint64(77)<<20)
	}
}

func TestGuardTreeRSSDefaultIsHostSized(t *testing.T) {
	for _, tc := range []struct {
		host, want uint64
	}{
		{0, 4 << 30},
		{8 << 30, 2 << 30},
		{16 << 30, 4 << 30},
		{32 << 30, 8 << 30},
		{512 << 30, 64 << 30},
	} {
		if got := guardTreeRSSDefault(tc.host); got != tc.want {
			t.Fatalf("guardTreeRSSDefault(%d)=%d want %d", tc.host, got, tc.want)
		}
	}
}

func TestGuardResourceMonitorFailureIsTypedAndVisible(t *testing.T) {
	snapshot := procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS, RootPID: 42, Processes: []procguard.MemoryProcess{{PID: 42}, {PID: 43}}}
	event := guardResourceMonitorFailure(42, snapshot, "CHILD_RESOURCE_MONITOR_ERROR", "ps failed")
	if event.Kind != guardChildResourceLimit || event.Resource == nil || !event.Resource.Stop || event.Resource.Reason != "CHILD_RESOURCE_MONITOR_ERROR" {
		t.Fatalf("event=%+v", event)
	}
	if !strings.Contains(event.Reason, "ps failed") || len(event.Resource.OwnedPIDs) != 2 {
		t.Fatalf("failure was not visible/owned: %+v", event)
	}
}

func TestGuardResourceReceiptsKeepCommitAndRSSFieldsDistinct(t *testing.T) {
	commitDecision := guardResourceDecision{Metric: procguard.MemoryMetricCommit, Reason: "CHILD_TREE_COMMIT_LIMIT", TreeBytes: 10, SystemBytes: 20, SystemLimit: 30}
	commit, err := json.Marshal(newGuardResourceReceipt("trace", "codex", 1, commitDecision))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(commit), `"memory_metric":"commit"`) || !strings.Contains(string(commit), `"tree_commit_bytes":10`) || strings.Contains(string(commit), `"tree_rss_bytes"`) {
		t.Fatalf("Windows-compatible commit receipt=%s", commit)
	}
	if got := guardResourceReason(commitDecision); got != "CHILD_TREE_COMMIT_LIMIT tree_commit=10 threshold=0 system_commit=20 limit=30 headroom=0 offender_pid=0" {
		t.Fatalf("Windows v1 reason changed: %q", got)
	}
	rss, err := json.Marshal(newGuardResourceReceipt("trace", "codex", 1, guardResourceDecision{Metric: procguard.MemoryMetricRSS, TreeBytes: 10}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(rss), `"memory_metric":"rss"`) || !strings.Contains(string(rss), `"tree_rss_bytes":10`) || strings.Contains(string(rss), `"tree_commit_bytes"`) {
		t.Fatalf("Darwin RSS receipt=%s", rss)
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
	wantBase := base
	if runtime.GOOS == "windows" {
		t.Setenv("APPDATA", base)
	} else {
		t.Setenv("HOME", base)
		if runtime.GOOS == "darwin" {
			wantBase = filepath.Join(base, "Library", "Application Support")
		} else {
			t.Setenv("XDG_CONFIG_HOME", base)
		}
	}
	got := guardResourceReceiptPath()
	want := filepath.Join(wantBase, "fak", "guard", "child-resource.jsonl")
	if got != want {
		t.Fatalf("default receipt path=%q want=%q", got, want)
	}
}
