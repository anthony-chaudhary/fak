package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
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

func TestGuardResourceMonitorFailureReceiptPersistsScrubbedDetail(t *testing.T) {
	rawDetail := "owned pids missing from rss census: [101 102]\n" +
		"collector=/vault/alice/private/ps.txt windows=C:\\private\\ps.exe host=db1.corp token=hunter2 address=10.2.3.4 " +
		strings.Repeat("safe-detail ", 80)
	snapshot := procguard.MemorySnapshot{
		Metric:  procguard.MemoryMetricRSS,
		RootPID: 42,
		Processes: []procguard.MemoryProcess{{
			PID:         42,
			Name:        "codex",
			CommandLine: "/Users/alice/bin/codex --api-key=hunter2",
		}},
	}
	event := guardResourceMonitorFailure(42, snapshot, "CHILD_RESOURCE_MONITOR_ERROR", rawDetail)
	if event.Resource == nil {
		t.Fatal("monitor failure has no resource decision")
	}
	receipt := newGuardResourceReceipt("trace", "codex", 42, *event.Resource)
	path := filepath.Join(t.TempDir(), "child-resource.jsonl")
	if err := appendGuardResourceReceipt(path, receipt); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got guardResourceReceipt
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode receipt: %v\n%s", err, data)
	}
	if got.Reason != "CHILD_RESOURCE_MONITOR_ERROR" || !strings.Contains(got.Detail, "owned pids missing from rss census: [101 102]") {
		t.Fatalf("typed public detail was not preserved: %+v", got)
	}
	if got.Detail == "" || len(got.Detail) > guardResourceDetailMaxBytes || strings.ContainsAny(got.Detail, "\r\n\t") {
		t.Fatalf("detail is not nonblank, bounded and one-line: %q", got.Detail)
	}
	for _, leaked := range []string{"/vault/alice", `C:\private`, "db1.corp", "hunter2", "10.2.3.4"} {
		if strings.Contains(got.Detail, leaked) || strings.Contains(event.Reason, leaked) {
			t.Errorf("resource detail leaked %q: receipt=%q event=%q", leaked, got.Detail, event.Reason)
		}
	}
	if got.OffenderCommand != "" || strings.Contains(string(data), snapshot.Processes[0].CommandLine) {
		t.Fatalf("receipt persisted raw offender command: %s", data)
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

func TestGuardResourcePolicyEdgeAndAdversarialInputs(t *testing.T) {
	const maxSafeMB = ^uint64(0) >> 20
	for _, key := range []string{"FAK_CHILD_MAX_MEMORY_MB", "FAK_CHILD_MAX_COMMIT_MB", "FAK_CHILD_MAX_RSS_MB", "FAK_SYSTEM_COMMIT_HEADROOM_MB", "FAK_CHILD_RESOURCE_POLL"} {
		t.Setenv(key, "")
	}
	defaults := guardResourcePolicyFromEnv()
	tests := []struct {
		name       string
		memory     string
		headroom   string
		poll       string
		wantMemory uint64
		wantHead   uint64
		wantPoll   time.Duration
		override   bool
	}{
		{name: "empty"},
		{name: "whitespace", memory: "  ", headroom: "\t", poll: "\n"},
		{name: "malformed", memory: "twelve", headroom: "1GB", poll: "soon"},
		{name: "hostile signs", memory: "-1", headroom: "+1", poll: "-1s"},
		{name: "zero cannot disable containment", memory: "0", headroom: "0", poll: "0s"},
		{name: "oversized memory cannot wrap", memory: "17592186044416", headroom: "17592186044416", poll: "999999999999999999999h"},
		{name: "largest safe memory", memory: strconv.FormatUint(maxSafeMB, 10), headroom: strconv.FormatUint(maxSafeMB, 10), poll: "100ms", wantMemory: maxSafeMB << 20, wantHead: maxSafeMB << 20, wantPoll: 100 * time.Millisecond, override: true},
		{name: "poll below floor", poll: "99ms"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("FAK_CHILD_MAX_MEMORY_MB", tt.memory)
			t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "")
			t.Setenv("FAK_CHILD_MAX_RSS_MB", "")
			t.Setenv("FAK_SYSTEM_COMMIT_HEADROOM_MB", tt.headroom)
			t.Setenv("FAK_CHILD_RESOURCE_POLL", tt.poll)
			wantMemory, wantHead, wantPoll := defaults.MaxTreeBytes, defaults.MinSystemHeadroom, defaults.PollInterval
			if tt.override {
				wantMemory, wantHead, wantPoll = tt.wantMemory, tt.wantHead, tt.wantPoll
			}
			got := guardResourcePolicyFromEnv()
			if got.MaxTreeBytes != wantMemory || got.MinSystemHeadroom != wantHead || got.PollInterval != wantPoll {
				t.Fatalf("policy=%+v, want memory=%d headroom=%d poll=%s", got, wantMemory, wantHead, wantPoll)
			}
		})
	}
}

func TestDecideGuardResourceEdgeAndAdversarialSnapshots(t *testing.T) {
	tests := []struct {
		name       string
		policy     guardResourcePolicy
		snapshot   procguard.MemorySnapshot
		wantStop   bool
		wantReason string
		wantPID    int
		wantHead   uint64
	}{
		{name: "empty snapshot", policy: guardResourcePolicy{MaxTreeBytes: 100}, snapshot: procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS}, wantStop: false},
		{name: "one byte below tree limit", policy: guardResourcePolicy{MaxTreeBytes: 100}, snapshot: procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS, TreeBytes: 99}, wantStop: false},
		{name: "exact tree limit", policy: guardResourcePolicy{MaxTreeBytes: 100}, snapshot: procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS, TreeBytes: 100}, wantStop: true, wantReason: "CHILD_TREE_RSS_LIMIT"},
		{name: "system counters cannot underflow", policy: guardResourcePolicy{MaxTreeBytes: 1000, MinSystemHeadroom: 1}, snapshot: procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, SystemBytes: 101, SystemLimit: 100}, wantStop: true, wantReason: "SYSTEM_COMMIT_HEADROOM", wantHead: 0},
		{name: "headroom exact threshold", policy: guardResourcePolicy{MaxTreeBytes: 1000, MinSystemHeadroom: 10}, snapshot: procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, SystemBytes: 90, SystemLimit: 100}, wantStop: true, wantReason: "SYSTEM_COMMIT_HEADROOM", wantHead: 10},
		{name: "hostile duplicate and negative pids remain attributed", policy: guardResourcePolicy{MaxTreeBytes: 1}, snapshot: procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS, TreeBytes: 1, Processes: []procguard.MemoryProcess{{PID: -1, Bytes: 9}, {PID: -1, Bytes: 10}, {PID: 0, Bytes: 8}}}, wantStop: true, wantReason: "CHILD_TREE_RSS_LIMIT", wantPID: -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := decideGuardResource(tt.policy, tt.snapshot)
			if d.Stop != tt.wantStop || d.Reason != tt.wantReason || d.Offender.PID != tt.wantPID || d.HeadroomBytes != tt.wantHead {
				t.Fatalf("decision=%+v", d)
			}
			if len(d.OwnedPIDs) != len(tt.snapshot.Processes) {
				t.Fatalf("owned pids=%v, processes=%v", d.OwnedPIDs, tt.snapshot.Processes)
			}
		})
	}
}

func TestGuardResourceReceiptAdversarialStringsRemainOneJSONRecord(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	d := guardResourceDecision{
		Stop: true, Reason: "HOSTILE\nREASON", Metric: procguard.MemoryMetricRSS,
		Offender:  procguard.MemoryProcess{PID: 7, Name: "name\n{\"forged\":true}", CommandLine: "cmd\r\nnext"},
		TreeBytes: 123, ThresholdBytes: 100,
	}
	if err := appendGuardResourceReceipt(path, newGuardResourceReceipt("trace\nnext", "agent\tname", 7, d)); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n"); len(lines) != 1 {
		t.Fatalf("hostile fields forged extra JSONL records: %q", data)
	}
	var got guardResourceReceipt
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("receipt is not valid JSON: %v: %q", err, data)
	}
	if got.TraceID != "trace\nnext" || got.OffenderName != "name\n{\"forged\":true}" || got.OffenderCommand != "" || got.TreeRSSBytes == nil || *got.TreeRSSBytes != 123 {
		t.Fatalf("receipt=%+v", got)
	}
}

func TestGuardResourceErrorPathsEdgeAndAdversarial(t *testing.T) {
	t.Run("empty receipt path", func(t *testing.T) {
		if err := appendGuardResourceReceipt(" \t\n", guardResourceReceipt{}); err == nil || !strings.Contains(err.Error(), "path is empty") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("directory in place of receipt", func(t *testing.T) {
		path := t.TempDir()
		if err := appendGuardResourceReceipt(path, guardResourceReceipt{}); err == nil || !strings.Contains(err.Error(), "open child resource receipt") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("receipt missing decision", func(t *testing.T) {
		if err := guardWriteResourceReceipt(guardChildWaitEvent{}, "trace", "agent", 1); err == nil || !strings.Contains(err.Error(), "missing decision") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("collector failure without processes keeps root", func(t *testing.T) {
		event := guardResourceMonitorFailure(-7, procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS}, "MONITOR_ERROR", "malformed\noutput")
		if event.Resource == nil || event.Resource.Offender.PID != -7 || !slices.Equal(event.Resource.OwnedPIDs, []int{-7}) || event.Resource.Detail != "malformed output" || event.Reason != "MONITOR_ERROR: malformed output" {
			t.Fatalf("event=%+v", event)
		}
	})
}
