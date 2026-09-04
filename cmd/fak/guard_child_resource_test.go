package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
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
	if !d.Stop || d.Reason != "SYSTEM_COMMIT_HEADROOM" || d.HeadroomBytes != 50 || d.ThresholdBytes != p.MinSystemHeadroom {
		t.Fatalf("decision=%+v", d)
	}
	if d.Offender.PID != 0 || d.Offender.PPID != 0 || d.Offender.Name != "host" {
		t.Fatalf("offender=%+v, want host with PID 0", d.Offender)
	}
	if reason := guardResourceReason(d); !strings.Contains(reason, "offender_pid=0") {
		t.Fatalf("reason missing offender_pid=0: %s", reason)
	}

	// Verify that even with existing large child processes, offender is attributed to host (PID 0)
	snapshotWithProcs := procguard.MemorySnapshot{
		Metric:      procguard.MemoryMetricCommit,
		TreeBytes:   10,
		SystemBytes: 950,
		SystemLimit: 1000,
		Processes: []procguard.MemoryProcess{
			{PID: 42, PPID: 1, Name: "child", Bytes: 500},
			{PID: 99, PPID: 42, Name: "grandchild", Bytes: 100},
		},
	}
	d2 := decideGuardResource(p, snapshotWithProcs)
	if !d2.Stop || d2.Reason != "SYSTEM_COMMIT_HEADROOM" {
		t.Fatalf("decision=%+v", d2)
	}
	if d2.Offender.PID != 0 || d2.Offender.PPID != 0 || d2.Offender.Name != "host" {
		t.Fatalf("offender=%+v, want host with PID 0 (did not attribute child's largest PID)", d2.Offender)
	}
	reason2 := guardResourceReason(d2)
	if !strings.Contains(reason2, "offender_pid=0") {
		t.Fatalf("guardResourceReason formatted child PID instead of offender_pid=0: %s", reason2)
	}
	if strings.Contains(reason2, "offender_pid=42") {
		t.Fatalf("guardResourceReason contained child PID 42: %s", reason2)
	}
	receipt := newGuardResourceReceipt("trace", "codex", 42, d2)
	if receipt.OffenderPID != 0 || receipt.OffenderName != "host" {
		t.Fatalf("receipt offender=%d (%s), want 0 (host)", receipt.OffenderPID, receipt.OffenderName)
	}
}

func TestDecideGuardResourceAllowsHealthyTree(t *testing.T) {
	p := guardResourcePolicy{Metric: procguard.MemoryMetricCommit, MaxTreeBytes: 100, MinSystemHeadroom: 10}
	d := decideGuardResource(p, procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, TreeBytes: 99, SystemBytes: 500, SystemLimit: 1000})
	if d.Stop {
		t.Fatalf("decision=%+v", d)
	}
}

func TestGuardResourcePolicyFromExplicitConfig(t *testing.T) {
	old := guardResourceConfigured
	t.Cleanup(func() { setGuardResourceConfig(old) })
	t.Setenv("FAK_SYSTEM_COMMIT_HEADROOM_MB", "456")
	t.Setenv("FAK_CHILD_RESOURCE_POLL", "5s")
	setGuardResourceConfig(guardResourceConfig{MaxMemoryMB: 123, PollInterval: 250 * time.Millisecond})
	p := guardResourcePolicyConfigured()
	wantHeadroom := uint64(456 << 20)
	if runtime.GOOS == "darwin" {
		wantHeadroom = 0
	}
	if p.MaxTreeBytes != 123<<20 || p.MinSystemHeadroom != wantHeadroom || p.PollInterval != 250*time.Millisecond {
		t.Fatalf("policy=%+v", p)
	}
	if runtime.GOOS == "darwin" && p.Metric != procguard.MemoryMetricRSS {
		t.Fatalf("darwin policy metric=%q want rss", p.Metric)
	}
}

func TestGuardResourceLegacyEnvironmentNoLongerOverridesConfig(t *testing.T) {
	old := guardResourceConfigured
	t.Cleanup(func() { setGuardResourceConfig(old) })
	t.Setenv("FAK_CHILD_MAX_MEMORY_MB", "77")
	t.Setenv("FAK_CHILD_MAX_RSS_MB", "88")
	t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "99")
	setGuardResourceConfig(guardResourceConfig{MaxMemoryMB: 66})
	if got := guardResourcePolicyConfigured().MaxTreeBytes; got != 66<<20 {
		t.Fatalf("explicit override=%d want %d", got, uint64(66)<<20)
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

func TestGuardResourceMonitorReasonsStayDistinctFromMeasuredLimits(t *testing.T) {
	tests := []struct {
		name   string
		reason string
		detail string
	}{
		{name: "protected live owned child", reason: "CHILD_RESOURCE_INSPECTION_DENIED", detail: "OpenProcess: Access is denied"},
		{name: "unsupported collector", reason: "CHILD_RESOURCE_COLLECTOR_UNAVAILABLE", detail: "native collector unavailable"},
		{name: "collector failure", reason: "CHILD_RESOURCE_COLLECTOR_FAILURE", detail: "malformed output"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			event := guardResourceMonitorFailure(42, procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, Processes: []procguard.MemoryProcess{{PID: 42}}}, tt.reason, tt.detail)
			if event.Kind != guardChildResourceLimit || event.Resource == nil || event.Resource.Stop || event.Resource.Reason != tt.reason {
				t.Fatalf("event=%+v", event)
			}
			receipt := newGuardResourceReceipt("trace", "codex", 42, *event.Resource)
			if receipt.Reason != tt.reason || receipt.Action != "observe_only" || !receipt.DescendantsSurvive || strings.Contains(receipt.Reason, "_LIMIT") {
				t.Fatalf("receipt=%+v", receipt)
			}
		})
	}
}

func TestGuardResourceMeasuredBreachesKeepLimitReasons(t *testing.T) {
	tests := []struct {
		name     string
		policy   guardResourcePolicy
		snapshot procguard.MemorySnapshot
		reason   string
	}{
		{name: "tree limit", policy: guardResourcePolicy{Metric: procguard.MemoryMetricCommit, MaxTreeBytes: 100}, snapshot: procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, TreeBytes: 101, Processes: []procguard.MemoryProcess{{PID: 42, Bytes: 101}}}, reason: "CHILD_TREE_COMMIT_LIMIT"},
		{name: "system headroom", policy: guardResourcePolicy{Metric: procguard.MemoryMetricCommit, MinSystemHeadroom: 100}, snapshot: procguard.MemorySnapshot{Metric: procguard.MemoryMetricCommit, SystemBytes: 951, SystemLimit: 1000, Processes: []procguard.MemoryProcess{{PID: 42}}}, reason: "SYSTEM_COMMIT_HEADROOM"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := decideGuardResource(tt.policy, tt.snapshot)
			if !d.Stop || d.Reason != tt.reason {
				t.Fatalf("decision=%+v", d)
			}
			if tt.reason == "SYSTEM_COMMIT_HEADROOM" && d.ThresholdBytes != tt.policy.MinSystemHeadroom {
				t.Fatalf("SYSTEM_COMMIT_HEADROOM threshold=%d want %d", d.ThresholdBytes, tt.policy.MinSystemHeadroom)
			}
			if got := newGuardResourceReceipt("trace", "codex", 42, d); got.Reason != tt.reason {
				t.Fatalf("receipt=%+v", got)
			}
		})
	}
}

func TestGuardResourceMonitorFailureIsTypedAndVisible(t *testing.T) {
	snapshot := procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS, RootPID: 42, Processes: []procguard.MemoryProcess{{PID: 42}, {PID: 43}}}
	event := guardResourceMonitorFailure(42, snapshot, "CHILD_RESOURCE_MONITOR_ERROR", "ps failed")
	if event.Kind != guardChildResourceLimit || event.Resource == nil || event.Resource.Stop || event.Resource.Reason != "CHILD_RESOURCE_MONITOR_ERROR" {
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
	old := guardResourceConfigured
	t.Cleanup(func() { setGuardResourceConfig(old) })
	setGuardResourceConfig(guardResourceConfig{})
	legacy := filepath.Join(t.TempDir(), "legacy-resource.jsonl")
	t.Setenv("FAK_CHILD_RESOURCE_JOURNAL", legacy)
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
	if got == legacy {
		t.Fatal("legacy resource journal unexpectedly controls the receipt path")
	}
}

func TestGuardResourcePolicyEdgeAndAdversarialInputs(t *testing.T) {
	old := guardResourceConfigured
	t.Cleanup(func() { setGuardResourceConfig(old) })
	setGuardResourceConfig(guardResourceConfig{})
	defaults := guardResourcePolicyConfigured()
	tests := []struct {
		name       string
		config     guardResourceConfig
		wantMemory uint64
		wantPoll   time.Duration
	}{
		{name: "zero keeps containment defaults"},
		{name: "poll below floor", config: guardResourceConfig{PollInterval: 99 * time.Millisecond}},
		{name: "oversized memory cannot wrap", config: guardResourceConfig{MaxMemoryMB: ^uint64(0)}},
		{name: "explicit values", config: guardResourceConfig{MaxMemoryMB: 512, PollInterval: 100 * time.Millisecond}, wantMemory: 512 << 20, wantPoll: 100 * time.Millisecond},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setGuardResourceConfig(tt.config)
			wantMemory, wantPoll := defaults.MaxTreeBytes, defaults.PollInterval
			if tt.wantMemory != 0 {
				wantMemory = tt.wantMemory
			}
			if tt.wantPoll != 0 {
				wantPoll = tt.wantPoll
			}
			got := guardResourcePolicyConfigured()
			if got.MaxTreeBytes != wantMemory || got.PollInterval != wantPoll {
				t.Fatalf("policy=%+v, want memory=%d poll=%s", got, wantMemory, wantPoll)
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
			if tt.wantReason == "SYSTEM_COMMIT_HEADROOM" && d.ThresholdBytes != tt.policy.MinSystemHeadroom {
				t.Fatalf("SYSTEM_COMMIT_HEADROOM threshold=%d want %d", d.ThresholdBytes, tt.policy.MinSystemHeadroom)
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
	t.Run("partial reap reports only protected survivors with typed recovery", func(t *testing.T) {
		alivePID := os.Getpid()
		event := guardChildWaitEvent{Resource: &guardResourceDecision{OwnedPIDs: []int{-51908, alivePID, -64852}}}
		err := guardWriteResourceReceipt(event, "trace", "codex", -51908)
		if err == nil {
			t.Fatal("expected protected survivor error")
		}
		got := err.Error()
		for _, want := range []string{"CHILD_RESOURCE_CONTAINMENT_SURVIVORS", fmt.Sprintf("owned processes still alive: [%d]", alivePID), "elevated Windows terminal", "service owner", "ownership is not verified"} {
			if !strings.Contains(got, want) {
				t.Fatalf("error %q missing %q", got, want)
			}
		}
		for _, reaped := range []string{"-51908", "-64852"} {
			if strings.Contains(got, reaped) {
				t.Fatalf("error %q reports reaped PID %s", got, reaped)
			}
		}
	})
	t.Run("collector failure without processes keeps root", func(t *testing.T) {
		event := guardResourceMonitorFailure(-7, procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS}, "MONITOR_ERROR", "malformed\noutput")
		if event.Resource == nil || event.Resource.Offender.PID != -7 || !slices.Equal(event.Resource.OwnedPIDs, []int{-7}) || event.Resource.Detail != "malformed output" || event.Reason != "MONITOR_ERROR: malformed output" {
			t.Fatalf("event=%+v", event)
		}
	})
}

func TestGuardResourceMonitorFailureReceiptBindsRunningActivation(t *testing.T) {
	oldIdentity := guardResourceBuildIdentity
	oldActivation := guardResourceActivationID
	guardResourceBuildIdentity = func() binaryIdentity {
		return binaryIdentity{Commit: "0123456789abcdef", ModuleVersion: "v1.2.3", Dirty: true, Stamped: true}
	}
	guardResourceActivationID = "activation-42"
	t.Cleanup(func() {
		guardResourceBuildIdentity = oldIdentity
		guardResourceActivationID = oldActivation
	})

	event := guardResourceMonitorFailure(42, procguard.MemorySnapshot{
		Metric:  procguard.MemoryMetricRSS,
		RootPID: 42,
		Processes: []procguard.MemoryProcess{{
			PID:  42,
			Name: "codex",
		}},
	}, "CHILD_RESOURCE_MONITOR_ERROR", `collector=C:\private\ps.exe host=db1.corp token=hunter2`)
	if event.Resource == nil {
		t.Fatal("monitor failure has no resource decision")
	}
	receipt := newGuardResourceReceipt("trace", "codex", 42, *event.Resource)
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.BuildCommit != "0123456789abcdef" || receipt.BuildModule != "v1.2.3" || !receipt.BuildDirty || receipt.ActivationID != "activation-42" {
		t.Fatalf("activation identity missing: %+v", receipt)
	}
	for _, leaked := range []string{`C:\private`, "db1.corp", "hunter2"} {
		if strings.Contains(string(data), leaked) {
			t.Fatalf("receipt identity leaked %q: %s", leaked, data)
		}
	}
	for _, forbiddenKey := range []string{"hostname", "executable", "executable_path"} {
		if strings.Contains(string(data), `"`+forbiddenKey+`"`) {
			t.Fatalf("receipt persisted forbidden key %q: %s", forbiddenKey, data)
		}
	}
}

func TestGuardChildResourceDarwinContainmentDeterminism(t *testing.T) {
	policy := guardResourcePolicy{
		Metric:       procguard.MemoryMetricRSS,
		MaxTreeBytes: 50 << 20,
	}
	snapshot := procguard.MemorySnapshot{
		Metric:    procguard.MemoryMetricRSS,
		TreeBytes: 55 << 20,
		Processes: []procguard.MemoryProcess{
			{PID: 105, Name: "child", Bytes: 30 << 20},
			{PID: 101, Name: "root", Bytes: 15 << 20},
			{PID: 109, Name: "worker", Bytes: 10 << 20},
		},
	}

	d1 := decideGuardResource(policy, snapshot)
	d2 := decideGuardResource(policy, snapshot)

	if d1.Reason != d2.Reason || d1.Stop != d2.Stop || d1.Offender.PID != d2.Offender.PID {
		t.Fatalf("decisions differ: %+v vs %+v", d1, d2)
	}
	if !slices.Equal(d1.OwnedPIDs, d2.OwnedPIDs) {
		t.Fatalf("owned PIDs order non-deterministic: %v vs %v", d1.OwnedPIDs, d2.OwnedPIDs)
	}

	rawDetail := "collector=/tmp/private/ps.txt token=secret-key pid=101"
	s1 := scrubGuardResourceDetail(rawDetail)
	s2 := scrubGuardResourceDetail(rawDetail)
	if s1 != s2 {
		t.Fatalf("scrubbing non-deterministic: %q vs %q", s1, s2)
	}

	// Concurrent race and determinism test across parallel workers
	const workers = 20
	errCh := make(chan error, workers)
	for i := 0; i < workers; i++ {
		go func(id int) {
			for j := 0; j < 50; j++ {
				d := decideGuardResource(policy, snapshot)
				if !d.Stop || d.Reason != "CHILD_TREE_RSS_LIMIT" || d.Offender.PID != 105 {
					errCh <- fmt.Errorf("unexpected decision in worker %d: %+v", id, d)
					return
				}
				if !slices.Equal(d.OwnedPIDs, []int{101, 105, 109}) {
					errCh <- fmt.Errorf("unexpected PIDs in worker %d: %v", id, d.OwnedPIDs)
					return
				}
			}
			errCh <- nil
		}(i)
	}

	for i := 0; i < workers; i++ {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
}

func TestFoldGuardResourceReceiptsByWeek(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "child-resource.jsonl")

	empty, err := foldChildResourceReceiptsByWeek(filepath.Join(dir, "nonexistent.jsonl"))
	if err != nil {
		t.Fatalf("missing ledger error: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("empty ledger count = %v, want 0", empty)
	}

	receipts := []guardResourceReceipt{
		{Schema: "fak.guard.child-resource.v1", At: "2026-08-10T12:00:00Z", TraceID: "t1", Action: "reap_tree"},
		{Schema: "fak.guard.child-resource.v1", At: "2026-08-11T14:00:00Z", TraceID: "t2", Action: "reap_tree"},
		{Schema: "fak.guard.child-resource.v1", At: "2026-08-25T09:00:00Z", TraceID: "t3", Action: "reap_tree"},
	}
	for _, r := range receipts {
		if err := appendGuardResourceReceipt(path, r); err != nil {
			t.Fatalf("append receipt: %v", err)
		}
	}

	counts, err := foldChildResourceReceiptsByWeek(path)
	if err != nil {
		t.Fatalf("fold receipts by week: %v", err)
	}

	if counts["2026-W33"] != 2 {
		t.Errorf("counts[2026-W33] = %d, want 2", counts["2026-W33"])
	}
	if counts["2026-W35"] != 1 {
		t.Errorf("counts[2026-W35] = %d, want 1", counts["2026-W35"])
	}
}

func TestGuardChildResourceHeadroomDebounceAndRecovery(t *testing.T) {
	t.Run("debounce grace period delays headroom intervention and invokes yield", func(t *testing.T) {
		stop := make(chan struct{})
		defer close(stop)

		var yieldCalled bool
		var yieldPIDs []int
		oldYield := guardYieldMemory
		guardYieldMemory = func(pids ...int) {
			yieldCalled = true
			yieldPIDs = append(yieldPIDs, pids...)
		}
		t.Cleanup(func() { guardYieldMemory = oldYield })

		policy := guardResourcePolicy{
			PollInterval:      10 * time.Millisecond,
			Metric:            procguard.MemoryMetricCommit,
			MaxTreeBytes:      1000,
			MinSystemHeadroom: 100,
			HeadroomDebounce:  60 * time.Millisecond,
			Stop:              stop,
		}
		snapshot := procguard.MemorySnapshot{
			Metric:      procguard.MemoryMetricCommit,
			RootPID:     42,
			TreeBytes:   10,
			SystemBytes: 950,
			SystemLimit: 1000,
			Processes:   []procguard.MemoryProcess{{PID: 42, Bytes: 10}},
		}

		started := time.Now()
		ch := startGuardChildResourceMonitorWithCollector(42, "trace-debounce", "test-agent", policy, func(pid int) (procguard.MemorySnapshot, bool, string) {
			return snapshot, true, ""
		})

		// At 25ms (halfway into 60ms debounce window), no event should be emitted yet
		time.Sleep(25 * time.Millisecond)
		select {
		case ev := <-ch:
			t.Fatalf("event emitted prematurely during debounce grace window: %+v", ev)
		default:
		}

		if !yieldCalled {
			t.Fatal("expected procguard.YieldMemory to be invoked on headroom refusal")
		}
		if len(yieldPIDs) == 0 || yieldPIDs[0] != 42 {
			t.Fatalf("expected YieldMemory to receive rootPID 42, got %v", yieldPIDs)
		}

		// Wait for event after debounce window expires
		select {
		case ev := <-ch:
			elapsed := time.Since(started)
			if elapsed < 50*time.Millisecond {
				t.Fatalf("event emitted too early (%v < 50ms)", elapsed)
			}
			if ev.Kind != guardChildResourceLimit || ev.Resource == nil {
				t.Fatalf("unexpected event: %+v", ev)
			}
			if ev.Resource.Reason != "SYSTEM_COMMIT_HEADROOM" {
				t.Fatalf("reason = %q, want SYSTEM_COMMIT_HEADROOM", ev.Resource.Reason)
			}
			if ev.Resource.Offender.PID != 0 || ev.Resource.Offender.Name != "host" {
				t.Fatalf("offender = %+v, want PID 0 (host)", ev.Resource.Offender)
			}
			if !strings.Contains(ev.Reason, "offender_pid=0") {
				t.Fatalf("reason missing offender_pid=0: %s", ev.Reason)
			}
		case <-time.After(500 * time.Millisecond):
			t.Fatal("timed out waiting for debounced headroom event")
		}
	})

	t.Run("headroom recovery resets grace timer without interrupting child", func(t *testing.T) {
		stop := make(chan struct{})
		defer close(stop)

		policy := guardResourcePolicy{
			PollInterval:      10 * time.Millisecond,
			Metric:            procguard.MemoryMetricCommit,
			MaxTreeBytes:      1000,
			MinSystemHeadroom: 100,
			HeadroomDebounce:  60 * time.Millisecond,
			Stop:              stop,
		}

		tickCount := 0
		ch := startGuardChildResourceMonitorWithCollector(42, "trace-recovery", "test-agent", policy, func(pid int) (procguard.MemorySnapshot, bool, string) {
			tickCount++
			systemBytes := uint64(500) // healthy (500 headroom >= 100)
			if tickCount <= 3 {
				// ticks 1, 2, 3: deficit (50 headroom < 100)
				systemBytes = 950
			}
			return procguard.MemorySnapshot{
				Metric:      procguard.MemoryMetricCommit,
				RootPID:     42,
				TreeBytes:   10,
				SystemBytes: systemBytes,
				SystemLimit: 1000,
				Processes:   []procguard.MemoryProcess{{PID: 42, Bytes: 10}},
			}, true, ""
		})

		// Wait 120ms (well beyond 60ms debounce window)
		time.Sleep(120 * time.Millisecond)

		select {
		case ev := <-ch:
			t.Fatalf("child was interrupted despite recovering: %+v", ev)
		default:
			// Child was not interrupted!
		}
	})

	t.Run("child tree limit does not debounce", func(t *testing.T) {
		stop := make(chan struct{})
		defer close(stop)

		policy := guardResourcePolicy{
			PollInterval:      10 * time.Millisecond,
			Metric:            procguard.MemoryMetricCommit,
			MaxTreeBytes:      100,
			MinSystemHeadroom: 100,
			HeadroomDebounce:  500 * time.Millisecond,
			Stop:              stop,
		}

		started := time.Now()
		ch := startGuardChildResourceMonitorWithCollector(42, "trace-tree-limit", "test-agent", policy, func(pid int) (procguard.MemorySnapshot, bool, string) {
			return procguard.MemorySnapshot{
				Metric:      procguard.MemoryMetricCommit,
				RootPID:     42,
				TreeBytes:   150, // exceeds MaxTreeBytes 100
				SystemBytes: 500,
				SystemLimit: 1000,
				Processes:   []procguard.MemoryProcess{{PID: 42, Bytes: 150}},
			}, true, ""
		})

		select {
		case ev := <-ch:
			elapsed := time.Since(started)
			if elapsed > 100*time.Millisecond {
				t.Fatalf("tree limit was delayed (%v > 100ms)", elapsed)
			}
			if ev.Resource.Reason != "CHILD_TREE_COMMIT_LIMIT" {
				t.Fatalf("reason = %q, want CHILD_TREE_COMMIT_LIMIT", ev.Resource.Reason)
			}
			if ev.Resource.Offender.PID != 42 {
				t.Fatalf("offender PID = %d, want 42", ev.Resource.Offender.PID)
			}
		case <-time.After(300 * time.Millisecond):
			t.Fatal("timed out waiting for immediate tree limit")
		}
	})

	t.Run("guardSuspendProcess called on headroom trip and guardResumeProcess on recovery", func(t *testing.T) {
		stop := make(chan struct{})
		defer close(stop)

		var suspendedPIDs []int
		var resumedPIDs []int
		oldSuspend := guardSuspendProcess
		oldResume := guardResumeProcess
		guardSuspendProcess = func(pid int) error {
			suspendedPIDs = append(suspendedPIDs, pid)
			return nil
		}
		guardResumeProcess = func(pid int) error {
			resumedPIDs = append(resumedPIDs, pid)
			return nil
		}
		t.Cleanup(func() {
			guardSuspendProcess = oldSuspend
			guardResumeProcess = oldResume
		})

		policy := guardResourcePolicy{
			PollInterval:      10 * time.Millisecond,
			Metric:            procguard.MemoryMetricCommit,
			MaxTreeBytes:      1000,
			MinSystemHeadroom: 100,
			HeadroomDebounce:  60 * time.Millisecond,
			Stop:              stop,
		}

		tickCount := 0
		ch := startGuardChildResourceMonitorWithCollector(42, "trace-suspend-resume", "test-agent", policy, func(pid int) (procguard.MemorySnapshot, bool, string) {
			tickCount++
			systemBytes := uint64(500) // healthy
			if tickCount <= 2 {
				// ticks 1, 2: deficit
				systemBytes = 950
			}
			return procguard.MemorySnapshot{
				Metric:      procguard.MemoryMetricCommit,
				RootPID:     42,
				TreeBytes:   10,
				SystemBytes: systemBytes,
				SystemLimit: 1000,
				Processes:   []procguard.MemoryProcess{{PID: 42, Bytes: 10}},
			}, true, ""
		})

		// Wait for both suspension and resumption to occur
		deadline := time.Now().Add(500 * time.Millisecond)
		for time.Now().Before(deadline) {
			if len(suspendedPIDs) > 0 && len(resumedPIDs) > 0 {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}

		if len(suspendedPIDs) == 0 || suspendedPIDs[0] != 42 {
			t.Fatalf("expected guardSuspendProcess to be called with PID 42, got %v", suspendedPIDs)
		}
		if len(resumedPIDs) == 0 || resumedPIDs[0] != 42 {
			t.Fatalf("expected guardResumeProcess to be called with PID 42, got %v", resumedPIDs)
		}

		select {
		case ev := <-ch:
			t.Fatalf("unexpected event on recovery: %+v", ev)
		default:
		}
	})

	t.Run("deferred resumption on monitor exit when suspended", func(t *testing.T) {
		stop := make(chan struct{})

		var resumedPIDs []int
		oldResume := guardResumeProcess
		guardResumeProcess = func(pid int) error {
			resumedPIDs = append(resumedPIDs, pid)
			return nil
		}
		t.Cleanup(func() { guardResumeProcess = oldResume })

		policy := guardResourcePolicy{
			PollInterval:      10 * time.Millisecond,
			Metric:            procguard.MemoryMetricCommit,
			MaxTreeBytes:      1000,
			MinSystemHeadroom: 100,
			HeadroomDebounce:  200 * time.Millisecond,
			Stop:              stop,
		}

		_ = startGuardChildResourceMonitorWithCollector(42, "trace-defer-resume", "test-agent", policy, func(pid int) (procguard.MemorySnapshot, bool, string) {
			return procguard.MemorySnapshot{
				Metric:      procguard.MemoryMetricCommit,
				RootPID:     42,
				TreeBytes:   10,
				SystemBytes: 950,
				SystemLimit: 1000,
				Processes:   []procguard.MemoryProcess{{PID: 42, Bytes: 10}},
			}, true, ""
		})

		time.Sleep(25 * time.Millisecond) // wait for at least one tick which suspends
		close(stop)                       // close stop channel to exit monitor
		time.Sleep(25 * time.Millisecond) // wait for defer to run

		if len(resumedPIDs) == 0 || resumedPIDs[0] != 42 {
			t.Fatalf("expected deferred guardResumeProcess on monitor exit, got %v", resumedPIDs)
		}
	})
}

func TestGuardHeadroomDebounceConfiguration(t *testing.T) {
	oldConfig := guardResourceConfigured
	t.Cleanup(func() { setGuardResourceConfig(oldConfig) })

	t.Run("default is 15s", func(t *testing.T) {
		setGuardResourceConfig(guardResourceConfig{})
		p := guardResourcePolicyConfigured()
		if p.HeadroomDebounce != 15*time.Second {
			t.Fatalf("HeadroomDebounce = %v, want %v", p.HeadroomDebounce, 15*time.Second)
		}
	})

	t.Run("flag configures policy.HeadroomDebounce", func(t *testing.T) {
		flagVal := 45 * time.Second
		resolved := resolveGuardHeadroomDebounce(flagVal, "")
		setGuardResourceConfig(guardResourceConfig{HeadroomDebounce: resolved})
		p := guardResourcePolicyConfigured()
		if p.HeadroomDebounce != 45*time.Second {
			t.Fatalf("HeadroomDebounce = %v, want %v", p.HeadroomDebounce, 45*time.Second)
		}
	})

	t.Run("env var configures policy.HeadroomDebounce when flag <= 0", func(t *testing.T) {
		resolved := resolveGuardHeadroomDebounce(0, "30s")
		setGuardResourceConfig(guardResourceConfig{HeadroomDebounce: resolved})
		p := guardResourcePolicyConfigured()
		if p.HeadroomDebounce != 30*time.Second {
			t.Fatalf("HeadroomDebounce = %v, want %v", p.HeadroomDebounce, 30*time.Second)
		}
	})

	t.Run("flag takes precedence over env var", func(t *testing.T) {
		flagVal := 25 * time.Second
		resolved := resolveGuardHeadroomDebounce(flagVal, "50s")
		setGuardResourceConfig(guardResourceConfig{HeadroomDebounce: resolved})
		p := guardResourcePolicyConfigured()
		if p.HeadroomDebounce != 25*time.Second {
			t.Fatalf("HeadroomDebounce = %v, want %v", p.HeadroomDebounce, 25*time.Second)
		}
	})
}
