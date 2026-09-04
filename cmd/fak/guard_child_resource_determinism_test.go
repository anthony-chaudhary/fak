package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func TestGuardChildResourceDarwin_Determinism(t *testing.T) {
	t.Run("byte_identical_decision_and_receipt", func(t *testing.T) {
		policy := guardResourcePolicy{
			Metric:       procguard.MemoryMetricRSS,
			MaxTreeBytes: 50 << 20,
			PollInterval: 100 * time.Millisecond,
		}
		snapshot := procguard.MemorySnapshot{
			Metric:    procguard.MemoryMetricRSS,
			RootPID:   1000,
			TreeBytes: 75 << 20,
			Processes: []procguard.MemoryProcess{
				{PID: 1002, PPID: 1000, Name: "claude-worker", Bytes: 40 << 20, CommandLine: "/app/worker --port 8080"},
				{PID: 1000, PPID: 1, Name: "claude", Bytes: 20 << 20, CommandLine: "/app/claude --agent"},
				{PID: 1003, PPID: 1002, Name: "node", Bytes: 15 << 20, CommandLine: "node index.js"},
			},
		}

		// Two identical runs must produce byte-identical decisions.
		d1 := decideGuardResource(policy, snapshot)
		d2 := decideGuardResource(policy, snapshot)

		if !d1.Stop || !d2.Stop {
			t.Fatalf("expected both runs to stop, got d1.Stop=%v d2.Stop=%v", d1.Stop, d2.Stop)
		}
		if d1.Reason != "CHILD_TREE_RSS_LIMIT" || d2.Reason != "CHILD_TREE_RSS_LIMIT" {
			t.Fatalf("expected CHILD_TREE_RSS_LIMIT, got d1=%q d2=%q", d1.Reason, d2.Reason)
		}
		if d1.Metric != procguard.MemoryMetricRSS || d2.Metric != procguard.MemoryMetricRSS {
			t.Fatalf("expected RSS metric, got d1=%s d2=%s", d1.Metric, d2.Metric)
		}
		if d1.Offender.PID != 1002 || d2.Offender.PID != 1002 {
			t.Fatalf("offender PID mismatch: d1=%d d2=%d, want 1002", d1.Offender.PID, d2.Offender.PID)
		}
		if !slices.Equal(d1.OwnedPIDs, []int{1000, 1002, 1003}) {
			t.Fatalf("d1.OwnedPIDs not sorted: %v", d1.OwnedPIDs)
		}
		if !slices.Equal(d1.OwnedPIDs, d2.OwnedPIDs) {
			t.Fatalf("owned PIDs differ: d1=%v d2=%v", d1.OwnedPIDs, d2.OwnedPIDs)
		}

		d1JSON, err := json.Marshal(d1)
		if err != nil {
			t.Fatalf("marshal d1: %v", err)
		}
		d2JSON, err := json.Marshal(d2)
		if err != nil {
			t.Fatalf("marshal d2: %v", err)
		}
		if !bytes.Equal(d1JSON, d2JSON) {
			t.Fatalf("decisions not byte-identical:\nd1=%s\nd2=%s", d1JSON, d2JSON)
		}
		if r1, r2 := guardResourceReason(d1), guardResourceReason(d2); r1 != r2 {
			t.Fatalf("guardResourceReason differ: %q vs %q", r1, r2)
		}

		// Two identical runs must produce byte-identical receipt structure.
		const traceID = "trace-darwin-rss-determinism"
		const agent = "claude"
		r1 := newGuardResourceReceipt(traceID, agent, snapshot.RootPID, d1)
		r2 := newGuardResourceReceipt(traceID, agent, snapshot.RootPID, d2)

		// Normalize At timestamp across runs to compare structural byte identity.
		r2.At = r1.At

		r1JSON, err := json.Marshal(r1)
		if err != nil {
			t.Fatalf("marshal r1: %v", err)
		}
		r2JSON, err := json.Marshal(r2)
		if err != nil {
			t.Fatalf("marshal r2: %v", err)
		}
		if !bytes.Equal(r1JSON, r2JSON) {
			t.Fatalf("receipts not byte-identical:\nr1=%s\nr2=%s", r1JSON, r2JSON)
		}

		// Verify receipt content adheres to Darwin RSS containment schema.
		if r1.Schema != "fak.guard.child-resource.v1" {
			t.Fatalf("schema = %q, want fak.guard.child-resource.v1", r1.Schema)
		}
		if r1.MemoryMetric != "rss" {
			t.Fatalf("memory_metric = %q, want rss", r1.MemoryMetric)
		}
		if r1.TreeRSSBytes == nil || *r1.TreeRSSBytes != 75<<20 {
			t.Fatalf("tree_rss_bytes = %v, want %d", r1.TreeRSSBytes, 75<<20)
		}
		if r1.TreeCommitBytes != nil {
			t.Fatalf("tree_commit_bytes unexpectedly set for Darwin RSS: %v", r1.TreeCommitBytes)
		}
		if r1.OffenderPID != 1002 || r1.OffenderName != "claude-worker" {
			t.Fatalf("offender info = %d (%s), want 1002 (claude-worker)", r1.OffenderPID, r1.OffenderName)
		}
		if r1.OffenderCommand != "" {
			t.Fatalf("offender command = %q, want empty", r1.OffenderCommand)
		}
		if r1.Action != "reap_tree" || r1.DescendantsSurvive {
			t.Fatalf("action=%q descendantsSurvive=%v, want reap_tree/false", r1.Action, r1.DescendantsSurvive)
		}
		if r1.Reason != "CHILD_TREE_RSS_LIMIT" {
			t.Fatalf("reason = %q, want CHILD_TREE_RSS_LIMIT", r1.Reason)
		}

		// Verify file persistence yields byte-identical bytes.
		dir := t.TempDir()
		p1 := filepath.Join(dir, "receipt1.jsonl")
		p2 := filepath.Join(dir, "receipt2.jsonl")
		if err := appendGuardResourceReceipt(p1, r1); err != nil {
			t.Fatalf("write p1: %v", err)
		}
		if err := appendGuardResourceReceipt(p2, r2); err != nil {
			t.Fatalf("write p2: %v", err)
		}
		data1, err := os.ReadFile(p1)
		if err != nil {
			t.Fatalf("read p1: %v", err)
		}
		data2, err := os.ReadFile(p2)
		if err != nil {
			t.Fatalf("read p2: %v", err)
		}
		if !bytes.Equal(data1, data2) {
			t.Fatalf("persisted receipts not byte-identical:\n1=%s\n2=%s", data1, data2)
		}
	})

	t.Run("equal_memory_tie_break_determinism", func(t *testing.T) {
		policy := guardResourcePolicy{
			Metric:       procguard.MemoryMetricRSS,
			MaxTreeBytes: 40 << 20,
		}
		// Two processes with identical RSS bytes; selection must deterministically pick lower PID.
		snapshot := procguard.MemorySnapshot{
			Metric:    procguard.MemoryMetricRSS,
			RootPID:   2000,
			TreeBytes: 50 << 20,
			Processes: []procguard.MemoryProcess{
				{PID: 2005, PPID: 2000, Name: "worker-b", Bytes: 25 << 20},
				{PID: 2002, PPID: 2000, Name: "worker-a", Bytes: 25 << 20},
			},
		}

		d1 := decideGuardResource(policy, snapshot)
		d2 := decideGuardResource(policy, snapshot)

		if d1.Offender.PID != 2002 || d2.Offender.PID != 2002 {
			t.Fatalf("offender PID tie-break failed: d1=%d d2=%d, want 2002", d1.Offender.PID, d2.Offender.PID)
		}
		d1JSON, _ := json.Marshal(d1)
		d2JSON, _ := json.Marshal(d2)
		if !bytes.Equal(d1JSON, d2JSON) {
			t.Fatalf("tie-break decisions not byte-identical:\nd1=%s\nd2=%s", d1JSON, d2JSON)
		}

		// Ensure input snapshot was not mutated in place.
		if snapshot.Processes[0].PID != 2005 || snapshot.Processes[1].PID != 2002 {
			t.Fatalf("input snapshot processes mutated in place: %+v", snapshot.Processes)
		}
	})

	t.Run("detail_scrubbing_determinism", func(t *testing.T) {
		raw := "collector=/tmp/private/token.txt secret=hunter2 host=api.corp ip=10.0.1.5 bearer token123 " +
			strings.Repeat("long-diagnostic ", 50)
		s1 := scrubGuardResourceDetail(raw)
		s2 := scrubGuardResourceDetail(raw)
		if s1 != s2 {
			t.Fatalf("detail scrubbing non-deterministic: %q vs %q", s1, s2)
		}
		for _, forbidden := range []string{"hunter2", "token123", "api.corp", "10.0.1.5"} {
			if strings.Contains(s1, forbidden) {
				t.Fatalf("scrubbed detail leaked %q: %s", forbidden, s1)
			}
		}
	})

	t.Run("darwin_runtime_defaults", func(t *testing.T) {
		if runtime.GOOS != "darwin" {
			t.Skip("darwin-only runtime defaults check")
		}
		old := guardResourceConfigured
		t.Cleanup(func() { setGuardResourceConfig(old) })
		setGuardResourceConfig(guardResourceConfig{})

		p := guardResourcePolicyConfigured()
		if p.Metric != procguard.MemoryMetricRSS {
			t.Fatalf("darwin default metric=%s, want rss", p.Metric)
		}
		if p.MinSystemHeadroom != 0 {
			t.Fatalf("darwin min system headroom=%d, want 0", p.MinSystemHeadroom)
		}
		if p.MaxTreeBytes < guardTreeRSSMinimum || p.MaxTreeBytes > guardTreeCommitDefault {
			t.Fatalf("darwin default max tree bytes out of bounds: %d", p.MaxTreeBytes)
		}
	})

	t.Run("concurrent_race_witness", func(t *testing.T) {
		policy := guardResourcePolicy{
			Metric:       procguard.MemoryMetricRSS,
			MaxTreeBytes: 50 << 20,
			PollInterval: 100 * time.Millisecond,
		}
		snapshot := procguard.MemorySnapshot{
			Metric:    procguard.MemoryMetricRSS,
			RootPID:   1000,
			TreeBytes: 75 << 20,
			Processes: []procguard.MemoryProcess{
				{PID: 1005, PPID: 1000, Name: "child-b", Bytes: 30 << 20},
				{PID: 1001, PPID: 1000, Name: "child-a", Bytes: 30 << 20},
				{PID: 1000, PPID: 1, Name: "root", Bytes: 15 << 20},
			},
		}

		baselineDecision := decideGuardResource(policy, snapshot)
		baselineDJSON, err := json.Marshal(baselineDecision)
		if err != nil {
			t.Fatal(err)
		}
		const traceID = "race-witness-trace"
		const agent = "codex"
		baselineReceipt := newGuardResourceReceipt(traceID, agent, snapshot.RootPID, baselineDecision)
		baselineRJSON, err := json.Marshal(baselineReceipt)
		if err != nil {
			t.Fatal(err)
		}

		const workers = 32
		const iterations = 100
		var wg sync.WaitGroup
		errCh := make(chan error, workers*iterations)

		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for iter := 0; iter < iterations; iter++ {
					// Concurrently call decideGuardResource on the shared snapshot.
					d := decideGuardResource(policy, snapshot)
					dJSON, err := json.Marshal(d)
					if err != nil {
						errCh <- fmt.Errorf("worker %d iter %d marshal decision: %w", workerID, iter, err)
						return
					}
					if !bytes.Equal(dJSON, baselineDJSON) {
						errCh <- fmt.Errorf("worker %d iter %d decision mismatch:\ngot=%s\nwant=%s", workerID, iter, dJSON, baselineDJSON)
						return
					}

					// Concurrently construct and marshal receipt.
					r := newGuardResourceReceipt(traceID, agent, snapshot.RootPID, d)
					r.At = baselineReceipt.At
					rJSON, err := json.Marshal(r)
					if err != nil {
						errCh <- fmt.Errorf("worker %d iter %d marshal receipt: %w", workerID, iter, err)
						return
					}
					if !bytes.Equal(rJSON, baselineRJSON) {
						errCh <- fmt.Errorf("worker %d iter %d receipt mismatch:\ngot=%s\nwant=%s", workerID, iter, rJSON, baselineRJSON)
						return
					}

					// Concurrently scrub diagnostic details.
					detail := scrubGuardResourceDetail(fmt.Sprintf("worker=%d iter=%d /vault/key token=secret123", workerID, iter))
					if strings.Contains(detail, "secret123") || strings.Contains(detail, "/vault/key") {
						errCh <- fmt.Errorf("worker %d iter %d leaked detail: %s", workerID, iter, detail)
						return
					}
				}
			}(w)
		}

		wg.Wait()
		close(errCh)
		for err := range errCh {
			t.Fatal(err)
		}
	})

	t.Run("monitor_collector_accounting_determinism", func(t *testing.T) {
		policy := guardResourcePolicy{
			Metric:       procguard.MemoryMetricRSS,
			MaxTreeBytes: 50 << 20,
			PollInterval: 10 * time.Millisecond,
		}
		snapshot := procguard.MemorySnapshot{
			Metric:    procguard.MemoryMetricRSS,
			RootPID:   4000,
			TreeBytes: 60 << 20,
			Processes: []procguard.MemoryProcess{
				{PID: 4001, PPID: 4000, Name: "worker", Bytes: 60 << 20},
			},
		}

		runMonitor := func() guardChildWaitEvent {
			stop := make(chan struct{})
			defer close(stop)
			p := policy
			p.Stop = stop
			ch := startGuardChildResourceMonitorWithCollector(4000, "trace-m", "codex", p, func(pid int) (procguard.MemorySnapshot, bool, string) {
				return snapshot, true, ""
			})
			select {
			case ev := <-ch:
				return ev
			case <-time.After(2 * time.Second):
				t.Fatal("monitor timed out")
				return guardChildWaitEvent{}
			}
		}

		ev1 := runMonitor()
		ev2 := runMonitor()

		if ev1.Kind != guardChildResourceLimit || ev2.Kind != guardChildResourceLimit {
			t.Fatalf("unexpected event kinds: ev1=%v ev2=%v", ev1.Kind, ev2.Kind)
		}
		if ev1.Reason != ev2.Reason {
			t.Fatalf("reasons differ: %q vs %q", ev1.Reason, ev2.Reason)
		}
		ev1JSON, _ := json.Marshal(ev1.Resource)
		ev2JSON, _ := json.Marshal(ev2.Resource)
		if !bytes.Equal(ev1JSON, ev2JSON) {
			t.Fatalf("monitor decisions not byte-identical:\n1=%s\n2=%s", ev1JSON, ev2JSON)
		}
	})
}
