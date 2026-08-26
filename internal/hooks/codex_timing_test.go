package hooks_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

type timingAttachment struct {
	Type       string      `json:"type"`
	DurationMS json.Number `json:"durationMs"`
	Hook       string      `json:"hook"`
	SessionID  string      `json:"sessionId"`
	TurnID     string      `json:"turnId"`
	CallID     string      `json:"callId"`
}

type timingRecord struct {
	Type       string           `json:"type"`
	Attachment timingAttachment `json:"attachment"`
}

func TestRunCodexHookTimedPreservesVerdictBytes(t *testing.T) {
	want := []byte("{\"decision\":\"deny\",\"reason\":\"literal bytes\\nkept\"}\n")
	var verdict bytes.Buffer
	var timing bytes.Buffer

	verdictErr, timingErr := hooks.RunCodexHookTimed(&verdict, &timing, "pretool", hooks.CodexHookCorrelation{
		SessionID: " session-1 ", TurnID: "turn-1", CallID: "call-1",
	}, func(w io.Writer) error {
		_, err := w.Write(want)
		return err
	})
	if verdictErr != nil || timingErr != nil {
		t.Fatalf("errors = verdict %v, timing %v", verdictErr, timingErr)
	}
	if !bytes.Equal(verdict.Bytes(), want) {
		t.Fatalf("verdict bytes changed:\n got %q\nwant %q", verdict.Bytes(), want)
	}

	var got timingRecord
	decoder := json.NewDecoder(&timing)
	decoder.UseNumber()
	if err := decoder.Decode(&got); err != nil {
		t.Fatalf("decode timing: %v", err)
	}
	if got.Type != "attachment" || got.Attachment.Type != "hook_codex" || got.Attachment.Hook != "pretool" {
		t.Fatalf("timing envelope = %+v", got)
	}
	if got.Attachment.SessionID != "session-1" || got.Attachment.TurnID != "turn-1" || got.Attachment.CallID != "call-1" {
		t.Fatalf("correlation = %+v", got.Attachment)
	}
	if duration, err := got.Attachment.DurationMS.Int64(); err != nil || duration < 0 {
		t.Fatalf("durationMs = %q, err %v", got.Attachment.DurationMS, err)
	}
}

func TestCodexHookTimingFixturesMatchTrajectoryConsumer(t *testing.T) {
	tests := []struct {
		name     string
		eligible int
		observed int
		p50, p95 *int64
		max      *int64
		refusals int
	}{
		{name: "complete", eligible: 3, observed: 3, p50: int64p(20), p95: int64p(30), max: int64p(30)},
		{name: "partial", eligible: 3, observed: 2, p50: int64p(30), p95: int64p(30), max: int64p(30)},
		{name: "malformed", eligible: 3, observed: 0, refusals: 2},
		{name: "absent", eligible: 0, observed: 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := filepath.Join("testdata", "codex_hook_timing", test.name+".jsonl")
			durations, malformed := readCompatibleDurations(t, fixture)
			if len(durations) != test.observed || malformed != test.refusals {
				t.Fatalf("coverage = %d/%d with %d malformed, want %d/%d with %d malformed", len(durations), test.eligible, malformed, test.observed, test.eligible, test.refusals)
			}
			p50, p95, max := timingStats(durations)
			assertInt64Ptr(t, "p50", p50, test.p50)
			assertInt64Ptr(t, "p95", p95, test.p95)
			assertInt64Ptr(t, "max", max, test.max)

			root := t.TempDir()
			body, err := os.ReadFile(fixture)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(root, "session.jsonl")
			if err := os.WriteFile(path, body, 0o600); err != nil {
				t.Fatal(err)
			}
			result, err := trajectory.RunAudit(trajectory.AuditOptions{
				Sources: []trajectory.AuditSource{{Name: trajectory.AuditSourceClaude, Root: root, RootLabel: "fixture"}},
				Now:     time.Now().Add(time.Second),
			})
			if err != nil {
				t.Fatalf("trajectory consumer: %v", err)
			}
			assertInt64Ptr(t, "consumer p95", result.Summary.HookP95MS, test.p95)
		})
	}
}

func readCompatibleDurations(t *testing.T, path string) ([]int64, int) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var durations []int64
	malformed := 0
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var record timingRecord
		decoder := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		decoder.UseNumber()
		if err := decoder.Decode(&record); err != nil {
			malformed++
			continue
		}
		if record.Type != "attachment" || len(record.Attachment.Type) < len("hook_") || record.Attachment.Type[:len("hook_")] != "hook_" {
			continue
		}
		duration, err := record.Attachment.DurationMS.Int64()
		if err == nil && duration >= 0 {
			durations = append(durations, duration)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return durations, malformed
}

func timingStats(values []int64) (p50, p95, max *int64) {
	if len(values) == 0 {
		return nil, nil, nil
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	percentile := func(n int) *int64 {
		index := (n*(len(sorted)-1) + 50) / 100
		return int64p(sorted[index])
	}
	return percentile(50), percentile(95), int64p(sorted[len(sorted)-1])
}

func int64p(value int64) *int64 { return &value }

func assertInt64Ptr(t *testing.T, name string, got, want *int64) {
	t.Helper()
	if got == nil || want == nil {
		if got != nil || want != nil {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
		return
	}
	if *got != *want {
		t.Fatalf("%s = %d, want %d", name, *got, *want)
	}
}
