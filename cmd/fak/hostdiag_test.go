package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostdiag"
)

func TestHostdiagHistoricalFixtureIsRetainedUnresolved(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	events := []hostdiag.ResourceEvent{{TimeMS: 1787710011683, Source: "Windows Error Reporting", EventID: 1001, RecordID: "111212", Name: "RADAR_PRE_LEAK_64", ReportID: "471a1974-e0e6-426d-86aa-4ed6533cde06", App: "fak.exe"}}
	data, _ := json.Marshal(events)
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"correlate", "--fixture", fixture, "--ledger", ledger}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var got hostdiag.Correlation
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &got); err != nil || got.Status != "historical_unresolved" || !got.Observational || got.ReportID != events[0].ReportID {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	if raw, err := os.ReadFile(ledger); err != nil || !strings.Contains(string(raw), hostdiag.CorrelationSchema) {
		t.Fatalf("ledger=%q err=%v", raw, err)
	}
}

func TestHostdiagFixtureIdentifiesSpanningCensus(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	event := hostdiag.ResourceEvent{TimeMS: 2000, Source: "Windows Error Reporting", EventID: 1001, RecordID: "2", Name: "RADAR_PRE_LEAK_64", App: "fak.exe"}
	data, _ := json.Marshal([]hostdiag.ResourceEvent{event})
	_ = os.WriteFile(fixture, data, 0o600)
	sample := hostdiag.NewProcessSample(timeUnixMilli(3000), 42, timeUnixMilli(1000), `C:\bin\fak.exe`, "sha", "rev", "guard", "g1", 10, 20, 0, 0)
	if err := appendHostdiagRow(ledger, sample, 4096); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"correlate", "--fixture", fixture, "--ledger", ledger, "--max-bytes", "4096"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var got hostdiag.Correlation
	_ = json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &got)
	if got.Status != "identified" || len(got.Candidates) != 1 || got.Candidates[0].CommandClass != "guard" {
		t.Fatalf("%+v", got)
	}
}

func TestAppendHostdiagRowBoundsCompleteLines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hostdiag.jsonl")
	for i := 1; i <= 40; i++ {
		row := hostdiag.Correlation{Schema: hostdiag.CorrelationSchema, CorrelationID: strings.Repeat("x", 60) + string(rune(i)), TimeMS: int64(i), Status: "historical_unresolved", Observational: true}
		if err := appendHostdiagRow(path, row, 700); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) > 700 || raw[len(raw)-1] != '\n' {
		t.Fatalf("bytes=%d err=%v", len(raw), err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("malformed retained row: %v", err)
		}
	}
}

func TestAppendHostdiagRowConcurrentComplete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hostdiag.jsonl")
	const writers = 12
	start := make(chan struct{})
	errs := make(chan error, writers)
	var group sync.WaitGroup
	for i := range writers {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			errs <- appendHostdiagRow(path, hostdiag.Correlation{Schema: hostdiag.CorrelationSchema, CorrelationID: fmt.Sprintf("c-%d", i), TimeMS: int64(i + 1), Status: "historical_unresolved", Observational: true}, 4096)
		}()
	}
	close(start)
	group.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != writers {
		t.Fatalf("rows=%d want=%d", len(lines), writers)
	}
	for _, line := range lines {
		if !json.Valid([]byte(line)) {
			t.Fatalf("malformed row %q", line)
		}
	}
}

func timeUnixMilli(ms int64) time.Time { return time.UnixMilli(ms) }
