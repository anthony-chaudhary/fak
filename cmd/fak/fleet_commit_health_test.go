package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestWriteFleetCommitThroughputMetricsRendersTopLevelZeroHealth(t *testing.T) {
	w := newPromWriter()
	writeFleetCommitThroughputMetrics(w, fleetmetrics.CommitThroughput{Measured: true, Previous: 2}, 4)
	got := w.String()
	for _, want := range []string{
		"fak_fleet_commits_per_10m 0",
		"fak_fleet_commits_previous_10m 2",
		"fak_fleet_commit_throughput_healthy 0",
		`fak_fleet_commit_throughput_state{state="stalled"} 1`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("exposition missing %q:\n%s", want, got)
		}
	}
}

func TestWriteFleetCommitThroughputMetricsRendersPositiveHealth(t *testing.T) {
	w := newPromWriter()
	writeFleetCommitThroughputMetrics(w, fleetmetrics.CommitThroughput{Measured: true, Current: 3}, 4)
	got := w.String()
	for _, want := range []string{
		"fak_fleet_commits_per_10m 3",
		"fak_fleet_commit_throughput_healthy 1",
		`fak_fleet_commit_throughput_state{state="healthy"} 1`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("exposition missing %q:\n%s", want, got)
		}
	}
}

func TestFormatCommitThroughputIsScannable(t *testing.T) {
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	got := formatCommitThroughput(fleetmetrics.CommitThroughput{Measured: true, Current: 1, Previous: 0, LatestCommitAt: now.Add(-2 * time.Minute)}, 3, now)
	if got != "commits/10m=1 previous=0 state=healthy latest_age=2m0s" {
		t.Fatalf("got %q", got)
	}
}

func TestRunFleetCommitHealthReportsActiveZeroAsUnhealthy(t *testing.T) {
	oldNow := fleetCommitHealthNow
	oldMeasure := fleetCommitHealthMeasure
	t.Cleanup(func() { fleetCommitHealthNow, fleetCommitHealthMeasure = oldNow, oldMeasure })
	now := time.Now().UTC().Truncate(time.Second)
	fleetCommitHealthNow = func() time.Time { return now }
	fleetCommitHealthMeasure = func(string, time.Time) fleetmetrics.CommitThroughput {
		return fleetmetrics.CommitThroughput{Measured: true, Window: fleetmetrics.CommitWindow}
	}

	registry := filepath.Join(t.TempDir(), "sessions.json")
	reg := session.NewRegistry(session.NewFileStore(registry))
	if _, err := reg.Register("worker-1", "test-host", session.DefaultState("worker-1"), time.Hour, now); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runFleetCommitHealth(&out, &errb, []string{"--json", "--registry", registry})
	if code != 3 {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errb.String())
	}
	var got fleetCommitHealthReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ActiveWorkers != 1 || got.Health.Healthy || got.Throughput.Measured == false {
		t.Fatalf("report=%+v", got)
	}
}
