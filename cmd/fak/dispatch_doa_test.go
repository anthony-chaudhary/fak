package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchdoa"
)

// doaOutageLog is the verbatim shape of the 350 records burned 2026-07-28..08-03
// (#5868), copied from .dispatch-runs/resolve-2419-20260728-213124.log. Trimmed of the
// middle of the usage block only — the classifier reads the head, and the first two
// lines plus the absence of a launch banner are the whole signal.
const doaOutageLog = `# fak-spawn 20260728-213124 issue=2419 lane=gateway backend=claude argv0=fak.exe
flag provided but not defined: -compact-solvency-floor
usage: fak guard [flags] -- <agent command...>
  e.g. fak guard -- claude
       fak guard --provider openai -- codex

81 flags in this build. 'fak guard -h -all' lists every one grouped.
`

// doaHealthyLog is a real post-outage record's head
// (.dispatch-runs/resolve-1348-20260807-055116.log, from the clean 08-04+ window). Its
// witness carried the SAME `reason: unknown` as every outage run; the guard's
// `— kernel-adjudicated:` launch banner is what tells them apart.
const doaHealthyLog = `# fak-spawn 20260807-055116 issue=1348 lane=gateway backend=claude argv0=fak.exe
fak guard: fleetspine: self-discovery on (group 239.255.70.65:4765 as "desktop-bb3fmhp")
fak guard 0.43.0 — kernel-adjudicated: claude -p --permission-mode bypassPermissions --model claude-opus-5
  build      : b225bb1ca20f
  gateway    : http://127.0.0.1:56430   (in-process; torn down when the command exits)
`

// plantRunLog writes one worker log into runsDir and back-dates it so the scan sees a
// FINISHED run inside the window rather than a still-settling live one.
func plantRunLog(t *testing.T, runsDir, name, body string, age time.Duration) string {
	t.Helper()
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(runsDir, name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	when := time.Now().Add(-age)
	if err := os.Chtimes(p, when, when); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestScanDispatchDOAAlarmsOnOutageWindow is the FIRES direction over real bytes: a
// runs directory holding the 2026-07-28 outage shape must alarm, name the argv-drift
// cause, and report the denominator alongside the rate.
func TestScanDispatchDOAAlarmsOnOutageWindow(t *testing.T) {
	runs := t.TempDir()
	for i := 0; i < 30; i++ {
		plantRunLog(t, runs, fmt.Sprintf("resolve-24%02d-20260728-2131%02d.log", i, i), doaOutageLog, 3*time.Hour)
	}
	for i := 0; i < 4; i++ {
		plantRunLog(t, runs, fmt.Sprintf("resolve-99%02d-20260728-2200%02d.log", i, i), doaHealthyLog+strings.Repeat("turn output\n", 800), 3*time.Hour)
	}
	rep := scanDispatchDOA(runs, dispatchDOAWindow, dispatchDOASettle, time.Now())
	if rep.Runs != 34 || rep.DOA != 30 {
		t.Fatalf("scan = %d DOA of %d runs, want 30 of 34 (the real 2026-07-28 day-one figures)", rep.DOA, rep.Runs)
	}
	if rep.Status != dispatchdoa.StatusAlarm {
		t.Fatalf("status = %q, want %q — day one must alarm, not wait six days", rep.Status, dispatchdoa.StatusAlarm)
	}
	if rep.Causes[dispatchdoa.CauseFlagParse] != 30 {
		t.Fatalf("causes = %v, want flag_parse=30", rep.Causes)
	}
	line := dispatchDOALine(rep, dispatchDOAWindow)
	for _, want := range []string{"ALARM", "30 of 34", "DIED ON ARRIVAL", "flag_parse=30"} {
		if !strings.Contains(line, want) {
			t.Fatalf("operator line %q missing %q", line, want)
		}
	}
}

// TestScanDispatchDOAClearOnHealthyWindow is the SILENT direction: the post-outage
// fleet must produce a clear verdict and name no evidence.
func TestScanDispatchDOAClearOnHealthyWindow(t *testing.T) {
	runs := t.TempDir()
	for i := 0; i < 12; i++ {
		plantRunLog(t, runs, fmt.Sprintf("resolve-13%02d-20260805-0551%02d.log", i, i),
			doaHealthyLog+strings.Repeat("turn output\n", 800), 90*time.Minute)
	}
	rep := scanDispatchDOA(runs, dispatchDOAWindow, dispatchDOASettle, time.Now())
	if rep.Runs != 12 || rep.DOA != 0 || rep.Status != dispatchdoa.StatusClear {
		t.Fatalf("scan(healthy) = %+v, want 12 runs / 0 DOA / clear", rep)
	}
	if line := dispatchDOALine(rep, dispatchDOAWindow); !strings.Contains(line, "0 dead on arrival") {
		t.Fatalf("operator line %q should confirm the check ran and found nothing", line)
	}
}

// TestScanDispatchDOASkipsLiveAndStaleRuns pins the two window guards. A worker
// spawned seconds ago has flushed only the spawn header, so it is momentarily SHAPED
// like a DOA record — the settle floor is what stops the detector from accusing a live
// worker. A run older than the window is out of frame entirely.
func TestScanDispatchDOASkipsLiveAndStaleRuns(t *testing.T) {
	runs := t.TempDir()
	// A worker that spawned 1 second ago and has written only the header.
	plantRunLog(t, runs, "resolve-1-20260807-000000.log", "# fak-spawn 20260807-000000 issue=1 lane=x backend=claude argv0=fak.exe\n", time.Second)
	if rep := scanDispatchDOA(runs, dispatchDOAWindow, dispatchDOASettle, time.Now()); rep.Runs != 0 {
		t.Fatalf("scan counted a still-settling live worker: %+v", rep)
	}
	// A real DOA record, but from a week ago: outside the window.
	plantRunLog(t, runs, "resolve-2-20260728-000000.log", doaOutageLog, 7*24*time.Hour)
	if rep := scanDispatchDOA(runs, dispatchDOAWindow, dispatchDOASettle, time.Now()); rep.Runs != 0 {
		t.Fatalf("scan counted a run outside the window: %+v", rep)
	}
	// The same record inside the window IS graded.
	plantRunLog(t, runs, "resolve-3-20260807-000000.log", doaOutageLog, 3*time.Hour)
	rep := scanDispatchDOA(runs, dispatchDOAWindow, dispatchDOASettle, time.Now())
	if rep.Runs != 1 || rep.DOA != 1 {
		t.Fatalf("scan = %+v, want the in-window DOA record graded", rep)
	}
}

// TestDispatchStatusCardCarriesSpawnHealth is the point of the whole change: the
// operator card an operator ALREADY reads must say the fleet is dying. Throughout the
// real outage this card printed "0 live worker(s)" and nothing else, which reads exactly
// like an idle fleet.
func TestDispatchStatusCardCarriesSpawnHealth(t *testing.T) {
	runs := filepath.Join(t.TempDir(), "runs")
	for i := 0; i < 6; i++ {
		plantRunLog(t, runs, fmt.Sprintf("resolve-%d-20260728-2131%02d.log", i+1, i), doaOutageLog, 2*time.Hour)
	}
	snap := dispatchStatusScan(runs, t.TempDir())
	if snap.SpawnHealth == nil {
		t.Fatal("snapshot carries no spawn_health section")
	}
	if snap.SpawnHealth.DOA != 6 || snap.SpawnHealth.Runs != 6 || snap.SpawnHealth.Status != dispatchdoa.StatusAlarm {
		t.Fatalf("spawn_health = %+v, want 6/6 alarm", snap.SpawnHealth)
	}
	if got := snap.SpawnHealth.NextActions[dispatchdoa.CauseFlagParse]; strings.TrimSpace(got) == "" {
		t.Fatalf("spawn_health.next_actions = %+v, want flag-parse remediation", snap.SpawnHealth.NextActions)
	}
	card := renderDispatchStatus(snap)
	if !strings.Contains(card, "0 live worker(s)") {
		t.Fatalf("card should still report the (misleading on its own) live count: %q", card)
	}
	for _, want := range []string{"spawn health: ALARM", "6 of 6", "DIED ON ARRIVAL"} {
		if !strings.Contains(card, want) {
			t.Fatalf("card %q missing %q", card, want)
		}
	}
	if md := renderDispatchStatusMarkdown(snap); !strings.Contains(md, "spawn health: ALARM") {
		t.Fatalf("markdown card missing the alarm: %q", md)
	}
}

// TestDispatchStatusOmitsSpawnHealthWithoutRuns keeps the change additive: a snapshot
// over an absent/empty runs dir is byte-identical to before #5868.
func TestDispatchStatusOmitsSpawnHealthWithoutRuns(t *testing.T) {
	snap := dispatchStatusScan(filepath.Join(t.TempDir(), "runs"), t.TempDir())
	if snap.SpawnHealth != nil {
		t.Fatalf("spawn_health = %+v, want nil on an empty runs dir", snap.SpawnHealth)
	}
	if strings.Contains(renderDispatchStatus(snap), "spawn health") {
		t.Fatal("card mentions spawn health with nothing to report")
	}
}

// The real-corpus both-directions regression over every retained
// .dispatch-runs/resolve-*.log record lives in internal/dispatchdoa
// (TestClassifyAgainstRealOutageCorpus) — it needs only the pure classifier.
