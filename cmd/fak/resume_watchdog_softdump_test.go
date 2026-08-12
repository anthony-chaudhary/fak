package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// softDumpStubs points the trajectory watchdog at one synthetic session whose anchor
// curve is STALL with its newest witnessed progress point stalledFor ago, and whose
// process is (or is not) live.
func softDumpStubs(t *testing.T, sid string, stalledFor time.Duration, alive bool, stamped bool) {
	t.Helper()
	pts := []trajctl.CurvePoint{}
	if stamped {
		pts = append(pts, trajctl.CurvePoint{Value: .4, UnixMillis: time.Now().Add(-stalledFor).UnixMilli()})
	}
	curve := trajctl.ObjectiveCurve{
		ObjectiveID: "issue-5287",
		Signal:      trajctl.SignalStall,
		Latest:      .4,
		Detail:      "flat witnessed progress",
		Methods:     []trajctl.MethodCurve{{Method: "commit-progress", Points: pts}},
	}
	oldAnchor, oldProcs, oldSoft := rwResumeAnchor, rwCollectProcCmdlines, rwSoftWatchdog
	rwResumeAnchor = func(string) resume.ResumeAnchor {
		return resume.ResumeAnchor{Schema: resume.ResumeAnchorSchema, Session: sid, ObjectiveID: curve.ObjectiveID, Objective: "capture the wedge", Curve: &curve, Present: true}
	}
	rwCollectProcCmdlines = func() ([]string, bool) {
		if !alive {
			return nil, true
		}
		return []string{"claude --resume " + sid}, true
	}
	rwSoftWatchdog = resume.NewSoftWatchdog(time.Minute)
	t.Cleanup(func() { rwResumeAnchor, rwCollectProcCmdlines, rwSoftWatchdog = oldAnchor, oldProcs, oldSoft })
}

func softDumpLedgerLines(t *testing.T, path string) []string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger %s: %v", path, err)
	}
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}

// #5287 acceptance: on the alive-but-stalled branch the watchdog writes a diagnostic
// state dump into the durable session record BEFORE the nudge/revive decision row,
// carrying the evidence (elapsed stall, last progress marker, the stalled session's
// live process) — and the intervention itself is unchanged.
func TestRwApplyTrajectoryWatchdogWritesSoftDumpBeforeDecision(t *testing.T) {
	const sid, trace = "soft-sid", "soft-trace"
	softDumpStubs(t, sid, 30*time.Minute, true, true)
	t.Cleanup(func() { sessionctl.ClearObjective(trace) })

	ledger := filepath.Join(t.TempDir(), "resume_ledger.jsonl")
	handled, got := rwApplyTrajectoryWatchdog(resume.WatchdogPlanRow{Session: sid}, nil, &rwProcScan{}, map[string]string{sid: trace}, ledger, true)
	if !handled || got.Action != resume.TrajectoryNudge {
		t.Fatalf("handled=%v decision=%+v, want the unchanged NUDGE", handled, got)
	}

	lines := softDumpLedgerLines(t, ledger)
	if len(lines) != 2 {
		t.Fatalf("want a soft dump then a decision, got %d rows:\n%s", len(lines), strings.Join(lines, "\n"))
	}
	var row resume.SoftDumpRow
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatalf("first ledger row is not a soft dump: %v (%s)", err, lines[0])
	}
	if row.Phase != resume.SoftDumpPhase || row.Schema != resume.SoftDumpSchema || row.Session != sid || row.Trace != trace || row.TS == "" {
		t.Fatalf("soft dump row envelope wrong: %+v", row)
	}
	if !row.Dump.Alive || !row.Dump.ProgressStalled ||
		row.Dump.LivenessVsProgress != resume.SoftSplitAliveWithoutProgress ||
		row.Dump.LastProgressMarker != "flat witnessed progress" ||
		row.Dump.PendingAction != "claude --resume "+sid ||
		row.Dump.ElapsedSinceProgressMillis < (29*time.Minute).Milliseconds() {
		t.Fatalf("soft dump carries no usable diagnostic: %+v", row.Dump)
	}
	if !strings.Contains(lines[1], `"phase":"trajectory_decision"`) {
		t.Fatalf("the dump must precede the nudge/revive decision row, got:\n%s", strings.Join(lines, "\n"))
	}

	// Exactly once per stall episode: the next tick on the same unbroken stall
	// re-runs the identical decision and adds no second dump.
	handledAgain, again := rwApplyTrajectoryWatchdog(resume.WatchdogPlanRow{Session: sid}, nil, &rwProcScan{}, map[string]string{sid: trace}, ledger, true)
	if !handledAgain || again != got {
		t.Fatalf("the soft watchdog must not alter the intervention: %+v vs %+v", again, got)
	}
	if n := strings.Count(strings.Join(softDumpLedgerLines(t, ledger), "\n"), resume.SoftDumpPhase); n != 1 {
		t.Fatalf("want exactly one dump per stall episode, got %d", n)
	}
}

// Soft observes, hard decides: a session with no proven soft-timeout elapse, and a
// dead session (the hard revive path's business), both capture nothing — and neither
// case changes the decision the watchdog would have made without the soft step.
func TestRwApplyTrajectoryWatchdogSoftDumpStaysGated(t *testing.T) {
	tests := []struct {
		name       string
		sid        string
		stalledFor time.Duration
		alive      bool
		stamped    bool
		want       resume.TrajectoryWatchdogAction
		handled    bool
	}{
		{"inside the soft grace window", "grace-sid", 10 * time.Second, true, true, resume.TrajectoryNudge, true},
		{"stall clock unknown", "noclock-sid", 30 * time.Minute, true, false, resume.TrajectoryNudge, true},
		{"dead session belongs to the hard path", "dead-sid", 30 * time.Minute, false, true, resume.TrajectoryReviveAnchor, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			softDumpStubs(t, tt.sid, tt.stalledFor, tt.alive, tt.stamped)
			t.Cleanup(func() { sessionctl.ClearObjective("t-" + tt.sid) })
			ledger := filepath.Join(t.TempDir(), "resume_ledger.jsonl")
			handled, got := rwApplyTrajectoryWatchdog(resume.WatchdogPlanRow{Session: tt.sid}, nil, &rwProcScan{}, map[string]string{tt.sid: "t-" + tt.sid}, ledger, true)
			if handled != tt.handled || got.Action != tt.want {
				t.Fatalf("handled=%v action=%s, want handled=%v action=%s", handled, got.Action, tt.handled, tt.want)
			}
			raw, err := os.ReadFile(ledger)
			if err != nil && !os.IsNotExist(err) {
				t.Fatalf("read ledger: %v", err)
			}
			if strings.Contains(string(raw), resume.SoftDumpPhase) {
				t.Fatalf("%s must capture nothing, ledger:\n%s", tt.name, raw)
			}
		})
	}
}
