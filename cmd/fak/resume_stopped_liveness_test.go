package main

// These tests drive the REAL `fak resume stopped` argv path (runResumeStopped) with a
// hermetic fleet home, a staged process table and a staged launch ledger. Driving argv is
// the point: internal/resume/stopped already has table tests for ClassifyWithLiveness, and
// those pass whether or not production ever reaches it — which is exactly the #5440 gap.
// Reverting cmd/fak/resume_stopped.go's ClassifyWithLiveness call back to the LivenessUnknown
// Classify wrapper must red TestResumeStoppedDriverLivenessThroughArgv's gone/live cases.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/resume/stopped"
)

// stoppedLivenessSID is the transcript stem (a session uuid) every case below triages.
const stoppedLivenessSID = "11111111-2222-3333-4444-555555555555"

// stoppedLivenessTriage is the decoded --json record the cases assert on.
type stoppedLivenessTriage struct {
	NResume        int            `json:"n_resume"`
	NDefer         int            `json:"n_defer"`
	NSkip          int            `json:"n_skip"`
	Resume         []stopped.Row  `json:"resume"`
	Defer          []stopped.Row  `json:"defer"`
	Rows           []stopped.Row  `json:"rows"`
	DriverLiveness map[string]any `json:"driver_liveness"`
}

// stageStoppedLivenessHome builds a self-contained fleet home holding ONE worker account
// with ONE mid-tool transcript: an assistant tool_use with no following tool_result, the
// exact tail that is ambiguous between a crash and a slow tool call. Everything lives in
// t.TempDir(), and the fleet policy/registry env is redirected there too, so the run never
// reads the operator's real accounts, ledger or process history.
func stageStoppedLivenessHome(t *testing.T) (home, regDir string) {
	t.Helper()
	home, regDir = t.TempDir(), t.TempDir()

	projDir := filepath.Join(home, ".claude-t5440", "projects", "proj-under-test")
	if err := os.MkdirAll(projDir, 0o755); err != nil {
		t.Fatalf("stage account dir: %v", err)
	}
	transcript := filepath.Join(projDir, stoppedLivenessSID+".jsonl")
	lines := []string{
		`{"type":"user","timestamp":"2026-07-28T00:00:00Z","cwd":"/proj","version":"1.0.0","sessionId":"` +
			stoppedLivenessSID + `","message":{"role":"user","content":[{"type":"text","text":"do the work"}]}}`,
		`{"type":"assistant","timestamp":"2026-07-28T00:01:00Z","sessionId":"` + stoppedLivenessSID +
			`","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash"}]}}`,
	}
	if err := os.WriteFile(transcript, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("stage transcript: %v", err)
	}
	// Backdate past stopped.LiveMinutes: a freshly-written file would take the mtime
	// freshness arm (DispLive) and never reach the mid-tool branch this test is about. Still
	// well inside the --window-h the cases pass.
	old := time.Now().Add(-30 * time.Minute)
	if err := os.Chtimes(transcript, old, old); err != nil {
		t.Fatalf("backdate transcript: %v", err)
	}

	policy := filepath.Join(regDir, "accounts_policy.json")
	if err := os.WriteFile(policy, []byte(`{"exclude":[],"include_only":[],"notes":{},"account_profiles":{}}`), 0o644); err != nil {
		t.Fatalf("stage accounts policy: %v", err)
	}
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", filepath.Join(home, ".config"))
	t.Setenv("FLEET_POLICY_PATH", policy)
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv("FAK_RESUME_TRIAGE_GATE", "")
	return home, regDir
}

// runStoppedLivenessTriage drives the real argv entry point and decodes its machine record.
func runStoppedLivenessTriage(t *testing.T, home string) stoppedLivenessTriage {
	t.Helper()
	var out, errBuf bytes.Buffer
	if code := runResumeStopped(&out, &errBuf, []string{"--home", home, "--window-h", "24", "--json"}); code != 0 {
		t.Fatalf("fak resume stopped exit=%d stderr=%s", code, errBuf.String())
	}
	var got stoppedLivenessTriage
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode triage json: %v\nraw=%s", err, out.String())
	}
	return got
}

// TestResumeStoppedDriverLivenessThroughArgv is the #5440 acceptance: all three verdicts,
// produced by the real command, from evidence the command itself observed.
func TestResumeStoppedDriverLivenessThroughArgv(t *testing.T) {
	const selfPID = 4242
	// A plausible table that always contains the scanner itself, so an absence in it means
	// something. Cases override this where the point IS that it does not.
	baseTable := []procguard.Proc{
		{PID: selfPID, Name: "fak", Cmdline: "fak resume stopped --json"},
		{PID: 700, Name: "sh", Cmdline: "/bin/sh -c make ci"},
	}
	driverCmd := "node /opt/claude/cli.js --resume " + stoppedLivenessSID

	cases := []struct {
		name string
		// procs/scanErr/self stage the observation; ledger stages the durable launch record
		// and identity the durable session-start identity store (the second recorded-pid
		// producer, #5542).
		procs      []procguard.Proc
		scanErr    string
		self       int
		ledger     string
		identity   string
		wantDisp   stopped.Disp
		wantLive   stopped.Liveness
		wantBucket string // "resume" | "defer" | "skip"
		wantBlock  string
	}{
		{
			// GONE: the launcher recorded pid 9001 for this session and 9001 is absent from a
			// table that can see its own reader. That is positive evidence of death, so the
			// mid-tool tail is the crash it looks like — and the row is OFFERED for resume.
			name:       "recorded driver pid absent from a complete table resumes",
			procs:      baseTable,
			self:       selfPID,
			ledger:     `{"ts":"t","session":"` + stoppedLivenessSID + `","pid":9001,"phase":"launched"}` + "\n",
			wantDisp:   stopped.DispStoppedMidtool,
			wantLive:   stopped.LivenessGone,
			wantBucket: "resume",
		},
		{
			// LIVE: a running process names the session on its command line. The mid-tool tail
			// is a slow tool call, not a death — never a resume candidate.
			name:       "a running process naming the session is left alone",
			procs:      append(append([]procguard.Proc{}, baseTable...), procguard.Proc{PID: 9001, Name: "node", Cmdline: driverCmd}),
			self:       selfPID,
			ledger:     `{"ts":"t","session":"` + stoppedLivenessSID + `","pid":9001,"phase":"launched"}` + "\n",
			wantDisp:   stopped.DispLive,
			wantLive:   stopped.LivenessLive,
			wantBucket: "skip",
		},
		{
			// UNKNOWN: the table is readable and complete, nothing names the session, and no
			// launcher ever recorded a driver pid for it. Absence from the table is then NOT
			// evidence of death — a driver launched without the id on its argv looks identical —
			// so the row keeps deferring with the ambiguity named.
			name:       "no recorded driver pid still defers as unknown",
			procs:      baseTable,
			self:       selfPID,
			ledger:     "",
			wantDisp:   stopped.DispMidtoolUnknown,
			wantLive:   stopped.LivenessUnknown,
			wantBucket: "defer",
			wantBlock:  stopped.MidtoolUnknownBlockedBy,
		},
		{
			// The host where liveness cannot be observed at all (#5385: the POSIX census comes
			// back empty on some hosts). A recorded pid is present and absent from the table —
			// but the table is not one, so the absence witnesses nothing. UNKNOWN, not gone.
			name:       "empty process table witnesses nothing even with a recorded pid",
			procs:      nil,
			self:       selfPID,
			ledger:     `{"ts":"t","session":"` + stoppedLivenessSID + `","pid":9001,"phase":"launched"}` + "\n",
			wantDisp:   stopped.DispMidtoolUnknown,
			wantLive:   stopped.LivenessUnknown,
			wantBucket: "defer",
			wantBlock:  stopped.MidtoolUnknownBlockedBy,
		},
		{
			// A table that cannot see the process that took it is provably incomplete, so a
			// missing pid may simply be a row the census dropped. UNKNOWN, not gone.
			name:       "table missing its own reader cannot witness gone",
			procs:      []procguard.Proc{{PID: 700, Name: "sh", Cmdline: "/bin/sh -c make ci"}},
			self:       selfPID,
			ledger:     `{"ts":"t","session":"` + stoppedLivenessSID + `","pid":9001,"phase":"launched"}` + "\n",
			wantDisp:   stopped.DispMidtoolUnknown,
			wantLive:   stopped.LivenessUnknown,
			wantBucket: "defer",
			wantBlock:  stopped.MidtoolUnknownBlockedBy,
		},
		{
			// A collector error is not a clean host. UNKNOWN, not gone.
			name:       "a failed scan witnesses nothing",
			procs:      baseTable,
			scanErr:    "ps: operation not permitted",
			self:       selfPID,
			ledger:     `{"ts":"t","session":"` + stoppedLivenessSID + `","pid":9001,"phase":"launched"}` + "\n",
			wantDisp:   stopped.DispMidtoolUnknown,
			wantLive:   stopped.LivenessUnknown,
			wantBucket: "defer",
			wantBlock:  stopped.MidtoolUnknownBlockedBy,
		},
		{
			// A deferral row in the launch ledger records no spawn, so it contributes no driver
			// pid — the triage must not read a never-started launch as a dead driver.
			name:       "a non-launch ledger row contributes no driver pid",
			procs:      baseTable,
			self:       selfPID,
			ledger:     `{"ts":"t","session":"` + stoppedLivenessSID + `","pid":9001,"phase":"deferred"}` + "\n",
			wantDisp:   stopped.DispMidtoolUnknown,
			wantLive:   stopped.LivenessUnknown,
			wantBucket: "defer",
			wantBlock:  stopped.MidtoolUnknownBlockedBy,
		},
		{
			// #5542, the case the whole ticket is about: a FIRST-GENERATION worker. Nothing ever
			// resumed it, so the launch ledger holds no pid for it and it deferred forever. Its
			// SessionStart hook witnessed the driver and recorded the pid on the identity row, and
			// that pid is absent from a table that can see its own reader — so the crash is
			// finally witnessed and the row is offered for resume.
			name:       "a session-start identity pid absent from a complete table resumes",
			procs:      baseTable,
			self:       selfPID,
			ledger:     "",
			identity:   `{"ts":"t","uuid":"` + stoppedLivenessSID + `","trace":"guard","pid":9001,"via":"guard-sessionstart"}` + "\n",
			wantDisp:   stopped.DispStoppedMidtool,
			wantLive:   stopped.LivenessGone,
			wantBucket: "resume",
		},
		{
			// The safety half of the same seam. A first-generation driver carries no session id on
			// its argv, so the command-line scan cannot find it — only the recorded pid can. It is
			// STILL RUNNING, so the mid-tool tail is a slow tool call and the row must be left
			// alone. Resolving this arm as gone would resume a lane under a live driver.
			name: "a running session-start identity pid is left alone",
			procs: append(append([]procguard.Proc{}, baseTable...),
				procguard.Proc{PID: 9001, Name: "claude", Cmdline: "claude -p do the work"}),
			self:       selfPID,
			ledger:     "",
			identity:   `{"ts":"t","uuid":"` + stoppedLivenessSID + `","trace":"guard","pid":9001,"via":"guard-sessionstart"}` + "\n",
			wantDisp:   stopped.DispLive,
			wantLive:   stopped.LivenessLive,
			wantBucket: "skip",
		},
		{
			// Every identity row written before the pid field existed decodes with pid 0, and a
			// producer that could not WITNESS its driver records 0 rather than guessing. Absence
			// must read as "not recorded", never as "gone" — so this row keeps deferring.
			name:       "an identity row carrying no pid is not evidence of death",
			procs:      baseTable,
			self:       selfPID,
			ledger:     "",
			identity:   `{"ts":"t","uuid":"` + stoppedLivenessSID + `","trace":"guard","via":"guard-sessionstart"}` + "\n",
			wantDisp:   stopped.DispMidtoolUnknown,
			wantLive:   stopped.LivenessUnknown,
			wantBucket: "defer",
			wantBlock:  stopped.MidtoolUnknownBlockedBy,
		},
		{
			// Precedence, pinned in the direction that matters for safety. The launcher recorded a
			// pid at the NEWEST spawn and that process is alive; the identity row is from the
			// generation that spawn replaced and its pid is long gone. The launch ledger must win,
			// or a stale identity pid would manufacture a `gone` for a session whose driver is
			// running right now.
			name: "a stale identity pid never overrides a live launch-ledger pid",
			procs: append(append([]procguard.Proc{}, baseTable...),
				procguard.Proc{PID: 9001, Name: "claude", Cmdline: "claude -p do the work"}),
			self:       selfPID,
			ledger:     `{"ts":"t","session":"` + stoppedLivenessSID + `","pid":9001,"phase":"launched"}` + "\n",
			identity:   `{"ts":"t","uuid":"` + stoppedLivenessSID + `","trace":"guard","pid":9002,"via":"guard-sessionstart"}` + "\n",
			wantDisp:   stopped.DispLive,
			wantLive:   stopped.LivenessLive,
			wantBucket: "skip",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			home, regDir := stageStoppedLivenessHome(t)
			if err := os.WriteFile(filepath.Join(regDir, "resume_ledger.jsonl"), []byte(c.ledger), 0o644); err != nil {
				t.Fatalf("stage launch ledger: %v", err)
			}
			if err := os.WriteFile(filepath.Join(regDir, "resume_identity.jsonl"), []byte(c.identity), 0o644); err != nil {
				t.Fatalf("stage identity store: %v", err)
			}
			procs, scanErr, self := c.procs, c.scanErr, c.self
			prevProcs, prevSelf := stoppedProcRelations, stoppedSelfPID
			t.Cleanup(func() { stoppedProcRelations, stoppedSelfPID = prevProcs, prevSelf })
			stoppedProcRelations = func() ([]procguard.Proc, string) { return procs, scanErr }
			stoppedSelfPID = func() int { return self }

			got := runStoppedLivenessTriage(t, home)
			if len(got.Rows) != 1 {
				t.Fatalf("want exactly 1 classified row, got %d: %+v", len(got.Rows), got.Rows)
			}
			row := got.Rows[0]
			if row.Disp != c.wantDisp {
				t.Fatalf("disp = %q, want %q (liveness=%q blocked_by=%q)", row.Disp, c.wantDisp, row.Liveness, row.BlockedBy)
			}
			if row.Liveness != c.wantLive {
				t.Fatalf("liveness = %q, want %q", row.Liveness, c.wantLive)
			}
			buckets := map[string]int{"resume": got.NResume, "defer": got.NDefer, "skip": got.NSkip}
			for name, n := range buckets {
				want := 0
				if name == c.wantBucket {
					want = 1
				}
				if n != want {
					t.Fatalf("bucket %s = %d, want %d (resume=%d defer=%d skip=%d)",
						name, n, want, got.NResume, got.NDefer, got.NSkip)
				}
			}
			if c.wantBlock != "" {
				if len(got.Defer) != 1 || got.Defer[0].BlockedBy != c.wantBlock {
					t.Fatalf("deferred blocked_by = %+v, want %q", got.Defer, c.wantBlock)
				}
			}
			if got.DriverLiveness == nil {
				t.Fatalf("--json carries no driver_liveness record")
			}
			reasons, _ := got.DriverLiveness["reasons"].(map[string]any)
			if why, _ := reasons[stoppedLivenessSID].(string); strings.TrimSpace(why) == "" {
				t.Fatalf("driver_liveness.reasons has no reason for the session: %+v", got.DriverLiveness)
			}
		})
	}
}

// TestResumeStoppedHumanRenderStatesWhatWasObservable pins that the operator render says
// out loud when driver liveness could not be observed on this host, so an unobservable host
// never looks the same as a host where every driver was checked and found alive.
func TestResumeStoppedHumanRenderStatesWhatWasObservable(t *testing.T) {
	prevProcs, prevSelf := stoppedProcRelations, stoppedSelfPID
	t.Cleanup(func() { stoppedProcRelations, stoppedSelfPID = prevProcs, prevSelf })

	for _, c := range []struct {
		name  string
		procs []procguard.Proc
		want  string
	}{
		{
			name:  "unobservable host says so",
			procs: nil,
			want:  "NOT OBSERVABLE on this host",
		},
		{
			name:  "observable host reports what it scanned",
			procs: []procguard.Proc{{PID: 4242, Cmdline: "fak resume stopped"}},
			want:  "2 processes scanned",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			home, _ := stageStoppedLivenessHome(t)
			procs := c.procs
			if len(procs) > 0 {
				// One extra row whose command line the platform would not surrender: it must be
				// counted as NOT EXAMINED, never folded into the examined-and-clean population.
				procs = append(append([]procguard.Proc{}, procs...), procguard.Proc{PID: 700})
			}
			stoppedProcRelations = func() ([]procguard.Proc, string) { return procs, "" }
			stoppedSelfPID = func() int { return 4242 }

			var out, errBuf bytes.Buffer
			if code := runResumeStopped(&out, &errBuf, []string{"--home", home, "--window-h", "24"}); code != 0 {
				t.Fatalf("exit=%d stderr=%s", code, errBuf.String())
			}
			if !strings.Contains(out.String(), c.want) {
				t.Fatalf("render does not state %q:\n%s", c.want, out.String())
			}
			if len(c.procs) > 0 && !strings.Contains(out.String(), "1 command lines not examined") {
				t.Fatalf("render hides the unexaminable remainder:\n%s", out.String())
			}
		})
	}
}

// TestStoppedDriverFactsNeverGuessesFromTheClock pins the constraint that makes the whole
// producer trustworthy: the evidence base is built from the process table and the recorded
// launch pid ONLY. Two folds that differ solely in how much wall time has passed — modelled
// here as identical inputs folded twice — must agree, and no input to the fold is a time.
func TestStoppedDriverFactsNeverGuessesFromTheClock(t *testing.T) {
	prevProcs, prevSelf := stoppedProcRelations, stoppedSelfPID
	t.Cleanup(func() { stoppedProcRelations, stoppedSelfPID = prevProcs, prevSelf })
	stoppedProcRelations = func() ([]procguard.Proc, string) {
		return []procguard.Proc{{PID: 11, Cmdline: "fak resume stopped"}}, ""
	}
	stoppedSelfPID = func() int { return 11 }
	regDir := t.TempDir()
	ledger := filepath.Join(regDir, "resume_ledger.jsonl")
	if err := os.WriteFile(ledger, []byte(`{"session":"`+stoppedLivenessSID+`","pid":9001,"phase":"launched"}`+"\n"), 0o644); err != nil {
		t.Fatalf("stage ledger: %v", err)
	}
	first, _ := foldStoppedDriverFacts(ledger, regDir).livenessFor(stoppedLivenessSID)
	second, _ := foldStoppedDriverFacts(ledger, regDir).livenessFor(stoppedLivenessSID)
	if first != stopped.LivenessGone || second != first {
		t.Fatalf("liveness must be stable and evidence-derived: first=%q second=%q", first, second)
	}
	// An unreadable command line is NOT EXAMINED, and must never render as "examined and
	// found dead": the recorded pid is running, so the verdict is unknown, not gone.
	stoppedProcRelations = func() ([]procguard.Proc, string) {
		return []procguard.Proc{{PID: 11, Cmdline: "fak resume stopped"}, {PID: 9001, Cmdline: ""}}, ""
	}
	facts := foldStoppedDriverFacts(ledger, regDir)
	if facts.cmdlineUnread != 1 {
		t.Fatalf("cmdline_not_examined = %d, want 1 (the unexaminable remainder must stay visible)", facts.cmdlineUnread)
	}
	if live, why := facts.livenessFor(stoppedLivenessSID); live != stopped.LivenessUnknown {
		t.Fatalf("running-but-unreadable driver = %q (%s), want unknown", live, why)
	}
}
