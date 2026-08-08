package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

// stageBareInteractiveSessionStart stages the world a plain, attended `fak guard -- claude`
// hook actually runs in, and returns the journal path plus the cwd the row must record.
//
// Every knob is set explicitly rather than inherited: this suite runs on the maintainers'
// shared host, where FAK_* tuning in the operator's own environment would otherwise decide
// what the test observes. The trace is the literal "guard" because that is what
// resolveGuardSessionID hands an ordinary attended launch — the degenerate constant this
// registration has to survive.
func stageBareInteractiveSessionStart(t *testing.T, uuid string) (journal, cwd string) {
	t.Helper()
	// The negframe journal the emit writes is workspace-relative, so run inside a scratch tree;
	// that also makes the recorded cwd a value the test owns.
	t.Chdir(t.TempDir())
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	journal = filepath.Join(t.TempDir(), "session-journal.jsonl")
	t.Setenv(sessionjournal.EnvPath, journal)
	t.Setenv(guardSessionJournalEnvMode, "") // no host opt-out leaking in
	t.Setenv(guardSessionStartEnvMode, "")   // affordance knob at its default
	t.Setenv("FLEET_REG_DIR", t.TempDir())   // identity join goes to a scratch store
	t.Setenv("CLAUDE_CODE_SESSION_ID", uuid)
	// The hook is a child of the driver it registers; stage that chain so the pid it records is
	// one the test built rather than the host's real (and slow to census) process tree.
	stageGuardSessionStartWitness(t, 4242, []procguard.Proc{
		{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart --trace guard"},
		{PID: 9001, Name: "claude", PPID: procguard.IntPtr(15696), Cmdline: "claude"},
		{PID: 15696, Name: "fak", Cmdline: "fak guard -- claude"},
	})
	return journal, cwd
}

// runBareInteractiveSessionStart fires the hook exactly as an attended `fak guard -- claude`
// wires it: mode on, the constant trace, and NO --managed (that flag marks a headless `-p`
// worker). Returns the folded journal.
func runBareInteractiveSessionStart(t *testing.T, journal string, extra ...string) []sessionjournal.Session {
	t.Helper()
	var out, errb bytes.Buffer
	argv := append([]string{"--mode", "on", "--trace", "guard"}, extra...)
	if code := runGuardSessionStart(&out, &errb, argv); code != 0 {
		t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start); stderr=%s", code, errb.String())
	}
	return sessionjournal.FoldEvents(sessionjournal.LoadFile(journal))
}

// TestGuardSessionStartRegistersBareInteractiveSession is the C3 acceptance witness (#3787):
// a plain interactive `fak guard -- claude` — no `--managed`, no durability opt-in, no
// session-registry.json — leaves a boot-stamped `open` row in the crash journal.
//
// Before this the hook persisted only the uuid<->trace join (#4112), so the session's own
// START went unrecorded and `fak sessionjournal report` had nothing to classify for the most
// common shape on the host. The registration is what makes the boot-epoch fold applicable to
// an attended session at all.
func TestGuardSessionStartRegistersBareInteractiveSession(t *testing.T) {
	const uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
	journal, cwd := stageBareInteractiveSessionStart(t, uuid)

	sessions := runBareInteractiveSessionStart(t, journal)
	if len(sessions) != 1 {
		t.Fatalf("recorded %d sessions, want exactly 1 open row: %+v", len(sessions), sessions)
	}
	s := sessions[0]

	// The transcript UUID is the fold key, NOT the trace: an attended launch's trace is the
	// constant "guard", so keying on it would fold every interactive session on the host into
	// one row.
	if s.ID != uuid {
		t.Fatalf("row id = %q, want the transcript uuid %q", s.ID, uuid)
	}
	// The cwd is the whole reason this beats resume's slug-recovered path (epic #3784): it is
	// recorded verbatim, not reconstructed from an irreversible project slug.
	if s.CWD != cwd {
		t.Fatalf("row cwd = %q, want %q", s.CWD, cwd)
	}
	if s.Agent != guardSessionJournalAgent {
		t.Fatalf("row agent = %q, want %q", s.Agent, guardSessionJournalAgent)
	}
	// The pid is the WITNESSED driver, never the hook's own ephemeral pid — the hook has exited
	// by the time anything folds this row, so recording itself would read as a dead session.
	if s.PID != 9001 {
		t.Fatalf("row pid = %d, want the witnessed driver 9001", s.PID)
	}
	if s.StartedAt.IsZero() {
		t.Fatalf("row carries no start time: %+v", s)
	}
	if s.Closed {
		t.Fatalf("an open row must not fold as closed: %+v", s)
	}
	// Boot-stamped wherever the host can name a boot epoch. macOS returns "unknown" today
	// (bootepoch_other.go), so the assertion follows what the platform can actually witness
	// rather than demanding a stamp the fold itself would skip.
	if want := sessionjournal.BootID(bootTimeForTest(t)); want != "" && s.Boot != want {
		t.Fatalf("row boot = %q, want the current boot id %q", s.Boot, want)
	}
}

// TestGuardSessionStartRegistrationSurvivesReboot is the point of registering at all: the row
// an attended session leaves is enough for the boot-epoch fold to name it a resume candidate
// after a machine-wide crash, carrying the cwd to relaunch in. Classified against an injected
// LATER boot instant, so it pins the fold rather than the host's real uptime.
func TestGuardSessionStartRegistrationSurvivesReboot(t *testing.T) {
	const uuid = "11111111-1111-1111-1111-111111111111"
	journal, cwd := stageBareInteractiveSessionStart(t, uuid)
	sessions := runBareInteractiveSessionStart(t, journal)

	// The reboot the epic is named for: the machine came up AFTER this session started, and the
	// session never wrote a clean close — so it cannot still be running.
	rebooted := sessions[0].StartedAt.Add(time.Minute)
	classified := sessionjournal.Classify(sessions, sessionjournal.ClassifyConfig{
		Now:      rebooted.Add(time.Minute),
		BootTime: rebooted,
	})
	if len(classified) != 1 {
		t.Fatalf("classified %d rows, want 1", len(classified))
	}
	if got := classified[0].Status; got != sessionjournal.StatusCrashed {
		t.Fatalf("status = %s, want CRASHED (it started before the current boot)", got)
	}
	if got := classified[0].Reason; got != sessionjournal.ReasonMachineReboot {
		t.Fatalf("reason = %s, want %s", got, sessionjournal.ReasonMachineReboot)
	}
	if classified[0].CWD != cwd {
		t.Fatalf("the resume candidate lost its cwd: %q, want %q", classified[0].CWD, cwd)
	}
}

// TestGuardSessionStartRegistersRegardlessOfAffordanceMode pins the UNCONDITIONAL half of the
// DoD from the other side. FAK_GUARD_AFFORDANCE_MODE governs the injected hint — a presentation
// knob — and must not decide whether a durable recovery row exists, exactly as the sibling
// identity join is already written ahead of that check.
func TestGuardSessionStartRegistersRegardlessOfAffordanceMode(t *testing.T) {
	const uuid = "22222222-2222-2222-2222-222222222222"
	journal, _ := stageBareInteractiveSessionStart(t, uuid)

	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "off", "--trace", "guard"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if out.Len() != 0 {
		t.Fatalf("off mode must still inject nothing, got: %q", out.String())
	}
	sessions := sessionjournal.FoldEvents(sessionjournal.LoadFile(journal))
	if len(sessions) != 1 || sessions[0].ID != uuid {
		t.Fatalf("the affordance knob suppressed the durable registration: %+v", sessions)
	}
}

// TestGuardSessionStartRegisterOptOut pins the lever: registration is on by DEFAULT (nothing
// opts in), and FAK_SESSION_JOURNAL_REGISTER=off is the opt-OUT a lean harness can set. The
// hook still exits 0 and still injects — the kill switch costs the row, never the start.
func TestGuardSessionStartRegisterOptOut(t *testing.T) {
	const uuid = "33333333-3333-3333-3333-333333333333"
	journal, _ := stageBareInteractiveSessionStart(t, uuid)
	t.Setenv(guardSessionJournalEnvMode, "OFF") // case-insensitive, like the affordance knob

	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--trace", "guard"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(out.String(), "fak_index_work") {
		t.Fatalf("the register opt-out also suppressed the affordance: %s", out.String())
	}
	if _, err := os.Stat(journal); !os.IsNotExist(err) {
		t.Fatalf("opted out, but a journal was written; stat err = %v", err)
	}
}

// TestGuardSessionJournalID pins the fold-key rule. The transcript UUID wins because the trace
// an attended launch carries is the shared constant "guard"; the trace is the fallback for a
// resumed child (CLAUDE_CODE_SESSION_ID stripped), which does carry a real per-session id. With
// neither, nothing identifies the session and nothing is recorded — an anonymous row would fold
// every unidentifiable session together.
func TestGuardSessionJournalID(t *testing.T) {
	for _, c := range []struct{ name, uuid, trace, want string }{
		{"the transcript uuid wins over the constant attended trace", "uuid-1", "guard", "uuid-1"},
		{"a resumed child falls back to its real trace", "", "trace-xyz", "trace-xyz"},
		{"neither id identifies anything", "", "", ""},
		{"whitespace is not an id", "   ", "  ", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv("CLAUDE_CODE_SESSION_ID", c.uuid)
			if got := guardSessionJournalID(c.trace); got != c.want {
				t.Fatalf("guardSessionJournalID(%q) = %q, want %q", c.trace, got, c.want)
			}
		})
	}
}

// TestGuardSessionStartUnidentifiableSessionWritesNothing is the fail-open half: a hook that
// can name neither id writes no row at all and still exits 0.
func TestGuardSessionStartUnidentifiableSessionWritesNothing(t *testing.T) {
	journal, _ := stageBareInteractiveSessionStart(t, "")

	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if _, err := os.Stat(journal); !os.IsNotExist(err) {
		t.Fatalf("an unidentifiable session must write no row; stat err = %v", err)
	}
}

// TestBareInteractiveGuardInstallsSessionStartHook closes the wiring end of the DoD: the row
// above is only reachable if an ATTENDED `fak guard -- claude` installs the hook at all. The
// existing install tests all pass managed=true (the headless `-p` shape), so without this a
// regression that wired the hook for fleet workers only would keep the interactive case — the
// one the issue names — silently unrecorded.
func TestBareInteractiveGuardInstallsSessionStartHook(t *testing.T) {
	cmd := []string{"claude"} // no -p: an attended TUI
	if guardSessionStartManaged(cmd) {
		t.Fatalf("an attended `claude` must not be admitted MANAGED — the fixture is wrong")
	}
	out, install, err := installGuardSessionStartHookAt(cmd, "on", false, "fak", t.TempDir(), "", "guard")
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if !install.Applied {
		t.Fatalf("an attended claude child got no SessionStart hook: %+v", install)
	}
	raw, err := os.ReadFile(install.SettingsPath)
	if err != nil {
		t.Fatalf("read written settings: %v", err)
	}
	if !strings.Contains(string(raw), "guard-sessionstart") {
		t.Fatalf("attended settings do not invoke guard-sessionstart: %s", raw)
	}
	if !strings.Contains(strings.Join(out, " "), "--settings "+install.SettingsPath) {
		t.Fatalf("command missing --settings %s: %v", install.SettingsPath, out)
	}
}

// bootTimeForTest reports the host's current boot instant, or the zero time where the platform
// cannot name one.
func bootTimeForTest(t *testing.T) time.Time {
	t.Helper()
	bt, _ := sessionjournal.BootTime(time.Now().UTC())
	return bt
}
