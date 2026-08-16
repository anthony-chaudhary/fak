package main

// serve_durability_test.go — the acceptance witness for #1365: durable session state is
// ON BY DEFAULT and the durability posture is reported by `fak doctor`.
//
// The three acceptance boxes map to the three groups below:
//  1. a no-flag `fak serve` persists + restores by default, and the opt-out works;
//  2. `fak doctor serve` reports the posture (path, what is persisted, signals handled);
//  3. no regression for the explicit `--session-state FILE` form.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// noEnv is the injected environment for a host with no FAK_SESSION_STATE set.
func noEnv(string) string { return "" }

// envWith returns an env reader that answers exactly one variable, so the resolution
// arms are driven without mutating the real process environment.
func envWith(key, value string) func(string) string {
	return func(k string) string {
		if k == key {
			return value
		}
		return ""
	}
}

// TestServeSessionStateDefaultsOn is acceptance box 1's first half: with NO --session-state
// flag and no env override, `fak serve` resolves a real snapshot path rather than the
// pre-#1365 "" (= off). This is the flip the issue asks for.
func TestServeSessionStateDefaultsOn(t *testing.T) {
	p := resolveServeSessionState("", noEnv)
	if !p.Enabled {
		t.Fatal("no-flag serve resolved durability OFF — #1365 requires session state ON by default")
	}
	if strings.TrimSpace(p.Path) == "" {
		t.Fatal("durability enabled but no snapshot path resolved")
	}
	if p.Source != "default" {
		t.Errorf("source = %q, want %q", p.Source, "default")
	}
	if filepath.Base(p.Path) != serveSessionStateFile {
		t.Errorf("default path = %q, want a path ending in %q", p.Path, serveSessionStateFile)
	}
	if !filepath.IsAbs(p.Path) && !strings.HasPrefix(p.Path, ".fak") {
		t.Errorf("default path %q is neither under the user config dir nor the .fak fallback", p.Path)
	}
}

// TestServeSessionStateDefaultRoundTripsAcrossRestart is acceptance box 1's load-bearing
// half: the DEFAULT-resolved path really does persist and restore drive state across a
// simulated process restart. It drives the same two serve-lifecycle hooks the boot and
// shutdown stages call, with the resolved default path standing in for the flag value —
// which is exactly the wiring resolveSessionPlane installs.
func TestServeSessionStateDefaultRoundTripsAcrossRestart(t *testing.T) {
	// Point the default at a temp state dir so the test never writes the real user config
	// dir; the resolver reads os.UserConfigDir, whose var is per-OS, so pin the env knob
	// instead — it takes the same "a path relocates it" branch an operator would use.
	statePath := filepath.Join(t.TempDir(), "default-state.snap")
	p := resolveServeSessionState("", envWith(serveSessionStateEnv, statePath))
	if !p.Enabled || p.Path != statePath {
		t.Fatalf("env-relocated posture = {enabled:%v path:%q}, want {true %q}", p.Enabled, p.Path, statePath)
	}
	if p.Source != "env" {
		t.Errorf("source = %q, want %q", p.Source, "env")
	}

	// Boot 1: a session drives to a non-default state, then the process stops gracefully.
	first := session.NewTable()
	first.Transition("sess-default", session.Throttled, "operator-dial-down")
	first.SetBudget("sess-default", session.Budget{TurnsLeft: 5, TokensLeft: 2048})
	dumpServeSessions(first, p.Path)

	// Boot 2: a FRESH table restores through the same resolved path, with no flag passed.
	second := session.NewTable()
	if err := restoreServeSessions(second, p.Path); err != nil {
		t.Fatalf("restore across the default path: %v", err)
	}
	got := second.Get("sess-default")
	if got.Run != session.Throttled {
		t.Errorf("restored run-state = %v, want Throttled (default-on durability must survive a restart)", got.Run)
	}
	if got.Budget.TurnsLeft != 5 || got.Budget.TokensLeft != 2048 {
		t.Errorf("restored budget = %+v, want {TurnsLeft:5 TokensLeft:2048}", got.Budget)
	}
}

// TestServeSessionStateDumpCreatesMissingParent covers the first boot on a host where the
// per-user fak state dir does not exist yet: the default flush must create it rather than
// silently lose the snapshot to a missing parent.
func TestServeSessionStateDumpCreatesMissingParent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "never", "created", "state.snap")
	tbl := session.NewTable()
	tbl.Transition("sess-a", session.Paused, "")
	dumpServeSessions(tbl, path)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("dump into a missing parent dir did not write %s: %v", path, err)
	}
}

// TestServeSessionStateOptOut is acceptance box 1's documented opt-out: "off" at EITHER
// knob disables persistence and yields the empty path the restore/dump helpers treat as a
// no-op — byte-for-byte the pre-#1365 behavior.
func TestServeSessionStateOptOut(t *testing.T) {
	for _, tc := range []struct {
		name       string
		flag       string
		env        func(string) string
		wantSource string
	}{
		{"flag-off", serveSessionStateOff, noEnv, "flag"},
		{"flag-off-uppercase", "OFF", noEnv, "flag"},
		{"env-off", "", envWith(serveSessionStateEnv, serveSessionStateOff), "env"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := resolveServeSessionState(tc.flag, tc.env)
			if p.Enabled {
				t.Fatalf("%q did not opt out of durability", tc.name)
			}
			if p.Path != "" {
				t.Errorf("opted-out path = %q, want empty (the helpers' no-op contract)", p.Path)
			}
			if p.Source != tc.wantSource {
				t.Errorf("source = %q, want %q", p.Source, tc.wantSource)
			}
		})
	}
}

// TestServeSessionStateExplicitFileWins is acceptance box 3: the explicit
// `--session-state FILE` form is unchanged, and it beats both the env knob and the
// default — a flag an operator typed is never overridden by ambient configuration.
func TestServeSessionStateExplicitFileWins(t *testing.T) {
	explicit := filepath.Join(t.TempDir(), "explicit.snap")
	p := resolveServeSessionState(explicit, envWith(serveSessionStateEnv, serveSessionStateOff))
	if !p.Enabled {
		t.Fatal("an explicit --session-state FILE must not be disabled by the env opt-out")
	}
	if p.Path != explicit {
		t.Errorf("path = %q, want the explicit flag value %q", p.Path, explicit)
	}
	if p.Source != "flag" {
		t.Errorf("source = %q, want %q", p.Source, "flag")
	}
}

// TestServeDurabilityPostureNamesWhatIsPersisted checks the posture actually carries the
// three things #1365 asks the doctor to answer: where, what, and which signals.
func TestServeDurabilityPostureNamesWhatIsPersisted(t *testing.T) {
	p := resolveServeSessionState("", noEnv)
	if len(p.Persisted) == 0 {
		t.Error("posture names nothing as persisted")
	}
	if len(p.Signals) == 0 {
		t.Error("posture names no flush signals")
	}
	// The signal list must be the one the serve loop actually registers, never a wider
	// advertised set — a doctor promising a flush this build cannot deliver is worse
	// than a doctor that stays quiet.
	if got, want := len(p.Signals), len(terminatingSignals()); got != want {
		t.Errorf("posture reports %d flush signals, but the serve loop registers %d", got, want)
	}
	if strings.TrimSpace(p.Registry) == "" {
		t.Error("posture does not name the session descriptor registry")
	}
}

// TestServeDurabilityRowReportsPosture is acceptance box 2 at the row level: an enabled
// posture is a green row naming the path, and a disabled one is a yellow row carrying the
// re-enable action.
func TestServeDurabilityRowReportsPosture(t *testing.T) {
	on := serveDurabilityRow(resolveServeSessionState("/tmp/fak-state.snap", noEnv))
	if on.Check != "session-durability" {
		t.Fatalf("check = %q, want session-durability", on.Check)
	}
	if on.Status != sevOK || on.Tier != "Ready" {
		t.Errorf("enabled row = {status:%q tier:%q}, want {ok Ready}", on.Status, on.Tier)
	}
	if !strings.Contains(on.Finding, "/tmp/fak-state.snap") {
		t.Errorf("enabled finding does not name the state path: %q", on.Finding)
	}

	off := serveDurabilityRow(resolveServeSessionState(serveSessionStateOff, noEnv))
	if off.Status != sevWarn {
		t.Errorf("disabled row status = %q, want warn", off.Status)
	}
	if off.Remediation == "" {
		t.Error("disabled row carries no remediation — an operator cannot act on it")
	}
}

// TestRunServeDoctorReportsDurability is acceptance box 2 end to end: `fak doctor serve
// --json` carries the durability posture and its readiness row, and the row's presence
// does not turn a healthy host red.
func TestRunServeDoctorReportsDurability(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runServeDoctor(&out, &errb, []string{"--json"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	var rep serveReadinessReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\noutput=%s", err, out.String())
	}
	if rep.Durability == nil {
		t.Fatal("doctor report carries no durability posture (#1365 acceptance)")
	}
	var row serveReadinessRow
	for _, r := range rep.Rows {
		if r.Check == "session-durability" {
			row = r
		}
	}
	if row.Check == "" {
		t.Fatal("no session-durability row in the doctor table")
	}
	if row.Finding == "" {
		t.Error("session-durability row has an empty finding")
	}
}

// TestWriteServeDurabilityBannerStatesPosture is the boot-output half of acceptance box 2:
// the banner names the path and the opt-out when on, and says so loudly when off.
func TestWriteServeDurabilityBannerStatesPosture(t *testing.T) {
	var on bytes.Buffer
	writeServeDurabilityBanner(&on, resolveServeSessionState("/tmp/fak-state.snap", noEnv))
	for _, want := range []string{"ON", "/tmp/fak-state.snap", serveSessionStateEnv} {
		if !strings.Contains(on.String(), want) {
			t.Errorf("enabled banner missing %q: %s", want, on.String())
		}
	}

	var off bytes.Buffer
	writeServeDurabilityBanner(&off, resolveServeSessionState(serveSessionStateOff, noEnv))
	if !strings.Contains(off.String(), "OFF") {
		t.Errorf("disabled banner does not state the OFF posture: %s", off.String())
	}
}
