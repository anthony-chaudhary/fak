package main

import (
	"bytes"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// TestStripCoreLockAllFlagPresent: the flag is detected and removed from argv.
func TestStripCoreLockAllFlagPresent(t *testing.T) {
	core, rest := stripCoreLockAllFlag([]string{"--quiet", "--core-lock-all", "--policy", "p.json"})
	if !core {
		t.Fatal("--core-lock-all must be detected")
	}
	if want := []string{"--quiet", "--policy", "p.json"}; !reflect.DeepEqual(rest, want) {
		t.Fatalf("flag not stripped cleanly: got %v want %v", rest, want)
	}
}

// TestStripCoreLockAllFlagAbsent: without the flag, core is false and argv is
// returned unchanged.
func TestStripCoreLockAllFlagAbsent(t *testing.T) {
	core, rest := stripCoreLockAllFlag([]string{"--quiet"})
	if core {
		t.Fatal("must not report core-lock when flag absent")
	}
	if want := []string{"--quiet"}; !reflect.DeepEqual(rest, want) {
		t.Fatalf("argv changed unexpectedly: %v", rest)
	}
}

// TestCoreLockAllRefusesWiden: under core-lock-all, an added Allow (widening) is
// refused.
func TestCoreLockAllRefusesWiden(t *testing.T) {
	cur := adjudicator.Policy{Allow: map[string]bool{"read_file": true}}
	proposed := adjudicator.Policy{Allow: map[string]bool{"read_file": true, "write_file": true}}
	admit, reason := coreLockAllReloadVerdict(true, cur, proposed)
	if admit {
		t.Fatalf("core-lock-all must refuse a widening reload (%s)", reason)
	}
}

// TestCoreLockAllAdmitsTighten: under core-lock-all, an added Deny (tightening)
// is admitted.
func TestCoreLockAllAdmitsTighten(t *testing.T) {
	cur := adjudicator.Policy{}
	proposed := adjudicator.Policy{Deny: map[string]abi.ReasonCode{"dangerous_tool": abi.ReasonPolicyBlock}}
	admit, reason := coreLockAllReloadVerdict(true, cur, proposed)
	if !admit {
		t.Fatalf("core-lock-all must admit a tighten-only reload (%s)", reason)
	}
}

// TestCoreLockAllInactiveAdmitsWiden: when the mode is off, even a widening
// reload is admitted here (normal gating applies elsewhere).
func TestCoreLockAllInactiveAdmitsWiden(t *testing.T) {
	cur := adjudicator.Policy{Allow: map[string]bool{"read_file": true}}
	proposed := adjudicator.Policy{Allow: map[string]bool{"read_file": true, "write_file": true}}
	if admit, _ := coreLockAllReloadVerdict(false, cur, proposed); !admit {
		t.Fatal("inactive core-lock-all must not refuse anything")
	}
}

// ---------------------------------------------------------------------------
// #5423 — the WIRING. Everything above this line passes just as well when the
// posture is dead code: a pure verdict function classifies correctly whether or
// not a live session can ever reach it, which is exactly the confusion this
// file's own header used to declare. The tests below drive the two ends.
// ---------------------------------------------------------------------------

// withCoreLockAll sets the session posture for one test and restores whatever it
// was, so an ordinary package test run is unaffected by the ones that clamp.
func withCoreLockAll(t *testing.T, active bool) {
	t.Helper()
	prev := guardCoreLockAllActive()
	setGuardCoreLockAll(active)
	t.Cleanup(func() { setGuardCoreLockAll(prev) })
}

// withLiveFloor installs p as the process-wide adjudicator floor for one test and
// restores the previous snapshot afterwards. The amendment sites read the live
// floor as the "current" side of their delta, so a test cannot drive them without
// standing one up.
func withLiveFloor(t *testing.T, p adjudicator.Policy) {
	t.Helper()
	prev := adjudicator.Default.PolicySnapshot()
	adjudicator.Default.SetPolicy(p)
	t.Cleanup(func() { adjudicator.Default.SetPolicy(prev) })
}

// TestGuardLaunchCoreLockAllPeelsOnlyGuardSideArgv pins the `--` boundary. The
// posture flag must come off the GUARD side of the launch line; an occurrence
// after `--` belongs to the wrapped child and must survive byte for byte, or
// guard would be silently editing the argv of the program it wraps.
func TestGuardLaunchCoreLockAllPeelsOnlyGuardSideArgv(t *testing.T) {
	active, rest := guardLaunchCoreLockAll([]string{"--quiet", "--core-lock-all", "--", "claude", "--core-lock-all"})
	if !active {
		t.Fatal("--core-lock-all before -- must arm the session posture")
	}
	want := []string{"--quiet", "--", "claude", "--core-lock-all"}
	if !reflect.DeepEqual(rest, want) {
		t.Fatalf("child argv was edited: got %v want %v", rest, want)
	}

	// The mirror case: the flag ONLY on the child side arms nothing and changes
	// nothing. A peel that ignored `--` would arm the posture here, which would
	// hand the wrapped agent a way to clamp its own operator's session.
	active, rest = guardLaunchCoreLockAll([]string{"--", "claude", "--core-lock-all"})
	if active {
		t.Fatal("a --core-lock-all that belongs to the wrapped child must NOT arm the guard posture")
	}
	if want := []string{"--", "claude", "--core-lock-all"}; !reflect.DeepEqual(rest, want) {
		t.Fatalf("child argv was edited: got %v want %v", rest, want)
	}
}

// TestGuardCoreLockAllIsPeeledBeforeTheFlagSetParse is the launch-end wiring
// witness, written against cmdGuard's own source bytes so it can be run at the
// pre-fix commit (where it FAILS: nothing named the helper outside the tests).
//
// The ORDER is the load-bearing part, not merely the call. cmdGuard's FlagSet is
// flag.ExitOnError, so if --core-lock-all ever reached fs.Parse the launch would
// abort with "flag provided but not defined" — the posture has to be peeled
// before the FlagSet is even constructed.
func TestGuardCoreLockAllIsPeeledBeforeTheFlagSetParse(t *testing.T) {
	src, err := os.ReadFile("guard.go")
	if err != nil {
		t.Fatal(err)
	}
	at := bytes.Index(src, []byte("func cmdGuard("))
	if at < 0 {
		t.Fatal("cmd/fak/guard.go no longer defines cmdGuard")
	}
	body := src[at:]
	peelAt := bytes.Index(body, []byte("guardLaunchCoreLockAll(argv)"))
	if peelAt < 0 {
		t.Fatal("cmdGuard never peels --core-lock-all: the posture has no launch mode, so no session can be clamped (#5423)")
	}
	setAt := bytes.Index(body, []byte("setGuardCoreLockAll("))
	if setAt < 0 {
		t.Fatal("cmdGuard peels the flag but never records the posture, so every amendment site reads false (#5423)")
	}
	parseAt := bytes.Index(body, []byte("flag.NewFlagSet(\"guard\""))
	if parseAt < 0 {
		t.Fatal("cmdGuard no longer builds its guard FlagSet")
	}
	if peelAt > parseAt || setAt > parseAt {
		t.Fatal("--core-lock-all must be peeled and recorded BEFORE the ExitOnError FlagSet is built, or the launch aborts on an undefined flag")
	}
}

// TestGuardCoreLockAllVerdictHasALiveCaller is the dead-code witness, the sibling
// of TestGuardSelfTightenGateHasALiveCaller (#5411). At the pre-fix commit the
// only files naming the verdict were guard_core_lock.go and its own test, so the
// posture refused nothing however correct its classification was.
func TestGuardCoreLockAllVerdictHasALiveCaller(t *testing.T) {
	live := make([]string, 0, 2)
	for _, f := range guardNonTestFilesNaming(t, "guardCoreLockAllAdmitAmendment(") {
		if f != "guard_core_lock.go" { // the definition itself proves nothing
			live = append(live, f)
		}
	}
	if len(live) == 0 {
		t.Fatal("guardCoreLockAllAdmitAmendment has NO non-test caller: the core-lock-all posture is inert — no live amendment is refused by it (#5423)")
	}
	t.Logf("core-lock-all posture is consulted from: %s", strings.Join(live, ", "))
}

// TestGuardCoreLockAllRefusesAWideningPolicyReload drives the --policy FILE
// reload seam end to end and asserts the LIVE floor is unchanged afterwards.
//
// It deliberately passes enforceWideningGate=false — the branch where the
// pre-existing widening gate does NOT run — so the assertion cannot be satisfied
// by that older gate. At the pre-fix commit this call installed the widening
// silently; the only thing that can refuse it here is the core-lock posture.
func TestGuardCoreLockAllRefusesAWideningPolicyReload(t *testing.T) {
	withCoreLockAll(t, true)
	withLiveFloor(t, adjudicator.Policy{Allow: map[string]bool{"read_file": true}})

	widened := policy.Runtime{Adjudicator: adjudicator.Policy{
		Allow: map[string]bool{"read_file": true, "bash": true},
	}}
	_, err := applyPolicyRuntimeLocked(widened, "test-floor.json", "sha256:test", "", false)
	if err == nil {
		t.Fatal("core-lock-all admitted a WIDENING policy reload: the posture is not enforced on the --policy reload path (#5423)")
	}
	if !strings.Contains(err.Error(), "core-lock-all") {
		t.Fatalf("refusal must NAME the posture that produced it, got: %v", err)
	}
	if live := adjudicator.Default.PolicySnapshot(); live.Allow["bash"] {
		t.Fatal("the refused floor was installed anyway: the gate returned an error but the widening still took effect")
	}
}

// TestGuardCoreLockAllInactiveStillInstallsAWideningPolicyReload is the control
// that keeps the test above honest. Without the posture the SAME call must still
// install the SAME widening — otherwise the test above would pass just as well if
// the gate refused everything unconditionally, which would be a behaviour change
// for every launch rather than the opt-in posture #5423 asks for.
func TestGuardCoreLockAllInactiveStillInstallsAWideningPolicyReload(t *testing.T) {
	withCoreLockAll(t, false)
	withLiveFloor(t, adjudicator.Policy{Allow: map[string]bool{"read_file": true}})

	widened := policy.Runtime{Adjudicator: adjudicator.Policy{
		Allow: map[string]bool{"read_file": true, "bash": true},
	}}
	if _, err := applyPolicyRuntimeLocked(widened, "test-floor.json", "sha256:test", "", false); err != nil {
		t.Fatalf("an ordinary (no --core-lock-all) launch must be unaffected, got: %v", err)
	}
	if live := adjudicator.Default.PolicySnapshot(); !live.Allow["bash"] {
		t.Fatal("the widening was not installed on an ordinary launch: #5423 changed behaviour outside its opt-in posture")
	}
}

// TestGuardCoreLockAllRefusesAWideningDefaultFloorReload drives the OTHER seam:
// the built-in-floor reload that the allow watcher and POST /v1/fak/policy/reload
// take on an ordinary `fak guard -- claude`. That path had no widening gate at
// all before #5423 — it called SetPolicy unconditionally — so this is the one
// that carries the most weight.
//
// The setup makes the re-derived floor a genuine widening: the live floor is the
// shipped built-in floor plus one extra Deny, so re-deriving the built-in floor
// REMOVES that deny, which policy.DiffAmendment classifies as widening.
func TestGuardCoreLockAllRefusesAWideningDefaultFloorReload(t *testing.T) {
	base, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	tighter := base.Adjudicator
	deny := make(map[string]abi.ReasonCode, len(tighter.Deny)+1)
	for tool, reason := range tighter.Deny {
		deny[tool] = reason
	}
	const extra = "fak_core_lock_test_only_tool"
	deny[extra] = abi.ReasonPolicyBlock
	tighter.Deny = deny

	withCoreLockAll(t, true)
	withLiveFloor(t, tighter)

	if _, _, err := guardReloadDefaultFloor(); err == nil {
		t.Fatal("core-lock-all admitted a reload that DROPS a deny: the built-in-floor reload path is not clamped (#5423)")
	} else if !strings.Contains(err.Error(), "core-lock-all") {
		t.Fatalf("refusal must NAME the posture that produced it, got: %v", err)
	}
	if live := adjudicator.Default.PolicySnapshot(); live.Deny[extra] != abi.ReasonPolicyBlock {
		t.Fatal("the refused reload dropped the deny anyway: the last-good floor did not stand")
	}
}
