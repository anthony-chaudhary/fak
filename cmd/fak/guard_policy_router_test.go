package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// The reachability witness for `fak guard policy` (#5424).
//
// The point of this ticket is that a human typing the real command gets the
// report, so these tests drive the REAL dispatch path — operator argv into
// cmdGuard, exit code + rendered report out. Calling runGuardPolicyDiff or
// policyDiffReport directly would witness the renderer and prove nothing about
// reachability, which is exactly the defect the ticket exists to close. Because
// cmdGuard owns os.Exit, the only way to observe the exit code is to re-exec this
// test binary as `fak guard policy …` — the same subprocess pattern
// TestGuardSessionsDispatch and TestGuardHelpHelperProcess already use in this
// package.
//
// Negative witness: delete the `policy` peel in cmdGuard (guard.go) and every
// dispatch test below goes red — argv falls through to the wrap-a-command parser
// and no report is ever rendered. Delete the `diff` row from guardPolicyVerbs
// (guard_policy.go) and the diff tests red with the router's own unknown-verb
// usage error.

// guardPolicyDispatchEnv carries the JSON-packed argv from the parent test to the
// re-exec'd child. Test-harness plumbing only: it selects nothing about product
// behavior, it just tells the child process which argv to replay.
const guardPolicyDispatchEnv = "FAK_TEST_GUARD_POLICY_DISPATCH_ARGV"

// TestGuardPolicyDispatchHelperProcess is the re-exec target: with the argv env
// set it calls the real cmdGuard, so the whole peel → router → report path runs
// in one process and its exit code is observable. Run bare (the normal `go test`
// pass) it instead asserts the verb table's own invariants, so it is never a
// silent no-op.
func TestGuardPolicyDispatchHelperProcess(t *testing.T) {
	raw := os.Getenv(guardPolicyDispatchEnv)
	if raw == "" {
		seen := map[string]bool{}
		for _, v := range guardPolicyVerbs {
			if v.Name == "" || v.Blurb == "" || v.Run == nil {
				t.Fatalf("guardPolicyVerbs row %+v is missing a name, blurb, or handler", v)
			}
			if seen[v.Name] {
				t.Fatalf("guardPolicyVerbs registers %q twice", v.Name)
			}
			seen[v.Name] = true
		}
		for _, want := range []string{"explain", "diff"} {
			if !seen[want] {
				t.Fatalf("guardPolicyVerbs is missing the %q verb; registered: %v", want, guardPolicyVerbNames())
			}
		}
		return
	}
	var argv []string
	if err := json.Unmarshal([]byte(raw), &argv); err != nil {
		t.Fatalf("unpack %s=%q: %v", guardPolicyDispatchEnv, raw, err)
	}
	cmdGuard(argv)
}

// runGuardPolicyCLI re-execs the test binary as `fak guard <argv...>` and returns
// the process exit code plus combined stdout+stderr.
func runGuardPolicyCLI(t *testing.T, argv ...string) (int, string) {
	t.Helper()
	packed, err := json.Marshal(argv)
	if err != nil {
		t.Fatalf("pack argv %v: %v", argv, err)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestGuardPolicyDispatchHelperProcess$")
	cmd.Env = append(os.Environ(), guardPolicyDispatchEnv+"="+string(packed))
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()
	code := 0
	if runErr != nil {
		exitErr, ok := runErr.(*exec.ExitError)
		if !ok {
			t.Fatalf("re-exec `fak guard %s`: %v\n%s", strings.Join(argv, " "), runErr, out.String())
		}
		code = exitErr.ExitCode()
	}
	return code, out.String()
}

// writeWidenedFloor writes a manifest that is the SHIPPED floor plus one extra
// allowed tool, and returns its path. That single added Allow entry is the
// deliberate widening the drift report must catch — it is classified by the same
// policy.DiffAmendment registry that governs admission, so the fixture cannot
// drift from the engine's own notion of "loosened".
func writeWidenedFloor(t *testing.T) string {
	t.Helper()
	m, err := policy.ParseManifest(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("parse embedded floor: %v", err)
	}
	m.Allow = append(m.Allow, "fak_test_widened_tool")
	body, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal widened floor: %v", err)
	}
	if _, err := policy.ParseManifest(body); err != nil {
		t.Fatalf("widened floor does not round-trip: %v", err)
	}
	path := filepath.Join(t.TempDir(), "widened-floor.json")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatalf("write widened floor: %v", err)
	}
	return path
}

// TestGuardPolicyDiffDispatchExitsOneOnWidenedFloor is the ticket's checkable
// witness: `fak guard policy diff --policy <widened>` typed as argv must reach
// the report and exit 1, naming the widening.
func TestGuardPolicyDiffDispatchExitsOneOnWidenedFloor(t *testing.T) {
	floor := writeWidenedFloor(t)
	code, out := runGuardPolicyCLI(t, "policy", "diff", "--policy", floor)
	if code != guardPolicyExitFlagged {
		t.Fatalf("`fak guard policy diff --policy <widened>` exit = %d, want %d (widen-drift is gate-able)\n%s", code, guardPolicyExitFlagged, out)
	}
	if !strings.Contains(out, "widen-drift from the shipped floor") {
		t.Fatalf("dispatch did not reach the drift report header:\n%s", out)
	}
	if !strings.Contains(out, "WIDENED") || !strings.Contains(out, "fak_test_widened_tool") {
		t.Fatalf("drift report did not name the deliberate widening:\n%s", out)
	}
}

// TestGuardPolicyExplainDispatch proves the sibling verb is reachable through the
// same table: argv in, the amendment-class report out, exit 0.
func TestGuardPolicyExplainDispatch(t *testing.T) {
	code, out := runGuardPolicyCLI(t, "policy", "explain")
	if code != guardPolicyExitOK {
		t.Fatalf("`fak guard policy explain` exit = %d, want %d\n%s", code, guardPolicyExitOK, out)
	}
	for _, want := range []string{"effective guard floor by amendment class", policyClassFrozen, policyClassGatedWiden} {
		if !strings.Contains(out, want) {
			t.Fatalf("explain output missing %q:\n%s", want, out)
		}
	}
}

// TestGuardPolicyUnknownVerbDispatch pins the router's refusal shape — the same
// error the negative witness produces when a verb row is deleted.
func TestGuardPolicyUnknownVerbDispatch(t *testing.T) {
	code, out := runGuardPolicyCLI(t, "policy", "no-such-verb")
	if code != guardPolicyExitUsage {
		t.Fatalf("unknown verb exit = %d, want %d\n%s", code, guardPolicyExitUsage, out)
	}
	if !strings.Contains(out, "unknown verb") || !strings.Contains(out, "explain") || !strings.Contains(out, "diff") {
		t.Fatalf("unknown-verb refusal must name the known verbs:\n%s", out)
	}
}

// TestGuardPolicyBareDispatchIsUsage proves `fak guard policy` with no verb is a
// usage error rather than a fall-through into the wrap-a-command parser (which is
// what the missing-router bug looked like from the terminal).
func TestGuardPolicyBareDispatchIsUsage(t *testing.T) {
	code, out := runGuardPolicyCLI(t, "policy")
	if code != guardPolicyExitUsage {
		t.Fatalf("bare `fak guard policy` exit = %d, want %d\n%s", code, guardPolicyExitUsage, out)
	}
	if !strings.Contains(out, "usage: fak guard policy <verb>") {
		t.Fatalf("bare `fak guard policy` must print the verb table:\n%s", out)
	}
}

// TestGuardPolicyIsAdvertisedInGuardHelp is the help gate: a subcommand an
// operator cannot discover from `fak guard -h` is only half-reachable.
func TestGuardPolicyIsAdvertisedInGuardHelp(t *testing.T) {
	out := runGuardHelp(t, "-h")
	if !strings.Contains(out, "policy explain|diff") {
		t.Fatalf("`fak guard -h` does not advertise the policy subcommand:\n%s", out)
	}
}
