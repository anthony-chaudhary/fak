package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ablate"
)

// smokeTraceArg is the repo-root frozen trace, addressed relative to this package
// dir (cmd/fak) — `go test` runs with CWD at the package, so the default
// testdata/tau2 lookup (relative to CWD) does not resolve here; pass it explicitly.
const smokeTraceArg = "../../testdata/tau2/tau2-smoke.json"

// runAB drives the testable runAblate core with captured streams, always pinning the
// explicit smoke trace so the run is independent of CWD.
func runAB(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	full := append([]string{"--trace", smokeTraceArg}, args...)
	code := runAblate(&out, &errb, full)
	return code, out.String(), errb.String()
}

// The built-in trace + a vDSO sweep prints a two-arm table with the per-arm delta.
func TestAblateTableTwoArms(t *testing.T) {
	code, out, errb := runAB("--sweep", "vdso")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	for _, want := range []string{"fak ablate", "workload hash", "all-off", "vdso", "deltas vs all-off"} {
		if !strings.Contains(out, want) {
			t.Fatalf("table missing %q:\n%s", want, out)
		}
	}
}

// --json emits a valid AblationReport whose arms share one workload hash and whose
// vdso arm served the fast path the all-off arm did not.
func TestAblateJSONReport(t *testing.T) {
	code, out, errb := runAB("--sweep", "vdso", "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var rep struct {
		WorkloadHash string `json:"workload_hash"`
		Baseline     string `json:"baseline_arm"`
		Runs         []struct {
			ArmID        string            `json:"arm_id"`
			Features     map[string]string `json:"features"`
			WorkloadHash string            `json:"workload_hash"`
			Arm          struct {
				VDSOHits int64 `json:"vdso_hits"`
			} `json:"arm"`
		} `json:"runs"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if len(rep.Runs) != 2 {
		t.Fatalf("want 2 arms, got %d", len(rep.Runs))
	}
	var offHits, onHits int64 = -1, -1
	for _, r := range rep.Runs {
		if r.WorkloadHash != rep.WorkloadHash {
			t.Errorf("arm %q workload hash %q != report %q", r.ArmID, r.WorkloadHash, rep.WorkloadHash)
		}
		switch r.ArmID {
		case "all-off":
			offHits = r.Arm.VDSOHits
		case "vdso":
			onHits = r.Arm.VDSOHits
		}
	}
	if offHits != 0 {
		t.Errorf("all-off vdso_hits = %d, want 0", offHits)
	}
	if onHits <= offHits {
		t.Errorf("vdso arm hits (%d) must exceed all-off (%d)", onHits, offHits)
	}
}

// A token in NO rung — neither the runtime vdso knob nor an env-gated feature — fails
// loud (usage exit 2). An env-gated feature is now accepted (routed to the subprocess
// rung), so the fail-loud case is a genuinely-unknown token.
func TestAblateUnknownFeatureUsageError(t *testing.T) {
	code, _, errb := runAB("--sweep", "bogus")
	if code != 2 {
		t.Fatalf("exit=%d, want 2 (usage error)", code)
	}
	if !strings.Contains(errb, "unknown runtime feature") {
		t.Fatalf("stderr missing the fail-loud reason:\n%s", errb)
	}
}

// fakeAblateArmRunner runs each arm IN-PROCESS (no spawn), reconstructing the trace exactly
// as SweepViaSubprocess canonicalizes it (via UnmarshalTrace) so the child's workload hash
// matches the parent's and no arm is dropped. It is the cmd analogue of the ablate package
// test's fake: it exercises the routing + report assembly of the env-gated path without the
// real re-exec.
func fakeAblateArmRunner(ctx context.Context, bin string, c ablate.FeatureConfig, traceJSON []byte) (ablate.AblationRun, error) {
	tr, err := ablate.UnmarshalTrace(traceJSON)
	if err != nil {
		return ablate.AblationRun{}, err
	}
	return ablate.RunOneArm(ctx, tr, "mock", c)
}

// withFakeArmRunner swaps the injected production re-exec for the in-process fake for the
// duration of a test, so the cmd suite never spawns the real fak binary.
func withFakeArmRunner(t *testing.T) {
	t.Helper()
	orig := ablateArmRunner
	t.Cleanup(func() { ablateArmRunner = orig })
	ablateArmRunner = fakeAblateArmRunner
}

// Acceptance #1: an all-env-gated sweep is no longer rejected — it runs every arm through
// the subprocess rung and emits one valid AblationReport whose arms share a workload hash.
func TestAblateEnvGatedSweepRunsAllArms(t *testing.T) {
	withFakeArmRunner(t)
	code, out, errb := runAB("--sweep",
		"normgate,radix,compressor,ifc,gitgate,ctxplan_seam,wire_screen,wire_redact", "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var rep struct {
		WorkloadHash string `json:"workload_hash"`
		Runs         []struct {
			ArmID        string `json:"arm_id"`
			WorkloadHash string `json:"workload_hash"`
		} `json:"runs"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	// all-off + one arm per feature (8) + all-on = 10 arms.
	if len(rep.Runs) != 10 {
		t.Fatalf("want 10 arms (all-off + 8 single + all-on), got %d", len(rep.Runs))
	}
	if rep.WorkloadHash == "" {
		t.Fatal("report workload hash is empty")
	}
	for _, r := range rep.Runs {
		if r.WorkloadHash != rep.WorkloadHash {
			t.Errorf("arm %q workload hash %q != report %q (identical-workload guard)", r.ArmID, r.WorkloadHash, rep.WorkloadHash)
		}
	}
}

// Acceptance #4: a mixed vdso,normgate sweep merges the vdso arm (vdso applied in-process
// within its child) and the normgate arm (env-gated, via the subprocess rung) under one
// report and one workload hash.
func TestAblateMixedSweepMergesVdsoAndEnvArm(t *testing.T) {
	withFakeArmRunner(t)
	code, out, errb := runAB("--sweep", "vdso,normgate", "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var rep struct {
		WorkloadHash string `json:"workload_hash"`
		Runs         []struct {
			ArmID    string            `json:"arm_id"`
			Features map[string]string `json:"features"`
		} `json:"runs"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	// all-off, vdso, normgate, all-on.
	if len(rep.Runs) != 4 {
		t.Fatalf("want 4 arms, got %d", len(rep.Runs))
	}
	byID := map[string]map[string]string{}
	for _, r := range rep.Runs {
		byID[r.ArmID] = r.Features
	}
	vdso, okV := byID["vdso"]
	norm, okN := byID["normgate"]
	if !okV || !okN {
		t.Fatalf("expected both a vdso and a normgate arm, got arms %v", keysOf(byID))
	}
	if vdso["vdso"] != "on" || vdso["normgate"] != "off" {
		t.Errorf("vdso arm descriptor = %v, want vdso=on normgate=off", vdso)
	}
	if norm["normgate"] != "on" || norm["vdso"] != "off" {
		t.Errorf("normgate arm descriptor = %v, want normgate=on vdso=off", norm)
	}
}

// keysOf lists a map's keys (a small test helper for a readable failure message).
func keysOf(m map[string]map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
