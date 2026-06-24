package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
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

// An env-gated feature is not a runtime knob this rung sweeps: fail loud (usage exit 2).
func TestAblateUnknownFeatureUsageError(t *testing.T) {
	code, _, errb := runAB("--sweep", "normgate")
	if code != 2 {
		t.Fatalf("exit=%d, want 2 (usage error)", code)
	}
	if !strings.Contains(errb, "unknown runtime feature") {
		t.Fatalf("stderr missing the fail-loud reason:\n%s", errb)
	}
}
