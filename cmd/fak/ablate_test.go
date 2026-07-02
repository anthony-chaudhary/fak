package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
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
	for _, want := range []string{"fak ablate", "workload hash", "all-off", "vdso", "provider_tokeq", "fak_tokeq", "prefix_mismatch", "deltas vs all-off"} {
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
			MechanismSavings struct {
				FakVDSOAvoidedCalls uint64 `json:"fak_vdso_avoided_calls"`
			} `json:"mechanism_savings"`
			PrefixIntegrity struct {
				Checked        bool   `json:"checked"`
				PrefixMismatch uint64 `json:"prefix_mismatch"`
			} `json:"prefix_integrity"`
			ProviderTokenEquiv float64 `json:"provider_tokeq"`
			FakTokenEquiv      float64 `json:"fak_tokeq"`
		} `json:"runs"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if len(rep.Runs) != 2 {
		t.Fatalf("want 2 arms, got %d", len(rep.Runs))
	}
	var offHits, onHits int64 = -1, -1
	var offAvoided, onAvoided uint64
	for _, r := range rep.Runs {
		if r.WorkloadHash != rep.WorkloadHash {
			t.Errorf("arm %q workload hash %q != report %q", r.ArmID, r.WorkloadHash, rep.WorkloadHash)
		}
		switch r.ArmID {
		case "all-off":
			offHits = r.Arm.VDSOHits
			offAvoided = r.MechanismSavings.FakVDSOAvoidedCalls
		case "vdso":
			onHits = r.Arm.VDSOHits
			onAvoided = r.MechanismSavings.FakVDSOAvoidedCalls
		}
	}
	if offHits != 0 {
		t.Errorf("all-off vdso_hits = %d, want 0", offHits)
	}
	if onHits <= offHits {
		t.Errorf("vdso arm hits (%d) must exceed all-off (%d)", onHits, offHits)
	}
	if onAvoided <= offAvoided {
		t.Errorf("vdso arm avoided calls (%d) must exceed all-off (%d)", onAvoided, offAvoided)
	}
}

func TestAblateWireCacheLeversReportFakDeltaAndPrefixIntegrity(t *testing.T) {
	withFakeArmRunner(t)
	code, out, errb := runAB("--sweep", "ttl_1h,uncached_trim", "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var rep struct {
		Runs []struct {
			ArmID           string             `json:"arm_id"`
			Features        map[string]string  `json:"features"`
			ProviderTokenEq float64            `json:"provider_tokeq"`
			FakTokenEq      float64            `json:"fak_tokeq"`
			PrefixIntegrity ablatePrefixShadow `json:"prefix_integrity"`
		} `json:"runs"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	byID := map[string]struct {
		ArmID           string             `json:"arm_id"`
		Features        map[string]string  `json:"features"`
		ProviderTokenEq float64            `json:"provider_tokeq"`
		FakTokenEq      float64            `json:"fak_tokeq"`
		PrefixIntegrity ablatePrefixShadow `json:"prefix_integrity"`
	}{}
	for _, r := range rep.Runs {
		byID[r.ArmID] = r
	}
	for _, id := range []string{"ttl_1h", "uncached_trim"} {
		r, ok := byID[id]
		if !ok {
			t.Fatalf("missing %s arm in %+v", id, byID)
		}
		if r.Features[id] != "on" {
			t.Fatalf("%s descriptor = %v, want %s=on", id, r.Features, id)
		}
		if r.ProviderTokenEq != 0 {
			t.Fatalf("%s provider_tokeq = %v, want 0 in mock wire-lever arm", id, r.ProviderTokenEq)
		}
		if r.FakTokenEq <= 0 {
			t.Fatalf("%s fak_tokeq = %v, want > 0", id, r.FakTokenEq)
		}
		if !r.PrefixIntegrity.Checked || r.PrefixIntegrity.PrefixMismatch != 0 {
			t.Fatalf("%s prefix integrity = %+v, want checked with prefix_mismatch=0", id, r.PrefixIntegrity)
		}
	}
}

type ablatePrefixShadow struct {
	Checked        bool   `json:"checked"`
	PrefixMismatch uint64 `json:"prefix_mismatch"`
}

func TestAblateCompressorReportsFakTokenEquivOnly(t *testing.T) {
	withFakeArmRunner(t)
	code, out, errb := runAB("--sweep", "compressor", "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var rep struct {
		Runs []struct {
			ArmID            string  `json:"arm_id"`
			ProviderTokenEq  float64 `json:"provider_tokeq"`
			FakTokenEq       float64 `json:"fak_tokeq"`
			MechanismSavings struct {
				FakCompactionShedTokens uint64 `json:"fak_compaction_shed_tokens"`
			} `json:"mechanism_savings"`
		} `json:"runs"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	var off, on *struct {
		ArmID            string  `json:"arm_id"`
		ProviderTokenEq  float64 `json:"provider_tokeq"`
		FakTokenEq       float64 `json:"fak_tokeq"`
		MechanismSavings struct {
			FakCompactionShedTokens uint64 `json:"fak_compaction_shed_tokens"`
		} `json:"mechanism_savings"`
	}
	for i := range rep.Runs {
		switch rep.Runs[i].ArmID {
		case "all-off":
			off = &rep.Runs[i]
		case "compressor":
			on = &rep.Runs[i]
		}
	}
	if off == nil || on == nil {
		t.Fatalf("expected all-off and compressor arms, got %+v", rep.Runs)
	}
	if off.FakTokenEq != 0 || off.ProviderTokenEq != 0 {
		t.Fatalf("all-off token-equiv = provider %v fak %v, want 0/0", off.ProviderTokenEq, off.FakTokenEq)
	}
	if on.ProviderTokenEq != 0 {
		t.Fatalf("compressor provider_tokeq = %v, want 0", on.ProviderTokenEq)
	}
	if on.FakTokenEq <= 0 || on.MechanismSavings.FakCompactionShedTokens == 0 {
		t.Fatalf("compressor fak split did not move: fak_tokeq=%v mechanisms=%+v", on.FakTokenEq, on.MechanismSavings)
	}
}

func TestAblateCompressorTableShowsOwnerDeltas(t *testing.T) {
	withFakeArmRunner(t)
	code, out, errb := runAB("--sweep", "compressor")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	for _, want := range []string{"provider_tokeq", "fak_tokeq", "prefix_mismatch", "compressor", "provider_tokeq 0", "fak_tokeq +", "prefix_mismatch +0"} {
		if !strings.Contains(out, want) {
			t.Fatalf("table missing %q:\n%s", want, out)
		}
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

func TestAblateFromSessionUsesCapturedUsage(t *testing.T) {
	path := writeAblateSessionFixture(t, `{
  "slice_id": "cmd-session",
  "turns": [{
    "usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 10},
    "calls": [{"tool": "calculate", "args": {"a": 1, "b": 2}, "class": "allow"}]
  }]
}`)
	var out, errb bytes.Buffer
	code := runAblate(&out, &errb, []string{"--from-session", path, "--sweep", "vdso", "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var rep struct {
		Provenance struct {
			EngineModel string `json:"engine_model"`
			SliceID     string `json:"slice_id"`
		} `json:"provenance"`
		Runs []struct {
			ArmID string `json:"arm_id"`
			Arm   struct {
				InTokens                    int64 `json:"input_tokens"`
				OutTokens                   int64 `json:"output_tokens"`
				ProviderCacheReadTokens     int64 `json:"provider_cache_read_tokens"`
				ProviderCacheCreationTokens int64 `json:"provider_cache_creation_tokens"`
			} `json:"arm"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if rep.Provenance.EngineModel != "session-cassette" || rep.Provenance.SliceID != "session:cmd-session" {
		t.Fatalf("provenance = %+v, want session cassette bound to session:cmd-session", rep.Provenance)
	}
	var off *struct {
		ArmID string `json:"arm_id"`
		Arm   struct {
			InTokens                    int64 `json:"input_tokens"`
			OutTokens                   int64 `json:"output_tokens"`
			ProviderCacheReadTokens     int64 `json:"provider_cache_read_tokens"`
			ProviderCacheCreationTokens int64 `json:"provider_cache_creation_tokens"`
		} `json:"arm"`
	}
	for i := range rep.Runs {
		if rep.Runs[i].ArmID == "all-off" {
			off = &rep.Runs[i]
			break
		}
	}
	if off == nil {
		t.Fatalf("missing all-off arm in %+v", rep.Runs)
	}
	if off.Arm.InTokens != 120 || off.Arm.OutTokens != 30 ||
		off.Arm.ProviderCacheReadTokens != 40 || off.Arm.ProviderCacheCreationTokens != 10 {
		t.Fatalf("all-off session usage = %+v, want captured 120/30/40/10", off.Arm)
	}
}

func TestAblateFromSessionRejectsEnvGatedSweep(t *testing.T) {
	path := writeAblateSessionFixture(t, `{
  "slice_id": "cmd-session",
  "turns": [{
    "usage": {"input_tokens": 10, "output_tokens": 2},
    "calls": [{"tool": "calculate", "args": {"a": 1, "b": 2}, "class": "allow"}]
  }]
}`)
	var out, errb bytes.Buffer
	code := runAblate(&out, &errb, []string{"--from-session", path, "--sweep", "normgate", "--json"})
	if code != 2 {
		t.Fatalf("exit=%d, want usage refusal; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "--from-session currently supports in-process sweeps only") {
		t.Fatalf("stderr missing from-session/env-gated refusal:\n%s", errb.String())
	}
}

func writeAblateSessionFixture(t *testing.T, body string) string {
	t.Helper()
	path := t.TempDir() + "/session.json"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
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
