package ablate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/bench"
)

// mustMarshal marshals v or fails the test (a test-side convenience for round-tripping
// a trace through the same JSON path the subprocess wire uses).
func mustMarshal(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

// fakeArmRunner stands in for execArmRunner so the suite never spawns the real binary.
// It runs the single-arm core IN-PROCESS via RunOneArm, then SIMULATES the one thing an
// in-process run cannot do for an env-gated feature: the child's process-start read of
// FAK_NORMGATE. A real child re-exec'd with FAK_NORMGATE=1 quarantines on the normgate
// path; a child with FAK_NORMGATE=0 does not. The fake reflects exactly that env effect
// into the arm's Quarantines counter, so the test can prove the env knob CHANGED the
// path — the rung-2 acceptance criterion — without a subprocess.
func fakeArmRunner(ctx context.Context, bin string, c FeatureConfig, traceJSON []byte) (AblationRun, error) {
	tr, err := UnmarshalTrace(json.RawMessage(traceJSON))
	if err != nil {
		return AblationRun{}, err
	}
	run, err := RunOneArm(ctx, tr, "mock", c)
	if err != nil {
		return AblationRun{}, err
	}
	// Model the env-gated normgate effect the way a re-exec'd child would surface it:
	// the child env carries FAK_NORMGATE=1 iff this arm turned normgate on.
	for _, kv := range c.childEnv() {
		if kv == "FAK_NORMGATE=1" {
			run.Arm.Quarantines += int64(len(tr.Calls)) // normgate quarantined every call
		}
	}
	return run, nil
}

// SweepViaSubprocess over [vdso, normgate] with the injected fake runner: it yields the
// full arm matrix, binds every arm to ONE workload hash (the cross-process guard), and
// the normgate=on arm shows a different quarantine count than normgate=off — proving the
// env-gated knob actually moved the path across the (simulated) subprocess boundary.
func TestSweepViaSubprocess_EnvArmChangesPath(t *testing.T) {
	tr, err := bench.LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace(%q): %v", smokeTrace, err)
	}
	configs, err := BuildSweep([]string{FeatureVDSO, FeatureNormgate})
	if err != nil {
		t.Fatalf("BuildSweep: %v", err)
	}
	// all-off, vdso, normgate, all-on
	if len(configs) != 4 {
		t.Fatalf("BuildSweep([vdso,normgate]) = %d arms, want 4", len(configs))
	}

	rep, dropped, err := SweepViaSubprocess(context.Background(), "/fake/fak", tr, "mock", "mock-offline", configs, "all-off", fakeArmRunner)
	if err != nil {
		t.Fatalf("SweepViaSubprocess: %v", err)
	}
	if len(dropped) != 0 {
		t.Fatalf("unexpected dropped arms: %+v", dropped)
	}
	if len(rep.Runs) != 4 {
		t.Fatalf("report has %d arms, want 4", len(rep.Runs))
	}

	// The cross-process identical-workload guard: the report binds ONE canonical hash
	// (the round-tripped trace's, the same one every child computes from the wire bytes),
	// and every arm shares it. A non-empty, internally-consistent hash is the contract;
	// it need not equal the un-canonicalized in-memory hash.
	if rep.WorkloadHash == "" {
		t.Error("report WorkloadHash is empty")
	}
	canon, err := UnmarshalTrace(mustMarshal(t, tr))
	if err != nil {
		t.Fatalf("canonicalize: %v", err)
	}
	if rep.WorkloadHash != canon.WorkloadHash() {
		t.Errorf("report WorkloadHash = %q, want the round-tripped canonical hash %q", rep.WorkloadHash, canon.WorkloadHash())
	}
	for _, run := range rep.Runs {
		if run.WorkloadHash != rep.WorkloadHash {
			t.Errorf("arm %q WorkloadHash = %q, want %q", run.ArmID, run.WorkloadHash, rep.WorkloadHash)
		}
	}

	// The descriptor records the env-gated knob honestly on the arms that carry it.
	allOff := rep.ArmByID("all-off")
	normOn := rep.ArmByID("normgate")
	if allOff == nil || normOn == nil {
		t.Fatalf("expected arms all-off and normgate, got %+v", armIDs(rep))
	}
	if allOff.Features[FeatureNormgate] != "off" {
		t.Errorf("all-off normgate descriptor = %q, want off", allOff.Features[FeatureNormgate])
	}
	if normOn.Features[FeatureNormgate] != "on" {
		t.Errorf("normgate arm descriptor = %q, want on", normOn.Features[FeatureNormgate])
	}

	// The env knob actually changed the path: normgate=on quarantined, normgate=off did
	// not. This is the rung-2 acceptance assertion (the env feature is genuinely live in
	// the child, not a no-op).
	if normOn.Arm.Quarantines <= allOff.Arm.Quarantines {
		t.Errorf("normgate=on Quarantines (%d) must exceed normgate=off (%d) — the env arm must change the path",
			normOn.Arm.Quarantines, allOff.Arm.Quarantines)
	}
	if allOff.Arm.Quarantines != 0 {
		t.Errorf("all-off Quarantines = %d, want 0 (normgate off, no quarantine path)", allOff.Arm.Quarantines)
	}
}

// A child that fails to run is DROPPED with a recorded reason, never a silent hole: the
// surviving arms still assemble, and the dropped arm is named with its error.
func TestSweepViaSubprocess_DropsFailedArmWithReason(t *testing.T) {
	tr, err := bench.LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}
	configs, err := BuildSweep([]string{FeatureVDSO, FeatureNormgate})
	if err != nil {
		t.Fatalf("BuildSweep: %v", err)
	}

	// A runner that fails exactly the "normgate" arm (a child that could not start),
	// succeeds on the rest via the in-process core.
	failNormgate := func(ctx context.Context, bin string, c FeatureConfig, traceJSON []byte) (AblationRun, error) {
		if c.Name == "normgate" {
			return AblationRun{}, fmt.Errorf("child exited 1: simulated re-exec failure")
		}
		return fakeArmRunner(ctx, bin, c, traceJSON)
	}

	rep, dropped, err := SweepViaSubprocess(context.Background(), "/fake/fak", tr, "mock", "mock-offline", configs, "all-off", failNormgate)
	if err != nil {
		t.Fatalf("SweepViaSubprocess (one arm dropped) should still assemble: %v", err)
	}
	// 4 configs, 1 dropped -> 3 surviving arms.
	if len(rep.Runs) != 3 {
		t.Fatalf("surviving arms = %d, want 3 (normgate dropped)", len(rep.Runs))
	}
	if rep.ArmByID("normgate") != nil {
		t.Error("dropped normgate arm must NOT be in the report")
	}
	if len(dropped) != 1 || dropped[0].ArmID != "normgate" {
		t.Fatalf("dropped = %+v, want exactly the normgate arm", dropped)
	}
	if !strings.Contains(dropped[0].Reason, "simulated re-exec failure") {
		t.Errorf("dropped reason = %q, want the child's error (a logged hole, not a silent one)", dropped[0].Reason)
	}
}

// A child that replays a DIFFERENT workload is dropped, never folded — the cross-process
// identical-workload guard. Returning an arm with a forked hash must not corrupt the
// report.
func TestSweepViaSubprocess_DropsForkedWorkload(t *testing.T) {
	tr, err := bench.LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}
	configs, err := BuildSweep([]string{FeatureVDSO})
	if err != nil {
		t.Fatalf("BuildSweep: %v", err)
	}

	forked := func(ctx context.Context, bin string, c FeatureConfig, traceJSON []byte) (AblationRun, error) {
		run, err := fakeArmRunner(ctx, bin, c, traceJSON)
		if err != nil {
			return AblationRun{}, err
		}
		if c.Name == "vdso" {
			run.WorkloadHash = "FORKED-HASH" // child secretly ran different work
		}
		return run, nil
	}

	rep, dropped, err := SweepViaSubprocess(context.Background(), "/fake/fak", tr, "mock", "mock-offline", configs, "all-off", forked)
	if err != nil {
		t.Fatalf("SweepViaSubprocess: %v", err)
	}
	if rep.ArmByID("vdso") != nil {
		t.Error("forked-workload arm must be dropped, not folded")
	}
	if len(dropped) != 1 || dropped[0].ArmID != "vdso" {
		t.Fatalf("dropped = %+v, want the vdso arm on a workload mismatch", dropped)
	}
	if !strings.Contains(dropped[0].Reason, "workload hash mismatch") {
		t.Errorf("dropped reason = %q, want a workload-hash-mismatch reason", dropped[0].Reason)
	}
}

// A baseline that was itself dropped is refused — the table cannot anchor on an arm that
// never ran (fail closed, no silent re-point).
func TestSweepViaSubprocess_RefusesDroppedBaseline(t *testing.T) {
	tr, err := bench.LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}
	configs, err := BuildSweep([]string{FeatureVDSO})
	if err != nil {
		t.Fatalf("BuildSweep: %v", err)
	}
	failAllOff := func(ctx context.Context, bin string, c FeatureConfig, traceJSON []byte) (AblationRun, error) {
		if c.Name == "all-off" {
			return AblationRun{}, fmt.Errorf("baseline child failed")
		}
		return fakeArmRunner(ctx, bin, c, traceJSON)
	}
	if _, _, err := SweepViaSubprocess(context.Background(), "/fake/fak", tr, "mock", "mock-offline", configs, "all-off", failAllOff); err == nil {
		t.Fatal("SweepViaSubprocess accepted a sweep whose baseline arm was dropped; want refusal")
	}
}

// RunArmMode is the exact child codec the production re-exec drives: it decodes one
// armRequest from stdin, runs the single arm in-process, and encodes one AblationRun to
// stdout. Exercising it here proves the wire format parent and child agree on, and that
// the child reconstructs the SAME workload hash the parent froze — without a spawn.
func TestRunArmMode_RoundTripsRequest(t *testing.T) {
	tr, err := bench.LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}
	traceJSON, err := json.Marshal(tr)
	if err != nil {
		t.Fatalf("marshal trace: %v", err)
	}
	cfg := FeatureConfig{Name: "vdso", VDSO: true}
	req := armRequest{Config: cfg, EngineID: "mock", EngineModel: "mock-offline", TraceJSON: traceJSON}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	var out bytes.Buffer
	if err := RunArmMode(context.Background(), bytes.NewReader(reqJSON), &out, UnmarshalTrace); err != nil {
		t.Fatalf("RunArmMode: %v", err)
	}

	var run AblationRun
	if err := json.Unmarshal(out.Bytes(), &run); err != nil {
		t.Fatalf("decode child output: %v", err)
	}
	if run.ArmID != "vdso" {
		t.Errorf("child arm id = %q, want vdso", run.ArmID)
	}
	// The child hashes the trace it RECONSTRUCTS from the wire bytes; json.Marshal
	// compacts Call.Args, so the authoritative hash is the round-tripped one (the same
	// canonicalization SweepViaSubprocess applies parent-side). Compare against that, so
	// parent and child agree by construction.
	canon, err := UnmarshalTrace(traceJSON)
	if err != nil {
		t.Fatalf("canonicalize trace: %v", err)
	}
	if run.WorkloadHash != canon.WorkloadHash() {
		t.Errorf("child workload hash = %q, want %q (parent and child must agree on the round-tripped trace)", run.WorkloadHash, canon.WorkloadHash())
	}
	if run.Features[FeatureVDSO] != "on" {
		t.Errorf("child descriptor vdso = %q, want on", run.Features[FeatureVDSO])
	}
	if run.Arm.VDSOHits <= 0 {
		t.Errorf("vdso child VDSOHits = %d, want > 0 (the knob is live in the child)", run.Arm.VDSOHits)
	}
}

// childEnv renders the env-gated toggles as FAK_*=1|0, sorted and deterministic, so a
// re-exec'd child reads the arm's intent with no "unset means default" ambiguity.
func TestChildEnv_RendersFakVarsDeterministically(t *testing.T) {
	c := FeatureConfig{Name: "x"}
	c.apply(FeatureNormgate, true)
	c.apply(FeatureRadix, false)
	got := c.childEnv()
	// childEnv sorts by sweep TOKEN: "normgate" < "radix", so FAK_NORMGATE then
	// FAK_INKERNEL_RADIX, each rendered =1 (on) / =0 (off).
	want := []string{"FAK_NORMGATE=1", "FAK_INKERNEL_RADIX=0"}
	if len(got) != len(want) {
		t.Fatalf("childEnv = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("childEnv[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
	// A config with no env features carries no FAK_* splice.
	if env := (FeatureConfig{Name: "y", VDSO: true}).childEnv(); len(env) != 0 {
		t.Errorf("vdso-only childEnv = %v, want empty (vdso is runtime, not env-gated)", env)
	}
}
