package turnbench

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// rawArgs marshals a test arg map to the Trace's json.RawMessage form.
func rawArgs(t *testing.T, m map[string]any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return b
}

// spineTrace is a 5-call airline-support trajectory whose tools are all served by the
// localtools engine, with two calls (book_flight, delete_account) that go through
// adjudication — the points where a policy difference actually bites the recorded
// trajectory.
func spineTrace(t *testing.T) *Trace {
	return &Trace{
		SliceID: "policy-replay-spine",
		Calls: []Call{
			{Tool: "get_user_details", Args: rawArgs(t, map[string]any{"user_id": "u1"})},
			{Tool: "search_direct_flight", Args: rawArgs(t, map[string]any{"origin": "SFO", "destination": "JFK", "date": "2026-07-01"})},
			{Tool: "calculate", Args: rawArgs(t, map[string]any{"a": 240, "b": 60})},
			{Tool: "book_flight", Args: rawArgs(t, map[string]any{"flight_id": "UA123"})},
			{Tool: "delete_account", Args: rawArgs(t, map[string]any{"user_id": "u1"})},
		},
	}
}

// spineArms returns the K policy variants the spine scores against one trajectory.
// Divergences are driven by tools the MONITOR POLICY fully controls (search_direct_flight,
// book_flight); delete_account is a policy-INDEPENDENT structural deny — the fold's
// most-restrictive rule refuses the destructive tool in EVERY arm (allowing it in the
// monitor does not serve it), so it is "denied" uniformly and never a divergence point.
//   - recorded         (reference): denies search_direct_flight, allows the rest used.
//   - equivalent-on-trace          : a DIFFERENT policy (also allows convert_currency, a
//     tool the trace never calls) that produces the SAME observed result on every
//     recorded call — the replay-equivalence the cube collapse rests on (exact).
//   - strict-no-book               : drops book_flight from Allow → it default-denies at
//     call 3 (a deny where the reference served → divergence@3).
//   - permissive-plus              : does NOT deny search_direct_flight → it is served at
//     call 1 (a serve where the reference denied → divergence@1).
func spineArms() []PolicyArm {
	denySearch := func(allow map[string]bool) adjudicator.Policy {
		return adjudicator.Policy{
			Allow: allow,
			Deny:  map[string]abi.ReasonCode{"search_direct_flight": abi.ReasonPolicyBlock},
		}
	}
	return []PolicyArm{
		{Name: "recorded", Policy: denySearch(map[string]bool{
			"get_user_details": true, "calculate": true, "book_flight": true,
		})},
		{Name: "equivalent-on-trace", Policy: denySearch(map[string]bool{
			"get_user_details": true, "calculate": true, "book_flight": true,
			"convert_currency": true, // extra allow the trace never exercises
		})},
		{Name: "strict-no-book", Policy: denySearch(map[string]bool{
			"get_user_details": true, "calculate": true,
		})},
		{Name: "permissive-plus", Policy: adjudicator.Policy{
			Allow: map[string]bool{
				"get_user_details": true, "search_direct_flight": true, "calculate": true, "book_flight": true,
			},
		}},
	}
}

func armByName(t *testing.T, rep *PolicyReplayReport, name string) PolicyArmResult {
	t.Helper()
	for _, a := range rep.Arms {
		if a.Name == name {
			return a
		}
	}
	t.Fatalf("arm %q not found in report", name)
	return PolicyArmResult{}
}

// TestPolicyReplay_SpineCollapsesPolicyComparison is the spine proof: K policies are
// scored against ONE recorded trajectory as model-free kernel replays, the per-policy
// verdict comparison is real (deny counts differ), and the divergence witness labels
// each arm exact|bounded so resolve-rate never crosses the measured/modeled wall.
func TestPolicyReplay_SpineCollapsesPolicyComparison(t *testing.T) {
	ctx := context.Background()
	tr := spineTrace(t)
	arms := spineArms()

	rep, err := RunPolicyReplay(ctx, tr, arms, "recorded", DefaultCostModel())
	if err != nil {
		t.Fatalf("RunPolicyReplay: %v", err)
	}

	// (1) Cube-collapse accounting: K full agent+model runs become 1 recording + K-1
	// model-free replays. (K=4 policies, 5-call trace.)
	if rep.Policies != 4 || rep.Calls != 5 {
		t.Fatalf("expected 4 policies over a 5-call trace, got policies=%d calls=%d", rep.Policies, rep.Calls)
	}
	if rep.ModelTurnsNaive != 20 || rep.ModelTurnsReplay != 5 || rep.ModelTurnsAvoided != 15 {
		t.Errorf("accounting: naive=%d replay=%d avoided=%d, want 20/5/15",
			rep.ModelTurnsNaive, rep.ModelTurnsReplay, rep.ModelTurnsAvoided)
	}
	if rep.DollarsAvoided <= 0 {
		t.Errorf("modeled dollars avoided should be positive, got %v", rep.DollarsAvoided)
	}
	// The K replays must cost FAR less wall-time than even a single policy's worth of
	// model turns (5 turns × 1.5s = 7.5s); in practice the 20 local syscalls are sub-ms.
	// With the arms now fanned out concurrently the wall is the OVERLAP span, which only
	// shrinks the bound — it can never approach the naive model latency.
	naiveLatency := time.Duration(rep.ModelTurnsNaive) * time.Duration(rep.Cost.ModelTurnLatencyMs) * time.Millisecond
	if time.Duration(rep.ReplayWallNs) >= naiveLatency {
		t.Errorf("replay wall %v should be << naive model latency %v", time.Duration(rep.ReplayWallNs), naiveLatency)
	}
	// The wall is a real measurement (never negative). It may legitimately read 0ns when
	// the concurrent fan-out completes inside one tick of a coarse OS monotonic clock
	// (Windows' timer is ~0.5-15ms; a warm sub-ms fan-out rounds to 0) — a sub-tick
	// measurement, not a missing one — so the lower bound is >=0, not >0.
	if rep.ReplayWallNs < 0 {
		t.Errorf("replay wall time must be a real (non-negative) measurement, got %d", rep.ReplayWallNs)
	}

	// (2) The per-policy comparison is REAL: different policies produce different deny
	// counts on the SAME recorded trajectory (this is measured, from k.Syscall).
	rec := armByName(t, rep, "recorded")
	equiv := armByName(t, rep, "equivalent-on-trace")
	strict := armByName(t, rep, "strict-no-book")
	plus := armByName(t, rep, "permissive-plus")
	if rec.Counters.Denies != 2 {
		t.Errorf("recorded: want 2 denies (search_direct_flight + structural delete_account), got %d", rec.Counters.Denies)
	}
	if strict.Counters.Denies != 3 {
		t.Errorf("strict-no-book: want 3 denies (search + book_flight + delete_account), got %d", strict.Counters.Denies)
	}
	if plus.Counters.Denies != 1 {
		t.Errorf("permissive-plus: want 1 deny (structural delete_account only), got %d", plus.Counters.Denies)
	}
	if !(rec.Counters.Denies != strict.Counters.Denies && strict.Counters.Denies != plus.Counters.Denies) {
		t.Errorf("expected three distinct deny counts; got rec=%d strict=%d plus=%d",
			rec.Counters.Denies, strict.Counters.Denies, plus.Counters.Denies)
	}

	// (3) The divergence witness: EXACT where the observed trajectory matches the
	// reference, BOUNDED@i at the first model-observed result flip.
	if rec.Replayability != "exact" || rec.FirstDivergence != -1 {
		t.Errorf("recorded vs itself must be exact, got %q (idx %d)", rec.Replayability, rec.FirstDivergence)
	}
	// A DIFFERENT policy that never bites this trajectory is still exact — replay
	// equivalence is the basis of the collapse.
	if equiv.Replayability != "exact" || equiv.FirstDivergence != -1 {
		t.Errorf("equivalent-on-trace must replay exactly, got %q (idx %d)", equiv.Replayability, equiv.FirstDivergence)
	}
	if strict.FirstDivergence != 3 {
		t.Errorf("strict-no-book must diverge at call 3 (book_flight), got idx %d (%q)", strict.FirstDivergence, strict.Replayability)
	}
	if strict.DivergenceTool != "book_flight" {
		t.Errorf("strict-no-book divergence tool: want book_flight, got %q", strict.DivergenceTool)
	}
	if plus.FirstDivergence != 1 {
		t.Errorf("permissive-plus must diverge at call 1 (search_direct_flight), got idx %d (%q)", plus.FirstDivergence, plus.Replayability)
	}
	if plus.DivergenceTool != "search_direct_flight" {
		t.Errorf("permissive-plus divergence tool: want search_direct_flight, got %q", plus.DivergenceTool)
	}

	// (4) Honesty split: 2 arms replay exactly (resolve-rate sound), 2 are bounded
	// (verdict counters real, resolve-rate counterfactual past the frontier).
	if rep.ExactArms != 2 || rep.BoundedArms != 2 {
		t.Errorf("want 2 exact + 2 bounded arms, got exact=%d bounded=%d", rep.ExactArms, rep.BoundedArms)
	}
	// A bounded arm must carry a non-empty witness note (no silent crossing).
	if strict.DivergenceNote == "" || plus.DivergenceNote == "" {
		t.Errorf("bounded arms must carry a divergence note; strict=%q plus=%q", strict.DivergenceNote, plus.DivergenceNote)
	}
}

// TestPolicyReplay_Deterministic asserts a fixed (trace, arms) yields the identical
// verdict surface across runs — the property that makes a replayed comparison a
// reproducible artifact rather than a sample.
func TestPolicyReplay_Deterministic(t *testing.T) {
	ctx := context.Background()
	tr := spineTrace(t)
	arms := spineArms()

	a, err := RunPolicyReplay(ctx, tr, arms, "recorded", DefaultCostModel())
	if err != nil {
		t.Fatalf("run a: %v", err)
	}
	b, err := RunPolicyReplay(ctx, tr, arms, "recorded", DefaultCostModel())
	if err != nil {
		t.Fatalf("run b: %v", err)
	}

	if a.Provenance.WorkloadHash != b.Provenance.WorkloadHash {
		t.Errorf("workload hash drifted: %s vs %s", a.Provenance.WorkloadHash, b.Provenance.WorkloadHash)
	}
	if len(a.Arms) != len(b.Arms) {
		t.Fatalf("arm count drifted: %d vs %d", len(a.Arms), len(b.Arms))
	}
	for i := range a.Arms {
		if a.Arms[i].Counters != b.Arms[i].Counters {
			t.Errorf("arm %q counters drifted: %+v vs %+v", a.Arms[i].Name, a.Arms[i].Counters, b.Arms[i].Counters)
		}
		if a.Arms[i].Class != b.Arms[i].Class {
			t.Errorf("arm %q class drifted: %+v vs %+v", a.Arms[i].Name, a.Arms[i].Class, b.Arms[i].Class)
		}
		if a.Arms[i].FirstDivergence != b.Arms[i].FirstDivergence {
			t.Errorf("arm %q divergence drifted: %d vs %d", a.Arms[i].Name, a.Arms[i].FirstDivergence, b.Arms[i].FirstDivergence)
		}
	}
}

// replaySerial runs the SAME arms RunPolicyReplay runs, but strictly one-at-a-time
// in the caller's goroutine, each against its OWN injected monitor (the per-kernel
// adjudicator chain). It is the in-test reference the concurrent driver is compared
// against: the counters/class/divergence surface must be IDENTICAL whether the arms
// fan out across goroutines or run serially, which is the acceptance proof that
// concurrency did not perturb the deterministic replay.
func replaySerial(ctx context.Context, t *Trace, arms []PolicyArm) ([]KernelCounters, []ClassBreakdown, [][]CallDisposition) {
	agent.Configure()
	base := abi.Adjudicators()
	cs := make([]KernelCounters, len(arms))
	cbs := make([]ClassBreakdown, len(arms))
	disps := make([][]CallDisposition, len(arms))
	for i, arm := range arms {
		adj := adjudicator.New(arm.Policy)
		chain := swapMonitor(base, adj)
		kc, cb, _, _, disp, err := replay(ctx, t, true, false, true, withAdjudicators(chain))
		if err != nil {
			panic(err)
		}
		cs[i], cbs[i], disps[i] = kc, cb, disp
	}
	return cs, cbs, disps
}

// TestPolicyReplay_ConcurrentEqualsSerial is the core acceptance for issue #500: the K
// arms now fan out across goroutines (each with its OWN injected adjudicator chain,
// NOT a process-global SetPolicy swap), and the result must be BIT-IDENTICAL to running
// them serially. It compares the concurrent RunPolicyReplay output to an in-test serial
// replay of the same arms: per-arm kernel counters, the per-call class breakdown, AND
// the ordered per-call dispositions must all match exactly. If concurrency perturbed any
// shared state (the monitor policy, the vDSO world, the CAS), a counter or a disposition
// would drift and this test would catch it.
func TestPolicyReplay_ConcurrentEqualsSerial(t *testing.T) {
	ctx := context.Background()
	tr := spineTrace(t)
	arms := spineArms()

	// Concurrent path (the production driver).
	rep, err := RunPolicyReplay(ctx, tr, arms, "recorded", DefaultCostModel())
	if err != nil {
		t.Fatalf("RunPolicyReplay (concurrent): %v", err)
	}

	// Serial reference: the same arms, same injected chains, one at a time.
	serCtr, serClass, serDisp := replaySerial(ctx, tr, arms)

	if len(rep.Arms) != len(arms) {
		t.Fatalf("arm count: concurrent=%d want=%d", len(rep.Arms), len(arms))
	}
	for i := range arms {
		if rep.Arms[i].Counters != serCtr[i] {
			t.Errorf("arm %q counters concurrent != serial:\n  concurrent=%+v\n  serial=%+v",
				arms[i].Name, rep.Arms[i].Counters, serCtr[i])
		}
		if rep.Arms[i].Class != serClass[i] {
			t.Errorf("arm %q class concurrent != serial:\n  concurrent=%+v\n  serial=%+v",
				arms[i].Name, rep.Arms[i].Class, serClass[i])
		}
		if len(serDisp[i]) != len(tr.Calls) {
			t.Fatalf("arm %q serial disposition count=%d want=%d", arms[i].Name, len(serDisp[i]), len(tr.Calls))
		}
	}
}

// TestPolicyReplay_ConcurrentRepeatable runs the concurrent driver many times and asserts
// the verdict surface NEVER drifts — the determinism gate under fan-out. A latent race or
// a shared-state leak between arms would surface as an intermittent counter/divergence
// drift across iterations; a clean per-kernel injection is invariant.
func TestPolicyReplay_ConcurrentRepeatable(t *testing.T) {
	ctx := context.Background()
	tr := spineTrace(t)
	arms := spineArms()

	first, err := RunPolicyReplay(ctx, tr, arms, "recorded", DefaultCostModel())
	if err != nil {
		t.Fatalf("run 0: %v", err)
	}
	for n := 1; n < 25; n++ {
		got, err := RunPolicyReplay(ctx, tr, arms, "recorded", DefaultCostModel())
		if err != nil {
			t.Fatalf("run %d: %v", n, err)
		}
		if got.ExactArms != first.ExactArms || got.BoundedArms != first.BoundedArms {
			t.Fatalf("run %d honesty split drifted: exact=%d/%d bounded=%d/%d",
				n, got.ExactArms, first.ExactArms, got.BoundedArms, first.BoundedArms)
		}
		for i := range got.Arms {
			if got.Arms[i].Counters != first.Arms[i].Counters {
				t.Fatalf("run %d arm %q counters drifted: %+v vs %+v",
					n, got.Arms[i].Name, got.Arms[i].Counters, first.Arms[i].Counters)
			}
			if got.Arms[i].FirstDivergence != first.Arms[i].FirstDivergence {
				t.Fatalf("run %d arm %q divergence drifted: %d vs %d",
					n, got.Arms[i].Name, got.Arms[i].FirstDivergence, first.Arms[i].FirstDivergence)
			}
			if got.Arms[i].Replayability != first.Arms[i].Replayability {
				t.Fatalf("run %d arm %q replayability drifted: %q vs %q",
					n, got.Arms[i].Name, got.Arms[i].Replayability, first.Arms[i].Replayability)
			}
		}
	}
}

// TestPolicyReplay_NoDivergenceControl is the anti-inflation control: replaying the
// reference policy against itself (every arm == reference) yields ZERO divergences and
// a zero floor-delta — the witness cannot manufacture a divergence where the policy did
// not change, the analogue of the turn-tax happy-path=0 control.
func TestPolicyReplay_NoDivergenceControl(t *testing.T) {
	ctx := context.Background()
	tr := spineTrace(t)
	ref := spineArms()[0] // "recorded"
	arms := []PolicyArm{ref, {Name: "ref-copy", Policy: ref.Policy}, {Name: "ref-copy-2", Policy: ref.Policy}}

	rep, err := RunPolicyReplay(ctx, tr, arms, "recorded", DefaultCostModel())
	if err != nil {
		t.Fatalf("RunPolicyReplay: %v", err)
	}
	if rep.BoundedArms != 0 || rep.ExactArms != 3 {
		t.Errorf("identical policies must all be exact; got exact=%d bounded=%d", rep.ExactArms, rep.BoundedArms)
	}
	for _, a := range rep.Arms {
		if a.FirstDivergence != -1 {
			t.Errorf("arm %q against an identical reference must not diverge, got idx %d", a.Name, a.FirstDivergence)
		}
		if a.Counters != rep.Arms[0].Counters {
			t.Errorf("arm %q counters must match the reference exactly: %+v vs %+v", a.Name, a.Counters, rep.Arms[0].Counters)
		}
	}
}

// redactTrace is a 2-call trajectory whose first call carries a secret-shaped arg
// ("password"). Both the served arm and the redact arm SERVE the call (a monitor REDACT
// is a pass, not a deny/quarantine), so observedClass buckets both as "served" — the
// exact-false trap the raw-verdict path (#501) has to catch.
func redactTrace(t *testing.T) *Trace {
	return &Trace{
		SliceID: "policy-replay-redact",
		Calls: []Call{
			{Tool: "get_user_details", Args: rawArgs(t, map[string]any{"user_id": "u1", "password": "hunter2"})},
			{Tool: "calculate", Args: rawArgs(t, map[string]any{"a": 240, "b": 60})},
		},
	}
}

// TestPolicyReplay_RedactTransformDivergence is the #501(a) proof: a redact-ONLY policy
// difference must surface as bounded@i, not a false exact. The reference serves the
// secret-shaped call verbatim; the redact arm rewrites "password" -> "[REDACTED]" before
// dispatch. observedClass calls BOTH "served" (a redact is a pass), so the class witness
// alone would (wrongly) report exact; the raw-verdict path catches the differing rewrite
// and labels the arm bounded@0.
func TestPolicyReplay_RedactTransformDivergence(t *testing.T) {
	ctx := context.Background()
	tr := redactTrace(t)
	allow := map[string]bool{"get_user_details": true, "calculate": true}
	arms := []PolicyArm{
		// Reference: serves the secret-shaped call verbatim (no redact).
		{Name: "served-raw", Policy: adjudicator.Policy{Allow: allow}},
		// A DIFFERENT policy that does NOT redact and serves identically -> still exact.
		{Name: "served-raw-copy", Policy: adjudicator.Policy{Allow: allow}},
		// Redact arm: same allow set, but rewrites "password" before dispatch. The model
		// would observe different args -> a redact divergence the class witness misses.
		{Name: "redact-password", Policy: adjudicator.Policy{Allow: allow, RedactFields: []string{"password"}}},
	}

	rep, err := RunPolicyReplay(ctx, tr, arms, "served-raw", DefaultCostModel())
	if err != nil {
		t.Fatalf("RunPolicyReplay: %v", err)
	}

	raw := armByName(t, rep, "served-raw")
	rawCopy := armByName(t, rep, "served-raw-copy")
	redact := armByName(t, rep, "redact-password")

	// The reference and an identical non-redacting policy replay exactly.
	if raw.Replayability != "exact" || raw.FirstDivergence != -1 {
		t.Errorf("served-raw vs itself must be exact, got %q (idx %d)", raw.Replayability, raw.FirstDivergence)
	}
	if rawCopy.Replayability != "exact" || rawCopy.FirstDivergence != -1 {
		t.Errorf("served-raw-copy must replay exactly, got %q (idx %d)", rawCopy.Replayability, rawCopy.FirstDivergence)
	}

	// The CRUX: the redact arm must NOT read as a false exact. observedClass calls call 0
	// "served" under both arms, so without the raw-verdict path this would be exact.
	if redact.FirstDivergence != 0 {
		t.Fatalf("redact-password must diverge at call 0 (the secret-shaped call), got idx %d (%q)",
			redact.FirstDivergence, redact.Replayability)
	}
	if redact.Replayability != "bounded@0" {
		t.Errorf("redact-password must be bounded@0, got %q", redact.Replayability)
	}
	if redact.DivergenceTool != "get_user_details" {
		t.Errorf("redact divergence tool: want get_user_details, got %q", redact.DivergenceTool)
	}
	if redact.DivergenceNote == "" {
		t.Errorf("a redact divergence must carry a witness note (no silent crossing)")
	}

	// Honesty split: 2 exact (reference + identical) + 1 bounded (the redact arm).
	if rep.ExactArms != 2 || rep.BoundedArms != 1 {
		t.Errorf("want 2 exact + 1 bounded, got exact=%d bounded=%d", rep.ExactArms, rep.BoundedArms)
	}

	// Anti-inflation: confirm the call's observed CLASS really is "served" under both arms
	// (so the divergence is genuinely the redact, not a deny/quarantine the class witness
	// would already catch). Both arms must serve call 0 with zero denies/quarantines.
	if raw.Counters.Denies != 0 || redact.Counters.Denies != 0 {
		t.Errorf("redact must be a PASS, not a deny: raw denies=%d redact denies=%d",
			raw.Counters.Denies, redact.Counters.Denies)
	}
	if redact.Counters.Transforms < 1 {
		t.Errorf("redact arm must record a monitor TRANSFORM, got transforms=%d", redact.Counters.Transforms)
	}
}
