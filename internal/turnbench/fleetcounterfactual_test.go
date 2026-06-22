package turnbench

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// floorTrace is a security-FLOOR trajectory: it fires a quarantine (a poisoned
// fetch_policy refund doc the ctx-MMU pages out at admission), a structural deny
// (delete_account is refused by the capability floor in EVERY arm), plus benign served
// calls. It is the engine-agnostic floor the fleet counterfactual reports on — the same
// shape internal/tracesink would capture from a live run.
func floorTrace(t *testing.T) *Trace {
	return &Trace{
		SliceID: "fleet-cf-floor",
		Calls: []Call{
			// Poisoned refund doc -> ctx-MMU quarantine at result admission.
			{Tool: "fetch_policy", Args: rawArgs(t, map[string]any{"topic": "refunds"}),
				Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}},
			{Tool: "get_user_details", Args: rawArgs(t, map[string]any{"user_id": "u1"})},
			{Tool: "search_direct_flight", Args: rawArgs(t, map[string]any{"origin": "SFO", "destination": "JFK", "date": "2026-07-01"})},
			{Tool: "book_flight", Args: rawArgs(t, map[string]any{"flight_id": "UA123"})},
			// Destructive sink -> policy-INDEPENDENT structural deny in every arm.
			{Tool: "delete_account", Args: rawArgs(t, map[string]any{"user_id": "u1"}),
				Meta: map[string]string{"readOnlyHint": "false", "idempotentHint": "false", "destructive": "true"}},
		},
	}
}

// floorArms are candidate policies over the floor trace. The reference allows the benign
// served tools; the candidates differ in how much floor they catch:
//   - reference     : allows the served tools; quarantine + structural delete deny fire.
//   - equivalent    : a DIFFERENT policy (extra allow the trace never calls) that observes
//     the SAME result on every recorded call — exact, full floor coverage, resolve-rate sound.
//   - strict-no-book: drops book_flight from Allow -> it default-denies at call 3, a deny
//     where the reference served -> BOUNDED@3; an EXTRA deny over the reference.
func floorArms() []PolicyArm {
	allow := map[string]bool{
		"fetch_policy": true, "get_user_details": true, "search_direct_flight": true, "book_flight": true,
	}
	cp := func(m map[string]bool) map[string]bool {
		out := map[string]bool{}
		for k, v := range m {
			out[k] = v
		}
		return out
	}
	return []PolicyArm{
		{Name: "reference", Policy: adjudicator.Policy{Allow: cp(allow)}},
		{Name: "equivalent", Policy: adjudicator.Policy{Allow: func() map[string]bool {
			m := cp(allow)
			m["convert_currency"] = true // extra allow the trace never exercises
			return m
		}()}},
		{Name: "strict-no-book", Policy: adjudicator.Policy{Allow: func() map[string]bool {
			m := cp(allow)
			delete(m, "book_flight")
			return m
		}()}},
	}
}

func policyByName(t *testing.T, rep *FleetCounterfactualReport, name string) PolicyFloorCoverage {
	t.Helper()
	for _, p := range rep.Coverage {
		if p.Policy == name {
			return p
		}
	}
	t.Fatalf("policy %q not found in coverage", name)
	return PolicyFloorCoverage{}
}

func cellOf(t *testing.T, rep *FleetCounterfactualReport, trace, policy string) FloorCell {
	t.Helper()
	for _, c := range rep.CellTable {
		if c.Trace == trace && c.Policy == policy {
			return c
		}
	}
	t.Fatalf("cell (%s, %s) not found", trace, policy)
	return FloorCell{}
}

// TestFleetCounterfactual_PerPolicyFloorCoverage is the issue-#502 headline proof: a
// recorded corpus scored against candidate policies emits a PER-POLICY floor-coverage
// report — the summed Quarantines/Denies/Transforms across the population per policy —
// computed at $0 model spend. The floor counters are MEASURED (real kernel events for
// every recorded call), so they are reported for ALL cells regardless of divergence.
func TestFleetCounterfactual_PerPolicyFloorCoverage(t *testing.T) {
	ctx := context.Background()
	// Two copies of the floor trace (distinct slice ids) make a small population so the
	// per-policy roll-up sums across MORE than one trace.
	tr1 := floorTrace(t)
	tr2 := floorTrace(t)
	tr2.SliceID = "fleet-cf-floor-2"
	corpus := []DivHistInput{
		{Trace: tr1, Arms: floorArms(), RefName: "reference"},
		{Trace: tr2, Arms: floorArms(), RefName: "reference"},
	}

	rep, err := RunFleetCounterfactual(ctx, corpus, DefaultCostModel())
	if err != nil {
		t.Fatalf("RunFleetCounterfactual: %v", err)
	}

	// $0 model spend is the headline — the whole report is model-free replay.
	if rep.ModelCallsSpent != 0 {
		t.Errorf("fleet counterfactual must spend ZERO model calls, got %d", rep.ModelCallsSpent)
	}
	if rep.Traces != 2 {
		t.Errorf("traces: want 2, got %d", rep.Traces)
	}
	// 3 arms, 1 reference => 2 candidate policies; 2 traces => 4 cells.
	if rep.Policies != 2 {
		t.Errorf("policies: want 2 candidates (3 arms minus the reference), got %d", rep.Policies)
	}
	if rep.Cells != 4 {
		t.Fatalf("cells: want 4 (2 candidates x 2 traces), got %d", rep.Cells)
	}

	// The floor really fires: each floor trace quarantines the poisoned fetch_policy and
	// denies the structural delete_account. Confirm against the per-cell counters.
	equivCell := cellOf(t, rep, "fleet-cf-floor", "equivalent")
	if equivCell.Floor.Quarantines != 1 {
		t.Errorf("equivalent cell must quarantine the 1 poisoned fetch_policy, got %d", equivCell.Floor.Quarantines)
	}
	if equivCell.Floor.Denies != 1 {
		t.Errorf("equivalent cell must deny the 1 structural delete_account, got %d", equivCell.Floor.Denies)
	}

	// PER-POLICY ROLL-UP across the population (the headline). The "equivalent" policy is
	// exact on both traces: 2 cells, each 1 quarantine + 1 deny => 2 quarantines + 2 denies.
	equiv := policyByName(t, rep, "equivalent")
	if equiv.Cells != 2 {
		t.Fatalf("equivalent: want 2 cells (1 per trace), got %d", equiv.Cells)
	}
	if equiv.Floor.Quarantines != 2 || equiv.Floor.Denies != 2 {
		t.Errorf("equivalent population floor: want 2 quarantines + 2 denies across 2 cells, got Q=%d D=%d",
			equiv.Floor.Quarantines, equiv.Floor.Denies)
	}

	// The "strict-no-book" policy denies book_flight on top of the structural delete, so it
	// catches MORE floor (2 denies per cell): the per-policy comparison is REAL and measured.
	strict := policyByName(t, rep, "strict-no-book")
	if strict.Floor.Denies <= equiv.Floor.Denies {
		t.Errorf("strict-no-book must deny MORE than equivalent (it drops book_flight): strict=%d equiv=%d",
			strict.Floor.Denies, equiv.Floor.Denies)
	}
	// 2 cells x (book_flight deny + structural delete deny) = 4 denies.
	if strict.Floor.Denies != 4 {
		t.Errorf("strict-no-book population denies: want 4 (book + delete, x2 traces), got %d", strict.Floor.Denies)
	}

	// The corpus-wide population floor is the sum over EVERY cell — a MEASURED headline.
	// equivalent (2Q+2D) + strict (2Q+4D) = 4 quarantines + 6 denies.
	if rep.PopulationFloor.Quarantines != 4 || rep.PopulationFloor.Denies != 6 {
		t.Errorf("population floor: want 4 quarantines + 6 denies, got Q=%d D=%d",
			rep.PopulationFloor.Quarantines, rep.PopulationFloor.Denies)
	}

	// Every cell is labeled exact|bounded@i (the per-cell witness).
	for _, c := range rep.CellTable {
		if c.Replayability == "" {
			t.Errorf("cell (%s, %s) is unlabeled — every cell must carry exact|bounded@i", c.Trace, c.Policy)
		}
	}
}

// TestFleetCounterfactual_BoundedCellRefusesResolveRate is the load-bearing honesty proof:
// a BOUNDED cell's floor counters ARE reported (measured), but its resolve-rate is NOT —
// it must be flagged "needs live re-run from frontier" and NEVER contribute a resolve-rate
// number. This test proves the split: the bounded cell has full floor counters yet
// ResolveRateReportable=false, and its policy row is likewise not resolve-rate reportable.
func TestFleetCounterfactual_BoundedCellRefusesResolveRate(t *testing.T) {
	ctx := context.Background()
	corpus := []DivHistInput{
		{Trace: floorTrace(t), Arms: floorArms(), RefName: "reference"},
	}

	rep, err := RunFleetCounterfactual(ctx, corpus, DefaultCostModel())
	if err != nil {
		t.Fatalf("RunFleetCounterfactual: %v", err)
	}

	// strict-no-book diverges at book_flight (a deny where the reference served) -> bounded.
	strictCell := cellOf(t, rep, "fleet-cf-floor", "strict-no-book")
	if strictCell.Exact {
		t.Fatalf("strict-no-book must be BOUNDED on the floor trace (it denies book_flight), got exact")
	}
	if strictCell.Replayability == "exact" || strictCell.Replayability == "" {
		t.Errorf("bounded cell must carry a bounded@i label, got %q", strictCell.Replayability)
	}

	// THE LOAD-BEARING SPLIT: the bounded cell's FLOOR counters are still reported (real
	// kernel events) — it caught the quarantine + denies — yet its RESOLVE-RATE is refused.
	if strictCell.Floor.Denies == 0 && strictCell.Floor.Quarantines == 0 {
		t.Errorf("a bounded cell must STILL report its measured floor counters, got empty floor")
	}
	if strictCell.ResolveRateReportable {
		t.Errorf("a bounded cell must NOT report a resolve-rate (it diverged); ResolveRateReportable must be false")
	}
	if strictCell.ResolveRateNote == "" {
		t.Errorf("a bounded cell must carry a resolve-rate refusal note (no silent crossing)")
	}

	// The policy row is likewise not resolve-rate reportable (it has a bounded cell), but
	// its floor coverage is intact and reported.
	strict := policyByName(t, rep, "strict-no-book")
	if strict.ResolveRateReportable {
		t.Errorf("strict-no-book has a bounded cell, so its resolve-rate must be refused at the policy level too")
	}
	if strict.ResolveRateNote == "" {
		t.Errorf("a policy with bounded cells must carry a resolve-rate refusal note")
	}
	if strict.Floor.Denies == 0 {
		t.Errorf("the bounded policy must STILL report its measured floor coverage, got 0 denies")
	}

	// CONTRAST: the exact "equivalent" cell DOES report a resolve-rate (it replays soundly).
	equivCell := cellOf(t, rep, "fleet-cf-floor", "equivalent")
	if !equivCell.Exact || !equivCell.ResolveRateReportable {
		t.Errorf("an exact cell MUST be resolve-rate reportable; exact=%v reportable=%v",
			equivCell.Exact, equivCell.ResolveRateReportable)
	}
	if equivCell.ResolveRateNote != "" {
		t.Errorf("an exact cell must NOT carry a refusal note, got %q", equivCell.ResolveRateNote)
	}

	// Belt-and-braces: NO bounded cell anywhere in the table reports a resolve-rate.
	for _, c := range rep.CellTable {
		if !c.Exact && c.ResolveRateReportable {
			t.Errorf("bounded cell (%s, %s) leaked a resolve-rate number — the honesty split is broken",
				c.Trace, c.Policy)
		}
	}
}

// TestFleetCounterfactual_NoDivergenceControlFullCoverage is the anti-inflation control: a
// corpus whose candidate arms are all IDENTICAL to the reference yields 100% EXACT cells
// AND full floor coverage — every cell reports its floor counters AND a sound resolve-rate.
// The witness can never manufacture a divergence where the policy did not change, so the
// honest baseline is 100% exact with every floor counter intact.
func TestFleetCounterfactual_NoDivergenceControlFullCoverage(t *testing.T) {
	ctx := context.Background()
	ref := floorArms()[0] // "reference"
	noDiv := []PolicyArm{
		ref,
		{Name: "ref-copy", Policy: ref.Policy},
		{Name: "ref-copy-2", Policy: ref.Policy},
	}
	corpus := []DivHistInput{
		{Trace: floorTrace(t), Arms: noDiv, RefName: "reference"},
	}

	rep, err := RunFleetCounterfactual(ctx, corpus, DefaultCostModel())
	if err != nil {
		t.Fatalf("RunFleetCounterfactual: %v", err)
	}

	// 3 arms, 1 reference => 2 candidate cells, both exact.
	if rep.Cells != 2 {
		t.Fatalf("cells: want 2 candidates, got %d", rep.Cells)
	}
	if rep.BoundedCells != 0 {
		t.Errorf("no-divergence control must have ZERO bounded cells, got %d", rep.BoundedCells)
	}
	if rep.ExactFraction != 1.0 {
		t.Fatalf("no-divergence control must read 100%% exact, got %v (exact=%d/cells=%d)",
			rep.ExactFraction, rep.ExactCells, rep.Cells)
	}

	// FULL floor coverage: every candidate cell reports its floor counters AND is
	// resolve-rate reportable (it replays the reference exactly). Each cell catches the
	// 1 quarantine + 1 structural deny.
	for _, c := range rep.CellTable {
		if !c.Exact || !c.ResolveRateReportable {
			t.Errorf("control cell (%s, %s) must be exact + resolve-rate reportable; exact=%v reportable=%v",
				c.Trace, c.Policy, c.Exact, c.ResolveRateReportable)
		}
		if c.Floor.Quarantines != 1 || c.Floor.Denies != 1 {
			t.Errorf("control cell (%s, %s) must report full floor (1Q+1D), got Q=%d D=%d",
				c.Trace, c.Policy, c.Floor.Quarantines, c.Floor.Denies)
		}
	}
	// Every policy row is resolve-rate reportable (no bounded cell anywhere).
	for _, p := range rep.Coverage {
		if !p.ResolveRateReportable {
			t.Errorf("control policy %q must be resolve-rate reportable (all cells exact)", p.Policy)
		}
		if p.ResolveRateNote != "" {
			t.Errorf("control policy %q must NOT carry a refusal note, got %q", p.Policy, p.ResolveRateNote)
		}
	}
}

// TestFleetCounterfactual_Deterministic asserts the report is a regenerable artifact: a
// fixed corpus yields byte-identical JSON across runs (modulo host-dependent provenance and
// the measured wall span), so the floor coverage is a reproducible number, not a sample.
func TestFleetCounterfactual_Deterministic(t *testing.T) {
	ctx := context.Background()
	mk := func() []DivHistInput {
		return []DivHistInput{{Trace: floorTrace(t), Arms: floorArms(), RefName: "reference"}}
	}

	a, err := RunFleetCounterfactual(ctx, mk(), DefaultCostModel())
	if err != nil {
		t.Fatalf("run a: %v", err)
	}
	b, err := RunFleetCounterfactual(ctx, mk(), DefaultCostModel())
	if err != nil {
		t.Fatalf("run b: %v", err)
	}

	// Zero the host-dependent provenance and the measured wall span so the comparison is
	// over the verdict/floor surface (which IS deterministic).
	a.Provenance, b.Provenance = Provenance{}, Provenance{}
	a.ReplayWallNs, b.ReplayWallNs = 0, 0
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Errorf("fleet counterfactual report drifted across runs:\n a=%s\n b=%s", ja, jb)
	}
}
