package depthadmit

import (
	"encoding/json"
	"strings"
	"testing"
)

// plan is a terse declared-plan builder: ids only, in declared order.
func plan(ids ...string) []Phase {
	out := make([]Phase, 0, len(ids))
	for _, id := range ids {
		out = append(out, Phase{ID: id})
	}
	return out
}

func TestFoldVerdicts(t *testing.T) {
	cases := []struct {
		name      string
		in        Input
		want      Verdict
		declared  int
		carried   int
		frontier  string // "" means the frontier must be nil
		remaining int
	}{
		{
			name: "no plan is undeclared, not shallow",
			in:   Input{},
			want: VerdictUndeclared,
		},
		{
			name: "a plan with witnesses but no declared phases is still undeclared",
			in:   Input{Witnessed: []string{"p1"}},
			want: VerdictUndeclared,
		},
		{
			name:     "declared and never walked is shallow",
			in:       Input{Plan: plan("p1", "p2", "p3")},
			want:     VerdictShallow,
			declared: 3, carried: 0, frontier: "p1", remaining: 3,
		},
		{
			name:     "partly walked is advancing",
			in:       Input{Plan: plan("p1", "p2", "p3"), Witnessed: []string{"p1"}},
			want:     VerdictAdvancing,
			declared: 3, carried: 1, frontier: "p2", remaining: 2,
		},
		{
			name:     "fully walked is carried with no frontier",
			in:       Input{Plan: plan("p1", "p2"), Witnessed: []string{"p2", "p1"}},
			want:     VerdictCarried,
			declared: 2, carried: 2, frontier: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Fold(tc.in)
			if got.Verdict != tc.want {
				t.Fatalf("verdict = %q, want %q", got.Verdict, tc.want)
			}
			if !ValidVerdict(got.Verdict) {
				t.Fatalf("verdict %q is outside the closed vocabulary", got.Verdict)
			}
			if got.Schema != Schema {
				t.Errorf("schema = %q, want %q", got.Schema, Schema)
			}
			if got.Coverage.Declared != tc.declared || got.Coverage.Carried != tc.carried {
				t.Errorf("coverage = %d/%d, want %d/%d",
					got.Coverage.Carried, got.Coverage.Declared, tc.carried, tc.declared)
			}
			if tc.frontier == "" {
				if got.Frontier != nil {
					t.Fatalf("frontier = %+v, want nil", got.Frontier)
				}
				return
			}
			if got.Frontier == nil {
				t.Fatalf("frontier = nil, want phase %q", tc.frontier)
			}
			if got.Frontier.PhaseID != tc.frontier {
				t.Errorf("frontier phase = %q, want %q", got.Frontier.PhaseID, tc.frontier)
			}
			if got.Frontier.Remaining != tc.remaining {
				t.Errorf("frontier remaining = %d, want %d", got.Frontier.Remaining, tc.remaining)
			}
		})
	}
}

// The frontier is the FIRST gap in declared order, not the last witnessed phase.
// A later phase landing early advances the count and leaves the frontier parked —
// that ordering is the whole reason the frontier is a usable next step.
func TestFoldFrontierIsFirstGapNotLastWitness(t *testing.T) {
	got := Fold(Input{Plan: plan("p1", "p2", "p3", "p4"), Witnessed: []string{"p3"}})
	if got.Verdict != VerdictAdvancing {
		t.Fatalf("verdict = %q, want %q", got.Verdict, VerdictAdvancing)
	}
	if got.Frontier == nil || got.Frontier.PhaseID != "p1" {
		t.Fatalf("frontier = %+v, want the first gap p1", got.Frontier)
	}
	if got.Frontier.Index != 0 {
		t.Errorf("frontier index = %d, want 0", got.Frontier.Index)
	}
	if got.Frontier.Remaining != 3 {
		t.Errorf("remaining = %d, want 3 (p1, p2, p4)", got.Frontier.Remaining)
	}
	want := []string{"p1", "p2", "p4"}
	if strings.Join(got.Coverage.Unwitnessed, ",") != strings.Join(want, ",") {
		t.Errorf("unwitnessed = %v, want %v (declared order)", got.Coverage.Unwitnessed, want)
	}
}

// A witness naming no declared phase is surfaced as Foreign and never credited:
// work that landed off the declared plan must not buy depth against it.
func TestFoldForeignWitnessIsNeverCredited(t *testing.T) {
	got := Fold(Input{Plan: plan("p1", "p2"), Witnessed: []string{"p1", "elsewhere", "also-elsewhere", "elsewhere"}})
	if got.Coverage.Carried != 1 {
		t.Fatalf("carried = %d, want 1 — a foreign witness must not count", got.Coverage.Carried)
	}
	want := []string{"elsewhere", "also-elsewhere"}
	if strings.Join(got.Coverage.Foreign, ",") != strings.Join(want, ",") {
		t.Errorf("foreign = %v, want %v (first-appearance order, deduped)", got.Coverage.Foreign, want)
	}
	// Foreign work alone does not block a close: the plan under-describing the
	// work is a planning signal, not a depth defect.
	full := Fold(Input{Plan: plan("p1"), Witnessed: []string{"p1", "elsewhere"}})
	if d := Admit(Input{Plan: plan("p1"), Witnessed: []string{"p1", "elsewhere"}}, ClosureMet); !d.Admitted {
		t.Errorf("met refused with a foreign witness on a fully carried plan: %s", d.Detail)
	}
	if len(full.Coverage.Foreign) != 1 {
		t.Errorf("foreign = %v, want it still reported on a carried plan", full.Coverage.Foreign)
	}
}

// Normalization is fail-closed: whitespace is trimmed, a repeated declared id
// counts once (so a duplicated phase cannot inflate the denominator), and a blank
// id is Malformed rather than a silently-satisfied phase.
func TestFoldNormalization(t *testing.T) {
	t.Run("trims whitespace on both sides", func(t *testing.T) {
		got := Fold(Input{Plan: []Phase{{ID: "  p1  "}}, Witnessed: []string{" p1 "}})
		if got.Verdict != VerdictCarried {
			t.Fatalf("verdict = %q, want %q — trimmed ids must match", got.Verdict, VerdictCarried)
		}
	})
	t.Run("a duplicated declared id counts once", func(t *testing.T) {
		got := Fold(Input{Plan: plan("p1", "p1", "p2"), Witnessed: []string{"p1", "p2"}})
		if got.Coverage.Declared != 2 {
			t.Fatalf("declared = %d, want 2 — a repeat must not inflate the denominator", got.Coverage.Declared)
		}
		if got.Verdict != VerdictCarried {
			t.Errorf("verdict = %q, want %q", got.Verdict, VerdictCarried)
		}
	})
	t.Run("a blank declared id is malformed, not declared", func(t *testing.T) {
		got := Fold(Input{Plan: []Phase{{ID: "p1"}, {ID: "   "}}, Witnessed: []string{"p1"}})
		if got.Coverage.Malformed != 1 {
			t.Fatalf("malformed = %d, want 1", got.Coverage.Malformed)
		}
		if got.Coverage.Declared != 1 {
			t.Errorf("declared = %d, want 1 — a blank id is not a declarable phase", got.Coverage.Declared)
		}
	})
	t.Run("a blank witnessed id credits nothing", func(t *testing.T) {
		got := Fold(Input{Plan: plan("p1"), Witnessed: []string{"", "   "}})
		if got.Coverage.Carried != 0 {
			t.Fatalf("carried = %d, want 0", got.Coverage.Carried)
		}
		if len(got.Coverage.Foreign) != 0 {
			t.Errorf("foreign = %v, want empty — a blank id is dropped, not reported as foreign", got.Coverage.Foreign)
		}
	})
}

// Fold is total and deterministic: same input, same report, and every input lands
// on a member of the closed vocabulary.
func TestFoldIsTotalAndDeterministic(t *testing.T) {
	inputs := []Input{
		{},
		{Plan: plan("a")},
		{Plan: plan("a", "b"), Witnessed: []string{"b"}},
		{Plan: []Phase{{ID: ""}}, Witnessed: []string{"x"}},
		{Witnessed: []string{"x", "y"}},
	}
	for i, in := range inputs {
		first, second := Fold(in), Fold(in)
		if !ValidVerdict(first.Verdict) {
			t.Errorf("input %d: verdict %q is outside the closed vocabulary", i, first.Verdict)
		}
		a, _ := json.Marshal(first)
		b, _ := json.Marshal(second)
		if string(a) != string(b) {
			t.Errorf("input %d: fold is not deterministic:\n %s\n %s", i, a, b)
		}
	}
}

// The core drive: `met` costs a carried plan. This is the hole the fold exists to
// close — before it, `fak trajctl close --status met` wrote `met` on a six-phase
// plan with one phase witnessed and nothing objected.
func TestAdmitMetRequiresACarriedPlan(t *testing.T) {
	cases := []struct {
		name    string
		in      Input
		admit   bool
		mention string // substring the detail must name
	}{
		{
			name:    "undeclared plan is refused: the claim is uncheckable",
			in:      Input{},
			admit:   false,
			mention: "no plan is declared",
		},
		{
			name:    "shallow is refused and names the frontier",
			in:      Input{Plan: plan("p1", "p2", "p3")},
			admit:   false,
			mention: `"p1"`,
		},
		{
			name:    "advancing is refused and names the frontier",
			in:      Input{Plan: plan("p1", "p2", "p3"), Witnessed: []string{"p1"}},
			admit:   false,
			mention: `"p2"`,
		},
		{
			name:  "carried is admitted",
			in:    Input{Plan: plan("p1", "p2"), Witnessed: []string{"p1", "p2"}},
			admit: true,
		},
		{
			name:    "a malformed plan is refused even when every real phase is carried",
			in:      Input{Plan: []Phase{{ID: "p1"}, {ID: ""}}, Witnessed: []string{"p1"}},
			admit:   false,
			mention: "blank id",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := Admit(tc.in, ClosureMet)
			if d.Admitted != tc.admit {
				t.Fatalf("admitted = %v, want %v (detail: %s)", d.Admitted, tc.admit, d.Detail)
			}
			if tc.admit {
				if d.Reason != "" {
					t.Errorf("reason = %q, want empty on an admission", d.Reason)
				}
				return
			}
			if d.Reason != RefusalReason {
				t.Errorf("reason = %q, want the closed token %q", d.Reason, RefusalReason)
			}
			if tc.mention != "" && !strings.Contains(d.Detail, tc.mention) {
				t.Errorf("detail %q does not name %q", d.Detail, tc.mention)
			}
		})
	}
}

// Abandoning is never refused — refusing it would only trap dead objectives open
// — but the depth reached is still recorded at the moment of the drop.
func TestAdmitAbandonIsNeverRefusedButIsRecorded(t *testing.T) {
	for _, in := range []Input{
		{},
		{Plan: plan("p1", "p2", "p3")},
		{Plan: plan("p1", "p2"), Witnessed: []string{"p1"}},
		{Plan: []Phase{{ID: ""}}},
	} {
		d := Admit(in, ClosureAbandoned)
		if !d.Admitted {
			t.Fatalf("abandon refused for %+v: %s", in, d.Detail)
		}
		if d.Reason != "" {
			t.Errorf("reason = %q, want empty", d.Reason)
		}
		if d.Report.Verdict == "" || !ValidVerdict(d.Report.Verdict) {
			t.Errorf("abandon must still carry a valid depth report, got %q", d.Report.Verdict)
		}
	}
	// The recorded depth is the real one, not a zero.
	d := Admit(Input{Plan: plan("p1", "p2", "p3"), Witnessed: []string{"p1", "p2"}}, ClosureAbandoned)
	if d.Report.Coverage.Carried != 2 || d.Report.Coverage.Declared != 3 {
		t.Errorf("recorded depth = %d/%d, want 2/3", d.Report.Coverage.Carried, d.Report.Coverage.Declared)
	}
}

// A closure outside the vocabulary fails closed rather than falling into the
// permissive arm.
func TestAdmitUnknownClosureFailsClosed(t *testing.T) {
	carried := Input{Plan: plan("p1"), Witnessed: []string{"p1"}}
	d := Admit(carried, Closure("retired"))
	if d.Admitted {
		t.Fatal("an unrecognized closure was admitted; it must fail closed")
	}
	if d.Reason != RefusalReason {
		t.Errorf("reason = %q, want %q", d.Reason, RefusalReason)
	}
	if ValidClosure(Closure("retired")) {
		t.Error("ValidClosure accepted a foreign value")
	}
	if !ValidClosure(ClosureMet) || !ValidClosure(ClosureAbandoned) {
		t.Error("ValidClosure rejected a member of its own vocabulary")
	}
}

// The allow-depth half: frontier movement separates deep work from thrash where a
// bare attempt counter cannot.
func TestPersist(t *testing.T) {
	fold := func(carried ...string) Report {
		return Fold(Input{Plan: plan("p1", "p2", "p3"), Witnessed: carried})
	}
	cases := []struct {
		name    string
		earlier Report
		later   Report
		want    Persistence
		delta   int
	}{
		{
			name:    "a phase carried that was not before is depth",
			earlier: fold("p1"),
			later:   fold("p1", "p2"),
			want:    PersistenceAdvanced,
			delta:   1,
		},
		{
			name:    "the same phases carried is thrash-shaped",
			earlier: fold("p1"),
			later:   fold("p1"),
			want:    PersistenceStuck,
		},
		{
			name:    "a lost witness is worse than stuck",
			earlier: fold("p1", "p2"),
			later:   fold("p1"),
			want:    PersistenceRegressed,
			delta:   -1,
		},
		{
			name:    "an undeclared side abstains rather than guessing",
			earlier: Fold(Input{}),
			later:   fold("p1"),
			want:    PersistenceUnknown,
		},
		{
			name:    "an undeclared later side also abstains",
			earlier: fold("p1"),
			later:   Fold(Input{}),
			want:    PersistenceUnknown,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Persist(tc.earlier, tc.later)
			if got.Persistence != tc.want {
				t.Fatalf("persistence = %q, want %q (detail: %s)", got.Persistence, tc.want, got.Detail)
			}
			if !ValidPersistence(got.Persistence) {
				t.Fatalf("persistence %q is outside the closed vocabulary", got.Persistence)
			}
			if got.CarriedDelta != tc.delta {
				t.Errorf("carried delta = %d, want %d", got.CarriedDelta, tc.delta)
			}
		})
	}
}

// Re-planning deeper is not progress: declaring more phases must not be the cheap
// way to buy another attempt.
func TestPersistAGrownPlanAloneIsNotAdvancing(t *testing.T) {
	earlier := Fold(Input{Plan: plan("p1", "p2"), Witnessed: []string{"p1"}})
	later := Fold(Input{Plan: plan("p1", "p2", "p3", "p4"), Witnessed: []string{"p1"}})
	got := Persist(earlier, later)
	if got.Persistence != PersistenceStuck {
		t.Fatalf("persistence = %q, want %q — declaring more phases is not carrying them",
			got.Persistence, PersistenceStuck)
	}
}

// Out-of-order progress still counts. The measure is carried COUNT, not frontier
// index, precisely so a later phase landing first is not scored as thrash.
func TestPersistOutOfOrderProgressCounts(t *testing.T) {
	earlier := Fold(Input{Plan: plan("p1", "p2", "p3"), Witnessed: []string{}})
	later := Fold(Input{Plan: plan("p1", "p2", "p3"), Witnessed: []string{"p3"}})
	if earlier.Frontier.PhaseID != later.Frontier.PhaseID {
		t.Fatalf("precondition: the frontier should be parked at p1 on both sides, got %q -> %q",
			earlier.Frontier.PhaseID, later.Frontier.PhaseID)
	}
	got := Persist(earlier, later)
	if got.Persistence != PersistenceAdvanced {
		t.Fatalf("persistence = %q, want %q — a parked frontier with more carried is still ground bought",
			got.Persistence, PersistenceAdvanced)
	}
}

// The handoff line names the concrete next phase, never a restatement of the goal:
// a successor handed the frontier continues, one handed the goal re-plans.
func TestHandoffLine(t *testing.T) {
	t.Run("mid-line names the next phase, its title, and what remains", func(t *testing.T) {
		r := Fold(Input{
			Plan:      []Phase{{ID: "p1", Title: "declare"}, {ID: "p2", Title: "wire the gate"}, {ID: "p3"}},
			Witnessed: []string{"p1"},
		})
		got := HandoffLine("obj-7", r)
		for _, want := range []string{"obj-7", "1/3", `"p2"`, "wire the gate", "2 remaining"} {
			if !strings.Contains(got, want) {
				t.Errorf("handoff line %q does not name %q", got, want)
			}
		}
	})
	t.Run("a carried line says so and offers no next phase", func(t *testing.T) {
		got := HandoffLine("obj-7", Fold(Input{Plan: plan("p1"), Witnessed: []string{"p1"}}))
		if !strings.Contains(got, "complete") {
			t.Errorf("handoff line %q does not report completion", got)
		}
		if strings.Contains(got, "next phase") {
			t.Errorf("handoff line %q offers a next phase on a carried plan", got)
		}
	})
	t.Run("an undeclared plan asks for the plan", func(t *testing.T) {
		got := HandoffLine("obj-7", Fold(Input{}))
		if !strings.Contains(got, "declares no plan") {
			t.Errorf("handoff line %q does not name the missing plan", got)
		}
	})
	t.Run("an empty objective id still renders", func(t *testing.T) {
		if got := HandoffLine("  ", Fold(Input{Plan: plan("p1")})); !strings.Contains(got, "unnamed") {
			t.Errorf("handoff line %q does not handle a blank objective id", got)
		}
	})
}

// The report round-trips through JSON: it is a wire value a successor process
// reads, not an in-memory convenience.
func TestReportJSONRoundTrip(t *testing.T) {
	want := Fold(Input{Plan: []Phase{{ID: "p1", Title: "declare"}, {ID: "p2"}}, Witnessed: []string{"p1", "stray"}})
	blob, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got Report
	if err := json.Unmarshal(blob, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Verdict != want.Verdict || got.Coverage.Carried != want.Coverage.Carried {
		t.Fatalf("round trip lost data: %+v != %+v", got, want)
	}
	if got.Frontier == nil || got.Frontier.PhaseID != "p2" {
		t.Fatalf("round trip lost the frontier: %+v", got.Frontier)
	}
	if len(got.Coverage.Foreign) != 1 || got.Coverage.Foreign[0] != "stray" {
		t.Errorf("round trip lost the foreign witness: %v", got.Coverage.Foreign)
	}
}
