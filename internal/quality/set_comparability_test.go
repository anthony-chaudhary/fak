package quality

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Tests for the set-wide comparability gate (#5660). The load-bearing one is
// TestEvaluateSetComparability_SetWideRefusesWhatTheFavoredPairAccepts: it runs the
// SHIPPED path twice over the same committed fixture — once on the two arms an
// operator would actually be reading, once on the complete selected set — and shows
// the pair certifying while the set refuses. That single pair of calls is the whole
// argument for the layer, because it demonstrates the defect (a pairwise-clean,
// set-confounded campaign) rather than asserting that it exists.

// scFixture is the committed on-disk shape of one campaign: the declaration and the
// COMPLETE selected arm set. Tests drive the shipped path from these files rather
// than from inline literals so the fixtures are auditable artifacts in their own
// right — the same convention post_selection_test.go uses.
type scFixture struct {
	Name     string              `json:"name"`
	Note     string              `json:"note"`
	Campaign CampaignDeclaration `json:"campaign"`
	Arms     []CampaignArm       `json:"arms"`
}

func loadSCFixture(t *testing.T, name string) scFixture {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "set_comparability", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	var f scFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatalf("decode fixture %s: %v", name, err)
	}
	if f.Note == "" {
		t.Fatalf("fixture %s carries no note: a fixture that does not say what it proves is not an artifact", name)
	}
	return f
}

// findingFor returns the first finding with the given code, or false.
func findingFor(d ComparabilityDecision, code ComparabilityFindingCode) (ComparabilityFinding, bool) {
	for _, f := range d.Findings {
		if f.Code == code {
			return f, true
		}
	}
	return ComparabilityFinding{}, false
}

// armByID pulls one arm out of a fixture by id so a test can name the arms it means.
func armByID(t *testing.T, arms []CampaignArm, id string) CampaignArm {
	t.Helper()
	for _, a := range arms {
		if a.ID == id {
			return a
		}
	}
	t.Fatalf("fixture has no arm %q", id)
	return CampaignArm{}
}

// The proof artifact #5660 asks for, both halves in one test: the favored pair of a
// three-arm campaign is spotless and certifies, and the SAME campaign evaluated over
// its complete selected set refuses and names the third arm. If this ever passes on
// the full set, the gate has silently degraded back into a pairwise check.
func TestEvaluateSetComparability_SetWideRefusesWhatTheFavoredPairAccepts(t *testing.T) {
	f := loadSCFixture(t, "third_arm_breaks_hold.json")

	// Half one: the two arms an operator actually reads the delta between. This has
	// to PASS, or the fixture would not be demonstrating the defect — it would just
	// be a broken campaign that any check catches.
	pair := []CampaignArm{
		armByID(t, f.Arms, "baseline-greedy"),
		armByID(t, f.Arms, "candidate-topp"),
	}
	pd := EvaluateSetComparability(f.Campaign, pair)
	if pd.Verdict != SetComparable {
		t.Fatalf("favored pair: verdict = %q, want %q — the fixture must be pairwise-clean for the set-wide refusal below to mean anything\n%s",
			pd.Verdict, SetComparable, ExplainSetComparability(pd))
	}

	// Half two: the complete selected set, which is what the campaign actually ran.
	d := EvaluateSetComparability(f.Campaign, f.Arms)
	if d.Verdict != SetNotComparable {
		t.Fatalf("full set: verdict = %q, want %q\n%s", d.Verdict, SetNotComparable, ExplainSetComparability(d))
	}
	if d.Comparable {
		t.Fatalf("full set: Comparable = true on a %q verdict — the boolean read must fail closed", d.Verdict)
	}
	if d.SetSize != 3 {
		t.Fatalf("SetSize = %d, want 3: the decision must record the COMPLETE set it judged", d.SetSize)
	}

	// The finding has to name the exact axis and the exact arms (#5660 done
	// condition), not merely report that something differed.
	fd, ok := findingFor(d, FindingHeldAxisDiffers)
	if !ok {
		t.Fatalf("no %q finding\n%s", FindingHeldAxisDiffers, ExplainSetComparability(d))
	}
	if fd.Axis != "tokenizer" {
		t.Fatalf("differing axis = %q, want %q", fd.Axis, "tokenizer")
	}
	for _, want := range []string{"baseline-greedy", "candidate-topp", "candidate-minp"} {
		if !strings.Contains(fd.Detail, want) {
			t.Fatalf("finding detail does not name arm %q: %s", want, fd.Detail)
		}
	}
	// The partition, not just the arm list: which arms hold which value is the thing
	// that tells an operator whether the odd one out is the third arm or the pair.
	if !strings.Contains(fd.Detail, "tokenizer-v2") || !strings.Contains(fd.Detail, "tokenizer-v1") {
		t.Fatalf("finding detail does not report the value partition: %s", fd.Detail)
	}
	// Only tokenizer differs — a gate that flagged the matched axes too would be
	// noise rather than localization.
	if n := countFindings(d.Findings, FindingHeldAxisDiffers); n != 1 {
		t.Fatalf("got %d differing-axis findings, want exactly 1 (only tokenizer varies)\n%s", n, ExplainSetComparability(d))
	}
}

// The other half of the proof artifact: a fully matched three-arm campaign must
// PASS. Without this a gate that refuses unconditionally would look correct.
func TestEvaluateSetComparability_FullyMatchedThreeArmCampaignPasses(t *testing.T) {
	f := loadSCFixture(t, "fully_matched.json")
	d := EvaluateSetComparability(f.Campaign, f.Arms)
	if d.Verdict != SetComparable || !d.Comparable {
		t.Fatalf("verdict = %q (comparable=%v), want %q\n%s", d.Verdict, d.Comparable, SetComparable, ExplainSetComparability(d))
	}
	if len(d.Findings) != 0 {
		t.Fatalf("a fully matched campaign produced %d finding(s):\n%s", len(d.Findings), ExplainSetComparability(d))
	}
	if d.Controls != 1 {
		t.Fatalf("Controls = %d, want 1", d.Controls)
	}
	for _, a := range d.Arms {
		if !a.Typed {
			t.Fatalf("arm %q untyped in a clean fixture: %s", a.ID, a.Reason)
		}
	}
	// The bound must state what the pass is limited to; a pass that reads as
	// unconditional is the failure mode this whole cohort is built against.
	if !strings.Contains(d.Bound, "did not declare") {
		t.Fatalf("bound does not scope the pass to the DECLARED axes: %q", d.Bound)
	}
}

// Unknown dominates different: the fixture has both an unwitnessed tokenizer and a
// provably differing engine, and the verdict must be could_not_establish — while
// still reporting the difference it did prove.
func TestEvaluateSetComparability_UnwitnessedAxisCannotBeGuessedPast(t *testing.T) {
	f := loadSCFixture(t, "unwitnessed_axis.json")
	d := EvaluateSetComparability(f.Campaign, f.Arms)
	if d.Verdict != SetCouldNotEstablish {
		t.Fatalf("verdict = %q, want %q\n%s", d.Verdict, SetCouldNotEstablish, ExplainSetComparability(d))
	}
	if d.Comparable {
		t.Fatal("Comparable = true on a could-not-establish verdict")
	}
	un, ok := findingFor(d, FindingHeldAxisUnwitnessed)
	if !ok {
		t.Fatalf("no %q finding\n%s", FindingHeldAxisUnwitnessed, ExplainSetComparability(d))
	}
	if un.Axis != "tokenizer" {
		t.Fatalf("unwitnessed axis = %q, want %q", un.Axis, "tokenizer")
	}
	if len(un.Arms) != 1 || un.Arms[0] != "candidate-minp" {
		t.Fatalf("unwitnessed finding names arms %v, want exactly [candidate-minp]", un.Arms)
	}
	// The proven difference is still reported — the demotion to could-not-establish
	// must not swallow evidence the gate actually has.
	df, ok := findingFor(d, FindingHeldAxisDiffers)
	if !ok {
		t.Fatalf("the provably differing 'engine' axis was dropped from the findings\n%s", ExplainSetComparability(d))
	}
	if df.Axis != "engine" {
		t.Fatalf("differing axis = %q, want %q", df.Axis, "engine")
	}
}

// scPairwiseOracle is an INDEPENDENT statement of the classic two-arm comparability
// rule, written straight from the two-arm reading rather than by calling the shipped
// path: for each held axis, an absent value on either side is unknown and unequal
// values are a difference; unknown wins over difference. The next test asserts the
// shipped set-wide gate reduces to exactly this at |arms| = 2 — that is what
// "preserve existing two-arm behavior as the size-two special case" (#5660) means
// operationally, and asserting it against a re-derived oracle is the only way the
// claim is worth anything.
func scPairwiseOracle(heldAxes []string, a, b CampaignArm) ComparabilityVerdict {
	unknown, differs := false, false
	for _, axis := range heldAxes {
		va, vb := strings.TrimSpace(a.Axes[axis]), strings.TrimSpace(b.Axes[axis])
		if va == "" || vb == "" {
			unknown = true
			continue
		}
		if va != vb {
			differs = true
		}
	}
	switch {
	case unknown:
		return SetCouldNotEstablish
	case differs:
		return SetNotComparable
	default:
		return SetComparable
	}
}

// Exhaustive size-two equivalence: two held axes, each arm's value drawn from
// {unwitnessed, "x", "y"} — all 81 assignments — and the shipped gate must agree
// with the pairwise oracle on every one. The generalization to N arms is only safe
// if it changed nothing at N=2.
func TestEvaluateSetComparability_SizeTwoReducesToThePairwiseRule(t *testing.T) {
	axes := []string{"model", "tokenizer"}
	values := []string{"", "x", "y"}
	decl := CampaignDeclaration{ID: "size-two-equivalence", HeldAxes: axes}

	checked := 0
	for _, am := range values {
		for _, at := range values {
			for _, bm := range values {
				for _, bt := range values {
					a := CampaignArm{ID: "control", Role: RoleControl, Axes: map[string]string{"model": am, "tokenizer": at}}
					b := CampaignArm{ID: "treatment", Role: RoleTreatment, Axes: map[string]string{"model": bm, "tokenizer": bt}}
					got := EvaluateSetComparability(decl, []CampaignArm{a, b}).Verdict
					want := scPairwiseOracle(axes, a, b)
					if got != want {
						t.Fatalf("size-two divergence at model=(%q,%q) tokenizer=(%q,%q): set-wide gate says %q, pairwise rule says %q",
							am, bm, at, bt, got, want)
					}
					checked++
				}
			}
		}
	}
	if checked != 81 {
		t.Fatalf("covered %d assignments, want 81 — the matrix is not exhaustive", checked)
	}
}

// Every typed refusal, each with the defect that produces it. A gate whose refusals
// are untyped forces a consumer to pattern-match prose.
func TestEvaluateSetComparability_TypedRefusals(t *testing.T) {
	ok := map[string]string{"model": "m"}
	decl := CampaignDeclaration{ID: "c", HeldAxes: []string{"model"}}

	cases := []struct {
		name string
		decl CampaignDeclaration
		arms []CampaignArm
		code ComparabilityFindingCode
	}{
		{
			name: "unnamed campaign",
			decl: CampaignDeclaration{HeldAxes: []string{"model"}},
			arms: []CampaignArm{{ID: "a", Role: RoleControl, Axes: ok}, {ID: "b", Role: RoleTreatment, Axes: ok}},
			code: FindingDeclarationInvalid,
		},
		{
			name: "no held axis declared is refused, never vacuously passed",
			decl: CampaignDeclaration{ID: "c"},
			arms: []CampaignArm{{ID: "a", Role: RoleControl, Axes: ok}, {ID: "b", Role: RoleTreatment, Axes: ok}},
			code: FindingDeclarationInvalid,
		},
		{
			name: "a single arm is not a set",
			decl: decl,
			arms: []CampaignArm{{ID: "a", Role: RoleControl, Axes: ok}},
			code: FindingDeclarationInvalid,
		},
		{
			name: "arm with no id",
			decl: decl,
			arms: []CampaignArm{{ID: "a", Role: RoleControl, Axes: ok}, {Role: RoleTreatment, Axes: ok}},
			code: FindingArmRoleUntyped,
		},
		{
			name: "duplicate arm id",
			decl: decl,
			arms: []CampaignArm{{ID: "a", Role: RoleControl, Axes: ok}, {ID: "a", Role: RoleTreatment, Axes: ok}},
			code: FindingArmRoleUntyped,
		},
		{
			name: "role outside the closed set",
			decl: decl,
			arms: []CampaignArm{{ID: "a", Role: RoleControl, Axes: ok}, {ID: "b", Role: "holdout", Axes: ok}},
			code: FindingArmRoleUntyped,
		},
		{
			name: "no control to read treatments against",
			decl: decl,
			arms: []CampaignArm{{ID: "a", Role: RoleTreatment, Axes: ok}, {ID: "b", Role: RoleTreatment, Axes: ok}},
			code: FindingArmRolesUnbalanced,
		},
		{
			name: "two controls leaves the baseline unstated",
			decl: decl,
			arms: []CampaignArm{{ID: "a", Role: RoleControl, Axes: ok}, {ID: "b", Role: RoleControl, Axes: ok}},
			code: FindingArmRolesUnbalanced,
		},
		{
			name: "blank axis value is unwitnessed, not matched",
			decl: decl,
			arms: []CampaignArm{{ID: "a", Role: RoleControl, Axes: ok}, {ID: "b", Role: RoleTreatment, Axes: map[string]string{"model": "   "}}},
			code: FindingHeldAxisUnwitnessed,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := EvaluateSetComparability(tc.decl, tc.arms)
			if d.Verdict != SetCouldNotEstablish {
				t.Fatalf("verdict = %q, want %q\n%s", d.Verdict, SetCouldNotEstablish, ExplainSetComparability(d))
			}
			if _, found := findingFor(d, tc.code); !found {
				t.Fatalf("no %q finding\n%s", tc.code, ExplainSetComparability(d))
			}
			for _, f := range d.Findings {
				if strings.TrimSpace(f.Detail) == "" {
					t.Fatalf("finding %q carries no detail", f.Code)
				}
			}
		})
	}
}

// The zero value must refuse. A contract whose empty input passes is a contract that
// certifies anything a caller forgets to fill in.
func TestEvaluateSetComparability_ZeroValueFailsClosed(t *testing.T) {
	d := EvaluateSetComparability(CampaignDeclaration{}, nil)
	if d.Verdict != SetCouldNotEstablish || d.Comparable {
		t.Fatalf("zero value: verdict = %q comparable = %v, want %q/false", d.Verdict, d.Comparable, SetCouldNotEstablish)
	}
	if len(d.Findings) == 0 {
		t.Fatal("zero value produced no findings: the refusal has to say why")
	}
	if d.Schema != SetComparabilitySchema {
		t.Fatalf("Schema = %q, want %q even on a refusal", d.Schema, SetComparabilitySchema)
	}
}

// A duplicated held axis must not double-report one defect.
func TestEvaluateSetComparability_RepeatedHeldAxisReportedOnce(t *testing.T) {
	decl := CampaignDeclaration{ID: "c", HeldAxes: []string{"model", " model ", "model"}}
	arms := []CampaignArm{
		{ID: "a", Role: RoleControl, Axes: map[string]string{"model": "x"}},
		{ID: "b", Role: RoleTreatment, Axes: map[string]string{"model": "y"}},
	}
	d := EvaluateSetComparability(decl, arms)
	if len(d.HeldAxes) != 1 {
		t.Fatalf("HeldAxes = %v, want the declared list de-duplicated to one entry", d.HeldAxes)
	}
	if n := countFindings(d.Findings, FindingHeldAxisDiffers); n != 1 {
		t.Fatalf("got %d differing-axis findings for one repeated axis, want 1", n)
	}
}

// The decision must be a pure function of its inputs: same campaign, byte-identical
// JSON. Map iteration over the axes or the arm partition would break this, and a
// verdict that does not replay cannot be an audit artifact.
func TestEvaluateSetComparability_ReplaysByteIdentically(t *testing.T) {
	f := loadSCFixture(t, "third_arm_breaks_hold.json")
	first, err := json.Marshal(EvaluateSetComparability(f.Campaign, f.Arms))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for i := 0; i < 32; i++ {
		again, err := json.Marshal(EvaluateSetComparability(f.Campaign, f.Arms))
		if err != nil {
			t.Fatalf("marshal %d: %v", i, err)
		}
		if string(again) != string(first) {
			t.Fatalf("run %d differs from run 0:\n first: %s\n again: %s", i, first, again)
		}
	}
}

// The public shape has to survive the JSON boundary a consumer reads it across.
func TestComparabilityDecision_PublicShapeRoundTrips(t *testing.T) {
	f := loadSCFixture(t, "third_arm_breaks_hold.json")
	d := EvaluateSetComparability(f.Campaign, f.Arms)
	raw, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back ComparabilityDecision
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	again, err := json.Marshal(back)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	if string(again) != string(raw) {
		t.Fatalf("round trip lost a field:\n before: %s\n  after: %s", raw, again)
	}
	if back.Schema != SetComparabilitySchema {
		t.Fatalf("Schema = %q, want %q", back.Schema, SetComparabilitySchema)
	}
}

// The operator readout must carry the set size, every arm, and every finding —
// including on a pass, where the bound is the only thing that scopes the claim.
func TestExplainSetComparability_NamesTheSetAndEveryFinding(t *testing.T) {
	for _, name := range []string{"third_arm_breaks_hold.json", "fully_matched.json", "unwitnessed_axis.json"} {
		t.Run(name, func(t *testing.T) {
			f := loadSCFixture(t, name)
			d := EvaluateSetComparability(f.Campaign, f.Arms)
			out := ExplainSetComparability(d)
			if !strings.Contains(out, f.Campaign.ID) {
				t.Fatalf("readout does not name the campaign:\n%s", out)
			}
			for _, a := range f.Arms {
				if !strings.Contains(out, a.ID) {
					t.Fatalf("readout drops arm %q:\n%s", a.ID, out)
				}
			}
			for _, fd := range d.Findings {
				if !strings.Contains(out, string(fd.Code)) {
					t.Fatalf("readout drops finding %q:\n%s", fd.Code, out)
				}
			}
			if !strings.Contains(out, d.Bound) {
				t.Fatalf("readout drops the bound:\n%s", out)
			}
		})
	}
}

// The zero-value decision must still render as a refusal rather than panicking or
// printing an empty pass.
func TestExplainSetComparability_ZeroValueRendersARefusal(t *testing.T) {
	out := ExplainSetComparability(ComparabilityDecision{})
	if !strings.Contains(out, "COULD NOT ESTABLISH") {
		t.Fatalf("zero-value readout is not a refusal:\n%s", out)
	}
	if !strings.Contains(out, "(unnamed campaign)") {
		t.Fatalf("zero-value readout does not flag the missing campaign id:\n%s", out)
	}
}
