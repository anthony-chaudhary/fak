package quality

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Tests for the post-selection controller (#5666). The load-bearing one is the
// NULL SIMULATION: it measures, on the shipped PriceSearch path, how often a sweep
// over m purely-null arms hands back a "certified improvement". Under the global
// null every certification is false by construction, so the measured rate IS the
// error rate — and it lands on 1-(1-alpha)^m for the unadjusted read while the
// priced read stays under alpha. That single measurement is the whole argument for
// the layer: the gap between those two numbers is the passing claim that search
// breadth manufactures.

const (
	psAlpha  = 0.05
	psTrials = 20000
	psSeed   = uint64(5666_2026)
	// psSlack is Monte-Carlo headroom. At 20000 trials the standard error of a rate
	// near 0.3 is sqrt(.3*.7/20000) ~= 0.0032, so 0.012 is nearly four standard
	// errors — loose enough that the assertion is about the procedure rather than
	// the draw, tight enough that the 0.05 vs 0.34 separation cannot hide inside it.
	// The PRNG is seed-pinned, so the measured value is identical on every run.
	psSlack = 0.012
	// psFuzz is the tolerance on a p-value arithmetic assertion. The adjustments are
	// exact scalings here, but comparing floats on the nose is a brittle way to
	// assert a statistical claim.
	psFuzz = 1e-12
)

const psReceiptSchema = "fak-quality-postselection-receipt/1"

var psFixtures = []string{
	"known_positive.json",
	"manufactured_by_breadth.json",
	"refused_file_drawer.json",
}

// psFixture is the committed on-disk shape of one sweep: the declaration and the
// complete family of arms it was drawn from. Tests drive the shipped path from
// these files rather than from inline literals so the fixtures are auditable
// artifacts in their own right.
type psFixture struct {
	Name          string             `json:"name"`
	Note          string             `json:"note"`
	Certification SweepCertification `json:"certification"`
	Family        []SearchArm        `json:"family"`
}

// psReceiptCase is one entry of the committed receipt: which fixture produced it
// and the full machine-readable decision the shipped contract returned.
type psReceiptCase struct {
	Fixture  string                `json:"fixture"`
	Name     string                `json:"name"`
	Note     string                `json:"note"`
	Decision PostSelectionDecision `json:"decision"`
}

type psReceipt struct {
	Schema string          `json:"schema"`
	Note   string          `json:"note"`
	Cases  []psReceiptCase `json:"cases"`
}

func psLoad(t *testing.T, name string) psFixture {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "post_selection", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	var f psFixture
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		t.Fatalf("decode fixture %s: %v", name, err)
	}
	return f
}

func psHasRefusal(d PostSelectionDecision, code PostSelectionRefusalCode) bool {
	for _, r := range d.Refusals {
		if r.Code == code {
			return true
		}
	}
	return false
}

// psArm is a clean, fully admissible arm. Refusal tests start from it and break
// exactly one thing, so each assertion isolates the refusal it names.
func psArm(id, point string, p float64) SearchArm {
	return SearchArm{
		ID:    id,
		Point: point,
		P:     p,
		Evidence: Evidence{
			CaseID: "fixture/case",
			State:  StatePass,
			Provenance: EvidenceProvenance{
				Model:     "fixture-7b",
				Tokenizer: "fixture-tokenizer@sha256:tokenizer-v1",
				Engine:    "fak/cpu",
				Oracle:    "greedy-token-diff",
				Revision:  "git:38185e22199bf791df73195b93ba29bbebeed8e3",
				Baseline:  "sha256:baseline-postselection-v1",
			},
		},
	}
}

// psSweep is a two-arm sweep whose winner is genuinely extremal and whose
// declaration is internally consistent — the baseline the refusal table perturbs.
func psSweep() (SweepCertification, []SearchArm) {
	return SweepCertification{
			Alpha: psAlpha, Rule: SelectMinP, Adjustment: AdjustBonferroni,
			DeclaredArms: 2, Winner: "arm-a",
		}, []SearchArm{
			psArm("arm-a", "threshold=0.5", 0.01),
			psArm("arm-b", "threshold=0.6", 0.40),
		}
}

// The known-positive fixture proves the contract is not a refusal machine: a
// four-arm sweep whose retained arm is strong enough to survive being priced over
// the whole family still certifies, and the price it paid is stated explicitly.
func TestPriceSearch_KnownPositiveFixture_SurvivesThePrice(t *testing.T) {
	f := psLoad(t, "known_positive.json")
	d := PriceSearch(f.Certification, f.Family)

	if len(d.Refusals) != 0 {
		t.Fatalf("known-positive fixture refused: %+v", d.Refusals)
	}
	if !d.Certified {
		t.Errorf("Certified = false, want true: adjusted p = %v must clear alpha %v", d.Adjusted, f.Certification.Alpha)
	}
	if !d.UnadjustedCertifies {
		t.Errorf("UnadjustedCertifies = false, want true")
	}
	if d.ManufacturedByBreadth {
		t.Errorf("ManufacturedByBreadth = true: this improvement survived its own price, so it was not manufactured")
	}
	if math.Abs(d.Unadjusted-0.004) > psFuzz {
		t.Errorf("Unadjusted = %v, want 0.004 (the retained arm's raw p)", d.Unadjusted)
	}
	// bonferroni over the complete family of 4: min(1, 4*0.004) = 0.016.
	if math.Abs(d.Adjusted-0.016) > psFuzz {
		t.Errorf("Adjusted = %v, want 0.016 (bonferroni over 4 arms)", d.Adjusted)
	}
	if math.Abs(d.SelectionPrice-0.012) > psFuzz {
		t.Errorf("SelectionPrice = %v, want 0.012 (0.016 - 0.004)", d.SelectionPrice)
	}
	if d.Winner == nil || d.Winner.ID != f.Certification.Winner {
		t.Fatalf("Winner = %+v, want the retained arm %q", d.Winner, f.Certification.Winner)
	}
	if d.FamilySize != 4 {
		t.Errorf("FamilySize = %d, want 4", d.FamilySize)
	}
}

// The adversarial fixture is the anti-vacuity proof. Nothing about it is
// malformed — every arm is admissible, the declaration is internally consistent,
// and the retained arm's raw p = 0.012 clears alpha. The contract therefore cannot
// refuse its way out; it has to actually price the search and return a NEGATIVE
// certification. If this ever certifies, the layer is decoration.
func TestPriceSearch_AdversarialFixture_BreadthCannotManufactureAPass(t *testing.T) {
	f := psLoad(t, "manufactured_by_breadth.json")
	d := PriceSearch(f.Certification, f.Family)

	if len(d.Refusals) != 0 {
		t.Fatalf("adversarial fixture must be PRICED, not refused — a refusal would prove nothing about the pricing: %+v", d.Refusals)
	}
	if !d.UnadjustedCertifies {
		t.Fatalf("UnadjustedCertifies = false: the fixture is only adversarial if the naive read DOES certify (raw p = %v vs alpha %v)", d.Unadjusted, f.Certification.Alpha)
	}
	if d.Certified {
		t.Errorf("Certified = true: an 8-arm sweep retaining p = %v was certified, so search breadth manufactured the pass", d.Unadjusted)
	}
	if !d.ManufacturedByBreadth {
		t.Errorf("ManufacturedByBreadth = false, want true: unadjusted certifies and priced does not is exactly this flag")
	}
	// bonferroni over the complete family of 8: min(1, 8*0.012) = 0.096 > 0.05.
	if math.Abs(d.Adjusted-0.096) > psFuzz {
		t.Errorf("Adjusted = %v, want 0.096 (bonferroni over 8 arms)", d.Adjusted)
	}
	if d.Adjusted <= f.Certification.Alpha {
		t.Errorf("adjusted p %v <= alpha %v: the fixture no longer demonstrates a manufactured pass", d.Adjusted, f.Certification.Alpha)
	}
}

// The refusal fixture is the file drawer: nine thresholds searched, three arms
// handed over. The contract must refuse rather than price the winner over the
// three arms it was shown — and it must preserve the naive read on the record, so
// an auditor can see exactly what claim was withheld.
func TestPriceSearch_FileDrawerFixture_RefusesRatherThanPricingWhatItWasShown(t *testing.T) {
	f := psLoad(t, "refused_file_drawer.json")
	d := PriceSearch(f.Certification, f.Family)

	if !psHasRefusal(d, RefuseFamilyIncomplete) {
		t.Fatalf("want refusal %q, got %+v", RefuseFamilyIncomplete, d.Refusals)
	}
	if d.Certified {
		t.Errorf("Certified = true on a refused certification: the contract must fail closed")
	}
	if d.Adjusted != 1 {
		t.Errorf("Adjusted = %v, want 1 on a refusal: no family was established, so no bound was bought", d.Adjusted)
	}
	if d.SelectionPrice != 0 {
		t.Errorf("SelectionPrice = %v, want 0 on a refusal: an unpriced search has no stated price", d.SelectionPrice)
	}
	if !d.UnadjustedCertifies {
		t.Errorf("UnadjustedCertifies = false: the refusal receipt must preserve the naive read (raw p = %v) that was withheld", d.Unadjusted)
	}
	if d.ManufacturedByBreadth {
		t.Errorf("ManufacturedByBreadth = true on a refusal: a refusal is a different failure and must not be laundered into the priced one")
	}
}

// Every typed refusal in the closed vocabulary, each isolated by breaking exactly
// one thing on an otherwise-clean sweep. A code that no input can produce is a code
// that does not exist, so this table is what keeps the vocabulary honest.
func TestPriceSearch_TypedRefusals(t *testing.T) {
	badP := psArm("arm-b", "threshold=0.6", 0.40)
	badP.P = 1.4
	noProv := psArm("arm-b", "threshold=0.6", 0.40)
	noProv.Evidence.Provenance.Tokenizer = ""
	notProduced := psArm("arm-b", "threshold=0.6", 0.40)
	notProduced.Evidence.State = StateMissing
	noPoint := psArm("arm-b", "", 0.40)

	cases := []struct {
		name  string
		want  PostSelectionRefusalCode
		mutIn func(*SweepCertification, *[]SearchArm)
	}{
		{"alpha out of range", RefuseCertificationInvalid, func(c *SweepCertification, _ *[]SearchArm) { c.Alpha = 0 }},
		{"alpha at 1", RefuseCertificationInvalid, func(c *SweepCertification, _ *[]SearchArm) { c.Alpha = 1 }},
		{"unknown selection rule", RefuseCertificationInvalid, func(c *SweepCertification, _ *[]SearchArm) { c.Rule = "best-looking" }},
		{"unknown adjustment", RefuseCertificationInvalid, func(c *SweepCertification, _ *[]SearchArm) { c.Adjustment = "vibes" }},
		{"no retained arm named", RefuseCertificationInvalid, func(c *SweepCertification, _ *[]SearchArm) { c.Winner = "" }},

		{"empty family", RefuseFamilyUndeclared, func(c *SweepCertification, f *[]SearchArm) { *f = nil }},
		{"family size never declared", RefuseFamilyUndeclared, func(c *SweepCertification, _ *[]SearchArm) { c.DeclaredArms = 0 }},

		{"arms evaluated but not reported", RefuseFamilyIncomplete, func(c *SweepCertification, _ *[]SearchArm) { c.DeclaredArms = 9 }},
		{"breadth understated", RefuseFamilyIncomplete, func(c *SweepCertification, _ *[]SearchArm) { c.DeclaredArms = 1 }},
		{"predeclared but searched", RefuseFamilyIncomplete, func(c *SweepCertification, _ *[]SearchArm) { c.Rule = SelectPredeclared }},

		{"winner absent from family", RefuseWinnerNotInFamily, func(c *SweepCertification, _ *[]SearchArm) { c.Winner = "arm-ghost" }},

		{"winner is not the minimum", RefuseWinnerNotExtremal, func(c *SweepCertification, _ *[]SearchArm) { c.Winner = "arm-b" }},

		{"p-value out of range", RefuseArmInadmissible, func(_ *SweepCertification, f *[]SearchArm) { (*f)[1] = badP }},
		{"incomplete provenance", RefuseArmInadmissible, func(_ *SweepCertification, f *[]SearchArm) { (*f)[1] = noProv }},
		{"evidence never produced", RefuseArmInadmissible, func(_ *SweepCertification, f *[]SearchArm) { (*f)[1] = notProduced }},
		{"arm names no searched point", RefuseArmInadmissible, func(_ *SweepCertification, f *[]SearchArm) { (*f)[1] = noPoint }},

		{"multi-arm sweep prices nothing", RefuseAdjustmentUndeclared, func(c *SweepCertification, _ *[]SearchArm) { c.Adjustment = AdjustNone }},
	}

	seen := map[PostSelectionRefusalCode]bool{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cert, family := psSweep()
			tc.mutIn(&cert, &family)
			d := PriceSearch(cert, family)

			if !psHasRefusal(d, tc.want) {
				t.Fatalf("want refusal %q, got %+v", tc.want, d.Refusals)
			}
			if d.Certified {
				t.Errorf("Certified = true alongside refusal %q: the contract must fail closed", tc.want)
			}
			if d.Adjusted != 1 {
				t.Errorf("Adjusted = %v, want 1 on a refusal", d.Adjusted)
			}
			for _, r := range d.Refusals {
				if r.Detail == "" {
					t.Errorf("refusal %q carries no detail: a bare code is not actionable", r.Code)
				}
			}
			if d.Bound == "" {
				t.Errorf("Bound is empty on a refusal: the artifact must say that no bound was established")
			}
		})
		seen[tc.want] = true
	}

	// Every code in the shipped vocabulary must be reachable from some input.
	for _, code := range []PostSelectionRefusalCode{
		RefuseCertificationInvalid, RefuseFamilyUndeclared, RefuseFamilyIncomplete,
		RefuseWinnerNotInFamily, RefuseWinnerNotExtremal, RefuseArmInadmissible,
		RefuseAdjustmentUndeclared,
	} {
		if !seen[code] {
			t.Errorf("refusal code %q is in the vocabulary but no case exercises it", code)
		}
	}
}

// A genuine predeclaration — one point, fixed before the evidence was seen — pays
// nothing, because there was no search to price. This is the boundary the whole
// layer is defined against: it is the ONLY shape entitled to quote its raw p.
func TestPriceSearch_PredeclaredPointCostsNothing(t *testing.T) {
	cert := SweepCertification{
		Alpha: psAlpha, Rule: SelectPredeclared, Adjustment: AdjustBonferroni,
		DeclaredArms: 1, Winner: "arm-a",
	}
	d := PriceSearch(cert, []SearchArm{psArm("arm-a", "threshold=0.5", 0.03)})

	if len(d.Refusals) != 0 {
		t.Fatalf("a single predeclared point refused: %+v", d.Refusals)
	}
	if !d.Certified {
		t.Errorf("Certified = false: a predeclared point at p = %v must certify at alpha %v", d.Unadjusted, psAlpha)
	}
	if d.SelectionPrice != 0 {
		t.Errorf("SelectionPrice = %v, want 0: there was no search to price", d.SelectionPrice)
	}
	if d.Adjusted != d.Unadjusted {
		t.Errorf("Adjusted %v != Unadjusted %v: both procedures must reduce to p exactly at m = 1", d.Adjusted, d.Unadjusted)
	}
	// The same raw p under a one-arm min-p sweep is identical arithmetic — the
	// difference between the two rules is what a wider family would have cost.
	if got := adjustForSelection(AdjustSidak, 0.03, 1); math.Abs(got-0.03) > psFuzz {
		t.Errorf("sidak at m = 1 = %v, want 0.03", got)
	}
}

// The zero value fails closed. A consumer that constructs a decision request from
// unset fields — the shape a partially-wired caller produces — gets a refusal, not
// a vacuous pass.
func TestPriceSearch_ZeroValueFailsClosed(t *testing.T) {
	d := PriceSearch(SweepCertification{}, nil)
	if d.Certified {
		t.Fatalf("the zero-value certification certified: %+v", d)
	}
	if len(d.Refusals) == 0 {
		t.Fatalf("the zero-value certification produced no refusal")
	}
	if d.Schema != PostSelectionSchema {
		t.Errorf("Schema = %q, want %q even on a refusal: the artifact must be identifiable", d.Schema, PostSelectionSchema)
	}
}

// The public result shape. Consumers pin the major schema tag, and a decision must
// round-trip through JSON unchanged — it is a receipt before it is a value.
func TestPostSelectionDecision_PublicShapeRoundTrips(t *testing.T) {
	if PostSelectionSchema != "fak-quality-postselection/1" {
		t.Fatalf("schema tag drifted to %q: a bump is a conscious migration, not a field edit", PostSelectionSchema)
	}
	f := psLoad(t, "known_positive.json")
	d := PriceSearch(f.Certification, f.Family)

	blob, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("marshal decision: %v", err)
	}
	var back PostSelectionDecision
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatalf("unmarshal decision: %v", err)
	}
	again, err := json.Marshal(back)
	if err != nil {
		t.Fatalf("re-marshal decision: %v", err)
	}
	if !bytes.Equal(blob, again) {
		t.Errorf("decision does not round-trip through JSON:\n first: %s\nsecond: %s", blob, again)
	}
	for _, field := range []string{
		`"schema"`, `"unadjusted_p"`, `"selection_adjusted_p"`, `"selection_price"`,
		`"unadjusted_certifies"`, `"certified"`, `"manufactured_by_breadth"`, `"bound"`, `"family_size"`,
	} {
		if !bytes.Contains(blob, []byte(field)) {
			t.Errorf("decision JSON is missing %s: the unadjusted and adjusted results must both be on the artifact", field)
		}
	}
}

// The committed receipt. Regenerate with:
//
//	FAK_UPDATE_POSTSELECTION_RECEIPT=1 go test ./internal/quality -run Receipt
//
// It records the accepted case, the adversarial negative, and the refusal case as
// the shipped path actually decides them, so the artifact cannot drift from the
// code without this test going red.
func TestPriceSearch_ReceiptMatchesCommittedGolden(t *testing.T) {
	receipt := psReceipt{
		Schema: psReceiptSchema,
		Note:   "Post-selection pricing receipt (#5666): each case is a committed sweep fixture run through the shipped PriceSearch path. Regenerate with FAK_UPDATE_POSTSELECTION_RECEIPT=1 go test ./internal/quality -run Receipt.",
	}
	for _, name := range psFixtures {
		f := psLoad(t, name)
		receipt.Cases = append(receipt.Cases, psReceiptCase{
			Fixture:  name,
			Name:     f.Name,
			Note:     f.Note,
			Decision: PriceSearch(f.Certification, f.Family),
		})
	}
	got, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	got = append(got, '\n')

	path := filepath.Join("testdata", "post_selection", "receipt_v1.json")
	if os.Getenv("FAK_UPDATE_POSTSELECTION_RECEIPT") == "1" {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatalf("write receipt: %v", err)
		}
		t.Logf("regenerated %s (%d bytes)", path, len(got))
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read committed receipt: %v", err)
	}
	if !bytes.Equal(bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n")), got) {
		t.Errorf("committed receipt %s is stale — regenerate with FAK_UPDATE_POSTSELECTION_RECEIPT=1 go test ./internal/quality -run Receipt", path)
	}

	// The receipt is only a proof if it actually contains both outcomes.
	var accepted, refused int
	for _, c := range receipt.Cases {
		if c.Decision.Certified {
			accepted++
		}
		if len(c.Decision.Refusals) > 0 {
			refused++
		}
	}
	if accepted == 0 {
		t.Errorf("receipt demonstrates no accepted case")
	}
	if refused == 0 {
		t.Errorf("receipt demonstrates no refusal case")
	}
}

// The measurement the layer exists for. Under the global null a p-value is uniform
// on [0, 1], so every "improvement" a null sweep certifies is false by
// construction. Running the shipped PriceSearch over m null arms therefore reads
// the error rate directly off the two fields: the unadjusted read lands on
// 1-(1-alpha)^m — 34% at m = 8, a gate an operator learns to ignore — while the
// bonferroni-priced read stays under alpha. The difference is not theory; it is
// the count of passing claims that search breadth alone produced.
func TestSelectionBreadthManufacturesPasses_NullSimulation(t *testing.T) {
	const m = 8
	rng := &splitMix64{state: psSeed}

	var unadjusted, priced, manufactured int
	for trial := 0; trial < psTrials; trial++ {
		family := make([]SearchArm, m)
		best, bestP := 0, math.Inf(1)
		for i := range family {
			p := rng.float64()
			family[i] = psArm(psArmID(i), "threshold", p)
			if p < bestP {
				best, bestP = i, p
			}
		}
		cert := SweepCertification{
			Alpha: psAlpha, Rule: SelectMinP, Adjustment: AdjustBonferroni,
			DeclaredArms: m, Winner: psArmID(best),
		}
		d := PriceSearch(cert, family)
		if len(d.Refusals) != 0 {
			t.Fatalf("trial %d: a well-formed null sweep was refused: %+v", trial, d.Refusals)
		}
		if d.UnadjustedCertifies {
			unadjusted++
		}
		if d.Certified {
			priced++
		}
		if d.ManufacturedByBreadth {
			manufactured++
		}
	}

	unadjustedRate := float64(unadjusted) / psTrials
	pricedRate := float64(priced) / psTrials
	wantUnadjusted := 1 - math.Pow(1-psAlpha, m) // 0.3366
	t.Logf("m=%d null sweeps: unadjusted certifies %.4f (analytic %.4f), priced certifies %.4f (alpha %.4f), manufactured %d/%d",
		m, unadjustedRate, wantUnadjusted, pricedRate, psAlpha, manufactured, psTrials)

	if math.Abs(unadjustedRate-wantUnadjusted) > psSlack {
		t.Errorf("unadjusted certification rate %.4f is not the analytic 1-(1-alpha)^%d = %.4f (slack %.4f)",
			unadjustedRate, m, wantUnadjusted, psSlack)
	}
	if unadjustedRate <= psAlpha {
		t.Errorf("unadjusted certification rate %.4f <= alpha %.4f: the simulation is not reproducing the selection effect at all",
			unadjustedRate, psAlpha)
	}
	if pricedRate > psAlpha+psSlack {
		t.Errorf("bonferroni-priced certification rate %.4f exceeds alpha %.4f (slack %.4f): the documented bound does not hold on the shipped path",
			pricedRate, psAlpha, psSlack)
	}
	// Every unadjusted certification of a null sweep IS a manufactured pass unless
	// the price also let it through, so the flag must account for the whole gap.
	if manufactured != unadjusted-priced {
		t.Errorf("ManufacturedByBreadth fired %d times, want %d (unadjusted %d - priced %d): the flag must name exactly the claims breadth produced",
			manufactured, unadjusted-priced, unadjusted, priced)
	}
}

func psArmID(i int) string { return "null-arm-" + string(rune('a'+i)) }

// The operator readout. The renderer's one non-negotiable job is that BOTH readings
// reach the reader on every outcome — a readout that prints only the flattering
// number reintroduces exactly the confusion the decision type removed.
func TestExplainPostSelection_RendersBothReadingsOnEveryOutcome(t *testing.T) {
	cases := []struct {
		name    string
		fixture string
		want    []string
		absent  []string
	}{
		{
			name:    "accepted",
			fixture: "known_positive.json",
			want: []string{
				"CERTIFIED", "unadjusted          p = 0.004", "selection-adjusted  p = 0.016",
				"bonferroni", "price of the search: +0.012", "bound:",
			},
			absent: []string{"MANUFACTURED BY BREADTH", "REFUSED"},
		},
		{
			name:    "priced negative",
			fixture: "manufactured_by_breadth.json",
			want: []string{
				"NOT CERTIFIED", "MANUFACTURED BY BREADTH",
				"unadjusted          p = 0.012", "selection-adjusted  p = 0.096",
			},
			absent: []string{"REFUSED"},
		},
		{
			name:    "refusal",
			fixture: "refused_file_drawer.json",
			want: []string{
				"REFUSED", string(RefuseFamilyIncomplete), "->",
				"unadjusted          p =", "selection-adjusted  p =", "no bound established",
			},
			absent: []string{"MANUFACTURED BY BREADTH", "CERTIFIED  "},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := psLoad(t, tc.fixture)
			out := ExplainPostSelection(PriceSearch(f.Certification, f.Family))
			for _, want := range tc.want {
				if !strings.Contains(out, want) {
					t.Errorf("readout is missing %q:\n%s", want, out)
				}
			}
			for _, absent := range tc.absent {
				if strings.Contains(out, absent) {
					t.Errorf("readout must not contain %q:\n%s", absent, out)
				}
			}
		})
	}
}

// Every typed refusal must survive into the readout with its detail, so an operator
// never has to read the JSON to learn which requirement went unmet.
func TestExplainPostSelection_CarriesEveryRefusalDetail(t *testing.T) {
	cert, family := psSweep()
	cert.DeclaredArms = 9 // the file drawer
	cert.Adjustment = "vibes"
	d := PriceSearch(cert, family)
	if len(d.Refusals) < 2 {
		t.Fatalf("want at least two refusals to test the listing, got %+v", d.Refusals)
	}
	out := ExplainPostSelection(d)
	for _, r := range d.Refusals {
		if !strings.Contains(out, string(r.Code)) {
			t.Errorf("readout drops refusal code %q:\n%s", r.Code, out)
		}
		if !strings.Contains(out, r.Detail) {
			t.Errorf("readout drops the detail for %q:\n%s", r.Code, out)
		}
	}
}

// The zero value renders rather than panicking: a partially-wired caller that prints
// whatever it got must still get a refusal on screen.
func TestExplainPostSelection_ZeroValueRendersARefusal(t *testing.T) {
	out := ExplainPostSelection(PriceSearch(SweepCertification{}, nil))
	if !strings.Contains(out, "REFUSED") {
		t.Errorf("zero-value readout does not say REFUSED:\n%s", out)
	}
	if !strings.Contains(out, "(no arm named)") {
		t.Errorf("zero-value readout does not name the missing retained arm:\n%s", out)
	}
}
