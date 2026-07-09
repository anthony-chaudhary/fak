package scorecard

import (
	"strings"
	"testing"
)

// approxEq compares two rounded 0..1 fractions; the sub-aspect math is deterministic so an
// exact-after-rounding compare is enough, but a tiny epsilon guards float formatting.
func approxEq(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

func TestObservationCompleteness(t *testing.T) {
	n := len(AThemeWitnesses)
	cases := []struct {
		name  string
		fires []int
		want  float64
	}{
		{"no fires is vacuously complete", nil, 1},
		{"all witnesses every fire", []int{n, n, n}, 1},
		{"half on a single fire", []int{n / 2}, float64(n/2) / float64(n)},
		{"mean of two fires", []int{n, 0}, 0.5},
		{"over-count clamps to 1", []int{n + 3}, 1},
	}
	for _, c := range cases {
		if got := ObservationCompleteness(c.fires); !approxEq(got, c.want) {
			t.Errorf("%s: ObservationCompleteness(%v)=%g want %g", c.name, c.fires, got, c.want)
		}
	}
}

func TestValuationBasisHonesty(t *testing.T) {
	cases := []struct {
		figures, withBasis int
		want               float64
	}{
		{0, 0, 1}, // nothing claimed -> nothing unbasis'd
		{4, 4, 1}, // all labelled
		{4, 1, 0.25},
		{4, 0, 0},
		{4, 9, 1}, // over-count clamps
	}
	for _, c := range cases {
		if got := ValuationBasisHonesty(c.figures, c.withBasis); !approxEq(got, c.want) {
			t.Errorf("ValuationBasisHonesty(%d,%d)=%g want %g", c.figures, c.withBasis, got, c.want)
		}
	}
}

func TestAblationCoverage(t *testing.T) {
	total := len(CThemeArms)
	cases := []struct {
		wired int
		want  float64
	}{
		{0, 0},
		{total, 1},
		{total / 2, float64(total/2) / float64(total)},
		{total + 5, 1}, // over-count clamps
	}
	for _, c := range cases {
		if got := AblationCoverage(c.wired); !approxEq(got, c.want) {
			t.Errorf("AblationCoverage(%d)=%g want %g", c.wired, got, c.want)
		}
	}
}

func TestGrossNetDivergence(t *testing.T) {
	cases := []struct {
		gross, net float64
		want       float64
	}{
		{0.034, 0.034, 0},    // aligned -> no divergence
		{0.26, 0.034, 0.226}, // the #2783 defect: 26% gross vs 3.4% net
		{0.034, 0.26, 0.226}, // symmetric in the arguments
		{2.0, 0.0, 1},        // out-of-band shares clamp to max divergence
	}
	for _, c := range cases {
		if got := GrossNetDivergence(c.gross, c.net); !approxEq(got, c.want) {
			t.Errorf("GrossNetDivergence(%g,%g)=%g want %g", c.gross, c.net, got, c.want)
		}
	}
}

// A fully honest corpus: every witness present, every $ figure based, every arm wired, gross
// and net aligned. The four sub-aspects all score 1.0, the D1 headline is clean, debt is 0.
func TestComposeD1Clean(t *testing.T) {
	n := len(AThemeWitnesses)
	p := ComposeD1(CacheValueFacts{
		FireWitnessCounts:      []int{n, n, n},
		DollarFigures:          5,
		DollarFiguresWithBasis: 5,
		AblationArmsWired:      len(CThemeArms),
		FakShareGross:          0.034,
		FakShareNet:            0.034,
	})
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("clean corpus should be OK, got ok=%v verdict=%q", p.OK, p.Verdict)
	}
	if p.Corpus["cachevalue_debt"] != 0 {
		t.Errorf("cachevalue_debt=%v want 0", p.Corpus["cachevalue_debt"])
	}
	if p.Corpus["value"] != 1.0 {
		t.Errorf("D1 headline value=%v want 1.0", p.Corpus["value"])
	}
	if p.Corpus["grade"] != "A" {
		t.Errorf("grade=%v want A", p.Corpus["grade"])
	}
	// each raw sub-aspect is stamped under its exact issue name and is individually 1.0
	for _, name := range []string{"observation_completeness", "valuation_basis_honesty", "ablation_coverage"} {
		if p.Corpus[name] != 1.0 {
			t.Errorf("corpus[%q]=%v want 1.0", name, p.Corpus[name])
		}
	}
	if p.Corpus["gross_net_divergence"] != 0.0 {
		t.Errorf("corpus[gross_net_divergence]=%v want 0.0", p.Corpus["gross_net_divergence"])
	}
}

// Each sub-aspect is INDIVIDUALLY RETIRABLE: degrade exactly one at a time and only that
// sub-aspect's defect appears, so debt == 1 and the corpus key drops below 1.0 for that one
// while the others stay clean.
func TestComposeD1IndividuallyRetirable(t *testing.T) {
	n := len(AThemeWitnesses)
	clean := CacheValueFacts{
		FireWitnessCounts:      []int{n},
		DollarFigures:          4,
		DollarFiguresWithBasis: 4,
		AblationArmsWired:      len(CThemeArms),
		FakShareGross:          0.10,
		FakShareNet:            0.10,
	}

	cases := []struct {
		name       string
		mutate     func(f *CacheValueFacts)
		wantKey    string
		otherClean []string
	}{
		{
			name:       "only observation degrades",
			mutate:     func(f *CacheValueFacts) { f.FireWitnessCounts = []int{n - 1} },
			wantKey:    "observation_completeness",
			otherClean: []string{"valuation_basis_honesty", "ablation_coverage", "gross_net_divergence"},
		},
		{
			name:       "only valuation basis degrades",
			mutate:     func(f *CacheValueFacts) { f.DollarFiguresWithBasis = 1 },
			wantKey:    "valuation_basis_honesty",
			otherClean: []string{"observation_completeness", "ablation_coverage", "gross_net_divergence"},
		},
		{
			name:       "only ablation coverage degrades",
			mutate:     func(f *CacheValueFacts) { f.AblationArmsWired = len(CThemeArms) - 1 },
			wantKey:    "ablation_coverage",
			otherClean: []string{"observation_completeness", "valuation_basis_honesty", "gross_net_divergence"},
		},
		{
			name:       "only gross/net divergence degrades",
			mutate:     func(f *CacheValueFacts) { f.FakShareGross = 0.26; f.FakShareNet = 0.034 },
			wantKey:    "gross_net_divergence",
			otherClean: []string{"observation_completeness", "valuation_basis_honesty", "ablation_coverage"},
		},
	}

	for _, c := range cases {
		f := clean
		c.mutate(&f)
		p := ComposeD1(f)
		if p.OK {
			t.Errorf("%s: expected debt, got OK", c.name)
		}
		if p.Corpus["cachevalue_debt"] != 1 {
			t.Errorf("%s: cachevalue_debt=%v want 1 (exactly one sub-aspect degraded)", c.name, p.Corpus["cachevalue_debt"])
		}
		// the degraded sub-aspect must own the single KPI defect
		var defectKeys []string
		for _, k := range p.KPIs {
			if len(k.Defects) > 0 {
				defectKeys = append(defectKeys, k.Key)
			}
		}
		if len(defectKeys) != 1 || defectKeys[0] != c.wantKey {
			t.Errorf("%s: defect owned by %v want [%s]", c.name, defectKeys, c.wantKey)
		}
		// gross_net_divergence is the one sub-aspect whose corpus value RISES when it degrades
		// (it is a divergence, not a quality fraction); the other three fall below 1.0.
		if c.wantKey != "gross_net_divergence" {
			if v, _ := p.Corpus[c.wantKey].(float64); v >= 1.0 {
				t.Errorf("%s: corpus[%q]=%v want < 1.0", c.name, c.wantKey, v)
			}
		}
	}
}

// Two clones at the same facts must fold identically -- the determinism law.
func TestComposeD1Deterministic(t *testing.T) {
	f := CacheValueFacts{
		FireWitnessCounts:      []int{3, 4, 5},
		DollarFigures:          7,
		DollarFiguresWithBasis: 5,
		AblationArmsWired:      2,
		FakShareGross:          0.20,
		FakShareNet:            0.05,
	}
	a := ComposeD1(f)
	b := ComposeD1(f)
	if a.Corpus["value"] != b.Corpus["value"] || a.Corpus["cachevalue_debt"] != b.Corpus["cachevalue_debt"] {
		t.Errorf("non-deterministic fold: value %v/%v debt %v/%v",
			a.Corpus["value"], b.Corpus["value"], a.Corpus["cachevalue_debt"], b.Corpus["cachevalue_debt"])
	}
}

// The four D1 sub-aspects #2815 names are an enumerable, canonical set: CacheValueSubAspectKeys
// lists exactly the four issue-contract keys in order, D1SubAspectKPIs yields one KPI per key in
// that same order, and folding that slice is byte-identical to ComposeD1 -- so the enumeration
// (the "individually retirable" contract as data) can never drift from the D1 headline.
func TestD1SubAspectKPIsEnumerationMatchesContractAndComposesToD1(t *testing.T) {
	wantKeys := []string{"observation_completeness", "valuation_basis_honesty", "ablation_coverage", "gross_net_divergence"}
	if len(CacheValueSubAspectKeys) != len(wantKeys) {
		t.Fatalf("CacheValueSubAspectKeys=%v want the four #2815 sub-aspects %v", CacheValueSubAspectKeys, wantKeys)
	}
	for i, want := range wantKeys {
		if CacheValueSubAspectKeys[i] != want {
			t.Errorf("CacheValueSubAspectKeys[%d]=%q want %q", i, CacheValueSubAspectKeys[i], want)
		}
	}

	f := CacheValueFacts{
		FireWitnessCounts:      []int{3, 5},
		DollarFigures:          7,
		DollarFiguresWithBasis: 5,
		AblationArmsWired:      2,
		FakShareGross:          0.20,
		FakShareNet:            0.05,
	}
	kpis := D1SubAspectKPIs(f)
	if len(kpis) != len(CacheValueSubAspectKeys) {
		t.Fatalf("D1SubAspectKPIs returned %d KPIs, want %d (one per sub-aspect)", len(kpis), len(CacheValueSubAspectKeys))
	}
	// each KPI is the sub-aspect at the matching canonical position -- individually addressable
	for i, key := range CacheValueSubAspectKeys {
		if kpis[i].Key != key {
			t.Errorf("D1SubAspectKPIs[%d].Key=%q want %q", i, kpis[i].Key, key)
		}
	}
	// folding the enumerated KPIs is identical to ComposeD1 -- enumeration and headline can't drift
	fromEnum := Fold(CacheValueScoreSchema, D1SubAspectKPIs(f), "cachevalue_debt", nil, Messages{Grade: GradeStrict})
	d1 := ComposeD1(f)
	if fromEnum.Corpus["value"] != d1.Corpus["value"] || fromEnum.Corpus["cachevalue_debt"] != d1.Corpus["cachevalue_debt"] {
		t.Errorf("enumerated fold diverged from ComposeD1: value %v/%v debt %v/%v",
			fromEnum.Corpus["value"], d1.Corpus["value"], fromEnum.Corpus["cachevalue_debt"], d1.Corpus["cachevalue_debt"])
	}
}

// honestBaseline is the pinned "last accepted honest" floor the D2 gate ratchets against:
// gross == net (no inflation) and every fak $ figure labelled. It mirrors #2783's corrected
// last-2h corpus (net ~3.4%).
func honestBaseline() CacheValueBaseline {
	return CacheValueBaseline{FakShareGross: 0.034, FakShareNet: 0.034, ValuationBasisHonesty: 1.0}
}

// loneDefectKeys returns the KPI keys that own a defect in a payload -- the fences that red.
func loneDefectKeys(p Payload) []string {
	var keys []string
	for _, k := range p.KPIs {
		if len(k.Defects) > 0 {
			keys = append(keys, k.Key)
		}
	}
	return keys
}

// THE ISSUE ACCEPTANCE (#2820): a merge that reintroduces unlabeled 1.0x-on-warm valuation --
// gross inflates to 26% while the true net stays 3.4%, and a $ figure loses its basis label --
// MUST be blocked by the D2 gate. This is the fails-before/passes-after witness: revert the D2
// fences and this test reds.
func TestComposeD2BlocksUnlabeled1xOnWarmGrossUp(t *testing.T) {
	candidate := CacheValueFacts{
		FireWitnessCounts:      []int{len(AThemeWitnesses)},
		DollarFigures:          4,
		DollarFiguresWithBasis: 3, // one figure reverted to an unlabeled 1.0x-on-warm dollar
		AblationArmsWired:      len(CThemeArms),
		FakShareGross:          0.26,  // 1.0x-on-warm inflates the gross share (the #2783 defect)
		FakShareNet:            0.034, // the true net is unchanged
	}
	block, reason := CacheValueGateBlocks(honestBaseline(), candidate)
	if !block {
		t.Fatalf("gate must BLOCK a reintroduced unlabeled 1.0x-on-warm gross-up; reason=%q", reason)
	}
	p := ComposeD2(honestBaseline(), candidate)
	if p.OK || p.Verdict != "ACTION" {
		t.Fatalf("blocked candidate must be !ok/ACTION, got ok=%v verdict=%q", p.OK, p.Verdict)
	}
	// the reward-hack, the basis regression, and the divergence ceiling all fence it (net held,
	// so net_nonregression stays green): three red fences.
	if got := p.Corpus["cachevalue_gate_debt"]; got != 3 {
		t.Errorf("cachevalue_gate_debt=%v want 3 (gross-up + basis + divergence)", got)
	}
	for _, want := range []string{"gross_up_without_net", "valuation_basis_nonregression", "divergence_ceiling"} {
		if !strings.Contains(reason, want) {
			t.Errorf("block reason missing the %q fence: %s", want, reason)
		}
	}
}

// An honest change -- net genuinely improves over the floor, gross tracks net, every figure
// labelled -- must PASS the gate. The gate is on net, not on gross moving at all.
func TestComposeD2PassesHonestNonRegressingNet(t *testing.T) {
	candidate := CacheValueFacts{
		FireWitnessCounts:      []int{len(AThemeWitnesses)},
		DollarFigures:          4,
		DollarFiguresWithBasis: 4, // every figure labelled
		AblationArmsWired:      len(CThemeArms),
		FakShareGross:          0.05, // gross tracks net...
		FakShareNet:            0.05, // ...and net genuinely improved over the 0.034 floor
	}
	block, reason := CacheValueGateBlocks(honestBaseline(), candidate)
	if block {
		t.Fatalf("gate must PASS an honest net gain (gross tracks net, figures labelled): %s", reason)
	}
	p := ComposeD2(honestBaseline(), candidate)
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("honest candidate must be ok/OK, got ok=%v verdict=%q", p.OK, p.Verdict)
	}
	if got := p.Corpus["cachevalue_gate_debt"]; got != 0 {
		t.Errorf("cachevalue_gate_debt=%v want 0", got)
	}
}

// A net regression alone (headline fak_share_net drops below the pinned floor) blocks the gate
// even with gross tracking net and every figure labelled -- the gate is on the net headline.
func TestComposeD2BlocksNetRegressionAlone(t *testing.T) {
	candidate := CacheValueFacts{
		DollarFigures:          4,
		DollarFiguresWithBasis: 4,
		AblationArmsWired:      len(CThemeArms),
		FakShareGross:          0.02,
		FakShareNet:            0.02, // fell below the 0.034 floor
	}
	p := ComposeD2(honestBaseline(), candidate)
	if p.OK {
		t.Fatalf("a net regression below the pinned floor must block the gate")
	}
	if keys := loneDefectKeys(p); len(keys) != 1 || keys[0] != "net_nonregression" {
		t.Errorf("net regression must own the single defect, got %v", keys)
	}
}

// A basis regression alone -- an unlabeled figure reappears while net holds and gross tracks
// net -- blocks the gate on the basis fence alone (the literal "unlabeled valuation").
func TestComposeD2BlocksBasisRegressionAlone(t *testing.T) {
	candidate := CacheValueFacts{
		DollarFigures:          4,
		DollarFiguresWithBasis: 3, // one figure lost its basis label
		AblationArmsWired:      len(CThemeArms),
		FakShareGross:          0.034,
		FakShareNet:            0.034,
	}
	p := ComposeD2(honestBaseline(), candidate)
	if p.OK {
		t.Fatalf("an unlabeled figure alone must block the gate")
	}
	if got := p.Corpus["cachevalue_gate_debt"]; got != 1 {
		t.Errorf("cachevalue_gate_debt=%v want 1 (basis fence only)", got)
	}
	if keys := loneDefectKeys(p); len(keys) != 1 || keys[0] != "valuation_basis_nonregression" {
		t.Errorf("basis regression must own the single defect, got %v", keys)
	}
}

// CacheValueControlPaneMembers registers BOTH workstream-D scores -- the D1 SCORE card and the
// D2 GATE card -- as control-pane members, each with its own schema and debt key.
func TestCacheValueControlPaneMembersRegistersBothScoreAndGate(t *testing.T) {
	facts := CacheValueFacts{
		FireWitnessCounts:      []int{len(AThemeWitnesses)},
		DollarFigures:          2,
		DollarFiguresWithBasis: 2,
		AblationArmsWired:      len(CThemeArms),
		FakShareGross:          0.034,
		FakShareNet:            0.034,
	}
	members := CacheValueControlPaneMembers(BaselineFromFacts(facts), facts)
	if len(members) != 2 {
		t.Fatalf("want two control-pane members (D1 score + D2 gate), got %d", len(members))
	}
	if members[0].Schema != CacheValueScoreSchema {
		t.Errorf("member[0] schema=%q want the D1 score card %q", members[0].Schema, CacheValueScoreSchema)
	}
	if members[1].Schema != CacheValueGateSchema {
		t.Errorf("member[1] schema=%q want the D2 gate card %q", members[1].Schema, CacheValueGateSchema)
	}
	if _, ok := members[0].Corpus["cachevalue_debt"]; !ok {
		t.Errorf("D1 member missing cachevalue_debt key")
	}
	if _, ok := members[1].Corpus["cachevalue_gate_debt"]; !ok {
		t.Errorf("D2 member missing cachevalue_gate_debt key")
	}
	// pinned against its own facts, the gate is clean -- a corpus never regresses against itself.
	if !members[1].OK {
		t.Errorf("gate pinned to its own baseline must be clean, got reason %q", members[1].Reason)
	}
}

// --- D3: the cache-value ACCURACY-debt scorer (issue #2814) ---

// A fully honest corpus -- gross == net, every $ figure labelled, every A-theme witness captured
// every fire -- carries ZERO accuracy debt and an empty worklist. This is the "clean record
// scores 0" half of the #2814 acceptance.
func TestCacheValueAccuracyDebtCleanScoresZero(t *testing.T) {
	n := len(AThemeWitnesses)
	total, worklist := CacheValueAccuracyDebt(CacheValueFacts{
		FireWitnessCounts:      []int{n, n, n},
		DollarFigures:          5,
		DollarFiguresWithBasis: 5,
		FakShareGross:          0.034,
		FakShareNet:            0.034,
	})
	if total != 0 {
		t.Errorf("clean corpus accuracy debt=%d want 0", total)
	}
	if len(worklist) != 0 {
		t.Errorf("clean corpus worklist=%v want empty", worklist)
	}
}

// THE #2814 WITNESS + committed snapshot: a valuation record with a known gross-net divergence,
// unlabeled valuation_basis figures, AND dropped A-theme witnesses produces a deterministic
// non-zero integer debt and a worst-first worklist pinned to exact values here. This test does
// not compile before CacheValueAccuracyDebt exists (fails-before / passes-after).
func TestCacheValueAccuracyDebtSnapshotAndWorstFirst(t *testing.T) {
	// gross 0.26 vs net 0.034 -> divergence 0.226 -> ceil((0.226-0.05)/0.05) = 4 units.
	// 4 figures, 1 based -> 3 unlabeled units. fires [3,5] over 5 witnesses -> (5-3)+0 = 2 units.
	total, worklist := CacheValueAccuracyDebt(CacheValueFacts{
		FireWitnessCounts:      []int{3, 5},
		DollarFigures:          4,
		DollarFiguresWithBasis: 1,
		FakShareGross:          0.26,
		FakShareNet:            0.034,
	})
	// pinned deterministic integer debt
	if total != 9 {
		t.Fatalf("accuracy debt=%d want 9 (4 divergence + 3 basis + 2 witness)", total)
	}
	// pinned worst-first worklist: divergence(4) > basis(3) > witness(2)
	wantClass := []string{AccuracyClassGrossNetDivergence, AccuracyClassUnlabeledBasis, AccuracyClassMissingWitness}
	wantDebt := []int{4, 3, 2}
	if len(worklist) != len(wantClass) {
		t.Fatalf("worklist len=%d want %d: %v", len(worklist), len(wantClass), worklist)
	}
	for i := range wantClass {
		if worklist[i].Class != wantClass[i] || worklist[i].Debt != wantDebt[i] {
			t.Errorf("worklist[%d]=%s/%d want %s/%d", i, worklist[i].Class, worklist[i].Debt, wantClass[i], wantDebt[i])
		}
	}
}

// Each accuracy class is INDIVIDUALLY RETIRABLE: degrade exactly one and only that class carries
// debt, so total == that class's units and the worklist is the single row.
func TestCacheValueAccuracyDebtIndividuallyRetirable(t *testing.T) {
	n := len(AThemeWitnesses)
	clean := CacheValueFacts{
		FireWitnessCounts:      []int{n},
		DollarFigures:          4,
		DollarFiguresWithBasis: 4,
		FakShareGross:          0.10,
		FakShareNet:            0.10,
	}
	cases := []struct {
		name      string
		mutate    func(f *CacheValueFacts)
		wantClass string
		wantDebt  int
	}{
		{"only divergence", func(f *CacheValueFacts) { f.FakShareGross = 0.26; f.FakShareNet = 0.034 }, AccuracyClassGrossNetDivergence, 4},
		{"only unlabeled basis", func(f *CacheValueFacts) { f.DollarFiguresWithBasis = 1 }, AccuracyClassUnlabeledBasis, 3},
		{"only missing witness", func(f *CacheValueFacts) { f.FireWitnessCounts = []int{n, n - 2, n - 1} }, AccuracyClassMissingWitness, 3},
	}
	for _, c := range cases {
		f := clean
		c.mutate(&f)
		total, worklist := CacheValueAccuracyDebt(f)
		if total != c.wantDebt {
			t.Errorf("%s: total=%d want %d", c.name, total, c.wantDebt)
		}
		if len(worklist) != 1 || worklist[0].Class != c.wantClass || worklist[0].Debt != c.wantDebt {
			t.Errorf("%s: worklist=%v want single [%s/%d]", c.name, worklist, c.wantClass, c.wantDebt)
		}
	}
}

// A sub-threshold gross-vs-net gap is within the healthy band and books NO divergence debt --
// the |gross-net| alarm (DivergenceDefectThreshold) is the same one D1 uses, so a healthy report
// stays clean here too (and the penalty is on |gross-net|, symmetric: net exceeding gross also
// diverges).
func TestCacheValueAccuracyDivergenceBandAndSymmetry(t *testing.T) {
	// within the alarm -> clean
	if u := divergenceDebtUnits(GrossNetDivergence(0.10, 0.06)); u != 0 { // gap 0.04 <= 0.05
		t.Errorf("sub-threshold gap booked %d divergence units want 0", u)
	}
	// net exceeding gross is still a divergence (|gross-net|), not a reward
	over, _ := CacheValueAccuracyDebt(CacheValueFacts{FakShareGross: 0.034, FakShareNet: 0.26, DollarFigures: 1, DollarFiguresWithBasis: 1, FireWitnessCounts: []int{len(AThemeWitnesses)}})
	if over != 4 {
		t.Errorf("net-exceeds-gross divergence debt=%d want 4 (|gross-net| is symmetric)", over)
	}
}

// Ties break by the canonical CacheValueAccuracyClasses order so the worst-first ordering is fully
// deterministic even when two classes carry equal debt.
func TestCacheValueAccuracyDebtTieBreakIsCanonical(t *testing.T) {
	// basis 2, witness 2, divergence 0 -> both tie at 2; canonical order puts basis before witness.
	n := len(AThemeWitnesses)
	_, worklist := CacheValueAccuracyDebt(CacheValueFacts{
		FireWitnessCounts:      []int{n - 2},
		DollarFigures:          4,
		DollarFiguresWithBasis: 2,
		FakShareGross:          0.10,
		FakShareNet:            0.10,
	})
	if len(worklist) != 2 || worklist[0].Class != AccuracyClassUnlabeledBasis || worklist[1].Class != AccuracyClassMissingWitness {
		t.Errorf("tie worklist=%v want [unlabeled_valuation_basis, missing_time_of_event_witness]", worklist)
	}
}

// ComposeD3 folds the accuracy debt into the family payload shape: the WEIGHTED integer debt is
// stamped under corpus["cachevalue_accuracy_debt"] (overriding Fold's raw defect count), the
// worklist is carried, and ok flips iff any class is in debt. The D3 payload is NOT a registered
// control-pane member -- CacheValueControlPaneMembers stays the D1 score + D2 gate (#2820).
func TestComposeD3PayloadStampsWeightedIntegerAndWorklist(t *testing.T) {
	dirty := ComposeD3(CacheValueFacts{
		FireWitnessCounts:      []int{3, 5},
		DollarFigures:          4,
		DollarFiguresWithBasis: 1,
		FakShareGross:          0.26,
		FakShareNet:            0.034,
	})
	if dirty.Schema != CacheValueAccuracyDebtSchema {
		t.Errorf("schema=%q want %q", dirty.Schema, CacheValueAccuracyDebtSchema)
	}
	if dirty.OK || dirty.Verdict != "ACTION" {
		t.Errorf("dirty corpus must be !ok/ACTION, got ok=%v verdict=%q", dirty.OK, dirty.Verdict)
	}
	if got := dirty.Corpus["cachevalue_accuracy_debt"]; got != 9 {
		t.Errorf("cachevalue_accuracy_debt=%v want 9 (weighted integer, not the 3-class defect count)", got)
	}
	wl, ok := dirty.Corpus["accuracy_worklist"].([]AccuracyDebtRow)
	if !ok || len(wl) != 3 || wl[0].Class != AccuracyClassGrossNetDivergence {
		t.Errorf("accuracy_worklist=%v want 3 rows worst-first (divergence first)", dirty.Corpus["accuracy_worklist"])
	}

	clean := ComposeD3(CacheValueFacts{
		FireWitnessCounts:      []int{len(AThemeWitnesses)},
		DollarFigures:          2,
		DollarFiguresWithBasis: 2,
		FakShareGross:          0.034,
		FakShareNet:            0.034,
	})
	if !clean.OK || clean.Verdict != "OK" {
		t.Errorf("clean corpus must be ok/OK, got ok=%v verdict=%q", clean.OK, clean.Verdict)
	}
	if got := clean.Corpus["cachevalue_accuracy_debt"]; got != 0 {
		t.Errorf("clean cachevalue_accuracy_debt=%v want 0", got)
	}

	// D3 must NOT have been registered as a control-pane member: that is #2820, so the roster
	// stays the D1 score + D2 gate (two members), never three.
	facts := CacheValueFacts{FireWitnessCounts: []int{len(AThemeWitnesses)}, DollarFigures: 2, DollarFiguresWithBasis: 2, AblationArmsWired: len(CThemeArms), FakShareGross: 0.034, FakShareNet: 0.034}
	if members := CacheValueControlPaneMembers(BaselineFromFacts(facts), facts); len(members) != 2 {
		t.Errorf("control-pane members=%d want 2 (D3 accuracy scorer is not registered -- that is #2820)", len(members))
	}
}

// Two folds at the same facts must be identical -- the determinism law, extended to D3.
func TestCacheValueAccuracyDebtDeterministic(t *testing.T) {
	f := CacheValueFacts{
		FireWitnessCounts:      []int{3, 4, 5},
		DollarFigures:          7,
		DollarFiguresWithBasis: 5,
		FakShareGross:          0.20,
		FakShareNet:            0.05,
	}
	t1, w1 := CacheValueAccuracyDebt(f)
	t2, w2 := CacheValueAccuracyDebt(f)
	if t1 != t2 || len(w1) != len(w2) {
		t.Fatalf("non-deterministic accuracy fold: total %d/%d worklist %d/%d", t1, t2, len(w1), len(w2))
	}
	for i := range w1 {
		if w1[i] != w2[i] {
			t.Errorf("worklist row %d differs: %v vs %v", i, w1[i], w2[i])
		}
	}
}

// Two folds at the same baseline+facts must be identical -- the determinism law, extended to D2.
func TestComposeD2Deterministic(t *testing.T) {
	b := CacheValueBaseline{FakShareGross: 0.2, FakShareNet: 0.05, ValuationBasisHonesty: 0.9}
	c := CacheValueFacts{DollarFigures: 7, DollarFiguresWithBasis: 5, FakShareGross: 0.3, FakShareNet: 0.04}
	p1 := ComposeD2(b, c)
	p2 := ComposeD2(b, c)
	if p1.OK != p2.OK || p1.Corpus["cachevalue_gate_debt"] != p2.Corpus["cachevalue_gate_debt"] {
		t.Errorf("non-deterministic D2 fold: ok %v/%v debt %v/%v",
			p1.OK, p2.OK, p1.Corpus["cachevalue_gate_debt"], p2.Corpus["cachevalue_gate_debt"])
	}
}
