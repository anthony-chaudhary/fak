package engine

import (
	"math"
	"testing"
)

// parityReading builds a measured sample carrying the axis's own unit, so a case
// only has to state the two numbers under comparison.
func parityReading(axis ParityAxis, fak, rival float64) ParitySample {
	req, _ := ParityRequirementFor(axis)
	return ParitySample{Axis: axis, Measured: true, Unit: req.Unit, Fak: fak, Rival: rival}
}

// TestServingParityVocabularyClosure pins each closed vocabulary to its exact
// membership: the empty string is fail-closed everywhere, and the axis order is
// decode, prefill, warm-ttft (the order every ordered report accessor inherits).
func TestServingParityVocabularyClosure(t *testing.T) {
	t.Parallel()

	axes := ParityAxes()
	if len(axes) != 3 {
		t.Fatalf("ParityAxes() has %d members, want 3: %v", len(axes), axes)
	}
	wantAxes := []ParityAxis{ParityAxisDecode, ParityAxisPrefill, ParityAxisWarmTTFT}
	for i, want := range wantAxes {
		if axes[i] != want {
			t.Errorf("ParityAxes()[%d] = %q, want %q (order is part of the contract)", i, axes[i], want)
		}
	}
	seenAxis := map[ParityAxis]bool{}
	for _, a := range axes {
		if !a.Valid() {
			t.Errorf("closed-vocabulary axis %q reports Valid()==false", a)
		}
		if seenAxis[a] {
			t.Errorf("ParityAxes() returned duplicate axis %q", a)
		}
		seenAxis[a] = true
	}
	for _, bogus := range []ParityAxis{"", "Decode", "warmttft", "ttft", "latency"} {
		if bogus.Valid() {
			t.Errorf("out-of-vocabulary axis %q reports Valid()==true", bogus)
		}
	}

	units := ParityUnits()
	if len(units) != 2 {
		t.Fatalf("ParityUnits() has %d members, want 2: %v", len(units), units)
	}
	wantUnits := map[ParityUnit]bool{ParityUnitTokensPerSecond: true, ParityUnitMilliseconds: true}
	for _, u := range units {
		if !wantUnits[u] {
			t.Errorf("ParityUnits() returned unexpected member %q", u)
		}
		if !u.Valid() {
			t.Errorf("closed-vocabulary unit %q reports Valid()==false", u)
		}
	}
	for _, bogus := range []ParityUnit{"", "tok/s", "MS", "seconds"} {
		if bogus.Valid() {
			t.Errorf("out-of-vocabulary unit %q reports Valid()==true", bogus)
		}
	}

	verdicts := ParityVerdicts()
	if len(verdicts) != 3 {
		t.Fatalf("ParityVerdicts() has %d members, want 3: %v", len(verdicts), verdicts)
	}
	wantVerdicts := map[ParityVerdict]bool{ParityUnmeasured: true, ParityMet: true, ParityMissed: true}
	for _, v := range verdicts {
		if !wantVerdicts[v] {
			t.Errorf("ParityVerdicts() returned unexpected member %q", v)
		}
		if !v.Valid() {
			t.Errorf("closed-vocabulary verdict %q reports Valid()==false", v)
		}
	}
	for _, bogus := range []ParityVerdict{"", "MET", "unknown", "pass"} {
		if bogus.Valid() {
			t.Errorf("out-of-vocabulary verdict %q reports Valid()==true", bogus)
		}
	}
}

// TestServingParityAxisUnitAndDirection pins unit coupling and comparison
// direction per axis: throughput axes are higher-is-better tokens/s, warm-ttft is
// lower-is-better milliseconds, and an invalid axis has neither a valid unit nor a
// direction.
func TestServingParityAxisUnitAndDirection(t *testing.T) {
	t.Parallel()

	cases := []struct {
		axis       ParityAxis
		wantUnit   ParityUnit
		wantHigher bool
	}{
		{ParityAxisDecode, ParityUnitTokensPerSecond, true},
		{ParityAxisPrefill, ParityUnitTokensPerSecond, true},
		{ParityAxisWarmTTFT, ParityUnitMilliseconds, false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(string(tc.axis), func(t *testing.T) {
			t.Parallel()
			if got := tc.axis.Unit(); got != tc.wantUnit {
				t.Errorf("%q.Unit() = %q, want %q", tc.axis, got, tc.wantUnit)
			}
			if got := tc.axis.HigherIsBetter(); got != tc.wantHigher {
				t.Errorf("%q.HigherIsBetter() = %v, want %v", tc.axis, got, tc.wantHigher)
			}
		})
	}

	invalid := ParityAxis("bogus")
	if invalid.HigherIsBetter() {
		t.Errorf("invalid axis %q reports HigherIsBetter()==true", invalid)
	}
	if u := invalid.Unit(); u.Valid() || u == ParityUnitTokensPerSecond || u == ParityUnitMilliseconds {
		t.Errorf("invalid axis %q.Unit() = %q, want a non-valid sentinel unit", invalid, u)
	}
}

// TestServingParityRequirementBars pins the exact per-axis bar table and that
// ParityRequirementFor agrees with it while rejecting an unknown axis.
func TestServingParityRequirementBars(t *testing.T) {
	t.Parallel()

	want := []struct {
		unit ParityUnit
		bar  float64
	}{
		{ParityUnitTokensPerSecond, 0.8},
		{ParityUnitTokensPerSecond, 0.5},
		{ParityUnitMilliseconds, 2.0},
	}
	reqs := ParityRequirements()
	if len(reqs) != len(want) {
		t.Fatalf("ParityRequirements() has %d members, want %d: %v", len(reqs), len(want), reqs)
	}
	axes := ParityAxes()
	for i, w := range want {
		if reqs[i].Unit != w.unit {
			t.Errorf("requirement[%d] (%q) unit = %q, want %q", i, axes[i], reqs[i].Unit, w.unit)
		}
		if reqs[i].Ratio != w.bar {
			t.Errorf("requirement[%d] (%q) bar = %v, want %v", i, axes[i], reqs[i].Ratio, w.bar)
		}
		got, ok := ParityRequirementFor(axes[i])
		if !ok {
			t.Errorf("ParityRequirementFor(%q) reported not found for a valid axis", axes[i])
		}
		if got != reqs[i] {
			t.Errorf("ParityRequirementFor(%q) = %+v, want %+v from ParityRequirements()", axes[i], got, reqs[i])
		}
	}

	if req, ok := ParityRequirementFor(ParityAxis("bogus")); ok {
		t.Errorf("ParityRequirementFor(unknown axis) reported found: %+v", req)
	} else if req != (ParityRequirement{}) {
		t.Errorf("ParityRequirementFor(unknown axis) = %+v, want zero value", req)
	}
}

// TestServingParityBoundarySemantics pins the inclusive bar comparison: a ratio
// exactly at the bar is Met on both directions, one step past is Missed, and a
// large win (or a warm-ttft far under the bar) is Met.
func TestServingParityBoundarySemantics(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		axis        ParityAxis
		sample      ParitySample
		wantVerdict ParityVerdict
		wantRatio   float64
	}{
		{"decode_at_bar_is_met", ParityAxisDecode, parityReading(ParityAxisDecode, 4, 5), ParityMet, 0.8},
		{"decode_below_bar_is_missed", ParityAxisDecode, parityReading(ParityAxisDecode, 79, 100), ParityMissed, 0.79},
		{"prefill_at_bar_is_met", ParityAxisPrefill, parityReading(ParityAxisPrefill, 1, 2), ParityMet, 0.5},
		{"warm_ttft_at_bar_is_met", ParityAxisWarmTTFT, parityReading(ParityAxisWarmTTFT, 200, 100), ParityMet, 2.0},
		{"warm_ttft_over_bar_is_missed", ParityAxisWarmTTFT, parityReading(ParityAxisWarmTTFT, 201, 100), ParityMissed, 2.01},
		{"decode_large_win_is_met", ParityAxisDecode, parityReading(ParityAxisDecode, 250, 100), ParityMet, 2.5},
		{"warm_ttft_twice_as_fast_is_met", ParityAxisWarmTTFT, parityReading(ParityAxisWarmTTFT, 50, 100), ParityMet, 0.5},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			grade, ok := GradeParity([]ParitySample{tc.sample}).Grade(tc.axis)
			if !ok {
				t.Fatalf("Grade(%q) not present after a matching sample", tc.axis)
			}
			if grade.Verdict != tc.wantVerdict {
				t.Errorf("verdict = %q, want %q (ratio %v, reason %q)", grade.Verdict, tc.wantVerdict, grade.Ratio, grade.Reason)
			}
			if grade.Ratio != tc.wantRatio {
				t.Errorf("ratio = %v, want %v", grade.Ratio, tc.wantRatio)
			}
			if got := grade.Met(); got != (tc.wantVerdict == ParityMet) {
				t.Errorf("Met() = %v, want %v for verdict %q", got, tc.wantVerdict == ParityMet, tc.wantVerdict)
			}
		})
	}
}

// TestServingParityFailClosedInputs pins that every malformed reading grades
// Unmeasured with a zero ratio and a non-empty reason, so no bad number can be
// read as a pass.
func TestServingParityFailClosedInputs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		sample ParitySample
	}{
		{"measured_false", ParitySample{Axis: ParityAxisDecode, Measured: false, Unit: ParityUnitTokensPerSecond, Fak: 1, Rival: 1}},
		{"unit_mismatch", ParitySample{Axis: ParityAxisDecode, Measured: true, Unit: ParityUnitMilliseconds, Fak: 1, Rival: 1}},
		{"zero_fak", parityReading(ParityAxisDecode, 0, 100)},
		{"zero_rival", parityReading(ParityAxisDecode, 100, 0)},
		{"negative_fak", parityReading(ParityAxisDecode, -5, 100)},
		{"nan_fak", parityReading(ParityAxisDecode, math.NaN(), 100)},
		{"pos_inf_rival", parityReading(ParityAxisDecode, 100, math.Inf(1))},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			grade, ok := GradeParity([]ParitySample{tc.sample}).Grade(ParityAxisDecode)
			if !ok {
				t.Fatalf("Grade(%q) not present although a sample for it was supplied", ParityAxisDecode)
			}
			if grade.Verdict != ParityUnmeasured {
				t.Errorf("verdict = %q, want %q (fail-closed)", grade.Verdict, ParityUnmeasured)
			}
			if grade.Ratio != 0 {
				t.Errorf("ratio = %v, want 0 for an unmeasured grade", grade.Ratio)
			}
			if grade.Reason == "" {
				t.Error("reason is empty for an unmeasured grade")
			}
			if grade.Met() {
				t.Error("Met() reports true for an unmeasured grade")
			}
		})
	}
}

// TestServingParityReportAggregation pins AllMet vs Complete: Complete only needs
// every axis graded, AllMet needs every axis Met, and the ordered accessors follow
// ParityAxes() order.
func TestServingParityReportAggregation(t *testing.T) {
	t.Parallel()

	t.Run("all_met", func(t *testing.T) {
		t.Parallel()
		rep := GradeParity([]ParitySample{
			parityReading(ParityAxisDecode, 100, 100),
			parityReading(ParityAxisPrefill, 100, 100),
			parityReading(ParityAxisWarmTTFT, 100, 100),
		})
		if !rep.AllMet() {
			t.Error("AllMet() = false with three met grades")
		}
		if !rep.Complete() {
			t.Error("Complete() = false with three met grades")
		}
		for _, a := range ParityAxes() {
			if _, ok := rep.Grade(a); !ok {
				t.Errorf("Grade(%q) not present", a)
			}
		}
		if got := rep.MetAxes(); !equalAxes(got, ParityAxes()) {
			t.Errorf("MetAxes() = %v, want %v", got, ParityAxes())
		}
		if got := rep.UnmeasuredAxes(); len(got) != 0 {
			t.Errorf("UnmeasuredAxes() = %v, want none", got)
		}
	})

	t.Run("all_missed_is_complete_not_all_met", func(t *testing.T) {
		t.Parallel()
		rep := GradeParity([]ParitySample{
			parityReading(ParityAxisDecode, 10, 100),
			parityReading(ParityAxisPrefill, 10, 100),
			parityReading(ParityAxisWarmTTFT, 300, 100),
		})
		if rep.AllMet() {
			t.Error("AllMet() = true with three missed grades")
		}
		if !rep.Complete() {
			t.Error("Complete() = false with three missed grades (a miss is still a reading)")
		}
		if got := rep.MissedAxes(); !equalAxes(got, ParityAxes()) {
			t.Errorf("MissedAxes() = %v, want %v in axis order", got, ParityAxes())
		}
		if got := rep.MetAxes(); len(got) != 0 {
			t.Errorf("MetAxes() = %v, want none", got)
		}
	})

	t.Run("one_axis_unmeasured", func(t *testing.T) {
		t.Parallel()
		rep := GradeParity([]ParitySample{
			parityReading(ParityAxisDecode, 100, 100),
			parityReading(ParityAxisPrefill, 100, 100),
		})
		if rep.AllMet() {
			t.Error("AllMet() = true while an axis has no reading")
		}
		if rep.Complete() {
			t.Error("Complete() = true while an axis has no reading")
		}
		if got := rep.UnmeasuredAxes(); !equalAxes(got, []ParityAxis{ParityAxisWarmTTFT}) {
			t.Errorf("UnmeasuredAxes() = %v, want [%q]", got, ParityAxisWarmTTFT)
		}
		grade, ok := rep.Grade(ParityAxisWarmTTFT)
		if !ok {
			t.Fatalf("Grade(%q) not present for a sample-less axis", ParityAxisWarmTTFT)
		}
		if grade.Verdict != ParityUnmeasured {
			t.Errorf("verdict for sample-less axis = %q, want %q", grade.Verdict, ParityUnmeasured)
		}
		if grade.Reason == "" {
			t.Error("reason is empty for a sample-less axis")
		}
	})

	t.Run("zero_value_report", func(t *testing.T) {
		t.Parallel()
		var rep ServingParityReport
		if rep.AllMet() {
			t.Error("zero-value report AllMet() = true")
		}
		if rep.Complete() {
			t.Error("zero-value report Complete() = true")
		}
		if got := rep.MetAxes(); len(got) != 0 {
			t.Errorf("zero-value MetAxes() = %v, want none", got)
		}
		if got := rep.MissedAxes(); len(got) != 0 {
			t.Errorf("zero-value MissedAxes() = %v, want none", got)
		}
		if got := rep.UnmeasuredAxes(); !equalAxes(got, ParityAxes()) {
			t.Errorf("zero-value UnmeasuredAxes() = %v, want %v", got, ParityAxes())
		}
	})
}

// TestServingParityInputImmutability pins that grading does not mutate the
// caller's slice.
func TestServingParityInputImmutability(t *testing.T) {
	t.Parallel()

	samples := []ParitySample{
		parityReading(ParityAxisDecode, 100, 100),
		parityReading(ParityAxisPrefill, 10, 100),
		parityReading(ParityAxisWarmTTFT, 300, 100),
	}
	before := make([]ParitySample, len(samples))
	copy(before, samples)

	_ = GradeParity(samples)

	if len(samples) != len(before) {
		t.Fatalf("input length changed: %d -> %d", len(before), len(samples))
	}
	for i := range before {
		if samples[i] != before[i] {
			t.Errorf("input sample[%d] mutated: %+v -> %+v", i, before[i], samples[i])
		}
	}
}

// TestServingParityFirstSampleWins pins that a duplicate reading for an axis does
// not overwrite the first one.
func TestServingParityFirstSampleWins(t *testing.T) {
	t.Parallel()

	first := parityReading(ParityAxisDecode, 100, 100) // ratio 1.0, Met
	second := parityReading(ParityAxisDecode, 10, 100) // ratio 0.1, Missed
	grade, ok := GradeParity([]ParitySample{first, second}).Grade(ParityAxisDecode)
	if !ok {
		t.Fatalf("Grade(%q) not present", ParityAxisDecode)
	}
	if grade.Verdict != ParityMet {
		t.Errorf("verdict = %q, want %q from the first sample (the later one must not win)", grade.Verdict, ParityMet)
	}
	if grade.Ratio != 1.0 {
		t.Errorf("ratio = %v, want 1.0 from the first sample", grade.Ratio)
	}
}

// TestServingParityNonFiniteRatio pins that a quotient overflowing to infinity is
// unmeasured on both polarity paths: both readings are finite and positive, so only
// the ratio check can reject it, and it must never read as a huge win.
func TestServingParityNonFiniteRatio(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		axis ParityAxis
	}{
		{"decode_throughput", ParityAxisDecode},
		{"warm_ttft_latency", ParityAxisWarmTTFT},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			sample := parityReading(tc.axis, math.MaxFloat64, math.SmallestNonzeroFloat64)
			grade, ok := GradeParity([]ParitySample{sample}).Grade(tc.axis)
			if !ok {
				t.Fatalf("Grade(%q) not present after a matching sample", tc.axis)
			}
			if grade.Verdict != ParityUnmeasured {
				t.Errorf("verdict = %q, want %q for an overflowing quotient", grade.Verdict, ParityUnmeasured)
			}
			if grade.Ratio != 0 {
				t.Errorf("ratio = %v, want 0 for an unmeasured grade", grade.Ratio)
			}
			if grade.Reason == "" {
				t.Error("reason is empty for an overflowing quotient")
			}
			if grade.Met() {
				t.Error("Met() reports true for an overflowing quotient")
			}
		})
	}
}

// TestServingParityGradeNotFound pins that a report holding only some axes answers
// not-found with the zero grade for an absent axis, while the ordered accessors
// exclude it from met and missed and count it as unmeasured.
func TestServingParityGradeNotFound(t *testing.T) {
	t.Parallel()

	present, ok := GradeParity([]ParitySample{parityReading(ParityAxisDecode, 100, 100)}).Grade(ParityAxisDecode)
	if !ok {
		t.Fatalf("Grade(%q) not present after a matching sample", ParityAxisDecode)
	}
	rep := ServingParityReport{Grades: []ParityGrade{present}}

	if got, ok := rep.Grade(ParityAxisDecode); !ok {
		t.Errorf("Grade(%q) reported not found for a held axis", ParityAxisDecode)
	} else if !got.Met() {
		t.Errorf("Grade(%q).Met() = false, want true for the held axis", ParityAxisDecode)
	}
	for _, absent := range []ParityAxis{ParityAxisPrefill, ParityAxisWarmTTFT} {
		grade, ok := rep.Grade(absent)
		if ok {
			t.Errorf("Grade(%q) reported found for an axis the report does not hold: %+v", absent, grade)
		}
		if grade != (ParityGrade{}) {
			t.Errorf("Grade(%q) = %+v, want the zero grade for an absent axis", absent, grade)
		}
	}
	if got := rep.MetAxes(); !equalAxes(got, []ParityAxis{ParityAxisDecode}) {
		t.Errorf("MetAxes() = %v, want [%q]", got, ParityAxisDecode)
	}
	if got := rep.MissedAxes(); len(got) != 0 {
		t.Errorf("MissedAxes() = %v, want none (an absent axis is never missed)", got)
	}
	if got := rep.UnmeasuredAxes(); !equalAxes(got, []ParityAxis{ParityAxisPrefill, ParityAxisWarmTTFT}) {
		t.Errorf("UnmeasuredAxes() = %v, want [%q %q]", got, ParityAxisPrefill, ParityAxisWarmTTFT)
	}
}

// TestServingParityCompleteRejectsInvalidVerdict pins that Complete and AllMet both
// reject a report whose verdict is outside the vocabulary, and a report that omits an
// axis entirely.
func TestServingParityCompleteRejectsInvalidVerdict(t *testing.T) {
	t.Parallel()

	met := func(axis ParityAxis) ParityGrade {
		req, _ := ParityRequirementFor(axis)
		return ParityGrade{Requirement: req, Ratio: 1, Verdict: ParityMet}
	}
	warmReq, _ := ParityRequirementFor(ParityAxisWarmTTFT)
	bogus := ParityGrade{Requirement: warmReq, Ratio: 1, Verdict: ParityVerdict("bogus")}

	cases := []struct {
		name   string
		grades []ParityGrade
	}{
		{"out_of_vocabulary_verdict", []ParityGrade{met(ParityAxisDecode), met(ParityAxisPrefill), bogus}},
		{"missing_axis", []ParityGrade{met(ParityAxisDecode), met(ParityAxisPrefill)}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rep := ServingParityReport{Grades: tc.grades}
			if rep.Complete() {
				t.Error("Complete() = true, want false")
			}
			if rep.AllMet() {
				t.Error("AllMet() = true, want false")
			}
		})
	}
}

func equalAxes(got, want []ParityAxis) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
