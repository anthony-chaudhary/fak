package benchscore

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

func goldenALFixtureCurve() GoldenALCurve {
	return GoldenALCurve{
		ID: GoldenALCurveID{
			Model:       "glm-5.3-flash",
			Method:      "dspark",
			Draft:       "glm-5.3-flash-dspark",
			DraftSample: "probabilistic",
			Verify:      "block",
		},
		SourceRun: "https://example.invalid/actions/runs/1",
		Acceptance: map[string]map[int]float64{
			"default":  {2: 2.00, 4: 3.20},
			"thinking": {4: 2.50},
		},
	}
}

const goldenALBefore = `# HELP spec_decode_num_accepted_tokens_total Accepted draft tokens.
# TYPE spec_decode_num_accepted_tokens_total counter
spec_decode_num_accepted_tokens_total{engine="0",model="m"} 100
spec_decode_num_accepted_tokens_total{engine="1",model="m{x}"} 50 1700000000000
spec_decode_num_drafts_total{engine="0",model="m"} 60
spec_decode_num_drafts_total{engine="1",model="m"} 40
spec_decode_num_accepted_tokens_total_per_pos{position="0"} 9999
`

const goldenALAfter = `spec_decode_num_accepted_tokens_total{engine="0",model="m"} 1300
spec_decode_num_accepted_tokens_total{engine="1",model="m{x}"} 1050 1700000009000 # {trace_id="abc"} 7.5
spec_decode_num_drafts_total{engine="0",model="m"} 560
spec_decode_num_drafts_total{engine="1",model="m"} 540
spec_decode_num_accepted_tokens_total_per_pos{position="0"} 123456
`

const (
	goldenALAcceptedName = "spec_decode_num_accepted_tokens_total"
	goldenALDraftsName   = "spec_decode_num_drafts_total"
)

// fak-test:runtime fast est=1ms
func TestGoldenALGate(t *testing.T) {
	t.Parallel()
	curve := goldenALFixtureCurve()

	t.Run("curve identity round-trips through its stem", func(t *testing.T) {
		const want = "glm-5.3-flash__dspark__glm-5.3-flash-dspark__probabilistic-block"
		if got := curve.ID.Stem(); got != want {
			t.Fatalf("Stem = %q, want %q", got, want)
		}
		back, err := ParseGoldenALCurveStem(want)
		if err != nil || back != curve.ID {
			t.Fatalf("ParseGoldenALCurveStem = %+v, %v; want %+v", back, err, curve.ID)
		}
		for _, bad := range []string{
			"",
			"model__mtp__native",
			"model__mtp__native__default",
			"model__mtp__native__default-default__extra",
			"model__mtp____default-default",
			"model__mtp__native__-default",
			"mo del__mtp__native__default-default",
		} {
			if _, err := ParseGoldenALCurveStem(bad); !errors.Is(err, ErrGoldenALCurve) {
				t.Fatalf("ParseGoldenALCurveStem(%q) = %v, want ErrGoldenALCurve", bad, err)
			}
		}
		// A trailing underscore would merge with the separator and split elsewhere.
		for _, mutate := range []func(*GoldenALCurveID){
			func(id *GoldenALCurveID) { id.Model = "glm_" },
			func(id *GoldenALCurveID) { id.Method = "mtp_" },
			func(id *GoldenALCurveID) { id.Draft = "native_" },
			func(id *GoldenALCurveID) { id.Verify = "a-b" },
		} {
			id := curve.ID
			mutate(&id)
			if err := id.Validate(); !errors.Is(err, ErrGoldenALCurve) {
				t.Fatalf("identity %+v (stem %q) admitted: %v", id, id.Stem(), err)
			}
		}
		// Curves that differ only in the verifier are different curves.
		other := curve.ID
		other.Verify = "default"
		if other.Stem() == curve.ID.Stem() {
			t.Fatal("sample-verify is not part of the curve identity")
		}
	})

	t.Run("curve needs provenance and a recorded point", func(t *testing.T) {
		noRun := goldenALFixtureCurve()
		noRun.SourceRun = "  "
		if _, err := EvaluateGoldenALGate(noRun, "default", 4, nil); !errors.Is(err, ErrGoldenALCurve) {
			t.Fatalf("curve without source_run judged: %v", err)
		}
		for name, point := range map[string]struct {
			tokens int
			al     float64
		}{
			"above draft_tokens+1": {4, 5.01},
			"below one":            {4, 0.9},
			"not a number":         {4, math.NaN()},
			"zero draft tokens":    {0, 1},
		} {
			impossible := goldenALFixtureCurve()
			impossible.Acceptance["default"][point.tokens] = point.al
			if err := impossible.Validate(); !errors.Is(err, ErrGoldenALCurve) {
				t.Fatalf("curve point %s admitted: %v", name, err)
			}
		}
		for _, point := range []struct {
			mode   string
			tokens int
		}{{"default", 3}, {"thinking", 2}, {"unknown", 4}} {
			if _, err := EvaluateGoldenALGate(curve, point.mode, point.tokens, &SpecDecodeCounters{1, 1}); !errors.Is(err, ErrGoldenALCurve) {
				t.Fatalf("unrecorded point %+v judged: %v", point, err)
			}
		}
	})

	t.Run("acceptance length derives from the counter increase", func(t *testing.T) {
		before, ok := ReadSpecDecodeCounters(goldenALBefore, goldenALAcceptedName, goldenALDraftsName)
		if !ok || before != (SpecDecodeCounters{AcceptedDraftTokens: 150, VerificationSteps: 100}) {
			t.Fatalf("before = %+v, %v", before, ok)
		}
		after, ok := ReadSpecDecodeCounters(goldenALAfter, goldenALAcceptedName, goldenALDraftsName)
		if !ok || after != (SpecDecodeCounters{AcceptedDraftTokens: 2350, VerificationSteps: 1100}) {
			t.Fatalf("after = %+v, %v", after, ok)
		}
		increase, ok := SpecDecodeIncrease(before, after)
		if !ok || increase != (SpecDecodeCounters{AcceptedDraftTokens: 2200, VerificationSteps: 1000}) {
			t.Fatalf("increase = %+v, %v", increase, ok)
		}
		v, err := EvaluateGoldenALGate(curve, "default", 4, &increase)
		if err != nil {
			t.Fatal(err)
		}
		// AL = 1 + 2200/1000 = 3.2 against golden 3.2. The pre-replay totals
		// (150 accepted over 100 steps) must not leak into the measurement.
		if v.AcceptanceLength != 3.2 || v.Ratio != 1 || v.Status != GoldenALPass {
			t.Fatalf("verdict = %+v, want AL 3.2 ratio 1 pass", v)
		}
		if v.AcceptedTokens != 2200 || v.Verifications != 1000 || v.GoldenAcceptanceLength != 3.2 {
			t.Fatalf("verdict evidence = %+v", v)
		}
		if v.Curve != curve.ID.Stem() || v.SourceRun != curve.SourceRun || v.ThinkingMode != "default" || v.DraftTokens != 4 {
			t.Fatalf("verdict identity = %+v", v)
		}
	})

	t.Run("fixed ratio band with inclusive ends", func(t *testing.T) {
		// Golden 2.00 at default/2 with 1000 steps: AL = 1 + accepted/1000.
		for _, tc := range []struct {
			accepted float64
			ratio    float64
			status   GoldenALStatus
		}{
			{898, 0.949, GoldenALBelow},
			{900, 0.95, GoldenALPass},
			{1000, 1, GoldenALPass},
			{1100, 1.05, GoldenALPass},
			{1102, 1.051, GoldenALAbove},
			{0, 0.5, GoldenALBelow},
		} {
			v, err := EvaluateGoldenALGate(curve, "default", 2, &SpecDecodeCounters{AcceptedDraftTokens: tc.accepted, VerificationSteps: 1000})
			if err != nil {
				t.Fatal(err)
			}
			if v.Ratio != tc.ratio || v.Status != tc.status {
				t.Fatalf("accepted=%v: ratio %v status %q, want %v %q", tc.accepted, v.Ratio, v.Status, tc.ratio, tc.status)
			}
			if !v.Measured() {
				t.Fatalf("accepted=%v: verdict %q not reported as measured", tc.accepted, v.Status)
			}
		}
		// Just outside the band before rounding, on it after: the verdict
		// follows the ratio the document reports.
		for _, accepted := range []float64{899_992, 1_100_008} {
			v, err := EvaluateGoldenALGate(curve, "default", 2, &SpecDecodeCounters{AcceptedDraftTokens: accepted, VerificationSteps: 1_000_000})
			if err != nil {
				t.Fatal(err)
			}
			if (v.Ratio != 0.95 && v.Ratio != 1.05) || v.Status != GoldenALPass {
				t.Fatalf("accepted=%v: ratio %v status %q, want an edge ratio and pass", accepted, v.Ratio, v.Status)
			}
		}
	})

	t.Run("report-only: no verdict fails the run", func(t *testing.T) {
		noCounters, err := EvaluateGoldenALGate(curve, "default", 4, nil)
		if err != nil || noCounters.Status != GoldenALNoCounters {
			t.Fatalf("nil increase = %+v, %v; want no-counters", noCounters, err)
		}
		noDrafting, err := EvaluateGoldenALGate(curve, "default", 4, &SpecDecodeCounters{})
		if err != nil || noDrafting.Status != GoldenALNoDrafting {
			t.Fatalf("zero verifications = %+v, %v; want no-drafting", noDrafting, err)
		}
		below, _ := EvaluateGoldenALGate(curve, "default", 4, &SpecDecodeCounters{AcceptedDraftTokens: 1, VerificationSteps: 1000})
		above, _ := EvaluateGoldenALGate(curve, "default", 2, &SpecDecodeCounters{AcceptedDraftTokens: 2000, VerificationSteps: 1000})
		pass, _ := EvaluateGoldenALGate(curve, "default", 2, &SpecDecodeCounters{AcceptedDraftTokens: 1000, VerificationSteps: 1000})
		if below.Status != GoldenALBelow || above.Status != GoldenALAbove || pass.Status != GoldenALPass {
			t.Fatalf("statuses = %q %q %q", below.Status, above.Status, pass.Status)
		}
		for _, v := range []GoldenALVerdict{noCounters, noDrafting, below, above, pass} {
			if v.ExitCode() != 0 {
				t.Fatalf("status %q exits %d, want 0", v.Status, v.ExitCode())
			}
			if !v.ReportOnly {
				t.Fatalf("status %q not marked report-only", v.Status)
			}
			raw, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			var doc map[string]any
			if err := json.Unmarshal(raw, &doc); err != nil {
				t.Fatal(err)
			}
			if doc["report_only"] != true || doc["source_run"] != curve.SourceRun || doc["status"] != string(v.Status) {
				t.Fatalf("status %q document = %s", v.Status, raw)
			}
		}
		for _, v := range []GoldenALVerdict{noCounters, noDrafting} {
			if v.Measured() || v.AcceptanceLength != 0 || v.Ratio != 0 {
				t.Fatalf("skipped gate %q carries a measurement: %+v", v.Status, v)
			}
		}
	})

	t.Run("absent or reset counters are not a measurement", func(t *testing.T) {
		if _, ok := ReadSpecDecodeCounters(goldenALBefore, goldenALAcceptedName, "missing_total"); ok {
			t.Fatal("missing verification counter read as present")
		}
		if _, ok := ReadSpecDecodeCounters("", goldenALAcceptedName, goldenALDraftsName); ok {
			t.Fatal("empty exposition read as present")
		}
		if _, ok := ReadSpecDecodeCounters("# TYPE "+goldenALAcceptedName+" counter\n", goldenALAcceptedName, goldenALDraftsName); ok {
			t.Fatal("comment-only exposition read as present")
		}
		// An engine restart resets its counters below the earlier scrape.
		before := SpecDecodeCounters{AcceptedDraftTokens: 500, VerificationSteps: 200}
		if _, ok := SpecDecodeIncrease(before, SpecDecodeCounters{AcceptedDraftTokens: 10, VerificationSteps: 300}); ok {
			t.Fatal("accepted counter reset read as an increase")
		}
		if _, ok := SpecDecodeIncrease(before, SpecDecodeCounters{AcceptedDraftTokens: 600, VerificationSteps: 5}); ok {
			t.Fatal("verification counter reset read as an increase")
		}
		v, err := EvaluateGoldenALGate(curve, "default", 4, &SpecDecodeCounters{AcceptedDraftTokens: -1, VerificationSteps: 10})
		if err != nil || v.Status != GoldenALNoCounters {
			t.Fatalf("negative increase = %+v, %v; want no-counters", v, err)
		}
		v, err = EvaluateGoldenALGate(curve, "default", 4, &SpecDecodeCounters{AcceptedDraftTokens: math.NaN(), VerificationSteps: 10})
		if err != nil || v.Status != GoldenALNoCounters {
			t.Fatalf("NaN increase = %+v, %v; want no-counters", v, err)
		}
	})
}
