package quality

import (
	"encoding/json"
	"testing"
)

// oraclesFor resolves a case's declared oracles or fails the test — the spine
// requires every named oracle to exist.
func oraclesFor(t *testing.T, c QualityCase) []Oracle {
	t.Helper()
	os, err := Lookup(c.Oracles)
	if err != nil {
		t.Fatalf("Lookup(%v): %v", c.Oracles, err)
	}
	return os
}

// TestSpineCleanCasePasses is contract items 1–4 on the happy path: the demo case
// run through reference vs a faithful engine passes every oracle and emits no
// failure bundle.
func TestSpineCleanCasePasses(t *testing.T) {
	c := DemoCase()
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("clean case should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean pass must not carry a failure bundle: %+v", res.FailureBundle)
	}
	if res.Schema != ResultSchema {
		t.Errorf("result schema = %q, want %q", res.Schema, ResultSchema)
	}
	if res.Manifest.ReferenceRunner != "reference" {
		t.Errorf("manifest reference runner = %q", res.Manifest.ReferenceRunner)
	}
}

// TestSpineDecodeDefectTripsDifferential is the epic Witness for the decode gate:
// an injected token flip must fail the greedy differential oracle AT the flipped
// index and produce a replay-complete bundle.
func TestSpineDecodeDefectTripsDifferential(t *testing.T) {
	c := DemoCase()
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("decode"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("decode defect must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "greedy-token-diff" {
		t.Errorf("first failing oracle = %q, want greedy-token-diff", fb.FailingOracle)
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 1 {
		t.Fatalf("expected first divergence at token 1, got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Reference != "increased" || fb.FirstDivergence.Engine != "decreased" {
		t.Errorf("divergence tokens = ref %q eng %q, want increased/decreased",
			fb.FirstDivergence.Reference, fb.FirstDivergence.Engine)
	}
	// Replay-complete: the bundle embeds the full case, so it round-trips through
	// JSON and reruns to the same failure with nothing external.
	blob, err := json.Marshal(fb)
	if err != nil {
		t.Fatalf("marshal bundle: %v", err)
	}
	var got FailureBundle
	if err := json.Unmarshal(blob, &got); err != nil {
		t.Fatalf("unmarshal bundle: %v", err)
	}
	replay, err := RunCase(got.Case, ReferenceRunner{}, DemoEngine("decode"), oraclesFor(t, got.Case))
	if err != nil {
		t.Fatalf("replay RunCase: %v", err)
	}
	if replay.Pass {
		t.Fatal("replayed bundle case must fail identically")
	}
}

// TestSpineReportDefectTripsRubric is the epic Witness for the report-quality gate:
// text that omits a required figure and asserts a forbidden claim fails the rubric
// even though (here) the tokens are unremarkable.
func TestSpineReportDefectTripsRubric(t *testing.T) {
	c := DemoCase()
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("report"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("report defect must not pass; got %s", Explain(res))
	}
	var sawRubricFail bool
	for _, v := range res.Verdicts {
		if v.Oracle == "grounding-rubric" && !v.Pass {
			sawRubricFail = true
		}
	}
	if !sawRubricFail {
		t.Fatalf("grounding rubric should have failed; got %s", Explain(res))
	}
}

// TestComparatorIsLoadBearing is spine contract item 5: the SAME decode defect that
// is caught with the differential comparator wired slips through UNDETECTED when it
// is not. That proves the comparator — not some incidental check — is what makes the
// fixture fail; wiring it is what turns the fixture green→red on a real regression.
func TestComparatorIsLoadBearing(t *testing.T) {
	c := DemoCase()
	defect := DemoEngine("decode")

	// Comparator NOT wired: judge with only the rubric, which the token-only decode
	// defect does not trip (text still contains "12%", no forbidden word here... it
	// does contain "decreased"), so to isolate the comparator we strip the rubric's
	// forbidden term for this negative control.
	loose := c
	loose.Rubric = RubricSpec{Required: []string{"12%"}, MinScore: 1}
	loose.Oracles = []string{"grounding-rubric"}
	unwired, err := RunCase(loose, ReferenceRunner{}, defect, oraclesFor(t, loose))
	if err != nil {
		t.Fatalf("unwired RunCase: %v", err)
	}
	if !unwired.Pass {
		t.Fatalf("without the differential comparator the decode defect should slip through; got %s", Explain(unwired))
	}

	// Comparator wired: the differential oracle catches the same defect.
	wired := loose
	wired.Oracles = []string{"greedy-token-diff", "grounding-rubric"}
	caught, err := RunCase(wired, ReferenceRunner{}, defect, oraclesFor(t, wired))
	if err != nil {
		t.Fatalf("wired RunCase: %v", err)
	}
	if caught.Pass {
		t.Fatal("with the differential comparator wired the decode defect must be caught")
	}
}

// TestUnknownOracleRefused proves a case naming an unshipped oracle is refused, not
// silently skipped to a misleading green.
func TestUnknownOracleRefused(t *testing.T) {
	if _, err := Lookup([]string{"greedy-token-diff", "no-such-oracle"}); err == nil {
		t.Fatal("Lookup must refuse an unknown oracle")
	}
}

// TestInvalidCaseRefused proves the admission gate rejects a malformed case.
func TestInvalidCaseRefused(t *testing.T) {
	bad := DemoCase()
	bad.Oracles = nil
	if _, err := RunCase(bad, ReferenceRunner{}, DemoEngine(""), nil); err == nil {
		t.Fatal("RunCase must refuse a case that declares no oracles")
	}
}
