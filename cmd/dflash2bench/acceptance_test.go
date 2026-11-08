package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"testing"
)

// closeTo pins a priced value within absolute tolerance; economics contracts
// are closed values, not error prose.
func closeTo(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v, want %v", got, want)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestRooflineVerifierBandwidthPlateauThenComputeKnee(t *testing.T) {
	v := rooflineVerifier{WeightBytes: 100, FlopsPerPos: 100, Bandwidth: 100, Compute: 1000}
	if err := v.valid(); err != nil {
		t.Fatal(err)
	}
	// Weight bytes move once per pass: every width up to the knee costs the
	// same 1.0s bandwidth floor; past the knee compute grows linearly in width.
	for w := 1; w <= 10; w++ {
		closeTo(t, v.seconds(w), 1.0)
	}
	closeTo(t, v.seconds(11), 1.1)
	closeTo(t, v.seconds(20), 2.0)
	if got := v.seconds(0); got != 0 {
		t.Fatalf("empty pass cost %v, want 0", got)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestGeometricAcceptanceClosedForm(t *testing.T) {
	table, err := GeometricAcceptance(0.8, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 0.8, 1.44, 1.952} // p*(1-p^L)/(1-p)
	for l, value := range table {
		closeTo(t, value, want[l])
	}
	zeros, err := GeometricAcceptance(0, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range zeros {
		if value != 0 {
			t.Fatalf("p=0 accepted %v, want 0", value)
		}
	}
	identity, err := GeometricAcceptance(1, 3)
	if err != nil {
		t.Fatal(err)
	}
	for l, value := range identity {
		if value != float64(l) {
			t.Fatalf("p=1 row %d accepted %v, want %d", l, value, l)
		}
	}
	for name, p := range map[string]float64{"negative": -0.1, "above one": 1.2, "nan": math.NaN()} {
		if _, err := GeometricAcceptance(p, 2); !errors.Is(err, ErrInvalidAcceptanceTable) {
			t.Fatalf("%s: err %v, want ErrInvalidAcceptanceTable", name, err)
		}
	}
	if _, err := GeometricAcceptance(0.5, -1); !errors.Is(err, ErrInvalidAcceptanceTable) {
		t.Fatalf("err %v, want ErrInvalidAcceptanceTable", err)
	}
}

// fakeRoofline pins the interior-optimum fixture: 1.0s bandwidth floor up to
// width 10, then 0.1s per extra verified position.
func fakeRoofline() rooflineVerifier {
	return rooflineVerifier{WeightBytes: 100, FlopsPerPos: 100, Bandwidth: 100, Compute: 1000}
}

// fak-test:runtime fast est=50ms lane=default
func TestPriceSpecDecodePicksInteriorBestNotLargest(t *testing.T) {
	accept, err := GeometricAcceptance(0.5, 8)
	if err != nil {
		t.Fatal(err)
	}
	verdict, err := priceSpecDecode(fakeRoofline(), drafterCost{PerToken: 0.05}, accept, defaultAmberMargin)
	if err != nil {
		t.Fatal(err)
	}
	if verdict.Schema != acceptanceSchema {
		t.Fatalf("schema %q", verdict.Schema)
	}
	closeTo(t, verdict.Plain.TokensPerSecond, 1.0)
	if len(verdict.Rounds) != 8 {
		t.Fatalf("rounds %d, want 8", len(verdict.Rounds))
	}
	// L=3: committed 1+0.875, round 0.05*3 + 1.0 = 1.15 -> best.
	best := verdict.Best
	if best.DraftLength != 3 {
		t.Fatalf("best draft length %d, want 3 (roofline-priced, not maximized)", best.DraftLength)
	}
	closeTo(t, best.AcceptedDraftTokens, 0.875)
	closeTo(t, best.CommittedTokens, 1.875)
	if best.VerifierWidth != 4 {
		t.Fatalf("verifier width %d, want 4 (anchor + full proposed chain)", best.VerifierWidth)
	}
	closeTo(t, best.RoundSeconds, 1.15)
	closeTo(t, verdict.Ratio, 1.875/1.15)
	if verdict.Decision != decisionGreen {
		t.Fatalf("decision %s, want GREEN", verdict.Decision)
	}
	// The largest draft length must lose to the priced optimum.
	largest := verdict.Rounds[len(verdict.Rounds)-1]
	if largest.TokensPerSecond >= best.TokensPerSecond {
		t.Fatalf("largest L=%d tps %v beat best %v", largest.DraftLength, largest.TokensPerSecond, best.TokensPerSecond)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestPriceSpecDecodeRedWhenDrafterCannotRepay(t *testing.T) {
	accept, err := GeometricAcceptance(0.5, 8)
	if err != nil {
		t.Fatal(err)
	}
	verdict, err := priceSpecDecode(fakeRoofline(), drafterCost{PerToken: 1.0}, accept, defaultAmberMargin)
	if err != nil {
		t.Fatal(err)
	}
	if verdict.Best.DraftLength != 0 {
		t.Fatalf("best draft length %d, want 0 (plain decode wins)", verdict.Best.DraftLength)
	}
	closeTo(t, verdict.Ratio, 1.0)
	if verdict.Decision != decisionRed {
		t.Fatalf("decision %s, want RED", verdict.Decision)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestPriceSpecDecodeAmberWindow(t *testing.T) {
	accept, err := GeometricAcceptance(0.5, 8)
	if err != nil {
		t.Fatal(err)
	}
	verdict, err := priceSpecDecode(fakeRoofline(), drafterCost{PerToken: 0.3}, accept, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if verdict.Best.DraftLength != 1 {
		t.Fatalf("best draft length %d, want 1", verdict.Best.DraftLength)
	}
	closeTo(t, verdict.Ratio, 1.5/1.3) // 1.1538..., inside (1, 1.5)
	if verdict.Decision != decisionAmber {
		t.Fatalf("decision %s, want AMBER", verdict.Decision)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestPriceSpecDecodeRejectsInvalidInputs(t *testing.T) {
	good, err := GeometricAcceptance(0.5, 2)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]error{
		"accepted above requested": func() error {
			bad := acceptanceMock{0, 1.5}
			_, err := priceSpecDecode(fakeRoofline(), drafterCost{}, bad, defaultAmberMargin)
			return err
		}(),
		"decreasing expectation": func() error {
			bad := acceptanceMock{0, 1, 0.5}
			_, err := priceSpecDecode(fakeRoofline(), drafterCost{}, bad, defaultAmberMargin)
			return err
		}(),
		"negative expectation": func() error {
			bad := acceptanceMock{0, -0.1}
			_, err := priceSpecDecode(fakeRoofline(), drafterCost{}, bad, defaultAmberMargin)
			return err
		}(),
		"empty table": func() error {
			_, err := priceSpecDecode(fakeRoofline(), drafterCost{}, nil, defaultAmberMargin)
			return err
		}(),
		"zero bandwidth": func() error {
			_, err := priceSpecDecode(rooflineVerifier{WeightBytes: 1, Compute: 1}, drafterCost{}, good, defaultAmberMargin)
			return err
		}(),
		"no verifier work priced": func() error {
			_, err := priceSpecDecode(rooflineVerifier{Bandwidth: 100, Compute: 1000}, drafterCost{}, good, defaultAmberMargin)
			return err
		}(),
		"negative weight bytes": func() error {
			_, err := priceSpecDecode(rooflineVerifier{WeightBytes: -1, Bandwidth: 1, Compute: 1}, drafterCost{}, good, defaultAmberMargin)
			return err
		}(),
		"negative drafter token cost": func() error {
			_, err := priceSpecDecode(fakeRoofline(), drafterCost{PerToken: -1}, good, defaultAmberMargin)
			return err
		}(),
		"negative amber margin": func() error {
			_, err := priceSpecDecode(fakeRoofline(), drafterCost{}, good, -0.1)
			return err
		}(),
	}
	want := map[string]error{
		"accepted above requested":    ErrInvalidAcceptanceTable,
		"decreasing expectation":      ErrInvalidAcceptanceTable,
		"negative expectation":        ErrInvalidAcceptanceTable,
		"empty table":                 ErrInvalidAcceptanceTable,
		"zero bandwidth":              ErrInvalidRoofline,
		"negative weight bytes":       ErrInvalidRoofline,
		"no verifier work priced":     ErrInvalidRoofline,
		"negative drafter token cost": ErrInvalidDrafterCost,
		"negative amber margin":       ErrInvalidAmberMargin,
	}
	for name, got := range cases {
		if !errors.Is(got, want[name]) {
			t.Fatalf("%s: err %v, want %v", name, got, want[name])
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestRunAcceptanceCLIPricesFromFlags(t *testing.T) {
	var out bytes.Buffer
	code := runAcceptanceCLI([]string{
		"--max-draft", "8",
		"--acceptance-p", "0.5",
		"--weight-bytes", "100", "--flops-per-pos", "100",
		"--bandwidth", "100", "--compute", "1000",
		"--drafter-per-token", "0.05",
	}, &out, io.Discard)
	if code != 0 {
		t.Fatalf("code %d", code)
	}
	var verdict specDecodeVerdict
	if err := json.Unmarshal(out.Bytes(), &verdict); err != nil {
		t.Fatalf("json: %v", err)
	}
	if verdict.Schema != acceptanceSchema || verdict.Best.DraftLength != 3 || verdict.Decision != decisionGreen {
		t.Fatalf("verdict %+v", verdict)
	}
	// The explicit table path prices identically to the geometric form.
	var out2 bytes.Buffer
	code = runAcceptanceCLI([]string{
		"--max-draft", "8",
		"--acceptance-table", "0,0.5,0.75,0.875,0.9375,0.96875,0.984375,0.9921875,0.99609375",
		"--weight-bytes", "100", "--flops-per-pos", "100",
		"--bandwidth", "100", "--compute", "1000",
		"--drafter-per-token", "0.05",
	}, &out2, io.Discard)
	if code != 0 {
		t.Fatalf("code %d", code)
	}
	var second specDecodeVerdict
	if err := json.Unmarshal(out2.Bytes(), &second); err != nil {
		t.Fatalf("json: %v", err)
	}
	if second.Best.DraftLength != 3 {
		t.Fatalf("table path best %d, want 3", second.Best.DraftLength)
	}
	closeTo(t, second.Ratio, verdict.Ratio)
	// Ambiguous or missing inputs fail fast at usage level.
	for _, args := range [][]string{
		{"--bandwidth", "100", "--compute", "1000"},                                                         // no acceptance source
		{"--acceptance-p", "0.5", "--acceptance-table", "0,0.5", "--bandwidth", "100", "--compute", "1000"}, // both
		{"--acceptance-p", "0.5", "--bandwidth", "0", "--compute", "1000"},                                  // zero bandwidth
		{"--acceptance-table", "0,0.5", "--bandwidth", "100", "--compute", "1000"},                          // table shorter than max-draft+1
	} {
		if code := runAcceptanceCLI(args, io.Discard, io.Discard); code == 0 {
			t.Fatalf("args %v accepted, want refusal", args)
		}
	}
}
