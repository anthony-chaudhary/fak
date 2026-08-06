package metrics

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// resolution_test.go — the fail-closed proof for the instrument-resolution
// contract (#5667). Every arm constructs metrics values in-process: no network,
// no model, no accelerator, no credential, so the whole set replays on a clean
// CPU-only checkout.
//
// The arms are ordered by the defect they kill:
//  1. unmeasured is not a measured zero (and a real zero is not unmeasured)
//  2. a claim below the grid refuses, and the required-threshold form of the
//     same question refuses on a small population
//  3. per-slice floors differ inside ONE artifact
//  4. the percentile ceiling is disclosed without moving the published value
//  5. the contract cannot pass vacuously
//  6. the vocabulary is closed and every typed refusal is reachable

// Arm 1a: a slice that observed nothing is UNMEASURED, not a measured zero.
// This is the representative defect: internal/bench's rate() returns 0 for an
// empty denominator, so "never measured" and "measured zero" are byte-identical
// in report.json. Give ResolveClaim that same 0-over-0 input and it must refuse.
func TestZeroDenominatorIsUnmeasuredNotZero(t *testing.T) {
	got := ResolveClaim(ResolutionClaim{
		Metric: "context_pollution_rate", Slice: "vdso_on",
		Denominator: 0,                   // nothing observed
		Claim:       0, ClaimKnown: true, // exactly what rate() would hand over
	})
	if got.Verdict != ResolutionUnmeasured {
		t.Fatalf("empty denominator verdict = %q, want %q", got.Verdict, ResolutionUnmeasured)
	}
	if got.Published() {
		t.Error("an unmeasured slice must not be published as a measurement")
	}
	if got.Quantum != 0 {
		t.Errorf("quantum = %v, want 0: no population means no grid", got.Quantum)
	}
}

// Arm 1b: the mirror image. An exact zero on a POPULATED slice IS a measurement
// — 0 quarantines out of 1,000 calls is real evidence — so it must stay
// MEASURABLE. A contract that refused here would trade one collapse for another.
func TestExactZeroOnPopulatedSliceIsMeasured(t *testing.T) {
	got := ResolveClaim(ResolutionClaim{
		Metric: "context_pollution_rate", Slice: "vdso_on",
		Denominator: 1000,
		Claim:       0, ClaimKnown: true,
	})
	if got.Verdict != ResolutionMeasurable {
		t.Fatalf("exact zero on 1000 calls = %q, want %q (reason: %s)", got.Verdict, ResolutionMeasurable, got.Reason)
	}
	if got.Quantum != 0.001 {
		t.Errorf("quantum = %v, want 0.001", got.Quantum)
	}
}

// Arm 2a: a claim finer than the grid refuses. A 12-call arm resolves nothing
// below 1/12 = 0.0833, so a claimed 0.05 effect is BELOW_RESOLUTION — not a
// measured zero, and not a failure.
func TestSubQuantumClaimRefuses(t *testing.T) {
	coarse := ResolveClaim(ResolutionClaim{
		Metric: "vdso_hit_rate", Slice: "vdso_on",
		Denominator: 12,
		Claim:       0.05, ClaimKnown: true,
	})
	if coarse.Verdict != ResolutionBelowResolution {
		t.Fatalf("0.05 on 12 calls = %q, want %q", coarse.Verdict, ResolutionBelowResolution)
	}
	if coarse.Published() {
		t.Error("a below-resolution claim must not be published as a measurement")
	}

	// The same claim on a population that can carry it is MEASURABLE. This is
	// the discrimination half: the signal must be able to succeed.
	fine := ResolveClaim(ResolutionClaim{
		Metric: "vdso_hit_rate", Slice: "vdso_on",
		Denominator: 1000,
		Claim:       0.05, ClaimKnown: true,
	})
	if fine.Verdict != ResolutionMeasurable {
		t.Fatalf("0.05 on 1000 calls = %q, want %q", fine.Verdict, ResolutionMeasurable)
	}
}

// Arm 2b: the contract's headline question in its own verb. "Could this
// population have shown a 5-point effect?" must answer no for a 12-call slice
// and yes for a 1,000-call slice, BEFORE any score from either is read as
// system quality.
func TestResolveThresholdPricesTheRequiredEffect(t *testing.T) {
	small := ResolveThreshold("score_delta", "slice-a", 12, 0.05)
	if small.Verdict != ResolutionBelowResolution {
		t.Fatalf("threshold 0.05 on n=12 = %q, want %q", small.Verdict, ResolutionBelowResolution)
	}
	big := ResolveThreshold("score_delta", "slice-b", 1000, 0.05)
	if big.Verdict != ResolutionMeasurable {
		t.Fatalf("threshold 0.05 on n=1000 = %q, want %q", big.Verdict, ResolutionMeasurable)
	}
	if small.Quantum <= big.Quantum {
		t.Errorf("a smaller population must have a coarser quantum: %v vs %v", small.Quantum, big.Quantum)
	}
}

// Arm 3: per-slice floors. ONE artifact carrying a 1,000-call arm and a 6-call
// arm must publish TWO different quanta. A global floor taken from the
// better-populated arm would pass a naive test while hiding exactly the
// distortion the contract names.
func TestPerSliceFloorsDifferInOneArtifact(t *testing.T) {
	r := &Report{}
	r.On = Arm{Label: "vdso_on", Calls: 1000, InTokens: 900, OutTokens: 100}
	r.Off = Arm{Label: "vdso_off", Calls: 6, InTokens: 1200, OutTokens: 300}

	rr := r.InstrumentResolution()
	if len(rr.Floors) != 2 {
		t.Fatalf("floors = %d, want one per arm", len(rr.Floors))
	}
	on, off := rr.Floors[0], rr.Floors[1]
	if on.RateQuantum == off.RateQuantum {
		t.Fatalf("both arms reported the same floor %v: the floor is not per-slice", on.RateQuantum)
	}
	if on.RateQuantum != 0.001 {
		t.Errorf("1000-call arm quantum = %v, want 0.001", on.RateQuantum)
	}
	if off.RateQuantum <= on.RateQuantum {
		t.Errorf("6-call arm quantum %v must be coarser than the 1000-call arm's %v", off.RateQuantum, on.RateQuantum)
	}
	if !on.Resolvable || !off.Resolvable {
		t.Error("both populated arms must be resolvable")
	}
}

// Arm 4: the percentile ceiling is disclosed, and the published percentile is
// NOT moved. On a 12-sample set Hist.pct answers a p99 request with the index-10
// sample — the value a p91 request also returns — so the label is finer than the
// grid. The contract says so out loud instead of interpolating.
func TestPercentileCeilingDisclosedWithoutMovingTheValue(t *testing.T) {
	var h Hist
	for i := 0; i < 12; i++ {
		h.RecordNs(int64(100 + i)) // 100..111, so the sample value names its own index
	}
	before := h.P99()

	small := ResolvePercentile("tool_call_p99_ns", "vdso_on", h.Count(), 99)
	if small.Verdict != ResolutionBelowResolution {
		t.Fatalf("p99 on 12 samples = %q, want %q (reason: %s)", small.Verdict, ResolutionBelowResolution, small.Reason)
	}
	// The disclosure is true: the "p99" of 12 samples is the index-10 sample,
	// not the 99th percentile and not even the maximum.
	if before != 110 {
		t.Fatalf("12-sample P99 = %d, want the index-10 sample 110", before)
	}
	if h.P99() == 111 {
		t.Error("12-sample P99 returned the maximum: the grid claim in resolution.go is wrong")
	}
	// A p91 request returns the SAME sample: the two labels are indistinguishable
	// at this population, which is exactly what BELOW_RESOLUTION reports.
	if h.pct(91) != before {
		t.Errorf("p91 = %d, p99 = %d: expected one grid point to serve both", h.pct(91), before)
	}
	if h.P99() != before {
		t.Error("disclosing the ceiling must not change the published percentile")
	}

	// p50 is resolvable on the same 12 samples: the ceiling is per-request, not
	// a blanket "small sample" refusal.
	if med := ResolvePercentile("tool_call_p50_ns", "vdso_on", h.Count(), 50); med.Verdict != ResolutionMeasurable {
		t.Errorf("p50 on 12 samples = %q, want %q", med.Verdict, ResolutionMeasurable)
	}
	// 5,000 samples resolve p99.
	if big := ResolvePercentile("tool_call_p99_ns", "vdso_on", 5000, 99); big.Verdict != ResolutionMeasurable {
		t.Errorf("p99 on 5000 samples = %q, want %q", big.Verdict, ResolutionMeasurable)
	}
	// 101 samples is the boundary: p99 lands exactly on its own grid point.
	if edge := ResolvePercentile("tool_call_p99_ns", "vdso_on", 101, 99); edge.Verdict != ResolutionMeasurable {
		t.Errorf("p99 on 101 samples = %q, want %q", edge.Verdict, ResolutionMeasurable)
	}
	if under := ResolvePercentile("tool_call_p99_ns", "vdso_on", 100, 99); under.Verdict != ResolutionBelowResolution {
		t.Errorf("p99 on 100 samples = %q, want %q", under.Verdict, ResolutionBelowResolution)
	}
	// A single sample resolves no percentile at all.
	if one := ResolvePercentile("tool_call_p50_ns", "vdso_on", 1, 50); one.Verdict != ResolutionBelowResolution {
		t.Errorf("p50 on 1 sample = %q, want %q", one.Verdict, ResolutionBelowResolution)
	}
	// No samples at all refuses as UNMEASURED, not as BELOW_RESOLUTION.
	if none := ResolvePercentile("tool_call_p99_ns", "vdso_on", 0, 99); none.Verdict != ResolutionUnmeasured {
		t.Errorf("p99 on 0 samples = %q, want %q", none.Verdict, ResolutionUnmeasured)
	}
}

// Arm 5a: the contract cannot pass vacuously. A fold with nothing in it refuses.
func TestEmptyFoldRefusesRatherThanPassing(t *testing.T) {
	rr := FoldResolution(nil)
	if rr.Conclusive {
		t.Fatal("a fold that priced nothing reported a conclusion")
	}
	if rr.Verdict != ResolutionUnmeasured {
		t.Errorf("empty fold verdict = %q, want %q", rr.Verdict, ResolutionUnmeasured)
	}
	err := rr.Refusal()
	if err == nil {
		t.Fatal("Refusal() on an empty fold = nil, want a refusal")
	}
	if !errors.Is(err, ErrResolutionRefused) {
		t.Errorf("refusal %v does not wrap ErrResolutionRefused", err)
	}
}

// Arm 5b: an all-unmeasured report must not read as a completed comparison, and
// must be distinguishable from a well-populated report whose effects are zero.
func TestAllUnmeasuredReportIsNotAGreenComparison(t *testing.T) {
	empty := &Report{}
	empty.On = Arm{Label: "vdso_on"}
	empty.Off = Arm{Label: "vdso_off"}
	vacuous := empty.InstrumentResolution()
	if vacuous.Conclusive {
		t.Fatal("a report with no observations reported a conclusion")
	}
	if vacuous.Verdict != ResolutionUnmeasured {
		t.Errorf("all-empty report verdict = %q, want %q", vacuous.Verdict, ResolutionUnmeasured)
	}
	if err := vacuous.Refusal(); !errors.Is(err, ErrResolutionRefused) {
		t.Errorf("all-empty report Refusal() = %v, want a wrapped refusal", err)
	}

	// The same report shape, populated and with genuinely zero effects, IS a
	// conclusion. Without this the refusal above would be unfalsifiable.
	real := &Report{}
	real.On = Arm{Label: "vdso_on", Calls: 1000, InTokens: 900, OutTokens: 100}
	real.Off = Arm{Label: "vdso_off", Calls: 1000, InTokens: 900, OutTokens: 100}
	real.KPIs = KPIs{VDSOHitRate: 0.75, PreflightCatchRate: 0.5, ContextPollutionRate: 0.25, TokensPerTask: 1}
	measured := real.InstrumentResolution()
	if !measured.Conclusive {
		t.Fatalf("a fully populated report refused: %s", measured.Reason)
	}
	if err := measured.Refusal(); err != nil {
		t.Errorf("Refusal() on a measurable report = %v, want nil", err)
	}
	if measured.Verdict == vacuous.Verdict {
		t.Error("an unmeasured report and a measured one carry the same verdict: the states collapsed")
	}
}

// The exact collapse the contract exists to break, stated at its sharpest: two
// reports whose published rate SCALAR is the identical float 0 — one because
// nothing was ever observed, one because 1,000 calls were observed and none
// polluted — must not be indistinguishable in the artifact. Before the readout
// the only thing separating them was an unlabelled `calls` field a consumer had
// to know to cross-check against a grid rule it had to derive itself.
func TestMeasuredZeroAndUnmeasuredZeroAreDistinguishable(t *testing.T) {
	never := &Report{}
	never.On = Arm{Label: "vdso_on", Calls: 0} // rate() would publish 0/0 as 0.0
	never.Off = Arm{Label: "vdso_off", Calls: 0}

	observed := &Report{}
	observed.On = Arm{Label: "vdso_on", Calls: 1000, Quarantines: 0, InTokens: 900, OutTokens: 100}
	observed.Off = Arm{Label: "vdso_off", Calls: 1000, InTokens: 900, OutTokens: 100}

	// Premise: the published scalar really is identical, so the readout is
	// carrying the whole distinction. If this ever stops holding the test below
	// is proving something easier than it claims.
	if never.KPIs.ContextPollutionRate != observed.KPIs.ContextPollutionRate {
		t.Fatalf("premise broken: the two reports no longer publish the same scalar (%v vs %v)",
			never.KPIs.ContextPollutionRate, observed.KPIs.ContextPollutionRate)
	}

	neverOut := readoutFor(t, never.InstrumentResolution(), "context_pollution_rate")
	obsOut := readoutFor(t, observed.InstrumentResolution(), "context_pollution_rate")

	if neverOut.Verdict != ResolutionUnmeasured {
		t.Errorf("0-call arm verdict = %q, want %q", neverOut.Verdict, ResolutionUnmeasured)
	}
	if obsOut.Verdict != ResolutionMeasurable {
		t.Errorf("1000-call arm verdict = %q, want %q", obsOut.Verdict, ResolutionMeasurable)
	}
	if neverOut.Published() == obsOut.Published() {
		t.Fatal("an unmeasured zero and a measured zero are still indistinguishable to a consumer")
	}
	// And the grid is disclosed only where one exists.
	if neverOut.Quantum != 0 || obsOut.Quantum != 0.001 {
		t.Errorf("quanta = %v (unobserved) / %v (observed), want 0 / 0.001", neverOut.Quantum, obsOut.Quantum)
	}
}

// readoutFor picks one metric's readout out of a fold, failing loudly when the
// fold does not carry it (so a renamed metric surfaces as a test error rather
// than a silently skipped assertion).
func readoutFor(t *testing.T, rr ResolutionReport, metric string) ResolutionReadout {
	t.Helper()
	for _, r := range rr.Readouts {
		if r.Metric == metric {
			return r
		}
	}
	t.Fatalf("fold carries no readout for %q", metric)
	return ResolutionReadout{}
}

// Arm 5c: a report that straddles the ceiling is MIXED and refuses. This is the
// state the contract forbids collapsing into one scalar.
func TestStraddlingReportIsMixedAndRefuses(t *testing.T) {
	r := &Report{}
	r.On = Arm{Label: "vdso_on", Calls: 12, InTokens: 900, OutTokens: 100}
	r.Off = Arm{Label: "vdso_off", Calls: 12, InTokens: 1000, OutTokens: 200}
	r.KPIs = KPIs{VDSOHitRate: 0.5, PreflightCatchRate: 0.25, ContextPollutionRate: 0.5, TokensPerTask: 83}
	r.TokenDeltaPct = 16.7

	rr := r.InstrumentResolution()
	if rr.Verdict != ResolutionMixed {
		t.Fatalf("straddling report verdict = %q, want %q (reason: %s)", rr.Verdict, ResolutionMixed, rr.Reason)
	}
	if rr.Conclusive {
		t.Error("a MIXED artifact must not report a conclusion")
	}
	if rr.Measurable == 0 || rr.BelowResolution == 0 {
		t.Errorf("MIXED must actually straddle: measurable=%d below=%d", rr.Measurable, rr.BelowResolution)
	}
}

// Arm 6a: the vocabulary is closed, every token is distinct, and every typed
// refusal the contract uses is reachable from the public API.
func TestResolutionVocabularyIsClosedAndFullyReachable(t *testing.T) {
	vocab := ResolutionVerdicts()
	if len(vocab) != 4 {
		t.Fatalf("vocabulary size = %d, want 4", len(vocab))
	}
	seen := map[ResolutionVerdict]bool{}
	for _, v := range vocab {
		if v == "" {
			t.Error("vocabulary carries an empty token")
		}
		if seen[v] {
			t.Errorf("vocabulary repeats %q", v)
		}
		seen[v] = true
	}

	reached := map[ResolutionVerdict]bool{
		ResolveClaim(ResolutionClaim{Metric: "m", Slice: "s", Denominator: 1000, Claim: 0.5, ClaimKnown: true}).Verdict: true,
		ResolveClaim(ResolutionClaim{Metric: "m", Slice: "s", Denominator: 12, Claim: 0.01, ClaimKnown: true}).Verdict:  true,
		ResolveClaim(ResolutionClaim{Metric: "m", Slice: "s", Denominator: 0, Claim: 0, ClaimKnown: true}).Verdict:      true,
	}
	mixed := FoldResolution(nil,
		ResolveClaim(ResolutionClaim{Metric: "m", Slice: "s", Denominator: 1000, Claim: 0.5, ClaimKnown: true}),
		ResolveClaim(ResolutionClaim{Metric: "m", Slice: "s", Denominator: 12, Claim: 0.01, ClaimKnown: true}),
	)
	reached[mixed.Verdict] = true
	for _, v := range vocab {
		if !reached[v] {
			t.Errorf("verdict %q is in the vocabulary but unreachable from the public API", v)
		}
	}
}

// Arm 6b: every fail-closed input path refuses. A default-to-MEASURABLE bug on
// any of these would be a silent fail-open.
func TestMissingEvidenceAlwaysRefuses(t *testing.T) {
	cases := []struct {
		name  string
		claim ResolutionClaim
	}{
		{"unnamed metric", ResolutionClaim{Slice: "s", Denominator: 10, Claim: 1, ClaimKnown: true}},
		{"unnamed slice", ResolutionClaim{Metric: "m", Denominator: 10, Claim: 1, ClaimKnown: true}},
		{"empty denominator", ResolutionClaim{Metric: "m", Slice: "s", Denominator: 0, Claim: 1, ClaimKnown: true}},
		{"negative denominator", ResolutionClaim{Metric: "m", Slice: "s", Denominator: -3, Claim: 1, ClaimKnown: true}},
		{"absent claim", ResolutionClaim{Metric: "m", Slice: "s", Denominator: 10}},
	}
	for _, tc := range cases {
		got := ResolveClaim(tc.claim)
		if got.Verdict != ResolutionUnmeasured {
			t.Errorf("%s: verdict = %q, want %q", tc.name, got.Verdict, ResolutionUnmeasured)
		}
		if got.Reason == "" {
			t.Errorf("%s: refusal carried no reason", tc.name)
		}
		if got.Published() {
			t.Errorf("%s: a refusal was published as a measurement", tc.name)
		}
	}
}

// The readout reaches the artifact consumers ingest: report.json carries the
// verdict beside the numbers it qualifies, and every pre-existing key keeps its
// name and value.
func TestReportJSONCarriesInstrumentResolution(t *testing.T) {
	r := &Report{}
	r.On = Arm{Label: "vdso_on", Calls: 1000, VDSOHits: 750, Quarantines: 250, InTokens: 900, OutTokens: 100}
	r.Off = Arm{Label: "vdso_off", Calls: 1000, InTokens: 1200, OutTokens: 300}
	r.KPIs = KPIs{VDSOHitRate: 0.75, PreflightCatchRate: 0.5, ContextPollutionRate: 0.25, TokensPerTask: 1}
	r.TokenDeltaPct = 33.33

	var generic map[string]any
	if err := json.Unmarshal(r.JSON(), &generic); err != nil {
		t.Fatalf("report JSON did not unmarshal: %v", err)
	}
	res, ok := generic["instrument_resolution"].(map[string]any)
	if !ok {
		t.Fatalf("report JSON missing \"instrument_resolution\": %s", r.JSON())
	}
	if res["schema"] != InstrumentResolutionSchema {
		t.Errorf("schema = %v, want %q", res["schema"], InstrumentResolutionSchema)
	}
	if res["verdict"] != string(ResolutionMeasurable) {
		t.Errorf("verdict = %v, want %q", res["verdict"], ResolutionMeasurable)
	}
	// Additive on the wire: the existing keys are untouched.
	for _, k := range []string{"vdso_on", "vdso_off", "kpis", "gate_primary", "token_delta_pct", "live_seam"} {
		if _, present := generic[k]; !present {
			t.Errorf("stamping the readout dropped pre-existing key %q", k)
		}
	}
	if got := generic["token_delta_pct"].(float64); got != 33.33 {
		t.Errorf("token_delta_pct = %v, want the unmodified 33.33", got)
	}
}

// The committed machine-readable receipt: one artifact carrying the accepted
// case, the refusal cases, and the adversarial all-unmeasured case, so the
// contract's behaviour is auditable without re-running anything.
// Regenerate after an intentional change with UPDATE_GOLDEN=1.
func TestInstrumentResolutionReceiptGolden(t *testing.T) {
	golden := filepath.Join("testdata", "instrument_resolution_receipt.json")
	got, err := json.MarshalIndent(buildResolutionReceipt(), "", "  ")
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	got = append(got, '\n')
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatalf("mkdir testdata: %v", err)
		}
		if err := os.WriteFile(golden, got, 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("receipt drifted from %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// receiptCase is one named case in the committed receipt.
type receiptCase struct {
	Name       string           `json:"name"`
	Role       string           `json:"role"`
	Resolution ResolutionReport `json:"resolution"`
	Refusal    string           `json:"refusal"`
}

// buildResolutionReceipt constructs the receipt deterministically from in-process
// values: no clock, no host, no I/O, so the golden is stable everywhere.
func buildResolutionReceipt() map[string]any {
	accepted := &Report{}
	accepted.On = Arm{Label: "vdso_on", Calls: 1000, VDSOHits: 750, Quarantines: 250, InTokens: 900, OutTokens: 100}
	accepted.Off = Arm{Label: "vdso_off", Calls: 1000, InTokens: 1200, OutTokens: 300}
	accepted.KPIs = KPIs{VDSOHitRate: 0.75, PreflightCatchRate: 0.5, ContextPollutionRate: 0.25, TokensPerTask: 1}
	accepted.TokenDeltaPct = 33.33

	refused := &Report{}
	refused.On = Arm{Label: "vdso_on", Calls: 12, VDSOHits: 6, Quarantines: 6, InTokens: 900, OutTokens: 100}
	refused.Off = Arm{Label: "vdso_off", Calls: 12, InTokens: 1000, OutTokens: 200}
	refused.KPIs = KPIs{VDSOHitRate: 0.5, PreflightCatchRate: 0.25, ContextPollutionRate: 0.5, TokensPerTask: 83}
	refused.TokenDeltaPct = 16.7

	vacuous := &Report{}
	vacuous.On = Arm{Label: "vdso_on"}
	vacuous.Off = Arm{Label: "vdso_off"}

	cases := []receiptCase{
		{
			Name: "accepted",
			Role: "known-positive: a 1000-call arm resolves every published scalar, so the artifact supports a conclusion",
		},
		{
			Name: "refused",
			Role: "refusal: a 12-call arm cannot resolve the p99 tail it publishes, so the artifact straddles the ceiling and refuses",
		},
		{
			Name: "adversarial_vacuous",
			Role: "adversarial: a report that observed nothing must not read as a green comparison — every slice is UNMEASURED, not zero",
		},
	}
	for i, r := range []*Report{accepted, refused, vacuous} {
		rr := r.InstrumentResolution()
		cases[i].Resolution = rr
		if err := rr.Refusal(); err != nil {
			cases[i].Refusal = err.Error()
		}
	}
	return map[string]any{
		"schema": InstrumentResolutionSchema,
		"issue":  "https://github.com/anthony-chaudhary/fak/issues/5667",
		"note": "Instrument-resolution ceilings for the A/B report shape. Generated by " +
			"TestInstrumentResolutionReceiptGolden (UPDATE_GOLDEN=1); pure in-process values, no host state.",
		"cases": cases,
	}
}
