package sweepcert

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestCanonicalEnvelopeDigestIsOrderStableAndBindsEveryIdentity(t *testing.T) {
	base := testEnvelope([]float64{1, 2, 4})
	first, err := CanonicalEnvelopeDigest(base)
	if err != nil {
		t.Fatal(err)
	}
	reversed := base
	reversed.Bindings = append([]Binding(nil), base.Bindings...)
	for left, right := 0, len(reversed.Bindings)-1; left < right; left, right = left+1, right-1 {
		reversed.Bindings[left], reversed.Bindings[right] = reversed.Bindings[right], reversed.Bindings[left]
	}
	second, err := CanonicalEnvelopeDigest(reversed)
	if err != nil || first != second || !strings.HasPrefix(first, "sha256:") {
		t.Fatalf("canonical digests = %q / %q, err=%v", first, second, err)
	}
	for i := range base.Bindings {
		changed := base
		changed.Bindings = append([]Binding(nil), base.Bindings...)
		changed.Bindings[i].Value += "-changed"
		got, err := CanonicalEnvelopeDigest(changed)
		if err != nil || got == first {
			t.Fatalf("binding %q did not alter digest: %q err=%v", changed.Bindings[i].Name, got, err)
		}
	}
	changedAxis := base
	changedAxis.Axis.Coordinates = []float64{1, 2, 8}
	if got, err := CanonicalEnvelopeDigest(changedAxis); err != nil || got == first {
		t.Fatalf("axis change did not alter digest: %q err=%v", got, err)
	}
}

func TestObservedExtremumCensorsOpenEndpointsAndPreservesRawValues(t *testing.T) {
	e := testEvidence([]float64{10, 20, 30})
	finding := ObservedExtremum(e, "throughput", Maximum)
	if finding.Status != FindingRightCensored || finding.PointID != "p3" {
		t.Fatalf("terminal extremum=%+v", finding)
	}
	if err := ValidateFinding(e, finding); err != nil {
		t.Fatal(err)
	}
	if got := *e.Points[2].Observations["throughput"].Value; got != 30 {
		t.Fatalf("raw value changed to %g", got)
	}
	e.Envelope.Axis.UpperClosed = true
	e.EnvelopeDigest, _ = CanonicalEnvelopeDigest(e.Envelope)
	restamp(&e)
	if got := ObservedExtremum(e, "throughput", Maximum); got.Status != FindingMeasured {
		t.Fatalf("closed terminal extremum=%+v", got)
	}
}

func TestConstrainedExtremumUsesTypedProvenanceAndAllConstraints(t *testing.T) {
	e := testEvidence([]float64{10, 20, 30})
	addMetric(t, &e, "latency", "ms", []float64{2, 4, 9})
	finding := ConstrainedExtremum(e, "throughput", Maximum, []Constraint{{Metric: "latency", Operator: AtOrBelow, Threshold: 5}})
	if finding.Status != FindingMeasured || finding.PointID != "p2" {
		t.Fatalf("constrained extremum=%+v", finding)
	}
	bad := e.Points[1].Observations["latency"]
	bad.Provenance.Method = "other"
	e.Points[1].Observations["latency"] = bad
	if got := ConstrainedExtremum(e, "throughput", Maximum, []Constraint{{Metric: "latency", Operator: AtOrBelow, Threshold: 5}}); got.Status != FindingInvalid || !strings.Contains(got.Reason, "provenance") {
		t.Fatalf("mixed provenance finding=%+v", got)
	}
}

func TestThresholdFoldsRefuseMultipleRegimesAndCensorEdges(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		fold   func(Evidence, string, ThresholdOperator, float64) Finding
		want   FindingStatus
	}{
		{name: "first point", values: []float64{10, 11, 12}, fold: FirstThreshold, want: FindingLeftCensored},
		{name: "beyond range", values: []float64{1, 2, 3}, fold: FirstThreshold, want: FindingRightCensored},
		{name: "multiple crossing", values: []float64{1, 10, 2, 11}, fold: FirstThreshold, want: FindingNotIdentifiable},
		{name: "isolated spike", values: []float64{1, 10, 2, 3}, fold: StableSuffixThreshold, want: FindingNotIdentifiable},
		{name: "later collapse", values: []float64{1, 10, 11, 2}, fold: StableSuffixThreshold, want: FindingNotIdentifiable},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.fold(testEvidence(tt.values), "throughput", AtOrAbove, 5)
			if got.Status != tt.want {
				t.Fatalf("finding=%+v want status %s", got, tt.want)
			}
		})
	}
}

func TestMissingIdentityInvalidReasonAndNonFiniteFailClosed(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Evidence)
		want FindingStatus
	}{
		{name: "missing", edit: func(e *Evidence) { e.Points[1].Status = PointNotMeasured }, want: FindingNotIdentifiable},
		{name: "identity", edit: func(e *Evidence) { e.Points[1].EnvelopeDigest = "sha256:other" }, want: FindingInvalid},
		{name: "undeclared invalidity", edit: func(e *Evidence) { e.Points[1].Status, e.Points[1].InvalidReason = PointInvalid, "mystery" }, want: FindingInvalid},
		{name: "non-finite", edit: func(e *Evidence) {
			e.Points[1].Observations["throughput"] = measured(math.NaN(), e.EnvelopeDigest, "tok/s")
		}, want: FindingInvalid},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := testEvidence([]float64{1, 2, 3})
			tt.edit(&e)
			got := ObservedExtremum(e, "throughput", Maximum)
			if got.Status != tt.want {
				t.Fatalf("finding=%+v want status %s", got, tt.want)
			}
			if tt.name == "missing" && (got.Interval == nil || got.Interval.LowerPointID != "p1" || got.Interval.UpperPointID != "p3") {
				t.Fatalf("missing point interval=%+v", got.Interval)
			}
			if e.Points[1].Observations["throughput"].Value == nil {
				t.Fatal("raw observation was erased")
			}
		})
	}
}

func TestFindingValidationRejectsMissingAndForgedReferences(t *testing.T) {
	e := testEvidence([]float64{1, 2, 1})
	finding := ObservedExtremum(e, "throughput", Maximum)
	finding.SupportingPoints = finding.SupportingPoints[:2]
	if err := ValidateFinding(e, finding); err == nil {
		t.Fatal("missing support reference was accepted")
	}
	finding = ObservedExtremum(e, "throughput", Maximum)
	finding.PointID = "forged"
	if err := ValidateFinding(e, finding); err == nil {
		t.Fatal("forged selected point was accepted")
	}
	thresholdEvidence := testEvidence([]float64{1, 2, 3})
	finding = FirstThreshold(thresholdEvidence, "throughput", AtOrAbove, 2)
	finding.Interval.UpperPointID = "forged"
	if err := ValidateFinding(thresholdEvidence, finding); err == nil {
		t.Fatal("forged interval point was accepted")
	}
}

func testEnvelope(coordinates []float64) Envelope {
	return Envelope{
		Axis: Axis{Name: "concurrency", Unit: "requests", Coordinates: coordinates},
		Bindings: []Binding{
			{Name: "model", Value: "qwen3.8"}, {Name: "workload", Value: "fixture"},
			{Name: "engine", Value: "fak-native"}, {Name: "configuration", Value: "cfg"},
			{Name: "capacity", Value: "4"}, {Name: "reset_order", Value: "ascending"},
			{Name: "environment", Value: "test"},
		},
	}
}

func testEvidence(values []float64) Evidence {
	coordinates := make([]float64, len(values))
	for i := range values {
		coordinates[i] = float64(i + 1)
	}
	envelope := testEnvelope(coordinates)
	digest, _ := CanonicalEnvelopeDigest(envelope)
	e := Evidence{Envelope: envelope, EnvelopeDigest: digest, DeclaredInvalidReasons: []string{"identity_drift"}}
	for i, value := range values {
		id := "p" + string(rune('1'+i))
		e.Points = append(e.Points, Point{ID: id, Coordinate: float64(i + 1), Status: PointMeasured, EnvelopeDigest: digest, Observations: map[string]Observation{"throughput": measured(value, digest, "tok/s")}})
	}
	return e
}

func addMetric(t *testing.T, e *Evidence, metric, unit string, values []float64) {
	t.Helper()
	if len(values) != len(e.Points) {
		t.Fatal("fixture metric length mismatch")
	}
	for i := range e.Points {
		e.Points[i].Observations[metric] = measured(values[i], e.EnvelopeDigest, unit)
	}
}

func restamp(e *Evidence) {
	for i := range e.Points {
		e.Points[i].EnvelopeDigest = e.EnvelopeDigest
		for name, observation := range e.Points[i].Observations {
			observation.Provenance.EnvelopeDigest = e.EnvelopeDigest
			e.Points[i].Observations[name] = observation
		}
	}
}

func measured(value float64, digest, unit string) Observation {
	return Observation{Status: ObservationMeasured, Value: &value, Provenance: Provenance{Source: "fixture", Method: "fixture-median", Unit: unit, EnvelopeDigest: digest}}
}

func TestQwen38NativeEnvelopeBindsRequiredProvenance(t *testing.T) {
	base := qwen38FixtureProvenance()
	envelope, err := NewQwen38NativeEnvelope([]float64{128, 512, 2048}, base, RangeClosure{})
	if err != nil {
		t.Fatal(err)
	}
	baseDigest, err := CanonicalEnvelopeDigest(envelope)
	if err != nil {
		t.Fatal(err)
	}
	mutations := []struct {
		name string
		edit func(*Qwen38NativeProvenance)
	}{
		{"artifact", func(p *Qwen38NativeProvenance) { p.Artifact = "fixture/qwen3.8-other" }},
		{"artifact digest", func(p *Qwen38NativeProvenance) { p.ArtifactDigest = "sha256:other" }},
		{"engine commit", func(p *Qwen38NativeProvenance) { p.EngineCommit = "fixture-commit-other" }},
		{"backend", func(p *Qwen38NativeProvenance) { p.Backend = "cuda-other" }},
		{"node", func(p *Qwen38NativeProvenance) { p.Node = "fixture-node-other" }},
		{"hardware", func(p *Qwen38NativeProvenance) { p.Hardware = "fixture-gpu-other" }},
		{"tokenizer", func(p *Qwen38NativeProvenance) { p.Tokenizer = "fixture-tokenizer-other" }},
		{"output", func(p *Qwen38NativeProvenance) { p.Output = "fixture-output-other" }},
		{"reset", func(p *Qwen38NativeProvenance) { p.Reset = "fixture-reset-other" }},
		{"order", func(p *Qwen38NativeProvenance) { p.Order = "fixture-order-other" }},
		{"capacity", func(p *Qwen38NativeProvenance) { p.Capacity = "fixture-capacity-other" }},
	}
	for _, tt := range mutations {
		t.Run(tt.name, func(t *testing.T) {
			changed := base
			tt.edit(&changed)
			other, err := NewQwen38NativeEnvelope([]float64{128, 512, 2048}, changed, RangeClosure{})
			if err != nil {
				t.Fatal(err)
			}
			digest, err := CanonicalEnvelopeDigest(other)
			if err != nil {
				t.Fatal(err)
			}
			if digest == baseDigest {
				t.Fatalf("changing %s did not change envelope digest", tt.name)
			}
		})
	}
}

func TestQwen38NativeEnvelopeRejectsMissingProvenanceAndNonNativeEngine(t *testing.T) {
	base := qwen38FixtureProvenance()
	tests := []struct {
		name string
		edit func(*Qwen38NativeProvenance)
	}{
		{"artifact", func(p *Qwen38NativeProvenance) { p.Artifact = "" }},
		{"artifact digest", func(p *Qwen38NativeProvenance) { p.ArtifactDigest = "" }},
		{"engine", func(p *Qwen38NativeProvenance) { p.Engine = "" }},
		{"engine commit", func(p *Qwen38NativeProvenance) { p.EngineCommit = "" }},
		{"backend", func(p *Qwen38NativeProvenance) { p.Backend = "" }},
		{"node", func(p *Qwen38NativeProvenance) { p.Node = "" }},
		{"hardware", func(p *Qwen38NativeProvenance) { p.Hardware = "" }},
		{"tokenizer", func(p *Qwen38NativeProvenance) { p.Tokenizer = "" }},
		{"output", func(p *Qwen38NativeProvenance) { p.Output = "" }},
		{"reset", func(p *Qwen38NativeProvenance) { p.Reset = "" }},
		{"order", func(p *Qwen38NativeProvenance) { p.Order = "" }},
		{"capacity", func(p *Qwen38NativeProvenance) { p.Capacity = "" }},
		{"llama fallback", func(p *Qwen38NativeProvenance) { p.Engine = "llama.cpp" }},
		{"vague native", func(p *Qwen38NativeProvenance) { p.Engine = "native" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := base
			tt.edit(&p)
			if _, err := NewQwen38NativeEnvelope([]float64{128, 512, 2048}, p, RangeClosure{}); err == nil {
				t.Fatal("invalid provenance was accepted")
			}
		})
	}
	if _, err := NewQwen38NativeEnvelope([]float64{128, 512}, base, RangeClosure{}); err == nil {
		t.Fatal("fewer than three prompt lengths were accepted")
	}
}

func TestQwen38NativeEvidencePreservesNotMeasuredRuns(t *testing.T) {
	e := qwen38FixtureEvidence(t, RangeClosure{}, []float64{10, 20, 30})
	e.Points[1].Status = PointNotMeasured
	e.Points[1].Observations["prefill_throughput"] = Observation{Status: ObservationNotMeasured, Reason: "fixture allocation failure"}
	if err := ValidateQwen38NativeEvidence(e); err != nil {
		t.Fatalf("explicit not-measured run rejected: %v", err)
	}
	finding := ObservedExtremum(e, "prefill_throughput", Maximum)
	if finding.Status != FindingNotIdentifiable {
		t.Fatalf("finding=%+v, want not identifiable", finding)
	}
	bad := e
	bad.Points = append([]Point(nil), e.Points...)
	bad.Points[1].Observations = map[string]Observation{}
	zero := 0.0
	bad.Points[1].Observations["prefill_throughput"] = Observation{Status: ObservationNotMeasured, Value: &zero, Reason: "fixture allocation failure"}
	if err := ValidateQwen38NativeEvidence(bad); err == nil {
		t.Fatal("numeric value on not-measured observation was accepted")
	}
	bad.Points[1].Observations["prefill_throughput"] = Observation{Status: ObservationNotMeasured}
	if err := ValidateQwen38NativeEvidence(bad); err == nil {
		t.Fatal("not-measured observation without reason was accepted")
	}
}

func TestQwen38NativeTerminalMaximumRequiresRangeClosureProof(t *testing.T) {
	open := qwen38FixtureEvidence(t, RangeClosure{}, []float64{10, 20, 30})
	finding := ObservedExtremum(open, "prefill_throughput", Maximum)
	if finding.Status != FindingRightCensored {
		t.Fatalf("open terminal maximum=%+v, want right-censored", finding)
	}
	closed := qwen38FixtureEvidence(t, RangeClosure{Proven: true, Evidence: "fixture capacity boundary witness"}, []float64{10, 20, 30})
	finding = ObservedExtremum(closed, "prefill_throughput", Maximum)
	if finding.Status != FindingMeasured {
		t.Fatalf("proven closed terminal maximum=%+v, want measured", finding)
	}
	if _, err := NewQwen38NativeEnvelope([]float64{128, 512, 2048}, qwen38FixtureProvenance(), RangeClosure{Proven: true}); err == nil {
		t.Fatal("range closure without evidence was accepted")
	}
}

func qwen38FixtureProvenance() Qwen38NativeProvenance {
	return Qwen38NativeProvenance{
		Artifact: "fixture/qwen3.8", ArtifactDigest: "sha256:fixture-artifact", Engine: Qwen38NativeEngine,
		EngineCommit: "fixture-commit", Backend: "fixture-cuda", Node: "fixture-node", Hardware: "fixture-gpu",
		Tokenizer: "sha256:fixture-tokenizer", Output: "fixture-logits-shape", Reset: "fixture-cold-reset",
		Order: "fixture-ascending", Capacity: "fixture-capacity",
	}
}

func qwen38FixtureEvidence(t *testing.T, closure RangeClosure, values []float64) Evidence {
	t.Helper()
	coordinates := []float64{128, 512, 2048}
	envelope, err := NewQwen38NativeEnvelope(coordinates, qwen38FixtureProvenance(), closure)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := CanonicalEnvelopeDigest(envelope)
	if err != nil {
		t.Fatal(err)
	}
	e := Evidence{Envelope: envelope, EnvelopeDigest: digest}
	for i, coordinate := range coordinates {
		e.Points = append(e.Points, Point{
			ID: fmt.Sprintf("prompt-%d", int(coordinate)), Coordinate: coordinate, Status: PointMeasured, EnvelopeDigest: digest,
			Observations: map[string]Observation{"prefill_throughput": measured(values[i], digest, "tok/s")},
		})
	}
	if err := ValidateQwen38NativeEvidence(e); err != nil {
		t.Fatal(err)
	}
	return e
}

func TestSweepCertPrefillResponseSurface(t *testing.T) {
	sc := NewSweepCert()

	// 1. Monotonic increasing prefill throughput surface (expected as chunking/batch efficiency scales)
	passingSamples := []PrefillSample{
		{PromptLength: 128, Throughput: 1200.0, LatencyMs: 106.6},
		{PromptLength: 512, Throughput: 2800.0, LatencyMs: 182.8},
		{PromptLength: 1024, Throughput: 4200.0, LatencyMs: 243.8},
		{PromptLength: 2048, Throughput: 5600.0, LatencyMs: 365.7},
	}

	criteria := &CertificationCriteria{
		MaxVariance:  1e8,
		MaxCV:        0.6,
		Monotonicity: MonotonicNonDecreasing,
	}

	res, err := sc.CertifyPrefillSurface(passingSamples, criteria)
	if err != nil {
		t.Fatalf("CertifyPrefillSurface failed: %v", err)
	}
	if !res.Passed || !res.Monotonic {
		t.Fatalf("expected passing monotonic surface, got res=%+v", res)
	}
	if res.Digest == "" || !strings.HasPrefix(res.Digest, "sha256:") {
		t.Fatalf("expected valid sha256 digest, got %q", res.Digest)
	}

	// 2. Monotonicity violation: throughput drops at 1024 tokens
	regressedSamples := []PrefillSample{
		{PromptLength: 128, Throughput: 1200.0},
		{PromptLength: 512, Throughput: 2800.0},
		{PromptLength: 1024, Throughput: 2100.0}, // Regression!
		{PromptLength: 2048, Throughput: 5600.0},
	}

	resRegressed, err := sc.CertifyPrefillSurface(regressedSamples, criteria)
	if err != nil {
		t.Fatalf("CertifyPrefillSurface failed: %v", err)
	}
	if resRegressed.Passed || resRegressed.Monotonic {
		t.Fatalf("expected failure on monotonic regression, got res=%+v", resRegressed)
	}
	if len(resRegressed.Violations) == 0 {
		t.Fatal("expected violations on regressed surface")
	}

	// 3. Strict variance ceiling violation
	strictCriteria := &CertificationCriteria{
		MaxVariance:  10.0, // unrealistically low variance ceiling
		Monotonicity: MonotonicNonDecreasing,
	}
	resHighVar, err := sc.CertifyPrefillSurface(passingSamples, strictCriteria)
	if err != nil {
		t.Fatalf("CertifyPrefillSurface failed: %v", err)
	}
	if resHighVar.Passed {
		t.Fatalf("expected failure on high variance, got res=%+v", resHighVar)
	}
}

func TestSweepCertTokenLatencyDistribution(t *testing.T) {
	sc := NewSweepCert()

	// 1. Stable ITL distribution with low jitter
	stableLatencies := TokenLatencyDistribution{
		LatenciesMs: []float64{15.2, 15.4, 15.1, 15.5, 15.3, 15.2, 15.6},
		MaxJitterMs: 2.0,
	}
	criteria := &CertificationCriteria{
		MaxVariance: 5.0,
		MaxCV:       0.1,
	}

	res, err := sc.CertifyTokenLatencyDistribution(stableLatencies, criteria)
	if err != nil {
		t.Fatalf("CertifyTokenLatencyDistribution failed: %v", err)
	}
	if !res.Passed {
		t.Fatalf("expected stable token latencies to pass, got %+v", res)
	}

	// 2. High jitter violation
	jitteryLatencies := TokenLatencyDistribution{
		LatenciesMs: []float64{15.0, 35.0, 15.0, 40.0}, // Jitter up to 25 ms
		MaxJitterMs: 5.0,
	}
	resJitter, err := sc.CertifyTokenLatencyDistribution(jitteryLatencies, criteria)
	if err != nil {
		t.Fatalf("CertifyTokenLatencyDistribution failed: %v", err)
	}
	if resJitter.Passed {
		t.Fatalf("expected jitter failure, got res=%+v", resJitter)
	}

	// 3. Variance limit violation
	resVar, err := sc.CertifyTokenLatencyDistribution(jitteryLatencies, &CertificationCriteria{MaxVariance: 1.0})
	if err != nil {
		t.Fatalf("CertifyTokenLatencyDistribution failed: %v", err)
	}
	if resVar.Passed {
		t.Fatalf("expected variance limit failure, got res=%+v", resVar)
	}
}

func TestSweepCertAgentTurns(t *testing.T) {
	sc := NewSweepCert()

	// 1. Valid multi-turn session with monotonically increasing cumulative tokens and bounded turn latency
	turns := []AgentTurnSample{
		{TurnIndex: 1, PromptTokens: 200, CompletionTokens: 50, CumulativeTokens: 250, TurnDurationMs: 850.0},
		{TurnIndex: 2, PromptTokens: 300, CompletionTokens: 60, CumulativeTokens: 360, TurnDurationMs: 920.0},
		{TurnIndex: 3, PromptTokens: 400, CompletionTokens: 45, CumulativeTokens: 445, TurnDurationMs: 890.0},
		{TurnIndex: 4, PromptTokens: 500, CompletionTokens: 80, CumulativeTokens: 580, TurnDurationMs: 1100.0},
	}

	criteria := &CertificationCriteria{
		MaxVariance:       1e5,
		MaxCV:             0.5,
		MaxTurnDurationMs: 5000.0,
	}

	res, err := sc.CertifyAgentTurns(turns, criteria)
	if err != nil {
		t.Fatalf("CertifyAgentTurns failed: %v", err)
	}
	if !res.Passed {
		t.Fatalf("expected passing agent turns, got res=%+v", res)
	}

	// 2. Cumulative token rollback violation (turns cannot lose progress)
	brokenTurns := []AgentTurnSample{
		{TurnIndex: 1, CumulativeTokens: 250, TurnDurationMs: 850.0},
		{TurnIndex: 2, CumulativeTokens: 150, TurnDurationMs: 920.0}, // Regressed!
	}
	resBroken, err := sc.CertifyAgentTurns(brokenTurns, criteria)
	if err != nil {
		t.Fatalf("CertifyAgentTurns failed: %v", err)
	}
	if resBroken.Passed {
		t.Fatalf("expected failure on regressed cumulative tokens, got res=%+v", resBroken)
	}

	// 3. Turn duration budget violation
	slowCriteria := &CertificationCriteria{
		MaxTurnDurationMs: 1000.0, // Turn 4 took 1100 ms
	}
	resSlow, err := sc.CertifyAgentTurns(turns, slowCriteria)
	if err != nil {
		t.Fatalf("CertifyAgentTurns failed: %v", err)
	}
	if resSlow.Passed {
		t.Fatalf("expected failure on slow turn, got res=%+v", resSlow)
	}
}

func TestSweepCertFailClosed(t *testing.T) {
	sc := NewSweepCert()

	// Empty prefill points
	if _, err := sc.CertifyPrefillSurface(nil, nil); err == nil {
		t.Fatal("expected error on empty prefill points")
	}

	// Non-finite prefill sample
	badPrefill := []PrefillSample{{PromptLength: 100, Throughput: math.NaN()}}
	if _, err := sc.CertifyPrefillSurface(badPrefill, nil); err == nil {
		t.Fatal("expected error on NaN throughput")
	}

	// Empty token latencies
	if _, err := sc.CertifyTokenLatencyDistribution(TokenLatencyDistribution{}, nil); err == nil {
		t.Fatal("expected error on empty token latencies")
	}

	// Negative latency
	badLatencies := TokenLatencyDistribution{LatenciesMs: []float64{-10.0}}
	if _, err := sc.CertifyTokenLatencyDistribution(badLatencies, nil); err == nil {
		t.Fatal("expected error on negative latency")
	}

	// Empty agent turns
	if _, err := sc.CertifyAgentTurns(nil, nil); err == nil {
		t.Fatal("expected error on empty agent turns")
	}

	// Negative turn duration
	badTurns := []AgentTurnSample{{TurnIndex: 1, TurnDurationMs: -1.0}}
	if _, err := sc.CertifyAgentTurns(badTurns, nil); err == nil {
		t.Fatal("expected error on negative turn duration")
	}
}
