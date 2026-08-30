package sweepcert

import (
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
