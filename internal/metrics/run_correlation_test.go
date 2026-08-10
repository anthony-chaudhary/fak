package metrics

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// producerIssuedSeries is the known-positive shape: every stream fak folds over
// one run copied the same producer-issued identity.
func producerIssuedSeries() []CorrelationSeries {
	return []CorrelationSeries{
		{Stream: "worker", RunID: "run-2f8c1a", Source: CorrelationSourceProducerIssued},
		{Stream: "tool", RunID: "run-2f8c1a", Source: CorrelationSourceProducerIssued},
		{Stream: "session", RunID: "run-2f8c1a", Source: CorrelationSourceProducerIssued},
		{Stream: "annotation", RunID: "run-2f8c1a", Source: CorrelationSourceProducerIssued},
		{Stream: "summary", RunID: "run-2f8c1a", Source: CorrelationSourceProducerIssued},
	}
}

func TestCorrelateRunSeriesAcceptsOneCopiedIdentity(t *testing.T) {
	got := CorrelateRunSeries(producerIssuedSeries())
	if !got.Accepted || got.Refusal != CorrelationRefusalNone {
		t.Fatalf("result = %+v, want accepted with no refusal", got)
	}
	if got.CanonicalRunID != "run-2f8c1a" || got.Streams != 5 || got.RefusedIndex != -1 {
		t.Fatalf("result = %+v, want canonical run-2f8c1a across 5 streams", got)
	}
	if got.Schema != RunCorrelationSchema {
		t.Fatalf("schema = %q, want %q", got.Schema, RunCorrelationSchema)
	}
	if err := got.RefusalError(); err != nil {
		t.Fatalf("accepted result still refused: %v", err)
	}
	if err := ValidateRunCorrelationResult(got); err != nil {
		t.Fatal(err)
	}
}

// TestCorrelateRunSeriesRefusesReconstructedIdentityThatMatches is the
// non-vacuity witness: every series carries the SAME identity string, so a
// string-equality correlation check passes. The contract still refuses, because
// one stream reconstructed the identity from a display label instead of copying
// it — equal strings are not agreement.
func TestCorrelateRunSeriesRefusesReconstructedIdentityThatMatches(t *testing.T) {
	series := producerIssuedSeries()
	series[3].Source = CorrelationSourceDisplayLabel

	for _, s := range series {
		if s.RunID != series[0].RunID {
			t.Fatalf("fixture is not adversarial: %q differs from %q", s.RunID, series[0].RunID)
		}
	}

	got := CorrelateRunSeries(series)
	if got.Accepted || got.Refusal != CorrelationRefusalFallbackIdentity {
		t.Fatalf("result = %+v, want refusal %q", got, CorrelationRefusalFallbackIdentity)
	}
	if got.RefusedStream != "annotation" || got.RefusedIndex != 3 {
		t.Fatalf("result = %+v, want the annotation series at index 3 named", got)
	}
	if got.CanonicalRunID != "" {
		t.Fatalf("refused result published canonical id %q", got.CanonicalRunID)
	}
	if err := ValidateRunCorrelationResult(got); err != nil {
		t.Fatal(err)
	}
}

func TestCorrelateRunSeriesTypedRefusals(t *testing.T) {
	tests := []struct {
		name   string
		series []CorrelationSeries
		want   CorrelationRefusal
		stream string
		index  int
	}{
		{"nothing presented", nil, CorrelationRefusalNoSeries, "", -1},
		{
			"stream unnamed",
			[]CorrelationSeries{
				{Stream: "worker", RunID: "run-1", Source: CorrelationSourceProducerIssued},
				{RunID: "run-1", Source: CorrelationSourceProducerIssued},
			},
			CorrelationRefusalUnnamedStream, "", 1,
		},
		{
			"identity absent",
			[]CorrelationSeries{
				{Stream: "worker", RunID: "run-1", Source: CorrelationSourceProducerIssued},
				{Stream: "tool", Source: CorrelationSourceProducerIssued},
			},
			CorrelationRefusalMissingIdentity, "tool", 1,
		},
		{
			"provenance absent",
			[]CorrelationSeries{
				{Stream: "worker", RunID: "run-1", Source: CorrelationSourceProducerIssued},
				{Stream: "tool", RunID: "run-1"},
			},
			CorrelationRefusalUnrecordedSource, "tool", 1,
		},
		{
			"nested identifier lifted",
			[]CorrelationSeries{
				{Stream: "worker", RunID: "run-1", Source: CorrelationSourceProducerIssued},
				{Stream: "summary", RunID: "run-1", Source: CorrelationSourceNestedIdentifier},
			},
			CorrelationRefusalFallbackIdentity, "summary", 1,
		},
		{
			"identities diverge",
			[]CorrelationSeries{
				{Stream: "worker", RunID: "run-1", Source: CorrelationSourceProducerIssued},
				{Stream: "tool", RunID: "run-2", Source: CorrelationSourceProducerIssued},
			},
			CorrelationRefusalDivergentIdentity, "tool", 1,
		},
		{
			"one stream agrees with itself",
			[]CorrelationSeries{
				{Stream: "worker", RunID: "run-1", Source: CorrelationSourceProducerIssued},
				{Stream: "worker", RunID: "run-1", Source: CorrelationSourceProducerIssued},
			},
			CorrelationRefusalSingleStream, "", -1,
		},
	}

	seen := map[CorrelationRefusal]bool{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CorrelateRunSeries(tt.series)
			if got.Accepted || got.Refusal != tt.want {
				t.Fatalf("result = %+v, want refusal %q", got, tt.want)
			}
			if got.RefusedStream != tt.stream || got.RefusedIndex != tt.index {
				t.Fatalf("result = %+v, want offender %q at index %d", got, tt.stream, tt.index)
			}
			if got.Reason == "" {
				t.Fatalf("refusal %q carries no reason", got.Refusal)
			}
			if err := ValidateRunCorrelationResult(got); err != nil {
				t.Fatal(err)
			}
			seen[got.Refusal] = true
		})
	}

	for _, refusal := range CorrelationRefusals() {
		if !seen[refusal] {
			t.Errorf("typed refusal %q is never exercised", refusal)
		}
	}
	if len(seen) != len(CorrelationRefusals()) {
		t.Errorf("exercised %d refusals, vocabulary declares %d", len(seen), len(CorrelationRefusals()))
	}
}

func TestRunCorrelationRefusalWrapsSentinel(t *testing.T) {
	err := CorrelateRunSeries(nil).RefusalError()
	if !errors.Is(err, ErrRunCorrelationRefused) {
		t.Fatalf("err = %v, want it to wrap ErrRunCorrelationRefused", err)
	}
}

func TestValidateRunCorrelationResultRejectsDrift(t *testing.T) {
	accepted := CorrelateRunSeries(producerIssuedSeries())

	forged := accepted
	forged.Series = append([]CorrelationSeries(nil), accepted.Series...)
	forged.Series[2].Source = CorrelationSourceDisplayLabel
	if err := ValidateRunCorrelationResult(forged); err == nil {
		t.Fatal("a receipt whose verdict contradicts its preserved series validated")
	}

	rescheme := accepted
	rescheme.Schema = "fak.run_correlation.v0"
	if err := ValidateRunCorrelationResult(rescheme); err == nil {
		t.Fatal("a receipt carrying an unknown schema validated")
	}

	refused := CorrelateRunSeries(nil)
	untyped := refused
	untyped.Refusal = CorrelationRefusalNone
	if err := ValidateRunCorrelationResult(untyped); err == nil {
		t.Fatal("a refused receipt with no typed refusal validated")
	}
}

func TestRunCorrelationVocabulariesAreClosed(t *testing.T) {
	sources := CorrelationSources()
	if len(sources) != 4 || sources[1] != CorrelationSourceProducerIssued {
		t.Fatalf("sources = %v, want the four-token vocabulary with producer_issued second", sources)
	}
	accepting := 0
	for _, source := range sources {
		series := []CorrelationSeries{
			{Stream: "worker", RunID: "run-1", Source: source},
			{Stream: "tool", RunID: "run-1", Source: source},
		}
		if CorrelateRunSeries(series).Accepted {
			accepting++
		}
	}
	if accepting != 1 {
		t.Fatalf("%d sources are accepted, want exactly producer_issued", accepting)
	}
}

// runCorrelationCase is one named case in the committed receipt.
type runCorrelationCase struct {
	Name        string               `json:"name"`
	Role        string               `json:"role"`
	Correlation RunCorrelationResult `json:"correlation"`
	Refusal     string               `json:"refusal"`
}

// buildRunCorrelationReceipt constructs the receipt deterministically from
// in-process values: no clock, no host, no I/O, so the golden is stable
// everywhere.
func buildRunCorrelationReceipt() map[string]any {
	divergent := producerIssuedSeries()
	divergent[1].RunID = "run-91b07d"

	reconstructed := producerIssuedSeries()
	reconstructed[3].Source = CorrelationSourceDisplayLabel

	single := []CorrelationSeries{
		{Stream: "worker", RunID: "run-2f8c1a", Source: CorrelationSourceProducerIssued},
		{Stream: "worker", RunID: "run-2f8c1a", Source: CorrelationSourceProducerIssued},
	}

	inputs := []struct {
		name   string
		role   string
		series []CorrelationSeries
	}{
		{
			"accepted",
			"known-positive: all five streams fak folds over one run copied the producer-issued identity, so the run correlates",
			producerIssuedSeries(),
		},
		{
			"refused",
			"refusal: the tool stream copied a different identity, so the series describe two runs and the fold refuses instead of splitting one run silently",
			divergent,
		},
		{
			"adversarial_equal_strings",
			"adversarial: every identity string matches, so a string-equality check passes vacuously — the annotation stream reconstructed its identity from a display label and is refused on provenance",
			reconstructed,
		},
		{
			"boundary_single_stream",
			"boundary: one stream repeated agrees with itself and witnesses no cross-stream correlation, so the contract fails closed rather than accepting a self-consistent set",
			single,
		},
	}

	cases := make([]runCorrelationCase, 0, len(inputs))
	for _, in := range inputs {
		result := CorrelateRunSeries(in.series)
		refusal := ""
		if err := result.RefusalError(); err != nil {
			refusal = err.Error()
		}
		cases = append(cases, runCorrelationCase{Name: in.name, Role: in.role, Correlation: result, Refusal: refusal})
	}

	return map[string]any{
		"schema": RunCorrelationSchema,
		"issue":  "https://github.com/anthony-chaudhary/fak/issues/5688",
		"note": "One canonical producer-issued correlation identity across a run's measurement series. Generated by " +
			"TestRunCorrelationReceiptGolden (UPDATE_GOLDEN=1); pure in-process values, no host state.",
		"cases": cases,
	}
}

func TestRunCorrelationReceiptGolden(t *testing.T) {
	golden := filepath.Join("testdata", "run_correlation_receipt.json")
	got, err := json.MarshalIndent(buildRunCorrelationReceipt(), "", "  ")
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

// TestRunCorrelationCommittedReceiptRevalidates re-reads the committed receipt
// from disk and re-derives every case's verdict from the series that case
// preserved, so the fixture cannot drift away from the contract it witnesses.
func TestRunCorrelationCommittedReceiptRevalidates(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "run_correlation_receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct {
		Schema string               `json:"schema"`
		Cases  []runCorrelationCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != RunCorrelationSchema {
		t.Fatalf("receipt schema = %q, want %q", receipt.Schema, RunCorrelationSchema)
	}
	if len(receipt.Cases) != 4 {
		t.Fatalf("receipt carries %d cases, want the accepted, refusal, adversarial, and boundary cases", len(receipt.Cases))
	}
	accepted, refused := 0, 0
	for _, c := range receipt.Cases {
		if err := ValidateRunCorrelationResult(c.Correlation); err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		if c.Correlation.Accepted {
			accepted++
			if c.Refusal != "" {
				t.Errorf("%s: accepted case carries refusal text %q", c.Name, c.Refusal)
			}
			continue
		}
		refused++
		if c.Refusal == "" {
			t.Errorf("%s: refused case carries no rendered refusal", c.Name)
		}
	}
	if accepted != 1 || refused != 3 {
		t.Fatalf("receipt has %d accepted / %d refused cases, want 1 accepted and 3 refused", accepted, refused)
	}
}
