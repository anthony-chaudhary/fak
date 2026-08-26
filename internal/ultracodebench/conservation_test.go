package ultracodebench

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func readConservationReceipt(t *testing.T, name string) ConservationReceipt {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	var r ConservationReceipt
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func TestObservedConservationReceiptMatchesCommittedReport(t *testing.T) {
	r := readConservationReceipt(t, "issue8676-balanced-receipt.json")
	got := EvaluateConservation(r)
	b, err := os.ReadFile("testdata/issue8676-balanced-report.json")
	if err != nil {
		t.Fatal(err)
	}
	var want ConservationReport
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("report mismatch\n got: %#v\nwant: %#v", got, want)
	}
	if got.ByClass[WorkScopedAway] != 627 || got.ByClass[WorkPrefixRead] != 373 {
		t.Fatalf("unexpected attribution: %#v", got.ByClass)
	}
}

func TestConservationAbstainsOnOverlappingOwnership(t *testing.T) {
	r := readConservationReceipt(t, "issue8676-balanced-receipt.json")
	r.Spans[1].Start = 600
	got := EvaluateConservation(r)
	if got.Verdict != ScopedPrefixAbstain || got.Conserved {
		t.Fatalf("overlap admitted: %#v", got)
	}
}

func TestConservationAbstainsInsteadOfInferringMissingCacheTokens(t *testing.T) {
	r := readConservationReceipt(t, "issue8676-balanced-receipt.json")
	r.Spans = r.Spans[:1]
	got := EvaluateConservation(r)
	if got.Verdict != ScopedPrefixAbstain || got.ByClass[WorkPrefixRead] != 0 {
		t.Fatalf("missing cache span inferred: %#v", got)
	}
}

func TestConservationAbstainsOnUnknownOwnershipOrMissingRawTelemetry(t *testing.T) {
	r := readConservationReceipt(t, "issue8676-balanced-receipt.json")
	r.Spans[1].Class = WorkUnknown
	if got := EvaluateConservation(r); got.Verdict != ScopedPrefixAbstain {
		t.Fatalf("unknown ownership admitted: %#v", got)
	}
	r = readConservationReceipt(t, "issue8676-balanced-receipt.json")
	r.RawRuntime = nil
	if got := EvaluateConservation(r); got.Verdict != ScopedPrefixAbstain {
		t.Fatalf("missing raw telemetry admitted: %#v", got)
	}
}

func TestConservationAbstainsOnUnequalOutcomeOrIncompleteEnvelope(t *testing.T) {
	r := readConservationReceipt(t, "issue8676-balanced-receipt.json")
	r.AcceptedOutcomeEqual = false
	if got := EvaluateConservation(r); got.Verdict != ScopedPrefixAbstain {
		t.Fatalf("unequal outcome admitted: %#v", got)
	}
	r = readConservationReceipt(t, "issue8676-balanced-receipt.json")
	r.Envelope.Tokenizer = ""
	if got := EvaluateConservation(r); got.Verdict != ScopedPrefixAbstain {
		t.Fatalf("incomplete envelope admitted: %#v", got)
	}
}
