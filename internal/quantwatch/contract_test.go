package quantwatch_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantwatch"
)

func TestOfflineFixtureRanksAndDeduplicatesDeterministically(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "ranking-v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	first := quantwatch.IngestSnapshot(raw)
	second := quantwatch.IngestSnapshot(raw)
	if first.Outcome != quantwatch.OutcomeRanked {
		t.Fatalf("outcome=%q reason=%q", first.Outcome, first.Reason)
	}
	if len(first.Items) != 3 {
		t.Fatalf("items=%d, want 3 after dedupe", len(first.Items))
	}
	if first.Items[0].ID != "github:acme/quant-runtime:v2.1.0" {
		t.Fatalf("first=%q", first.Items[0].ID)
	}
	if first.Items[1].ID != "arxiv:2608.00002" {
		t.Fatalf("second=%q", first.Items[1].ID)
	}
	if first.Items[2].ID != "arxiv:2607.99999" {
		t.Fatalf("third=%q", first.Items[2].ID)
	}
	if first.Items[0].Score != second.Items[0].Score {
		t.Fatal("ranking is not deterministic")
	}
	if first.Deduplicated != 1 {
		t.Fatalf("deduplicated=%d", first.Deduplicated)
	}
	if first.Items[0].Claims.HardwareEnvelope.Status != quantwatch.ClaimUnknown {
		t.Fatal("watchlist invented a hardware envelope")
	}
}

func TestUnknownVersionAbstainsTyped(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "unknown-version.json"))
	if err != nil {
		t.Fatal(err)
	}
	got := quantwatch.IngestSnapshot(raw)
	if got.Outcome != quantwatch.OutcomeAbstain || got.Reason != quantwatch.ReasonUnknownVersion {
		t.Fatalf("got outcome=%q reason=%q", got.Outcome, got.Reason)
	}
	if len(got.Sources) != 1 || !got.Sources[0].Abstained {
		t.Fatalf("missing abstention receipt: %#v", got.Sources)
	}
}

func TestUnsupportedSourceIsTyped(t *testing.T) {
	raw := []byte(`{"schema":"fak.quantwatch.snapshot/v1","query_time":"2026-08-10T12:00:00Z","sources":[{"kind":"social_post","name":"feed","records":[]}]}`)
	got := quantwatch.IngestSnapshot(raw)
	if got.Outcome != quantwatch.OutcomeUnsupported || got.Reason != quantwatch.ReasonSourceNotHandled {
		t.Fatalf("got outcome=%q reason=%q", got.Outcome, got.Reason)
	}
}

func TestStrictSnapshotUnknownFieldRefuses(t *testing.T) {
	raw := []byte(`{"schema":"fak.quantwatch.snapshot/v1","query_time":"2026-08-10T12:00:00Z","sources":[],"future_field":true}`)
	got := quantwatch.IngestSnapshot(raw)
	if got.Outcome != quantwatch.OutcomeRefused || got.Reason != quantwatch.ReasonMalformedInput {
		t.Fatalf("got outcome=%q reason=%q", got.Outcome, got.Reason)
	}
}

func TestPublisherMeasurementIsNotPromotedToWitnessedHardware(t *testing.T) {
	raw := []byte(`{"schema":"fak.quantwatch.snapshot/v1","query_time":"2026-08-10T12:00:00Z","sources":[{"kind":"arxiv","name":"paper","records":[{"id":"paper:1","title":"Quantization","url":"https://example.invalid/paper","published_at":"2026-08-09T00:00:00Z","claims":{"hardware_envelope":{"status":"measured"}}}]}]}`)
	got := quantwatch.IngestSnapshot(raw)
	if got.Outcome != quantwatch.OutcomeRanked {
		t.Fatalf("outcome=%q", got.Outcome)
	}
	if got.Items[0].Claims.HardwareEnvelope.Status != quantwatch.ClaimReported {
		t.Fatalf("hardware status=%q, want reported (not independently measured)", got.Items[0].Claims.HardwareEnvelope.Status)
	}
}
