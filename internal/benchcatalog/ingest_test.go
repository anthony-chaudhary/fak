package benchcatalog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixturesDir holds the committed snapshot fixtures the ingestor reads. Each
// good fixture names its official source URL in the file itself, so the witness
// is self-documenting: the test data IS the citation.
const fixturesDir = "testdata/modelscore-ingest"

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(fixturesDir, name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return data
}

// TestBenchmarkIngestThreeSnapshotsEmitValidatedRows is the Done-condition
// witness: given the three good fixture snapshots (Terminal-Bench 2.1, SWE-bench
// Verified, FrontierSWE), the ingestor emits validated modelscore rows — one
// model's evidence carrying its rows from every benchmark it appears in, each in
// its own native unit with source, version, and date provenance intact.
func TestBenchmarkIngestThreeSnapshotsEmitValidatedRows(t *testing.T) {
	tb := readFixture(t, "terminal-bench-2.1.json")
	swe := readFixture(t, "swe-bench-verified.json")
	fsw := readFixture(t, "frontier-swe.json")

	reg, err := IngestBytes(
		[]string{"terminal-bench-2.1.json", "swe-bench-verified.json", "frontier-swe.json"},
		tb, swe, fsw,
	)
	if err != nil {
		t.Fatalf("ingest three good snapshots: %v", err)
	}

	// opus appears in all three benchmarks (twice in FrontierSWE: mean@5 and
	// best@5), so its evidence must carry every row, never collapsed.
	prof, ok := reg.Profile("opus")
	if !ok {
		t.Fatal("opus missing from ingested registry")
	}
	if got := len(prof.Benchmarks); got != 4 {
		t.Fatalf("opus benchmark rows = %d, want 4 (terminal-bench + swe-bench + frontier mean@5 + frontier best@5)", got)
	}

	// The terminal-bench row must keep its harness — the model alone did not earn
	// an agent-plus-model number.
	tbRow, ok := prof.Benchmark("terminal-bench")
	if !ok {
		t.Fatal("opus has no terminal-bench row")
	}
	if tbRow.Harness == "" {
		t.Fatal("terminal-bench row dropped its harness (it scores an agent PLUS a model)")
	}
	if tbRow.Version != "2.1" { //boundarylint:ignore CHANGE_DETECTOR_TEST fixture round-trip fidelity: ingest must preserve the version verbatim
		t.Fatalf("terminal-bench version = %q, want 2.1", tbRow.Version)
	}
	if tbRow.Unit != "pct-resolved" {
		t.Fatalf("terminal-bench unit = %q, want native pct-resolved (never normalized)", tbRow.Unit)
	}

	// Provenance must survive the ingest: source URL, capture date, and a
	// confidence folded from the official trust class.
	if !strings.Contains(tbRow.Provenance.Source, "tbench.ai") {
		t.Fatalf("terminal-bench source = %q, want the tbench.ai leaderboard URL", tbRow.Provenance.Source)
	}
	if tbRow.Provenance.AsOf == "" {
		t.Fatal("terminal-bench row lost its as_of capture date")
	}
	if tbRow.Provenance.Confidence != ConfidenceOfficial.Weight() {
		t.Fatalf("terminal-bench confidence = %v, want official weight %v", tbRow.Provenance.Confidence, ConfidenceOfficial.Weight())
	}

	// SWE-bench and FrontierSWE are DIFFERENT axes, not one rank: the units must
	// stay distinct on opus's profile.
	sweRow, ok := prof.Benchmark("swe-bench-verified")
	if !ok {
		t.Fatal("opus has no swe-bench-verified row")
	}
	fswRow, ok := prof.Benchmark("frontier-swe")
	if !ok {
		t.Fatal("opus has no frontier-swe row")
	}
	if sweRow.Unit == fswRow.Unit {
		t.Fatalf("swe-bench and frontier-swe collapsed to one unit %q — they are different axes", sweRow.Unit)
	}

	// A community/vendor row is still ingested, just weighted lower — trust is
	// evidence about the evidence, not an admission gate at ingest.
	glm, ok := reg.Profile("glm-5.2")
	if !ok {
		t.Fatal("glm-5.2 (community/vendor rows) was dropped; lower trust must still ingest")
	}
	if got := len(glm.Benchmarks); got != 2 {
		t.Fatalf("glm-5.2 rows = %d, want 2 (terminal-bench community + swe-bench vendor)", got)
	}
}

// TestBenchmarkIngestRefusesMissingProvenance is the fail-closed half of the Done
// condition: a row missing source, version, date, model, or metric — or an
// agent-plus-model row with no harness — is refused loud, never half-ingested.
func TestBenchmarkIngestRefusesMissingProvenance(t *testing.T) {
	cases := []struct {
		fixture string
		wantSub string
	}{
		{"bad-missing-source.json", "source is required"},
		{"bad-terminal-no-harness.json", "harness is required"},
		{"bad-missing-version.json", "version is required"},
	}
	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			data := readFixture(t, tc.fixture)
			_, err := IngestBytes([]string{tc.fixture}, data)
			if err == nil {
				t.Fatalf("%s: ingest accepted an under-provenanced row, want refusal", tc.fixture)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("%s: error = %q, want it to mention %q", tc.fixture, err.Error(), tc.wantSub)
			}
		})
	}
}

// TestBenchmarkIngestRefusesInProcessMissingModelAndMetric covers the remaining
// two required fields (model, metric) via in-process Snapshot values, so the
// witness does not depend on a fixture for every refusal branch and the field
// list stays exhaustive against the Done condition.
func TestBenchmarkIngestRefusesInProcessMissingModelAndMetric(t *testing.T) {
	base := func() Snapshot {
		return Snapshot{
			Schema:    IngestSchema,
			Benchmark: "swe-bench-verified",
			Version:   "verified",
			Rows: []SnapshotRow{{
				Model:      "opus",
				Score:      74.5,
				Metric:     "pct-resolved",
				Source:     "https://www.swebench.com/",
				AsOf:       "2026-07-06",
				Confidence: ConfidenceOfficial,
			}},
		}
	}

	t.Run("missing model", func(t *testing.T) {
		s := base()
		s.Rows[0].Model = ""
		if _, err := Ingest(s); err == nil || !strings.Contains(err.Error(), "model is required") {
			t.Fatalf("err = %v, want a model-required refusal", err)
		}
	})
	t.Run("missing metric", func(t *testing.T) {
		s := base()
		s.Rows[0].Metric = ""
		if _, err := Ingest(s); err == nil || !strings.Contains(err.Error(), "metric is required") {
			t.Fatalf("err = %v, want a metric-required refusal", err)
		}
	})
	t.Run("missing date", func(t *testing.T) {
		s := base()
		s.Rows[0].AsOf = ""
		if _, err := Ingest(s); err == nil || !strings.Contains(err.Error(), "as_of date is required") {
			t.Fatalf("err = %v, want an as_of-required refusal", err)
		}
	})
	t.Run("unknown confidence label", func(t *testing.T) {
		s := base()
		s.Rows[0].Confidence = SourceConfidence("offical") // typo
		if _, err := Ingest(s); err == nil || !strings.Contains(err.Error(), "not one of") {
			t.Fatalf("err = %v, want an unknown-confidence refusal", err)
		}
	})
}

// TestBenchmarkIngestConfidenceLadder pins the source-confidence trust ordering:
// official > vendor_reported > community > unknown, and an unrecognized label
// is treated as the fail-safe unknown weight (never a panic).
func TestBenchmarkIngestConfidenceLadder(t *testing.T) {
	if !(ConfidenceOfficial.Weight() > ConfidenceVendorReported.Weight() &&
		ConfidenceVendorReported.Weight() > ConfidenceCommunity.Weight() &&
		ConfidenceCommunity.Weight() > ConfidenceUnknown.Weight()) {
		t.Fatalf("confidence ladder not monotonic: official=%v vendor=%v community=%v unknown=%v",
			ConfidenceOfficial.Weight(), ConfidenceVendorReported.Weight(),
			ConfidenceCommunity.Weight(), ConfidenceUnknown.Weight())
	}
	for _, c := range []SourceConfidence{ConfidenceOfficial, ConfidenceVendorReported, ConfidenceCommunity, ConfidenceUnknown} {
		w := c.Weight()
		if w < 0 || w > 1 {
			t.Fatalf("confidence %q weight %v out of [0,1]", c, w)
		}
	}
	if got := SourceConfidence("garbage").Weight(); got != ConfidenceUnknown.Weight() {
		t.Fatalf("unrecognized confidence weight = %v, want fail-safe unknown weight %v", got, ConfidenceUnknown.Weight())
	}
}

// TestBenchmarkIngestRejectsBadSchemaTag guards the on-disk schema tag: a forked
// or missing tag fails loud rather than silently ingesting the wrong shape.
func TestBenchmarkIngestRejectsBadSchemaTag(t *testing.T) {
	_, err := ParseSnapshot([]byte(`{"schema":"wrong","benchmark":"x","version":"1"}`))
	if err == nil || !strings.Contains(err.Error(), "schema") {
		t.Fatalf("err = %v, want a schema-tag refusal", err)
	}
}
