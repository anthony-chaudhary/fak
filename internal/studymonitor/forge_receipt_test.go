package studymonitor

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInventoryReportAcceptsValidatedStudyForgeReceipt(t *testing.T) {
	dir := t.TempDir()
	mapPath := writeInventoryMapForForgeTest(t, dir)
	receiptPath := writeStudyForgeReceiptForTest(t, dir, validStudyForgeReceiptForTest())
	registry := forgeInventoryRegistryForTest(mapPath, receiptPath)

	report := BuildInventoryReportWithMapFiles(registry, dir)
	if !report.OK {
		t.Fatalf("report = %+v, want validated receipt to satisfy forge source class", report)
	}
	row := report.Repositories[0]
	if row.ForgeReceiptPath != filepath.Base(receiptPath) {
		t.Fatalf("forge_receipt_path = %q", row.ForgeReceiptPath)
	}
	for _, evidence := range row.SourceEvidence {
		if evidence.Class == "open_closed_issues_prs_discussions" {
			t.Fatal("test accidentally satisfied forge class with free-form source_evidence")
		}
	}
}

func TestDecodeStudyForgeReceiptAcceptsStandaloneReceipt(t *testing.T) {
	want := validStudyForgeReceiptForTest()
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeStudyForgeReceiptEvidence(data)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != want.Schema || got.Repository != want.Repository || len(got.Sources) != len(want.Sources) {
		t.Fatalf("decoded receipt = %+v", got)
	}
}

func TestInventoryReportRejectsPartialOrInconsistentStudyForgeReceipt(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*StudyForgeReceiptEvidence)
		want   string
	}{
		{
			name: "partial top level",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Status = "partial"
			},
			want: `status must be complete, got "partial"`,
		},
		{
			name: "missing discussions",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Sources = append(receipt.Sources[:2], receipt.Sources[3:]...)
			},
			want: "missing source discussions",
		},
		{
			name: "duplicate identity count",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Sources[0].UniqueCount--
			},
			want: "source issues normalized_count does not match unique_count",
		},
		{
			name: "partial source",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Sources[1].Status = "partial"
			},
			want: "source pulls status must be complete",
		},
		{
			name: "revision mismatch",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Revision = "def"
			},
			want: "revision does not match checked_revision",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			mapPath := writeInventoryMapForForgeTest(t, dir)
			receipt := validStudyForgeReceiptForTest()
			tt.mutate(&receipt)
			receiptPath := writeStudyForgeReceiptForTest(t, dir, receipt)

			report := BuildInventoryReportWithMapFiles(forgeInventoryRegistryForTest(mapPath, receiptPath), dir)
			if report.OK {
				t.Fatalf("report = %+v, want blocker", report)
			}
			reasons := strings.Join(report.Repositories[0].Reasons, "\n")
			if !strings.Contains(reasons, "studyforge receipt is invalid: "+tt.want) {
				t.Fatalf("reasons = %q, want %q", reasons, tt.want)
			}
			if !strings.Contains(reasons, "missing source classes: open_closed_issues_prs_discussions") {
				t.Fatalf("invalid receipt must not satisfy forge source class: %q", reasons)
			}
		})
	}
}

func validStudyForgeReceiptForTest() StudyForgeReceiptEvidence {
	sources := make([]StudyForgeSourceReceiptEvidence, 0, len(requiredStudyForgeSources))
	for _, name := range requiredStudyForgeSources {
		sources = append(sources, StudyForgeSourceReceiptEvidence{
			Name:            name,
			Status:          "complete",
			Pages:           []json.RawMessage{json.RawMessage(`{"number":1}`)},
			FetchedCount:    2,
			NormalizedCount: 2,
			UniqueCount:     2,
			Checksum:        "sha256:" + name,
		})
	}
	// The issues census legitimately fetches mixed pull-request rows and then
	// partitions them out of its normalized issue-only index.
	sources[0].FetchedCount = 3
	return StudyForgeReceiptEvidence{
		Schema:        StudyForgeReceiptSchema,
		Repository:    "owner/repo",
		Revision:      "abc",
		Cutoff:        "2026-08-26T22:00:00Z",
		Status:        "complete",
		Sources:       sources,
		IndexChecksum: "sha256:index",
	}
}

func writeStudyForgeReceiptForTest(t *testing.T, dir string, receipt StudyForgeReceiptEvidence) string {
	t.Helper()
	// Capture writes the normalized corpus envelope; the monitor also accepts a
	// standalone receipt for content-addressed receipt-only artifacts.
	envelope := struct {
		Schema  string                    `json:"schema"`
		Receipt StudyForgeReceiptEvidence `json:"receipt"`
		Records []any                     `json:"records"`
	}{Schema: studyForgeCorpusSchema, Receipt: receipt, Records: []any{}}
	data, err := json.MarshalIndent(envelope, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "forge-receipt.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeInventoryMapForForgeTest(t *testing.T, dir string) string {
	t.Helper()
	var data bytes.Buffer
	if err := WriteInventoryMapJSON(&data, validInventoryMapForTest()); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "inventory-map.json")
	if err := os.WriteFile(path, data.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func forgeInventoryRegistryForTest(mapPath, receiptPath string) Registry {
	return Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{{
		Repository:      "owner/repo",
		URL:             "https://example/repo",
		Status:          "candidate",
		Priority:        1,
		Why:             "fresh",
		LastChecked:     "2026-08-26",
		CheckedRevision: "abc",
		Inventory: &InventoryContract{
			MapPath:          filepath.Base(mapPath),
			ForgeReceiptPath: filepath.Base(receiptPath),
			IndexedRevision:  "abc",
			SourceClasses:    append([]string(nil), RequiredInventorySourceClasses...),
			SourceEvidence: []InventorySourceEvidence{
				{Class: "fak_selfquery_witness", Evidence: []string{"fak capabilities owner/repo"}},
				{Class: "candidate_matrix", Evidence: []string{"docs/notes/STUDY.md#candidate-matrix"}},
				{Class: "issue_tracking", Evidence: []string{"#123"}},
			},
			SubsystemCount:     1,
			CompletenessCritic: "local map and external receipts reconciled",
		},
	}}}
}
