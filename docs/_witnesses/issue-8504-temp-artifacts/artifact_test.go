package issue8504witness

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type item struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	Reason string `json:"reason"`
}

type summary struct {
	MatchingCount  int   `json:"matching_count"`
	MatchingBytes  int64 `json:"matching_bytes"`
	EligibleCount  int   `json:"eligible_count"`
	EligibleBytes  int64 `json:"eligible_bytes"`
	PreservedCount int   `json:"preserved_count"`
	PreservedBytes int64 `json:"preserved_bytes"`
	ReapedCount    int   `json:"reaped_count"`
	ReapedBytes    int64 `json:"reaped_bytes"`
}

type run struct {
	ExitCode int `json:"exit_code"`
	Receipt  struct {
		Schema         string   `json:"schema"`
		Mode           string   `json:"mode"`
		Root           string   `json:"root"`
		MinAgeSeconds  int64    `json:"min_age_seconds"`
		Inspection     string   `json:"inspection"`
		EligibleItems  []item   `json:"eligible_items"`
		ReapedItems    []item   `json:"reaped_items"`
		PreservedItems []item   `json:"preserved_items"`
		Summary        summary  `json:"summary"`
		Warnings       []string `json:"warnings"`
	} `json:"receipt"`
}

type witness struct {
	Schema        string `json:"schema"`
	FailingBefore run    `json:"failing_before"`
	Apply         run    `json:"apply"`
	PostApply     run    `json:"post_apply"`
	DefectLedger  struct {
		SurfacedCount   int    `json:"surfaced_count"`
		MarkerKeyPrefix string `json:"marker_key_prefix"`
		FiledIssues     []int  `json:"filed_issues"`
	} `json:"defect_ledger"`
}

func TestLiveDogfoodReceiptClosesTheObservedBacklog(t *testing.T) {
	raw, err := os.ReadFile("receipt.json")
	if err != nil {
		t.Fatal(err)
	}
	var got witness
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-issue-8504-dogfood/1" {
		t.Fatalf("schema = %q", got.Schema)
	}
	assertHeader(t, got.FailingBefore, "preview")
	assertHeader(t, got.Apply, "apply")
	assertHeader(t, got.PostApply, "preview")

	before := got.FailingBefore.Receipt.Summary
	applied := got.Apply.Receipt.Summary
	after := got.PostApply.Receipt.Summary
	if before.EligibleCount != 8 || before.EligibleBytes != 854104623 {
		t.Fatalf("failing-before backlog drifted: %+v", before)
	}
	if applied.EligibleCount != before.EligibleCount || applied.EligibleBytes != before.EligibleBytes || applied.ReapedCount != before.EligibleCount || applied.ReapedBytes != before.EligibleBytes {
		t.Fatalf("apply did not reap the complete observed backlog: before=%+v apply=%+v", before, applied)
	}
	if after.EligibleCount != 0 || after.EligibleBytes != 0 || after.MatchingCount != applied.PreservedCount || after.MatchingBytes != applied.PreservedBytes {
		t.Fatalf("post-apply preview does not preserve only the fresh set: apply=%+v after=%+v", applied, after)
	}

	if len(got.FailingBefore.Receipt.EligibleItems) != before.EligibleCount || len(got.Apply.Receipt.ReapedItems) != applied.ReapedCount || len(got.Apply.Receipt.PreservedItems) != applied.PreservedCount {
		t.Fatalf("item counts do not bind summaries: before=%d reaped=%d preserved=%d", len(got.FailingBefore.Receipt.EligibleItems), len(got.Apply.Receipt.ReapedItems), len(got.Apply.Receipt.PreservedItems))
	}
	assertItems(t, got.FailingBefore.Receipt.EligibleItems, "eligible", before.EligibleBytes)
	assertItems(t, got.Apply.Receipt.ReapedItems, "reaped", applied.ReapedBytes)
	assertItems(t, got.Apply.Receipt.PreservedItems, "fresh", applied.PreservedBytes)

	if got.DefectLedger.SurfacedCount != len(got.DefectLedger.FiledIssues) || got.DefectLedger.SurfacedCount != 0 || got.DefectLedger.MarkerKeyPrefix == "" {
		t.Fatalf("defect ledger is incomplete: %+v", got.DefectLedger)
	}
	lower := strings.ToLower(string(raw))
	for _, forbidden := range []string{"appdata", "commandline", "executablepath"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("public receipt contains private field %q", forbidden)
		}
	}
}

func assertHeader(t *testing.T, got run, mode string) {
	t.Helper()
	if got.ExitCode != 0 || got.Receipt.Schema != "fak-temp-artifacts/1" || got.Receipt.Mode != mode || got.Receipt.Root != "<OS_TEMP>" || got.Receipt.MinAgeSeconds != 86400 || got.Receipt.Inspection != "complete" || len(got.Receipt.Warnings) != 0 {
		t.Fatalf("%s receipt header is not witnessed: %+v", mode, got)
	}
}

func assertItems(t *testing.T, items []item, reason string, wantBytes int64) {
	t.Helper()
	var bytes int64
	seen := map[string]bool{}
	for _, got := range items {
		if filepath.Base(got.Name) != got.Name || !strings.HasPrefix(got.Name, "fak-") || got.Reason != reason || got.Bytes <= 0 || seen[strings.ToLower(got.Name)] {
			t.Fatalf("invalid %s item: %+v", reason, got)
		}
		seen[strings.ToLower(got.Name)] = true
		bytes += got.Bytes
	}
	if bytes != wantBytes {
		t.Fatalf("%s bytes = %d, want %d", reason, bytes, wantBytes)
	}
}
