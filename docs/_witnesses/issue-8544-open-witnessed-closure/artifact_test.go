package openwitnessed8544

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type sourceAudit struct {
	Issues []struct {
		Number int    `json:"number"`
		Bucket string `json:"bucket"`
	} `json:"issues"`
}

type beforeArtifact struct {
	Schema string `json:"schema"`
	Issue  int    `json:"issue"`
	DryRun struct {
		Verdict         string         `json:"verdict"`
		Live            bool           `json:"live"`
		CandidatesTotal int            `json:"candidates_total"`
		PlannedCount    int            `json:"planned_count"`
		Counts          map[string]int `json:"counts"`
		ClosedNumbers   []int          `json:"closed_numbers"`
	} `json:"dry_run"`
	Rows []beforeRow `json:"rows"`
}

type beforeRow struct {
	Number            int    `json:"number"`
	RecordedSHA       string `json:"recorded_sha"`
	DOSVerdict        string `json:"dos_verdict"`
	DOSWitness        string `json:"dos_witness"`
	GitHubStateBefore string `json:"github_state_before"`
	Action            string `json:"action"`
	TypedReason       string `json:"typed_reason"`
}

type reconciliationArtifact struct {
	Schema       string `json:"schema"`
	Issue        int    `json:"issue"`
	LiveCloseArm struct {
		Verdict       string         `json:"verdict"`
		Live          bool           `json:"live"`
		PushedGate    string         `json:"pushed_gate"`
		Counts        map[string]int `json:"counts"`
		ClosedNumbers []int          `json:"closed_numbers"`
	} `json:"live_close_arm"`
	Reconciliation struct {
		Rows                int            `json:"rows"`
		UniqueIssueNumbers  int            `json:"unique_issue_numbers"`
		AncestryChecked     int            `json:"ancestry_checked"`
		AncestryConfirmed   int            `json:"ancestry_confirmed"`
		GitHubStateChecked  int            `json:"github_state_checked"`
		ClosedWitnessed     int            `json:"closed_witnessed"`
		OpenTypedExceptions int            `json:"open_typed_exceptions"`
		DispositionCounts   map[string]int `json:"disposition_counts"`
	} `json:"reconciliation"`
	Rows []reconciliationRow `json:"rows"`
}

type reconciliationRow struct {
	Number             int    `json:"number"`
	RecordedSHA        string `json:"recorded_sha"`
	OriginMainAncestor bool   `json:"origin_main_ancestor"`
	DOSVerdict         string `json:"dos_verdict"`
	DOSWitness         string `json:"dos_witness"`
	GitHubState        string `json:"github_state"`
	Disposition        string `json:"disposition"`
	TypedReason        string `json:"typed_reason"`
}

func readArtifact[T any](t *testing.T, path string) T {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var artifact T
	if err := json.Unmarshal(raw, &artifact); err != nil {
		t.Fatal(err)
	}
	return artifact
}

func TestIssue8544ReconcilesEveryOpenWitnessedRow(t *testing.T) {
	source := readArtifact[sourceAudit](t, "../issue-8238-closure-honesty/complete-window.json")
	before := readArtifact[beforeArtifact](t, "failing-before.json")
	final := readArtifact[reconciliationArtifact](t, "reconciliation.json")

	sourceRows := make(map[int]struct{})
	for _, row := range source.Issues {
		if row.Bucket == "OPEN_WITNESSED" {
			sourceRows[row.Number] = struct{}{}
		}
	}
	if len(sourceRows) != 54 {
		t.Fatalf("source OPEN_WITNESSED rows=%d, want 54", len(sourceRows))
	}
	if before.Schema != "fak-issue-8544-failing-before/1" || before.Issue != 8544 {
		t.Fatalf("before schema=%q issue=%d", before.Schema, before.Issue)
	}
	if before.DryRun.Live || before.DryRun.Verdict != "PLANNED" || before.DryRun.CandidatesTotal != 54 || before.DryRun.PlannedCount != 54 {
		t.Fatalf("before dry run=%+v", before.DryRun)
	}
	if before.DryRun.Counts["would_close"] != 0 || len(before.DryRun.ClosedNumbers) != 0 {
		t.Fatalf("before unexpectedly closeable: counts=%v closed=%v", before.DryRun.Counts, before.DryRun.ClosedNumbers)
	}
	expectedSkips := map[string]int{
		"skipped_reopened":            15,
		"skipped_unwitnessed":         1,
		"skipped_incomplete_evidence": 32,
		"skipped_partial":             2,
		"skipped_nonresolving":        4,
	}
	for key, want := range expectedSkips {
		if got := before.DryRun.Counts[key]; got != want {
			t.Fatalf("before %s=%d, want %d", key, got, want)
		}
	}

	beforeRows := make(map[int]beforeRow, len(before.Rows))
	for _, row := range before.Rows {
		if _, ok := sourceRows[row.Number]; !ok {
			t.Fatalf("before issue #%d absent from source cohort", row.Number)
		}
		if _, duplicate := beforeRows[row.Number]; duplicate {
			t.Fatalf("duplicate before issue #%d", row.Number)
		}
		if row.GitHubStateBefore != "OPEN" || row.TypedReason == "" {
			t.Fatalf("before issue #%d state=%q reason=%q", row.Number, row.GitHubStateBefore, row.TypedReason)
		}
		beforeRows[row.Number] = row
	}
	if len(beforeRows) != len(sourceRows) {
		t.Fatalf("before rows=%d source=%d", len(beforeRows), len(sourceRows))
	}

	if final.Schema != "fak-issue-8544-open-witnessed-reconciliation/1" || final.Issue != 8544 {
		t.Fatalf("final schema=%q issue=%d", final.Schema, final.Issue)
	}
	if !final.LiveCloseArm.Live || final.LiveCloseArm.Verdict != "NO_CLOSES" || final.LiveCloseArm.PushedGate != "active" {
		t.Fatalf("live close arm=%+v", final.LiveCloseArm)
	}
	if final.LiveCloseArm.Counts["failed"] != 0 || len(final.LiveCloseArm.ClosedNumbers) != 0 {
		t.Fatalf("live close failures=%d closed=%v", final.LiveCloseArm.Counts["failed"], final.LiveCloseArm.ClosedNumbers)
	}
	meta := final.Reconciliation
	if meta.Rows != 54 || meta.UniqueIssueNumbers != 54 || meta.AncestryChecked != 54 || meta.AncestryConfirmed != 53 || meta.GitHubStateChecked != 54 || meta.ClosedWitnessed != 0 || meta.OpenTypedExceptions != 54 {
		t.Fatalf("reconciliation summary=%+v", meta)
	}
	expectedDispositions := map[string]int{
		"OPEN_TYPED_REOPENED":            15,
		"OPEN_TYPED_UNWITNESSED":         1,
		"OPEN_TYPED_INCOMPLETE_EVIDENCE": 32,
		"OPEN_TYPED_PARTIAL":             2,
		"OPEN_TYPED_NONRESOLVING":        4,
	}
	if len(meta.DispositionCounts) != len(expectedDispositions) {
		t.Fatalf("disposition kinds=%v", meta.DispositionCounts)
	}
	for disposition, want := range expectedDispositions {
		if got := meta.DispositionCounts[disposition]; got != want {
			t.Fatalf("disposition %s=%d, want %d", disposition, got, want)
		}
	}

	seen := make(map[int]struct{}, len(final.Rows))
	for _, row := range final.Rows {
		beforeRow, ok := beforeRows[row.Number]
		if !ok {
			t.Fatalf("final issue #%d absent from before receipt", row.Number)
		}
		if _, duplicate := seen[row.Number]; duplicate {
			t.Fatalf("duplicate final issue #%d", row.Number)
		}
		seen[row.Number] = struct{}{}
		if row.RecordedSHA != beforeRow.RecordedSHA || row.DOSVerdict != beforeRow.DOSVerdict || row.DOSWitness != beforeRow.DOSWitness {
			t.Fatalf("issue #%d before/final witness drift", row.Number)
		}
		if row.GitHubState != "OPEN" || !strings.HasPrefix(row.Disposition, "OPEN_TYPED_") || row.TypedReason == "" {
			t.Fatalf("issue #%d state=%q disposition=%q reason=%q", row.Number, row.GitHubState, row.Disposition, row.TypedReason)
		}
		if row.Number == 6349 {
			if row.OriginMainAncestor || row.DOSVerdict != "CLAIM_UNWITNESSED" || row.DOSWitness != "subject-only" {
				t.Fatalf("issue #6349 exception=%+v", row)
			}
			continue
		}
		if !row.OriginMainAncestor || row.DOSVerdict != "OK" || row.DOSWitness != "diff-witnessed" {
			t.Fatalf("issue #%d ancestry/DOS=%+v", row.Number, row)
		}
	}
	if len(seen) != len(sourceRows) {
		t.Fatalf("final rows=%d source=%d", len(seen), len(sourceRows))
	}
}
