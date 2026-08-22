package closurehonesty8238

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

type auditArtifact struct {
	Schema   string         `json:"schema"`
	Counts   map[string]int `json:"counts"`
	Totals   auditTotals    `json:"totals"`
	Issues   []auditIssue   `json:"issues"`
	Coverage auditCoverage  `json:"coverage"`
}

type auditTotals struct {
	IssuesAudited int `json:"issues_audited"`
}

type auditCoverage struct {
	Complete         bool   `json:"complete"`
	Verdict          string `json:"verdict"`
	IssuesTruncated  bool   `json:"issues_truncated"`
	CommitsTruncated bool   `json:"commits_truncated"`
	IssuesFetched    int    `json:"issues_fetched"`
	CommitsScanned   int    `json:"commits_scanned"`
	CommitsTotal     int    `json:"commits_total"`
}

type auditIssue struct {
	Number int    `json:"number"`
	Bucket string `json:"bucket"`
}

type reconciliationArtifact struct {
	Schema        string `json:"schema"`
	AuditSnapshot struct {
		CompleteWindowSHA256 string `json:"complete_window_sha256"`
		CoverageComplete     bool   `json:"coverage_complete"`
	} `json:"audit_snapshot"`
	InitialReport struct {
		MappingBasis string `json:"mapping_basis"`
	} `json:"initial_truncated_report"`
	Reconciliation struct {
		Rows                          int            `json:"rows"`
		UniqueIssueNumbers            int            `json:"unique_issue_numbers"`
		ActionRows                    int            `json:"action_rows"`
		ActionRowsWithOpenRepairIssue int            `json:"action_rows_with_open_repair_issue"`
		DispositionCounts             map[string]int `json:"disposition_counts"`
	} `json:"reconciliation"`
	Rows []reconciliationRow `json:"rows"`
}

type reconciliationRow struct {
	Number         int          `json:"number"`
	SourceBucket   string       `json:"source_bucket"`
	Disposition    string       `json:"disposition"`
	ActionRequired bool         `json:"action_required"`
	RepairIssue    *repairIssue `json:"repair_issue"`
}

type repairIssue struct {
	Number int    `json:"number"`
	URL    string `json:"url"`
	State  string `json:"state"`
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

func TestIssue8238ArtifactsCloseEveryAuditRow(t *testing.T) {
	before := readArtifact[auditArtifact](t, "default-window-before.json")
	if before.Schema != "fleet-issue-closure-audit/1" || before.Coverage.Complete {
		t.Fatalf("before receipt schema=%q complete=%t", before.Schema, before.Coverage.Complete)
	}
	if !before.Coverage.IssuesTruncated || !before.Coverage.CommitsTruncated {
		t.Fatalf("before receipt did not capture both truncated windows: %+v", before.Coverage)
	}

	completeRaw, err := os.ReadFile("complete-window.json")
	if err != nil {
		t.Fatal(err)
	}
	var complete auditArtifact
	if err := json.Unmarshal(completeRaw, &complete); err != nil {
		t.Fatal(err)
	}
	if complete.Schema != "fleet-issue-closure-audit/1" || !complete.Coverage.Complete || complete.Coverage.Verdict != "COVERAGE_COMPLETE" {
		t.Fatalf("complete receipt schema=%q coverage=%+v", complete.Schema, complete.Coverage)
	}
	if complete.Coverage.IssuesFetched != 8505 || complete.Coverage.CommitsScanned != 12328 || complete.Coverage.CommitsTotal != 12328 {
		t.Fatalf("complete receipt window changed: %+v", complete.Coverage)
	}
	if complete.Totals.IssuesAudited != len(complete.Issues) {
		t.Fatalf("complete issues total=%d rows=%d", complete.Totals.IssuesAudited, len(complete.Issues))
	}

	reconciliation := readArtifact[reconciliationArtifact](t, "reconciliation.json")
	if reconciliation.Schema != "fleet-issue-closure-reconciliation/1" || !reconciliation.AuditSnapshot.CoverageComplete {
		t.Fatalf("reconciliation schema=%q complete=%t", reconciliation.Schema, reconciliation.AuditSnapshot.CoverageComplete)
	}
	completeSum := sha256.Sum256(completeRaw)
	if got := hex.EncodeToString(completeSum[:]); got != reconciliation.AuditSnapshot.CompleteWindowSHA256 {
		t.Fatalf("complete receipt sha256=%s, ledger=%s", got, reconciliation.AuditSnapshot.CompleteWindowSHA256)
	}
	if reconciliation.InitialReport.MappingBasis != "COMPLETE_EXTANT_REPOSITORY_SUPERSET" {
		t.Fatalf("initial mapping basis=%q", reconciliation.InitialReport.MappingBasis)
	}
	if reconciliation.Reconciliation.Rows != len(complete.Issues) || len(reconciliation.Rows) != len(complete.Issues) {
		t.Fatalf("reconciliation rows declared=%d ledger=%d audit=%d", reconciliation.Reconciliation.Rows, len(reconciliation.Rows), len(complete.Issues))
	}

	auditRows := make(map[int]string, len(complete.Issues))
	for _, row := range complete.Issues {
		if _, exists := auditRows[row.Number]; exists {
			t.Fatalf("duplicate audit issue #%d", row.Number)
		}
		auditRows[row.Number] = row.Bucket
	}
	dispositions := make(map[string]int)
	actions := 0
	for _, row := range reconciliation.Rows {
		bucket, exists := auditRows[row.Number]
		if !exists {
			t.Fatalf("ledger issue #%d absent from complete audit", row.Number)
		}
		if bucket != row.SourceBucket {
			t.Fatalf("issue #%d bucket audit=%s ledger=%s", row.Number, bucket, row.SourceBucket)
		}
		delete(auditRows, row.Number)
		dispositions[row.Disposition]++
		if !row.ActionRequired {
			continue
		}
		actions++
		if row.RepairIssue == nil || row.RepairIssue.State != "OPEN" || row.RepairIssue.Number == 0 || row.RepairIssue.URL == "" {
			t.Fatalf("action issue #%d lacks an open repair issue: %+v", row.Number, row.RepairIssue)
		}
	}
	if len(auditRows) != 0 {
		t.Fatalf("%d complete-audit rows disappeared from reconciliation", len(auditRows))
	}
	if actions != 56 || reconciliation.Reconciliation.ActionRows != actions || reconciliation.Reconciliation.ActionRowsWithOpenRepairIssue != actions {
		t.Fatalf("action rows calculated=%d declared=%d open-repair=%d", actions, reconciliation.Reconciliation.ActionRows, reconciliation.Reconciliation.ActionRowsWithOpenRepairIssue)
	}
	if len(dispositions) != len(reconciliation.Reconciliation.DispositionCounts) {
		t.Fatalf("disposition kinds calculated=%d declared=%d", len(dispositions), len(reconciliation.Reconciliation.DispositionCounts))
	}
	for disposition, count := range dispositions {
		if reconciliation.Reconciliation.DispositionCounts[disposition] != count {
			t.Fatalf("disposition %s calculated=%d declared=%d", disposition, count, reconciliation.Reconciliation.DispositionCounts[disposition])
		}
	}
}
