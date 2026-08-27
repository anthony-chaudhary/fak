package studytickets

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studyforge"
	"github.com/anthony-chaudhary/fak/internal/studyprio"
)

func finalForgePath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("FAK_STUDYTICKETS_FORGE")
	if path == "" {
		t.Skip("set FAK_STUDYTICKETS_FORGE to run the complete real-forge witness")
	}
	return path
}

func repoOptions(t *testing.T) BuildOptions {
	return BuildOptions{
		PriorityPath:       "../../docs/research/vllm-priority-2026-08-27/ledger.json",
		JoinPath:           "../../docs/research/vllm-fak-join-2026-08-27/ledger.json",
		ForgePath:          finalForgePath(t),
		AdjacencyPath:      "../../docs/research/inventory/vllm-related-system-adjacency-v1.json",
		ClassificationPath: "../../docs/research/vllm-classification-2026-08-26/index.json",
	}
}

func TestBuildRealClosureLedgerDeterministic(t *testing.T) {
	opts := repoOptions(t)
	first, firstReport, err := Build(opts)
	if err != nil {
		t.Fatal(err)
	}
	second, secondReport, err := Build(opts)
	if err != nil {
		t.Fatal(err)
	}
	firstJSON, _ := MarshalLedger(first)
	secondJSON, _ := MarshalLedger(second)
	if string(firstJSON) != string(secondJSON) {
		t.Fatal("ledger generation is not deterministic")
	}
	firstMD, _ := MarshalReport(firstReport)
	secondMD, _ := MarshalReport(secondReport)
	if string(firstMD) != string(secondMD) {
		t.Fatal("report generation is not deterministic")
	}

	if got, want := first.Coverage.JoinClusters, 193; got != want {
		t.Fatalf("join clusters=%d want %d", got, want)
	}
	if got, want := first.Coverage.ActionableClusters, 183; got != want {
		t.Fatalf("actionable clusters=%d want %d", got, want)
	}
	if got, want := first.Coverage.UncoveredActionable, 5; got != want {
		t.Fatalf("uncovered=%d want %d", got, want)
	}
	if got, want := first.Coverage.MappedSourceClusters, 5; got != want {
		t.Fatalf("mapped=%d want %d", got, want)
	}
	if first.Coverage.SelectedUnmapped != 0 || first.Coverage.Unclassified != 0 || first.Coverage.UnmappedActionable != 0 || first.Coverage.ClosureLeftovers != 0 {
		t.Fatalf("nonzero leftovers: %+v", first.Coverage)
	}
	if first.Coverage.CreatedCount != 2 || first.Coverage.ReusedCount != 0 {
		t.Fatalf("construction counts: %+v", first.Coverage)
	}
	if len(first.Tickets) != 2 || len(first.Queue) != 2 {
		t.Fatalf("tickets=%d queue=%d", len(first.Tickets), len(first.Queue))
	}
	wantIssues := map[string]int{"native-vllm-ir": 9377, "allocator-fragmentation": 9378}
	for _, ticket := range first.Tickets {
		if wantIssues[ticket.CandidateID] != ticket.Issue {
			t.Fatalf("mapping %s -> #%d", ticket.CandidateID, ticket.Issue)
		}
	}
	if first.Queue[0].CandidateID != "native-vllm-ir" || first.Queue[0].Issue != 9377 || first.Queue[1].CandidateID != "allocator-fragmentation" || first.Queue[1].Issue != 9378 {
		t.Fatalf("queue drift: %+v", first.Queue)
	}
	if first.Adjacency.MemberCount != 14 || first.Adjacency.CompleteClassCount != 29 || first.Adjacency.PartialClassCount != 4 || first.Adjacency.InaccessibleCount != 9 || first.Sources.Adjacency.RecordCount != 14 || first.Capture.RecordCount != 9537 || first.Sources.Join.RecordCount != 193 {
		t.Fatalf("source capture accounting drift: adjacency=%+v captures=%+v", first.Adjacency, first.Capture)
	}
}

func TestValidateFilesMatchesCheckedInArtifacts(t *testing.T) {
	opts := repoOptions(t)
	if err := ValidateFiles(ValidateOptions{BuildOptions: opts, LedgerPath: "../../docs/research/vllm-ticket-closure-2026-08-27/ledger.json", ReportPath: "../../docs/research/vllm-ticket-closure-2026-08-27/README.md"}); err != nil {
		t.Fatal(err)
	}
}

func TestBuildRejectsClosedMappedIssue(t *testing.T) {
	candidate, queue, issue := realCandidateIssue(t, "native-vllm-ir", 9377)
	issue.State = "closed"
	assertInvalidContains(t, validateIssue(candidate, queue, issue), "not open")
}

func TestBuildRejectsMissingRequiredSection(t *testing.T) {
	candidate, queue, issue := realCandidateIssue(t, "native-vllm-ir", 9377)
	issue.Body = strings.Replace(issue.Body, "## Witness", "## Missing Witness", 1)
	assertInvalidContains(t, validateIssue(candidate, queue, issue), "Witness section")
}

func TestBuildRejectsMissingStableClusterLink(t *testing.T) {
	candidate, queue, issue := realCandidateIssue(t, "native-vllm-ir", 9377)
	issue.Body = strings.Replace(issue.Body, "architecture_runtime:body:vllm-ir", "architecture_runtime:body:missing", 1)
	assertInvalidContains(t, validateIssue(candidate, queue, issue), "source cluster")
}

func TestValidateRejectsDuplicateIssueReuse(t *testing.T) {
	ledger, _, err := Build(repoOptions(t))
	if err != nil {
		t.Fatal(err)
	}
	ledger.Tickets[1].Issue = ledger.Tickets[0].Issue
	assertInvalidContains(t, Validate(ledger), "ticket issue")
}

func TestValidateRejectsQueueDriftAndDependencyViolation(t *testing.T) {
	t.Run("rank", func(t *testing.T) {
		ledger, _, err := Build(repoOptions(t))
		if err != nil {
			t.Fatal(err)
		}
		ledger.Queue[0], ledger.Queue[1] = ledger.Queue[1], ledger.Queue[0]
		assertInvalidContains(t, Validate(ledger), "rank")
	})
	t.Run("dependency", func(t *testing.T) {
		ledger, _, err := Build(repoOptions(t))
		if err != nil {
			t.Fatal(err)
		}
		ledger.Queue[0].Dependencies = []string{"allocator-fragmentation"}
		assertInvalidContains(t, Validate(ledger), "dependency")
	})
}

func TestValidateFilesRejectsSourceChecksumDrift(t *testing.T) {
	opts := repoOptions(t)
	ledger, report, err := Build(opts)
	if err != nil {
		t.Fatal(err)
	}
	ledgerData, _ := MarshalLedger(ledger)
	reportData, _ := MarshalReport(report)
	ledgerPath := writeBytes(t, "ledger.json", ledgerData)
	reportPath := writeBytes(t, "README.md", reportData)
	forgeCopy, err := os.ReadFile(opts.ForgePath)
	if err != nil {
		t.Fatal(err)
	}
	forgePath := writeBytes(t, "forge.json", forgeCopy)
	opts.ForgePath = forgePath
	if err := ValidateFiles(ValidateOptions{BuildOptions: opts, LedgerPath: ledgerPath, ReportPath: reportPath}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected checksum drift, got %v", err)
	}
}

func realCandidateIssue(t *testing.T, candidateID string, issueNumber int) (studyprio.Candidate, studyprio.QueueEntry, studyforge.Record) {
	t.Helper()
	priority := readPriority(t, repoOptions(t).PriorityPath)
	var candidate studyprio.Candidate
	for _, c := range priority.Candidates {
		if c.ID == candidateID {
			candidate = c
			break
		}
	}
	var queue studyprio.QueueEntry
	for _, q := range priority.Queue {
		if q.CandidateID == candidateID {
			queue = q
			break
		}
	}
	data, err := os.ReadFile(finalForgePath(t))
	if err != nil {
		t.Fatal(err)
	}
	var corpus studyforge.Corpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatal(err)
	}
	issue := *findIssue(t, &corpus, issueNumber)
	return candidate, queue, issue
}
func findIssue(t *testing.T, c *studyforge.Corpus, number int) *studyforge.Record {
	t.Helper()
	for i := range c.Records {
		if c.Records[i].Number == number {
			return &c.Records[i]
		}
	}
	t.Fatalf("issue #%d missing", number)
	return nil
}
func readPriority(t *testing.T, path string) studyprio.Ledger {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var p studyprio.Ledger
	if err := json.Unmarshal(data, &p); err != nil {
		t.Fatal(err)
	}
	return p
}
func writeJSON(t *testing.T, name string, v any) string {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, '\n')
	return writeBytes(t, name, data)
}
func writeBytes(t *testing.T, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
func assertInvalidContains(t *testing.T, err error, want string) {
	t.Helper()
	if !errors.Is(err, ErrInvalid) || !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(want)) {
		t.Fatalf("error=%v want invalid containing %q", err, want)
	}
}
