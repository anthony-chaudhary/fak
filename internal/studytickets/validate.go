package studytickets

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

func Validate(ledger Ledger) error {
	if ledger.Schema != Schema || ledger.ParentIssue != ParentIssue {
		return invalidf("schema or parent issue mismatch")
	}
	if ledger.Coverage.JoinClusters != 193 || ledger.Coverage.UncoveredActionable != 5 ||
		ledger.Coverage.QueueSelections != 2 || ledger.Coverage.MappedSourceClusters != 5 ||
		ledger.Coverage.CreatedCount != 2 || ledger.Coverage.ReusedCount != 0 ||
		ledger.Coverage.ClosureLeftovers != 0 || ledger.Coverage.SelectedUnmapped != 0 ||
		ledger.Coverage.Unclassified != 0 || ledger.Coverage.UnmappedActionable != 0 {
		return invalidf("closure counts are not complete")
	}
	wantCounts := map[string]int{"landed": 4, "open_exact": 2, "partial": 168, "conflict": 13, "obsolete": 1, "uncovered": 5}
	if len(ledger.Coverage.DispositionCounts) != len(wantCounts) {
		return invalidf("disposition vocabulary drift")
	}
	for _, count := range ledger.Coverage.DispositionCounts {
		if want, ok := wantCounts[count.Disposition]; !ok || count.Count != want {
			return invalidf("disposition count drift for %s", count.Disposition)
		}
	}
	if len(ledger.Tickets) != 2 || len(ledger.Queue) != 2 {
		return invalidf("closure must contain two tickets and two queue entries")
	}
	if err := validateDependencyOrder(ledger.Queue); err != nil {
		return err
	}
	ticketBySelection := map[string]Ticket{}
	issueSeen := map[int]bool{}
	for _, ticket := range ledger.Tickets {
		if ticket.CandidateID == "" || ticketBySelection[ticket.CandidateID].CandidateID != "" {
			return invalidf("duplicate or empty ticket candidate")
		}
		if ticket.Issue <= 0 || issueSeen[ticket.Issue] {
			return invalidf("duplicate or empty ticket issue")
		}
		issueSeen[ticket.Issue] = true
		if issueMapping[ticket.CandidateID] != ticket.Issue || ticket.State != "open" ||
			!ticket.PurposeBuilt || ticket.ReusedExistingWork || ticket.URL == "" ||
			ticket.RecordSHA256 == "" || len(ticket.SourceClusters) == 0 {
			return invalidf("ticket %s receipt incomplete", ticket.CandidateID)
		}
		ticketBySelection[ticket.CandidateID] = ticket
	}
	for _, entry := range ledger.Queue {
		ticket := ticketBySelection[entry.CandidateID]
		if ticket.Issue != entry.Issue || ticket.QueueRank != entry.Rank || ticket.Horizon != entry.Horizon ||
			!equalStrings(ticket.Dependencies, entry.Dependencies) {
			return invalidf("queue drift for %s", entry.CandidateID)
		}
	}
	if ledger.Capture.Status != "complete" || ledger.Capture.Repository != "anthony-chaudhary/fak" ||
		ledger.Capture.RecordCount <= 0 || len(ledger.Capture.CompleteSources) != 6 {
		return invalidf("capture receipt incomplete")
	}
	if ledger.CorpusReceipt.VLLMRecords != 53848 || ledger.CorpusReceipt.RelatedRepositories != 14 ||
		ledger.CorpusReceipt.RelatedForgeComplete != 14 || ledger.CorpusReceipt.RelatedRecords != 121063 ||
		ledger.CorpusReceipt.FAKRecords != ledger.Capture.RecordCount || ledger.CorpusReceipt.VLLMRevision == "" ||
		ledger.CorpusReceipt.VLLMCutoff == "" || ledger.CorpusReceipt.VLLMIndexChecksum == "" {
		return invalidf("corpus coverage receipt incomplete")
	}
	if ledger.Adjacency.ID == "" || ledger.Adjacency.MemberCount != 14 ||
		ledger.Adjacency.CompleteClassCount+ledger.Adjacency.PartialClassCount+ledger.Adjacency.InaccessibleCount == 0 {
		return invalidf("adjacency receipt incomplete")
	}
	if len(ledger.SamplingEvidence) != len(sampleClusterIDs) || len(ledger.RefreshObligations) == 0 {
		return invalidf("sampling or refresh obligations missing")
	}
	for _, source := range []SourceReceipt{ledger.Sources.Priority, ledger.Sources.Join, ledger.Sources.Forge, ledger.Sources.Adjacency} {
		if source.Path == "" || source.SHA256 == "" || source.Schema == "" {
			return invalidf("source receipt incomplete")
		}
	}
	return nil
}

func ValidateFiles(opts ValidateOptions) error {
	ledger, report, err := Build(opts.BuildOptions)
	if err != nil {
		return err
	}
	wantLedger, err := MarshalLedger(ledger)
	if err != nil {
		return err
	}
	wantReport, err := MarshalReport(report)
	if err != nil {
		return err
	}
	if err := compareFile(opts.LedgerPath, wantLedger); err != nil {
		return err
	}
	if err := compareFile(opts.ReportPath, wantReport); err != nil {
		return err
	}
	return nil
}

func compareFile(path string, want []byte) error {
	got, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("studytickets: read generated artifact %s: %w", path, err)
	}
	if !bytes.Equal(got, want) {
		return invalidf("generated artifact drift: %s", path)
	}
	return nil
}

func MarshalLedger(ledger Ledger) ([]byte, error) {
	data, err := json.MarshalIndent(ledger, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}
