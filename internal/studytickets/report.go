package studytickets

import (
	"bytes"
	"fmt"
	"sort"
)

func MarshalReport(report Report) ([]byte, error) {
	if report.Schema != ReportSchema || report.LedgerSHA256 == "" {
		return nil, invalidf("report receipt incomplete")
	}
	ledger := report.Detail
	var b bytes.Buffer
	fmt.Fprintln(&b, "# vLLM ticket closure")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Machine ledger: `ledger.json` (`%s`). Parent study: #%d.\n\n", report.LedgerSHA256, ledger.ParentIssue)
	fmt.Fprintln(&b, "## Verdict")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- **Complete:** %d join clusters have dispositions; %d uncovered actionable clusters map through %d priority candidates to %d open captured tickets; closure leftovers are %d.\n",
		ledger.Coverage.JoinClusters, ledger.Coverage.UncoveredActionable, ledger.Coverage.QueueSelections, len(ledger.Tickets), ledger.Coverage.ClosureLeftovers)
	fmt.Fprintf(&b, "- **Ticket accounting:** created=%d, reused=%d. Here “created” means constructed specifically for these candidates before this offline build; the build itself performs no GitHub mutation.\n",
		ledger.Coverage.CreatedCount, ledger.Coverage.ReusedCount)
	fmt.Fprintln(&b, "- **Honesty:** landed/open/partial/conflict/obsolete source dispositions remain separate counts; only the five `uncovered` clusters enter the two new tickets.")
	fmt.Fprintln(&b)

	fmt.Fprintln(&b, "## Corpus coverage")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- vLLM: **%s records** at `%s`, cutoff `%s` (index `%s`).\n", formatCount(ledger.CorpusReceipt.VLLMRecords), ledger.CorpusReceipt.VLLMRevision, ledger.CorpusReceipt.VLLMCutoff, ledger.CorpusReceipt.VLLMIndexChecksum)
	fmt.Fprintf(&b, "- Related systems: **%d/%d forge captures complete**, totaling **%s records**. Runtime-tree partial and inaccessible classes remain explicit below.\n", ledger.CorpusReceipt.RelatedForgeComplete, ledger.CorpusReceipt.RelatedRepositories, formatCount(ledger.CorpusReceipt.RelatedRecords))
	fmt.Fprintf(&b, "- Final FAK ticket corpus: **%s records**, including open issues #9377 and #9378.\n\n", formatCount(ledger.CorpusReceipt.FAKRecords))

	fmt.Fprintln(&b, "## Source receipts")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| source | schema | revision / cutoff | count | checksum |")
	fmt.Fprintln(&b, "|---|---|---|---:|---|")
	for _, row := range []SourceReceipt{ledger.Sources.Priority, ledger.Sources.Join, ledger.Sources.Forge, ledger.Sources.Adjacency} {
		rev := row.Revision
		if row.Cutoff != "" {
			if rev != "" {
				rev += "<br>"
			}
			rev += row.Cutoff
		}
		fmt.Fprintf(&b, "| `%s` | `%s` | %s | %d | `%s` |\n", row.Path, row.Schema, rev, row.RecordCount, row.SHA256)
	}
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Capture receipt: repository `%s`, revision `%s`, cutoff `%s`, status `%s`, records=%d, sources=%v, receipt `%s`, index `%s`.\n\n",
		ledger.Capture.Repository, ledger.Capture.Revision, ledger.Capture.Cutoff, ledger.Capture.Status,
		ledger.Capture.RecordCount, ledger.Capture.CompleteSources, ledger.Capture.ReceiptSHA256, ledger.Capture.IndexChecksum)

	fmt.Fprintln(&b, "## Closure counts")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| disposition | total | actionable |")
	fmt.Fprintln(&b, "|---|---:|---:|")
	for _, count := range ledger.Coverage.DispositionCounts {
		fmt.Fprintf(&b, "| `%s` | %d | %d |\n", count.Disposition, count.Count, count.Actionable)
	}
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Selected/unmapped=%d; unclassified=%d; unmapped actionable=%d; closure leftovers=%d.\n\n",
		ledger.Coverage.SelectedUnmapped, ledger.Coverage.Unclassified, ledger.Coverage.UnmappedActionable, ledger.Coverage.ClosureLeftovers)

	fmt.Fprintln(&b, "## Ticket mapping and queue")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| rank | candidate | issue | horizon | source clusters |")
	fmt.Fprintln(&b, "|---:|---|---|---|---|")
	ticketBySelection := map[string]Ticket{}
	for _, ticket := range ledger.Tickets {
		ticketBySelection[ticket.CandidateID] = ticket
	}
	for _, entry := range ledger.Queue {
		ticket := ticketBySelection[entry.CandidateID]
		var clusters []string
		for _, source := range ticket.SourceClusters {
			clusters = append(clusters, "`"+source.ClusterID+"`")
		}
		fmt.Fprintf(&b, "| %d | `%s` | [#%d](%s) | `%s` | %s |\n", entry.Rank, entry.CandidateID, entry.Issue, ticket.URL, entry.Horizon, joinComma(clusters))
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "Each issue is captured once, open at the forge cutoff, carries the parent and stable-cluster links, required issue sections, native engine/model constraint, expected horizon labels, and a unique queue slot. Dependencies are empty for both candidates.")
	fmt.Fprintln(&b)

	fmt.Fprintln(&b, "## Adjacency coverage")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Adjacency `%s` has %d members and %d complete, %d partial, and %d inaccessible/missing source-class receipts.\n\n",
		ledger.Adjacency.ID, ledger.Adjacency.MemberCount, ledger.Adjacency.CompleteClassCount, ledger.Adjacency.PartialClassCount, ledger.Adjacency.InaccessibleCount)
	renderAdjacencyClasses(&b, "Partial classes", ledger.Adjacency.PartialClasses)
	renderAdjacencyClasses(&b, "Inaccessible / missing classes", ledger.Adjacency.InaccessibleClasses)

	fmt.Fprintln(&b, "## Sampling evidence")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| cluster | disposition | actionable | artifacts | confidence | evidence |")
	fmt.Fprintln(&b, "|---|---|---|---:|---|---|")
	for _, sample := range ledger.SamplingEvidence {
		fmt.Fprintf(&b, "| `%s` | `%s` | %t | %d | `%s` | `%s` |\n",
			sample.ClusterID, sample.Disposition, sample.Actionable, sample.ArtifactCount, sample.Confidence, sample.EvidenceSHA256)
	}
	fmt.Fprintln(&b)

	fmt.Fprintln(&b, "## Refresh obligations")
	fmt.Fprintln(&b)
	for _, obligation := range ledger.RefreshObligations {
		fmt.Fprintf(&b, "- %s\n", obligation)
	}
	return b.Bytes(), nil
}

func renderAdjacencyClasses(b *bytes.Buffer, title string, rows []AdjacencyClass) {
	fmt.Fprintf(b, "### %s\n\n", title)
	if len(rows) == 0 {
		fmt.Fprintln(b, "None.")
		fmt.Fprintln(b)
		return
	}
	rows = append([]AdjacencyClass(nil), rows...)
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Repository+rows[i].Class < rows[j].Repository+rows[j].Class
	})
	fmt.Fprintln(b, "| repository | class | status | note |")
	fmt.Fprintln(b, "|---|---|---|---|")
	for _, row := range rows {
		fmt.Fprintf(b, "| `%s` | `%s` | `%s` | %s |\n", row.Repository, row.Class, row.Status, row.Notes)
	}
	fmt.Fprintln(b)
}

func formatCount(value int) string {
	digits := fmt.Sprintf("%d", value)
	for index := len(digits) - 3; index > 0; index -= 3 {
		digits = digits[:index] + "," + digits[index:]
	}
	return digits
}

func joinComma(values []string) string {
	var out string
	for i, value := range values {
		if i > 0 {
			out += ", "
		}
		out += value
	}
	return out
}
