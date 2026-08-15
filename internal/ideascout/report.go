package ideascout

// Reporting and fixture input: reading candidate/issue corpora off disk for a
// replay, and rendering a run record for a human reader.

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

func ReadCandidates(path string) ([]Candidate, error) {
	var out []Candidate
	if err := readJSONFile(path, &out); err == nil {
		return out, nil
	}
	var wrapped struct {
		Candidates []Candidate `json:"candidates"`
	}
	if err := readJSONFile(path, &wrapped); err != nil {
		return nil, err
	}
	return wrapped.Candidates, nil
}

func ReadExistingIssues(path string) ([]ExistingIssue, error) {
	var out []ExistingIssue
	if err := readJSONFile(path, &out); err == nil {
		return out, nil
	}
	var wrapped struct {
		Issues []ExistingIssue `json:"issues"`
	}
	if err := readJSONFile(path, &wrapped); err != nil {
		return nil, err
	}
	return wrapped.Issues, nil
}

func readJSONFile(path string, out any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, out)
}

func RenderHuman(w io.Writer, result RunResult, cfg Config) {
	fmt.Fprintf(w, "idea-scout %s - %s\n", result.Date, result.Mode)
	fmt.Fprintf(w, "  gathered %d candidates from %d topics x (%s)\n", result.CandidatesGathered, result.Topics, sourceDisplayList())
	fmt.Fprintf(w, "  dedup index: %d source stamps from %d filed issue(s) (label-targeted, complete) + %d recent issue(s) for the near-dup rungs\n",
		result.DedupIndex.FiledStamps, result.DedupIndex.FiledIssuesScanned, result.DedupIndex.WindowIssuesScanned)
	var parts []string
	keys := make([]string, 0, len(result.Skipped))
	for k := range result.Skipped {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if result.Skipped[k] != 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", k, result.Skipped[k]))
		}
	}
	if len(parts) == 0 {
		parts = append(parts, "none")
	}
	fmt.Fprintf(w, "  deduped/dropped: %s\n", strings.Join(parts, ", "))
	renderConversion(w, result)
	if len(result.Planned) == 0 {
		fmt.Fprintln(w, "  -> nothing new worth filing today.")
	} else {
		verb := "would file"
		switch {
		case result.FilingGate.Paused:
			// Never "FILED" under a held gate: a live run that filed nothing because
			// the conversion gate stopped it must not read like one that filed.
			verb = "did NOT file (gate paused)"
		case result.Mode == "live":
			verb = "FILED"
		}
		fmt.Fprintf(w, "  -> %s %d issue(s) (cap %d, min-score %d):\n", verb, len(result.Planned), cfg.MaxIssues, cfg.MinScore)
		for _, issue := range result.Planned {
			fmt.Fprintf(w, "     [%3d] %s\n", issue.Score, issue.Title)
			fmt.Fprintf(w, "           %s  labels=%s\n", issue.URL, strings.Join(issue.Labels, ","))
		}
	}
	if len(result.Errors) > 0 {
		fmt.Fprintln(w, "  errors:")
		for _, e := range result.Errors {
			fmt.Fprintf(w, "     ! %s\n", e)
		}
	}
	if result.Mode == "dry-run" && len(result.Planned) > 0 && !result.FilingGate.Paused {
		fmt.Fprintln(w, "\n  dry-run - file these for real with: fak idea-scout --live")
	}
}

// renderConversion prints the three things #6506 found missing from a run that
// still exited 0: what the existing stock looks like, whether any of it ever
// converted, and whether a source lane was down for the whole run.
func renderConversion(w io.Writer, result RunResult) {
	b := result.Backlog
	fmt.Fprintf(w, "  backlog: %d filed - %d open (%d untriaged, oldest %dd, median %dd) - %d closed\n",
		b.Filed, b.Open, b.Untriaged, b.OldestOpenDays, b.MedianOpenDays, b.Closed)
	fmt.Fprintf(w, "  conversion: %d triaged (%s), %d converted (%s, upper bound), %d no-action (%s)%s\n",
		b.Triaged, pct(b.TriageRate), b.Converted, pct(b.ConversionRate), b.NoAction, pct(b.NoActionRate),
		unclassifiedNote(b))
	for _, lane := range result.SourceHealth {
		if lane.Status == LaneOK {
			continue
		}
		fmt.Fprintf(w, "  source %s: %s (%d/%d topic queries failed)\n", lane.Lane, strings.ToUpper(lane.Status), lane.Failed, lane.Attempted)
	}
	if result.Status == StatusDegraded {
		fmt.Fprintln(w, "  status: DEGRADED - a source lane failed on every topic that armed it, so the candidate pool is incomplete")
	}
	if result.FilingGate.Paused {
		fmt.Fprintf(w, "  FILING PAUSED (%s): %s\n", result.FilingGate.Reason, result.FilingGate.Detail)
		fmt.Fprintln(w, "     drain the stock first (triage or close the open idea-scout issues); prefer a witnessed")
		fmt.Fprintln(w, "     study + gap-witness pass over raw candidate filing while the backlog is above the cap.")
	}
}

func unclassifiedNote(b BacklogStats) string {
	if b.Unclassified == 0 {
		return ""
	}
	return fmt.Sprintf(" [%d of %d carry no state, so the ledger is partial]", b.Unclassified, b.Filed)
}

func pct(rate float64) string {
	return fmt.Sprintf("%.1f%%", rate*100)
}
