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
	fmt.Fprintf(w, "  gathered %d candidates from %d topics x (arXiv + GitHub + Hacker News + Reddit)\n", result.CandidatesGathered, result.Topics)
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
	if len(result.Planned) == 0 {
		fmt.Fprintln(w, "  -> nothing new worth filing today.")
	} else {
		verb := "would file"
		if result.Mode == "live" {
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
	if result.Mode == "dry-run" && len(result.Planned) > 0 {
		fmt.Fprintln(w, "\n  dry-run - file these for real with: fak idea-scout --live")
	}
}
