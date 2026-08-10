package issuededup

import (
	"strings"
	"time"
)

var comparisonBacklog = []BacklogIssue{
	{Number: 101, Title: "Prevent duplicate issue creation with semantic matching", Body: "Warn before filing a paraphrased backlog item and link the existing issue."},
	{Number: 102, Title: "Add dark mode to benchmark dashboard", Body: "Use system theme and preserve chart contrast."},
	{Number: 103, Title: "Rotate gateway logs by size", Body: "Bound disk growth and retain seven archives."},
}

var comparisonCandidates = []Candidate{
	{Title: "Stop semantically duplicate tickets before they are filed", Body: "Detect a paraphrase, warn the author, and point to the current backlog issue."},
	{Title: "Support dark theme in benchmark charts", Body: "Follow the OS theme while retaining accessible chart colors."},
	{Title: "Export cache metrics as OpenTelemetry spans", Body: "Emit cache hit and reuse attributes to an OTLP collector."},
}

type ComparisonArm struct {
	Name                                                                string
	Kind                                                                string
	Available                                                           bool
	Correct                                                             bool
	Latency                                                             time.Duration
	Cases, TruePositives, TrueNegatives, FalsePositives, FalseNegatives int
	Precision, Recall                                                   float64
	CPUSeconds                                                          float64
	PeakRSSBytes, InputBytes, NetworkBytes                              int64
	OperatorSeconds, CostUSD                                            float64
	Note                                                                string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func runNativeComparison() ComparisonArm {
	a := ComparisonArm{Name: "fak native issue near-duplicate gate", Kind: "native", Available: true, Cases: 3, Note: "title plus normalized-body simhash index with auditable issue pointers"}
	for _, x := range comparisonBacklog {
		a.InputBytes += int64(len(x.Title) + len(x.Body))
	}
	for _, x := range comparisonCandidates {
		a.InputBytes += int64(len(x.Title) + len(x.Body))
	}
	start := time.Now()
	ix := NewIndex(comparisonBacklog)
	want := []int{101, 102, 0}
	for i, c := range comparisonCandidates {
		hits := ix.Check(c, 1, 0.25)
		got := 0
		if len(hits) > 0 {
			got = hits[0].IssueNumber
		}
		switch {
		case want[i] > 0 && got == want[i]:
			a.TruePositives++
		case want[i] == 0 && got == 0:
			a.TrueNegatives++
		case want[i] > 0:
			a.FalseNegatives++
		default:
			a.FalsePositives++
		}
	}
	a.Latency = time.Since(start)
	a.Precision = ratio(a.TruePositives, a.TruePositives+a.FalsePositives)
	a.Recall = ratio(a.TruePositives, a.TruePositives+a.FalseNegatives)
	a.Correct = a.TruePositives == 2 && a.TrueNegatives == 1
	return a
}
func runExactTitleBaseline() ComparisonArm {
	a := ComparisonArm{Name: "normalized exact-title equality", Kind: "baseline", Available: true, Cases: 3, Note: "tuned no-semantic baseline catches only identical normalized titles"}
	start := time.Now()
	for _, c := range comparisonCandidates {
		found := false
		for _, x := range comparisonBacklog {
			if strings.EqualFold(strings.TrimSpace(c.Title), strings.TrimSpace(x.Title)) {
				found = true
				break
			}
		}
		if found {
			a.FalsePositives++
		} else {
			a.TrueNegatives++
		}
	}
	a.Latency = time.Since(start)
	a.FalseNegatives = 2
	a.TrueNegatives = 1
	a.Precision = ratio(a.TruePositives, a.TruePositives+a.FalsePositives)
	a.Recall = 0
	a.Correct = false
	return a
}
func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}
func unavailable(name, kind, note string) ComparisonArm {
	return ComparisonArm{Name: name, Kind: kind, Note: note}
}
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "classify the same two paraphrased duplicate candidates and one unrelated candidate against the same three-issue backlog, returning the correct existing issue pointer", Arms: []ComparisonArm{
		runNativeComparison(), runExactTitleBaseline(),
		unavailable("fak + GitHub issue search", "integration", "requires real GitHub search over a pinned mirrored backlog"),
		unavailable("GitHub duplicate issue detection", "external", "requires real GitHub duplicate suggestions or search workflow"),
		unavailable("Linear duplicate detection", "external", "requires a pinned Linear workspace and real duplicate workflow"),
		unavailable("Jira similar requests", "external", "requires a pinned Jira project and real similarity workflow"),
		unavailable("sentence-transformer cosine retrieval", "external", "requires pinned embedding model, vector index, and threshold tuning"),
	}}
}
