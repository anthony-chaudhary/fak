package issuecentrality

import (
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

const SampleSchema = "fak-issue-centrality-sample/1"

const FamilyUnknown = "unclassified-family"

var familyLabels = []string{
	"managed-context", "prompt-caching", "context-engineering", "agent-runtime",
	"agentic-serving", "model-support", "model-arch", "model-routing", "gpu",
	"dispatch", "orchestration", "observability", "security", "trust-floor",
	"harness-native", "agent-framework", "integration", "mcp", "benchmark",
	"performance", "operator", "ux", "documentation", "testing", "dev-ex",
	"deployment", "ci-cd", "enterprise", "distributed", "research",
}

type PortfolioIssue struct {
	Number    int        `json:"number"`
	Title     string     `json:"title"`
	Body      string     `json:"body"`
	Labels    []Label    `json:"labels"`
	Milestone *Milestone `json:"milestone"`
	UpdatedAt string     `json:"updatedAt"`
}

type Label struct {
	Name string `json:"name"`
}

type Milestone struct {
	Title string `json:"title"`
}

type SampleOptions struct {
	PerFamily int
}

type SampleRow struct {
	Number        int      `json:"number"`
	Title         string   `json:"title"`
	Family        string   `json:"family"`
	Reasons       []string `json:"selection_reasons"`
	Decision      string   `json:"decision"`
	Rationale     string   `json:"rationale"`
	Centrality    string   `json:"centrality"`
	NamedOutcome  string   `json:"named_outcome_or_obligation,omitempty"`
	EvidenceGaps  []string `json:"evidence_gaps"`
	RepairActions []string `json:"repair_actions"`
}

type FamilyCount struct {
	Family     string `json:"family"`
	Population int    `json:"population"`
	Selected   int    `json:"selected"`
}

type Sample struct {
	Schema        string        `json:"schema"`
	Total         int           `json:"total"`
	P0Total       int           `json:"p0_total"`
	GenNowTotal   int           `json:"gen_now_total"`
	Milestoneless int           `json:"milestoneless_total"`
	Rows          []SampleRow   `json:"rows"`
	Families      []FamilyCount `json:"families"`
}

func BuildSample(issues []PortfolioIssue, opt SampleOptions) Sample {
	if opt.PerFamily < 1 {
		opt.PerFamily = 1
	}
	out := Sample{Schema: SampleSchema, Total: len(issues), Rows: []SampleRow{}, Families: []FamilyCount{}}
	selected := make(map[int]*SampleRow)
	milestoneless := make(map[string][]PortfolioIssue)
	for _, issue := range issues {
		labels := labelSet(issue.Labels)
		if labels["priority/P0"] {
			out.P0Total++
			addSampleRow(selected, issue, primaryFamily(labels), "priority/P0")
		}
		if labels["gen/now"] {
			out.GenNowTotal++
			addSampleRow(selected, issue, primaryFamily(labels), "gen/now")
		}
		if issue.Milestone == nil {
			out.Milestoneless++
			family := primaryFamily(labels)
			milestoneless[family] = append(milestoneless[family], issue)
		}
	}
	families := make([]string, 0, len(milestoneless))
	for family := range milestoneless {
		families = append(families, family)
	}
	sort.Strings(families)
	for _, family := range families {
		rows := milestoneless[family]
		sort.SliceStable(rows, func(i, j int) bool {
			left, leftOK := parseUpdatedAt(rows[i].UpdatedAt)
			right, rightOK := parseUpdatedAt(rows[j].UpdatedAt)
			if leftOK != rightOK {
				return leftOK
			}
			if leftOK && !left.Equal(right) {
				return left.After(right)
			}
			return rows[i].Number < rows[j].Number
		})
		picked := 0
		for _, issue := range rows {
			if _, alreadySelected := selected[issue.Number]; alreadySelected {
				continue
			}
			addSampleRow(selected, issue, family, "milestoneless-stratum")
			picked++
			if picked >= opt.PerFamily {
				break
			}
		}
		out.Families = append(out.Families, FamilyCount{Family: family, Population: len(rows), Selected: picked})
	}
	for _, row := range selected {
		sort.Strings(row.Reasons)
		out.Rows = append(out.Rows, *row)
	}
	sort.SliceStable(out.Rows, func(i, j int) bool { return out.Rows[i].Number < out.Rows[j].Number })
	return out
}

func addSampleRow(rows map[int]*SampleRow, issue PortfolioIssue, family, reason string) {
	if row, ok := rows[issue.Number]; ok {
		if !contains(row.Reasons, reason) {
			row.Reasons = append(row.Reasons, reason)
		}
		return
	}
	frame := issuepolicy.AssessProblemFrame(issuepolicy.IssueDraft{Number: issue.Number, Title: issue.Title, Body: issue.Body})
	decision, rationale, gaps, repairs := frameDisposition(frame)
	named := frame.CentralityTarget
	if frame.Centrality == issuepolicy.CentralityCore && named == "" {
		named = issue.Title
	}
	rows[issue.Number] = &SampleRow{
		Number: issue.Number, Title: issue.Title, Family: family, Reasons: []string{reason},
		Decision: decision, Rationale: rationale, Centrality: frame.Centrality, NamedOutcome: named,
		EvidenceGaps: gaps, RepairActions: repairs,
	}
}

func frameDisposition(frame issuepolicy.ProblemFrame) (string, string, []string, []string) {
	switch {
	case !frame.Enforced || frame.Centrality == issuepolicy.CentralityUnclassified:
		return "unknown-with-missing-evidence", "retain unknown until the issue itself supplies a canonical problem frame; metadata is not classification evidence", []string{"problem_frame_unclassified"}, []string{"review issue evidence and explicitly declare Centrality plus P1-P4 before queue action"}
	case frame.Ready:
		return "retain", "canonical problem frame is structurally valid; retain its declared qualitative centrality without converting it to a score", []string{}, []string{}
	default:
		return "reframe", "the issue declares the canonical vocabulary but its frame is incomplete or ceremonial", append([]string(nil), frame.Reasons...), append([]string(nil), frame.RepairActions...)
	}
}

func labelSet(labels []Label) map[string]bool {
	out := make(map[string]bool, len(labels))
	for _, label := range labels {
		out[strings.TrimSpace(label.Name)] = true
	}
	return out
}

func primaryFamily(labels map[string]bool) string {
	for _, family := range familyLabels {
		if labels[family] {
			return family
		}
	}
	return FamilyUnknown
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func parseUpdatedAt(value string) (time.Time, bool) {
	parsed, err := time.Parse(time.RFC3339, strings.TrimSpace(value))
	return parsed, err == nil
}
