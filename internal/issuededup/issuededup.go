package issuededup

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/simhash"
)

// DefaultThreshold is the minimum cosine similarity that counts as dup-risk.
// Below it a backlog neighbor is normal same-repo vocabulary overlap; at or
// above it the candidate should be shown its twin before filing. Tuned against
// the fixture pair in issuededup_test.go: a paraphrased twin scores well above
// it on the title+body axis while an unrelated candidate sharing the same
// issue-template skeleton stays well below.
const DefaultThreshold = 0.80

// DefaultTopK bounds how many dup-risk verdicts one candidate reports. A
// producer acts on the top match; a couple more give the reviewer context.
const DefaultTopK = 3

// MatchedOn values name the axis a verdict matched on, so the report is
// auditable: a title hit is a near-identical restatement; a title+body hit is a
// paraphrase the title alone did not betray.
const (
	MatchedOnTitle     = "title"
	MatchedOnTitleBody = "title+body"
)

// Candidate is the issue a producer is about to file. Number, when set, names
// the candidate's own already-filed issue so re-reviewing an existing issue
// never flags itself.
type Candidate struct {
	Number int    `json:"number,omitempty"`
	Title  string `json:"title"`
	Body   string `json:"body,omitempty"`
}

// Label is one gh label name on a backlog issue. It is bonus evidence for the
// retrospective census (a shared label strengthens a proposed cluster) and is
// never read by the write-time gate, whose backlog read omits labels.
type Label struct {
	Name string `json:"name"`
}

// BacklogIssue is one row of read-only `gh issue list --json number,title,body`
// output (the write-time gate) or `...,labels` (the census) — the only shape the
// index is built from, so the whole path stays offline-safe (cached reads in,
// verdicts out, no gh write anywhere). Labels is optional: the write-time gate
// leaves it nil, the census fills it from the labels field for shared-label
// evidence.
type BacklogIssue struct {
	Number int     `json:"number"`
	Title  string  `json:"title"`
	Body   string  `json:"body,omitempty"`
	Labels []Label `json:"labels,omitempty"`
}

// Verdict is one dup-risk match. It always carries the matched issue's number
// and title — an auditable pointer to the suspected twin, never a bare score.
type Verdict struct {
	IssueNumber int     `json:"issue_number"`
	Title       string  `json:"title,omitempty"`
	Similarity  float64 `json:"similarity"`
	MatchedOn   string  `json:"matched_on"`
}

// Index is the backlog near-dup index: one title vector and one title+body
// vector per open issue. Build it once from a cached backlog read, then Check
// each candidate against it. Like simhash.Index it is not safe for concurrent
// mutation; build, then query.
type Index struct {
	titles   simhash.Index
	combined simhash.Index
	titleOf  map[int]string
}

// NewIndex builds the index over issues. Rows without a positive number or a
// non-empty title are skipped — they cannot yield an auditable verdict. A
// repeated issue number replaces the earlier row (simhash.Index.Add semantics),
// so overlapping paged reads never double-count one issue.
func NewIndex(issues []BacklogIssue) *Index {
	ix := &Index{titleOf: make(map[int]string, len(issues))}
	for _, is := range issues {
		title := strings.TrimSpace(is.Title)
		if is.Number <= 0 || title == "" {
			continue
		}
		id := strconv.Itoa(is.Number)
		ix.titles.AddText(id, title, "")
		ix.combined.AddText(id, title+"\n"+normalizeBody(is.Body), "")
		ix.titleOf[is.Number] = title
	}
	return ix
}

// Len is the number of indexed issues.
func (ix *Index) Len() int { return ix.titles.Len() }

// Check returns the dup-risk verdicts for c against the indexed backlog,
// descending by similarity (ties break by issue number ascending). k <= 0 uses
// DefaultTopK; threshold <= 0 uses DefaultThreshold. Each backlog issue is
// scored on both axes and reports its best; c.Number, when set, is excluded so
// an existing issue re-reviewed against a backlog containing itself passes.
func (ix *Index) Check(c Candidate, k int, threshold float64) []Verdict {
	if k <= 0 {
		k = DefaultTopK
	}
	if threshold <= 0 {
		threshold = DefaultThreshold
	}
	title := strings.TrimSpace(c.Title)
	if title == "" || ix.Len() == 0 {
		return nil
	}
	best := map[int]Verdict{}
	fold := func(matches []simhash.Match, axis string) {
		for _, m := range matches {
			number, err := strconv.Atoi(m.ID)
			if err != nil || number == c.Number || m.Score < threshold {
				continue
			}
			prev, seen := best[number]
			if !seen || m.Score > prev.Similarity {
				best[number] = Verdict{
					IssueNumber: number,
					Title:       ix.titleOf[number],
					Similarity:  m.Score,
					MatchedOn:   axis,
				}
			}
		}
	}
	// Fold title first so an exact-tie across axes reports the more specific one.
	fold(ix.titles.TopK(simhash.Embed(title), 0), MatchedOnTitle)
	fold(ix.combined.TopK(simhash.Embed(title+"\n"+normalizeBody(c.Body)), 0), MatchedOnTitleBody)

	out := make([]Verdict, 0, len(best))
	for _, v := range best {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Similarity != out[j].Similarity {
			return out[i].Similarity > out[j].Similarity
		}
		return out[i].IssueNumber < out[j].IssueNumber
	})
	if k < len(out) {
		out = out[:k]
	}
	return out
}

// ParseBacklog decodes cached `gh issue list --json number,title,body` output.
// It tolerates a UTF-8 BOM (PowerShell redirection stamps one) and returns the
// rows as-is; NewIndex does the filtering.
func ParseBacklog(b []byte) ([]BacklogIssue, error) {
	b = []byte(strings.TrimPrefix(string(b), "\ufeff"))
	var issues []BacklogIssue
	if err := json.Unmarshal(b, &issues); err != nil {
		return nil, fmt.Errorf("backlog is not a gh issue list --json number,title,body array: %w", err)
	}
	return issues, nil
}

// normalizeBody strips the parts of an issue body every producer shares — the
// marker comment and the template's section headings — so body similarity
// measures the prose that differs between issues, not the skeleton they all
// carry. A heading is '#'-run + space (markdown); a bare "#2504" reference line
// is prose and survives.
func normalizeBody(body string) string {
	if body == "" {
		return ""
	}
	lines := strings.Split(body, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		t := strings.TrimSpace(strings.TrimPrefix(line, "\ufeff"))
		if t == "" {
			continue
		}
		if strings.HasPrefix(t, "<!--") && strings.HasSuffix(t, "-->") {
			continue
		}
		if h := strings.TrimLeft(t, "#"); len(h) < len(t) && (h == "" || strings.HasPrefix(h, " ")) {
			continue
		}
		out = append(out, t)
	}
	return strings.Join(out, "\n")
}
