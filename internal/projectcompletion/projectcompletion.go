package projectcompletion

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

const Schema = "fak.project-completion.v1"

type Issue struct {
	Number      int
	Title       string
	State       string
	ProjectWork issuepolicy.ProjectWorkReadout
}
type Bucket struct {
	Standard string  `json:"standard"`
	Points   float64 `json:"points"`
	Issues   int     `json:"issues"`
	// Label is the maturity-qualified completion label ("demo-complete",
	// "production-complete") for operator renders; Standard stays the stable
	// machine field existing JSON consumers already read (#4640).
	Label string `json:"label,omitempty"`
}
type UnknownIssue struct {
	Number  int      `json:"number,omitempty"`
	Title   string   `json:"title"`
	Status  string   `json:"status"`
	Invalid []string `json:"invalid,omitempty"`
}
type Report struct {
	Schema                   string         `json:"schema"`
	Parent                   string         `json:"parent,omitempty"`
	BaselinePoints           float64        `json:"baseline_points"`
	DeclaredContribution     float64        `json:"declared_contribution_points"`
	ProductionCompletePoints float64        `json:"production_complete_points"`
	ProductionCompletePct    float64        `json:"production_complete_percent"`
	ClosedByStandard         []Bucket       `json:"closed_by_standard,omitempty"`
	OpenPoints               float64        `json:"open_points"`
	Unknown                  []UnknownIssue `json:"unknown,omitempty"`
	DenominatorDrift         []string       `json:"denominator_drift,omitempty"`
	Confidence               string         `json:"confidence"`
}

// Summarize computes progress against the declared parent production baseline.
// Closed demo/prototype/research work stays visible but receives no production credit.
func Summarize(issues []Issue) Report {
	r := Report{Schema: Schema, Confidence: "complete"}
	buckets := map[string]*Bucket{}
	parents := map[string]float64{}
	for _, issue := range issues {
		pw := issue.ProjectWork
		if pw.Status != issuepolicy.ProjectWorkValid {
			r.Unknown = append(r.Unknown, UnknownIssue{issue.Number, issue.Title, pw.Status, append([]string(nil), pw.Invalid...)})
			continue
		}
		if old, ok := parents[pw.Parent]; ok && !same(old, pw.ParentBaseline) {
			r.DenominatorDrift = append(r.DenominatorDrift, fmt.Sprintf("parent %q has both %.2f and %.2f baseline points", pw.Parent, old, pw.ParentBaseline))
		} else {
			parents[pw.Parent] = pw.ParentBaseline
		}
		r.DeclaredContribution += pw.Contribution
		if !strings.EqualFold(strings.TrimSpace(issue.State), "closed") {
			r.OpenPoints += pw.Contribution
			continue
		}
		b := buckets[pw.CompletionStandard]
		if b == nil {
			b = &Bucket{Standard: pw.CompletionStandard, Label: MaturityLabel(pw.CompletionStandard)}
			buckets[pw.CompletionStandard] = b
		}
		b.Points += pw.Contribution
		b.Issues++
		if pw.ProductionCredit {
			r.ProductionCompletePoints += pw.Contribution
		}
	}
	if len(parents) == 1 {
		for k, v := range parents {
			r.Parent, r.BaselinePoints = k, v
		}
	} else if len(parents) > 1 {
		keys := make([]string, 0, len(parents))
		for k := range parents {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		r.DenominatorDrift = append(r.DenominatorDrift, "multiple parents: "+strings.Join(keys, ", "))
	}
	if r.BaselinePoints > 0 {
		r.ProductionCompletePct = 100 * r.ProductionCompletePoints / r.BaselinePoints
		if r.DeclaredContribution > r.BaselinePoints+0.000001 {
			r.DenominatorDrift = append(r.DenominatorDrift, fmt.Sprintf("declared contributions %.2f exceed baseline %.2f", r.DeclaredContribution, r.BaselinePoints))
		}
	}
	for _, b := range buckets {
		r.ClosedByStandard = append(r.ClosedByStandard, *b)
	}
	sort.Slice(r.ClosedByStandard, func(i, j int) bool { return r.ClosedByStandard[i].Standard < r.ClosedByStandard[j].Standard })
	sort.Slice(r.Unknown, func(i, j int) bool { return r.Unknown[i].Number < r.Unknown[j].Number })
	sort.Strings(r.DenominatorDrift)
	if len(r.Unknown) > 0 || len(r.DenominatorDrift) > 0 || r.BaselinePoints <= 0 || r.DeclaredContribution+0.000001 < r.BaselinePoints {
		r.Confidence = "incomplete"
	}
	return r
}

// MaturityLabel renders a completion standard as an explicit maturity-qualified
// completion label. Operator status must never render a bare "complete" for a
// non-production leaf (#4640): a closed toy bring-up reads "demo-complete", and
// a close with no declared standard reads as undeclared, never as complete.
func MaturityLabel(standard string) string {
	s := strings.ToLower(strings.TrimSpace(standard))
	if s == "" {
		return "closed (maturity undeclared)"
	}
	return s + "-complete"
}

// RenderText is the operator-facing status render for a Report. Every closed
// bucket carries its maturity-qualified label so a closed toy bring-up stays
// visibly demo-complete while the parent remains partially production-complete.
// JSON consumers read the stable Report fields instead of parsing this text.
func RenderText(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "production complete: %.2f/%.2f points (%.1f%%) [%s]\n", r.ProductionCompletePoints, r.BaselinePoints, r.ProductionCompletePct, r.Confidence)
	for _, bucket := range r.ClosedByStandard {
		label := bucket.Label
		if label == "" {
			label = MaturityLabel(bucket.Standard)
		}
		fmt.Fprintf(&b, "closed %-14s %.2f points (%d issues)\n", label, bucket.Points, bucket.Issues)
	}
	fmt.Fprintf(&b, "open: %.2f points; declared: %.2f points\n", r.OpenPoints, r.DeclaredContribution)
	for _, u := range r.Unknown {
		fmt.Fprintf(&b, "unknown #%d %s: %s\n", u.Number, u.Title, u.Status)
	}
	for _, d := range r.DenominatorDrift {
		fmt.Fprintf(&b, "denominator drift: %s\n", d)
	}
	return b.String()
}

func same(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 0.000001
}
