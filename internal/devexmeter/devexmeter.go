package devexmeter

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

const (
	// RowSchema is the JSONL schema for one dev-ex friction observation.
	RowSchema = "fak.devexmeter.row.v1"
	// GateSchema is the JSON schema for the close-gate result.
	GateSchema = "fak.devexmeter.gate.v1"

	VerdictPass   = "PASS"
	VerdictNotYet = "NOT_YET"
	VerdictSkip   = "SKIP"
)

// Issue is the GitHub issue surface the close gate needs. Labels are raw label
// names; ClassFromIssue accepts either a friction/<class> label or a body marker.
type Issue struct {
	Number int      `json:"number"`
	Title  string   `json:"title,omitempty"`
	Body   string   `json:"body,omitempty"`
	Labels []string `json:"labels,omitempty"`
}

// Row records one measured value for a dev-ex friction class over a named
// window. Lower Value is better. Window must be before or after. Samples can
// weight repeated observations inside a window; <=0 means weight 1.
type Row struct {
	Schema  string  `json:"schema,omitempty"`
	Issue   int     `json:"issue,omitempty"`
	Class   string  `json:"class"`
	Window  string  `json:"window"`
	Value   float64 `json:"value"`
	Samples int     `json:"samples,omitempty"`
	Source  string  `json:"source,omitempty"`
}

// WindowStat is the folded metric for one window.
type WindowStat struct {
	Rows    int      `json:"rows"`
	Samples int      `json:"samples"`
	Value   float64  `json:"value"`
	Sources []string `json:"sources,omitempty"`
}

// Fold is the before/after measurement for one issue/class pair.
type Fold struct {
	Issue  int        `json:"issue"`
	Class  string     `json:"class"`
	Before WindowStat `json:"before"`
	After  WindowStat `json:"after"`
}

// GateResult decides whether a dev-ex issue may close green.
type GateResult struct {
	Schema         string   `json:"schema"`
	OK             bool     `json:"ok"`
	Verdict        string   `json:"verdict"`
	Issue          int      `json:"issue,omitempty"`
	Class          string   `json:"class,omitempty"`
	Before         *float64 `json:"before,omitempty"`
	After          *float64 `json:"after,omitempty"`
	Delta          *float64 `json:"delta,omitempty"`
	Reason         string   `json:"reason"`
	MissingWitness []string `json:"missing_witness,omitempty"`
}

// ParseIssue decodes the gh issue JSON shape. GitHub labels may arrive either
// as strings or as objects containing a name field.
func ParseIssue(data []byte) (Issue, error) {
	var raw struct {
		Number int             `json:"number"`
		Title  string          `json:"title"`
		Body   string          `json:"body"`
		Labels json.RawMessage `json:"labels"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return Issue{}, fmt.Errorf("devexmeter: parse issue: %w", err)
	}
	issue := Issue{Number: raw.Number, Title: raw.Title, Body: raw.Body}
	if len(bytes.TrimSpace(raw.Labels)) == 0 {
		return issue, nil
	}
	var stringLabels []string
	if err := json.Unmarshal(raw.Labels, &stringLabels); err == nil {
		issue.Labels = stringLabels
		return issue, nil
	}
	var objectLabels []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw.Labels, &objectLabels); err != nil {
		return Issue{}, fmt.Errorf("devexmeter: parse issue labels: %w", err)
	}
	for _, label := range objectLabels {
		if strings.TrimSpace(label.Name) != "" {
			issue.Labels = append(issue.Labels, label.Name)
		}
	}
	return issue, nil
}

// ParseLedger reads newline-delimited Row JSON. Blank lines are ignored; bad
// rows fail closed with a located error.
func ParseLedger(data []byte) ([]Row, error) {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var rows []Row
	line := 0
	for sc.Scan() {
		line++
		raw := bytes.TrimSpace(sc.Bytes())
		if len(raw) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, fmt.Errorf("devexmeter: ledger line %d: %w", line, err)
		}
		if err := validateRow(r); err != nil {
			return nil, fmt.Errorf("devexmeter: ledger line %d: %w", line, err)
		}
		rows = append(rows, r)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("devexmeter: scan ledger: %w", err)
	}
	return rows, nil
}

func validateRow(r Row) error {
	if normalizeClass(r.Class) == "" {
		return fmt.Errorf("class is required")
	}
	switch normalizeWindow(r.Window) {
	case "before", "after":
	default:
		return fmt.Errorf("window must be before or after, got %q", r.Window)
	}
	if math.IsNaN(r.Value) || math.IsInf(r.Value, 0) || r.Value < 0 {
		return fmt.Errorf("value must be a finite non-negative number")
	}
	return nil
}

// FoldRows folds all rows matching issue/class. Rows with Issue == 0 are
// class-wide and can witness a class-level issue; issue-specific rows also match.
func FoldRows(rows []Row, issue int, class string) Fold {
	class = normalizeClass(class)
	var before, after statBuilder
	for _, r := range rows {
		if normalizeClass(r.Class) != class {
			continue
		}
		if r.Issue != 0 && issue != 0 && r.Issue != issue {
			continue
		}
		switch normalizeWindow(r.Window) {
		case "before":
			before.add(r)
		case "after":
			after.add(r)
		}
	}
	return Fold{
		Issue:  issue,
		Class:  class,
		Before: before.stat(),
		After:  after.stat(),
	}
}

type statBuilder struct {
	rows    int
	weight  int
	sum     float64
	sources map[string]bool
}

func (b *statBuilder) add(r Row) {
	w := r.Samples
	if w <= 0 {
		w = 1
	}
	b.rows++
	b.weight += w
	b.sum += r.Value * float64(w)
	if s := strings.TrimSpace(r.Source); s != "" {
		if b.sources == nil {
			b.sources = map[string]bool{}
		}
		b.sources[s] = true
	}
}

func (b statBuilder) stat() WindowStat {
	st := WindowStat{Rows: b.rows, Samples: b.weight}
	if b.weight > 0 {
		st.Value = round4(b.sum / float64(b.weight))
	}
	if len(b.sources) > 0 {
		st.Sources = make([]string, 0, len(b.sources))
		for source := range b.sources {
			st.Sources = append(st.Sources, source)
		}
		sort.Strings(st.Sources)
	}
	return st
}

// GateIssue returns PASS only when a dev-ex issue with a friction class has both
// before and after windows and the after value is strictly lower.
func GateIssue(issue Issue, rows []Row) GateResult {
	class, tagged := ClassFromIssue(issue)
	result := GateResult{
		Schema: GateSchema,
		Issue:  issue.Number,
		Class:  class,
	}
	if !hasDevExLabel(issue.Labels) {
		result.OK = true
		result.Verdict = VerdictSkip
		result.Reason = "issue is not labeled dev-ex; devexmeter close gate does not apply"
		return result
	}
	if !tagged {
		result.OK = true
		result.Verdict = VerdictSkip
		result.Reason = "dev-ex issue has no friction class tag; no measured friction claim to gate"
		return result
	}

	fold := FoldRows(rows, issue.Number, class)
	if fold.Before.Rows == 0 || fold.After.Rows == 0 {
		result.Verdict = VerdictNotYet
		if fold.Before.Rows == 0 {
			result.MissingWitness = append(result.MissingWitness, "before meter window for friction class "+class)
		}
		if fold.After.Rows == 0 {
			result.MissingWitness = append(result.MissingWitness, "after meter window for friction class "+class)
		}
		result.Reason = "not yet: closing a dev-ex friction issue requires before/after meter evidence"
		return result
	}

	before := fold.Before.Value
	after := fold.After.Value
	delta := round4(after - before)
	result.Before = &before
	result.After = &after
	result.Delta = &delta
	if after < before {
		result.OK = true
		result.Verdict = VerdictPass
		result.Reason = fmt.Sprintf("PASS: %s friction dropped %.4g -> %.4g (delta %.4g)", class, before, after, delta)
		return result
	}
	result.Verdict = VerdictNotYet
	result.MissingWitness = []string{fmt.Sprintf("strict drop for friction class %s: after %.4g must be lower than before %.4g", class, after, before)}
	result.Reason = fmt.Sprintf("not yet: %s friction did not drop (%.4g -> %.4g, delta %.4g)", class, before, after, delta)
	return result
}

// ClassFromIssue finds the friction class named by a dev-ex issue. Labels are
// preferred: friction/<class>, friction:<class>, friction-class/<class>, or
// friction-class:<class>. The body may also carry "Friction-Class: <class>".
func ClassFromIssue(issue Issue) (string, bool) {
	for _, label := range issue.Labels {
		if class, ok := classFromLabel(label); ok {
			return class, true
		}
	}
	if class := classFromBody(issue.Body); class != "" {
		return class, true
	}
	return "", false
}

func hasDevExLabel(labels []string) bool {
	for _, label := range labels {
		n := strings.ToLower(strings.TrimSpace(label))
		if n == "dev-ex" || n == "devex" || n == "developer-experience" {
			return true
		}
	}
	return false
}

func classFromLabel(label string) (string, bool) {
	n := strings.TrimSpace(label)
	lower := strings.ToLower(n)
	for _, prefix := range []string{"friction/", "friction:", "friction-class/", "friction-class:"} {
		if strings.HasPrefix(lower, prefix) {
			return normalizeClass(n[len(prefix):]), normalizeClass(n[len(prefix):]) != ""
		}
	}
	return "", false
}

var classBodyRE = regexp.MustCompile(`(?im)^\s*friction[-_ ]class\s*:\s*([A-Za-z0-9_.:/-]+)\s*$`)

func classFromBody(body string) string {
	m := classBodyRE.FindStringSubmatch(body)
	if len(m) != 2 {
		return ""
	}
	return normalizeClass(m[1])
}

func normalizeClass(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	s = strings.ReplaceAll(s, "_", "-")
	s = strings.Trim(s, "/:")
	return s
}

func normalizeWindow(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}
