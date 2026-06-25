package swebenchsota

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Schema is the versioned identity stamped on every emitted snapshot.
const Schema = "fak.swebench-sota-snapshot.v1"

// DefaultURL is the official SWE-bench leaderboard page.
const DefaultURL = "https://www.swebench.com/"

// userAgent matches the Python tool's request header.
const userAgent = "fak-swebench-sota-snapshot/1.0"

// scriptRe matches the embedded leaderboard JSON script tag. It mirrors the
// Python re.DOTALL|re.IGNORECASE pattern: (?is) enables those flags, .*? is the
// lazy body capture.
var scriptRe = regexp.MustCompile(`(?is)<script\s+type="application/json"\s+id="leaderboard-data">\s*(.*?)\s*</script>`)

// Group is one leaderboard group ("Verified", "bash-only", ...). The raw JSON
// rows are kept as decoded maps so every field the page carries (and any field a
// future page adds) survives the round trip, matching the Python dict handling.
type Group struct {
	Name    string
	Results []map[string]any
}

// RowSummary is the per-row projection emitted in a snapshot. The field set and
// JSON names mirror tools/swebench_sota_snapshot.py::row_summary exactly.
type RowSummary struct {
	Rank                int `json:"rank"`
	Name                any `json:"name"`
	ResolvedPct         any `json:"resolved_pct"`
	Date                any `json:"date"`
	MiniSWEAgentVersion any `json:"mini_swe_agent_version"`
	Cost                any `json:"cost"`
	InstanceCost        any `json:"instance_cost"`
	Site                any `json:"site"`
	Folder              any `json:"folder"`
	OSModel             any `json:"os_model"`
	OSSystem            any `json:"os_system"`
	Warning             any `json:"warning"`
}

// GroupSummary is the per-group SOTA fold emitted in a snapshot.
type GroupSummary struct {
	Name      any          `json:"name"`
	Entries   int          `json:"entries"`
	Top       *RowSummary  `json:"top"`
	TopN      []RowSummary `json:"top_n"`
	FocalRows []RowSummary `json:"focal_rows"`
}

// ComparisonScope records the group/pattern choices and the honesty caveat.
type ComparisonScope struct {
	OverallGroup      string `json:"overall_group"`
	SameScaffoldGroup string `json:"same_scaffold_group"`
	FocalPattern      string `json:"focal_pattern"`
	Caveat            string `json:"caveat"`
}

// Snapshot is the versioned snapshot document.
type Snapshot struct {
	Schema           string          `json:"schema"`
	GeneratedAt      string          `json:"generated_at"`
	SourceURL        string          `json:"source_url"`
	SourceRef        string          `json:"source_ref"`
	SourceSHA256     string          `json:"source_sha256"`
	Benchmark        string          `json:"benchmark"`
	Metric           string          `json:"metric"`
	Instances        int             `json:"instances"`
	ComparisonScope  ComparisonScope `json:"comparison_scope"`
	OverallSOTA      GroupSummary    `json:"overall_sota"`
	SameScaffoldSOTA GroupSummary    `json:"same_scaffold_sota"`
}

// Options control how a snapshot is built from a source HTML string.
type Options struct {
	URL               string
	OverallGroup      string
	SameScaffoldGroup string
	FocalPattern      string
	Limit             int
}

// caveat is the verbatim honesty caveat from the Python tool.
const caveat = "This is a public full-500 leaderboard context snapshot. It is not " +
	"a substitute for the battery's live 20-task raw-vLLM vs fak-gateway run."

// defaults fills zero-valued Options with the Python argparse defaults.
func (o Options) defaults() Options {
	if o.URL == "" {
		o.URL = DefaultURL
	}
	if o.OverallGroup == "" {
		o.OverallGroup = "Verified"
	}
	if o.SameScaffoldGroup == "" {
		o.SameScaffoldGroup = "bash-only"
	}
	if o.FocalPattern == "" {
		o.FocalPattern = `\bGLM-5\b`
	}
	if o.Limit <= 0 {
		o.Limit = 10
	}
	return o
}

// ExtractLeaderboard finds the embedded leaderboard JSON script tag in source,
// HTML-unescapes it, parses it, and returns the list of groups. It is the pure
// half: no network, deterministic given the input string.
func ExtractLeaderboard(source string) ([]Group, error) {
	m := scriptRe.FindStringSubmatch(source)
	if m == nil {
		return nil, fmt.Errorf("official leaderboard JSON script tag not found")
	}
	var data []map[string]any
	if err := json.Unmarshal([]byte(html.UnescapeString(m[1])), &data); err != nil {
		// Distinguish a non-list payload the way the Python tool does.
		var any0 any
		if err2 := json.Unmarshal([]byte(html.UnescapeString(m[1])), &any0); err2 == nil {
			return nil, fmt.Errorf("leaderboard JSON is not a list")
		}
		return nil, fmt.Errorf("parse leaderboard JSON: %w", err)
	}
	groups := make([]Group, 0, len(data))
	for _, g := range data {
		grp := Group{}
		if n, ok := g["name"].(string); ok {
			grp.Name = n
		}
		if rows, ok := g["results"].([]any); ok {
			for _, r := range rows {
				if rm, ok := r.(map[string]any); ok {
					grp.Results = append(grp.Results, rm)
				}
			}
		}
		groups = append(groups, grp)
	}
	return groups, nil
}

// GroupByName returns the group with the given name, or an error if absent.
func GroupByName(groups []Group, name string) (Group, error) {
	for _, g := range groups {
		if g.Name == name {
			return g, nil
		}
	}
	return Group{}, fmt.Errorf("leaderboard group %q not found", name)
}

// rowResolved returns the row's resolved value as a float and whether it is a
// number (the Python isinstance(..., (int, float)) filter).
func rowResolved(row map[string]any) (float64, bool) {
	v, ok := row["resolved"]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	default:
		return 0, false
	}
}

// rowDate returns the row's date as a string ("" when absent/null).
func rowDate(row map[string]any) string {
	if s, ok := row["date"].(string); ok {
		return s
	}
	return ""
}

// SortedRows filters a group to rows with a numeric resolved value and sorts
// them by (resolved, date) descending — the SOTA fold ordering.
func SortedRows(g Group) []map[string]any {
	filtered := make([]map[string]any, 0, len(g.Results))
	for _, row := range g.Results {
		if _, ok := rowResolved(row); ok {
			filtered = append(filtered, row)
		}
	}
	sort.SliceStable(filtered, func(i, j int) bool {
		ri, _ := rowResolved(filtered[i])
		rj, _ := rowResolved(filtered[j])
		if ri != rj {
			return ri > rj
		}
		return rowDate(filtered[i]) > rowDate(filtered[j])
	})
	return filtered
}

func rowSummary(row map[string]any, rank int) RowSummary {
	return RowSummary{
		Rank:                rank,
		Name:                row["name"],
		ResolvedPct:         row["resolved"],
		Date:                row["date"],
		MiniSWEAgentVersion: row["mini-swe-agent_version"],
		Cost:                row["cost"],
		InstanceCost:        row["instance_cost"],
		Site:                row["site"],
		Folder:              row["folder"],
		OSModel:             row["os_model"],
		OSSystem:            row["os_system"],
		Warning:             row["warning"],
	}
}

// SummarizeGroup folds a group into its SOTA top row, a top-N window, and the
// focal rows whose name matches focalRe (skipped when focalRe is nil).
func SummarizeGroup(g Group, limit int, focalRe *regexp.Regexp) (GroupSummary, error) {
	rows := SortedRows(g)
	var name any
	if g.Name != "" {
		name = g.Name
	}
	out := GroupSummary{Name: name, Entries: len(rows), TopN: []RowSummary{}, FocalRows: []RowSummary{}}
	if len(rows) > 0 {
		t := rowSummary(rows[0], 1)
		out.Top = &t
	}
	for i, row := range rows {
		if i >= limit {
			break
		}
		out.TopN = append(out.TopN, rowSummary(row, i+1))
	}
	if focalRe != nil {
		for i, row := range rows {
			n, _ := row["name"].(string)
			if focalRe.MatchString(n) {
				out.FocalRows = append(out.FocalRows, rowSummary(row, i+1))
			}
		}
	}
	return out, nil
}

// BuildSnapshot runs the full pure pipeline over a source HTML string and a
// source reference label, producing the versioned snapshot. generatedAt is
// injected so callers can stamp time.Now (or a fixed value in tests).
func BuildSnapshot(source, sourceRef, generatedAt string, opt Options) (Snapshot, error) {
	opt = opt.defaults()
	groups, err := ExtractLeaderboard(source)
	if err != nil {
		return Snapshot{}, err
	}
	var focalRe *regexp.Regexp
	if opt.FocalPattern != "" {
		focalRe, err = regexp.Compile("(?i)" + opt.FocalPattern)
		if err != nil {
			return Snapshot{}, fmt.Errorf("compile focal pattern: %w", err)
		}
	}

	overallGroup, err := GroupByName(groups, opt.OverallGroup)
	if err != nil {
		return Snapshot{}, err
	}
	sameGroup, err := GroupByName(groups, opt.SameScaffoldGroup)
	if err != nil {
		return Snapshot{}, err
	}
	overall, err := SummarizeGroup(overallGroup, opt.Limit, focalRe)
	if err != nil {
		return Snapshot{}, err
	}
	same, err := SummarizeGroup(sameGroup, opt.Limit, focalRe)
	if err != nil {
		return Snapshot{}, err
	}

	sum := sha256.Sum256([]byte(source))
	return Snapshot{
		Schema:       Schema,
		GeneratedAt:  generatedAt,
		SourceURL:    opt.URL,
		SourceRef:    sourceRef,
		SourceSHA256: hex.EncodeToString(sum[:]),
		Benchmark:    "SWE-bench Verified",
		Metric:       "resolved_pct",
		Instances:    500,
		ComparisonScope: ComparisonScope{
			OverallGroup:      opt.OverallGroup,
			SameScaffoldGroup: opt.SameScaffoldGroup,
			FocalPattern:      opt.FocalPattern,
			Caveat:            caveat,
		},
		OverallSOTA:      overall,
		SameScaffoldSOTA: same,
	}, nil
}

// Check returns the list of structural problems with a built snapshot, matching
// tools/swebench_sota_snapshot.py::check_snapshot.
func Check(doc Snapshot) []string {
	var problems []string
	if doc.Schema != Schema {
		problems = append(problems, fmt.Sprintf("schema=%q", doc.Schema))
	}
	if doc.Benchmark != "SWE-bench Verified" {
		problems = append(problems, fmt.Sprintf("benchmark=%q", doc.Benchmark))
	}
	for _, kv := range []struct {
		key string
		grp GroupSummary
	}{{"overall_sota", doc.OverallSOTA}, {"same_scaffold_sota", doc.SameScaffoldSOTA}} {
		top := kv.grp.Top
		if top == nil {
			problems = append(problems, kv.key+".top missing")
			continue
		}
		if !isNumber(top.ResolvedPct) {
			problems = append(problems, fmt.Sprintf("%s.top.resolved_pct=%v", kv.key, top.ResolvedPct))
		}
		if !truthyName(top.Name) {
			problems = append(problems, kv.key+".top.name missing")
		}
	}
	if len(doc.SameScaffoldSOTA.FocalRows) == 0 {
		problems = append(problems, "same_scaffold_sota.focal_rows missing")
	}
	return problems
}

func isNumber(v any) bool {
	switch v.(type) {
	case float64, int, int64:
		return true
	default:
		return false
	}
}

// truthyName mirrors the Python `if not top.get("name")` falsiness check: a nil
// or empty string is "missing".
func truthyName(v any) bool {
	if v == nil {
		return false
	}
	if s, ok := v.(string); ok {
		return s != ""
	}
	return true
}

// RenderMarkdown renders the house-style markdown report for a snapshot, matching
// tools/swebench_sota_snapshot.py::render_markdown.
func RenderMarkdown(doc Snapshot) string {
	lines := []string{
		"# SWE-bench SOTA Snapshot",
		"",
		fmt.Sprintf("- Generated: `%s`", doc.GeneratedAt),
		fmt.Sprintf("- Source: `%s`", doc.SourceURL),
		fmt.Sprintf("- Source SHA256: `%s`", doc.SourceSHA256),
		fmt.Sprintf("- Benchmark: `%s` (%d instances)", doc.Benchmark, doc.Instances),
		"",
		"| scope | top row | resolved | date |",
		"|---|---|---:|---|",
	}
	for _, kv := range []struct {
		grp   GroupSummary
		label string
	}{{doc.OverallSOTA, "overall"}, {doc.SameScaffoldSOTA, "same scaffold"}} {
		var name, resolved, date any
		if kv.grp.Top != nil {
			name, resolved, date = kv.grp.Top.Name, kv.grp.Top.ResolvedPct, kv.grp.Top.Date
		}
		lines = append(lines, fmt.Sprintf("| %s | `%s` | %s | %s |",
			kv.label, fmtAny(name), fmtAny(resolved), fmtAny(date)))
	}
	focal := doc.SameScaffoldSOTA.FocalRows
	if len(focal) > 0 {
		lines = append(lines, "", "## Focal Rows", "", "| rank | name | resolved | date |", "|---:|---|---:|---|")
		for _, row := range focal {
			lines = append(lines, fmt.Sprintf("| %d | `%s` | %s | %s |",
				row.Rank, fmtAny(row.Name), fmtAny(row.ResolvedPct), fmtAny(row.Date)))
		}
	}
	lines = append(lines, "", doc.ComparisonScope.Caveat, "")
	return strings.Join(lines, "\n")
}

// fmtAny renders a JSON-decoded value the way Python's f-string would: a nil is
// "None", a whole-number float drops its trailing ".0".
func fmtAny(v any) string {
	switch n := v.(type) {
	case nil:
		return "None"
	case string:
		return n
	case float64:
		if n == float64(int64(n)) {
			return fmt.Sprintf("%d", int64(n))
		}
		return fmt.Sprintf("%g", n)
	default:
		return fmt.Sprintf("%v", n)
	}
}

// Fetch retrieves the leaderboard page over HTTP with the tool's User-Agent and
// the given timeout, returning the decoded body. This is the impure half.
func Fetch(url string, timeout time.Duration) (string, error) {
	client := &http.Client{Timeout: timeout}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("fetch %s: HTTP %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}
