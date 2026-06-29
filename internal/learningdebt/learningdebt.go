package learningdebt

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	Schema          = "fak.learning-debt-dispatch.v1"
	SeenSchema      = "fak.learning-debt-dispatch.seen.v1"
	DefaultCacheRel = ".fak/learning-debt-dispatch/seen.json"
)

var markerRE = regexp.MustCompile(`<!--\s*fak-learning-debt-key:\s*([^>\s]+)\s*-->`)

type Defect struct {
	Key    string  `json:"key"`
	Doc    string  `json:"doc"`
	Class  string  `json:"class"`
	Exact  string  `json:"exact"`
	Source string  `json:"source"`
	Score  string  `json:"score,omitempty"`
	Grade  string  `json:"grade,omitempty"`
	Rank   int     `json:"rank"`
	Prio   float64 `json:"priority"`
}

type SeenRecord struct {
	FiledAt  string `json:"filed_at"`
	Doc      string `json:"doc"`
	Class    string `json:"class"`
	Exact    string `json:"exact"`
	IssueURL string `json:"issue_url,omitempty"`
}

type SeenCache struct {
	Schema string                `json:"schema"`
	Seen   map[string]SeenRecord `json:"seen"`
}

type Issue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
	State  string `json:"state"`
	URL    string `json:"url"`
}

type PlanRow struct {
	Action string `json:"action"`
	Key    string `json:"key"`
	Title  string `json:"title"`
	Body   string `json:"-"`
	Doc    string `json:"doc"`
	Class  string `json:"class"`
	Exact  string `json:"exact"`
	Source string `json:"source"`
	Score  string `json:"score,omitempty"`
	Grade  string `json:"grade,omitempty"`
}

type Stats struct {
	TotalDefects     int `json:"total_defects"`
	Planned          int `json:"planned"`
	SkippedSeen      int `json:"skipped_seen"`
	SkippedIssueBody int `json:"skipped_issue_body"`
	SkippedWithinRun int `json:"skipped_within_run"`
	SkippedCap       int `json:"skipped_cap"`
	Cap              int `json:"cap"`
}

type SyncRow struct {
	Key    string `json:"key"`
	OK     bool   `json:"ok"`
	Stdout string `json:"stdout,omitempty"`
	Stderr string `json:"stderr,omitempty"`
}

type Result struct {
	Schema    string    `json:"schema"`
	Mode      string    `json:"mode"`
	Scorecard string    `json:"scorecard"`
	Cache     string    `json:"cache"`
	Stats     Stats     `json:"stats"`
	Planned   []PlanRow `json:"planned"`
	Synced    []SyncRow `json:"synced"`
}

type Runner func(args []string) (stdout, stderr string, ok bool)

func LoadPayload(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var data any
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, err
	}
	m, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("scorecard payload must be a JSON object")
	}
	return m, nil
}

func LoadSeen(path string) (SeenCache, error) {
	cache := SeenCache{Schema: SeenSchema, Seen: map[string]SeenRecord{}}
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return cache, nil
	}
	if err != nil {
		return cache, err
	}
	if strings.TrimSpace(string(b)) == "" {
		return cache, nil
	}
	if err := json.Unmarshal(b, &cache); err != nil {
		return cache, err
	}
	if cache.Schema == "" {
		cache.Schema = SeenSchema
	}
	if cache.Seen == nil {
		cache.Seen = map[string]SeenRecord{}
	}
	return cache, nil
}

func SaveSeen(path string, cache SeenCache) error {
	if cache.Schema == "" {
		cache.Schema = SeenSchema
	}
	if cache.Seen == nil {
		cache.Seen = map[string]SeenRecord{}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(cache, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func ExtractDefects(payload map[string]any) []Defect {
	priorities := priorityIndex(payload)
	var defects []Defect

	for _, raw := range asSlice(payload["docs"]) {
		doc := asMap(raw)
		if doc == nil {
			continue
		}
		path := toStr(doc["path"], "")
		if path == "" {
			continue
		}
		rank, prio := 9999, 0.0
		if p, ok := priorities[path]; ok {
			rank, prio = p.rank, p.prio
		}
		for _, d := range asSlice(doc["defects"]) {
			exact := strings.TrimSpace(toStr(d, ""))
			if exact == "" {
				continue
			}
			class := defectClass(exact)
			defects = append(defects, newDefect(path, class, exact, "docs",
				toStr(doc["score"], ""), toStr(doc["grade"], ""), rank, prio))
		}
	}

	coverage := asMap(payload["coverage"])
	for _, d := range asSlice(coverage["defects"]) {
		exact := strings.TrimSpace(toStr(d, ""))
		if exact == "" {
			continue
		}
		class := defectClass(exact)
		doc := coverageSubject(class, exact)
		defects = append(defects, newDefect(doc, class, exact, "coverage", "", "", 9999, 0))
	}

	stamp := asMap(payload["stamp_freshness"])
	if toBool(stamp["stale_stamp"]) {
		exact := strings.TrimSpace(toStr(stamp["reason"], ""))
		if exact == "" {
			exact = "stale-stamp"
		}
		class := strings.TrimSpace(toStr(stamp["flag"], "stale-stamp"))
		if class == "" {
			class = "stale-stamp"
		}
		doc := strings.TrimSpace(toStr(stamp["doc"], "docs/LEARNING-SCORECARD.md"))
		defects = append(defects, newDefect(doc, class, exact, "stamp_freshness", "", "", 9998, 0))
	}

	sort.SliceStable(defects, func(i, j int) bool {
		a, b := defects[i], defects[j]
		if a.Rank != b.Rank {
			return a.Rank < b.Rank
		}
		if a.Prio != b.Prio {
			return a.Prio > b.Prio
		}
		if a.Doc != b.Doc {
			return a.Doc < b.Doc
		}
		if a.Class != b.Class {
			return a.Class < b.Class
		}
		return a.Exact < b.Exact
	})
	return defects
}

func BuildPlan(defects []Defect, seen SeenCache, existing []Issue, cap int, scorecardPath string) ([]PlanRow, Stats) {
	if cap < 0 {
		cap = 0
	}
	stats := Stats{TotalDefects: len(defects), Cap: cap}
	existingKeys := existingByKey(existing)
	within := map[string]bool{}
	plan := make([]PlanRow, 0, min(cap, len(defects)))
	for _, d := range defects {
		if seen.Seen != nil {
			if _, ok := seen.Seen[d.Key]; ok {
				stats.SkippedSeen++
				continue
			}
		}
		if _, ok := existingKeys[d.Key]; ok {
			stats.SkippedIssueBody++
			continue
		}
		if within[d.Key] {
			stats.SkippedWithinRun++
			continue
		}
		within[d.Key] = true
		if len(plan) >= cap {
			stats.SkippedCap++
			continue
		}
		plan = append(plan, PlanRow{
			Action: "create",
			Key:    d.Key,
			Title:  issueTitle(d),
			Body:   IssueBody(d, scorecardPath),
			Doc:    d.Doc,
			Class:  d.Class,
			Exact:  d.Exact,
			Source: d.Source,
			Score:  d.Score,
			Grade:  d.Grade,
		})
	}
	stats.Planned = len(plan)
	return plan, stats
}

func IssueBody(d Defect, scorecardPath string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-learning-debt-key: %s -->\n", d.Key)
	b.WriteString("# Learning-debt triage\n\n")
	fmt.Fprintf(&b, "- Stable key: `%s`\n", d.Key)
	fmt.Fprintf(&b, "- Scorecard JSON: `%s`\n", scorecardPath)
	fmt.Fprintf(&b, "- Source: `%s`\n", d.Source)
	fmt.Fprintf(&b, "- Doc/topic: `%s`\n", d.Doc)
	fmt.Fprintf(&b, "- Defect class: `%s`\n", d.Class)
	if d.Score != "" {
		fmt.Fprintf(&b, "- Doc score: `%s`\n", d.Score)
	}
	if d.Grade != "" {
		fmt.Fprintf(&b, "- Doc grade: `%s`\n", d.Grade)
	}
	b.WriteString("\nExact scorecard defect:\n\n```text\n")
	b.WriteString(d.Exact)
	b.WriteString("\n```\n\n")
	b.WriteString("Suggested next action:\n\n")
	b.WriteString("Retire this HARD learning-debt defect in the cited doc/topic, then rerun ")
	b.WriteString("`python tools/learning_scorecard.py --json` and this dispatcher.\n\n")
	b.WriteString("Managed by `fak learning-debt-dispatch`; the HTML marker above is the dedup key.\n")
	return b.String()
}

func MarkerKey(body string) string {
	m := markerRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

func Sync(plan []PlanRow, repo string, labels []string, runner Runner) []SyncRow {
	run := runner
	if run == nil {
		run = defaultRunner
	}
	rows := make([]SyncRow, 0, len(plan))
	for _, row := range plan {
		args := []string{"issue", "create", "--title", row.Title, "--body", row.Body}
		for _, label := range labels {
			if strings.TrimSpace(label) != "" {
				args = append(args, "--label", label)
			}
		}
		if repo != "" {
			args = append(args, "--repo", repo)
		}
		stdout, stderr, ok := run(args)
		rows = append(rows, SyncRow{
			Key:    row.Key,
			OK:     ok,
			Stdout: strings.TrimSpace(stdout),
			Stderr: strings.TrimSpace(stderr),
		})
	}
	return rows
}

func MarkSuccessful(cache *SeenCache, plan []PlanRow, synced []SyncRow, now time.Time) {
	if cache.Schema == "" {
		cache.Schema = SeenSchema
	}
	if cache.Seen == nil {
		cache.Seen = map[string]SeenRecord{}
	}
	byKey := map[string]PlanRow{}
	for _, row := range plan {
		byKey[row.Key] = row
	}
	for _, row := range synced {
		if !row.OK {
			continue
		}
		planned, ok := byKey[row.Key]
		if !ok {
			continue
		}
		cache.Seen[row.Key] = SeenRecord{
			FiledAt:  now.UTC().Format(time.RFC3339),
			Doc:      planned.Doc,
			Class:    planned.Class,
			Exact:    planned.Exact,
			IssueURL: firstLine(row.Stdout),
		}
	}
}

func FetchExistingIssues(repo string, limit int) ([]Issue, error) {
	args := []string{"issue", "list", "--state", "all", "--limit", strconv.Itoa(limit),
		"--json", "number,title,body,state,url"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, stderr, ok := defaultRunner(args)
	if !ok {
		return nil, fmt.Errorf("gh %s: %s", strings.Join(args, " "), strings.TrimSpace(stderr))
	}
	if strings.TrimSpace(stdout) == "" {
		return nil, nil
	}
	var issues []Issue
	if err := json.Unmarshal([]byte(stdout), &issues); err != nil {
		return nil, err
	}
	return issues, nil
}

func Render(r Result) string {
	lines := []string{
		fmt.Sprintf("learning-debt-dispatch: %s  planned=%d cap=%d total_defects=%d",
			r.Mode, len(r.Planned), r.Stats.Cap, r.Stats.TotalDefects),
		fmt.Sprintf("  scorecard: %s", r.Scorecard),
		fmt.Sprintf("  seen-cache: %s", r.Cache),
	}
	if len(r.Planned) == 0 {
		lines = append(lines, "  no new learning-debt issues to file")
	} else {
		for _, row := range r.Planned {
			lines = append(lines, fmt.Sprintf("  [create] %s  doc=%s class=%s",
				row.Title, row.Doc, row.Class))
		}
	}
	lines = append(lines, fmt.Sprintf("  dedup: seen=%d issue-body=%d within-run=%d cap-skipped=%d",
		r.Stats.SkippedSeen, r.Stats.SkippedIssueBody, r.Stats.SkippedWithinRun, r.Stats.SkippedCap))
	if r.Mode == "dry-run" {
		lines = append(lines, "  dry-run: pass --live to create issues and update the seen-cache")
	}
	return strings.Join(lines, "\n")
}

func defaultRunner(args []string) (string, string, bool) {
	cmd := exec.Command("gh", args...)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err == nil
}

func existingByKey(issues []Issue) map[string]Issue {
	out := map[string]Issue{}
	for _, issue := range issues {
		if key := MarkerKey(issue.Body); key != "" {
			out[key] = issue
		}
	}
	return out
}

func priorityIndex(payload map[string]any) map[string]struct {
	rank int
	prio float64
} {
	out := map[string]struct {
		rank int
		prio float64
	}{}
	corpus := asMap(payload["corpus"])
	for i, raw := range asSlice(corpus["priorities"]) {
		p := asMap(raw)
		path := toStr(p["path"], "")
		if path == "" {
			continue
		}
		out[path] = struct {
			rank int
			prio float64
		}{rank: i, prio: toFloat(p["priority"], 0)}
	}
	return out
}

func newDefect(doc, class, exact, source, score, grade string, rank int, prio float64) Defect {
	if rank == 0 && prio == 0 {
		rank = 9999
	}
	d := Defect{
		Doc:    doc,
		Class:  class,
		Exact:  exact,
		Source: source,
		Score:  score,
		Grade:  grade,
		Rank:   rank,
		Prio:   prio,
	}
	d.Key = stableKey(d)
	return d
}

func stableKey(d Defect) string {
	sum := sha256.Sum256([]byte(d.Doc + "\x00" + d.Class + "\x00" + d.Exact))
	return "learning-debt/" + slug(d.Class) + "/" + hex.EncodeToString(sum[:])[:16]
}

func defectClass(exact string) string {
	before, _, ok := strings.Cut(exact, ":")
	if ok && strings.TrimSpace(before) != "" {
		return strings.TrimSpace(before)
	}
	return "unknown"
}

func coverageSubject(class, exact string) string {
	_, after, ok := strings.Cut(exact, ":")
	if !ok || strings.TrimSpace(after) == "" {
		return "coverage"
	}
	subject := strings.TrimSpace(after)
	if strings.HasPrefix(class, "uncovered learning topic") {
		return "topic:" + subject
	}
	return subject
}

func issueTitle(d Defect) string {
	title := "learning-debt: " + d.Doc + " [" + d.Class + "]"
	if len(title) <= 120 {
		return title
	}
	return strings.TrimSpace(title[:117]) + "..."
}

func slug(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "defect"
	}
	return out
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}

func asMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return nil
}

func asSlice(v any) []any {
	if xs, ok := v.([]any); ok {
		return xs
	}
	return nil
}

func toStr(v any, def string) string {
	if v == nil {
		return def
	}
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case bool:
		if x {
			return "true"
		}
		return "false"
	case json.Number:
		return x.String()
	default:
		return fmt.Sprintf("%v", x)
	}
}

func toFloat(v any, def float64) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case json.Number:
		f, err := x.Float64()
		if err == nil {
			return f
		}
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(x), 64)
		if err == nil {
			return f
		}
	}
	return def
}

func toBool(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		return strings.EqualFold(strings.TrimSpace(x), "true")
	default:
		return false
	}
}
