package issuesmallness

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

const (
	Schema = "fak-issue-smallness-lint/1"
	Pass   = "pass"
	Warn   = "warn"
	Fail   = "fail"
)

type Result struct {
	Verdict       string   `json:"verdict"`
	Count         int      `json:"count"`
	Items         []string `json:"items"`
	SectionSource string   `json:"section_source"`
	WitnessCount  int      `json:"witness_count"`
	WitnessItems  []string `json:"witness_items"`
	Reason        string   `json:"reason"`
}

type Issue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
}

type IssueReport struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Result
}

type OpenReport struct {
	Schema  string         `json:"schema"`
	Mode    string         `json:"mode"`
	Scanned int            `json:"scanned"`
	Counts  map[string]int `json:"counts"`
	Flagged []IssueReport  `json:"flagged"`
}

var (
	headingRE     = regexp.MustCompile(`(?m)^#{1,6}\s*(.+?)\s*$`)
	bulletRE      = regexp.MustCompile(`(?m)^\s*(?:[-*]|\d+[.)])\s+(.+)$`)
	leadingVerbRE = regexp.MustCompile(`(?i)^(add|fix|remove|rewrite|refactor|create|build|update|write|implement|replace|delete|migrate|design|extend|wire|document|test|lint|register|surface|expose|ship)\b`)
)

func LintBody(body string) Result {
	goal := ExtractSection(body, "goal")
	done := ExtractSection(body, "done condition", "done-condition", "acceptance")
	witness := ExtractSection(body, "witness")

	source := "body"
	section := body
	if goal != "" {
		source = "goal"
		section = goal
	} else if done != "" {
		source = "done"
		section = done
	}

	items := FindDeliverables(section)
	witnessItems := FindDeliverables(witness)
	count := len(items)
	witnessCount := len(witnessItems)

	verdict := Pass
	reasons := []string{}
	switch {
	case count >= 3:
		verdict = Fail
		reasons = append(reasons, fmt.Sprintf("%d distinct deliverables found in %s - split before dispatch", count, source))
	case count == 2:
		verdict = Warn
		reasons = append(reasons, fmt.Sprintf("2 distinct deliverables found in %s - confirm they are one unit of work", source))
	default:
		reasons = append(reasons, fmt.Sprintf("%d deliverable(s) found in %s", count, source))
	}
	if witnessCount != 1 {
		verdict = Fail
		reasons = append(reasons, fmt.Sprintf("%d witness item(s) found - exactly one witness is required", witnessCount))
	}

	return Result{
		Verdict:       verdict,
		Count:         count,
		Items:         items,
		SectionSource: source,
		WitnessCount:  witnessCount,
		WitnessItems:  witnessItems,
		Reason:        strings.Join(reasons, "; "),
	}
}

func ExtractSection(body string, headingSubstrings ...string) string {
	matches := headingRE.FindAllStringSubmatchIndex(body, -1)
	for i, m := range matches {
		title := strings.ToLower(body[m[2]:m[3]])
		found := false
		for _, sub := range headingSubstrings {
			if strings.Contains(title, strings.ToLower(sub)) {
				found = true
				break
			}
		}
		if !found {
			continue
		}
		start := m[1]
		end := len(body)
		if i+1 < len(matches) {
			end = matches[i+1][0]
		}
		return strings.TrimSpace(body[start:end])
	}
	return ""
}

func FindDeliverables(section string) []string {
	section = strings.TrimSpace(section)
	if section == "" {
		return nil
	}
	var bullets []string
	for _, m := range bulletRE.FindAllStringSubmatch(section, -1) {
		if len(m) > 1 {
			bullets = append(bullets, strings.TrimSpace(m[1]))
		}
	}
	if len(bullets) > 0 {
		return dedupeKeepOrder(bullets)
	}

	flat := flatten(section)
	if flat == "" {
		return nil
	}
	clauses := splitTaskClauses(flat)
	if len(clauses) <= 1 {
		return []string{flat}
	}
	var imperative []string
	for _, clause := range clauses {
		if leadingVerbRE.MatchString(clause) {
			imperative = append(imperative, clause)
		}
	}
	if len(imperative) >= 2 {
		return dedupeKeepOrder(imperative)
	}
	return []string{flat}
}

func ReportOpen(issues []Issue) OpenReport {
	counts := map[string]int{Pass: 0, Warn: 0, Fail: 0}
	flagged := []IssueReport{}
	for _, issue := range issues {
		res := LintBody(issue.Body)
		counts[res.Verdict]++
		row := IssueReport{Number: issue.Number, Title: issue.Title, Result: res}
		if res.Verdict != Pass {
			flagged = append(flagged, row)
		}
	}
	sort.Slice(flagged, func(i, j int) bool { return flagged[i].Number < flagged[j].Number })
	return OpenReport{Schema: Schema, Mode: "open", Scanned: len(issues), Counts: counts, Flagged: flagged}
}

func HasFailReport(r OpenReport) bool {
	return r.Counts[Fail] > 0
}

func dedupeKeepOrder(items []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, item := range items {
		item = strings.TrimSpace(item)
		key := strings.ToLower(item)
		if item == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, item)
	}
	return out
}

func flatten(s string) string {
	parts := []string{}
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			parts = append(parts, line)
		}
	}
	return strings.Join(parts, " ")
}

func splitTaskClauses(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ";") {
		for _, clause := range splitAndThen(part) {
			clause = strings.TrimSpace(clause)
			clause = strings.TrimPrefix(clause, ",")
			clause = strings.TrimSpace(clause)
			if clause != "" {
				out = append(out, clause)
			}
		}
	}
	return out
}

func splitAndThen(s string) []string {
	lower := strings.ToLower(s)
	needle := "and then"
	idx := strings.Index(lower, needle)
	if idx < 0 {
		return []string{s}
	}
	head := strings.TrimSpace(s[:idx])
	tail := strings.TrimSpace(s[idx+len(needle):])
	if strings.HasSuffix(head, ",") {
		head = strings.TrimSpace(strings.TrimSuffix(head, ","))
	}
	if head == "" {
		return []string{tail}
	}
	if tail == "" {
		return []string{head}
	}
	return []string{head, tail}
}
