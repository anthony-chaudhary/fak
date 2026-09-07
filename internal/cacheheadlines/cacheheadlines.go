// Package cacheheadlines enforces the cache-headline gate: a cache "win" headline
// must name its plane and provenance.
//
// A cache headline — "99% cache", "cache win", "cache is 99% of the story" — that omits
// which PLANE (provider / kernel / context / forecast) and provenance the number belongs to
// fails review.
package cacheheadlines

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// DefaultFix is the canonical remediation message for a cache headline violation.
const DefaultFix = "name the plane + provenance the number belongs to — provider prompt-cache (OBSERVED rebate), fak kernel KV reuse (WITNESSED), or O(1) context (WITNESSED/FORECAST) — never a blended \"99% cache\" / \"cache win\""

// Finding represents one cache headline violation in a file.
type Finding struct {
	File string `json:"file"`
	Line int    `json:"line"`
	Text string `json:"text"`
	Fix  string `json:"fix"`
}

// HeadlinePatterns defines known legacy cache-headline shapes.
// A line is a candidate if it matches one of these, and violates if it lacks a co-located label.
var HeadlinePatterns = []*regexp.Regexp{
	// a bare "cache win" / "cache-win" / "cache wins" headline
	regexp.MustCompile(`(?i)\bcache[\s-]+wins?\b`),
	// a percentage directly qualifying "cache" as a blended win: "99% cache"
	regexp.MustCompile(`(?i)\b\d{2,3}\s*%\s*cache\b`),
	// the legacy "(provider) cache is 99% of the story" phrasing
	regexp.MustCompile(`(?i)\bcache\s+is\s+\d{1,3}\s*%`),
	regexp.MustCompile(`(?i)\b\d{1,3}\s*%\s+of\s+the\s+story\b`),
}

// LabelRE matches plane keywords and provenance verbs that make a headline honest.
// Plane keywords: provider, kernel, context, forecast
// Provenance keywords: observed, witnessed, forecast, decision, measured, modeled, hypothesis
var LabelRE = regexp.MustCompile(
	`(?i)(?:` +
		`\b(?:` +
		`provider|kernel|context|forecast(?:ed)?|` +
		`observed|witnessed|decision|measured|modeled|hypothesis|` +
		`provenance|per-plane|per-mechanism|plane|` +
		`vcache|v-cache|radixkv|radix|paged|prefix|` +
		`radixattention|pagedattention|sglang|vllm|llama|` +
		`prompt-?cache|engine|rebate|cost[\s/-]*latency|` +
		`owner|attribution|reuse|kv` +
		`)\b` +
		`|\bo\(1\)(?:[^\w]|$)` +
		`)`,
)

// AllowPatterns defines carve-outs where legacy phrasing is allowed (e.g. meta-critique / removal).
var AllowPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\blegacy\b`),
	regexp.MustCompile(`(?i)\bremove\b`),
	regexp.MustCompile(`(?i)\bmislead`),
	regexp.MustCompile(`(?i)\bhide[sn]?\b`),
	regexp.MustCompile(`(?i)\bblend(?:ed|s)?\b`),
	regexp.MustCompile(`(?i)\bfails? review\b`),
	regexp.MustCompile(`(?i)\bomits?\b`),
}

// QuotedHeadlineRE matches headline phrases quoted in text (e.g. "cache win", '99% cache', `cache wins`),
// representing quoted references rather than direct assertions.
var QuotedHeadlineRE = regexp.MustCompile(`(?i)["'\x60][^"'\x60\r\n]*\b(?:cache[\s-]+wins?|\d{2,3}\s*%\s*cache|cache\s+is\s+\d{1,3}\s*%|\d{1,3}\s*%\s+of\s+the\s+story)\b[^"'\x60\r\n]*["'\x60]`)

// Aliases matching uppercase names for parity with Python / prompt declarations.
var (
	HEADLINE_PATTERNS  = HeadlinePatterns
	LABEL_RE           = LabelRE
	ALLOW_PATTERNS     = AllowPatterns
	QUOTED_HEADLINE_RE = QuotedHeadlineRE
	FIX                = DefaultFix
)

// ScanGlobs specifies file extensions to scan.
var ScanGlobs = []string{
	"*.md", "*.html", "*.txt",
}

// SkipPrefixes specifies directory prefixes excluded from scanning.
var SkipPrefixes = []string{
	"docs/releases/", // dated history, immutable
	"vendor/", "node_modules/",
}

// SkipBasenames specifies filenames excluded from scanning.
var SkipBasenames = []string{
	"llms-full.txt", // generated mirror (regenerates from llms.txt + docs)
	"check_cache_headlines.py",
	"check_cache_headlines_test.py",
	"cacheheadlines.go",
	"cacheheadlines_test.go",
	"check_cache_headlines.go",
	"check_cache_headlines_test.go",
}

// ShouldSkip reports whether relpath should be skipped based on prefixes and basenames.
func ShouldSkip(relpath string) bool {
	normalized := filepath.ToSlash(relpath)
	base := filepath.Base(normalized)
	for _, pre := range SkipPrefixes {
		if strings.HasPrefix(normalized, pre) {
			return true
		}
	}
	for _, b := range SkipBasenames {
		if base == b {
			return true
		}
	}
	return false
}

// MatchesScanGlob reports whether relpath matches any of the scan globs.
func MatchesScanGlob(relpath string) bool {
	base := filepath.Base(relpath)
	for _, g := range ScanGlobs {
		if strings.HasPrefix(g, "*") {
			if strings.HasSuffix(base, g[1:]) {
				return true
			}
		} else if base == g {
			return true
		}
	}
	return false
}

// CheckLine checks a single line. It returns isViolation=true and the matched headline if the line violates the gate.
func CheckLine(line string) (isViolation bool, matchedHeadline string) {
	line = strings.TrimRight(line, "\r\n")
	var matched string
	for _, p := range HeadlinePatterns {
		if loc := p.FindString(line); loc != "" {
			matched = loc
			break
		}
	}
	if matched == "" {
		return false, ""
	}
	if LabelRE.MatchString(line) {
		return false, ""
	}
	for _, p := range AllowPatterns {
		if p.MatchString(line) {
			return false, ""
		}
	}
	if QuotedHeadlineRE != nil && QuotedHeadlineRE.MatchString(line) {
		stripped := QuotedHeadlineRE.ReplaceAllString(line, " ")
		hasUnquoted := false
		for _, p := range HeadlinePatterns {
			if p.MatchString(stripped) {
				hasUnquoted = true
				break
			}
		}
		if !hasUnquoted {
			return false, ""
		}
	}
	return true, matched
}

// ScanText scans the given text (e.g. file content) line by line and returns findings.
// Line numbers are 1-based.
func ScanText(text string, filename string) []Finding {
	var findings []Finding
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if bad, _ := CheckLine(line); bad {
			trimmed := strings.TrimSpace(line)
			if len(trimmed) > 160 {
				trimmed = trimmed[:160]
			}
			findings = append(findings, Finding{
				File: filename,
				Line: i + 1,
				Text: trimmed,
				Fix:  DefaultFix,
			})
		}
	}
	return findings
}

// AuditTree scans the tracked tree under root using git ls-files for cache headline violations.
func AuditTree(root string) ([]Finding, error) {
	absRoot, err := filepath.Abs(root)
	if err == nil {
		root = absRoot
	}
	cmd := exec.Command("git", "ls-files", "-z", "*.md", "*.html", "*.txt")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files: %w", err)
	}

	var findings []Finding
	parts := bytes.Split(out, []byte{0})
	for _, part := range parts {
		if len(part) == 0 {
			continue
		}
		relpath := filepath.ToSlash(string(part))
		if ShouldSkip(relpath) || !MatchesScanGlob(relpath) {
			continue
		}
		full := filepath.Join(root, filepath.FromSlash(relpath))
		data, err := os.ReadFile(full)
		if err != nil {
			continue
		}
		fileFindings := ScanText(string(data), relpath)
		findings = append(findings, fileFindings...)
	}
	return findings, nil
}

type stagedLine struct {
	lineNo int
	text   string
}

var hunkRE = regexp.MustCompile(`^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@`)

func stagedAddedLines(root string, relpath string) ([]stagedLine, error) {
	cmd := exec.Command("git", "diff", "--cached", "--unified=0", "--no-ext-diff", "--", relpath)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git diff --cached on %s: %w", relpath, err)
	}
	var lines []stagedLine
	newLine := -1
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		raw := scanner.Text()
		if m := hunkRE.FindStringSubmatch(raw); m != nil {
			n, err := strconv.Atoi(m[1])
			if err == nil {
				newLine = n
			}
			continue
		}
		if newLine < 0 {
			continue
		}
		if strings.HasPrefix(raw, "+++") {
			continue
		}
		if strings.HasPrefix(raw, "+") {
			lines = append(lines, stagedLine{lineNo: newLine, text: raw[1:]})
			newLine++
			continue
		}
		if strings.HasPrefix(raw, "-") {
			continue
		}
		if strings.HasPrefix(raw, " ") {
			newLine++
		}
	}
	return lines, scanner.Err()
}

// AuditStaged scans staged additions from the git index under root for cache headline violations.
func AuditStaged(root string) ([]Finding, error) {
	absRoot, err := filepath.Abs(root)
	if err == nil {
		root = absRoot
	}
	cmd := exec.Command("git", "diff", "--cached", "--name-only", "--diff-filter=AM", "-z")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git diff --cached --name-only: %w", err)
	}

	var findings []Finding
	parts := bytes.Split(out, []byte{0})
	for _, part := range parts {
		if len(part) == 0 {
			continue
		}
		relpath := filepath.ToSlash(string(part))
		if ShouldSkip(relpath) || !MatchesScanGlob(relpath) {
			continue
		}
		addedLines, err := stagedAddedLines(root, relpath)
		if err != nil {
			return nil, err
		}
		for _, sl := range addedLines {
			if bad, _ := CheckLine(sl.text); bad {
				trimmed := strings.TrimSpace(sl.text)
				if len(trimmed) > 160 {
					trimmed = trimmed[:160]
				}
				findings = append(findings, Finding{
					File: relpath,
					Line: sl.lineNo,
					Text: trimmed,
					Fix:  DefaultFix,
				})
			}
		}
	}
	return findings, nil
}
