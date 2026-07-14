package docfreshrsi

import (
	"regexp"
	"strings"
)

// VersionClaim is a live, version-dependent assertion that lacks a freshness
// pointer. Historical sections are excluded because their assertions are
// intentionally snapshots rather than current operational guidance.
type VersionClaim struct {
	Path      string `json:"path"`
	Line      int    `json:"line"`
	Signature string `json:"signature"`
	Text      string `json:"text"`
}

type claimSignature struct {
	name string
	re   *regexp.Regexp
}

var versionClaimSignatures = []claimSignature{
	{"current/latest", regexp.MustCompile(`(?i)\b(current|currently|latest|newest)\b`)},
	{"version floor", regexp.MustCompile(`(?i)\b(requires?|needs?|supports?)\s+(go|python|node|cuda|codex|claude)(?:\s+version)?\s+v?\d`)},
	{"version comparison", regexp.MustCompile(`(?i)\b(go|python|node|cuda|codex|claude)\s+v?\d+(?:\.\d+)*(?:\+|\s+or\s+(?:newer|later)|\s+and\s+(?:newer|later))`)},
	{"release assertion", regexp.MustCompile(`(?i)\b(shipped|released|available|deprecated|removed)\s+(?:in|since|as of)\s+v?\d`)},
}

var freshnessPointers = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bas\s+of\s+\d{4}-\d{2}-\d{2}\b`),
	regexp.MustCompile(`(?i)\blast\s+verified\s*[:@]\s*\d{4}-\d{2}-\d{2}\b`),
	regexp.MustCompile(`(?i)\bverified\s+\d{4}-\d{2}-\d{2}\b`),
	regexp.MustCompile(`(?i)\b(?:source|freshness)\s*:\s*https?://\S+`),
	regexp.MustCompile(`(?i)\b(?:source|freshness)\s*:\s*[^\s]+@[0-9a-f]{7,40}\b`),
	regexp.MustCompile(`(?i)\bhttps?://\S+`),
	regexp.MustCompile(`(?i)\b[^\s]+@[0-9a-f]{7,40}\b`),
}

// ScanVersionClaims returns every unpointed live version assertion in stable
// line order. Markdown fenced code and explicitly historical sections are
// skipped; a freshness pointer may be on the claim line or the immediately
// following non-empty line.
func ScanVersionClaims(path, text string) []VersionClaim {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	var out []VersionClaim
	inFence := false
	historicalDepth := 0
	for i, raw := range lines {
		trimmed := strings.TrimSpace(raw)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			continue
		}
		if level, title, ok := markdownHeading(trimmed); ok {
			if historicalDepth > 0 && level <= historicalDepth {
				historicalDepth = 0
			}
			if historicalHeading(title) {
				historicalDepth = level
			}
			continue
		}
		if inFence || historicalDepth > 0 || trimmed == "" {
			continue
		}
		for _, sig := range versionClaimSignatures {
			loc := sig.re.FindStringIndex(trimmed)
			if loc == nil {
				continue
			}
			if freshnessPointed(trimmed[loc[1]:]) || nextLinePoints(lines, i+1) {
				break
			}
			out = append(out, VersionClaim{Path: path, Line: i + 1, Signature: sig.name, Text: trimmed})
			break
		}
	}
	return out
}

func freshnessPointed(line string) bool {
	for _, re := range freshnessPointers {
		if re.MatchString(line) {
			return true
		}
	}
	return false
}

func nextLinePoints(lines []string, start int) bool {
	if start >= len(lines) {
		return false
	}
	line := strings.TrimSpace(lines[start])
	return line != "" && freshnessPointed(line)
}

func markdownHeading(line string) (int, string, bool) {
	level := 0
	for level < len(line) && line[level] == '#' {
		level++
	}
	if level == 0 || level > 6 || level >= len(line) || line[level] != ' ' {
		return 0, "", false
	}
	return level, strings.TrimSpace(line[level+1:]), true
}

func historicalHeading(title string) bool {
	title = strings.ToLower(title)
	return strings.Contains(title, "historical") || strings.Contains(title, "history") ||
		strings.Contains(title, "archive") || strings.Contains(title, "changelog")
}
