package issuepolicy

// issuedraft.go — the issue-DRAFT parsing seam of the candidate contract, split out of
// contract.go (#5849) so that file stays under the god-file ceiling. Everything here reads
// a raw markdown issue body: the fenced ```routing block, the heading index, and the
// label/notes/paths sections CandidateFromIssueDraft recovers from prose. The candidate
// scoring, dispatch classification, and required-section checks stay in contract.go.

import (
	"strconv"
	"strings"
)

// routingBlock is the parsed content of a fenced ```routing key:value block.
// The block is the machine-authored, unambiguous form of the routing metadata
// CandidateFromIssueDraft otherwise recovers from prose sections. Each *Set flag
// records whether that key was PRESENT with a usable value, so a caller can
// prefer the block per key and fall back to the prose section parse only for the
// keys the block omits.
type routingBlock struct {
	lane          string
	laneSet       bool
	paths         []string
	pathsSet      bool
	expectedSteps int
	stepsSet      bool
}

// parseRoutingBlock scans body for the first fenced ```routing key:value block
// and returns its parsed routing fields plus whether such a block was found at
// all. Recognized keys: lane, paths (comma/space separated, each run through the
// same cleanPathHint as the prose path parse), and expected_steps (a positive
// int). A key the block omits — or leaves empty / unparseable — stays unset so
// the caller keeps the prose fallback for that key.
func parseRoutingBlock(body string) (routingBlock, bool) {
	var rb routingBlock
	inBlock := false
	found := false
	for _, raw := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		if !inBlock {
			if routingFenceOpenRE.MatchString(raw) {
				inBlock = true
				found = true
			}
			continue
		}
		if routingFenceCloseRE.MatchString(raw) {
			break
		}
		key, val, ok := strings.Cut(raw, ":")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(strings.Trim(key, "`*_-# ")))
		key = strings.ReplaceAll(key, " ", "_")
		val = strings.TrimSpace(val)
		switch key {
		case "lane":
			if lane := strings.TrimSpace(strings.Trim(val, "`")); lane != "" {
				rb.lane = lane
				rb.laneSet = true
			}
		case "paths", "path":
			var paths []string
			for _, tok := range strings.FieldsFunc(val, func(r rune) bool {
				return r == ',' || r == ' ' || r == '\t'
			}) {
				if p := cleanPathHint(tok); p != "" {
					paths = append(paths, p)
				}
			}
			if len(paths) > 0 {
				rb.paths = compact(paths)
				rb.pathsSet = true
			}
		case "expected_steps":
			if n := parseExpectedSteps(val); n > 0 {
				rb.expectedSteps = n
				rb.stepsSet = true
			}
		}
	}
	return rb, found
}

func prefixedSectionValue(section, prefix string) string {
	prefix = strings.ToLower(strings.TrimSpace(prefix)) + ":"
	for _, line := range strings.Split(section, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToLower(line), prefix) {
			return strings.TrimSpace(strings.Trim(line[len(prefix):], "` "))
		}
	}
	return ""
}

func issueDraftKey(d IssueDraft) string {
	if key := issueDraftMarkerKey(d.Body); key != "" {
		return key
	}
	if d.Number > 0 {
		return "issue/" + strconv.Itoa(d.Number)
	}
	slug := slugKeyPart(d.Title)
	if slug == "" {
		slug = "unknown"
	}
	return "manual/" + slug
}

func issueDraftMarkerKey(body string) string {
	m := issueMarkerKeyRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

func markdownSections(body string) map[string]string {
	out := map[string]string{}
	var current string
	var buf []string
	flush := func() {
		if current == "" {
			return
		}
		out[current] = strings.TrimSpace(strings.Join(buf, "\n"))
	}
	for _, raw := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		line := strings.TrimSpace(raw)
		if m := markdownHeadingRE.FindStringSubmatch(line); m != nil {
			flush()
			current = normalizeHeading(m[1])
			buf = nil
			continue
		}
		if current != "" {
			buf = append(buf, raw)
		}
	}
	flush()
	return out
}

func normalizeHeading(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "`*_:# ")
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

func issueDraftLabels(labels []IssueLabel) []string {
	out := make([]string, 0, len(labels))
	for _, label := range labels {
		out = append(out, label.Name)
	}
	return compact(out)
}

func issueDraftNotes(section string) []string {
	var out []string
	for _, line := range strings.Split(section, "\n") {
		line = trimListPrefix(line)
		if line != "" && !strings.EqualFold(line, "none") {
			out = append(out, line)
		}
	}
	return compact(out)
}

func issueDraftAgentNotes(section string) []string {
	notes := issueDraftNotes(section)
	out := notes[:0]
	for _, note := range notes {
		if agentSectionValue(note) != "" {
			out = append(out, note)
		}
	}
	return compact(out)
}

func agentSectionValue(s string) string {
	s = strings.TrimSpace(s)
	switch strings.ToLower(s) {
	case "", "not specified.", "not specified", "none named.", "none named", "no special coordination beyond the lane lease.":
		return ""
	default:
		return s
	}
}

func issueDraftPaths(section string) []string {
	var out []string
	for _, m := range codeSpanRE.FindAllStringSubmatch(section, -1) {
		if path := cleanPathHint(m[1]); path != "" {
			out = append(out, path)
		}
	}
	for _, line := range strings.Split(section, "\n") {
		line = trimListPrefix(line)
		if path := cleanPathHint(line); path != "" {
			out = append(out, path)
		}
	}
	return compact(out)
}
