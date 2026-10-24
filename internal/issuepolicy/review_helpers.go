package issuepolicy

import (
	"regexp"
	"sort"
	"strings"
)

func containsPrivateBoundary(c Candidate) bool {
	fields := []string{
		c.Title, c.ParentRef, c.CurrentState, c.WhyNow, c.WorkingSpine,
		c.PriorityContext, c.InScope, c.OutOfScope, c.DoneCondition, c.Witness, c.AcceptanceGate,
	}
	fields = append(fields, c.Paths...)
	fields = append(fields, c.BoundaryNotes...)
	for _, path := range c.Paths {
		if strings.Contains(strings.ToLower(path), "fak-private") {
			return true
		}
	}
	for _, field := range fields {
		if containsPrivateCompanionResource(field) {
			return true
		}
	}
	text := strings.ToLower(strings.Join(fields, "\n"))
	for _, needle := range []string{
		"slack control",
		"slack-control",
		"gpu-server reservation",
		"operator-only evidence",
		"credential dump",
		"secret key",
		"api key",
		"private token",
	} {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

var companionIssueReferenceRE = regexp.MustCompile(`^(?:fak-private#|anthony-chaudhary/fak-private#|https://github\.com/anthony-chaudhary/fak-private/issues/)[1-9][0-9]*$`)
var companionIssueLinkRE = regexp.MustCompile(`\[([^\[\]\r\n]*)\]\(\s*https://github\.com/anthony-chaudhary/fak-private/issues/[1-9][0-9]*\s*\)`)
var companionIssueDefinitionRE = regexp.MustCompile(`(?m)^[ \t]{0,3}\[([^\[\]\r\n]*)\]:[ \t]*https://github\.com/anthony-chaudhary/fak-private/issues/[1-9][0-9]*[ \t]*\r?$`)
var companionDestinationSyntaxRE = regexp.MustCompile(`\]\s*[(:]`)
var companionHTMLAttributeRE = regexp.MustCompile(`(?s)<\s*[a-z][^>]*=`)

func containsPrivateCompanionResource(field string) bool {
	field = strings.ToLower(field)
	if !strings.Contains(field, "fak-private") {
		return false
	}
	// Raw HTML destinations are outside the supported canonical Markdown
	// grammar. Refuse before whitespace can turn an href into a prose reference.
	if companionHTMLAttributeRE.MatchString(field) {
		return true
	}
	// A repository name next to an unqualified path is still ambiguous. Keep
	// this check field-local so an unrelated public Candidate.Paths entry does
	// not turn a harmless coordination note into a private resource reference.
	hasPath := len(issueDraftPaths(field)) != 0
	// Preserve destination context before tokenization. Only complete canonical
	// inline links and reference-definition lines may be masked. Unsupported
	// structures remain visible to the conservative destination-syntax check.
	for _, pattern := range []*regexp.Regexp{companionIssueLinkRE, companionIssueDefinitionRE} {
		var masked strings.Builder
		end := 0
		for _, match := range pattern.FindAllStringSubmatchIndex(field, -1) {
			if !companionReferenceSpanBoundary(field, match[0], match[1]) {
				continue
			}
			label := strings.TrimSpace(field[match[2]:match[3]])
			qualified := companionIssueReferenceRE.MatchString(label)
			if strings.Contains(label, "fak-private") && label != "fak-private" && !qualified {
				return true
			}
			// A label is visible evidence too; masking its canonical destination
			// must not erase an unqualified path next to an independent bare name.
			if !qualified && strings.ContainsAny(label, "/\\") {
				hasPath = true
			}
			masked.WriteString(field[end:match[0]])
			masked.WriteByte(' ')
			end = match[1]
		}
		if end > 0 {
			masked.WriteString(field[end:])
			field = masked.String()
		}
	}
	// This still applies when masking removed the field's last companion name.
	if companionDestinationSyntaxRE.MatchString(field) {
		return true
	}
	hasName := false
	for _, token := range strings.Fields(field) {
		// Keep internal URL delimiters intact: a canonical issue prefix followed
		// by a semicolon, comma or parenthesized resource suffix is not an issue.
		token = trimCompanionReferenceToken(token)
		if !strings.Contains(token, "fak-private") {
			// Path extraction rejects some punctuation, Markdown targets and drive
			// colons. A remaining slash-bearing token is still ambiguous beside a
			// bare companion name, even when cleanPathHint cannot extract it.
			if strings.ContainsAny(token, "/\\") {
				hasPath = true
			}
			continue
		}
		if token == "fak-private" {
			hasName = true
			continue
		}
		if !companionIssueReferenceRE.MatchString(token) {
			return true
		}
	}
	return hasName && hasPath
}

func companionReferenceSpanBoundary(field string, start, end int) bool {
	if start > 0 && !strings.ContainsRune(" \t\r\n\v\f", rune(field[start-1])) {
		return false
	}
	for end < len(field) && strings.ContainsRune(".,;:!?", rune(field[end])) {
		end++
	}
	return end == len(field) || strings.ContainsRune(" \t\r\n\v\f", rune(field[end]))
}

func trimCompanionReferenceToken(token string) string {
	for {
		token = strings.TrimRight(token, ".,;:!?")
		if len(token) < 2 {
			return token
		}
		// Only matching outer wrappers can be removed. Stripping internal
		// punctuation would turn a private locator into an allowed issue prefix.
		pair := string([]byte{token[0], token[len(token)-1]})
		switch pair {
		case "()", "[]", "<>", "\"\"", "''", "``", "**":
			token = token[1 : len(token)-1]
		default:
			return token
		}
	}
}

func compact(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

type reasonSet map[string]bool

func (s reasonSet) add(reason string) { s[reason] = true }
func (s reasonSet) has(reason string) bool {
	return s[reason]
}
func (s reasonSet) list() []string {
	out := make([]string, 0, len(s))
	for reason := range s {
		out = append(out, reason)
	}
	sort.Strings(out)
	return out
}

func removeReasons(list []string, toRemove ...string) []string {
	removeSet := make(map[string]bool, len(toRemove))
	for _, r := range toRemove {
		removeSet[r] = true
	}
	out := make([]string, 0, len(list))
	for _, r := range list {
		if !removeSet[r] {
			out = append(out, r)
		}
	}
	return out
}

func removeStrings(list []string, toRemove ...string) []string {
	removeSet := make(map[string]bool, len(toRemove))
	for _, s := range toRemove {
		removeSet[s] = true
	}
	out := make([]string, 0, len(list))
	for _, s := range list {
		if !removeSet[s] {
			out = append(out, s)
		}
	}
	return out
}

func hasProblemFrameAndDescriptions(c Candidate, reasons reasonSet) bool {
	if reasons.has(ReasonProblemFrameIncomplete) {
		return false
	}
	if reasons.has(ReasonPrivateBoundary) ||
		reasons.has(ReasonLiveUnarmored) ||
		reasons.has(ReasonNoiseIncomplete) ||
		reasons.has(ReasonAgentIncomplete) ||
		reasons.has(ReasonNotDispatchLeaf) ||
		reasons.has(ReasonOversizedSteps) ||
		reasons.has(ReasonUnexpandedTemplate) {
		return false
	}
	hasDesc := strings.TrimSpace(c.Title) != "" && (strings.TrimSpace(c.CurrentState) != "" ||
		strings.TrimSpace(c.InScope) != "" ||
		strings.TrimSpace(c.DoneCondition) != "" ||
		strings.TrimSpace(c.WhyNow) != "" ||
		strings.TrimSpace(c.BetterBecause) != "" ||
		strings.TrimSpace(c.Body) != "")
	return hasDesc
}
