package negframe

import (
	"regexp"
	"strings"
)

// TriggerResult is the parse-time decision used to wake a positive-state
// operator. Token is the lexicon span that caused the decision; Category lets
// callers route the turn without running document scoring.
type TriggerResult struct {
	Negation bool     `json:"negation"`
	Token    string   `json:"token,omitempty"`
	Category Category `json:"category,omitempty"`
}

// triggerOnlyRules extend the shared document lexicon for parse-time concepts
// that are useful operator gates but are too broad to count as prose debt.
// Existing negation forms stay in rules; Trigger scans both tables, so there is
// one owner for every lexical form rather than a second trigger token list.
var triggerOnlyRules = []reframeRule{
	{Pattern: wordPattern(`not`), Category: Prohibition},
	{Pattern: wordPattern(`only`), Category: Exception},
	{Pattern: wordPattern(`except`), Category: Exception},
}

// Trigger classifies one turn cheaply. Compiled package-level regexps are
// reused, fenced and inline code is opaque, and the first lexical match in text
// order wins (with the more-specific shared rules winning ties).
func Trigger(text string) TriggerResult {
	bestStart := -1
	bestOrder := 0
	var bestRule reframeRule
	bestToken := ""

	order := 0
	visit := func(r reframeRule, prose string, offset int) {
		loc := r.Pattern.FindStringIndex(prose)
		if loc == nil {
			order++
			return
		}
		start := offset + loc[0]
		if bestStart < 0 || start < bestStart || (start == bestStart && order < bestOrder) {
			bestStart = start
			bestOrder = order
			bestRule = r
			bestToken = strings.ToLower(prose[loc[0]:loc[1]])
		}
		order++
	}

	inFence := false
	offset := 0
	for _, raw := range strings.SplitAfter(text, "\n") {
		line := strings.TrimSuffix(raw, "\n")
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			offset += len(raw)
			continue
		}
		if !inFence {
			for _, span := range proseSpans(line) {
				for _, r := range rules {
					visit(r, span.text, offset+span.start)
				}
				for _, r := range triggerOnlyRules {
					visit(r, span.text, offset+span.start)
				}
			}
		}
		offset += len(raw)
	}
	if bestStart < 0 {
		return TriggerResult{}
	}
	return TriggerResult{Negation: true, Token: bestToken, Category: bestRule.Category}
}

type proseSpan struct {
	start int
	text  string
}

// proseSpans returns text outside inline backtick spans. An unmatched backtick
// makes the remainder opaque, which is the conservative false-positive choice.
func proseSpans(line string) []proseSpan {
	var spans []proseSpan
	start := 0
	for start < len(line) {
		open := strings.IndexByte(line[start:], '`')
		if open < 0 {
			if start < len(line) {
				spans = append(spans, proseSpan{start: start, text: line[start:]})
			}
			break
		}
		open += start
		if open > start {
			spans = append(spans, proseSpan{start: start, text: line[start:open]})
		}
		close := strings.IndexByte(line[open+1:], '`')
		if close < 0 {
			break
		}
		start = open + 1 + close + 1
	}
	return spans
}

func wordPattern(word string) *regexp.Regexp {
	return regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(word) + `\b`)
}
