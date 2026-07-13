package quality

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

// instrInstructionFollowing is the instruction-following oracle for executive
// reports (#4561): the report must obey every EXPLICIT instruction the case
// declares — a length cap, required fields, forbidden content, and format
// constraints. It is the compliance axis of the report-quality layer —
// claim-grounding (#4551) checks WHAT the report asserts is backed by
// evidence; this oracle checks the report was written the way it was TOLD to
// be, which is the direction a fluent-but-disobedient report enters from (a
// 300-word essay where a 40-word bulleted rollup was ordered).
//
// Instructions travel on the case in two places:
//
//  1. Rubric.Required / Rubric.Forbidden: each required entry is one
//     "must include" instruction and each forbidden entry one "must not
//     mention" instruction, both matched as case-insensitive substrings of
//     eng.Text.
//
//  2. A spec line in the Prompt, marked and JSON-encoded so it is an explicit,
//     machine-checkable order rather than prose the oracle guesses at:
//
//     INSTRUCTIONS: {"max_words":40,"max_chars":0,"line_prefix":"- ","starts_with":""}
//
//     max_words / max_chars (runes) are length caps (0 = no cap);
//     line_prefix orders every non-empty line to start with the prefix (a
//     bullet-list constraint); starts_with orders the report to open with a
//     fixed heading. A Prompt with no marker line declares no spec
//     instructions; a marker line whose JSON does not parse is skipped, never
//     panicked on (a malformed order is not an adjudicable instruction).
//
// Every declared instruction is adjudicated independently in a fixed order —
// length caps, format constraints, required fields, forbidden content — so
// the FIRST violation reported is deterministic. Score = obeyed instructions
// / declared instructions; Pass iff Score >= Rubric.MinScore (default 1:
// every explicit instruction must be followed). On failure Detail names the
// violated instruction with its evidence — the word count over the cap, the
// offending line, the omitted field, the forbidden phrase — localizing the
// disobedience per the spine contract. A case declaring no instructions
// passes at score 1 with a Detail note (there was nothing to disobey).
type instrInstructionFollowing struct{}

func (instrInstructionFollowing) Name() string { return "instruction-following" }
func (instrInstructionFollowing) Kind() string { return "rubric" }

func init() { Register(instrInstructionFollowing{}) }

func (instrInstructionFollowing) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "instruction-following", Kind: "rubric", Pass: true, Score: 1}
	total, violations := instrEvaluate(instrParseSpec(c.Prompt), c.Rubric, eng.Text)
	if total == 0 {
		v.Detail = "no explicit instructions declared; nothing to check"
		return v
	}
	obeyed := total - len(violations)
	v.Score = float64(obeyed) / float64(total)
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: every explicit instruction must be followed
	}
	if v.Score < min {
		v.Pass = false
		v.Detail = fmt.Sprintf("instruction following %.2f < %.2f (%d/%d instructions obeyed); first violated instruction: %s",
			v.Score, min, obeyed, total, violations[0])
		return v
	}
	if len(violations) > 0 {
		v.Detail = fmt.Sprintf("instruction following %.2f >= %.2f (%d/%d obeyed; tolerated violation: %s)",
			v.Score, min, obeyed, total, violations[0])
		return v
	}
	v.Detail = fmt.Sprintf("all %d explicit instruction(s) followed", total)
	return v
}

// instrSpecMarker prefixes the machine-checkable instruction spec line inside
// a case's Prompt. Everything after the marker on that line is the JSON spec.
const instrSpecMarker = "INSTRUCTIONS:"

// instrSpec is the prompt-carried instruction spec: the length caps and format
// constraints the report was explicitly ordered to satisfy. Zero values mean
// "not ordered" — an absent cap is not a cap of zero.
type instrSpec struct {
	MaxWords   int    `json:"max_words,omitempty"`
	MaxChars   int    `json:"max_chars,omitempty"`
	LinePrefix string `json:"line_prefix,omitempty"`
	StartsWith string `json:"starts_with,omitempty"`
}

// instrParseSpec extracts the first well-formed instruction spec from the
// prompt. Lines without the marker, and marker lines whose JSON payload does
// not parse, are skipped deterministically.
func instrParseSpec(prompt string) instrSpec {
	for _, line := range strings.Split(prompt, "\n") {
		rest, ok := strings.CutPrefix(strings.TrimSpace(line), instrSpecMarker)
		if !ok {
			continue
		}
		var s instrSpec
		if err := json.Unmarshal([]byte(strings.TrimSpace(rest)), &s); err != nil {
			continue // malformed order: not an adjudicable instruction
		}
		return s
	}
	return instrSpec{}
}

// instrEvaluate adjudicates every explicit instruction the case declares
// against the report text. It returns the number of instructions checked and
// the violations in the documented fixed order — length caps, format
// constraints, required fields, forbidden content — each naming the violated
// instruction with its evidence. Empty rubric entries declare nothing and are
// skipped.
func instrEvaluate(spec instrSpec, r RubricSpec, text string) (total int, violations []string) {
	low := strings.ToLower(text)

	if spec.MaxWords > 0 {
		total++
		if n := len(strings.Fields(text)); n > spec.MaxWords {
			violations = append(violations,
				fmt.Sprintf("length cap violated: report runs %d words, cap is %d", n, spec.MaxWords))
		}
	}
	if spec.MaxChars > 0 {
		total++
		if n := utf8.RuneCountInString(text); n > spec.MaxChars {
			violations = append(violations,
				fmt.Sprintf("length cap violated: report runs %d characters, cap is %d", n, spec.MaxChars))
		}
	}
	if spec.StartsWith != "" {
		total++
		if !strings.HasPrefix(strings.TrimSpace(text), spec.StartsWith) {
			violations = append(violations,
				fmt.Sprintf("format violated: report does not open with the required heading %q", spec.StartsWith))
		}
	}
	if spec.LinePrefix != "" {
		total++
		for i, line := range strings.Split(text, "\n") {
			t := strings.TrimSpace(line)
			if t == "" {
				continue
			}
			if !strings.HasPrefix(t, spec.LinePrefix) {
				violations = append(violations,
					fmt.Sprintf("format violated: line %d %q does not start with the required prefix %q", i+1, t, spec.LinePrefix))
				break // one format instruction, one verdict: the first bad line localizes it
			}
		}
	}
	for _, req := range r.Required {
		if req == "" {
			continue
		}
		total++
		if !strings.Contains(low, strings.ToLower(req)) {
			violations = append(violations,
				fmt.Sprintf("required field omitted: report does not include %q", req))
		}
	}
	for _, f := range r.Forbidden {
		if f == "" {
			continue
		}
		total++
		if strings.Contains(low, strings.ToLower(f)) {
			violations = append(violations,
				fmt.Sprintf("forbidden content included: report contains %q", f))
		}
	}
	return total, violations
}
