package issuepolicy

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	EnvelopeUndeclared    = "undeclared"
	EnvelopeTargetMissing = "target_missing"
	EnvelopeInvalid       = "invalid"
	EnvelopeStatusGap     = "gap"
	EnvelopeMet           = "met"
	EnvelopeNotRequired   = "not_required"
)

// EnvelopeValue is one typed dimension in an operating-envelope declaration.
// Operator is interpreted from the target: >= is a minimum capacity, <= is a
// maximum tolerated quantity, and = requires an exact match.
type EnvelopeValue struct {
	Dimension     string  `json:"dimension"`
	Operator      string  `json:"operator,omitempty"`
	Value         float64 `json:"value,omitempty"`
	Unit          string  `json:"unit,omitempty"`
	NotApplicable bool    `json:"not_applicable,omitempty"`
	Reason        string  `json:"reason,omitempty"`
}

type EnvelopeGap struct {
	Dimension string         `json:"dimension"`
	Reason    string         `json:"reason"`
	Target    *EnvelopeValue `json:"target,omitempty"`
	Witnessed *EnvelopeValue `json:"witnessed,omitempty"`
}

// OperatingEnvelopeReadout is emitted by every issue-contract review. Legacy
// issues remain undeclared until migration, but an explicit production claim
// is fail-closed unless its target is declared and witnessed.
type OperatingEnvelopeReadout struct {
	Status             string          `json:"status"`
	Required           bool            `json:"required"`
	CompletionStandard string          `json:"completion_standard,omitempty"`
	Target             []EnvelopeValue `json:"target,omitempty"`
	Witnessed          []EnvelopeValue `json:"witnessed,omitempty"`
	Gaps               []EnvelopeGap   `json:"gaps,omitempty"`
	Invalid            []string        `json:"invalid,omitempty"`
}

var envelopeLineRE = regexp.MustCompile(`(?i)^\s*[-*]?\s*([a-z][a-z0-9 _./-]*?)\s*:\s*(?:(>=|<=|=)\s*)?([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$`)
var envelopeNARE = regexp.MustCompile(`(?i)^\s*[-*]?\s*([a-z][a-z0-9 _./-]*?)\s*:\s*(?:n/?a|not[- ]applicable)\s*(?:\((.+)\)|[-—:]\s*(.+))\s*$`)

func operatingEnvelope(c Candidate) OperatingEnvelopeReadout {
	standard := normalizeCompletionStandard(c.CompletionStandard)
	out := OperatingEnvelopeReadout{
		Status:             EnvelopeUndeclared,
		Required:           standard == "production",
		CompletionStandard: standard,
	}
	var targetInvalid, witnessedInvalid []string
	out.Target, targetInvalid = parseEnvelope(c.TargetEnvelope, true)
	out.Witnessed, witnessedInvalid = parseEnvelope(c.WitnessedEnvelope, false)
	out.Invalid = append(targetInvalid, witnessedInvalid...)
	if out.Required && len(out.Target) == 0 {
		out.Status = EnvelopeTargetMissing
		return out
	}
	if len(out.Invalid) > 0 {
		out.Status = EnvelopeInvalid
		return out
	}
	if len(out.Target) == 0 {
		if len(out.Witnessed) > 0 {
			out.Status = EnvelopeNotRequired
		}
		return out
	}
	out.Gaps = compareEnvelopes(out.Target, out.Witnessed)
	if len(out.Gaps) > 0 {
		out.Status = EnvelopeStatusGap
	} else {
		out.Status = EnvelopeMet
	}
	return out
}

func normalizeCompletionStandard(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	if s == "" {
		return ""
	}
	for _, nonProduction := range []string{"research", "experiment", "prototype", "demo", "development", "dev", "integrated", "staging"} {
		if strings.HasPrefix(s, nonProduction) {
			return nonProduction
		}
	}
	if strings.HasPrefix(s, "production") || strings.Contains(s, "production complete") {
		return "production"
	}
	return s
}

func parseEnvelope(raw string, target bool) ([]EnvelopeValue, []string) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	values := map[string]EnvelopeValue{}
	var invalid []string
	// record admits one parsed dimension, refusing a second entry for a dimension already
	// claimed on an earlier line. First-wins is deliberate: a contract that states a dimension
	// twice is ambiguous, so the repeat is reported as invalid rather than silently overriding.
	// Both the not-applicable and the measured-value branch admit through here.
	record := func(v EnvelopeValue) {
		if _, duplicate := values[v.Dimension]; duplicate {
			invalid = append(invalid, fmt.Sprintf("%s: duplicate dimension", v.Dimension))
			return
		}
		values[v.Dimension] = v
	}
	for _, original := range strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n") {
		line := strings.TrimSpace(original)
		if line == "" {
			continue
		}
		if m := envelopeNARE.FindStringSubmatch(line); m != nil {
			reason := strings.TrimSpace(m[2])
			if reason == "" {
				reason = strings.TrimSpace(m[3])
			}
			dim := normalizeEnvelopeToken(m[1])
			if reason == "" {
				invalid = append(invalid, fmt.Sprintf("%s: not-applicable requires a reason", dim))
				continue
			}
			record(EnvelopeValue{Dimension: dim, NotApplicable: true, Reason: reason})
			continue
		}
		m := envelopeLineRE.FindStringSubmatch(line)
		if m == nil {
			invalid = append(invalid, fmt.Sprintf("unparseable envelope entry %q; want '- dimension: >= value unit'", line))
			continue
		}
		dim, op := normalizeEnvelopeToken(m[1]), m[2]
		if op == "" {
			if target {
				op = ">="
			} else {
				op = "="
			}
		}
		value, err := strconv.ParseFloat(m[3], 64)
		unit := normalizeEnvelopeToken(m[4])
		if err != nil || value < 0 || unit == "" {
			invalid = append(invalid, fmt.Sprintf("%s: invalid non-negative value/unit", dim))
			continue
		}
		record(EnvelopeValue{Dimension: dim, Operator: op, Value: value, Unit: unit})
	}
	out := make([]EnvelopeValue, 0, len(values))
	for _, v := range values {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Dimension < out[j].Dimension })
	sort.Strings(invalid)
	return out, invalid
}

func compareEnvelopes(target, witnessed []EnvelopeValue) []EnvelopeGap {
	observed := make(map[string]EnvelopeValue, len(witnessed))
	for _, value := range witnessed {
		observed[value.Dimension] = value
	}
	var gaps []EnvelopeGap
	for i := range target {
		t := target[i]
		if t.NotApplicable {
			continue
		}
		w, ok := observed[t.Dimension]
		if !ok || w.NotApplicable {
			gaps = append(gaps, EnvelopeGap{Dimension: t.Dimension, Reason: "missing witnessed dimension", Target: &t})
			continue
		}
		if w.Unit != t.Unit {
			gaps = append(gaps, EnvelopeGap{Dimension: t.Dimension, Reason: fmt.Sprintf("unit mismatch: target %s, witnessed %s", t.Unit, w.Unit), Target: &t, Witnessed: &w})
			continue
		}
		met := false
		switch t.Operator {
		case ">=":
			met = w.Value >= t.Value
		case "<=":
			met = w.Value <= t.Value
		case "=":
			met = w.Value == t.Value
		}
		if !met {
			gaps = append(gaps, EnvelopeGap{Dimension: t.Dimension, Reason: fmt.Sprintf("witnessed %g %s does not satisfy %s %g %s", w.Value, w.Unit, t.Operator, t.Value, t.Unit), Target: &t, Witnessed: &w})
		}
	}
	sort.Slice(gaps, func(i, j int) bool { return gaps[i].Dimension < gaps[j].Dimension })
	return gaps
}

func normalizeEnvelopeToken(raw string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(raw))), " ")
}
