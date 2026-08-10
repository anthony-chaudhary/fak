package issuepolicy

import (
	"fmt"
	"sort"
	"strings"
)

var validEvidenceStages = map[string]bool{
	"toy": true, "development": true, "representative": true, "soak": true,
	"target-load": true, "overload": true, "degradation": true, "recovery": true,
}
var validEvidenceProvenance = map[string]bool{
	"witnessed": true, "observed": true, "modeled": true, "extrapolated": true,
}

type ScaleEvidenceRecord struct {
	Stage       string          `json:"stage"`
	Provenance  string          `json:"provenance"`
	Duration    string          `json:"duration,omitempty"`
	Workload    string          `json:"workload,omitempty"`
	Environment string          `json:"environment,omitempty"`
	Values      []EnvelopeValue `json:"values"`
}

type ScaleEvidenceReadout struct {
	Records        []ScaleEvidenceRecord `json:"records,omitempty"`
	Qualifying     []EnvelopeValue       `json:"qualifying,omitempty"`
	RequiredStages []string              `json:"required_stages,omitempty"`
	MissingStages  []string              `json:"missing_stages,omitempty"`
	Invalid        []string              `json:"invalid,omitempty"`
}

// Scale evidence is a sequence of semicolon-delimited records:
// stage; provenance; dimension: value unit[, dimension: value unit]; key=value...
// Direct witnessed/observed records can satisfy an envelope; modeled and
// extrapolated records stay visible but never silently become production proof.
func scaleEvidence(c Candidate) ScaleEvidenceReadout {
	out := ScaleEvidenceReadout{RequiredStages: parseEvidenceStages(c.RequiredScaleStages)}
	if strings.TrimSpace(c.ScaleEvidence) == "" {
		return out
	}
	best := map[string]EnvelopeValue{}
	for lineNo, raw := range strings.Split(strings.ReplaceAll(c.ScaleEvidence, "\r\n", "\n"), "\n") {
		line := strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(raw), "-*"))
		if line == "" {
			continue
		}
		parts := strings.Split(line, ";")
		if len(parts) < 3 {
			out.Invalid = append(out.Invalid, fmt.Sprintf("line %d: want stage; provenance; dimensions", lineNo+1))
			continue
		}
		record := ScaleEvidenceRecord{Stage: normalizeEnvelopeToken(parts[0]), Provenance: normalizeEnvelopeToken(parts[1])}
		if !validEvidenceStages[record.Stage] {
			out.Invalid = append(out.Invalid, fmt.Sprintf("line %d: invalid stage %q", lineNo+1, record.Stage))
			continue
		}
		if !validEvidenceProvenance[record.Provenance] {
			out.Invalid = append(out.Invalid, fmt.Sprintf("line %d: invalid provenance %q", lineNo+1, record.Provenance))
			continue
		}
		var valueLines []string
		for _, item := range strings.Split(parts[2], ",") {
			valueLines = append(valueLines, strings.TrimSpace(item))
		}
		values, invalid := parseEnvelope(strings.Join(valueLines, "\n"), false)
		if len(invalid) > 0 || len(values) == 0 {
			out.Invalid = append(out.Invalid, fmt.Sprintf("line %d: %s", lineNo+1, strings.Join(invalid, "; ")))
			continue
		}
		record.Values = values
		for _, attr := range parts[3:] {
			kv := strings.SplitN(strings.TrimSpace(attr), "=", 2)
			if len(kv) != 2 {
				out.Invalid = append(out.Invalid, fmt.Sprintf("line %d: invalid attribute %q", lineNo+1, attr))
				continue
			}
			switch normalizeEnvelopeToken(kv[0]) {
			case "duration":
				record.Duration = strings.TrimSpace(kv[1])
			case "workload":
				record.Workload = strings.TrimSpace(kv[1])
			case "environment":
				record.Environment = strings.TrimSpace(kv[1])
			default:
				out.Invalid = append(out.Invalid, fmt.Sprintf("line %d: unknown attribute %q", lineNo+1, kv[0]))
			}
		}
		out.Records = append(out.Records, record)
		if record.Provenance == "witnessed" || record.Provenance == "observed" {
			for _, value := range values {
				prior, ok := best[value.Dimension]
				if !ok || value.Value > prior.Value {
					best[value.Dimension] = value
				}
			}
		}
	}
	present := map[string]bool{}
	for _, record := range out.Records {
		present[record.Stage] = true
	}
	for _, stage := range out.RequiredStages {
		if !present[stage] {
			out.MissingStages = append(out.MissingStages, stage)
		}
	}
	for _, value := range best {
		out.Qualifying = append(out.Qualifying, value)
	}
	sort.Slice(out.Qualifying, func(i, j int) bool { return out.Qualifying[i].Dimension < out.Qualifying[j].Dimension })
	sort.Strings(out.Invalid)
	sort.Strings(out.MissingStages)
	return out
}

func parseEvidenceStages(raw string) []string {
	seen := map[string]bool{}
	var out []string
	for _, item := range strings.FieldsFunc(raw, func(r rune) bool { return r == ',' || r == '\n' }) {
		stage := normalizeEnvelopeToken(strings.TrimLeft(strings.TrimSpace(item), "-*"))
		if stage != "" && !seen[stage] {
			seen[stage] = true
			out = append(out, stage)
		}
	}
	sort.Strings(out)
	return out
}

func renderEvidenceEnvelope(values []EnvelopeValue) string {
	var lines []string
	for _, value := range values {
		lines = append(lines, fmt.Sprintf("- %s: %g %s", value.Dimension, value.Value, value.Unit))
	}
	return strings.Join(lines, "\n")
}
