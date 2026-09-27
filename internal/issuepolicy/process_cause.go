package issuepolicy

import (
	"fmt"
	"regexp"
	"strings"
)

const ProcessCauseLabelPrefix = "process-cause:"

var processCausePrimaries = map[string]bool{
	"concurrency":        true,
	"infrastructure-lag": true,
	"model-failure":      true,
	"harness-failure":    true,
	"scoping-failure":    true,
	"verification-gap":   true,
	"handoff-failure":    true,
	"other":              true,
	"unknown":            true,
	"none":               true,
}

var concurrencyCauseDetails = map[string]bool{
	"shared-state":        true,
	"lease-contention":    true,
	"integration-order":   true,
	"resource-contention": true,
	"ownership-overlap":   true,
}

var processCauseFieldRE = regexp.MustCompile(`(?i)^(?:\*\*|__)?\s*(process\s+cause(?:\s+detail)?)\s*(?:\*\*|__)?\s*:\s*(?:\*\*|__)?\s*(.*?)\s*$`)

// ProcessCauseReadout is the closed-vocabulary filing-time cause declaration.
// Numbered issues may carry an invalid/absent readout for descriptive backlog
// audits; new drafts are required to have a valid one by ReviewIssueDraft.
type ProcessCauseReadout struct {
	Primary  string   `json:"primary,omitempty"`
	Detail   string   `json:"detail,omitempty"`
	Declared bool     `json:"declared"`
	Valid    bool     `json:"valid"`
	Errors   []string `json:"errors,omitempty"`
}

// AssessProcessCause parses Process cause declarations outside fenced examples.
// Duplicate, malformed, unknown, and cross-field-inconsistent declarations fail
// closed so issue creation cannot silently choose among competing causes.
func AssessProcessCause(body string) ProcessCauseReadout {
	var out ProcessCauseReadout
	var primaries, details []string
	var fence byte
	var fenceWidth int

	for lineNo, raw := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		if marker, width, closing := processCauseFence(raw, fence, fenceWidth); marker != 0 {
			if closing {
				fence, fenceWidth = 0, 0
			} else if fence == 0 {
				fence, fenceWidth = marker, width
			}
			continue
		}
		if fence != 0 {
			continue
		}

		line := strings.TrimSpace(raw)
		if len(line) >= 2 && (line[0] == '-' || line[0] == '*' || line[0] == '+') && (line[1] == ' ' || line[1] == '\t') {
			line = strings.TrimSpace(line[2:])
		}
		match := processCauseFieldRE.FindStringSubmatch(line)
		if match == nil {
			probe := strings.ToLower(strings.TrimLeft(line, "*_ \t"))
			if strings.HasPrefix(probe, "process cause") {
				out.Errors = append(out.Errors, fmt.Sprintf("line %d: malformed Process cause declaration", lineNo+1))
			}
			continue
		}
		value := strings.ToLower(strings.Trim(strings.TrimSpace(match[2]), "`*_ "))
		if strings.EqualFold(strings.Join(strings.Fields(match[1]), " "), "process cause detail") {
			details = append(details, value)
		} else {
			primaries = append(primaries, value)
		}
	}

	out.Declared = len(primaries) > 0
	if len(primaries) != 1 {
		switch len(primaries) {
		case 0:
			out.Errors = append(out.Errors, "missing Process cause declaration")
		default:
			out.Errors = append(out.Errors, "duplicate Process cause declarations")
		}
	} else {
		out.Primary = primaries[0]
		if !processCausePrimaries[out.Primary] {
			out.Errors = append(out.Errors, fmt.Sprintf("unknown Process cause %q", out.Primary))
		}
	}

	if len(details) > 1 {
		out.Errors = append(out.Errors, "duplicate Process cause detail declarations")
	} else if len(details) == 1 {
		out.Detail = details[0]
	}
	if out.Primary == "concurrency" {
		if out.Detail == "" {
			out.Errors = append(out.Errors, "Process cause concurrency requires Process cause detail")
		} else if !concurrencyCauseDetails[out.Detail] {
			out.Errors = append(out.Errors, fmt.Sprintf("unknown concurrency Process cause detail %q", out.Detail))
		}
	} else if len(details) != 0 {
		out.Errors = append(out.Errors, "Process cause detail is only valid for concurrency")
	}

	out.Valid = len(out.Errors) == 0
	return out
}

// ProcessCauseLabel returns the one canonical label corresponding to a valid
// primary declaration.
func ProcessCauseLabel(primary string) string {
	return ProcessCauseLabelPrefix + strings.ToLower(strings.TrimSpace(primary))
}

// TagGeneratedIssue keeps a machine-authored issue's explicit cause when valid
// and otherwise records the caller's honest fallback. It refuses malformed or
// conflicting declarations instead of silently overwriting an authored cause.
func TagGeneratedIssue(body, fallback string) (string, string, error) {
	primary := strings.ToLower(strings.TrimSpace(fallback))
	if !processCausePrimaries[primary] || primary == "concurrency" {
		return "", "", fmt.Errorf("invalid generated issue process cause fallback %q", fallback)
	}
	readout := AssessProcessCause(body)
	if readout.Valid {
		return body, ProcessCauseLabel(readout.Primary), nil
	}
	if readout.Declared || len(readout.Errors) != 1 || readout.Errors[0] != "missing Process cause declaration" {
		return "", "", fmt.Errorf("invalid generated issue process cause: %s", strings.Join(readout.Errors, "; "))
	}
	body = strings.TrimRight(body, "\r\n") + "\n\nProcess cause: " + primary + "\n"
	return body, ProcessCauseLabel(primary), nil
}

func processCauseFence(line string, active byte, activeWidth int) (marker byte, width int, closing bool) {
	trimmed := strings.TrimLeft(line, " \t")
	if trimmed == "" || (trimmed[0] != '`' && trimmed[0] != '~') {
		return 0, 0, false
	}
	marker = trimmed[0]
	for width < len(trimmed) && trimmed[width] == marker {
		width++
	}
	if width < 3 {
		return 0, 0, false
	}
	if active == 0 {
		return marker, width, false
	}
	if marker != active || width < activeWidth || strings.TrimSpace(trimmed[width:]) != "" {
		return 0, 0, false
	}
	return marker, width, true
}
