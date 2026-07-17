package agent

import (
	"bytes"
	"encoding/json"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

var positiveResidueKey = regexp.MustCompile(`(?i)^([a-z][a-z0-9_.-]{0,63})\s*(?:=|:)\s*(.+)$`)

type positiveResidueResult struct {
	Text           string
	RestoreID      string
	RestoreBytes   []byte
	DroppedBytes   int
	AssertionsKept int
}

type residueAssertion struct {
	key   string
	value string
}

// extractPositiveResidue is deliberately conservative. It recognizes explicit
// "fact: key=value" / "state: key=value" assertions and removes a key when a
// later "fixed:", "abandoned:", "superseded:", or "negated:" marker names it.
// Unknown transcript prose never becomes asserted state.
func extractPositiveResidue(elems []json.RawMessage) positiveResidueResult {
	if len(elems) == 0 {
		return positiveResidueResult{}
	}
	raw, _ := json.Marshal(elems)
	result := positiveResidueResult{
		RestoreBytes: raw,
		DroppedBytes: len(raw),
	}
	if len(raw) > 0 {
		result.RestoreID = originatingTaskDigestID(raw)
	}

	active := make(map[string]residueAssertion)
	for _, elem := range elems {
		text, ok := elementTextContent(elem)
		if !ok {
			continue
		}
		for _, line := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
			line = strings.TrimSpace(strings.TrimLeft(line, "-* "))
			if residual, ok := negframe.ResolveResidual(line); ok {
				key := "residual:" + strings.ToLower(line)
				active[key] = residueAssertion{key: key, value: residual.Positive}
				continue
			}
			lower := strings.ToLower(line)
			switch {
			case strings.HasPrefix(lower, "fact:"), strings.HasPrefix(lower, "state:"):
				_, rest, _ := strings.Cut(line, ":")
				match := positiveResidueKey.FindStringSubmatch(strings.TrimSpace(rest))
				if len(match) != 3 {
					continue
				}
				key := strings.ToLower(strings.TrimSpace(match[1]))
				value := strings.TrimSpace(match[2])
				if value != "" {
					active[key] = residueAssertion{key: key, value: value}
				}
			case strings.HasPrefix(lower, "fixed:"), strings.HasPrefix(lower, "abandoned:"), strings.HasPrefix(lower, "superseded:"), strings.HasPrefix(lower, "negated:"):
				_, rest, _ := strings.Cut(line, ":")
				key := strings.ToLower(strings.TrimSpace(rest))
				if match := positiveResidueKey.FindStringSubmatch(key); len(match) == 3 {
					key = strings.ToLower(strings.TrimSpace(match[1]))
				}
				delete(active, key)
			}
		}
	}
	if len(active) == 0 {
		return result
	}
	keys := make([]string, 0, len(active))
	for key := range active {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var out bytes.Buffer
	for i, key := range keys {
		if i > 0 {
			out.WriteString("; ")
		}
		if strings.HasPrefix(key, "residual:") {
			out.WriteString(active[key].value)
		} else {
			out.WriteString(key)
			out.WriteByte('=')
			out.WriteString(active[key].value)
		}
	}
	result.Text = out.String()
	result.AssertionsKept = len(keys)
	return result
}
