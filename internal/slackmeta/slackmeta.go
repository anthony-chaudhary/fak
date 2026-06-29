// Package slackmeta renders the common metadata line every fak Slack report carries.
//
// The score is intentionally simple and self-declared: a card counts the facts it is
// surfacing as signal and the wrapper/context it adds as noise. It is not a channel
// readback or a Slack engagement metric; it is a per-message "how dense is this report?"
// label so operators can tell status heartbeats from high-signal reports at a glance.
package slackmeta

import (
	"fmt"
	"strings"
)

// Score is a per-message signal/noise self-score for a Slack report.
type Score struct {
	Signal int     `json:"signal"`
	Noise  int     `json:"noise"`
	Ratio  float64 `json:"ratio"`
	Basis  string  `json:"basis,omitempty"`
}

// New builds a Score, clamping negative counts and applying a one-unit noise floor so
// the ratio is always finite and comparable across reports.
func New(signal, noise int, basis string) Score {
	if signal < 0 {
		signal = 0
	}
	if noise < 1 {
		noise = 1
	}
	return Score{
		Signal: signal,
		Noise:  noise,
		Ratio:  float64(signal) / float64(noise),
		Basis:  strings.TrimSpace(basis),
	}
}

// Line renders the compact metadata line used in Slack text fallbacks and context
// blocks.
func (s Score) Line() string {
	if s.Noise < 1 {
		s = New(s.Signal, s.Noise, s.Basis)
	}
	line := fmt.Sprintf("S/N self-score %.1f (%d signal / %d noise)", s.Ratio, s.Signal, s.Noise)
	if s.Basis != "" {
		line += ": " + s.Basis
	}
	return line
}

// AppendText appends the S/N line to a text fallback unless it already carries one.
func AppendText(text string, s Score) string {
	if strings.Contains(text, "S/N self-score") {
		return text
	}
	if strings.TrimSpace(text) == "" {
		return s.Line()
	}
	return strings.TrimRight(text, "\n") + "\n_" + s.Line() + "_"
}

// ContextBlock returns a Slack Block Kit context block carrying the S/N line.
func ContextBlock(s Score) map[string]any {
	return map[string]any{
		"type": "context",
		"elements": []any{
			map[string]any{"type": "mrkdwn", "text": s.Line()},
		},
	}
}

// AppendContext appends a Slack context block carrying the S/N line unless the existing
// blocks already include one.
func AppendContext(blocks []any, s Score) []any {
	for _, blk := range blocks {
		if containsSN(blk) {
			return blocks
		}
	}
	return append(blocks, ContextBlock(s))
}

// NonEmpty counts non-blank strings.
func NonEmpty(values ...string) int {
	n := 0
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			n++
		}
	}
	return n
}

func containsSN(v any) bool {
	switch t := v.(type) {
	case string:
		return strings.Contains(t, "S/N self-score")
	case map[string]any:
		for _, vv := range t {
			if containsSN(vv) {
				return true
			}
		}
	case []any:
		for _, vv := range t {
			if containsSN(vv) {
				return true
			}
		}
	}
	return false
}
