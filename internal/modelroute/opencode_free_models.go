package modelroute

import "strings"

// OpenCodeFreeModel is one free-tier model discovered from `opencode models`
// output (one "provider/model" id per line). It carries the quality-gate
// verdict so callers can split survivors from rejected candidates without
// re-running the gate.
type OpenCodeFreeModel struct {
	// ID is the full "provider/model" id, e.g. "opencode/ll-alpha-free".
	ID string `json:"id"`
	// Provider is the provider prefix, e.g. "opencode".
	Provider string `json:"provider"`
	// Name is the model name without the provider prefix, e.g. "ll-alpha-free".
	Name string `json:"name"`
	// QualityOK reports whether the model clears the free-tier quality gate.
	QualityOK bool `json:"quality_ok"`
	// Reason is the rejection reason when QualityOK is false.
	Reason string `json:"reason,omitempty"`
}

// ParseOpenCodeFreeModels parses the plain-text output of `opencode models`
// (one "provider/model" id per line) and returns the entries whose model name
// ends with freeSuffix (default "-free" when empty). Each entry is run through
// the free-tier quality gate (FreeQualityReason) with tool-capable=true and
// context-tokens=0 (operator-verified at bind time), so deny-token rejections
// (toys, non-chat endpoints, proxy slugs) surface in Reason.
//
// This is the discovery half of the auto-update story: OpenCode's free roster
// changes often, and hardcoding ids in opencode.json / the router routing table
// / the free-tier roster drifts the moment OpenCode renames or retires one.
// Parsing the live `opencode models` output keeps FAK ops honest about what is
// actually served, instead of trusting a stale committed list.
func ParseOpenCodeFreeModels(lines []string, freeSuffix string) []OpenCodeFreeModel {
	if freeSuffix == "" {
		freeSuffix = "-free"
	}
	var out []OpenCodeFreeModel
	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		slash := strings.Index(line, "/")
		if slash <= 0 || slash == len(line)-1 {
			continue
		}
		provider := line[:slash]
		name := line[slash+1:]
		if !strings.HasSuffix(name, freeSuffix) {
			continue
		}
		cand := OpenCodeFreeModel{
			ID:       line,
			Provider: provider,
			Name:     name,
		}
		if reason := FreeQualityReason(name, true, 0); reason != "" {
			cand.QualityOK = false
			cand.Reason = reason
		} else {
			cand.QualityOK = true
		}
		out = append(out, cand)
	}
	return out
}

// SplitOpenCodeFreeModels partitions discovered candidates into survivors
// (quality-gated) and rejected (with reasons), for machine-readable reporting.
func SplitOpenCodeFreeModels(cands []OpenCodeFreeModel) (survivors, rejected []OpenCodeFreeModel) {
	for _, c := range cands {
		if c.QualityOK {
			survivors = append(survivors, c)
		} else {
			rejected = append(rejected, c)
		}
	}
	return survivors, rejected
}
