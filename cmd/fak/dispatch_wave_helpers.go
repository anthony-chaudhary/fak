package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func dispatchWaveSkippedCandidates(candidates []dispatchWaveCandidate) []dispatchWaveCandidate {
	out := make([]dispatchWaveCandidate, 0, len(candidates))
	for _, cand := range candidates {
		if !cand.Selected {
			out = append(out, cand)
		}
	}
	return out
}

func dispatchWaveIssueLabel(cand dispatchWaveCandidate) string {
	if cand.Issue <= 0 {
		return "-"
	}
	return fmt.Sprintf("#%d", cand.Issue)
}

func dispatchWaveScopeLabel(cand dispatchWaveCandidate) string {
	switch {
	case len(cand.Tree) == 0:
		return "unknown"
	case cand.Scoped:
		return "issue"
	default:
		return "lane"
	}
}

func dispatchWaveCollisionLabel(ids []string) string {
	if len(ids) == 0 {
		return "-"
	}
	return strings.Join(ids, ",")
}

func accountFromWaveLane(m dispatchtick.AccountWaveLane) dispatchtick.Account {
	return dispatchtick.Account{
		Tag:   firstString(m.Tag, m.Account),
		Tier:  m.SelectedTier,
		Model: m.Model,
		Dir:   m.ConfigDir,
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func scrubDispatchSecrets(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			if dispatchSecretKey(k) {
				continue
			}
			out[k] = scrubDispatchSecrets(val)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = scrubDispatchSecrets(val)
		}
		return out
	default:
		return v
	}
}

func dispatchSecretKey(k string) bool {
	k = strings.ToLower(strings.TrimSpace(k))
	return strings.Contains(k, "token") || strings.Contains(k, "secret") || strings.Contains(k, "api_key") || strings.Contains(k, "apikey")
}
