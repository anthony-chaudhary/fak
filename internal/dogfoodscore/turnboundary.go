package dogfoodscore

import "strings"

// TurnBoundaryResult is the pre-final decision for one transcript. A fresh
// harness Stop-hook failure refuses success narration until the agent handles
// the failure and produces another assistant turn.
type TurnBoundaryResult struct {
	AllowFinal   bool   `json:"allow_final"`
	FreshFailure bool   `json:"fresh_stop_hook_failure"`
	Reason       string `json:"reason"`
	HarnessLine  string `json:"harness_line,omitempty"`
}

const turnBoundaryStopFailure = "fresh Stop-hook failure follows the latest assistant turn; handle it before final success narration"

// CheckTurnBoundary scans the current transcript at the point immediately
// before final copy is emitted. It fails closed when a genuine harness
// Stop-hook error is newer than the latest assistant event. Assistant prose
// that merely quotes a hook error is not harness evidence.
func CheckTurnBoundary(raw []byte) TurnBoundaryResult {
	result := TurnBoundaryResult{AllowFinal: true, Reason: "no fresh Stop-hook failure"}
	latestAssistant := -1
	latestFailure := -1
	failureLine := ""

	for i, rawLine := range strings.Split(string(raw), "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		if assistantText(line) != "" {
			latestAssistant = i
			continue
		}
		if stopErrorRe.MatchString(line) {
			latestFailure = i
			failureLine = clip(line, harnessLineClip)
		}
	}

	if latestFailure > latestAssistant {
		result.AllowFinal = false
		result.FreshFailure = true
		result.Reason = turnBoundaryStopFailure
		result.HarnessLine = failureLine
	}
	return result
}
