package dogfoodscore

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

// TurnBoundaryResult is the pre-final decision for one transcript. A fresh
// harness Stop-hook failure refuses success narration until the agent handles
// the failure and produces another assistant turn.
type TurnBoundaryResult struct {
	AllowFinal   bool   `json:"allow_final"`
	FreshFailure bool   `json:"fresh_stop_hook_failure"`
	Reason       string `json:"reason"`
	HarnessLine  string `json:"harness_line,omitempty"`
	// Reachable reports whether a transcript was actually read. False means the
	// check is unwitnessed, not clean — AllowFinal stays true because there is
	// nothing to refuse on, but the caller must not read that as a green.
	Reachable  bool   `json:"transcript_reachable"`
	Transcript string `json:"transcript,omitempty"`
}

const (
	turnBoundaryStopFailure = "fresh Stop-hook failure follows the latest assistant turn; handle it before final success narration"
	turnBoundaryUnreachable = "no transcript reachable from here — the pre-final check is unwitnessed, not clean"
)

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

// CheckTurnBoundaryLatest runs the pre-final check against the newest transcript
// for this workspace. This is the seam a session calls at its OWN turn boundary,
// before final success copy is emitted, instead of learning about the conflation
// from a scorecard run after the narration already landed.
func CheckTurnBoundaryLatest(opts Options) TurnBoundaryResult {
	path := latestTranscript(opts.normalize())
	if path == "" {
		return TurnBoundaryResult{AllowFinal: true, Reason: turnBoundaryUnreachable}
	}
	result := CheckTurnBoundary(readFile(path))
	result.Reachable = true
	result.Transcript = path
	return result
}

// latestTranscript picks the most recently modified transcript across this
// workspace's Claude project roots — the live session's, at the moment the check
// runs. Empty when none is reachable.
func latestTranscript(opts Options) string {
	best := ""
	var bestMod time.Time
	for _, root := range transcriptRoots(opts) {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".jsonl") {
				continue
			}
			info, err := entry.Info()
			if err != nil {
				continue
			}
			if mod := info.ModTime().UTC(); best == "" || mod.After(bestMod) {
				best, bestMod = filepath.Join(root, entry.Name()), mod
			}
		}
	}
	return best
}
