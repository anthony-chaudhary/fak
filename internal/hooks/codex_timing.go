package hooks

import (
	"encoding/json"
	"io"
	"strings"
	"time"
)

const codexHookTimingType = "hook_codex"

// CodexHookCorrelation carries the stable identifiers needed to join a hook
// observation back to the Codex session, turn, and tool call that caused it.
// Empty identifiers are omitted rather than invented.
type CodexHookCorrelation struct {
	SessionID string
	TurnID    string
	CallID    string
}

type codexHookTimingRecord struct {
	Type       string                    `json:"type"`
	Attachment codexHookTimingAttachment `json:"attachment"`
}

type codexHookTimingAttachment struct {
	Type       string `json:"type"`
	DurationMS int64  `json:"durationMs"`
	Hook       string `json:"hook"`
	SessionID  string `json:"sessionId,omitempty"`
	TurnID     string `json:"turnId,omitempty"`
	CallID     string `json:"callId,omitempty"`
}

// RunCodexHookTimed writes hook verdict bytes only to verdict and writes one
// trajectory-compatible timing attachment only to timing. The two returned
// errors stay separate so telemetry failure cannot be mistaken for a hook
// decision. time.Since uses the monotonic reading embedded by time.Now.
func RunCodexHookTimed(verdict, timing io.Writer, hook string, correlation CodexHookCorrelation, run func(io.Writer) error) (verdictErr, timingErr error) {
	started := time.Now()
	verdictErr = run(verdict)
	timingErr = writeCodexHookTiming(timing, hook, correlation, time.Since(started))
	return verdictErr, timingErr
}

func writeCodexHookTiming(w io.Writer, hook string, correlation CodexHookCorrelation, elapsed time.Duration) error {
	if elapsed < 0 {
		elapsed = 0
	}
	record := codexHookTimingRecord{
		Type: "attachment",
		Attachment: codexHookTimingAttachment{
			Type:       codexHookTimingType,
			DurationMS: elapsed.Milliseconds(),
			Hook:       strings.TrimSpace(hook),
			SessionID:  strings.TrimSpace(correlation.SessionID),
			TurnID:     strings.TrimSpace(correlation.TurnID),
			CallID:     strings.TrimSpace(correlation.CallID),
		},
	}
	return json.NewEncoder(w).Encode(record)
}
