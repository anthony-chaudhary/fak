package fleetmon

import "time"

// ProgressState separates productive liveness from process liveness. It is
// derived only from durable worker artifacts; CPU use and unrelated filesystem
// mtimes are deliberately excluded.
type ProgressState string

const (
	Progressing       ProgressState = "PROGRESSING"
	QuietWithinWindow ProgressState = "QUIET_WITHIN_WINDOW"
	Wedged            ProgressState = "WEDGED"
	ProgressUnknown   ProgressState = "UNKNOWN"
)

// ProgressEvidence is the newest durable progress marker observed for a worker.
// Provenance names the artifact and is required whenever At is known.
type ProgressEvidence struct {
	At         time.Time
	Provenance string
}

// EvaluateProgress classifies a live worker against a declared progress window.
// moved means the durable artifact advanced since the previous inventory sample.
func EvaluateProgress(live, livenessKnown, moved bool, evidence ProgressEvidence, now time.Time, window time.Duration) ProgressState {
	if !livenessKnown || !live || evidence.At.IsZero() || evidence.Provenance == "" {
		return ProgressUnknown
	}
	if moved {
		return Progressing
	}
	if window <= 0 {
		window = DefaultThresholds().StaleTranscript
	}
	if now.Sub(evidence.At) <= window {
		return QuietWithinWindow
	}
	return Wedged
}
