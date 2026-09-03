package gateway

import (
	"strings"
)

// StopHoldbackBuffer is an incremental stop-string holdback buffer for streaming generation
// (#10719, borrowed from vLLM detokenizer).
// It delays emitting up to max(len(stop)) - 1 characters during streaming generation until
// either the stop condition resolves (and the stop sequence is trimmed) or generation finishes
// (and the held buffer is flushed) — preventing partial stop markers from prematurely leaking
// across SSE chunk boundaries to streaming clients.
type StopHoldbackBuffer struct {
	stops       []string
	maxHoldback int
	buffered    strings.Builder
	stopped     bool
	matchedStop string
}

// NewStopHoldbackBuffer creates an incremental holdback buffer for the specified stop sequences.
func NewStopHoldbackBuffer(stops []string) *StopHoldbackBuffer {
	maxLen := 0
	cleanStops := make([]string, 0, len(stops))
	for _, s := range stops {
		if s != "" {
			cleanStops = append(cleanStops, s)
			if len(s) > maxLen {
				maxLen = len(s)
			}
		}
	}
	holdback := 0
	if maxLen > 1 {
		holdback = maxLen - 1
	}
	return &StopHoldbackBuffer{
		stops:       cleanStops,
		maxHoldback: holdback,
	}
}

// Append processes an incoming text chunk delta during streaming.
// It returns any text that is safe to emit immediately.
// If a stop sequence is encountered, it sets Stopped() = true, records MatchedStop(),
// and returns the text prior to the stop sequence (never emitting the stop sequence).
// Any trailing text that could be a partial prefix of a stop sequence is held in buffer.
func (b *StopHoldbackBuffer) Append(chunk string) string {
	if b == nil || chunk == "" || b.stopped {
		return ""
	}
	b.buffered.WriteString(chunk)
	current := b.buffered.String()

	// Find the earliest matching stop string in the accumulated text
	earliestIdx := -1
	var earliestStop string
	for _, stop := range b.stops {
		idx := strings.Index(current, stop)
		if idx >= 0 {
			if earliestIdx == -1 || idx < earliestIdx || (idx == earliestIdx && len(stop) > len(earliestStop)) {
				earliestIdx = idx
				earliestStop = stop
			}
		}
	}

	if earliestIdx >= 0 {
		// Stop sequence found: terminate stream holdback
		b.stopped = true
		b.matchedStop = earliestStop
		toEmit := current[:earliestIdx]
		b.buffered.Reset()
		return toEmit
	}

	// No full stop string matched yet.
	// Emit text up to (len(current) - maxHoldback), keeping the tail buffered in case it forms a partial stop.
	if b.maxHoldback > 0 && len(current) > b.maxHoldback {
		safeLen := len(current) - b.maxHoldback
		toEmit := current[:safeLen]
		tail := current[safeLen:]
		b.buffered.Reset()
		b.buffered.WriteString(tail)
		return toEmit
	} else if b.maxHoldback == 0 {
		toEmit := current
		b.buffered.Reset()
		return toEmit
	}

	return ""
}

// Flush is called when streaming finishes. If generation was not terminated by a stop
// sequence, Flush emits all remaining buffered characters.
func (b *StopHoldbackBuffer) Flush() string {
	if b == nil || b.stopped {
		return ""
	}
	tail := b.buffered.String()
	b.buffered.Reset()
	return tail
}

// Stopped reports whether streaming was halted by a matched stop sequence.
func (b *StopHoldbackBuffer) Stopped() bool {
	if b == nil {
		return false
	}
	return b.stopped
}

// MatchedStop returns the matched stop sequence that halted generation, if any.
func (b *StopHoldbackBuffer) MatchedStop() string {
	if b == nil {
		return ""
	}
	return b.matchedStop
}
