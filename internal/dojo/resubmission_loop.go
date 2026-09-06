package dojo

// ResubmissionLoopSample captures an observation slice for loop detection.
type ResubmissionLoopSample struct {
	SessionID     string
	Turn          int
	ToolName      string
	ArgumentsHash string
	WasElided     bool
	PromptTokens  int
	YieldIssued   bool
}

// DetectResubmissionLoops inspects a trace of session samples and returns counts of:
// 1. toolCallLoops: occurrences where an agent calls the identical tool + arguments >= 3 consecutive times after an elided result.
// 2. uncompactedYieldLoops: occurrences where a subturn yield was issued but the subsequent turn resubmitted with >= pre-yield tokens.
func DetectResubmissionLoops(samples []ResubmissionLoopSample) (toolCallLoops int, uncompactedYieldLoops int) {
	if len(samples) == 0 {
		return 0, 0
	}

	lastToolSig := ""
	toolStreak := 0
	elisionActive := false

	var pendingYieldTokens int
	awaitingYieldContinuation := false

	lastSessionID := ""
	for i, s := range samples {
		if i > 0 && s.SessionID != lastSessionID {
			lastToolSig = ""
			toolStreak = 0
			elisionActive = false
			awaitingYieldContinuation = false
			pendingYieldTokens = 0
		}
		lastSessionID = s.SessionID

		// 1. Tool-call resubmission loop detection
		if s.WasElided {
			elisionActive = true
		}

		sig := s.ToolName + ":" + s.ArgumentsHash
		if s.ToolName != "" {
			if sig == lastToolSig {
				toolStreak++
				if elisionActive && toolStreak == 3 {
					toolCallLoops++
				}
			} else {
				lastToolSig = sig
				toolStreak = 1
				elisionActive = s.WasElided
			}
		} else {
			lastToolSig = ""
			toolStreak = 0
		}

		// 2. Yield echo / uncompacted continuation detection
		if awaitingYieldContinuation {
			if s.PromptTokens >= pendingYieldTokens && pendingYieldTokens > 0 {
				uncompactedYieldLoops++
			}
			awaitingYieldContinuation = false
			pendingYieldTokens = 0
		}

		if s.YieldIssued {
			awaitingYieldContinuation = true
			pendingYieldTokens = s.PromptTokens
		}
	}

	return toolCallLoops, uncompactedYieldLoops
}
