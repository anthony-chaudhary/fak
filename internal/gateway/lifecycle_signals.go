package gateway

// LifecycleSignals is the in-process snapshot guard lifecycle actuators consume.
// It is deliberately the same information the /metrics compatibility surface renders,
// but reading it here cannot race a loopback socket or silently fail open.
type LifecycleSignals struct {
	DenyAllConsecutive         int
	DenyAllSameConsecutive     int
	DenyAllSameConsecutiveSeen bool
	ToolFeedbackConsecutive    int
	FakVerbCalls               int
	FakVerbCallsSeen           bool
	HarnessPosture             string
	RelayWouldRotate           bool
	RelayWouldRotateSeen       bool
}

// LifecycleSignalsSnapshot reads the guard lifecycle counters directly from this
// Server's metrics state. A nil/bare server returns a valid conservative snapshot.
func (s *Server) LifecycleSignalsSnapshot() LifecycleSignals {
	if s == nil || s.metrics == nil {
		return LifecycleSignals{
			DenyAllSameConsecutiveSeen: true,
			FakVerbCallsSeen:           true,
			HarnessPosture:             "block",
		}
	}
	_, deny := s.metrics.denyAllSnapshot()
	same := s.metrics.denyAllSameSnapshot()
	_, feedback := s.metrics.toolFeedbackSnapshot()
	verbs := s.metrics.fakVerbCallsSnapshot()
	coherence := s.metrics.harnessCoherenceSummary()
	return LifecycleSignals{
		DenyAllConsecutive:         int(deny),
		DenyAllSameConsecutive:     int(same),
		DenyAllSameConsecutiveSeen: true,
		ToolFeedbackConsecutive:    int(feedback),
		FakVerbCalls:               int(verbs),
		FakVerbCallsSeen:           true,
		HarnessPosture:             coherence.Posture,
	}
}
