package gateway

// recordAdjudicationOutcome keeps the existing metrics fold and evaluates the
// default-off reload canary at the same once-per-served-turn seam.
func (s *Server) recordAdjudicationOutcome(signal adjudicationOutcomeSignal, fingerprint string) {
	s.metrics.recordAdjudicationOutcome(signal, fingerprint)
	if s.policyCanaryTurns <= 0 {
		return
	}
	s.policyCanaryMu.Lock()
	if s.policyCanaryRemaining == 0 || s.policyCanaryRollback == nil {
		s.policyCanaryMu.Unlock()
		return
	}
	s.policyCanaryRemaining--
	if signal == adjudicationOutcomeDenyAll {
		s.policyCanaryConsecutive++
	} else {
		s.policyCanaryConsecutive = 0
	}
	if s.policyCanaryConsecutive >= s.policyCanaryTurns {
		rollback := s.policyCanaryRollback
		s.policyCanaryRollback = nil
		s.policyCanaryRemaining = 0
		consecutive := s.policyCanaryConsecutive
		s.policyCanaryConsecutive = 0
		s.policyCanaryMu.Unlock()
		rollback()
		s.metrics.policyCanaryRollbacks.Add(1)
		s.policyCanaryRolledBack.Store(true)
		s.logf("gateway: POLICY CANARY ROLLBACK: deny-all streak reached %d served turns", consecutive)
		return
	}
	if s.policyCanaryRemaining == 0 {
		s.policyCanaryRollback = nil
		s.policyCanaryConsecutive = 0
	}
	s.policyCanaryMu.Unlock()
}

func (s *Server) armPolicyCanary(rollback func()) {
	if s.policyCanaryTurns <= 0 || rollback == nil {
		return
	}
	s.policyCanaryMu.Lock()
	s.policyCanaryRemaining = s.policyCanaryTurns
	s.policyCanaryConsecutive = 0
	s.policyCanaryRollback = rollback
	s.policyCanaryMu.Unlock()
}
