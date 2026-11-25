package model

// RecordVerificationOutcome updates execution statistics for an externally
// driven verification round after its correction/bonus emission decision.
func (e *SpeculativeEngine) RecordVerificationOutcome(generated, accepted, rollback int, bonusEmitted bool) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.stats.DraftTokensGenerated += generated
	e.stats.DraftTokensAccepted += accepted
	if bonusEmitted {
		e.stats.BonusTokensEmitted++
	}
	e.stats.VerificationRounds++
	e.stats.RollbackTokensCount += rollback
	if e.stats.VerificationRounds > 0 {
		e.stats.MeanAcceptanceRate = float64(e.stats.DraftTokensAccepted) / float64(e.stats.VerificationRounds)
	}
	e.recordRollingAcceptanceLocked(generated, accepted)
}

// recordRollingAcceptanceLocked feeds one round's accepted/rejected draft tokens
// into the trailing window and trips the serial fallback once the window is full
// and its acceptance rate is below the configured floor. Callers hold e.mu.
func (e *SpeculativeEngine) recordRollingAcceptanceLocked(generated, accepted int) {
	if generated <= 0 {
		return
	}
	if accepted < 0 {
		accepted = 0
	}
	if accepted > generated {
		accepted = generated
	}
	e.ensureWindowLocked()
	for i := 0; i < generated; i++ {
		outcome := i < accepted
		if len(e.windowOutcomes) < e.cfg.AcceptanceWindow {
			e.windowOutcomes = append(e.windowOutcomes, outcome)
			continue
		}
		e.windowOutcomes[e.windowHead] = outcome
		e.windowHead = (e.windowHead + 1) % e.cfg.AcceptanceWindow
	}
	e.evaluateFallbackLocked()
}

// ensureWindowLocked initializes the rolling window storage and clamps a stale
// head index after a config change. Callers hold e.mu.
func (e *SpeculativeEngine) ensureWindowLocked() {
	if e.cfg.AcceptanceWindow <= 0 {
		e.cfg.AcceptanceWindow = 32
	}
	if cap(e.windowOutcomes) < e.cfg.AcceptanceWindow {
		grown := make([]bool, len(e.windowOutcomes), e.cfg.AcceptanceWindow)
		copy(grown, e.windowOutcomes)
		e.windowOutcomes = grown
	}
	if len(e.windowOutcomes) > e.cfg.AcceptanceWindow {
		e.windowOutcomes = e.windowOutcomes[len(e.windowOutcomes)-e.cfg.AcceptanceWindow:]
		e.windowHead = 0
	}
	if len(e.windowOutcomes) > 0 && e.windowHead >= len(e.windowOutcomes) {
		e.windowHead = 0
	}
}

// evaluateFallbackLocked decides whether the rolling window trips the serial
// fallback. The window must be full before the floor is evaluated, so a partial
// window warns without aborting. Once tripped the fallback is sticky until
// Reset, so a single healthy round cannot silently resurrect speculation
// mid-request. Callers hold e.mu.
func (e *SpeculativeEngine) evaluateFallbackLocked() {
	if !e.cfg.FallbackToSerial || e.cfg.MinAcceptanceRate <= 0 {
		return
	}
	if e.inFallback {
		return
	}
	if len(e.windowOutcomes) < e.cfg.AcceptanceWindow {
		return
	}
	accepted := 0
	for _, ok := range e.windowOutcomes {
		if ok {
			accepted++
		}
	}
	rate := float64(accepted) / float64(len(e.windowOutcomes))
	if rate < e.cfg.MinAcceptanceRate {
		e.inFallback = true
		e.tripwireTripped = true
		e.fallbackReason = "rolling acceptance rate below floor"
	}
}

// ResetAcceptanceMonitor clears only the rolling acceptance window and any
// active serial fallback, leaving cumulative statistics and boundary logits
// intact. The planner calls this at the start of each request so one degraded
// request cannot force every later request onto serial decode.
func (e *SpeculativeEngine) ResetAcceptanceMonitor() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.windowOutcomes = e.windowOutcomes[:0]
	e.windowHead = 0
	e.inFallback = false
	e.fallbackReason = ""
	e.tripwireTripped = false
}
