package gateway

import "fmt"

// admitLocked reserves the shared pool before charging local admission state. Another
// controller can draw after the budget snapshot, so only a full Draw authorizes admission.
// A failed partial draw is returned without granting the request. Caller holds c.mu.
func (c *AdmissionController) admitLocked(req SeqRequest) *AdmissionError {
	if c.pool != nil {
		granted, ok := c.pool.Draw(req.Tokens)
		if !ok {
			c.pool.Return(granted)
			return &AdmissionError{
				Verdict: VerdictShed,
				Budget:  "fleet_tokens",
				Reason:  fmt.Sprintf("session pool budget exhausted (%d granted < %d requested)", granted, req.Tokens),
			}
		}
	}
	c.running[req.TraceID] = req
	c.tokens += req.Tokens
	c.stats.Admitted++
	c.chargeTenantTokensLocked(req)
	return nil
}

// recordReservationShedLocked keeps the fast-path refusal counter and receipt together.
// Queued requests retain their place instead and must not be counted as shed.
func (c *AdmissionController) recordReservationShedLocked(req SeqRequest, err *AdmissionError) {
	c.stats.Shed++
	c.recordDecisionLocked(req, VerdictShed, err.Budget, err.Reason)
}
