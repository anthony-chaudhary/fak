package routing

// ExhaustedWalkRetryable is the fabric's verdict on a walk that ended with
// every rung failed and no upstream answer relayed: whether the caller must
// read it as a capacity shortfall to retry later rather than a dead end.
//
// failures are the classified failures of every dialed rung; starvedRungs counts
// rungs passed over because their credential pool had no eligible key (a
// credential verdict platform/keyring owns: rate-limited keys cool down, and a
// paused or walled key is a capacity loss to the caller, never a verdict on its
// request).
//
// It is retryable when at least one rung failed transiently or was starved and
// no rung returned a FATAL verdict. Every other class reaching an exhausted walk
// is FAILOVER_NEXT, i.e. backend-specific by the class's own Action: a missing
// model, a credit wall, or a substitute rung's rejected payload names THAT rung,
// so relaying it would hide the transient rung that may serve the same request
// once it recovers (live 2026-10-06: three Halos 503, then opencode-go's 400
// reached the client, which opencode treats as a non-retryable bad request).
// A FATAL verdict is a caller bug every engine shares and stays the caller's.
func ExhaustedWalkRetryable(failures []Failure, starvedRungs int) bool {
	capacity := starvedRungs > 0
	for _, f := range failures {
		if f.Action == ActionFatal {
			return false
		}
		if f.Class.Transient() {
			capacity = true
		}
	}
	return capacity
}
