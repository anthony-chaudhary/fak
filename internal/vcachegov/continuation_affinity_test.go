package vcachegov

// The continuation-affinity bridge test was removed with the affinity router
// (#5190): it fed session continuation keys through the dead AffinityHeader.
// The session-side property it exercised — a budget-reset continuation preserving
// its parent's cache-affinity key — is covered by
// TestContinuationCacheAffinityDecisionPreservesAcrossContinuations in
// internal/session.
