package gateway

// The UPGRADE_NEVER_FIRED live watchdog (#3620, cache-verify epic #3569): the managed-cache
// posture surfaces claim the 1h-TTL upgrade lever is ACTIVE, but the counter family
// (fak_gateway_cache_ttl_upgrade_total) witnesses only ATTEMPTS — a session can stay
// posture=ACTIVE for its whole life while every attempt is refused (volatile_head /
// no_stable_breakpoint), so the operator pays the 5m re-write the posture claimed to remove
// and nothing says so. The fold below turns the running upgraded-vs-refused outcome ratio
// into an explicit finding both live surfaces raise: the /debug/vars managed_cache.finding
// field (managedCacheVars) and the guard exit banner (cmd/fak formatAuditSummary).
//
// The fold is deliberately WITNESSED-only: refusal rows accrue only while the lever is on
// (see gatewayMetrics.ttlUpgrades) and only on a wire that has the Anthropic 1h-TTL lever
// (the upgrade transform never runs elsewhere), so a passive session, a lever-less
// (openai-responses) wire, or a cold process can never trip it — no posture flag needs to be
// threaded in, and the banner can raise the same finding from the summary alone.

// upgradeNeverFiredMinTurns is the refused-attempt floor the watchdog keys on: below it a
// zero-upgrade session is just short (the first turns of a run legitimately refuse until a
// stable head exists), at or past it the lever has had the ~3 requests its own break-even
// story needs (see cmd/fak guard_managed_cache.go) and still never fired — the
// ACTIVE-but-never-fired pathology, not a cold start.
const upgradeNeverFiredMinTurns = 3

// TTLUpgradeAttempts is the total managed-cache 1h TTL-upgrade attempts this session
// witnessed: turns that actually fired (CacheTTLUpgraded, both authoring outcomes) plus
// every refused turn in the per-reason breakdown. This is the denominator of the
// upgraded-vs-refused ratio the #3620 watchdog tracks.
func (s AdjudicationSummary) TTLUpgradeAttempts() uint64 {
	n := s.CacheTTLUpgraded
	for _, c := range s.CacheTTLUpgradeReasons {
		n += c
	}
	return n
}

// UpgradeNeverFired reports the #3620 finding: the session accrued at least
// upgradeNeverFiredMinTurns upgrade-eligible turns (every one refused) and not one
// "upgraded" outcome was observed. A single fired upgrade clears it for the session's
// lifetime; a session below the floor, with the lever off, or on a wire without the
// 1h-TTL lever never raises it (no outcomes accrue there at all).
func (s AdjudicationSummary) UpgradeNeverFired() bool {
	return s.CacheTTLUpgraded == 0 && s.TTLUpgradeAttempts() >= upgradeNeverFiredMinTurns
}
