package leaseref

// ttl_floor.go closes the "no expiry" hole in the lock-lease contract (fak-private#3076).
//
// THE DEFECT. Record.TTLSeconds == 0 used to mean "no expiry": expired() short-circuited
// false, so such a lease never entered Live's expired partition, Reap (which deletes only
// what Live called expired) could never collect it, and AcquireFenced could never take it
// over. A holder that crashed without releasing therefore wedged its lane for the life of
// the repository. The remotes accumulated eleven such refs, 16-30 days old, every one of
// them written through `fak leaseref acquire` with its old `--ttl 0` default. Nothing in
// the tree relies on a forever-lease: every production writer that cares passes a real TTL
// (dispatch tick, guard arbitrate, knownbad, the coordinator wire already refuses ttl<=0),
// and the contract and intent leases already clamp ttl<=0 to a one-hour default.
//
// THE FIX, two halves:
//
//   - WRITE side: every lock-lease write (Acquire, and the fenced acquire/renew CAS in
//     commitFenced) clamps a non-positive TTL to DefaultLeaseTTLSeconds, so a ttl<=0 lease
//     can no longer be minted. Clamping rather than refusing keeps every existing caller
//     (CLI default, HTTP lease-write endpoint, scripts) working, matching the clamp the
//     contract (DefaultContractTTLSeconds) and intent (DefaultIntentTTLSeconds) writers use.
//     A renew of a legacy ttl-0 lease upgrades it to a bounded lease the same way.
//
//   - READ side: a LEGACY ttl<=0 record (written by an older binary, or still arriving from
//     a peer that has not upgraded) is judged by AGE: it is expired once its last activity
//     (the renew-aware effectiveActiveAt) is at least LegacyNoTTLMaxAgeSeconds old, so the
//     normal reaper collects it and a peer can take the lane over through AcquireFenced's
//     generation-bumping transition. A record with no acquired/renewed stamp has no age and
//     fails closed to not-expired, the same posture the audit's age rung takes.

// DefaultLeaseTTLSeconds is the lifetime a lock lease gets when its writer supplies no
// positive TTL: one hour, identical to DefaultContractTTLSeconds and
// DefaultIntentTTLSeconds. A holder that needs longer renews (RenewFenced) or asks for an
// explicit --ttl.
const DefaultLeaseTTLSeconds int64 = 3600

// LegacyNoTTLMaxAgeSeconds is the age past which a LEGACY ttl<=0 record counts as expired:
// seven days, 168x DefaultLeaseTTLSeconds and ~250x the longest legitimate un-renewed hold
// in the tree (a dispatch worker's ~40 min lane lease). It is deliberately far more generous
// than the audit's 24 h age rung (leaserefNoTTLStaleAgeS in cmd/fak), because crossing this
// bound makes the record REAPABLE (a ref delete), not merely reported.
const LegacyNoTTLMaxAgeSeconds int64 = 7 * 24 * 60 * 60

// normalizeLeaseTTL returns ttl, or DefaultLeaseTTLSeconds when ttl is not positive. It is
// the single write-side clamp every lock-lease write path routes through.
func normalizeLeaseTTL(ttl int64) int64 {
	if ttl <= 0 {
		return DefaultLeaseTTLSeconds
	}
	return ttl
}
