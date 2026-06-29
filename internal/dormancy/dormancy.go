// Package dormancy is the dormancy clock + horizon bucketer: the one place that
// turns "how long was this agent/session/lease off?" into a first-class, measured
// quantity. It is the keystone every other random-time-horizons rung keys on
// (epic #1178): a durable, monotonic LastActiveAt Stamp and a PURE Horizon bucketer
// that maps a dormancy gap onto the band {warm, cool, cold, frozen, ancient} whose
// thresholds map to what actually decays at that scale.
//
// # The gap it closes
//
// fak already has every ORGAN of dormancy handling, but none treats the LENGTH of the
// off-gap as first-class: internal/resume prices a cold resume from idle-vs-cache-TTL
// (but shadow-only), internal/loopmgr records each loop's LastEventUnixNano,
// internal/leaseref stamps a lease's AcquiredAt + TTL, and internal/session tracks a
// run-state with no last-active time at all. Every one of them runs the SAME path
// whether the agent slept five minutes or five months. This leaf is the shared notion
// of "how long dormant" the others were missing — one durable stamp and one pure
// bucket, so a longer gap can drive strictly more revalidation before the first
// post-wake action is admitted (the Phase-2/3 rehydration rungs, #1181-#1192).
//
// # The bands (what has decayed at each scale)
//
//	warm    < ~5 min   nothing — the provider prompt cache is still warm
//	cool    < ~1 h     prompt cache cold (TTL aged out), KV gone
//	cold    < ~24 h    + credentials may be expired, recalled memory aging
//	frozen  < ~30 d    + lease re-granted, plan stale, repo HEAD moved on
//	ancient ≥ ~30 d    + model renamed/deprecated, the world materially changed
//
// The first two thresholds are NOT arbitrary: ~5 min and ~1 h are exactly the provider
// prompt-cache TTLs internal/resume already encodes (its TTL5m.Seconds()==300 and
// TTL1h.Seconds()==3600 — the Anthropic default ephemeral breakpoint and the extended
// tier). ~24 h and ~30 d are the credential-expiry and "world materially changed"
// scales. Like the cache multipliers internal/resume redeclares, these are stable
// published/operational economics, not a fak measurement, so this leaf states them as
// named constants rather than importing the cost-projection package.
//
// # Honest fences
//
//   - PURE and stdlib-only. Bucket is a total function of a gap; the Stamp is plain
//     unix-nanos arithmetic. No clock is read inside this package, no I/O, no model —
//     the caller supplies `now`, which is what makes the bucket "derivable without I/O"
//     and the time-travel harness (#1192) able to fast-forward hours→months with an
//     injected clock and zero real waits.
//   - MEASURE, do not act. This leaf is Phase 1 (#1179): it computes the band and
//     nothing else. No rehydration rung, lease fence, credential refresh, or cache
//     decision lives here — those compose this band in Phase 2/3. Adding the field is
//     a no-behavior-change increment: the stamp has zero readers until a consumer acts.
//   - CONSERVATIVE on ambiguity. An UNKNOWN gap (a never-stamped record) buckets to
//     Ancient, not Warm: when we cannot measure how long something slept, the safe
//     re-entry is to revalidate everything, never to assume the cache is warm. And the
//     band boundaries are half-open on the COLD side (a gap exactly at a threshold
//     tips to the colder band), mirroring internal/resume's `idle >= cutoff` cold rule.
//   - MONOTONIC stamp. Refresh never moves a stamp backwards on a wall-clock that
//     stepped back (NTP slew, a restore onto a skewed host), so an elapsed gap is never
//     spuriously inflated by a backwards clock — the monotonic-clock discipline the
//     lease/fencing SOTA (etcd/Chubby) calls for, applied to the dormancy stamp.
//
// This is a tier-1 foundation leaf: stdlib-only, imports nothing internal, registers
// nothing, off the request path. session/loop/lease surface a LastActiveAt from it; the
// pure decision lives here.
package dormancy

import "time"

// The band thresholds, as named durations. The first two are the provider prompt-cache
// TTLs internal/resume encodes (300s / 3600s); the last two are the credential-expiry
// and world-changed scales. A gap is bucketed half-open on the cold side: gap < WarmMax
// is warm, WarmMax <= gap < CoolMax is cool, and so on — so a gap landing exactly on a
// threshold takes the COLDER band (the same conservative `idle >= cutoff` tie-break
// internal/resume uses for its cold/warm posture).
const (
	// WarmMax — under ~5 min nothing has decayed: the provider prompt cache is still
	// warm (the 5-minute ephemeral TTL internal/resume's TTL5m.Seconds() encodes).
	WarmMax = 5 * time.Minute
	// CoolMax — under ~1 h the prompt cache has aged out and the KV is gone, but creds
	// and recall are still fresh (the 1-hour extended TTL, resume's TTL1h.Seconds()).
	CoolMax = time.Hour
	// ColdMax — under ~24 h credentials may have expired and recalled memory is aging.
	ColdMax = 24 * time.Hour
	// FrozenMax — under ~30 d a lease has likely been re-granted, the plan is stale, and
	// repo HEAD has moved on; at or past it the world has materially changed (Ancient).
	FrozenMax = 30 * 24 * time.Hour
)

// Horizon is the dormancy band a gap falls in — the single input every rehydration rung
// keys on. The bands are ORDERED warm < cool < cold < frozen < ancient (a higher value
// is a longer gap and strictly more decayed), so a consumer can gate "run this rung only
// past Cold" with a plain comparison (see AtLeast). The zero value is Warm, the
// least-stale band; a deliberately-unknown gap is Ancient, the most-stale (see Bucket).
type Horizon uint8

const (
	// Warm: < WarmMax. Nothing decayed — resume verbatim.
	Warm Horizon = iota
	// Cool: [WarmMax, CoolMax). Prompt cache cold, KV gone — re-warm the plan.
	Cool
	// Cold: [CoolMax, ColdMax). + creds may be expired, recall aging — refresh + revalidate.
	Cold
	// Frozen: [ColdMax, FrozenMax). + lease re-granted, plan stale — fence + freshness-check.
	Frozen
	// Ancient: >= FrozenMax (and the UNKNOWN-gap band). + model/world changed — full revalidation.
	Ancient
)

// The canonical lowercase wire tokens — the single definition of the strings this band
// serializes to (a ledger row, a metric label, a fak dormancy view). Untyped string
// constants so a caller can use them in a typed-constant context without re-spelling.
const (
	TokenWarm    = "warm"
	TokenCool    = "cool"
	TokenCold    = "cold"
	TokenFrozen  = "frozen"
	TokenAncient = "ancient"
)

// Bucket is THE pure horizon bucketer: the band a dormancy gap falls in, with no clock
// and no I/O. Boundaries are half-open on the cold side (gap < WarmMax is Warm, a gap
// exactly at a threshold tips to the colder band), mirroring internal/resume's
// `idle >= cutoff` cold rule. A NEGATIVE gap — a now earlier than the stamp, i.e. a
// backwards wall-clock reading — is treated as Warm (zero elapsed): a clock that ran
// backwards is never evidence of staleness. This is the function the issue names
// "Horizon(gap)" and the one the table test walks over its boundaries.
func Bucket(gap time.Duration) Horizon {
	switch {
	case gap < WarmMax: // includes negative (backwards-clock) gaps: not-yet-stale
		return Warm
	case gap < CoolMax:
		return Cool
	case gap < ColdMax:
		return Cold
	case gap < FrozenMax:
		return Frozen
	default:
		return Ancient
	}
}

// String renders a Horizon as its canonical lowercase wire token. An out-of-range value
// renders "unknown" rather than panicking — a wire-derived band is never trusted to be
// in range.
func (h Horizon) String() string {
	switch h {
	case Warm:
		return TokenWarm
	case Cool:
		return TokenCool
	case Cold:
		return TokenCold
	case Frozen:
		return TokenFrozen
	case Ancient:
		return TokenAncient
	}
	return "unknown"
}

// ParseHorizon maps a wire token back to a Horizon. The bool is false for any token
// outside the closed band vocabulary, so a caller fails closed at the edge rather than
// coercing an unknown string to Warm.
func ParseHorizon(s string) (Horizon, bool) {
	switch s {
	case TokenWarm:
		return Warm, true
	case TokenCool:
		return Cool, true
	case TokenCold:
		return Cold, true
	case TokenFrozen:
		return Frozen, true
	case TokenAncient:
		return Ancient, true
	}
	return 0, false
}

// AtLeast reports whether this band is at least as stale as o — the gate a rehydration
// rung uses to fire "only past Cold" (h.AtLeast(Cold)) without re-deriving the ordering.
// It is a plain comparison on the ordered enum, exposed as a method so consumers key on
// the named relation instead of a bare `>=` on an int.
func (h Horizon) AtLeast(o Horizon) bool { return h >= o }

// Stamp is a durable LastActiveAt marker: the wall-clock instant a session/loop/lease
// was last active, stored as unix NANOSECONDS so it survives a process restart and a
// JSON round-trip. (A time.Time's monotonic reading does not survive serialization; a
// unix-nanos int does, and Refresh restores the monotonic-advance discipline at the
// arithmetic level.) The zero value is "never stamped / unknown" — IsZero true — and a
// zero Stamp buckets to Ancient, the conservative most-stale band.
type Stamp struct {
	// LastActiveUnixNano is the last-active instant in unix nanoseconds, or 0 = unset.
	LastActiveUnixNano int64 `json:"last_active_unix_nano,omitempty"`
}

// At constructs a Stamp marking the given instant. A zero time yields the zero
// (unknown) Stamp, so At(time.Time{}) and the zero value agree.
func At(t time.Time) Stamp {
	if t.IsZero() {
		return Stamp{}
	}
	return Stamp{LastActiveUnixNano: t.UnixNano()}
}

// FromUnixNano constructs a Stamp from a raw unix-nanos value — the form loopmgr's
// LastEventUnixNano already carries — without routing through a time.Time. A
// non-positive value yields the zero (unknown) Stamp.
func FromUnixNano(unixNano int64) Stamp {
	if unixNano <= 0 {
		return Stamp{}
	}
	return Stamp{LastActiveUnixNano: unixNano}
}

// FromUnix constructs a Stamp from a unix SECONDS value — the form leaseref.Record's
// AcquiredAt carries. A non-positive value yields the zero (unknown) Stamp.
func FromUnix(unixSec int64) Stamp {
	if unixSec <= 0 {
		return Stamp{}
	}
	return Stamp{LastActiveUnixNano: unixSec * int64(time.Second)}
}

// IsZero reports whether the Stamp is unset (never marked active). A zero Stamp is
// "unknown dormancy", distinct from a stamp of the unix epoch.
func (s Stamp) IsZero() bool { return s.LastActiveUnixNano == 0 }

// Time returns the last-active instant as a UTC time.Time, or the zero time when unset.
func (s Stamp) Time() time.Time {
	if s.IsZero() {
		return time.Time{}
	}
	return time.Unix(0, s.LastActiveUnixNano).UTC()
}

// GapAt returns the elapsed dormancy at instant now and whether it is KNOWN. ok is false
// for a zero (never-stamped) Stamp — the caller must not read the returned 0 as "warm";
// an unknown gap is unmeasured, not short. A now earlier than the stamp (a backwards
// wall-clock) clamps to 0, never a negative gap.
func (s Stamp) GapAt(now time.Time) (gap time.Duration, ok bool) {
	if s.IsZero() {
		return 0, false
	}
	d := now.UnixNano() - s.LastActiveUnixNano
	if d < 0 {
		d = 0
	}
	return time.Duration(d), true
}

// HorizonAt is the band this Stamp is in at instant now: Bucket of the elapsed gap. A
// zero (unknown) Stamp returns Ancient, NOT Warm — when we cannot measure how long
// something has been dormant, the conservative re-entry is to revalidate everything. It
// is the no-I/O derivation the acceptance asks for: a session/loop exposing its
// LastActiveAt can produce its band with this method and no clock of its own beyond the
// supplied now.
func (s Stamp) HorizonAt(now time.Time) Horizon {
	gap, ok := s.GapAt(now)
	if !ok {
		return Ancient // unknown gap => most-stale band (revalidate everything)
	}
	return Bucket(gap)
}

// Refresh returns a Stamp re-marked at now, MONOTONICALLY: if now is not strictly after
// the existing stamp (a wall-clock that stepped backwards via NTP slew or a restore onto
// a skewed host), the existing stamp is kept unchanged. A LastActiveAt that only ever
// advances is what keeps an elapsed gap from being spuriously inflated by a backwards
// clock — the monotonic-clock discipline the lease/fencing SOTA calls for, at the
// durable-stamp level. Refresh on a zero Stamp simply adopts now (the first mark).
func (s Stamp) Refresh(now time.Time) Stamp {
	n := now.UnixNano()
	if n <= s.LastActiveUnixNano {
		return s
	}
	return Stamp{LastActiveUnixNano: n}
}
