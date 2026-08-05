package stallscan

// arming.go — the "was this host actually MEASURED?" question, as a pure decision.
//
// WHY THIS EXISTS. A churn gate that reads a signal ledger has two failure modes that
// look IDENTICAL from the admission path: a genuinely calm host, and a host nobody is
// measuring. Both present as "no burst". The first is a fact; the second is the absence
// of a fact. Folding them together is how an anti-churn reaper ships, passes its tests,
// and then sits inert for weeks while the box keeps freezing — every tick reporting a
// clean bill of health it never actually took.
//
// This is the general shape of a GHOST slowdown: not a metric that reads bad, but a
// metric that reads FINE because nothing populated it. The defense is to make
// unmeasured a first-class state that is never spelled the same way as healthy. So the
// rule this file encodes is:
//
//	absence of signal is NEVER evidence of absence of load.
//
// Callers ask Armed() before trusting a reading, and render Reason() when it is false so
// an operator sees "not measured (no ledger)" rather than a silent zero. The gate still
// FAILS OPEN — an unmeasured host must not block dispatch — but it now fails open
// LOUDLY, which is the whole difference between a knob that works and a knob that was
// never connected.
//
// Pure by construction: the caller does the file I/O and clock read, this decides. That
// keeps every state (missing / garbled / stale / disabled / armed) testable without a
// filesystem, and keeps the anti-churn path from itself spawning anything.

import (
	"fmt"
	"time"
)

// ArmState names why a churn signal is or is not usable. Its zero value is the
// empty string, which Normalized maps onto ArmStateMissing — so a struct nobody
// filled in reads as "not measured" rather than as a blank that renders like a
// pass. That is the failure direction this whole file exists to enforce.
type ArmState string

// Normalized maps the empty zero value onto ArmStateMissing. Renderers must use it
// rather than the raw field: an unpopulated Arming would otherwise serialize as
// `"state": ""`, an unexplained blank that a reader skims straight past — the exact
// silence that let an inert gate pass for a healthy one.
func (s ArmState) Normalized() ArmState {
	if s == "" {
		return ArmStateMissing
	}
	return s
}

const (
	// ArmStateMissing: no reading at all — no self-monitor has ever written one,
	// or the ledger was removed. The overwhelmingly common inert case.
	ArmStateMissing ArmState = "missing"
	// ArmStateGarbled: a reading exists but could not be parsed. Distinct from
	// missing because it means a writer IS running and producing the wrong shape
	// — a bug to fix, not a monitor to install.
	ArmStateGarbled ArmState = "garbled"
	// ArmStateStale: a well-formed reading that is older than the freshness bound.
	// The burst it saw may have long drained, so it must not gate — but it also
	// must not be reported as calm.
	ArmStateStale ArmState = "stale"
	// ArmStateDisabled: an operator turned the term off. Inert on purpose, which
	// is the one inert state that needs no alarm.
	ArmStateDisabled ArmState = "disabled"
	// ArmStateArmed: a fresh, parsed reading. Only in this state does a zero burst
	// count actually mean "this host is calm".
	ArmStateArmed ArmState = "armed"
)

// Arming is the verdict: whether a churn reading may be trusted, and if not, why.
type Arming struct {
	State ArmState `json:"state"`
	// AgeSeconds is how old the reading is, when there was one to age.
	AgeSeconds float64 `json:"age_seconds,omitempty"`
	// SpawnBurst is the measured burst the reading carried. Meaningful ONLY when
	// State is ArmStateArmed; readers must gate on Armed() before using it.
	SpawnBurst int `json:"spawn_burst,omitempty"`
	// SpawnWindowSeconds is the span SpawnBurst was counted over, carried through
	// so the consumer can compare a RATE. Zero means the reading predates the
	// window being recorded, in which case the count is only comparable against
	// the legacy count threshold. Carried alongside the count rather than folded
	// into a pre-divided rate so a reader can still audit the raw numbers.
	SpawnWindowSeconds float64 `json:"spawn_window_seconds,omitempty"`
	// Detail is a short human-readable explanation, always populated.
	Detail string `json:"detail"`
}

// Armed reports whether the reading may be trusted as a measurement of this host.
// Every state other than ArmStateArmed — including the zero value — answers false.
func (a Arming) Armed() bool { return a.State == ArmStateArmed }

// Status is the state to render: the field, with the zero value normalized to
// ArmStateMissing so an unpopulated struct names itself instead of going blank.
func (a Arming) Status() ArmState { return a.State.Normalized() }

// Reason renders the arming state for an operator or a tick payload.
func (a Arming) Reason() string { return a.Detail }

// LedgerRead is what the impure shell managed to get from the signal ledger. The
// zero value means "nothing found", which classifies as ArmStateMissing.
type LedgerRead struct {
	// Found is true when a candidate record was located at all.
	Found bool
	// Parsed is true when that record decoded into a usable reading. Found &&
	// !Parsed is the garbled case.
	Parsed bool
	// Timestamp is the reading's own clock stamp; zero when unparsed/absent.
	Timestamp time.Time
	// SpawnBurst is the measured gross birth count the reading carried.
	SpawnBurst int
	// SpawnWindowSeconds is the span that count covers; 0 when the reading did
	// not record one (older ledger records).
	SpawnWindowSeconds float64
}

// ClassifyArming decides whether a churn reading is trustworthy. Pure: same input,
// same output, no I/O and no clock read of its own (now is passed in).
//
// enabled=false means an operator disabled the term; that short-circuits every other
// state because an intentionally-off gate needs no missing-monitor alarm. A freshness
// of zero or less disables the staleness check, in which case any parsed reading arms.
func ClassifyArming(r LedgerRead, now time.Time, freshness time.Duration, enabled bool) Arming {
	if !enabled {
		return Arming{
			State:  ArmStateDisabled,
			Detail: "churn term disabled by configuration — host churn is not being gated on",
		}
	}
	if !r.Found {
		return Arming{
			State:  ArmStateMissing,
			Detail: "no stallscan reading on this host — churn is NOT MEASURED, so a calm verdict here is an absence of data, not an absence of load; start the self-monitor (`fak stallscan --watch`) to arm it",
		}
	}
	if !r.Parsed {
		return Arming{
			State:  ArmStateGarbled,
			Detail: "stallscan reading present but unparseable — a writer is running and emitting the wrong shape; churn is NOT MEASURED until its format is fixed",
		}
	}
	age := now.Sub(r.Timestamp)
	ageSec := age.Seconds()
	if ageSec < 0 {
		ageSec = 0 // a reading stamped slightly ahead of us is not "negative age"
	}
	if freshness > 0 && age > freshness {
		return Arming{
			State:      ArmStateStale,
			AgeSeconds: ageSec,
			Detail: fmt.Sprintf("stallscan reading is %.0fs old (older than the %.0fs freshness bound) — the burst it saw may have drained, so it cannot gate; churn is effectively NOT MEASURED right now",
				ageSec, freshness.Seconds()),
		}
	}
	armed := Arming{
		State:              ArmStateArmed,
		AgeSeconds:         ageSec,
		SpawnBurst:         r.SpawnBurst,
		SpawnWindowSeconds: r.SpawnWindowSeconds,
	}
	// Render the rate when the window is known, because the rate is the number the
	// threshold is actually calibrated in. Keep the raw count and window in the
	// text either way so the derived figure stays auditable.
	if r.SpawnWindowSeconds > 0 {
		armed.Detail = fmt.Sprintf("armed on a %.0fs-old reading: %.0f process(es)/sec born (%d in %.2fs)",
			ageSec, float64(r.SpawnBurst)/r.SpawnWindowSeconds, r.SpawnBurst, r.SpawnWindowSeconds)
		return armed
	}
	armed.Detail = fmt.Sprintf("armed on a %.0fs-old reading: %d process(es) born in the last sample window (window not recorded, so this count cannot be read as a rate)", ageSec, r.SpawnBurst)
	return armed
}
