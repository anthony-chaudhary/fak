package accounts

// removalguard.go — the removal-on-a-blip guard (#4676): the pure predicate that answers
// "would retiring this seat throw away one the probe ledger currently calls HEALTHY?"
//
// WHY THIS EXISTS. `fak accounts remove` is a soft, reversible tombstone — but it also
// FREEZES the health evidence. The fleet's probe loop stops sampling a tombstoned seat, so
// the last-ever probe is whatever stood at removal time, forever. A seat retired on a
// TRANSIENT blip therefore can never accumulate the fresh datum that would justify
// un-removing it: the removal itself guarantees the mistake cannot self-correct.
//
// Observed on july17-netra (2026-07-13): every live probe read OK — the last one 09:05Z,
// status OK, 1101ms — one session hit a transient INFRA_ORG_DISABLED at 02:45 (the SAME
// session had run fine on another seat), the seat was tombstoned at 15:45Z, and it then sat
// dark-but-healthy for ~27h with ZERO probes after the tombstone.
//
// The fix this file carries is the cheap half of the two the issue offers: rather than keep
// probing a dead seat (a new loop, new state, new failure modes), refuse to CREATE the frozen
// state silently. GradeRemoval gates the removal itself when the seat's newest probe is a
// fresh OK with no entitlement wall; RestoreNotice reads the SAME frozen last probe back out
// for seats already tombstoned, which is exactly the "removed-but-healthy — restore?" signal,
// because for a seat removed on a blip that frozen probe IS an OK.
//
// It is deliberately pure over plain values. The ledger reader (internal/accountprobe) owns
// which registry dir a host means and how a row is dated; this file owns only the judgment,
// so the judgment is testable in this lane with no filesystem and no clock.

import (
	"fmt"
	"strings"
	"time"
)

// Probe-status vocabulary the account prober writes (tools/account_probe.py's closed set,
// mirrored in Go by internal/accountprobe). Only the ones this predicate branches on are
// named; anything else falls through as "not a clean health signal", which never gates.
const (
	ProbeStatusOK     = "OK"     // the seat answered a live `claude -p` probe
	ProbeStatusAuth   = "AUTH"   // auth required / credential rejected
	ProbeStatusAccess = "ACCESS" // entitlement wall ("Claude subscription access disabled")
	ProbeStatusCredit = "CREDIT" // billing wall — out of credits
)

// entitlementWall is the closed set of probe statuses produced by an AUTHORIZATION or BILLING
// decision rather than by a transient fault or a quota window. It mirrors
// fleetaccounts.entitlementKinds: no amount of waiting turns a 403 oauth_org_not_allowed into
// a 200, so a seat whose newest probe is one of these has a real reason to be retired and this
// guard must never stand in the way of that (the "no false gate" half of #4676's DoD).
//
// LIMIT is deliberately ABSENT: a usage wall expires on its own, so a LIMIT probe is not
// evidence of a dead seat. It still does not GATE here — only a positive OK does — but it is
// reported as "no fresh health signal" rather than as a wall, so the printed reason stays true.
var entitlementWall = map[string]bool{
	ProbeStatusAuth:   true,
	ProbeStatusAccess: true,
	ProbeStatusCredit: true,
}

// ProbeFreshHours is how old a seat's newest probe row may be and still count as evidence
// about the seat's health NOW. 24h mirrors accountprobe.SeatCoverageMaxAgeMin (1440 minutes),
// the probe-coverage budget the roster already runs, so a seat leaves this guard's view at
// exactly the moment it leaves the roster's — one number, not two that can disagree.
const ProbeFreshHours = 24.0

// RestoreNoticeHours bounds how long after a tombstone the "removed-but-healthy — restore?"
// notice stays live. 48h is the window #4676 proposes: long enough to cover the ~27h the
// observed seat sat dark (a notice that expired first would have said nothing), short enough
// that a deliberate retirement stops nagging within a couple of days.
const RestoreNoticeHours = 48.0

// SeatProbe is what a decision path knows about ONE seat's newest probe-ledger row, as plain
// values. It is NOT the ledger reader's own type on purpose: this keeps the predicate pure and
// testable in this lane, and leaves "which registry dir does this host mean" — a genuinely
// ambiguous, host-shaped question — with the caller that can answer it.
type SeatProbe struct {
	// Status is the newest row's verdict in the prober's closed vocabulary
	// (OK / AUTH / ACCESS / CREDIT / LIMIT / APIERR / TRANSPORT). Empty means the ledger
	// carries no row for this seat at all — never probed.
	Status string
	// At is when that row was recorded. HasAt is false when the seat was never probed or its
	// timestamp would not parse; an undatable row is evidence the prober touched the seat at
	// some point, never evidence about its health now.
	At    time.Time
	HasAt bool
}

// RemovalVerdict is the closed answer vocabulary for "what does the ledger say about retiring
// this seat right now". Only RemovalHealthy gates.
type RemovalVerdict string

const (
	// RemovalHealthy: a FRESH probe read OK and named no entitlement wall. Retiring this seat
	// discards a seat the fleet can currently serve from, and freezes that OK as its last word.
	RemovalHealthy RemovalVerdict = "healthy"
	// RemovalWalled: a fresh probe named an entitlement wall (AUTH/ACCESS/CREDIT). This is the
	// removal the operator means; it must pass without friction.
	RemovalWalled RemovalVerdict = "walled"
	// RemovalUnknown: no fresh evidence of health either way — never probed, undatable, stale,
	// or a fresh-but-inconclusive status (LIMIT/APIERR/TRANSPORT). Absent evidence is NOT a
	// block: gating on it would be self-sealing, since a tombstoned seat is never probed again
	// and could therefore never clear a gate imposed for want of a probe.
	RemovalUnknown RemovalVerdict = "unknown"
)

// RemovalCheck is the graded answer plus the measurements it was made from, so a caller can
// print the evidence rather than only the label.
type RemovalCheck struct {
	// Verdict is the grade. Read it through Blip() rather than comparing, so a future fourth
	// state cannot silently start gating.
	Verdict RemovalVerdict
	// Status is the newest row's probe status, upper-cased and trimmed; "" when never probed.
	Status string
	// AgeHours is hours between that row and now; meaningful only when HasAge.
	AgeHours float64
	HasAge   bool
	// Reason is a one-line operator-facing explanation of the grade, always set.
	Reason string
}

// Blip reports whether this removal would retire a currently-healthy seat — the one condition
// #4676 asks `fak accounts remove` to gate on.
func (c RemovalCheck) Blip() bool { return c.Verdict == RemovalHealthy }

// GradeRemoval decides whether p is fresh evidence that the seat is healthy. now is injected
// for determinism (pass time.Now().UTC() in production).
//
// The bar is deliberately narrow — only a fresh, positive OK gates. Everything else passes,
// including a fresh wall (the removal the operator means) and every flavour of absent or
// inconclusive evidence. That asymmetry is the point: a false gate blocks legitimate fleet
// hygiene on a seat that genuinely cannot serve, which is worse than missing a blip.
func GradeRemoval(p SeatProbe, now time.Time) RemovalCheck {
	c := RemovalCheck{Verdict: RemovalUnknown, Status: strings.ToUpper(strings.TrimSpace(p.Status))}
	switch {
	case c.Status == "":
		c.Reason = "no probe row for this seat — nothing witnesses it as healthy"
		return c
	case !p.HasAt || p.At.IsZero():
		c.Reason = fmt.Sprintf("newest probe (%s) carries no usable timestamp — not evidence about now", c.Status)
		return c
	}
	c.AgeHours, c.HasAge = now.Sub(p.At).Hours(), true
	// A future-dated row is as unusable as an ancient one: a clock that disagrees by more than
	// the budget is not a clock this guard can reason from, so both directions age out.
	if c.AgeHours > ProbeFreshHours || c.AgeHours < -ProbeFreshHours {
		c.Reason = fmt.Sprintf("newest probe (%s) is %s the %.0fh freshness budget", c.Status, ageOffsetText(c.AgeHours), ProbeFreshHours)
		return c
	}
	if c.Status != ProbeStatusOK {
		if entitlementWall[c.Status] {
			c.Verdict = RemovalWalled
			c.Reason = fmt.Sprintf("newest probe %.1fh ago is %s — a real entitlement wall, not a blip", c.AgeHours, c.Status)
			return c
		}
		c.Reason = fmt.Sprintf("newest probe %.1fh ago is %s — no clean health signal either way", c.AgeHours, c.Status)
		return c
	}
	c.Verdict = RemovalHealthy
	c.Reason = fmt.Sprintf("newest probe %.1fh ago is OK with no auth/access wall — this seat is currently healthy", c.AgeHours)
	return c
}

// ageOffsetText renders a probe age as an operator-readable distance from the freshness
// budget, keeping the future-dated case ("dated %.1fh in the future") from printing as a
// nonsensical negative age.
func ageOffsetText(ageHours float64) string {
	if ageHours < 0 {
		return fmt.Sprintf("dated %.1fh in the FUTURE, past", -ageHours)
	}
	return fmt.Sprintf("%.1fh old, past", ageHours)
}

// RemovedHealthySeat is one TOMBSTONED seat whose frozen last probe says it was healthy when
// it was retired, still inside the restore window — the "removed-but-healthy — restore?"
// finding #4676 asks the status surface to carry.
type RemovedHealthySeat struct {
	Name string `json:"name"`
	// RehomeTo is the handle the tombstone falls forward to, so the notice names what is
	// carrying the load in the meantime.
	RehomeTo string `json:"rehome_to,omitempty"`
	// TombstonedAt / TombstoneReason are the registry's own audit fields, echoed verbatim.
	TombstonedAt    string `json:"tombstoned_at,omitempty"`
	TombstoneReason string `json:"tombstone_reason,omitempty"`
	// ProbeStatus is the frozen last probe's verdict (always OK for a seat in this list) and
	// ProbeAgeHours how old that probe was AT THE TOMBSTONE — not now. Measuring it against
	// the tombstone is the whole point: after the tombstone the prober stops, so measuring
	// against now would age every such seat out of the window it is supposed to be visible in.
	ProbeStatus   string  `json:"probe_status"`
	ProbeAgeHours float64 `json:"probe_age_hours"`
	// RemovedHoursAgo is how long the seat has been dark.
	RemovedHoursAgo float64 `json:"removed_hours_ago"`
}

// Hint is the one-line operator-facing restore hint for this seat.
func (s RemovedHealthySeat) Hint() string {
	return fmt.Sprintf("REMOVED-BUT-HEALTHY: %s was tombstoned %.1fh ago but its last probe (%.1fh earlier, status %s) was healthy — "+
		"no probe has run since, and none will while it is tombstoned. Restore with `fak accounts restore --name %s` if that removal was a blip.",
		s.Name, s.RemovedHoursAgo, s.ProbeAgeHours, s.ProbeStatus, s.Name)
}

// RestoreNotice grades ONE tombstoned seat as removed-but-healthy. probes is the seat's frozen
// last probe row; now is injected for determinism.
//
// It reports ok=false for an ACTIVE seat (nothing to restore), for a tombstone older than
// RestoreNoticeHours (the decision has settled), for a tombstone whose timestamp will not
// parse, and for any last probe that is not an OK dated within ProbeFreshHours OF THE TOMBSTONE.
// That last clause is what keeps the notice honest: a seat whose newest probe predates its
// removal by days was already unmeasured when it was retired, and calling it "healthy" would
// be inventing evidence the ledger never carried.
func RestoreNotice(h Home, p SeatProbe, now time.Time) (RemovedHealthySeat, bool) {
	if h.Active() {
		return RemovedHealthySeat{}, false
	}
	tomb, ok := parseTombstoneTime(h.TombstonedAt)
	if !ok {
		return RemovedHealthySeat{}, false
	}
	removedHours := now.Sub(tomb).Hours()
	if removedHours < 0 || removedHours > RestoreNoticeHours {
		return RemovedHealthySeat{}, false
	}
	// Grade the probe AS OF THE TOMBSTONE. GradeRemoval is reused verbatim so the notice and
	// the gate can never disagree about what "healthy" means.
	c := GradeRemoval(p, tomb)
	if !c.Blip() {
		return RemovedHealthySeat{}, false
	}
	return RemovedHealthySeat{
		Name:            h.Name,
		RehomeTo:        h.RehomeTo,
		TombstonedAt:    h.TombstonedAt,
		TombstoneReason: h.TombstoneReason,
		ProbeStatus:     c.Status,
		ProbeAgeHours:   c.AgeHours,
		RemovedHoursAgo: removedHours,
	}, true
}

// RemovedButHealthy folds RestoreNotice over the whole registry, returning the tombstoned
// seats a status surface should offer to restore, in registry order. probes maps SEAT NAME to
// that seat's frozen last probe row; a seat absent from the map is simply never healthy-looking,
// so a caller that cannot resolve a ledger at all gets an empty list rather than a false claim.
func (r Registry) RemovedButHealthy(probes map[string]SeatProbe, now time.Time) []RemovedHealthySeat {
	var out []RemovedHealthySeat
	for _, h := range r.Homes {
		if s, ok := RestoreNotice(h, probes[h.Name], now); ok {
			out = append(out, s)
		}
	}
	return out
}

// parseTombstoneTime parses the RFC3339 stamp `remove` writes into Home.TombstonedAt,
// tolerating the date-only form a hand-edited registry can carry.
func parseTombstoneTime(raw string) (time.Time, bool) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05", "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), true
		}
	}
	return time.Time{}, false
}
