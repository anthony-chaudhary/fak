package accountprobe

import (
	"fmt"
	"strings"
	"time"
)

// SeatHealth grades what the probe ledger can honestly say about ONE seat's health right
// now. It is the per-seat sibling of RegHealth (regdir.go), which asks the same question one
// level up: there, "can this registry derive a block at all"; here, "has this seat been
// probed recently enough for its silence to mean anything".
//
// The distinction it exists to keep is the whole of #5391. A ledger can be present, current
// and busy — the prober writes opencode-* rows to the minute — while a claude seat's newest
// row is eight days old. Every read-side fold then finds no fresh verdict for that seat,
// falls through, and publishes it as available, so a seat that is UNMEASURED and a seat that
// was MEASURED AND FOUND HEALTHY collapse into the same answer. The collapse is expensive in
// one specific direction: an entitlement failure answers 403, a 403 burns no quota, so the
// dead seat reports full headroom and a headroom-weighted allocator ranks it among the
// emptiest in the fleet — the fleet routes the most work to the one seat that cannot work.
//
// Unmeasured is deliberately NOT blocked. Treating absent evidence as a block strands seats
// and is self-sealing: the roster routes the work that runs the prober, so a block imposed
// for want of a probe forbids the very probe that would clear it. This repo has already paid
// for that once, and #5391 files itself as a coverage gap for exactly that reason. So this
// is a THIRD state, reportable on its own terms and convertible to neither of the other two.
type SeatHealth string

const (
	// SeatHealthFresh: the seat's newest probe row is dated and within
	// SeatCoverageMaxAgeMin. This seat's health is measured, so its silence is a finding.
	SeatHealthFresh SeatHealth = "probe-fresh"
	// SeatHealthStale: a probe row exists but is older than the budget — or carries a
	// timestamp that cannot be dated at all, which is the same thing for this purpose,
	// since an undatable row is not evidence about NOW. Unmeasured.
	SeatHealthStale SeatHealth = "probe-stale"
	// SeatHealthNever: the ledger carries no row for this seat at all. Unmeasured, and the
	// shape #5391 names outright — "never probed" and "probed OK" must not both read as
	// available.
	SeatHealthNever SeatHealth = "probe-never"
)

// Measured reports whether this grade licenses reading the seat's probe silence as health.
// Only SeatHealthFresh does. A caller holding false must say "unknown" — never "OK", and
// (see the SeatHealth doc) never "blocked" either.
func (h SeatHealth) Measured() bool { return h == SeatHealthFresh }

// SeatCoverageMaxAgeMin is the probe-coverage budget in minutes: how old a seat's newest
// probe row may be before the ledger stops counting as evidence about that seat.
//
// 1440 (24h) is a CHOSEN number, not a measured one. Two things chose it. It sits between
// the two populations #5391 actually observed on one host — opencode-* rows current to
// within minutes, claude rows 8-9 days old — with room on both sides, so it separates them
// instead of sitting on top of either. And it equals the entitlement-verdict freshness
// window the roster already runs (fleetaccounts.ProbeLedgerEntitlementFreshMin, default
// 1440), so a seat leaves coverage at exactly the moment its last entitlement verdict also
// ages out: past this line the ledger can say nothing about the seat by either route, and
// one number describes both.
//
// What would have to be observed to justify a different value: an inter-probe gap
// distribution per account class. If the p99 gap for seats that are in fact healthy exceeds
// 24h, this bound grades working seats stale and must rise. If the prober's nominal cadence
// is minutes for every class, a far tighter bound (a small multiple of that cadence) would
// catch an outage in hours rather than in a day, and the only reason not to adopt it here is
// that no such distribution has been recorded in this repo. Neither number is in hand, so
// this stays a declared budget rather than a derived one.
const SeatCoverageMaxAgeMin = 1440.0

// SeatCoverage is one seat's probe-evidence grade: the state, and the measurements the
// grade was made from so a caller can render the gap instead of only naming it.
type SeatCoverage struct {
	// Account is the seat as the ledger names it (the config-dir basename).
	Account string
	// Health is the grade. Read it through Health.Measured() rather than comparing to
	// SeatHealthFresh, so a future fourth state does not silently read as healthy.
	Health SeatHealth
	// AgeMin is minutes between the newest dated row and now. Meaningful only when HasAge;
	// it is left at zero otherwise, and a zero with HasAge false means "no age was
	// established", never "probed this instant".
	AgeMin float64
	// HasAge distinguishes a seat whose age is known from one with no datable row at all
	// (SeatHealthNever, or a SeatHealthStale row whose timestamp would not parse).
	HasAge bool
	// Status is the newest row's probe status verdict (OK / AUTH / LIMIT / …), empty when
	// the seat has no row. It is carried for reporting only: a STALE verdict is not a
	// current one, and this type takes no position on what it used to mean.
	Status string
}

// gradeFrom grades one seat against an already-read newest-row-per-account map. It is the
// single place the three states are decided; GradeSeat and GradeSeats differ only in how
// many times they pay for the ledger read.
func gradeFrom(latest map[string]LedgerEntry, account string, now time.Time) SeatCoverage {
	cov := SeatCoverage{Account: account, Health: SeatHealthNever}
	e, ok := latest[account]
	if !ok {
		return cov
	}
	cov.Status = e.Status
	when := parseLedgerTime(e.TS)
	if when == nil {
		// A row that cannot be dated cannot be shown to be recent. It is evidence that the
		// prober touched this seat at some point, which is why it is not "never", but it is
		// not evidence about the seat's health now.
		cov.Health = SeatHealthStale
		return cov
	}
	cov.AgeMin, cov.HasAge = now.Sub(*when).Seconds()/60.0, true
	if cov.AgeMin > SeatCoverageMaxAgeMin {
		cov.Health = SeatHealthStale
		return cov
	}
	cov.Health = SeatHealthFresh
	return cov
}

// GradeSeat grades one seat's probe evidence from the ledger under rd (rd "" resolves the
// default registry, as everywhere else in this package). now is injected for determinism;
// pass time.Now().UTC() in production.
//
// It cannot distinguish "there is no ledger at all" from "the ledger has never mentioned
// this seat": both are SeatHealthNever, which is the same instruction to the caller — do not
// read this seat as measured. Use GradeSeats, whose report carries whether the ledger said
// anything at all, or ResolveRegDir().BlocksDerivable(), when that difference matters.
func GradeSeat(account, rd string, now time.Time) SeatCoverage {
	return gradeFrom(LastProbeByAccount(rd), account, now)
}

// CoverageReport is the fold over a whole enrolled roster: every seat's grade, the counts,
// and — separately from the counts — whether the ledger said anything at all.
type CoverageReport struct {
	// LedgerPath is the file the grades were read from, so an operator reading a note can
	// go look at it rather than rediscovering which registry this host means.
	LedgerPath string
	// Seats holds one grade per enrolled seat, in the order the caller supplied them.
	Seats []SeatCoverage
	// Fresh/Stale/Never count the grades. Fresh+Stale+Never == len(Seats).
	Fresh int
	Stale int
	Never int
	// LedgerAccounts is how many distinct accounts the ledger carries a row for, INCLUDING
	// ones not enrolled here. It is what separates "the prober is running and has simply
	// never mentioned these seats" — the #5391 shape, since the same ledger was current to
	// the minute for other accounts — from "nothing has been recorded at all".
	LedgerAccounts int
	// Sufficient is false when the ledger yielded no rows whatever: unreadable, absent, or
	// empty. It exists so an empty read is reported as insufficient rather than as a
	// measurement of zero health — see MeasuredFraction.
	Sufficient bool
}

// GradeSeats grades every enrolled seat against one read of the ledger under rd. Passing an
// empty accounts slice yields an empty, insufficient-unless-the-ledger-has-rows report; it
// never errors, matching the rest of this package's best-effort read surface.
func GradeSeats(accounts []string, rd string, now time.Time) CoverageReport {
	latest := LastProbeByAccount(rd)
	rep := CoverageReport{
		LedgerPath:     ProbeLedgerPath(rd),
		LedgerAccounts: len(latest),
		Sufficient:     len(latest) > 0,
	}
	for _, a := range accounts {
		cov := gradeFrom(latest, a, now)
		switch cov.Health {
		case SeatHealthFresh:
			rep.Fresh++
		case SeatHealthStale:
			rep.Stale++
		default:
			rep.Never++
		}
		rep.Seats = append(rep.Seats, cov)
	}
	return rep
}

// Unmeasured counts the seats whose health this ledger cannot currently speak to.
func (r CoverageReport) Unmeasured() int { return r.Stale + r.Never }

// MeasuredFraction returns the share of enrolled seats that are actually measured, and
// whether that share means anything.
//
// ok is false when no seat was graded, and — the point of the second return — when the read
// was empty. A ledger that yields nothing has not established that 0% of seats are healthy;
// it has established nothing, and a caller that formats a bare 0.0 turns an absent
// measurement into a damning one. That inversion is the same mistake as reading an absent
// probe row as a healthy seat, made in the other direction, so it gets the same treatment:
// insufficient is a state, not a number.
//
// A ledger that DOES carry rows but none for the enrolled seats returns (0, true), because
// there the zero is a genuine finding: the prober is writing, and it is writing about
// something else. That is the case #5391 was filed on.
func (r CoverageReport) MeasuredFraction() (float64, bool) {
	if len(r.Seats) == 0 || !r.Sufficient {
		return 0, false
	}
	return float64(r.Fresh) / float64(len(r.Seats)), true
}

// Note renders the one-line operator-visible coverage report, or "" when every enrolled seat
// is measured. It is the observability half of the fix, and the direct answer to #5391's
// "make the gap visible in whatever operator view already shows seat health": the grades
// stop a consumer from misreading a gap, but only a printed line stops the gap itself from
// going unnoticed for the eight days it took to notice this one. It mirrors ForkNote's
// contract — silent when there is nothing to say, one line when there is.
func (r CoverageReport) Note() string {
	if !r.Sufficient {
		return fmt.Sprintf("accountprobe: seat probe coverage UNKNOWN — no probe rows under %s; %d seat(s) unmeasured",
			r.LedgerPath, len(r.Seats))
	}
	if r.Unmeasured() == 0 {
		return ""
	}
	var worst []string
	for _, s := range r.Seats {
		if s.Health.Measured() {
			continue
		}
		if s.HasAge {
			worst = append(worst, fmt.Sprintf("%s (%s ago)", s.Account, humanAgeMin(s.AgeMin)))
			continue
		}
		worst = append(worst, fmt.Sprintf("%s (%s)", s.Account, s.Health))
	}
	return fmt.Sprintf("accountprobe: seat probe coverage INCOMPLETE — %d/%d seats unmeasured"+
		" (%d stale past %s, %d never probed) under %s: %s",
		r.Unmeasured(), len(r.Seats), r.Stale, humanAgeMin(SeatCoverageMaxAgeMin), r.Never,
		r.LedgerPath, strings.Join(worst, ", "))
}

// humanAgeMin renders a minute count at the coarsest unit that keeps it readable, so an
// eight-day gap reads as "8.0d" instead of as five digits an operator has to divide.
func humanAgeMin(min float64) string {
	switch {
	case min >= 1440:
		return fmt.Sprintf("%.1fd", min/1440)
	case min >= 60:
		return fmt.Sprintf("%.1fh", min/60)
	default:
		return fmt.Sprintf("%.0fm", min)
	}
}
