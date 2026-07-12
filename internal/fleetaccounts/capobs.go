package fleetaccounts

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"
)

// deriveCapObservation distills the probe-ledger history for account into the two signals
// the cap-disambiguation cycles consume (see capstate.go): OKStreak, the run of consecutive
// OK verdicts at the tail of the ledger, and FirstSeen, the start of the current contiguous
// blocked episode. It is the bridge that turns the append-only prober record into a
// CapObservation — the aging valve keys off "how long has this seat been walled" and the
// override keys off "how many OKs in a row now say it recovered", and only the timestamped
// ledger knows either.
//
// regDir "" resolves accountprobe's default path. An account with no ledger history yields
// the zero CapObservation, which keeps DisambiguateCap on its legacy single-shot path — so
// a seat the prober has never touched is entirely unaffected. Derivation is time-independent
// (the now-comparison lives inside DisambiguateCap), so no clock is taken here.
func deriveCapObservation(account, regDir string) CapObservation {
	var mine []accountprobe.LedgerEntry
	for _, e := range accountprobe.ReadLedger(accountprobe.ProbeLedgerPath(regDir)) {
		if e.Account == account {
			mine = append(mine, e)
		}
	}
	if len(mine) == 0 {
		return CapObservation{}
	}

	var obs CapObservation
	// OKStreak: consecutive OK verdicts at the tail, newest-first until the first non-OK.
	for i := len(mine) - 1; i >= 0; i-- {
		if !probeIsOK(mine[i].Status) {
			break
		}
		obs.OKStreak++
	}

	// FirstSeen is only meaningful while the seat is CURRENTLY blocked (the tail is non-OK):
	// it is the timestamp of the first entry in that trailing blocked run — when the current
	// episode began. A tail of OKs means there is no live episode, so aging stays dormant and
	// the override (keyed on OKStreak) is the relevant cycle instead.
	if !probeIsOK(mine[len(mine)-1].Status) {
		episodeStart := mine[len(mine)-1]
		for i := len(mine) - 1; i >= 0; i-- {
			if probeIsOK(mine[i].Status) {
				break
			}
			episodeStart = mine[i]
		}
		if when := parseUTC(episodeStart.TS); when != nil {
			obs.FirstSeen, obs.HasFirstSeen = *when, true
		}
	}
	return obs
}

// probeIsOK reports whether a ledger status string is the clean-availability "OK" verdict.
func probeIsOK(status string) bool {
	return strings.EqualFold(strings.TrimSpace(status), "OK")
}
