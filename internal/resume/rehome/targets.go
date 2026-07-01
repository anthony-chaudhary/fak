package rehome

import (
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// DefaultRehomeCap is the fleet burst-spread ceiling: how many sessions one account
// may hold (already-live + just-assigned in this pass) before it drops out of the
// re-home candidate pool. A re-home adds a fresh autonomous `claude --resume` to the
// target, and an account already near its session ceiling limit-walls the moment the
// burst lands. Mirrors fleet_sessions.REHOME_CAP.
const DefaultRehomeCap = 4

// CapUnbounded is the sentinel a single interactive resume passes to RehomeTargets to
// get the least-loaded ranking WITHOUT the burst-spread stampede gate — so a lone
// resume is not stranded on PIN_BLOCKED when the only healthy account is merely over
// the fleet cap. It is the Go analogue of resume_resolver passing cap=math.inf.
const CapUnbounded = math.MaxInt

// RehomeCap resolves the burst-spread ceiling, honoring a FAK_REHOME_CAP override for
// hosts with fatter accounts, else DefaultRehomeCap. Mirrors the module-level
// REHOME_CAP = int(os.environ.get("FAK_REHOME_CAP", "4")).
func RehomeCap() int {
	if v := strings.TrimSpace(os.Getenv("FAK_REHOME_CAP")); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return DefaultRehomeCap
}

// Target is one candidate account row for re-home ranking, mirroring the availability
// dicts fleet_sessions.rehome_targets consumes.
type Target struct {
	Account        string
	Available      bool
	LiveSessions   int
	ActiveSessions int
	Tag            string
	ConfigDir      string
	// VerdictSource is the health-evidence class account_availability stamps:
	// "probe"/"passive" are positive evidence the account serves now; "carried"/"none"
	// are not. It is only a ranking tie-break here (probe/passive sort ahead), not an
	// admission gate. Absent (the resume_resolver availability shape) reads as "none".
	VerdictSource string
}

// hasPositiveEvidence reports whether a target's health verdict rests on positive
// evidence it is serving right now (a fresh probe or a real in-window session row),
// rather than a stale carried verdict or the absence-of-evidence default. Mirrors
// fleet_sessions._has_positive_evidence.
func hasPositiveEvidence(t Target) bool {
	switch t.VerdictSource {
	case "probe", "passive":
		return true
	default:
		return false
	}
}

// RehomeTargets returns the available Claude worker accounts a throttled session can
// move to, least loaded first. It is the Go port of fleet_sessions.rehome_targets: a
// pure ranker with no admission side effects.
//
// opencode accounts are excluded (a Claude transcript can only resume under another
// Claude config dir), as is excludeAccount (the throttled owner). assigned is the
// per-account count this pass has ALREADY re-homed onto each target (account basename
// -> n), folded into the load so a burst spreads across healthy accounts instead of
// all picking the same momentary least-loaded one. An account whose (already-live +
// just-assigned) load reaches cap drops out of the pool entirely.
//
// cap is the admission ceiling to apply: pass RehomeCap() for the normal burst-spread
// path, or CapUnbounded for a single interactive resume that must not be capped.
// Ranking is by (live + assigned) load, then proven-health (positive evidence first),
// then active sessions, then tag/account — matching the Python sort key exactly.
func RehomeTargets(avail []Target, excludeAccount string, assigned map[string]int, cap int) []Target {
	if assigned == nil {
		assigned = map[string]int{}
	}
	cands := make([]Target, 0, len(avail))
	for _, a := range avail {
		if !a.Available || a.Account == excludeAccount || !strings.HasPrefix(a.Account, ".claude") {
			continue
		}
		if a.LiveSessions+assigned[a.Account] >= cap {
			continue
		}
		cands = append(cands, a)
	}
	sort.SliceStable(cands, func(i, j int) bool {
		li := cands[i].LiveSessions + assigned[cands[i].Account]
		lj := cands[j].LiveSessions + assigned[cands[j].Account]
		if li != lj {
			return li < lj
		}
		ui, uj := unprovenRank(cands[i]), unprovenRank(cands[j])
		if ui != uj {
			return ui < uj
		}
		if cands[i].ActiveSessions != cands[j].ActiveSessions {
			return cands[i].ActiveSessions < cands[j].ActiveSessions
		}
		return targetTagKey(cands[i]) < targetTagKey(cands[j])
	})
	return cands
}

func unprovenRank(t Target) int {
	if hasPositiveEvidence(t) {
		return 0
	}
	return 1
}

func targetTagKey(t Target) string {
	if t.Tag != "" {
		return t.Tag
	}
	return t.Account
}
