package wipattr

import "strings"

// SweepRisk is the closed sweep-guard vocabulary (#3879). A broad `git add` (add -A /
// add .) stages every dirty hunk indiscriminately; on a shared working tree that can
// co-mingle or destroy a peer session's in-flight WIP. SweepGuard grades each hunk
// SAFE or HAZARD so the guard can warn before that sweep happens.
type SweepRisk string

const (
	// SweepSafe: the hunk is OWNED solely by the self session — a broad add stages
	// only your own work.
	SweepSafe SweepRisk = "SAFE"
	// SweepHazard: the hunk belongs (wholly or partly) to someone other than self, or
	// to no checkpoint at all — sweeping it endangers a peer's WIP.
	SweepHazard SweepRisk = "HAZARD"
)

// SweepVerdict is the per-hunk sweep decision, enriched with the peer's liveness.
type SweepVerdict struct {
	File   string    `json:"file"`
	State  AttrState `json:"state"`
	Owner  string    `json:"owner,omitempty"`
	Owners []string  `json:"owners,omitempty"`
	Risk   SweepRisk `json:"risk"`
	Reason string    `json:"reason"`
}

// SweepGuard classifies each attributed hunk as SAFE or HAZARD to stage in a broad
// `git add`, given the self session id and the set of currently-live session ids.
// The rule is FAIL-SAFE: a hunk is SAFE only when it is OWNED solely by self; every
// other state — owned by a peer, SHARED, or ORPHAN (no checkpoint records it, so it
// may be a peer's uncheckpointed WIP) — is a HAZARD. Liveness sharpens the reason (a
// LIVE peer is the acute case) but never downgrades a hazard to safe. One verdict per
// input attribution (totality); input order is preserved (Attribute already sorted it).
func SweepGuard(attrs []Attribution, self string, live map[string]bool) []SweepVerdict {
	out := make([]SweepVerdict, 0, len(attrs))
	for _, a := range attrs {
		v := SweepVerdict{File: a.File, State: a.State, Owner: a.Owner, Owners: a.Owners}
		switch a.State {
		case AttrOwned:
			switch {
			case a.Owner == self && self != "":
				v.Risk, v.Reason = SweepSafe, "owned by self — safe to stage"
			case live[a.Owner]:
				v.Risk, v.Reason = SweepHazard, "owned by LIVE peer "+a.Owner+" — do not sweep"
			default:
				v.Risk, v.Reason = SweepHazard, "owned by peer "+a.Owner+" (not live) — reconcile, do not sweep"
			}
		case AttrShared:
			v.Risk, v.Reason = SweepHazard, "shared across "+strings.Join(a.Owners, ",")+" — ambiguous ownership, do not sweep"
		default: // AttrOrphan
			v.Risk, v.Reason = SweepHazard, "unattributed (orphan) — may be a peer's uncheckpointed WIP"
		}
		out = append(out, v)
	}
	return out
}

// SweepHazards returns just the HAZARD verdicts — the set a broad add would endanger,
// and the exit-3 gate condition for the sweep-guard.
func SweepHazards(vs []SweepVerdict) []SweepVerdict {
	out := make([]SweepVerdict, 0)
	for _, v := range vs {
		if v.Risk == SweepHazard {
			out = append(out, v)
		}
	}
	return out
}
