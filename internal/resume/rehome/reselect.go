package rehome

import "sort"

// ProbeResult is one account's fresh live-availability verdict, mirroring the dict
// resume_resolver's probe_fn returns. A nil *ProbeResult means the probe could not be
// run (Python's None) — the caller trusts the ranking rather than inventing a verdict.
type ProbeResult struct {
	Available    bool
	BlockReason  string
	BlockKind    string
	StatusSource string
}

// ProbeFunc live-probes one account (by basename + config dir) and returns its fresh
// availability, or nil when no probe could be run. It is injected so the decision is
// unit-testable with no real prober. Mirrors resume_resolver.probe_fn.
type ProbeFunc func(account, configDir string) *ProbeResult

// ReselectMode is the typed decision ReselectDuplicateOwner returns.
type ReselectMode int

const (
	// ReselectKeep: the freshest copy serves (or is unprobeable), or nothing better
	// could be confirmed — keep the normal LocateOwner pick.
	ReselectKeep ReselectMode = iota
	// ReselectPin: a serving sibling whose content is at parity with (or ahead of) the
	// walled freshest — pin it, no copy needed.
	ReselectPin
	// ReselectRehome: the freshest is walled and the only serving siblings are content
	// BEHIND it. Re-home the freshest's FULL content onto the least-stale serving
	// sibling and pin there.
	ReselectRehome
)

// Reselection is the typed result of ReselectDuplicateOwner.
type Reselection struct {
	Mode ReselectMode
	// Owner is set for ReselectPin: the serving sibling to pin.
	Owner *Owner
	// Source/Target are set for ReselectRehome: copy Source's content onto Target.
	Source *Owner
	Target *Owner
}

// ReselectDuplicateOwner is the Go port of resume_resolver._reselect_duplicate_owner.
// For a session duplicated across accounts (the signature of a prior re-home), it
// confirms the newest-mtime owner actually serves; if it does not, it falls back to
// another copy's account that a live probe confirms serving — but only safely.
//
// Candidates are tried newest-mtime first, host LAST (the same ordering as the primary
// owner pick), so an older copy is chosen only when the freshest is provably walled.
// Pinning an older copy is safe only while it holds the SAME content as the walled
// freshest (these append-only transcripts grow, so a strictly-smaller sibling is
// missing the newest turns). So a serving sibling at size >= freshest pins directly;
// a serving-but-content-behind sibling instead becomes a re-home TARGET carrying the
// freshest's full content. Returns ReselectKeep when the normal pick should stand.
func ReselectDuplicateOwner(sid, home string, probe ProbeFunc) Reselection {
	matches := LocateMatches(sid, home)
	if len(matches) <= 1 {
		return Reselection{Mode: ReselectKeep}
	}
	ordered := orderHostLastNewest(matches)
	first := ordered[0]
	probed := probe(first.Account, first.ConfigDir)
	if probed == nil || probed.Available {
		return Reselection{Mode: ReselectKeep} // freshest serves (or unprobeable)
	}

	accts := sortedAccounts(matches)
	stamp := func(m Match) *Owner {
		return &Owner{Match: m, DupCount: len(matches), AllAccounts: accts}
	}

	var behindServing *Match
	for i := 1; i < len(ordered); i++ {
		cand := ordered[i]
		p := probe(cand.Account, cand.ConfigDir)
		if p == nil || !p.Available {
			continue
		}
		if cand.Size >= first.Size {
			return Reselection{Mode: ReselectPin, Owner: stamp(cand)} // at parity -> pin
		}
		if behindServing == nil {
			c := cand
			behindServing = &c
		}
	}
	if behindServing != nil {
		return Reselection{Mode: ReselectRehome, Source: stamp(first), Target: stamp(*behindServing)}
	}
	return Reselection{Mode: ReselectKeep}
}

// orderHostLastNewest returns matches ordered newest-mtime first, restricted to
// non-host accounts when any exist (host last), mirroring the ordered pool the Python
// owner pick and re-selection share.
func orderHostLastNewest(matches []Match) []Match {
	pool := make([]Match, 0, len(matches))
	for _, m := range matches {
		if !m.IsHost {
			pool = append(pool, m)
		}
	}
	if len(pool) == 0 {
		pool = append(pool, matches...)
	}
	sort.SliceStable(pool, func(i, j int) bool { return pool[i].ModTime.After(pool[j].ModTime) })
	return pool
}

func sortedAccounts(matches []Match) []string {
	accts := make([]string, 0, len(matches))
	for _, m := range matches {
		accts = append(accts, m.Account)
	}
	sort.Strings(accts)
	return accts
}
