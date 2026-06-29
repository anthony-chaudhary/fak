package marketing

import (
	"time"
)

// collect.go — the end-to-end fold a caller (CLI, bgloop Tick) uses: enumerate ships in a
// range, apply the CLAIMS.md honesty gate, and hand back the marketable ships + activity +
// what was withheld. It is the seam between the witnessed-atom layer (ship.go) and the
// generators (generate.go), so a caller never wires the gate by hand and can never skip it.

// Collected is the result of gathering and gating a commit range: the marketable ships (safe
// to claim), the honest activity tally, and the ships the CLAIMS.md gate held back (surfaced,
// never silent).
type Collected struct {
	Ships    []Ship
	Activity Activity
	Excluded []ExcludedShip
}

// Gather enumerates the ships in revRange (see CollectShips) at root, loads the CLAIMS.md
// ledger, and applies the honesty gate. It is the one place CollectShips and FilterMarketable
// are composed, so every artifact path goes through the gate.
func Gather(root, revRange string) (Collected, error) {
	ships, act, err := CollectShips(root, revRange)
	if err != nil {
		return Collected{}, err
	}
	ledger := LoadClaims(root)
	marketable, excluded := FilterMarketable(ledger, ships)
	return Collected{Ships: marketable, Activity: act, Excluded: excluded}, nil
}

// DigestFrom builds a weekly-digest Artifact from a gathered range, stamped with when (use a
// zero time to omit the dated title). It is the CLI/bgloop default generator.
func (c Collected) DigestFrom(when time.Time) Artifact {
	return WeeklyDigest(c.Ships, c.Activity, c.Excluded, when)
}

// EpicFrom builds an epic-blurb Artifact from a gathered range — the ships that closed an
// epic, grouped under its title. The caller supplies the epicTitle (e.g. the GitHub issue
// title) and the range it scoped (the gh-poll / issue-close integration is the caller's job,
// keeping the tier-1 core free of a gh dependency).
func (c Collected) EpicFrom(epicTitle string) Artifact {
	return EpicBlurb(epicTitle, c.Ships, c.Excluded)
}

// ReleaseFrom builds a release-highlight Artifact from a gathered range — the ships in a
// release. version is the tag (e.g. "v0.18.0"); notesLead is an optional one-line summary the
// caller pulls from docs/releases/v*.md.
func (c Collected) ReleaseFrom(version, notesLead string) Artifact {
	return ReleaseHighlight(version, notesLead, c.Ships, c.Excluded)
}
