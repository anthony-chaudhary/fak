package main

import (
	"context"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

// guard_skew.go — the ATTESTED-but-stale freshness signal for the guard banner and the live info
// pane. It is the sharper sibling of guard_freshness.go's guardUnattestedBuildWarning: that path
// answers "can this binary attest a commit AT ALL?" from the build-stamp string; this path answers
// "the binary CAN attest its commit — is that commit provably behind or off the trunk?" and needs
// git ancestry (internal/versionskew). Kept in its own file so the git-touching classifier and the
// two pure one-line renderers sit together, apart from the string-only staleness helpers.

var (
	guardBuildSkewOnce sync.Once
	guardBuildSkewVal  versionskew.Assessment
)

// guardBuildSkewAssessment classifies THIS running binary against origin/main by git ANCESTRY,
// ONCE per process. The running binary's embedded rev is a session constant, so the verdict is
// cached: the startup banner prints it once and the live info pane can re-read it every frame
// without re-shelling git. It runs git in the process cwd and does NOT fetch — guard boot and the
// pane must stay responsive, and a stamp that is a strict ancestor of even a slightly-old local
// origin/main is still a genuine Skewed. When origin/main cannot be resolved (fresh clone, no
// remote, git absent) the verdict is the honest Unknown and nothing warns. The trade-off of
// caching: a binary that was the tip at boot but is overtaken by a later origin/main push
// mid-session is not re-flagged — the dominant "launched already behind" case is caught at first
// read, and re-shelling git per frame for hours is the worse default.
func guardBuildSkewAssessment() versionskew.Assessment {
	guardBuildSkewOnce.Do(func() {
		guardBuildSkewVal = versionskew.AssessStamp(context.Background(), versionskew.RealRunner, "", "origin/main", binstamp.Self())
	})
	return guardBuildSkewVal
}

// guardSkewBuildWarning is the one-line WARN the guard startup banner prints under its `build`
// row when the running binary CAN attest its commit but that commit is provably behind or off
// origin/main — the attested twin of guardUnattestedBuildWarning. Skewed (a strict ancestor of
// the trunk tip) and Diverged (off the trunk line entirely) each get a distinct message, both
// naming the same re-exec footgun the unattested warning does (the default guard path re-execs
// THIS file) and the durable fix. Every other verdict returns "" so the caller can emit it
// unconditionally: Fresh and Ahead (a newer local build) are not stale; Dirty is a dev build;
// Unstamped is owned by guardUnattestedBuildWarning (the banner's own no-VCS row), so keeping it
// silent here is what prevents a double-warn; and Unknown is the honest residual nothing refuses.
func guardSkewBuildWarning(a versionskew.Assessment) string {
	switch a.Verdict {
	case versionskew.Skewed:
		return fmt.Sprintf("  build WARN : this guard was built from %s, but origin/main is at %s (provably BEHIND); the default guard path re-execs THIS file, so run `fak self-update` or rebuild in-repo with `go build ./cmd/fak`, then relaunch.\n",
			shortLaunchRev(a.Running), shortLaunchRev(a.TrunkTip))
	case versionskew.Diverged:
		return fmt.Sprintf("  build WARN : this guard was built from %s, which is OFF the trunk line (origin/main is at %s); rebuild/install fak from origin/main, then relaunch.\n",
			shortLaunchRev(a.Running), shortLaunchRev(a.TrunkTip))
	}
	return ""
}

// guardInfoSkewNote is the info PANE's persistent twin of guardSkewBuildWarning — the same
// provably-BEHIND (Skewed) / OFF-trunk (Diverged) classification, phrased for the pane header
// that stays on screen the whole session after the banner's line has scrolled off. It returns ""
// for every non-refusable verdict, and leaves Unstamped to guardInfoStalenessNote so the pane
// never double-warns about the same binary.
func guardInfoSkewNote(a versionskew.Assessment) string {
	switch a.Verdict {
	case versionskew.Skewed:
		return fmt.Sprintf("stale-build WARN: this fak is built from %s but origin/main is at %s (provably BEHIND) — `fak self-update` or `go build ./cmd/fak`, then relaunch",
			shortLaunchRev(a.Running), shortLaunchRev(a.TrunkTip))
	case versionskew.Diverged:
		return fmt.Sprintf("stale-build WARN: this fak is built from %s, OFF the trunk line (origin/main at %s) — rebuild from origin/main, then relaunch",
			shortLaunchRev(a.Running), shortLaunchRev(a.TrunkTip))
	}
	return ""
}
