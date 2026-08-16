package main

import (
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

// stallscan_skew.go — the BINARY/SOURCE SKEW GUARD for `fak stallscan` (#3668, the sibling
// finding to #3400's stale `repoguard.exe`).
//
// WHY THIS EXISTS, and why a generic "your fak is old" banner is not enough. stallscan grew its
// axes one at a time (handle-total 30947eee2, thread-leak 70defaa11, growth/trajectory b2eef8226,
// reboot verdict 196d195ab). A deployed `fak.exe` built between two of those commits does not
// FAIL on the axis it lacks — it reports that axis EMPTY. Observed live on 2026-07-09: the
// deployed binary predated the thread axis by 13 minutes, so
// `tools/.bin/fak.exe stallscan --json --once` printed `thread_leak_process: ""` and
// `top_threads: []` while WindowsTerminal held 3,206 threads. An empty axis is indistinguishable
// from a healthy host, so the stale detector reads as a CLEAN BILL OF HEALTH. That is the whole
// defect: not a wrong answer, a confidently silent one.
//
// So the guard is deliberately attached to the DETECTOR'S OWN OUTPUT rather than to a startup
// banner somewhere else: whoever reads the fingerprint (a human, the --watch JSONL trail, a
// downstream pager) is the one who must be told the reading may be from stale code. The verdict
// travels in-band, on the same record, or it does not help.
//
// It classifies by git ANCESTRY via internal/versionskew — the same kernel `fak guard`'s banner
// uses — and warns on exactly the four REFUSABLE tokens (SKEWED behind trunk, DIVERGED off it,
// UNSTAMPED, DIRTY: all four mean "the axes in this file may not be the axes in the source").
// FRESH and AHEAD are current enough; UNKNOWN is the honest residual and stays silent — the guard
// never invents a skew it could not establish.
//
// KNOWN RESIDUAL (honest, not a bug): resolving SKEWED/DIVERGED needs a git repo in the process
// cwd. Under the FakStallMonitor scheduled task the cwd is not the checkout, so those two collapse
// to UNKNOWN and stay quiet. UNSTAMPED and DIRTY need no git and still fire there — and those are
// the tokens a hand-copied or dev-tree deploy actually produces.

// stallBuildSkew is the machine-readable half of the guard: the closed-set verdict token plus the
// two revisions that decided it, so a JSONL consumer can match on `verdict` instead of parsing
// prose, and the human note that explains the consequence. It is emitted ONLY when the guard
// fires, so a fingerprint from a current binary keeps its existing shape byte for byte.
type stallBuildSkew struct {
	Verdict  string `json:"verdict"`             // SKEWED | DIVERGED | UNSTAMPED | DIRTY
	Running  string `json:"running,omitempty"`   // the binary's embedded rev ("" when unstamped)
	TrunkTip string `json:"trunk_tip,omitempty"` // origin/main at read time ("" when unresolvable)
	Note     string `json:"note"`                // the operator-facing consequence + the fix
}

// stallscanBuildSkew reads THIS binary's skew assessment. It is a var so a test can drive the
// guard from a synthetic verdict without a git checkout, and it delegates to the guard verb's
// cached assessment so the git probe happens at most ONCE per process — the property that makes it
// safe inside the --watch loop, which must never become the churn it measures.
var stallscanBuildSkew = guardBuildSkewAssessment

// stallscanSkewGuard is the pure kernel: assessment in, either a warning or nil. nil means "this
// reading can be trusted to be as current as the source" — the caller emits nothing at all, so no
// existing output or record changes shape.
func stallscanSkewGuard(a versionskew.Assessment) *stallBuildSkew {
	if !a.Verdict.Refusable() {
		return nil
	}
	var head string
	switch a.Verdict {
	case versionskew.Skewed:
		head = "this stallscan binary was built from " + shortLaunchRev(a.Running) +
			", but origin/main is at " + shortLaunchRev(a.TrunkTip) + " (provably BEHIND)"
	case versionskew.Diverged:
		head = "this stallscan binary was built from " + shortLaunchRev(a.Running) +
			", which is OFF the trunk line (origin/main at " + shortLaunchRev(a.TrunkTip) + ")"
	case versionskew.Unstamped:
		head = "this stallscan binary carries NO VCS stamp, so which commit its axes came from cannot be attested"
	case versionskew.Dirty:
		head = "this stallscan binary was built from a DIRTY tree at " + shortLaunchRev(a.Running) +
			", so its embedded rev does not describe what it actually contains"
	}
	return &stallBuildSkew{
		Verdict:  a.Verdict.String(),
		Running:  a.Running,
		TrunkTip: a.TrunkTip,
		Note: head + " — an axis added after this build reports EMPTY here, not missing, " +
			"so a quiet handle/thread/reboot line may be a stale-binary artifact rather than a healthy host. " +
			"Redeploy the detector (`go build -o tools/.bin/fak.exe ./cmd/fak`, or `fak self-update`) and re-run before trusting a calm reading.",
	}
}
