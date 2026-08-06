package hooks

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// gate_barecommitsweep.go — the BARE_COMMIT_SWEEP gate (issue #3615). It closes the raw-git
// bypass in safecommit's shared-trunk commit discipline.
//
// safecommit refuses to fold FOREIGN staged hunks into a commit: its PRESTAGED_PATH_OVERLAP
// guard (internal/safecommit/prestaged.go) runs `git status --porcelain -- <declared paths>`
// and blocks when a requested path already carries a peer's staged work. But that guard only
// runs INSIDE `fak commit`/safecommit, and it is scoped to the DECLARED pathspec — two holes:
//
//  1. A bare `git commit` (no pathspec) sweeps the WHOLE staged index into one commit,
//     including hunks the committer never declared — exactly the shared-trunk corruption the
//     pathspec discipline exists to prevent. Scoped to a declared pathspec, safecommit's own
//     guard cannot even see the foreign extras a bare commit would fold in.
//  2. A worker that runs raw `git add -A && git commit` never touches safecommit at all, so
//     nothing gates it.
//
// This pre-commit gate is the hook-layer sibling that covers BOTH: it sees the staged set the
// commit is about to record and, when the commit did NOT come through fak's vetted path, flags
// it as an unvetted bare sweep and names exactly what would land. The handshake is the
// FAK_SAFECOMMIT_VETTED marker safecommit sets on its own `git commit` exec (runner.go): a
// vetted commit already ran the path-scoped PRESTAGED_PATH_OVERLAP guard on an explicit
// pathspec, so this gate must never double-refuse it.
//
// ADVISORY BY DEFAULT (DefaultMode "warn"), exactly like E2E_OVER_MOCKS / PRIOR_ART: out of
// the box it never reds a shared trunk — it prints what a bare commit would sweep and the
// pathspec fix. Set FLEET_BARE_COMMIT_GUARD=block to hard-enforce it (refuse the commit),
// ALLOW_BARE_COMMIT=1 to skip it once, or FAK_PRESTAGED_PATH_GUARD=off to disable the whole
// prestaged-overlap family (this gate and safecommit's own guard together).

const (
	// bareCommitVettedEnv is the handshake marker safecommit sets on its `git commit` exec
	// (internal/safecommit/runner.go). Present => this commit's pathspec was already vetted by
	// the path-scoped PRESTAGED_PATH_OVERLAP guard, so the bare-sweep gate stands down.
	bareCommitVettedEnv = "FAK_SAFECOMMIT_VETTED"
	// bareCommitPreStagedEnv is safecommit's own prestaged-guard env. =off disables the whole
	// prestaged-overlap family, this hook-layer sibling included, so a worker who opts the
	// family out for an intentional one-shot opts out both layers with one switch.
	bareCommitPreStagedEnv = "FAK_PRESTAGED_PATH_GUARD"
	// bareCommitListCap bounds how many staged paths the finding lists before "(+N more)" —
	// enough to see the sweep without a wall of text on a whole-index bare commit.
	bareCommitListCap = 12
)

// gateBareCommitSweep fires ONE BARE_COMMIT_SWEEP finding when the staged set is non-empty and
// the commit is UNVETTED (no FAK_SAFECOMMIT_VETTED handshake) — the shape a raw `git commit` /
// `git add -A && git commit` outside `fak commit` presents. A vetted commit, an empty staged
// set, or an explicit family opt-out (FAK_PRESTAGED_PATH_GUARD=off) each returns clean.
func gateBareCommitSweep(d *StagedDiff) ([]Finding, error) {
	// Vetted commit: safecommit already ran the path-scoped prestaged-overlap guard on an
	// explicit pathspec. Never double-refuse the safe path.
	if bareCommitVetted() {
		return nil, nil
	}
	// Family opt-out: FAK_PRESTAGED_PATH_GUARD=off silences safecommit's own guard and this
	// hook-layer sibling together.
	if strings.EqualFold(strings.TrimSpace(os.Getenv(bareCommitPreStagedEnv)), "off") {
		return nil, nil
	}
	// Nothing staged => an empty commit (--allow-empty, a merge) sweeps nothing.
	if len(d.StagedPaths) == 0 {
		d.NoteCandidates("BARE_COMMIT_SWEEP", 0, "staged path(s) that would be swept")
		return nil, nil
	}

	paths := make([]string, 0, len(d.StagedPaths))
	for _, p := range d.StagedPaths {
		paths = append(paths, strings.ReplaceAll(p, "\\", "/"))
	}
	sort.Strings(paths)
	// The whole staged set IS this gate's domain: it asks what an unvetted bare commit would
	// fold in, so the denominator is the size of that sweep (#5602).
	d.NoteCandidates("BARE_COMMIT_SWEEP", len(paths), "staged path(s) that would be swept")

	return []Finding{{
		Gate:   "BARE_COMMIT_SWEEP",
		File:   paths[0],
		Line:   0,
		Detail: bareCommitDetail(paths),
	}}, nil
}

// bareCommitVetted reports whether the FAK_SAFECOMMIT_VETTED handshake marker is set (1/true/on)
// — i.e. this git commit was issued by safecommit, which already vetted the pathspec.
func bareCommitVetted() bool {
	v := strings.TrimSpace(os.Getenv(bareCommitVettedEnv))
	return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "on")
}

// bareCommitDetail renders the one-line advisory: how many staged paths a bare commit would
// sweep, a capped list of them, and the pathspec fix + escape hatches.
func bareCommitDetail(paths []string) string {
	shown := paths
	extra := 0
	if len(shown) > bareCommitListCap {
		extra = len(shown) - bareCommitListCap
		shown = shown[:bareCommitListCap]
	}
	list := strings.Join(shown, ", ")
	if extra > 0 {
		list = fmt.Sprintf("%s (+%d more)", list, extra)
	}
	return fmt.Sprintf(
		"a commit outside `fak commit` would sweep %d staged path(s) into one commit: %s — "+
			"none declared to fak (no FAK_SAFECOMMIT_VETTED handshake), so a peer's staged hunk in the "+
			"index would land under your message. Commit through fak's vetted path instead: "+
			"`fak commit --path <yours> ...` — the only form that sets the vetted marker. "+
			"(A raw `git commit -- <yours>` is NOT offered here: a pre-commit hook never sees the "+
			"pathspec, so that form stays unvetted and would re-draw this same advisory.) "+
			"(advisory; FLEET_BARE_COMMIT_GUARD=block enforces, ALLOW_BARE_COMMIT=1 skips once, "+
			"FAK_PRESTAGED_PATH_GUARD=off disables the family)",
		len(paths), list,
	)
}
