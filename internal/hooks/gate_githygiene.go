package hooks

import (
	"sort"
	"strings"
)

// gate_githygiene.go — the GIT_HYGIENE_BYPASS advisory gate (#5588). When a staged commit adds
// hand-rolled git-lock reclamation or object-database maintenance OUTSIDE the packages that own
// those decisions, this gate names the evidence-gated route (`fak git-daily`, internal/gitdaily)
// instead of letting the fifth private copy of "just remove index.lock" reach the trunk.
//
// WHY THIS IS THE NUDGE THE SPINE NEEDS. internal/gitdaily exists because both halves of git
// hygiene on a hot shared clone are decided by EVIDENCE, not by intent: a lock is only reclaimed
// when a dead PID or a structurally-inert filename proves it abandoned, and the object DB is only
// folded through gitgate's lock-deferential tiers under a quiet window. An ad-hoc `os.Remove` of a
// live `.git/index.lock` races the writer that owns it and corrupts the very index the removal was
// meant to unwedge; an ad-hoc `git gc --prune` on a clone with in-flight worktrees deletes objects
// a peer still references. Both are silent at review time and expensive at 03:00, which is exactly
// the shape a commit-boundary nudge catches cheaply.
//
// ADVISORY BY DESIGN, LIKE PRIOR_ART. Gate.DefaultMode is "warn": it prints and never refuses.
// FLEET_GIT_HYGIENE_GUARD=block hard-enforces it once a soak validates the signal;
// ALLOW_GIT_HYGIENE_BYPASS=1 skips it once. Advisory-first is the issue's explicit out-of-scope
// line — a block-mode default is a later change that must carry its own evidence.
//
// FAIL-OPEN, STRUCTURALLY. The check is pure string work over the staged diff already in memory:
// no git subprocess, no disk read, no clock. It cannot return an error, so it can never wedge the
// trunk on unreachable evidence (the issue's first confusion risk) — there is no evidence to be
// unreachable.

// gitHygieneOwnerPrefixes are the packages that OWN lock reclamation and object maintenance. Code
// there is the sanctioned implementation, not a bypass of it, so the gate is silent on their paths:
//
//	internal/gitdaily    the daily tick itself (the spine this gate defends)
//	internal/gitgate     the lock-deferential object-maintenance tiers
//	internal/treedoctor  dead-holder commit lock, renamed-aside residue, stale ref locks
//	internal/leaseref    the refs/fak/locks/*.lock orphan reaper
//	internal/commitlane  the two-sample frozen-index-lock evidence gate
//	internal/flock       the advisory-lock primitive every serializer above is built on
//	internal/hooks       this gate and its fixtures, which must spell the needles out loud
var gitHygieneOwnerPrefixes = []string{
	"internal/gitdaily/",
	"internal/gitgate/",
	"internal/treedoctor/",
	"internal/leaseref/",
	"internal/commitlane/",
	"internal/flock/",
	"internal/hooks/",
}

// gitHygieneSourceExts limits the gate to files that can actually RUN a reclamation: Go, shell,
// PowerShell, Python. A doc or a fixture JSON that merely spells `git gc` describes the behaviour
// rather than performing it, and nudging on prose would train the reader to ignore the gate.
var gitHygieneSourceExts = []string{".go", ".sh", ".bash", ".ps1", ".py"}

// gitHygieneLockNeedles name a GIT transaction lock specifically — not any `.lock` file. The
// distinction matters: `os.Remove(path + ".lock")` in an ordinary serializer is that serializer
// releasing its own lock, while these six names are files git (or fak's git lane) owns and whose
// removal is only safe behind the evidence internal/gitdaily applies.
var gitHygieneLockNeedles = []string{
	"index.lock",
	"packed-refs.lock",
	"fak-commit.lock",
	"auto_merge.lock",
	"next-index-",
	"refs/fak/locks",
}

// gitHygieneRemovalVerbs are the ways a line can DELETE one of the locks above. Naming a lock is
// not a bypass — diagnostics, error strings and refusal text all name index.lock and should stay
// silent — so a finding needs a removal verb and a lock name on the SAME added line.
var gitHygieneRemovalVerbs = []string{
	"os.remove",
	"remove-item",
	"unlink",
	"del ",
	"rm ",
	"rm(",
	"delete(",
	"deletefile",
}

// gitHygieneMaintNeedles are object-database maintenance invocations. Unlike the lock family these
// need no second token: running `git gc` / `git repack` / `git prune` from new code IS the bypass,
// because gitgate's tiers exist precisely to decide WHETHER that is safe right now.
var gitHygieneMaintNeedles = []string{
	"git gc",
	"git repack",
	"git prune",
	"gc --prune",
	"repack -a",
}

// gitHygieneAttestation is the token an author adds — in a comment, a doc line, or a trailer — to
// silence the advisory, mirroring PRIOR_ART's `Prior-art:`. A pre-commit gate reads the staged
// diff, not the commit message, so the attestation lives in an added line.
const gitHygieneAttestation = "git-hygiene:"

// gitHygieneRoutedTokens mean the new code already routes through the sanctioned tick, so the
// nudge has nothing to add. This is the "silent on a clean commit" half of the done condition:
// a caller that shells `fak git-daily` names it, and naming it is the fix the gate asks for.
var gitHygieneRoutedTokens = []string{"git-daily", "gitdaily"}

// gateGitHygieneBypass emits ONE advisory finding per staged source file that adds an ad-hoc lock
// reclamation or object-maintenance call outside the owner packages. Findings are deduped by file
// and sorted by path so the output is deterministic.
func gateGitHygieneBypass(d *StagedDiff) ([]Finding, error) {
	// The denominator is the hygiene-CAPABLE staged surface: the source files this gate could
	// have judged. Recorded before the suppression check below for the same reason PRIOR_ART
	// records its own there (#5602) — a suppressed commit that touched nine such files must not
	// report the same zero as one that touched none.
	judged := 0
	for _, raw := range d.StagedPaths {
		if gitHygieneInScope(raw) {
			judged++
		}
	}
	d.NoteCandidates("GIT_HYGIENE_BYPASS", judged, "staged hygiene-capable source file(s)")

	// Suppression is whole-diff, like PRIOR_ART's: an author who attests, or who routes the new
	// code through the daily tick, has already done what the advisory would ask for.
	for _, al := range d.AddedLines() {
		low := strings.ToLower(al.Text)
		if strings.Contains(low, gitHygieneAttestation) {
			return nil, nil
		}
		for _, tok := range gitHygieneRoutedTokens {
			if strings.Contains(low, tok) {
				return nil, nil
			}
		}
	}

	type hit struct {
		line int
		what string
	}
	hits := map[string]hit{}
	for file, lines := range d.AddedByFile {
		if !gitHygieneInScope(file) {
			continue
		}
		norm := strings.ReplaceAll(file, "\\", "/")
		for _, al := range lines {
			what := gitHygieneClassify(al.Text)
			if what == "" {
				continue
			}
			if _, seen := hits[norm]; !seen {
				hits[norm] = hit{line: al.New, what: what}
			}
			break
		}
	}

	files := make([]string, 0, len(hits))
	for f := range hits {
		files = append(files, f)
	}
	sort.Strings(files)

	findings := make([]Finding, 0, len(files))
	for _, f := range files {
		h := hits[f]
		findings = append(findings, Finding{
			Gate:     "GIT_HYGIENE_BYPASS",
			File:     f,
			Line:     h.line,
			Detail:   h.what + " added outside the packages that own it — route it through the evidence-gated daily tick (`fak git-daily`, internal/gitdaily) so a lock is only reclaimed on dead-PID proof and the object DB only folds under gitgate's tiers. Add a \"git-hygiene:\" note to silence.",
			Advisory: true,
		})
	}
	if len(findings) == 0 {
		return nil, nil
	}
	return findings, nil
}

// gitHygieneInScope reports whether a staged path is a source file outside the owner packages.
func gitHygieneInScope(p string) bool {
	norm := strings.ToLower(strings.ReplaceAll(p, "\\", "/"))
	for _, prefix := range gitHygieneOwnerPrefixes {
		if strings.HasPrefix(norm, prefix) {
			return false
		}
	}
	for _, ext := range gitHygieneSourceExts {
		if strings.HasSuffix(norm, ext) {
			return true
		}
	}
	return false
}

// gitHygieneClassify returns a short description of the bypass an added line performs, or "" when
// the line is clean. Lock removal needs a removal verb AND a git lock name on the same line;
// object maintenance needs only the invocation.
func gitHygieneClassify(text string) string {
	low := strings.ToLower(text)
	for _, needle := range gitHygieneMaintNeedles {
		if strings.Contains(low, needle) {
			return "ad-hoc object-database maintenance (" + needle + ")"
		}
	}
	for _, lock := range gitHygieneLockNeedles {
		if !strings.Contains(low, lock) {
			continue
		}
		for _, verb := range gitHygieneRemovalVerbs {
			if strings.Contains(low, verb) {
				return "ad-hoc reclamation of the git lock " + lock
			}
		}
	}
	return ""
}
