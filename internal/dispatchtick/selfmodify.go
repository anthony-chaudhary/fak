package dispatchtick

import "strings"

// SelfSourceTreePrefixes are the repository sub-trees that compile into the running
// fak orchestrator binary -- the Go module's own source. A worker spawned under
// `fak guard` cannot SHIP an edit to fak's own running code: it is editing the very
// binary (and the guard) adjudicating it, the structural self-modification hazard
// #1397 names. The dispatcher must not route such work to a self-guarded worker; the
// observed cost when it does is real (10- and 42-turn investigations that both
// terminally blocked with 0 commit on #1338's cmd/fak work). cmd/** is the CLI and the
// orchestrator verbs; internal/** is every kernel subsystem. A lane rooted in either is
// operator-gated for a guarded worker -- it belongs on an unguarded/operator or
// worktree-isolated path (#1334), not a self-guarded worker.
//
// This is a PRUDENTIAL pre-route over the whole Go module, deliberately broader than
// the guard's literal self_modify_globs default (.git/, .env, id_rsa, ...): the
// dispatcher refuses to spawn into fak's own source AT ALL rather than discover the
// block tool-call by tool-call after a worker has already burned turns.
var SelfSourceTreePrefixes = []string{"cmd/", "internal/"}

// IsSelfSourceTree reports whether one lane-tree glob is rooted in fak's own running
// source (cmd/** or internal/**). A leading "./" or "fak/" module prefix is tolerated
// so a tree written as fak/internal/... still matches, and backslashes are normalized
// so a Windows-authored glob matches the same as a POSIX one.
func IsSelfSourceTree(glob string) bool {
	g := strings.ReplaceAll(strings.TrimSpace(glob), "\\", "/")
	g = strings.TrimPrefix(g, "./")
	g = strings.TrimPrefix(g, "fak/")
	for _, prefix := range SelfSourceTreePrefixes {
		if strings.HasPrefix(g, prefix) {
			return true
		}
	}
	return false
}

// SelfModifyHold is the pure pre-route verdict for one dispatch pick: a GUARDED worker
// aimed at a lane whose tree is part of fak's own running source can do real
// investigation but can never SHIP -- the only safe ships for a self-guarded worker are
// the non-self-modify lanes (docs, tools/*.py, .github, examples). Rather than spawn a
// doomed worker that burns turns and lands 0 commits, the tick HOLDS the pick.
//
// It returns held=true only when guarded AND at least one lane tree is self-source, and
// names the first offending tree as the witness. An UNGUARDED worker (the guard disabled,
// or a worktree-isolated/operator path) is never held -- that is exactly the escape #1397
// points operators toward.
func SelfModifyHold(guarded bool, laneTree []string) (held bool, tree string) {
	if !guarded {
		return false, ""
	}
	for _, t := range laneTree {
		if IsSelfSourceTree(t) {
			return true, t
		}
	}
	return false, ""
}
