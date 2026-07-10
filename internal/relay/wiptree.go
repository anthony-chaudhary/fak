// Rung C6 (issue #3878): re-materialize the working-tree delta on resume. A baton's
// ProgressCursor.WipTree names a refs/fak/wip/<session> checkpoint object (spine #3872) —
// the uncommitted bytes the predecessor leg wrote but never committed. reload.go (#1877)
// re-verifies the start_sha anchor; this rung, gated on that anchor being fresh, re-applies
// the WipTree delta onto the resumed working tree so the successor picks up the CODE, not
// just the conversation. Fail-closed by construction: a delta that no longer applies cleanly
// to the current HEAD is DEFERRED with RELAY_WIP_STALE, never force-applied over a diverged
// tree — the same discipline stale.go uses for a diverged cursor, aimed at the delta.
//
// Pure core / injected port split (the resolve.go idiom): the fold decides over a WipRestorer
// interface and does no git I/O itself, so RematerializeWip / ResumeRematerialize are
// unit-testable with a fake restorer; GitWipRestorer is the production wiring that shells git.
package relay

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ReasonWipStale is the relay reason token emitted when a baton's WipTree delta cannot be
// re-materialized cleanly onto the resumed tree (a diverged base, a vanished object, or an
// unreachable store). Like ReasonBatonStale it is a member of the closed reason vocabulary
// the successor routes on — here it means "defer the delta, do not clobber", not "abort the
// resume": the conversation still reloads; only the uncommitted bytes are held back.
const ReasonWipStale = "RELAY_WIP_STALE"

// WipApplyVerdict is the closed outcome of asking the store to re-materialize a WipTree
// checkpoint object onto the current working tree. It mirrors ResolveVerdict's tri-state so
// an unreachable store (WipUnavailable) is never mistaken for a delta that applied cleanly.
type WipApplyVerdict string

const (
	// WipApplied means the checkpoint object resolved and its delta applied cleanly to the
	// current HEAD — the predecessor's uncommitted bytes are back in the working tree.
	WipApplied WipApplyVerdict = "applied"
	// WipConflict means the object resolved but its delta does NOT apply to the current tree
	// (the base diverged, or a peer already changed the same lines). A conforming restorer
	// reports this WITHOUT mutating the tree, so resume can defer with nothing half-written.
	WipConflict WipApplyVerdict = "conflict"
	// WipUnavailable means no verdict could be reached: the object is missing or the store is
	// unreachable. Kept distinct from WipConflict so an unreachable store is never read as a
	// clean apply — fail closed, not a false positive.
	WipUnavailable WipApplyVerdict = "unavailable"
)

// WipApplyResult is the typed outcome of one re-materialize attempt: the verdict plus a short
// human-readable detail (display-only, never consumed as progress).
type WipApplyResult struct {
	Verdict WipApplyVerdict `json:"verdict"`
	Detail  string          `json:"detail"`
}

// WipRestorer re-materializes a WipTree checkpoint object onto the working tree. It is a
// per-store port (like Resolver) so the resume fold is unit-testable without a live repo: a
// fake restorer returns a fixed verdict; GitWipRestorer provides the production wiring. A
// conforming restorer MUST be fail-closed — it never mutates the tree on a path that returns
// WipConflict or WipUnavailable, so the caller can defer without a half-applied delta.
type WipRestorer interface {
	Restore(objectID string) WipApplyResult
}

// RematerializeVerdict is the closed outcome of the resume re-materialize step.
type RematerializeVerdict string

const (
	// WipAbsent means the baton carried no WipTree — there is nothing to re-materialize and
	// the successor resumes exactly as a pre-#3878 baton did (back-compat).
	WipAbsent RematerializeVerdict = "absent"
	// WipRematerialized means the WipTree delta was re-applied cleanly onto the working tree.
	WipRematerialized RematerializeVerdict = "rematerialized"
	// WipDeferred means the delta could not be re-applied cleanly (or the anchor was stale)
	// and was left untouched for the operator/successor to reconcile — it carries
	// ReasonWipStale. Never a clobber.
	WipDeferred RematerializeVerdict = "deferred"
)

// RematerializeOutcome is the typed result of the resume re-materialize step: the verdict, the
// ReasonWipStale token when deferred, the WipTree object it acted on, and display-only detail.
// It carries no applied bytes and no progress number — the delta lives in git; this is a
// pointer-shaped report of what happened to it, so it keeps the baton's no-`claimed` posture.
type RematerializeOutcome struct {
	Verdict RematerializeVerdict `json:"verdict"`
	Reason  string               `json:"reason,omitempty"`
	WipTree string               `json:"wip_tree,omitempty"`
	Detail  string               `json:"detail,omitempty"`
}

// RematerializeWip re-materializes cur.WipTree through the injected restorer and folds the
// store verdict into the closed resume outcome. An empty WipTree is WipAbsent (nothing to do).
// A clean apply is WipRematerialized. A conflict or an unavailable object is WipDeferred with
// ReasonWipStale — fail-closed, since neither proves the delta is safe to force onto the
// resumed tree. Pure over the injected restorer: it reads no clock and does I/O only through
// the port. Callers that must also gate on the start_sha anchor use ResumeRematerialize.
func RematerializeWip(cur ProgressCursor, restore WipRestorer) RematerializeOutcome {
	if cur.WipTree == "" {
		return RematerializeOutcome{Verdict: WipAbsent}
	}
	res := restore.Restore(cur.WipTree)
	switch res.Verdict {
	case WipApplied:
		return RematerializeOutcome{Verdict: WipRematerialized, WipTree: cur.WipTree, Detail: res.Detail}
	case WipConflict:
		return RematerializeOutcome{Verdict: WipDeferred, Reason: ReasonWipStale, WipTree: cur.WipTree, Detail: res.Detail}
	default: // WipUnavailable — fail closed to a deferral, never a clobber.
		return RematerializeOutcome{Verdict: WipDeferred, Reason: ReasonWipStale, WipTree: cur.WipTree, Detail: res.Detail}
	}
}

// ResumeRematerialize is the resume-path composition a driver calls: it re-materializes the
// WipTree delta ONLY after VerifyReload confirms the start_sha anchor is fresh — exactly the
// order issue #3878 specifies ("after VerifyReload confirms start_sha, re-materialize the
// wip_tree delta before the first resumed turn"). If start_sha is stale the whole cursor is
// untrustworthy, so re-applying the delta over a diverged base is unsafe: the outcome is
// WipDeferred / ReasonWipStale WITHOUT ever calling the restorer, and the successor re-derives
// from durable state (the RELAY_BATON_STALE path) instead of clobbering the tree. A baton with
// no WipTree short-circuits to WipAbsent so an old baton resumes exactly as today. Pure over
// the two injected ports.
func ResumeRematerialize(b Baton, r Resolver, restore WipRestorer) RematerializeOutcome {
	if b.ProgressCursor.WipTree == "" {
		return RematerializeOutcome{Verdict: WipAbsent}
	}
	if rr := VerifyReload(b.ProgressCursor, r); rr.Verdict != ReloadFresh {
		return RematerializeOutcome{
			Verdict: WipDeferred,
			Reason:  ReasonWipStale,
			WipTree: b.ProgressCursor.WipTree,
			Detail:  "start_sha anchor is not fresh; re-materializing the wip_tree delta over a diverged base is unsafe: " + rr.Reason,
		}
	}
	return RematerializeWip(b.ProgressCursor, restore)
}

// GitWipRestorer is the production WipRestorer: it re-materializes a checkpoint object's delta
// onto the working tree of the git repo rooted at dir, fail-closed. The delta is the patch
// `git diff <object>^1 <object>` — the change the checkpoint captured against the HEAD it was
// taken from, the same form cmd/fak/wip.go restore uses. That patch is TEST-applied with
// `git apply --check` before the real apply, so a delta that no longer fits the current tree
// yields WipConflict WITHOUT mutating a single byte; only a clean check is followed by the real
// `git apply`. A missing/unresolvable object, or git that cannot be run at all, is
// WipUnavailable (fail closed) — never a false WipApplied.
type GitWipRestorer struct {
	dir string
}

// NewGitWipRestorer builds a GitWipRestorer over the repo rooted at dir ("" = the current
// working directory, matching git's own default).
func NewGitWipRestorer(dir string) GitWipRestorer { return GitWipRestorer{dir: dir} }

// Restore implements WipRestorer against a live git repo. See GitWipRestorer for the contract.
func (g GitWipRestorer) Restore(objectID string) WipApplyResult {
	obj := strings.TrimSpace(objectID)
	if obj == "" {
		return WipApplyResult{Verdict: WipUnavailable, Detail: "empty wip_tree object id"}
	}
	// 1) the object must resolve as a commit in the store (a stash-create checkpoint object).
	typ, err := g.git("cat-file", "-t", obj)
	if err != nil {
		return WipApplyResult{Verdict: WipUnavailable, Detail: "git cat-file failed: " + err.Error()}
	}
	if strings.TrimSpace(typ) != "commit" {
		return WipApplyResult{Verdict: WipUnavailable, Detail: "wip_tree object is not a commit: " + obj}
	}
	// 2) materialize the checkpoint's captured delta as a unified diff against its base HEAD.
	patch, err := g.git("diff", obj+"^1", obj)
	if err != nil {
		return WipApplyResult{Verdict: WipUnavailable, Detail: "git diff failed: " + err.Error()}
	}
	if strings.TrimSpace(patch) == "" {
		return WipApplyResult{Verdict: WipApplied, Detail: "empty delta; nothing to re-materialize"}
	}
	// 3) fail-closed: test-apply first so a conflict never leaves a half-written tree.
	if err := g.apply(patch, true); err != nil {
		return WipApplyResult{Verdict: WipConflict, Detail: "delta does not apply to the current tree: " + err.Error()}
	}
	if err := g.apply(patch, false); err != nil {
		return WipApplyResult{Verdict: WipConflict, Detail: "delta failed to apply after a clean --check: " + err.Error()}
	}
	return WipApplyResult{Verdict: WipApplied, Detail: "re-materialized wip_tree delta " + obj}
}

// git runs a read-only-ish git subcommand in g.dir and returns its stdout. A non-zero git
// exit (an unknown object, a bad ref) is surfaced as an error with git's stderr — the caller
// maps that to WipUnavailable; a failure to run git at all surfaces the same way.
func (g GitWipRestorer) git(args ...string) (string, error) {
	full := args
	if g.dir != "" {
		full = append([]string{"-C", g.dir}, args...)
	}
	cmd := exec.Command("git", full...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("git %s: %s", args[0], strings.TrimSpace(string(ee.Stderr)))
		}
		return "", err
	}
	return string(out), nil
}

// apply pipes patch to `git apply` (working tree only — never the index or a commit). With
// check=true it runs `--check`, the non-mutating dry run that decides WipConflict before any
// bytes move; with check=false it performs the real apply. --whitespace=nowarn keeps a benign
// trailing-newline delta from being rejected (mirrors cmd/fak/wip.go's wipApplyPatch).
func (g GitWipRestorer) apply(patch string, check bool) error {
	args := []string{"apply", "--whitespace=nowarn"}
	if check {
		args = append(args, "--check")
	}
	if g.dir != "" {
		args = append([]string{"-C", g.dir}, args...)
	}
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Stdin = strings.NewReader(patch)
	var errb strings.Builder
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%v: %s", err, strings.TrimSpace(errb.String()))
	}
	return nil
}
