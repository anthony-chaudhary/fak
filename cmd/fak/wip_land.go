package main

// wip_land.go — `fak wip land`: turning a session's checkpoint ref into a real commit.
// Holds the land verb, its scope-aware core (wipLandWith), the patch/delta plumbing it
// uses to materialize the snapshot, and the default audit-OK commit subject. Split
// verbatim out of wip.go — no behaviour change — to hold the internal/godfileceiling
// 1500-line god-file cap; the tests for this group already live in wip_land_test.go.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// ---- land (#3876: stamp-on-recover) ----

// wipLandResult is the JSON/plain outcome of a land.
type wipLandResult struct {
	Session      string   `json:"session"`
	Object       string   `json:"object,omitempty"`       // the checkpoint commit landed
	Files        []string `json:"files,omitempty"`        // the exact pathspec committed
	Excluded     []string `json:"excluded,omitempty"`     // in the snapshot, deliberately NOT landed (#5539)
	Materialized string   `json:"materialized,omitempty"` // applied | present | empty | conflict
	Subject      string   `json:"subject,omitempty"`      // the commit subject used
	SHA          string   `json:"committed_sha,omitempty"`
	Committed    bool     `json:"committed"`
	Verified     bool     `json:"verified"`
	Grade        string   `json:"grade,omitempty"`
	Reason       string   `json:"reason,omitempty"` // a closed refusal token when not committed
}

func runWipLand(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip land", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	session := fs.String("session", "", "session id to land (default: $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID)")
	message := fs.String("m", "", "commit subject (default: an audit-OK recovery subject naming the leaf + session)")
	push := fs.Bool("push", false, "push after a verified commit")
	var paths pathList
	fs.Var(&paths, "path", "a repo-relative path to land (repeatable); narrows BOTH the pathspec committed and the patch materialized, leaving the rest of the snapshot alone")
	all := fs.Bool("all", false, "land the WHOLE snapshot, a concurrent peer's captured edits included — the declared escape from the TREE_WIDE_SNAPSHOT refusal")
	asJSON := fs.Bool("json", false, "emit the land result as JSON")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}
	if *all && len(paths) > 0 {
		fmt.Fprintln(stderr, "fak wip land: --all and --path are mutually exclusive (--all takes the whole snapshot; --path declares a subset)")
		return 2
	}
	// Session: an optional positional wins, else --session, else the env default the
	// checkpoint verb uses — so `fak wip land` with no args lands your own checkpoint.
	sess := strings.TrimSpace(*session)
	if rest := fs.Args(); len(rest) > 0 {
		if len(rest) != 1 {
			fmt.Fprintln(stderr, "fak wip land: at most one <session> argument (flags must precede it, e.g. `fak wip land -C <repo> --json <session>`)")
			return 2
		}
		sess = strings.TrimSpace(rest[0])
	}
	if sess == "" {
		sess = firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
	}
	if sess == "" {
		fmt.Fprintln(stderr, "fak wip land: no session id (pass <session>/--session or set $CLAUDE_CODE_SESSION_ID)")
		return 2
	}
	if !wipref.ValidSession(sess) {
		fmt.Fprintf(stderr, "fak wip land: invalid session id %q (must be one safe ref segment)\n", sess)
		return 2
	}

	res, code, err := wipLandWith(context.Background(), *repo, sess, wipLandOptions{
		Message: strings.TrimSpace(*message),
		Push:    *push,
		Paths:   paths,
		All:     *all,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak wip land: %v\n", err)
	}
	if *asJSON {
		if jc := encodeJSONOrFail(stdout, stderr, res, "fak wip land"); jc != 0 {
			return jc
		}
		return code
	}
	if code == 0 && res.Committed {
		fmt.Fprintf(stdout, "landed %s: checkpoint %s (%d file(s)) committed %s [%s]\n",
			sess, res.Object, len(res.Files), res.SHA, res.Grade)
		fmt.Fprintf(stdout, "  subject: %s\n", res.Subject)
		if len(res.Excluded) > 0 {
			fmt.Fprintf(stdout, "  excluded %d captured file(s) outside the declared scope: %s\n",
				len(res.Excluded), strings.Join(res.Excluded, ", "))
		}
		fmt.Fprintln(stdout, "  the delta is now in HEAD; `fak wip reap` will clear the checkpoint ref.")
		return 0
	}
	if code == 0 && res.Materialized == "empty" {
		fmt.Fprintf(stdout, "nothing to land for %s: the checkpoint delta is empty\n", sess)
		return 0
	}
	return code
}

// wipLand lands a session's checkpoint with no declared scope — the original signature,
// kept for callers that have no paths to declare. It is a thin wrapper over wipLandWith,
// so those callers still get the #5539 refusal when the snapshot is unattributable.
func wipLand(ctx context.Context, repo, session, message string, push bool) (wipLandResult, int, error) {
	return wipLandWith(ctx, repo, session, wipLandOptions{Message: message, Push: push})
}

// wipLandWith turns a session's checkpoint into a real commit. It resolves the checkpoint,
// materializes its delta into the WORKING TREE ONLY (never the index — so safecommit's
// prestaged-overlap guard stays clean) when the delta is not already present, and
// REFUSES rather than clobbering when the tree has diverged (a peer edited the same
// files, or the owner kept editing past the checkpoint — re-checkpoint then land). It
// then commits exactly the delta's file set through safecommit.Commit, whose realRunner
// sets FAK_SAFECOMMIT_VETTED so the BARE_COMMIT_SWEEP gate stands down and which verifies
// only those paths landed (PATHSPEC_RACE). The default subject is shaped to grade the dos
// commit-audit OK (a _CODE_VERBS word leads, no whole-word no-claim marker, source is
// touched → diff-witnessed).
//
// Because the CAPTURE is tree-wide (wipCheckpointScoped), the object routinely holds a
// concurrent peer's edits, and land is the one verb that turns that snapshot into an
// irreversible act. So land resolves a SCOPE first (wipResolveLandScope) and commits only
// the files in it, refusing outright when the snapshot is unattributable. The scope also
// narrows the patch materialized, not just the pathspec committed — otherwise a `--path`
// land would still write a peer's hunks into the working tree.
//
// Returns (result, exitCode, err): exit 0 committed/empty, 3 a checkable refusal
// (TREE_WIDE_SNAPSHOT, SCOPE_MATCHED_NOTHING, TREE_DIVERGED), 1 a runtime error or a
// safecommit refusal.
func wipLandWith(ctx context.Context, repo, session string, opts wipLandOptions) (wipLandResult, int, error) {
	res := wipLandResult{Session: session}

	ref := wipref.SessionRef(session)
	obj, _, code, err := gitWip(ctx, repo, nil, "rev-parse", "--verify", "--quiet", ref)
	if err != nil {
		return res, 1, fmt.Errorf("git rev-parse: %w", err)
	}
	obj = strings.TrimSpace(obj)
	if code != 0 || obj == "" {
		return res, 1, fmt.Errorf("no checkpoint for session %q", session)
	}
	res.Object = obj

	captured, err := wipDeltaFiles(ctx, repo, obj)
	if err != nil {
		return res, 1, err
	}
	res.Files = captured
	if len(captured) == 0 {
		res.Materialized = "empty"
		res.Reason = "EMPTY_DELTA"
		return res, 0, nil // an empty checkpoint is a clean no-op, not an error
	}

	// Decide WHAT may land before anything is materialized, so a refusal leaves the
	// working tree exactly as it found it.
	files, excluded, reason, err := wipResolveLandScope(ctx, repo, session, ref, obj, captured, opts)
	if reason != "" {
		res.Files, res.Excluded, res.Reason = nil, captured, reason
		return res, 3, err
	}
	if err != nil {
		return res, 1, err
	}
	res.Files, res.Excluded = files, excluded

	diffArgs := append([]string{"diff", obj + "^1", obj, "--"}, files...)
	patch, _, dcode, err := gitWip(ctx, repo, nil, diffArgs...)
	if err != nil {
		return res, 1, fmt.Errorf("git diff: %w", err)
	}
	if dcode != 0 {
		return res, 1, fmt.Errorf("git diff exited %d", dcode)
	}
	if strings.TrimSpace(patch) == "" {
		res.Materialized = "empty"
		res.Reason = "EMPTY_DELTA"
		return res, 0, nil // an empty checkpoint is a clean no-op, not an error
	}

	// Materialize the checkpoint delta into the working tree if it is not already there,
	// using the same `git apply --check` discriminator wipDeltaApplies/reconcile use.
	switch {
	case wipPatchChecks(ctx, repo, patch, false): // forward-applies: tree is at baseline
		if err := wipApplyPatch(ctx, repo, patch); err != nil {
			return res, 1, err
		}
		res.Materialized = "applied"
	case wipPatchChecks(ctx, repo, patch, true): // reverse-applies: delta already present
		res.Materialized = "present"
	default:
		res.Materialized = "conflict"
		res.Reason = "TREE_DIVERGED"
		return res, 3, fmt.Errorf("working tree diverges from the %q checkpoint delta — re-checkpoint then land, or resolve with `fak wip reconcile`", session)
	}

	subject := opts.Message
	if subject == "" {
		subject = wipLandSubject(session, files)
	}
	res.Subject = subject

	cr, err := safecommit.Commit(ctx, safecommit.Options{
		Dir:     repo,
		Paths:   files,
		Message: subject,
		SignOff: true,
		Push:    opts.Push,
		// Scope the advisory commit lock to the TARGET repo, not the process cwd:
		// safecommit's default lock path is derived from the current working directory,
		// so a land invoked with -C on a different repo (a fleet host landing a crashed
		// peer) would otherwise serialize on the wrong .git. See wipCommitLockPath.
		Lock: safecommit.LockOptions{Path: wipCommitLockPath(ctx, repo)},
	})
	if err != nil {
		res.Reason = firstNonEmpty(cr.Reason, "COMMIT_ERROR")
		return res, 1, fmt.Errorf("safecommit: %w", err)
	}
	res.SHA, res.Committed, res.Verified, res.Grade = cr.SHA, cr.Committed, cr.Verified, cr.Grade
	if cr.Reason != "" || !cr.Committed {
		res.Reason = firstNonEmpty(cr.Reason, "COMMIT_REFUSED")
		return res, 1, fmt.Errorf("safecommit refused (%s): %s", res.Reason, strings.TrimSpace(cr.Detail))
	}
	return res, 0, nil
}

// wipPatchChecks reports whether the RAW patch applies cleanly to the current working
// tree — forward (reverse=false, "not yet applied") or reversed (reverse=true, "already
// present"). Same `git apply --check` gate as wipDeltaApplies, generalized so land can
// tell a clean baseline from an already-materialized delta from a true divergence. The
// untrimmed patch is fed so its trailing newline survives the check.
func wipPatchChecks(ctx context.Context, repo, patch string, reverse bool) bool {
	if strings.TrimSpace(patch) == "" {
		return false
	}
	args := []string{"apply", "--check"}
	if reverse {
		args = append(args, "-R")
	}
	args = append(args, "-")
	_, _, code, err := gitWipStdin(ctx, repo, patch, args...)
	return err == nil && code == 0
}

// wipCommitLockPath resolves the advisory commit-lock path for the TARGET repo
// (<git-dir>/fak-commit.lock) so a land invoked with -C locks the repo it commits into,
// not the process's cwd repo — safecommit's realLock otherwise derives the lock path from
// the current working directory. "" on failure lets safecommit fall back to its default.
func wipCommitLockPath(ctx context.Context, repo string) string {
	gd, err := gitWipOut(ctx, repo, nil, "rev-parse", "--absolute-git-dir")
	if err != nil || strings.TrimSpace(gd) == "" {
		return ""
	}
	return filepath.Join(strings.TrimSpace(gd), "fak-commit.lock")
}

// wipDeltaFiles enumerates the exact set of files a checkpoint's delta touches
// (`git diff --name-only <obj>^ <obj>`) — the explicit pathspec land stages. This is the
// precise per-file set, NOT the coarser Stamp.Leaves (which folds files to directories).
func wipDeltaFiles(ctx context.Context, repo, obj string) ([]string, error) {
	out, err := gitWipOut(ctx, repo, nil, "diff", "--name-only", obj+"^", obj)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, ln := range strings.Split(out, "\n") {
		if s := strings.TrimSpace(ln); s != "" {
			files = append(files, s)
		}
	}
	return files, nil
}

// wipLandSubject builds the default commit subject, shaped to grade the dos commit-audit
// OK: the _CODE_VERBS word "land" leads the description after the scope (→ code_effect),
// it carries NO whole-word no-claim marker (notably not "wip"), and a normal source
// recovery touches a .go file (→ diff-witnessed). A caller's -m overrides this. See
// [[dos-commit-audit-ok-grammar]].
func wipLandSubject(session string, files []string) string {
	return fmt.Sprintf("feat(%s): land %d recovered working-tree file(s) from the %s checkpoint",
		wipLandScope(files), len(files), session)
}

// wipLandScope picks the commit scope: the dominant top-level path segment of the delta
// (ties broken by the lexically smallest, so the subject is deterministic), falling back
// to "cmd" for a root-level-only delta. Never "wip" — a marker scope would flip the audit
// to ABSTAIN.
func wipLandScope(files []string) string {
	counts := map[string]int{}
	for _, f := range files {
		top := f
		if i := strings.IndexByte(f, '/'); i >= 0 {
			top = f[:i]
		}
		counts[top]++
	}
	best, bestN := "", 0
	for k, n := range counts {
		if n > bestN || (n == bestN && k < best) {
			best, bestN = k, n
		}
	}
	if best == "" || best == "wip" {
		best = "cmd"
	}
	return best
}
