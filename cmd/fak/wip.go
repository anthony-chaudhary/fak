package main

// wip.go — `fak wip`, the working-tree checkpoint/restore spine (#3872). It gives
// a session a durable, gc-safe snapshot of its uncommitted tracked changes under
// refs/fak/wip/<session> (a sibling of the lease refs, refs/fak/locks/*) WITHOUT
// touching the index, the working tree, or any branch/HEAD.
//
//	fak wip checkpoint [--session <id>] [-C <repo>]   # snapshot the tracked delta
//	fak wip status [-C <repo>] [--json]               # list the live checkpoints
//	fak wip restore <session> [-C <repo>] [--apply]   # re-materialize the delta
//	fak wip land [<session>] [-C <repo>] [-m <subj>]  # commit the delta (audit-OK)
//	fak wip selfcheck                                 # checkpoint->wipe->restore proof
//
// This shell owns only the git I/O the pure core (internal/wipref) must not: it
// captures the delta into a temp-index tree, mints a commit-tree object stamped
// with the wipref.Stamp, and points the ref at it. The ref->object edge keeps the
// snapshot reachable until the ref is deleted (retention). Restore is a plain
// `git diff <commit>^1 <commit>` — the apply-able delta.

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipattr"
	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

func cmdWip(argv []string) { os.Exit(runWip(os.Stdout, os.Stderr, argv)) }

// runWip dispatches `fak wip`. Exit codes: 0 ok, 1 runtime error, 2 usage error.
func runWip(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		wipUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "checkpoint":
		return runWipCheckpoint(stdout, stderr, argv[1:])
	case "autocheckpoint", "auto-checkpoint":
		return runWipAutoCheckpoint(stdout, stderr, argv[1:])
	case "status":
		return runWipStatus(stdout, stderr, argv[1:])
	case "sync":
		return runWipSync(stdout, stderr, argv[1:])
	case "remote-drain":
		return runWipRemoteDrain(stdout, stderr, argv[1:])
	case "restore":
		return runWipRestore(stdout, stderr, argv[1:])
	case "land":
		return runWipLand(stdout, stderr, argv[1:])
	case "fence":
		return runWipFence(stdout, stderr, argv[1:])
	case "unfence":
		return runWipUnfence(stdout, stderr, argv[1:])
	case "reap":
		return runWipReap(stdout, stderr, argv[1:])
	case "attribute", "attr":
		return runWipAttribute(stdout, stderr, argv[1:])
	case "owner", "own":
		return runWipOwner(stdout, stderr, argv[1:])
	case "blocked":
		return runWipBlocked(stdout, stderr, argv[1:])
	case "inventory":
		return runWIPInventory(argv[1:], stdout, stderr)
	case "queue":
		return runWIPQueue(argv[1:], stdout, stderr)
	case "readiness":
		return runWIPReadiness(argv[1:], stdout, stderr)
	case "lifecycle":
		return runWIPLifecycle(argv[1:], stdout, stderr)
	case "reconcile":
		return runWipReconcile(stdout, stderr, argv[1:])
	case "sweep-guard", "sweepguard":
		return runWipSweepGuard(stdout, stderr, argv[1:])
	case "admit":
		return runWipAdmit(stdout, stderr, argv[1:])
	case "selfcheck", "--selfcheck", "-selfcheck":
		return runWipSelfcheck(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		wipUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak wip: unknown subcommand %q\n", argv[0])
		wipUsage(stderr)
		return 2
	}
}

func wipUsage(w io.Writer) {
	fmt.Fprint(w, `fak wip — working-tree checkpoint/restore over refs/fak/wip/* (#3872)

  fak wip checkpoint [--session <id>] [-C <repo>] [--path <p>]... [--buildable=<bool>] [--json]
      Snapshot the current tracked working-tree delta into a gc-safe object under
      refs/fak/wip/<session>, WITHOUT touching the index, working tree, or a branch.
      Session defaults to $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID.
      The capture is deliberately TREE-WIDE (a shared tree's peer edits included) so a
      crashed session loses nothing. --path does NOT narrow it: it records what THIS
      session claims, in the stamp, so a later 'fak wip land' — possibly run by a fleet
      host that cannot ask the dead session — commits only those paths.

  fak wip autocheckpoint [--reason compaction|stop|doomloop|manual] [--session <id>] [-C <repo>] [--strict] [--json]
      Best-effort capture of the session's WIP at a risky boundary (#3877): the
      compaction step-advice, the guard Stop hook, or a doomloop NUDGE. Unlike
      checkpoint, it NEVER hard-fails by default (exit 0 even on a capture error, a
      missing session, or a clean tree) so it can never break the boundary it fires
      at; pass --strict to surface a capture failure as exit 1.

  fak wip status [-C <repo>] [--remote R] [--json]
      List the live working-tree checkpoints (one per session), sorted by session, each
      graded REPLICATED / STALE_REMOTE / LOCAL_ONLY against the remote (default origin).
      That column is the difference between the two failures a checkpoint can face:
      LOCAL_ONLY survives THIS SESSION dying (what autocheckpoint protects against) and
      does NOT survive this checkout or machine going away; STALE_REMOTE means an older
      checkpoint made it off-machine but the current delta did not. The verdict is read
      from this clone's mirror of the remote — a LOCAL ref read, never a network probe,
      so status still answers at a boundary where the network is what failed. It is a
      last-known claim as of your last 'fak wip sync', not a live interrogation.
      The report therefore also carries the mirror's PROVENANCE: NEVER_SYNCED / FRESH /
      STALE, when the last sync ran and whether it FETCHed the remote's whole namespace
      or only PUSHed this clone's. Read that line before reading anything absent from
      the mirror as absent from the remote — a clone that has never fetched, or fetched
      last Tuesday, reports the same emptiness as a peer that genuinely holds nothing.

  fak wip sync [-C <repo>] [--remote R] [--push-only|--fetch-only] [--json]
      REPLICATE the refs/fak/wip/* namespace to a remote (default origin): push this
      clone's checkpoints, then refresh the read-only mirror the status column reads.
      OPT-IN by design and never run for you — a checkpoint is a TREE-WIDE capture of a
      dirty working tree, so publishing it off-machine is a privacy and bandwidth
      decision you make deliberately. --push-only publishes without downloading a peer
      host's captured trees; --fetch-only imports theirs without publishing yours.
      Push runs FIRST and a failed push STOPS the sync, so a sync that errors has
      changed nothing locally. Both refspecs are FORCED (a checkpoint commit is parented
      on HEAD, never on the previous checkpoint, so two checkpoints of one session are
      siblings and a plain update is rejected) and confined to the checkpoint namespaces
      — no branch, HEAD, or tag ever moves. Deletions do not ride a refspec: a landed or
      reaped checkpoint converges on peers via their own 'fak wip reap', never a prune.
      The FETCH lands in refs/fak/remotewip/<remote>/*, deliberately NOT in the live
      namespace: every wip verb that reads refs/fak/wip/* is tree-relative and one of
      them ('reap') deletes, so a peer host's checkpoints must never join that set.
      A completed sync stamps refs/fak/checkpointsync/<remote> with the time and the
      direction, so a later reader can date this clone's picture of the remote instead
      of mistaking an unfetched mirror for an empty one. The stamp is local-only — the
      push refspec cannot carry it — and a sync that fails leaves none.

  fak wip remote-drain [-C <repo>] [--remote R] [--apply] [--allow-peer] [--json]
      Report remotely stored checkpoints and delete only those whose complete delta is
      independently witnessed in the remote default branch. Report mode is read-only.
      Age is never evidence. Own sessions are eligible by default; peers require
      --allow-peer and the same containment proof. Unlanded or unknown work is kept.

  fak wip restore <session> [-C <repo>] [--apply]
      Print the checkpointed delta as an apply-able diff (default) or, with --apply,
      re-materialize it onto the current working tree.

  fak wip land [<session>] [-C <repo>] [-m <subject>] [--path <p>]... [--all] [--push] [--json]
      Turn a session's checkpoint into a real commit: materialize its delta into the
      working tree (refusing, never clobbering, if the tree has diverged), then commit
      EXACTLY the declared file set through safecommit (explicit pathspec, vetted so the
      bare-commit guard stands down). The default subject is shaped to grade the dos
      commit-audit OK; -m overrides it. Session defaults to $CLAUDE_CODE_SESSION_ID.
      Because the capture is tree-wide, land REFUSES (exit 3, TREE_WIDE_SNAPSHOT) to
      commit an undeclared snapshot while another session holds a live checkpoint — that
      would land a peer's work under your name. Declare your paths with --path (which
      narrows both the commit and the patch applied), rely on the scope the checkpoint
      stamped, or take the whole snapshot deliberately with --all.

  fak wip fence <file> [--feature <slug>] | --all-untracked [-C <repo>]
      Prepend //go:build wip_<slug> (+ blank line) so a not-yet-compiling untracked .go
      stays OUT of the default build (green for peers/CI) and IN under -tags wip_<slug>.
      Idempotent; reversible with 'fak wip unfence'. --all-untracked bulk-fences a
      poisoned tree.

  fak wip unfence <file> [-C <repo>]
      Remove a //go:build wip_<slug> fence once the defining symbol has landed. Idempotent.

  fak wip reap [-C <repo>] [--json] [--dry-run] [--census]
      Delete redundant checkpoint refs whose delta has LANDED in HEAD (the owner
      committed it). Fail-safe: an unlanded checkpoint is always kept.
      --census is a READ-ONLY reporting mode (deletes nothing): it classifies every
      refs/fak/wip/* ref by owner-state — LANDED / LIVE / CLOSED_CLEAN_ESTIMATE /
      CLOSED_DIRTY_RECOVERABLE / UNKNOWN — and prints the counts (+ a --json breakdown),
      so #5340 can size how many dead-session checkpoints are safe to collect.

  fak wip attribute [-C <repo>] [--json] [--orphans]
      Attribute every dirty working-tree hunk to the session that checkpointed it
      (OWNED), to several (SHARED), or to none (ORPHAN — unattributed, at-risk WIP).
      With --orphans, print only the ORPHAN hunks (exit 3 if any exist).

  fak wip owner [-C <repo>] [--ttl <dur>] [--json] [--unclaimed] [<path>...]
      Answer "whose is this?" for a CREATED path — the case attribute structurally
      cannot see, because a file absent from HEAD produces no 'git diff HEAD' hunk.
      Evidence is the checkpoint capture itself (read-tree HEAD + add -A records every
      untracked path as an ADDITION), so each created path grades CLAIMED_LIVE (one
      fresh claimant), AMBIGUOUS (several — tree-wide capture cannot name an author, so
      none is chosen), CLAIMED_EXPIRED (named owner, check-in overdue — NEVER a reap
      licence), or UNCLAIMED (no fresh checkpoint records it: at risk from any broad
      add/clean). Defaults to every untracked path; --unclaimed prints only the at-risk
      set and exits 3 if any exist. Read-only. The --ttl claim window is also the cost
      bound: only checkpoints inside it (or held by a live session) are read.

  fak wip admit [-C <repo>] --self <session> [--intend <glob>]... [--strict] [--ceiling N] [--json]
      Read-only start-of-task admission. Refuse hard peer collisions and untracked
      WIP before beginning another unit; optionally promote soft intent/self-WIP
      pressure to HOLD with --strict.

  fak wip blocked [-C <repo>] [--ledger <path>] [--stale-days N] [--landable] [--json]
      Rank the dirty working tree by the dispatch admissions each path has REFUSED
      (parsed from the loop ledger's DIRTY_PATH_COLLISION / SAME_ISSUE_WIP rows), so
      the orphan actually throttling the fleet sorts first. Verdicts: LAND (blocking
      and its whole change set is idle — the lever), WAIT (blocking but the set is
      live; the refusal is correct), IDLE, ACTIVE. Staleness is judged on the change
      SET, never one file's mtime, so the stale half of a live set is never offered.
      With --landable, print only the LAND rows (exit 3 if any exist).

  fak wip inventory [--json] [--root DIR] [--max-untracked-age DURATION]
      Read-only census of main WIP, ignored files, worktrees, stale residue, and checkpoints.

  fak wip queue [--json] [-C DIR]
      Prioritize every sanctioned worktree and local-only checkpoint into one
      deterministic, read-only action queue with the exact next command.
  fak wip readiness --json [-C DIR] [--max-age DURATION] [--remote NAME]
      Join the canonical WIP surfaces into one freshness-stamped receipt reusable
      by fresh-start admission without authorizing cleanup.

  fak wip reconcile [-C <repo>] [--json] [--reclaim] [--file-ticket] [--dry-run]
      For every checkpoint whose owning session no longer holds a live lease
      (CRASHED), decide the one safe action: DISCARD_WITNESSED (delta landed in
      HEAD), RECLAIM (unlanded, applies cleanly), or QUARANTINE (unlanded, conflicts).
      A live owner's checkpoint is SKIPped. Advisory: prints decisions, mutates nothing.
      With --reclaim, print ONLY the RECLAIM rows as a recovery worklist ranked
      most-decayed-first by BASE DRIFT (commits HEAD has advanced past the checkpoint's
      base) then age; exit 3 if any exist. RECLAIM decays into QUARANTINE as the tree
      moves, so the drift column is that verdict's remaining life — act on the top row
      first, running the exact argv that row prints: 'fak wip reconcile adopt <session>'
      for a row nobody holds, 'fak wip reconcile resume <session>' for one this session
      already claimed, and nothing at all for a row a live peer holds (wait for that
      claim to finish or lapse). Adopting takes the witnessed claim BEFORE the delta is
      re-materialized, which is what keeps two successors reading one queue from
      recovering the same checkpoint twice (#5998).
      With --file-ticket, bind each QUARANTINE orphan to ONE idempotent GitHub tracking
      ticket (keyed by session+start-SHA; a matching ticket already open is reused, not
      duplicated). --dry-run (and an unavailable gh) prints the exact ticket instead of
      filing it; the ticket pass never changes the reconcile exit code.

  fak wip sweep-guard [-C <repo>] [--session <id>] [--json]
      Warn before a broad 'git add' sweeps WIP that is not yours. Grades every dirty
      hunk SAFE (owned solely by the self session) or HAZARD (owned by a peer — LIVE
      peers flagged sharply — SHARED, or ORPHAN). Exit 3 if any hunk is a HAZARD, 0 if
      the sweep is clean. Advisory: it inspects and warns, it never stages anything.

  fak wip selfcheck [--json]
      Prove checkpoint -> git checkout -- . -> restore reproduces the delta
      byte-identical, and that status lists the checkpoint. Exit 0 on PASS.
`)
}

// ---- git runner (dir + extra env aware; raw stdout preserved for patches) ----

// gitWip runs git in dir (optional) with extra env (optional), returning RAW
// stdout, stderr, the exit code, and an exec error. A non-zero git exit is
// reported in code, not err; err is non-nil only when git could not be executed.
// stdout is NOT trimmed — a patch's exact bytes (trailing newline) must survive.
func gitWip(ctx context.Context, dir string, env []string, args ...string) (stdout, stderr string, code int, err error) {
	gitWipSpawns.Add(1) // observability only (wip_spawns.go): the O(1)-spawns witness
	cmd := exec.CommandContext(ctx, "git", args...)
	configureDispatchHelperCommand(cmd)
	if dir != "" {
		cmd.Dir = dir
	}
	if len(env) > 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	var o, e strings.Builder
	cmd.Stdout = &o
	cmd.Stderr = &e
	runErr := cmd.Run()
	if runErr == nil {
		return o.String(), e.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(runErr, &ee) {
		return o.String(), e.String(), ee.ExitCode(), nil // git ran, non-zero
	}
	return "", e.String(), -1, runErr // git not executable
}

// gitWipStdin runs git like gitWip but feeds stdin (a patch) to the process — used by
// `git apply --check -` to test whether a delta would apply cleanly without mutating
// the tree. Same error contract as gitWip: a non-zero git exit is reported in code.
func gitWipStdin(ctx context.Context, dir, stdin string, args ...string) (stdout, stderr string, code int, err error) {
	gitWipSpawns.Add(1) // observability only (wip_spawns.go): the O(1)-spawns witness
	cmd := exec.CommandContext(ctx, "git", args...)
	configureDispatchHelperCommand(cmd)
	if dir != "" {
		cmd.Dir = dir
	}
	cmd.Stdin = strings.NewReader(stdin)
	var o, e strings.Builder
	cmd.Stdout = &o
	cmd.Stderr = &e
	runErr := cmd.Run()
	if runErr == nil {
		return o.String(), e.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(runErr, &ee) {
		return o.String(), e.String(), ee.ExitCode(), nil
	}
	return "", e.String(), -1, runErr
}

// gitWipOut is the must-succeed convenience: TRIMMED stdout, or an error carrying
// git's stderr when git is not runnable or exits non-zero.
func gitWipOut(ctx context.Context, dir string, env []string, args ...string) (string, error) {
	out, errStr, code, err := gitWip(ctx, dir, env, args...)
	if err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	if code != 0 {
		return "", fmt.Errorf("git %s exited %d: %s", strings.Join(args, " "), code, strings.TrimSpace(errStr))
	}
	return strings.TrimSpace(out), nil
}

// ---- checkpoint ----

// wipCheckpointFault is a test-only crash injection seam. Production leaves it
// nil. It runs immediately before/after the durable ref CAS so the crash matrix
// can prove which claims survive each boundary without mocking Git.
var wipCheckpointFault func(point string) error

func wipCheckpointFaultAt(point string) error {
	if wipCheckpointFault != nil {
		return wipCheckpointFault(point)
	}
	return nil
}

// wipCheckpointResult is the JSON/plain result of a checkpoint.
type wipCheckpointResult struct {
	Session    string   `json:"session"`
	Ref        string   `json:"ref"`
	Object     string   `json:"object,omitempty"`
	StartSHA   string   `json:"start_sha"`
	Leaves     []string `json:"leaves"`
	Scope      []string `json:"scope,omitempty"` // the paths this session CLAIMS of the tree-wide capture (#5539)
	Buildable  bool     `json:"buildable"`
	Clean      bool     `json:"clean"`
	Superseded bool     `json:"superseded,omitempty"` // a newer concurrent checkpoint won the ref (#3873)
}

func runWipCheckpoint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip checkpoint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	session := fs.String("session", "", "session id to checkpoint under (default: $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID)")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	buildable := fs.Bool("buildable", true, "record the checkpoint as buildable (advisory stamp field)")
	var scope pathList
	// No backticks in this usage string: flag.UnquoteUsage reads the first backticked
	// span as the flag's VALUE NAME, so `fak wip land` would render as the argument name.
	fs.Var(&scope, "path", "a repo-relative path this session CLAIMS of the (still tree-wide) capture (repeatable); recorded in the stamp so a later 'fak wip land' commits only these")
	asJSON := fs.Bool("json", false, "emit the checkpoint result as JSON")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}

	sess := strings.TrimSpace(*session)
	if sess == "" {
		sess = firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
	}
	if sess == "" {
		fmt.Fprintln(stderr, "fak wip checkpoint: no session id (pass --session or set $CLAUDE_CODE_SESSION_ID)")
		return 2
	}
	if !wipref.ValidSession(sess) {
		fmt.Fprintf(stderr, "fak wip checkpoint: invalid session id %q (must be one safe ref segment)\n", sess)
		return 2
	}

	res, err := wipCheckpointScoped(context.Background(), *repo, sess, *buildable, time.Now().Unix(), scope)
	if code, done := emitResultOrError(stdout, stderr, "fak wip checkpoint", *asJSON, res, err); done {
		return code
	}
	if res.Clean {
		fmt.Fprintf(stdout, "clean: nothing to checkpoint for session %s\n", sess)
		return 0
	}
	if res.Superseded {
		fmt.Fprintf(stdout, "superseded: a newer checkpoint already holds session %s -> %s\n",
			sess, shortWipSHA(res.Object))
		return 0
	}
	fmt.Fprintf(stdout, "checkpointed session %s at %s (%d leaves) -> %s\n",
		sess, shortWipSHA(res.StartSHA), len(res.Leaves), shortWipSHA(res.Object))
	// Echo the claim back: a --path that silently recorded nothing would look identical
	// to one that recorded the scope, and the scope is what a later land obeys.
	if len(res.Scope) > 0 {
		fmt.Fprintf(stdout, "  claimed scope (what `fak wip land` will commit): %s\n", strings.Join(res.Scope, ", "))
	}
	return 0
}

// wipAutoCheckpointReasons is the closed set of risky-boundary labels an auto-checkpoint
// may carry (#3877): the compaction step-advice boundary, the guard Stop hook, the doomloop
// NUDGE, or a manual invocation. A capture at any of these is best-effort by construction —
// it must never break the boundary it fires at.
var wipAutoCheckpointReasons = map[string]bool{
	"compaction": true, "stop": true, "doomloop": true, "manual": true,
}

// wipAutoCheckpointResult is the JSON/plain result of an auto-checkpoint. `captured` is the
// one bit that says a NEW ref was written; `skipped` names why it was a no-op otherwise
// (no-session | invalid-session | clean | superseded | capture-error).
type wipAutoCheckpointResult struct {
	Reason   string `json:"reason"`
	Session  string `json:"session,omitempty"`
	Captured bool   `json:"captured"`
	Leaves   int    `json:"leaves,omitempty"`
	Object   string `json:"object,omitempty"`
	Skipped  string `json:"skipped,omitempty"`
	Error    string `json:"error,omitempty"`
}

// runWipAutoCheckpoint is the hook-facing entrypoint (#3877): capture the session's WIP at a
// risky boundary. Unlike `checkpoint`, it is best-effort — a failed capture, a missing
// session, or a clean tree is reported but exits 0, so wiring it into the compaction /
// guard-Stop / doomloop-NUDGE boundaries can never block them. --strict flips a genuine
// capture failure to exit 1 for callers that want to know.
func runWipAutoCheckpoint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip autocheckpoint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	session := fs.String("session", "", "session id (default: $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID)")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	reason := fs.String("reason", "manual", "risky-boundary label: compaction|stop|doomloop|manual")
	strict := fs.Bool("strict", false, "surface a capture failure as exit 1 (default: best-effort exit 0)")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	// The capture walks the whole worktree against a FRESH temp index seeded by `read-tree
	// HEAD`, which carries no stat cache, so `add -A` re-hashes every tracked file. Measured
	// on the reference box (2026-08-05, n=5, 12,227 files / 44.5 MB tracked): mean 1.33s and
	// ~64,757 page faults per checkpoint -- about eight process-creations' worth of fault
	// cost, once per turn per session. That is affordable when the box is healthy and
	// unbounded when it is not, and this runs at a Stop boundary where wedging the agent is
	// the worse failure. The deadline is ~22x the measured mean so an ordinary turn never
	// trips it; a stalled host degrades to a countable skip instead of an open-ended stall.
	timeout := fs.Duration("timeout", 30*time.Second, "give up on the capture after this long (0 disables the deadline)")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}

	rsn := strings.ToLower(strings.TrimSpace(*reason))
	if !wipAutoCheckpointReasons[rsn] {
		fmt.Fprintf(stderr, "fak wip autocheckpoint: unknown --reason %q (want compaction|stop|doomloop|manual)\n", *reason)
		return 2
	}
	out := wipAutoCheckpointResult{Reason: rsn}

	sess := strings.TrimSpace(*session)
	if sess == "" {
		sess = firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
	}
	out.Session = sess
	switch {
	case sess == "":
		out.Skipped = "no-session"
	case !wipref.ValidSession(sess):
		out.Skipped = "invalid-session"
	default:
		ctx := context.Background()
		if *timeout > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, *timeout)
			defer cancel()
		}
		res, err := wipCheckpoint(ctx, *repo, sess, true, time.Now().Unix())
		switch {
		case err != nil && ctx.Err() != nil:
			// Spelled apart from capture-error on purpose: a deadline means the host was too
			// slow to walk the tree, not that the capture is broken. Callers that count skips
			// can see host pressure rather than reading it as a fak defect.
			out.Skipped, out.Error = "capture-timeout", fmt.Sprintf("checkpoint exceeded %s: %v", *timeout, err)
		case err != nil:
			out.Skipped, out.Error = "capture-error", err.Error()
		case res.Clean:
			out.Skipped = "clean"
		case res.Superseded:
			out.Skipped, out.Object = "superseded", res.Object
		default:
			out.Captured, out.Object, out.Leaves = true, res.Object, len(res.Leaves)
		}
	}

	// Best-effort exit policy: only --strict escalates. A capture-error or capture-timeout
	// under --strict is a real failure (exit 1); a missing/invalid session under --strict is a
	// usage error (exit 2). Without --strict every outcome is exit 0 so the boundary is never
	// blocked -- which is the posture the Stop hook relies on.
	if *strict {
		switch out.Skipped {
		case "capture-error", "capture-timeout":
			fmt.Fprintf(stderr, "fak wip autocheckpoint: %s\n", out.Error)
			return 1
		case "no-session", "invalid-session":
			fmt.Fprintf(stderr, "fak wip autocheckpoint: %s\n", out.Skipped)
			return 2
		}
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, out, "fak wip autocheckpoint")
	}
	switch {
	case out.Captured:
		fmt.Fprintf(stdout, "auto-checkpointed [%s] session %s (%d leaves) -> %s\n", rsn, sess, out.Leaves, shortWipSHA(out.Object))
	case out.Skipped == "clean":
		fmt.Fprintf(stdout, "auto-checkpoint [%s]: clean, nothing to capture\n", rsn)
	default:
		fmt.Fprintf(stdout, "auto-checkpoint [%s]: skipped (%s)\n", rsn, out.Skipped)
	}
	return 0
}

// wipCheckpoint captures the tree-wide delta with NO declared scope — the original
// signature, kept for every caller that has no claim to record. See wipCheckpointScoped
// for what a declared scope buys and why it is the LAND side, not the capture side,
// that needs it.
func wipCheckpoint(ctx context.Context, repo, session string, buildable bool, nowUnix int64) (wipCheckpointResult, error) {
	return wipCheckpointScoped(ctx, repo, session, buildable, nowUnix, nil)
}

// wipCheckpointScoped captures the working-tree delta — tracked modifications AND
// untracked non-ignored files (#4336) — into a stamped commit and anchors it at
// refs/fak/wip/<session>. It uses a THROWAWAY index (GIT_INDEX_FILE) seeded from
// HEAD, so `git add -A` stages the delta there without ever touching the real
// index or working tree, and `git write-tree` captures it. `.gitignore` is
// respected (add -A never stages ignored paths), so build artifacts stay out of
// the snapshot. The clean verdict is decided on the tree written AFTER untracked
// staging: a tree identical to HEAD's means a clean tree — reported, no ref
// written — so a pure-untracked WIP is never misreported as clean.
//
// scope does NOT narrow the capture: on a shared working tree the snapshot must stay
// tree-wide or a crashed session's undeclared edits become unrecoverable, which is
// strictly worse than storing too much. What scope narrows is the LAND — it is recorded
// in the stamp (wipref.Stamp.Scope) so a LATER process, a fleet host recovering a session
// that can no longer be asked what it owned, commits only what that session claimed, with
// no flags of its own (#5539). A nil/empty scope declares nothing.
func wipCheckpointScoped(ctx context.Context, repo, session string, buildable bool, nowUnix int64, scope []string) (wipCheckpointResult, error) {
	res := wipCheckpointResult{Session: session, Ref: wipref.SessionRef(session), Buildable: buildable, Leaves: []string{}}
	res.Scope = wipNormalizeScope(scope)

	head, err := gitWipOut(ctx, repo, nil, "rev-parse", "HEAD")
	if err != nil {
		return res, fmt.Errorf("resolve HEAD: %w", err)
	}
	res.StartSHA = head

	tmp, err := os.MkdirTemp("", "fak-wip-idx-")
	if err != nil {
		return res, err
	}
	defer os.RemoveAll(tmp)
	idxEnv := []string{"GIT_INDEX_FILE=" + filepath.Join(tmp, "index")}
	if _, err := gitWipOut(ctx, repo, idxEnv, "read-tree", "HEAD"); err != nil {
		return res, fmt.Errorf("seed temp index: %w", err)
	}
	if _, err := gitWipOut(ctx, repo, idxEnv, "add", "-A"); err != nil {
		return res, fmt.Errorf("stage working-tree changes: %w", err)
	}
	tree, err := gitWipOut(ctx, repo, idxEnv, "write-tree")
	if err != nil {
		return res, fmt.Errorf("write tree: %w", err)
	}

	headTree, err := gitWipOut(ctx, repo, nil, "rev-parse", "HEAD^{tree}")
	if err != nil {
		return res, err
	}
	if tree == headTree {
		res.Clean = true
		return res, nil
	}

	// Tree-based debounce against the session ref: the clean-check above only fires when the
	// working tree matches HEAD, so a persistently-dirty tree checkpointed twice (unchanged
	// between the two runs) writes the SAME tree each time. Minting a fresh commit for that
	// unchanged tree stamps a newer CheckpointedAt that wins wipAnchorCAS and churns the ref;
	// worse, the commit hash is wall-clock-second sensitive (author/committer date + the
	// CheckpointedAt stamp), so an unchanged tree would non-deterministically mint a duplicate
	// ref whenever the two runs straddle a second boundary. If the current checkpoint already
	// captured this exact tree, keep it — an unchanged tree never needs a new checkpoint.
	// The one thing that reopens the debounce is a CHANGED CLAIM: the stamped scope is
	// what a later land reads, so re-declaring it over an unchanged tree must actually
	// rewrite the stamp rather than be absorbed as "nothing changed" (#5539).
	if curOID, has, cerr := wipCurrentOID(ctx, repo, res.Ref); cerr == nil && has {
		if curTree, terr := gitWipOut(ctx, repo, nil, "rev-parse", curOID+"^{tree}"); terr == nil && curTree == tree {
			cur, rerr := wipRecordAt(ctx, repo, res.Ref, curOID)
			if rerr == nil && wipSameScope(cur.Stamp.Scope, res.Scope) {
				res.Object, res.Superseded = curOID, true
				return res, nil
			}
		}
	}

	names, err := gitWipOut(ctx, repo, nil, "diff", "--name-only", head, tree)
	if err != nil {
		return res, err
	}
	res.Leaves = wipLeavesFromNames(names)
	host := wipLocalHost(repo)
	deltaBytes := wipDeltaBytes(ctx, repo, head, tree)

	msg, err := wipref.EncodeStamp(wipref.Stamp{
		SessionID:      session,
		StartSHA:       head,
		Leaves:         res.Leaves,
		Scope:          res.Scope,
		Buildable:      buildable,
		CheckpointedAt: nowUnix,
		Host:           host,
		DeltaBytes:     deltaBytes,
	})
	if err != nil {
		return res, err
	}
	commit, err := gitWipOut(ctx, repo, nil, "commit-tree", tree, "-p", head, "-m", msg)
	if err != nil {
		return res, fmt.Errorf("mint checkpoint commit: %w", err)
	}
	cand := wipref.RefRecord{
		Ref:    wipref.SessionRef(session),
		Object: commit,
		Stamp: wipref.Stamp{
			SessionID:      session,
			StartSHA:       head,
			Leaves:         res.Leaves,
			Scope:          res.Scope,
			Buildable:      buildable,
			CheckpointedAt: nowUnix,
			Host:           host,
			DeltaBytes:     deltaBytes,
		},
	}
	if err := wipCheckpointFaultAt("before-ref-update"); err != nil {
		return res, err
	}
	object, superseded, err := wipAnchorCAS(ctx, repo, cand.Ref, cand)
	if err != nil {
		return res, err
	}
	res.Object, res.Superseded = object, superseded
	if err := wipCheckpointFaultAt("after-ref-update"); err != nil {
		return res, err
	}
	return res, nil
}

// wipAnchorCAS points ref at cand.Object under a last-writer-wins compare-and-swap,
// mirroring leaseref's fence CAS (internal/leaseref/fence.go): read the ref's current
// object, let wipref.Reconcile decide the winner, then `git update-ref ref new old` —
// which fails if the ref advanced under us — and retry on a lost CAS. It never holds a
// lock across a git subprocess. Two guard processes checkpointing the same session
// thus converge to one valid ref (the last writer, by CheckpointedAt), never a torn
// one. Returns the object the ref converged to and whether cand was SUPERSEDED (an
// equal-or-newer checkpoint already held the ref, so cand did not land — not an error,
// the later writer simply won).
func wipAnchorCAS(ctx context.Context, repo, ref string, cand wipref.RefRecord) (object string, superseded bool, err error) {
	const maxAttempts = 16
	for attempt := 0; attempt < maxAttempts; attempt++ {
		oldOID, hadRef, err := wipCurrentOID(ctx, repo, ref)
		if err != nil {
			return "", false, err
		}
		var cur wipref.RefRecord
		if hadRef {
			if cur, err = wipRecordAt(ctx, repo, ref, oldOID); err != nil {
				return "", false, err
			}
		}
		if _, changed := wipref.Reconcile(cur, cand); !changed {
			return oldOID, true, nil // a newer/equal checkpoint already holds the ref
		}
		ok, err := wipCasUpdateRef(ctx, repo, ref, cand.Object, oldOID, hadRef)
		if err != nil {
			return "", false, err
		}
		if ok {
			return cand.Object, false, nil
		}
		// CAS lost: the ref moved between our read and our write — re-read and re-decide.
	}
	return "", false, fmt.Errorf("update ref %s: compare-and-swap contended after %d attempts", ref, maxAttempts)
}

// wipCurrentOID resolves ref to its current object id and existence — the CAS
// old-value read, mirroring leaseref.currentOID. A non-executable git is the only
// hard error; a missing ref is (‑, false, nil).
func wipCurrentOID(ctx context.Context, repo, ref string) (oid string, has bool, err error) {
	out, _, code, err := gitWip(ctx, repo, nil, "rev-parse", "--verify", "--quiet", ref)
	if err != nil {
		return "", false, fmt.Errorf("resolve %s: %w", ref, err)
	}
	if code != 0 {
		return "", false, nil
	}
	return strings.TrimSpace(out), true, nil
}

// wipRecordAt reads the stamp of the checkpoint commit ref currently points at, so
// Reconcile can order the incumbent against the candidate. A missing/unparseable
// stamp yields a zero Stamp (CheckpointedAt 0), which loses to any real candidate.
func wipRecordAt(ctx context.Context, repo, ref, oid string) (wipref.RefRecord, error) {
	msg, err := gitWipOut(ctx, repo, nil, "log", "-1", "--format=%B", oid)
	if err != nil {
		return wipref.RefRecord{}, err
	}
	stamp, _ := wipref.DecodeStamp(msg)
	return wipref.RefRecord{Ref: ref, Object: oid, Stamp: stamp}, nil
}

// wipCasUpdateRef performs the OLD-VALUE compare-and-swap ref write, with a reflog
// entry so a just-superseded checkpoint object stays gc-reachable through the reflog
// expiry window (not only while the ref points at it). A create uses git's zero-OID
// "must not exist" sentinel, so even the first anchor fails closed if a peer created
// the ref first. ok=false is a lost CAS (a value), matching leaseref; git could not
// be executed is the only hard error.
func wipCasUpdateRef(ctx context.Context, repo, ref, newOID, oldOID string, hadRef bool) (bool, error) {
	old := oldOID
	if !hadRef {
		z, err := wipZeroOID(ctx, repo)
		if err != nil {
			return false, err
		}
		old = z
	}
	_, _, code, err := gitWip(ctx, repo, nil, "update-ref", "--create-reflog", ref, newOID, old)
	if err != nil {
		return false, fmt.Errorf("update-ref %s: %w", ref, err)
	}
	return code == 0, nil // non-zero: CAS lost (ref advanced, or a create raced)
}

// wipZeroOID returns git's all-zeros object id for this repo's hash algorithm — the
// update-ref old-value sentinel meaning "the ref must not currently exist". Mirrors
// leaseref.zeroOID: 64 zeros under sha256, else 40.
func wipZeroOID(ctx context.Context, repo string) (string, error) {
	out, _, code, err := gitWip(ctx, repo, nil, "rev-parse", "--show-object-format")
	if err != nil {
		return "", fmt.Errorf("probe object format: %w", err)
	}
	if code == 0 && strings.TrimSpace(out) == "sha256" {
		return strings.Repeat("0", 64), nil
	}
	return strings.Repeat("0", 40), nil
}

// wipLeavesFromNames folds a `git diff --name-only` listing into the sorted unique
// set of parent directories — a coarse per-file lane for the stamp. (Real lane
// attribution is a later cut; the spine just records where the delta landed.)
func wipLeavesFromNames(names string) []string {
	set := map[string]bool{}
	for _, ln := range strings.Split(names, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		set[path.Dir(ln)] = true
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ---- status ----

func runWipStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	remote := fs.String("remote", "origin", "grade replication against this remote's mirror (a local ref read, never a network probe)")
	asJSON := fs.Bool("json", false, "emit the status report as JSON")
	fleet := fs.Bool("fleet", false, "enumerate checkpoint refs mirrored from peer hosts")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	if *fleet {
		report, err := wipStatusFor(context.Background(), *repo, *remote, time.Now().Unix())
		if err != nil {
			fmt.Fprintf(stderr, "fak wip status --fleet: %v\n", err)
			return 1
		}
		fleetReport, err := wipFleetStatus(context.Background(), *repo, *remote, time.Now().Unix())
		if err != nil {
			fmt.Fprintf(stderr, "fak wip status --fleet: %v\n", err)
			return 1
		}
		report.Fleet = &fleetReport
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, report, "fak wip status --fleet")
		}
		if report.Count == 0 {
			fmt.Fprintln(stdout, "no working-tree checkpoints")
		}
		for _, session := range report.Sessions {
			fmt.Fprintf(stdout, "%s\t%s\t%d leaves\tage=%ds\tbuildable=%v\t%s\n", session.Session, shortWipSHA(session.StartSHA), len(session.Leaves), session.AgeSeconds, session.Buildable, session.Replication)
		}
		wipFleetRender(stdout, fleetReport)
		if report.Mirror != nil {
			fmt.Fprintln(stdout, wipMirrorLine(*report.Mirror))
		}
		return 0
	}

	report, err := wipStatusFor(context.Background(), *repo, *remote, time.Now().Unix())
	if err != nil {
		fmt.Fprintf(stderr, "fak wip status: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak wip status")
	}
	if report.Count == 0 {
		fmt.Fprintln(stdout, "no working-tree checkpoints")
		return 0
	}
	for _, s := range report.Sessions {
		fmt.Fprintf(stdout, "%s\t%s\t%d leaves\tage=%ds\tbuildable=%v\t%s\n",
			s.Session, shortWipSHA(s.StartSHA), len(s.Leaves), s.AgeSeconds, s.Buildable, s.Replication)
	}
	fmt.Fprintln(stdout, wipReplicationSummary(report.Replicated, report.StaleRemote, report.LocalOnly))
	if report.Mirror != nil {
		fmt.Fprintln(stdout, wipMirrorLine(*report.Mirror))
	}
	return 0
}

// wipStatus reads every live checkpoint ref and hands the records to the pure fold,
// computing each checkpoint's age against nowUnix. Kept as the no-remote entry point
// for callers with no replication question to ask; it grades every row against the
// default remote's mirror, exactly as `fak wip status` does with no --remote.
func wipStatus(ctx context.Context, repo string, nowUnix int64) (wipref.StatusReport, error) {
	return wipStatusFor(ctx, repo, "origin", nowUnix)
}

// wipStatusFor is wipStatus plus the replication verdict: it reads the live checkpoint
// refs AND this clone's mirror of remote, then folds the two together. Both reads are
// local — see wipMirrorIndex for why status must not touch the network.
//
// It also attaches the mirror's PROVENANCE (#5556). The three replication counts are only
// as good as the mirror they were graded against, and a mirror nobody has refreshed
// grades everything LOCAL_ONLY — which is the safe answer for durability but says nothing
// about WHY. The attached view carries the last sync's time and direction so a reader can
// tell "the remote does not have it" from "this clone has not looked".
func wipStatusFor(ctx context.Context, repo, remote string, nowUnix int64) (wipref.StatusReport, error) {
	recs, err := wipListRecords(ctx, repo)
	if err != nil {
		return wipref.StatusReport{}, err
	}
	mirror, err := wipMirrorIndex(ctx, repo, remote)
	if err != nil {
		return wipref.StatusReport{}, err
	}
	rep := wipref.FoldWithMirror(recs, mirror, nowUnix)
	view, err := wipMirrorView(ctx, repo, remote, len(mirror), nowUnix, 0)
	if err != nil {
		return wipref.StatusReport{}, err
	}
	rep.Mirror = &view
	return rep, nil
}

// wipListRecords reads every live checkpoint ref and decodes its stamp from the
// commit message — the shared raw listing behind the status fold, the reap fold,
// reconcile, and attribute. It pulls the ref name, the object, AND the stamp-bearing
// commit message in a SINGLE `git for-each-ref` (%(contents)) instead of a `git log`
// per ref: the local checkpoint namespace routinely holds thousands of refs, and a
// subprocess-per-ref fan-out made status/reconcile/attribute O(refs) in git spawns —
// slow enough to time out (>2m at ~4k refs), which is what left the reconciliation
// spine effectively unrunnable. Fields are NUL-separated and every record ends with a
// NUL, so a multi-line message survives intact — a commit object can never contain a
// NUL — and splitting the whole stream on NUL yields [refname, objectname, contents]
// triples. A ref whose stamp is missing or unparseable is still listed, labelled from
// its ref name (so nothing silently vanishes from a maintenance pass).
func wipListRecords(ctx context.Context, repo string) ([]wipref.RefRecord, error) {
	pattern := strings.TrimSuffix(wipref.RefNamespace, "/")
	out, errStr, code, err := gitWip(ctx, repo, nil,
		"for-each-ref", "--format=%(refname)%00%(objectname)%00%(contents)%00", pattern)
	if err != nil {
		return nil, fmt.Errorf("git for-each-ref: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("git for-each-ref exited %d: %s", code, strings.TrimSpace(errStr))
	}
	fields := strings.Split(out, "\x00")
	var recs []wipref.RefRecord
	// Consume [refname, objectname, contents] triples. for-each-ref appends a newline
	// after each record's trailing NUL, so every refname after the first carries a
	// leading '\n' — TrimSpace clears it. The short tail after the final NUL is ignored.
	for i := 0; i+2 < len(fields); i += 3 {
		ref := strings.TrimSpace(fields[i])
		obj := strings.TrimSpace(fields[i+1])
		if ref == "" || obj == "" {
			continue
		}
		stamp, ok := wipref.DecodeStamp(fields[i+2])
		if !ok {
			stamp = wipref.Stamp{SessionID: wipref.SessionFromRef(ref)}
		}
		recs = append(recs, wipref.RefRecord{Ref: ref, Object: obj, Stamp: stamp})
	}
	return recs, nil
}

// ---- reap (#3873) ----

// wipReapResult is the JSON/plain result of a reap pass: the refs deleted (or, in a
// dry run, that would be) and the refs kept, each with the fold's reason.
type wipReapResult struct {
	Reaped []wipref.ReapVerdict `json:"reaped"`
	Kept   []wipref.ReapVerdict `json:"kept"`
	DryRun bool                 `json:"dry_run"`
}

func runWipReap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	asJSON := fs.Bool("json", false, "emit the reap result as JSON")
	dryRun := fs.Bool("dry-run", false, "report what would be reaped without deleting any ref")
	census := fs.Bool("census", false, "read-only: classify every checkpoint ref by owner-state and print the counts (deletes NOTHING; #5340)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	// --census is a SEPARATE, read-only reporting path: it classifies every ref and
	// never touches the delete path below (no update-ref -d). It intentionally ignores
	// --dry-run — the census has no mutate mode to preview.
	if *census {
		return runWipCensus(context.Background(), stdout, stderr, *repo, *asJSON)
	}
	res, err := wipReap(context.Background(), *repo, *dryRun)
	if code, done := emitResultOrError(stdout, stderr, "fak wip reap", *asJSON, res, err); done {
		return code
	}
	if len(res.Reaped) == 0 {
		fmt.Fprintln(stdout, "no checkpoints to reap")
		return 0
	}
	verb := "reaped"
	if res.DryRun {
		verb = "would reap"
	}
	for _, v := range res.Reaped {
		fmt.Fprintf(stdout, "%s %s (%s) -> %s\n", verb, v.Session, v.Reason, shortWipSHA(v.Object))
	}
	return 0
}

// wipReap resolves every live checkpoint's owner state (LANDED when its delta is
// already present in HEAD, else UNKNOWN — the fail-safe keep the spine can prove
// without a session registry), folds them through the pure wipref.Reap, and deletes
// each DELETE verdict's ref under an OLD-VALUE compare-and-swap so a ref that a
// concurrent checkpoint advanced since the listing is left intact, not reaped out
// from under it. With dryRun it computes verdicts but issues no deletes.
func wipReap(ctx context.Context, repo string, dryRun bool) (wipReapResult, error) {
	recs, err := wipListRecords(ctx, repo)
	if err != nil {
		return wipReapResult{}, err
	}
	owners := make(map[string]wipref.OwnerState, len(recs))
	for _, r := range recs {
		st, err := wipOwnerState(ctx, repo, r)
		if err != nil {
			return wipReapResult{}, err
		}
		owners[wipSessionOf(r)] = st
	}
	res := wipReapResult{DryRun: dryRun, Reaped: []wipref.ReapVerdict{}, Kept: []wipref.ReapVerdict{}}
	for _, v := range wipref.Reap(recs, owners) {
		if v.Action != wipref.ReapDelete {
			res.Kept = append(res.Kept, v)
			continue
		}
		if dryRun {
			res.Reaped = append(res.Reaped, v)
			continue
		}
		receipt, err := wiplifecycle.Begin(repo, "checkpoint-reap", "", time.Now())
		if err != nil {
			return wipReapResult{}, fmt.Errorf("begin checkpoint reap receipt for %s: %w", v.Session, err)
		}
		deleted, err := wipDeleteRef(ctx, repo, v.Ref, v.Object)
		if err != nil {
			return wipReapResult{}, err
		}
		if deleted {
			if _, err := wiplifecycle.Finish(repo, receipt.OperationID, time.Now()); err != nil {
				return wipReapResult{}, fmt.Errorf("finish checkpoint reap receipt for %s: %w", v.Session, err)
			}
			res.Reaped = append(res.Reaped, v)
		} else {
			if _, err := wiplifecycle.Finish(repo, receipt.OperationID, time.Now()); err != nil {
				return wipReapResult{}, fmt.Errorf("finish skipped checkpoint reap receipt for %s: %w", v.Session, err)
			}
			res.Kept = append(res.Kept, v) // ref advanced under us: a concurrent checkpoint won
		}
	}
	return res, nil
}

// wipOwnerState returns the ONLY owner state the spine can positively prove from git
// alone: OwnerLanded when HEAD's version of exactly the files the checkpoint changed
// already equals the checkpoint's (the owner committed the delta), else OwnerUnknown
// — which the fold keeps. It never reports a delete-eligible state on uncertainty, so
// reap cannot destroy an unlanded snapshot. (LIVE / CLOSED_* are resolved by later
// cuts with a session registry; the pure fold already handles the full vocabulary.)
func wipOwnerState(ctx context.Context, repo string, rec wipref.RefRecord) (wipref.OwnerState, error) {
	names, err := gitWipOut(ctx, repo, nil, "diff", "--name-only", rec.Object+"^", rec.Object)
	if err != nil {
		return wipref.OwnerUnknown, nil // no resolvable parent/delta: fail-safe keep
	}
	var files []string
	for _, n := range strings.Split(names, "\n") {
		if n = strings.TrimSpace(n); n != "" {
			files = append(files, n)
		}
	}
	if len(files) == 0 {
		return wipref.OwnerUnknown, nil
	}
	args := append([]string{"diff", "--quiet", "HEAD", rec.Object, "--"}, files...)
	_, _, code, err := gitWip(ctx, repo, nil, args...)
	if err != nil {
		return wipref.OwnerUnknown, nil
	}
	if code == 0 {
		return wipref.OwnerLanded, nil // HEAD already carries exactly this delta
	}
	return wipref.OwnerUnknown, nil // unlanded (or diverged): keep, fail-safe
}

// wipDeleteRef removes ref under an OLD-VALUE compare-and-swap (`update-ref -d ref
// oldOID`), so a ref advanced by a concurrent checkpoint since the reap listing is
// left alone (deleted=false) rather than removed. Once the ref is gone the object
// becomes gc-eligible (its reflog window aside) — the retention edge ends here. A
// non-executable git is the only hard error.
func wipDeleteRef(ctx context.Context, repo, ref, oldOID string) (bool, error) {
	_, _, code, err := gitWip(ctx, repo, nil, "update-ref", "-d", ref, oldOID)
	if err != nil {
		return false, fmt.Errorf("delete %s: %w", ref, err)
	}
	return code == 0, nil
}

// wipSessionOf recovers a record's session id from its stamp, falling back to the
// ref name — the same identity rule the pure fold uses.
func wipSessionOf(rec wipref.RefRecord) string {
	if rec.Stamp.SessionID != "" {
		return rec.Stamp.SessionID
	}
	return wipref.SessionFromRef(rec.Ref)
}

// ---- attribute (#3874) ----

// wipAttributeResult is the JSON/plain result of an attribution pass: every dirty
// hunk's verdict plus the count that are ORPHAN (the at-risk set).
type wipAttributeResult struct {
	Attributions []wipattr.Attribution `json:"attributions"`
	Orphans      int                   `json:"orphans"`
}

// runWipAttribute exits 3 (in --orphans mode) when any dirty hunk is unattributed —
// the one-bit signal a sweep-guard (#3879) or CI gate keys on. Plain/JSON listing
// without --orphans is informational and exits 0.
func runWipAttribute(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip attribute", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	asJSON := fs.Bool("json", false, "emit the attribution result as JSON")
	orphansOnly := fs.Bool("orphans", false, "print only ORPHAN hunks; exit 3 if any exist")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	res, err := wipAttribute(context.Background(), *repo)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip attribute: %v\n", err)
		return 1
	}
	rows := res.Attributions
	if *orphansOnly {
		rows = wipattr.Orphans(rows)
	}
	switch {
	case *asJSON:
		if code := encodeJSONOrFail(stdout, stderr,
			wipAttributeResult{Attributions: rows, Orphans: res.Orphans}, "fak wip attribute"); code != 0 {
			return code
		}
	case len(rows) == 0 && *orphansOnly:
		fmt.Fprintln(stdout, "no orphan hunks: every dirty hunk is attributed")
	case len(rows) == 0:
		fmt.Fprintln(stdout, "no dirty hunks to attribute")
	default:
		for _, a := range rows {
			fmt.Fprintf(stdout, "%s\t%s\t%s\n", a.State, a.File, wipAttrOwnerLabel(a))
		}
	}
	if *orphansOnly && res.Orphans > 0 {
		return 3
	}
	return 0
}

// wipAttribute reads the live tracked working-tree delta (`git diff HEAD`) and every
// session's checkpoint delta (`git diff <obj>^ <obj>`), parses both to hunks, and
// folds them through the pure wipattr.Attribute — classifying every dirty hunk as
// OWNED / SHARED / ORPHAN. All git parsing lives here; the classification is pure.
func wipAttribute(ctx context.Context, repo string) (wipAttributeResult, error) {
	attrs, err := wipBuildAttributions(ctx, repo)
	if err != nil {
		return wipAttributeResult{}, err
	}
	return wipAttributeResult{Attributions: attrs, Orphans: len(wipattr.Orphans(attrs))}, nil
}

// wipBuildAttributions computes the per-hunk attribution of the current working-tree
// delta against every session's checkpoint hunks — the shared input for `wip attribute`
// and `wip sweep-guard`.
func wipBuildAttributions(ctx context.Context, repo string) ([]wipattr.Attribution, error) {
	liveDiff, err := gitWipOut(ctx, repo, nil, "diff", "HEAD", "--")
	if err != nil {
		return nil, fmt.Errorf("read working-tree diff: %w", err)
	}
	dirty := wipattr.ParseHunks(liveDiff)
	recs, err := wipListRecords(ctx, repo)
	if err != nil {
		return nil, err
	}
	checkpoints := make(map[string][]wipattr.Hunk, len(recs))
	for _, r := range recs {
		cpDiff, derr := gitWipOut(ctx, repo, nil, "diff", r.Object+"^", r.Object, "--")
		if derr != nil {
			continue // a root-parent or unreadable checkpoint contributes no ownership
		}
		checkpoints[wipSessionOf(r)] = wipattr.ParseHunks(cpDiff)
	}
	return wipattr.Attribute(dirty, checkpoints), nil
}

// wipAttrOwnerLabel renders the owner column for the plain listing.
func wipAttrOwnerLabel(a wipattr.Attribution) string {
	switch a.State {
	case wipattr.AttrOwned:
		return a.Owner
	case wipattr.AttrShared:
		return strings.Join(a.Owners, ",")
	default:
		return "-"
	}
}

// ---- sweep-guard (#3879) ----

// wipSweepResult is the JSON/plain result of a sweep-guard pass.
type wipSweepResult struct {
	Self     string                 `json:"self"`
	Verdicts []wipattr.SweepVerdict `json:"verdicts"`
	Hazards  int                    `json:"hazards"`
}

// runWipSweepGuard warns before a broad `git add` would sweep a peer's WIP. It exits 3
// when any dirty hunk is a HAZARD (owned by a peer, SHARED, or ORPHAN), 0 when every
// hunk is owned by self (or the tree is clean). Advisory: it inspects and warns, it
// never stages anything.
func runWipSweepGuard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip sweep-guard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	session := fs.String("session", "", "the self session id (default: $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID)")
	asJSON := fs.Bool("json", false, "emit every verdict as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	self := *session
	if self == "" {
		self = firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
	}
	res, err := wipSweepGuard(context.Background(), *repo, self)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip sweep-guard: %v\n", err)
		return 1
	}
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, res, "fak wip sweep-guard"); code != 0 {
			return code
		}
		if res.Hazards > 0 {
			return 3
		}
		return 0
	}
	if res.Hazards == 0 {
		fmt.Fprintln(stdout, "sweep-guard: clean — every dirty hunk is owned by self (or the tree is clean)")
		return 0
	}
	fmt.Fprintf(stderr, "sweep-guard: %d hazard hunk(s) — a broad `git add` would sweep WIP that is not yours:\n", res.Hazards)
	for _, v := range res.Verdicts {
		if v.Risk == wipattr.SweepHazard {
			fmt.Fprintf(stderr, "  %s\t%s\t%s\n", v.File, v.State, v.Reason)
		}
	}
	fmt.Fprintln(stderr, "checkpoint your own work, then stage explicit paths (git add <path>) instead of -A.")
	return 3
}

// wipSweepGuard attributes the working-tree delta, then grades each hunk SAFE/HAZARD
// against the self session and the live-lease set (peers still holding a lock under
// refs/fak/locks/*). Read-only: it stages nothing.
func wipSweepGuard(ctx context.Context, repo, self string) (wipSweepResult, error) {
	attrs, err := wipBuildAttributions(ctx, repo)
	if err != nil {
		return wipSweepResult{}, err
	}
	live, err := wipLiveSessions(ctx, repo)
	if err != nil {
		return wipSweepResult{}, err
	}
	verdicts := wipattr.SweepGuard(attrs, self, live)
	return wipSweepResult{Self: self, Verdicts: verdicts, Hazards: len(wipattr.SweepHazards(verdicts))}, nil
}

// ---- restore ----

func runWipRestore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip restore", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	apply := fs.Bool("apply", false, "apply the checkpointed delta to the working tree (default: print the diff)")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}
	rest := fs.Args()
	if len(rest) != 1 {
		fmt.Fprintln(stderr, "fak wip restore: exactly one <session> argument is required")
		return 2
	}
	sess := strings.TrimSpace(rest[0])
	if !wipref.ValidSession(sess) {
		fmt.Fprintf(stderr, "fak wip restore: invalid session id %q\n", sess)
		return 2
	}
	code, err := wipRestore(context.Background(), *repo, sess, *apply, stdout)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip restore: %v\n", err)
	}
	return code
}

// wipRestore resolves the checkpoint commit and materializes its delta as the
// patch `git diff <commit>^1 <commit>` — the change captured against the HEAD the
// checkpoint was taken from. With apply it feeds that patch to `git apply` onto the
// working tree; otherwise it prints it. Returns (exitCode, err).
func wipRestore(ctx context.Context, repo, session string, apply bool, stdout io.Writer) (int, error) {
	ref := wipref.SessionRef(session)
	commit, _, code, err := gitWip(ctx, repo, nil, "rev-parse", "--verify", "--quiet", ref)
	if err != nil {
		return 1, fmt.Errorf("git rev-parse: %w", err)
	}
	commit = strings.TrimSpace(commit)
	if code != 0 || commit == "" {
		return 1, fmt.Errorf("no checkpoint for session %q", session)
	}

	patch, errStr, dcode, err := gitWip(ctx, repo, nil, "diff", commit+"^1", commit)
	if err != nil {
		return 1, fmt.Errorf("git diff: %w", err)
	}
	if dcode != 0 {
		return 1, fmt.Errorf("git diff exited %d: %s", dcode, strings.TrimSpace(errStr))
	}
	if !apply {
		fmt.Fprint(stdout, patch)
		if patch != "" && !strings.HasSuffix(patch, "\n") {
			fmt.Fprintln(stdout)
		}
		return 0, nil
	}
	if strings.TrimSpace(patch) == "" {
		return 0, nil // empty delta -> nothing to re-materialize
	}
	if err := wipApplyPatch(ctx, repo, patch); err != nil {
		return 1, err
	}
	return 0, nil
}

// wipApplyPatch pipes a unified diff to `git apply` (working tree only, never the
// index or a commit). --whitespace=nowarn keeps a benign trailing-newline delta
// from being rejected.
func wipApplyPatch(ctx context.Context, repo, patch string) error {
	cmd := exec.CommandContext(ctx, "git", "apply", "--whitespace=nowarn")
	configureDispatchHelperCommand(cmd)
	if repo != "" {
		cmd.Dir = repo
	}
	cmd.Stdin = strings.NewReader(patch)
	var errb strings.Builder
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("git apply: %v: %s", err, strings.TrimSpace(errb.String()))
	}
	return nil
}

// wipPlumbBaseCommit mints a commit from the current index with plumbing only
// (write-tree + commit-tree, no parent), returning its sha. It lets a throwaway
// repo get a base commit without porcelain `git commit`, which would consult — and
// might demand — the caller's commit-signing config. Used by the selfcheck.
func wipPlumbBaseCommit(ctx context.Context, dir, msg string) (string, error) {
	tree, err := gitWipOut(ctx, dir, nil, "write-tree")
	if err != nil {
		return "", err
	}
	return gitWipOut(ctx, dir, nil, "commit-tree", tree, "-m", msg)
}
