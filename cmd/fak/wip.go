package main

// wip.go — `fak wip`, the working-tree checkpoint/restore spine (#3872). It gives
// a session a durable, gc-safe snapshot of its uncommitted tracked changes under
// refs/fak/wip/<session> (a sibling of the lease refs, refs/fak/locks/*) WITHOUT
// touching the index, the working tree, or any branch/HEAD.
//
//	fak wip checkpoint [--session <id>] [-C <repo>]   # snapshot the tracked delta
//	fak wip status [-C <repo>] [--json]               # list the live checkpoints
//	fak wip restore <session> [-C <repo>] [--apply]   # re-materialize the delta
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
	case "status":
		return runWipStatus(stdout, stderr, argv[1:])
	case "restore":
		return runWipRestore(stdout, stderr, argv[1:])
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

  fak wip checkpoint [--session <id>] [-C <repo>] [--buildable=<bool>] [--json]
      Snapshot the current tracked working-tree delta into a gc-safe object under
      refs/fak/wip/<session>, WITHOUT touching the index, working tree, or a branch.
      Session defaults to $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID.

  fak wip status [-C <repo>] [--json]
      List the live working-tree checkpoints (one per session), sorted by session.

  fak wip restore <session> [-C <repo>] [--apply]
      Print the checkpointed delta as an apply-able diff (default) or, with --apply,
      re-materialize it onto the current working tree.

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
	cmd := exec.CommandContext(ctx, "git", args...)
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

// wipCheckpointResult is the JSON/plain result of a checkpoint.
type wipCheckpointResult struct {
	Session   string   `json:"session"`
	Ref       string   `json:"ref"`
	Object    string   `json:"object,omitempty"`
	StartSHA  string   `json:"start_sha"`
	Leaves    []string `json:"leaves"`
	Buildable bool     `json:"buildable"`
	Clean     bool     `json:"clean"`
}

func runWipCheckpoint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip checkpoint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	session := fs.String("session", "", "session id to checkpoint under (default: $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID)")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	buildable := fs.Bool("buildable", true, "record the checkpoint as buildable (advisory stamp field)")
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

	res, err := wipCheckpoint(context.Background(), *repo, sess, *buildable, time.Now().Unix())
	if err != nil {
		fmt.Fprintf(stderr, "fak wip checkpoint: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, res, "fak wip checkpoint")
	}
	if res.Clean {
		fmt.Fprintf(stdout, "clean: nothing to checkpoint for session %s\n", sess)
		return 0
	}
	fmt.Fprintf(stdout, "checkpointed session %s at %s (%d leaves) -> %s\n",
		sess, shortWipSHA(res.StartSHA), len(res.Leaves), shortWipSHA(res.Object))
	return 0
}

// wipCheckpoint captures the tracked working-tree delta into a stamped commit and
// anchors it at refs/fak/wip/<session>. It uses a THROWAWAY index (GIT_INDEX_FILE)
// seeded from HEAD, so `git add -u` stages tracked modifications there without ever
// touching the real index or working tree, and `git write-tree` captures them. A
// tree identical to HEAD's means a clean tree — reported, no ref written.
func wipCheckpoint(ctx context.Context, repo, session string, buildable bool, nowUnix int64) (wipCheckpointResult, error) {
	res := wipCheckpointResult{Session: session, Ref: wipref.SessionRef(session), Buildable: buildable, Leaves: []string{}}

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
	if _, err := gitWipOut(ctx, repo, idxEnv, "add", "-u"); err != nil {
		return res, fmt.Errorf("stage tracked changes: %w", err)
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

	names, err := gitWipOut(ctx, repo, nil, "diff", "--name-only", head, tree)
	if err != nil {
		return res, err
	}
	res.Leaves = wipLeavesFromNames(names)

	msg, err := wipref.EncodeStamp(wipref.Stamp{
		SessionID:      session,
		StartSHA:       head,
		Leaves:         res.Leaves,
		Buildable:      buildable,
		CheckpointedAt: nowUnix,
	})
	if err != nil {
		return res, err
	}
	commit, err := gitWipOut(ctx, repo, nil, "commit-tree", tree, "-p", head, "-m", msg)
	if err != nil {
		return res, fmt.Errorf("mint checkpoint commit: %w", err)
	}
	if _, err := gitWipOut(ctx, repo, nil, "update-ref", wipref.SessionRef(session), commit); err != nil {
		return res, fmt.Errorf("update ref: %w", err)
	}
	res.Object = commit
	return res, nil
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
	asJSON := fs.Bool("json", false, "emit the status report as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	report, err := wipStatus(context.Background(), *repo, time.Now().Unix())
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
		fmt.Fprintf(stdout, "%s\t%s\t%d leaves\tage=%ds\tbuildable=%v\n",
			s.Session, shortWipSHA(s.StartSHA), len(s.Leaves), s.AgeSeconds, s.Buildable)
	}
	return 0
}

// wipStatus reads every live checkpoint ref, decodes its stamp from the commit
// message, and hands the records to the pure fold. A ref whose stamp is missing or
// unparseable still lists, labelled from its ref name.
func wipStatus(ctx context.Context, repo string, nowUnix int64) (wipref.StatusReport, error) {
	pattern := strings.TrimSuffix(wipref.RefNamespace, "/")
	out, errStr, code, err := gitWip(ctx, repo, nil, "for-each-ref", "--format=%(refname) %(objectname)", pattern)
	if err != nil {
		return wipref.StatusReport{}, fmt.Errorf("git for-each-ref: %w", err)
	}
	if code != 0 {
		return wipref.StatusReport{}, fmt.Errorf("git for-each-ref exited %d: %s", code, strings.TrimSpace(errStr))
	}
	var recs []wipref.RefRecord
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		fields := strings.Fields(ln)
		if len(fields) < 2 {
			continue
		}
		ref, obj := fields[0], fields[1]
		msg, merr := gitWipOut(ctx, repo, nil, "log", "-1", "--format=%B", obj)
		stamp, ok := wipref.DecodeStamp(msg)
		if merr != nil || !ok {
			stamp = wipref.Stamp{SessionID: wipref.SessionFromRef(ref)}
		}
		recs = append(recs, wipref.RefRecord{Ref: ref, Object: obj, Stamp: stamp})
	}
	return wipref.Fold(recs, nowUnix), nil
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

// ---- selfcheck ----

func runWipSelfcheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip selfcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	asJSON := fs.Bool("json", false, "emit the selfcheck verdict as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	ctx := context.Background()
	fail := func(msg string) int { return wipSelfcheckVerdict(stdout, stderr, *asJSON, false, msg) }

	dir, err := os.MkdirTemp("", "fak-wip-selfcheck-")
	if err != nil {
		fmt.Fprintf(stderr, "fak wip selfcheck: %v\n", err)
		return 1
	}
	defer os.RemoveAll(dir)

	// A throwaway repo with one committed tracked file — the base state. The base
	// commit is minted with plumbing (write-tree + commit-tree + update-ref HEAD),
	// never porcelain `git commit`: commit-tree never auto-signs, so the selfcheck
	// neither depends on nor disables the caller's commit-signing config.
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "wip@selfcheck.local"},
		{"config", "user.name", "wip selfcheck"},
	} {
		if _, err := gitWipOut(ctx, dir, nil, args...); err != nil {
			fmt.Fprintf(stderr, "fak wip selfcheck: %v\n", err)
			return 1
		}
	}
	file := filepath.Join(dir, "note.txt")
	base := []byte("committed base line\n")
	if err := os.WriteFile(file, base, 0o644); err != nil {
		fmt.Fprintf(stderr, "fak wip selfcheck: %v\n", err)
		return 1
	}
	if _, err := gitWipOut(ctx, dir, nil, "add", "note.txt"); err != nil {
		fmt.Fprintf(stderr, "fak wip selfcheck: %v\n", err)
		return 1
	}
	baseCommit, err := wipPlumbBaseCommit(ctx, dir, "base")
	if err != nil {
		fmt.Fprintf(stderr, "fak wip selfcheck: %v\n", err)
		return 1
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "HEAD", baseCommit); err != nil {
		fmt.Fprintf(stderr, "fak wip selfcheck: %v\n", err)
		return 1
	}

	// Dirty the tracked file — this uncommitted delta is what the checkpoint must
	// preserve across a destructive `git checkout -- .`.
	dirty := []byte("committed base line\nWIP: an uncommitted edit worth keeping\n")
	if err := os.WriteFile(file, dirty, 0o644); err != nil {
		fmt.Fprintf(stderr, "fak wip selfcheck: %v\n", err)
		return 1
	}

	res, err := wipCheckpoint(ctx, dir, "selfcheck", true, time.Now().Unix())
	if err != nil {
		return fail(fmt.Sprintf("checkpoint failed: %v", err))
	}
	if res.Clean {
		return fail("checkpoint reported a clean tree despite an uncommitted edit")
	}

	// Wipe the delta the way an errant `git checkout -- .` would.
	if _, err := gitWipOut(ctx, dir, nil, "checkout", "--", "."); err != nil {
		return fail(fmt.Sprintf("checkout to wipe delta failed: %v", err))
	}
	if wiped, _ := os.ReadFile(file); string(wiped) == string(dirty) {
		return fail("checkout did not clear the delta — test precondition broken")
	}

	// status must list exactly the one checkpoint we took.
	report, err := wipStatus(ctx, dir, time.Now().Unix())
	if err != nil {
		return fail(fmt.Sprintf("status failed: %v", err))
	}
	if report.Count != 1 || report.Sessions[0].Session != "selfcheck" {
		return fail(fmt.Sprintf("status did not list the checkpoint (count=%d)", report.Count))
	}

	if _, err := wipRestore(ctx, dir, "selfcheck", true, io.Discard); err != nil {
		return fail(fmt.Sprintf("restore failed: %v", err))
	}
	restored, err := os.ReadFile(file)
	if err != nil {
		return fail(fmt.Sprintf("read restored file: %v", err))
	}
	if string(restored) != string(dirty) {
		return fail("restored working tree does not match the pre-checkpoint delta byte-for-byte")
	}

	return wipSelfcheckVerdict(stdout, stderr, *asJSON, true,
		"checkpoint -> git checkout -- . -> restore reproduced the delta byte-identical; status listed it")
}

func wipSelfcheckVerdict(stdout, stderr io.Writer, asJSON, pass bool, detail string) int {
	if asJSON {
		if code := encodeJSONOrFail(stdout, stderr, map[string]any{"pass": pass, "detail": detail}, "fak wip selfcheck"); code != 0 {
			return code
		}
	} else if pass {
		fmt.Fprintf(stdout, "PASS: %s\n", detail)
	} else {
		fmt.Fprintf(stdout, "FAIL: %s\n", detail)
	}
	if pass {
		return 0
	}
	return 1
}

func shortWipSHA(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}
