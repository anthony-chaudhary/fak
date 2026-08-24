package main

// session_fork.go — `fak session fork`, the operator verb that SNAPSHOT-AND-BRANCHES a
// running session into a divergent continuation (issue #2761, child of the out-of-band
// operator control epic #2753). It is the exploration primitive the epic names: spawn a
// divergent line (e.g. after a redirect) without losing the current one — the original keeps
// its place, the fork diverges from a pinned branch point.
//
//	fak session fork <parent-image-dir> --out <fork-dir> --checkpoint <branch-point-dir>
//	    [--id <fork-id>] [--reason R] [--to-model M] [--to-host H] [--registry PATH] [--json]
//
// It is an OFFLINE verb (like `fak session branch` / `fak session checkpoint`): it never
// dials a live gateway. It composes internal/sessionimage.ForkDir, which pins the parent's
// current state as an immutable checkpoint (the branch point, #2760 — source read-only) and
// then forks that checkpoint into a fresh-trace session (#1200 — copy-on-write share of the
// parent's pages, a parent_id link, and the recorded lineage). The parent bundle is only
// read, never written, so the running session the operator forked is unaffected.

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
)

// runSessionFork is the testable core of `fak session fork`: it returns the process exit code
// (0 ok, 1 a runtime error, 2 a usage error) and takes its streams explicitly so a test can
// drive it and assert the rendered output.
func runSessionFork(stdout, stderr io.Writer, argv []string) int {
	// The parent checkpoint is the one leading positional; it comes before any flags so
	// `fak session fork <dir> --out <dir>` parses cleanly (Go's flag package stops at the first
	// non-flag token, so the parent dir must be split off before Parse — the same positionals-
	// first discipline the other offline session verbs use).
	if len(argv) == 0 || strings.HasPrefix(argv[0], "-") {
		fmt.Fprintln(stderr, "fak session fork: want <parent-image-dir> first, then flags")
		return 2
	}
	parentArg := argv[0]
	flagArgs := argv[1:]

	fs := flag.NewFlagSet("session fork", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	out := fs.String("out", "", "destination directory for the divergent fork image (required)")
	checkpoint := fs.String("checkpoint", "", "destination directory for the pinned, immutable branch-point checkpoint (required)")
	id := fs.String("id", "", "the fork's fresh durable session id / trace (default: <parent-id>-fork)")
	reason := fs.String("reason", "", "operator note folded into the checkpoint's and fork's migration-log entries")
	toModel := fs.String("to-model", "", "re-home the fork to a different model (default: inherit the parent's)")
	toHost := fs.String("to-host", "", "re-home the fork to a different host (default: inherit the parent's)")
	registry := fs.String("registry", "", "also register the fork as a new C1 descriptor in this registry file (with a parent_id link)")
	asJSON := fs.Bool("json", false, "emit the fork result (branch-point + fork Meta) as JSON instead of the human summary")
	if rc, ok := parseFlagsOrHelp(fs, flagArgs); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session fork: unexpected argument %q (the parent dir comes first, then flags)\n", fs.Arg(0))
		return 2
	}
	parentDir := pathutil.ExpandTilde(parentArg)
	if strings.TrimSpace(*out) == "" {
		fmt.Fprintln(stderr, "fak session fork: --out <fork-dir> is required")
		return 2
	}
	if strings.TrimSpace(*checkpoint) == "" {
		fmt.Fprintln(stderr, "fak session fork: --checkpoint <branch-point-dir> is required (fork pins the branch point before it diverges)")
		return 2
	}
	forkDir := pathutil.ExpandTilde(*out)
	checkpointDir := pathutil.ExpandTilde(*checkpoint)

	// Load the parent to derive the fork id default (and fail closed early on a truncated or
	// missing bundle, with a clearer message than the op's wrapped error). ForkDir re-loads and
	// re-verifies it, so this is a convenience read, not the integrity gate.
	parent, err := sessionimage.LoadDir(parentDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak session fork: load parent %q: %v\n", parentDir, err)
		return 1
	}
	target := newSessionChildTarget(*id, parent.Meta.SessionID, "fork", *toModel, *toHost, *reason)

	res, code := runSessionBranchOperation(stderr, "fork", *registry, func() (sessionimage.ForkResult, sessionimage.Meta, error) {
		res, err := sessionimage.ForkDir(parentDir, checkpointDir, forkDir, sessionimage.ForkOptions{
			ForkID: target.id, ToModel: target.model, ToHost: target.host, Reason: target.reason,
		})
		return res, res.Fork, err
	})
	if code != 0 {
		return code
	}

	if *asJSON {
		return emitSessionJSON(stdout, stderr, res)
	}
	fmt.Fprintf(stdout, "forked %s -> %s (parent_id=%s)\n", res.Fork.ParentID, res.Fork.SessionID, res.Fork.ParentID)
	if len(res.Fork.Migrations) > 0 {
		fmt.Fprintf(stdout, "  lineage:      %s\n", res.Fork.Migrations[len(res.Fork.Migrations)-1].Reason)
	}
	fmt.Fprintf(stdout, "  branch point: %s (checkpoint of %s, %d shared parts)\n", checkpointDir, res.BranchPoint.SessionID, len(res.BranchPoint.Parts))
	fmt.Fprintf(stdout, "  fork image:   %s (%d shared parts, copy-on-write)\n", forkDir, len(res.Fork.Parts))
	if strings.TrimSpace(*registry) != "" {
		fmt.Fprintf(stdout, "  registry:     new descriptor %s registered with parent_id=%s\n", res.Fork.SessionID, res.Fork.ParentID)
	}
	return 0
}
