package main

// session_checkpoint.go — `fak session checkpoint`, the operator verb that takes an
// on-demand durable SNAPSHOT of a session (issue #2760, the out-of-band operator control
// epic #2753). It is the substrate fork, safe experimentation, and crash-durable resume
// all build on: capture a running session's addressable state now, restore from it later.
//
//	fak session checkpoint <image-dir> --out <snap-dir> [--reason R] [--json]
//
// It is an OFFLINE verb (like `fak session branch` / `reset-diff`): it never dials a live
// gateway. It composes internal/sessionimage.SnapshotDir — which LoadDir-verifies the
// source bundle (a torn mid-write bundle fails closed), shares the content-addressed recall
// pages copy-on-write (no deep copy), PRESERVES the session id (a checkpoint is the same
// session captured, not a fork), and records the "checkpoint of <id> at <sha>" lineage in
// the snapshot's image.json migration log. The source bundle is only READ, never written,
// so the session that owns it keeps running, unaffected.
//
// Contrast `fak session branch`, which re-keys the fork to a NEW durable id with a
// parent_id link; a checkpoint keeps the id, because it IS that session's point-in-time
// image.

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
)

// runSessionCheckpoint is the testable core of `fak session checkpoint`: it returns the
// process exit code (0 ok, 1 a runtime error, 2 a usage error) and takes its streams
// explicitly so a test can drive it and assert the rendered output.
func runSessionCheckpoint(stdout, stderr io.Writer, argv []string) int {
	// The source checkpoint is the one leading positional; it comes before any flags so
	// `fak session checkpoint <dir> --out <dir>` parses cleanly (Go's flag package stops at
	// the first non-flag token, so the source must be split off before Parse — the same
	// positionals-first discipline `fak session branch` uses).
	if len(argv) == 0 || strings.HasPrefix(argv[0], "-") {
		fmt.Fprintln(stderr, "fak session checkpoint: want <image-dir> first, then flags")
		return 2
	}
	srcArg := argv[0]
	flagArgs := argv[1:]

	fs := flag.NewFlagSet("session checkpoint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	out := fs.String("out", "", "destination directory for the snapshot image (required)")
	reason := fs.String("reason", "", "operator note folded into the snapshot's migration-log entry")
	asJSON := fs.Bool("json", false, "emit the snapshot Meta as JSON instead of the human summary")
	if rc, ok := parseFlagsOrHelp(fs, flagArgs); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session checkpoint: unexpected argument %q (the image dir comes first, then flags)\n", fs.Arg(0))
		return 2
	}

	// The op's shape rule is the closed CHECKPOINT_MALFORMED refusal — an empty destination
	// has nowhere to write. Validating through the typed op keeps the CLI and the op contract
	// in lockstep rather than re-inventing the check.
	if r := (sessionctl.Checkpoint{Dest: *out}).Validate(); r != nil {
		fmt.Fprintf(stderr, "fak session checkpoint: %v\n", r)
		return 2
	}
	srcDir := pathutil.ExpandTilde(srcArg)
	snapDir := pathutil.ExpandTilde(strings.TrimSpace(*out))

	meta, err := sessionimage.SnapshotDir(srcDir, snapDir, sessionimage.SnapshotOptions{Reason: *reason})
	if err != nil {
		fmt.Fprintf(stderr, "fak session checkpoint: %v\n", err)
		return 1
	}

	if *asJSON {
		return emitSessionJSON(stdout, stderr, meta)
	}
	fmt.Fprintf(stdout, "checkpointed %s -> %s (source unaffected)\n", meta.SessionID, snapDir)
	if len(meta.Migrations) > 0 {
		fmt.Fprintf(stdout, "  lineage: %s\n", meta.Migrations[len(meta.Migrations)-1].Reason)
	}
	fmt.Fprintf(stdout, "  image:   %s (%d shared parts, copy-on-write)\n", snapDir, len(meta.Parts))
	return 0
}
