package main

// wip_selfcheck.go — `fak wip selfcheck`, split out of wip.go (#3022) to keep
// that file under the god-file growth ceiling. Behavior-preserving relocation of
// the checkpoint -> wipe -> restore proof harness; no logic change.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

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

	// An untracked new file alongside the tracked edit (#4336) — the most common
	// shape of new-leaf WIP; the checkpoint must capture BOTH categories.
	untracked := filepath.Join(dir, "newleaf.txt")
	untrackedBody := []byte("WIP: a brand-new untracked file worth keeping\n")
	if err := os.WriteFile(untracked, untrackedBody, 0o644); err != nil {
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

	// Wipe the delta the way an errant `git checkout -- .` plus `git clean -fd`
	// would: revert the tracked edit AND delete the untracked file.
	if _, err := gitWipOut(ctx, dir, nil, "checkout", "--", "."); err != nil {
		return fail(fmt.Sprintf("checkout to wipe delta failed: %v", err))
	}
	if wiped, _ := os.ReadFile(file); string(wiped) == string(dirty) {
		return fail("checkout did not clear the delta — test precondition broken")
	}
	if _, err := gitWipOut(ctx, dir, nil, "clean", "-fd"); err != nil {
		return fail(fmt.Sprintf("clean to wipe untracked file failed: %v", err))
	}
	if _, err := os.Stat(untracked); !os.IsNotExist(err) {
		return fail("git clean did not remove the untracked file — test precondition broken")
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
	restoredUntracked, err := os.ReadFile(untracked)
	if err != nil {
		return fail(fmt.Sprintf("restore did not re-materialize the untracked file: %v", err))
	}
	if string(restoredUntracked) != string(untrackedBody) {
		return fail("restored untracked file does not match the pre-checkpoint bytes")
	}

	return wipSelfcheckVerdict(stdout, stderr, *asJSON, true,
		"checkpoint -> checkout+clean wipe -> restore reproduced tracked edit AND untracked file byte-identical; status listed it")
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
