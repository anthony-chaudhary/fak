package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

const (
	syncExitOK       = 0
	syncExitUsage    = 2
	syncExitRefused  = 3
	syncExitInternal = 4
)

func cmdSync(argv []string) { os.Exit(runSync(os.Stdout, os.Stderr, argv)) }

func runSync(stdout, stderr io.Writer, argv []string) int {
	command := "check"
	if len(argv) > 0 {
		switch argv[0] {
		case "check", "apply", "push":
			command = argv[0]
			argv = argv[1:]
		case "help", "-h", "--help":
			syncUsage(stdout)
			return syncExitOK
		default:
			if !strings.HasPrefix(argv[0], "-") {
				fmt.Fprintf(stderr, "fak sync: unknown command %q (want check, apply, or push)\n", argv[0])
				syncUsage(stderr)
				return syncExitUsage
			}
		}
	}

	fs := flag.NewFlagSet("sync "+command, flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", ".", "repo path (default: cwd)")
	remote := fs.String("remote", "origin", "remote name")
	branch := fs.String("branch", "", "branch to sync (default: current branch)")
	fetch := fs.Bool("fetch", false, "git fetch <remote> <branch> before assessing")
	retries := fs.Int("retries", 3, "push: total attempts before giving up on a moving trunk")
	asJSON := fs.Bool("json", false, "emit the assessment as JSON")
	if err := fs.Parse(argv); err != nil {
		return syncExitUsage
	}

	// push is the push-side sibling of check/apply: a safe `git push` that retries a
	// transient non-fast-forward race (a peer landed between fetch and push, but HEAD
	// already contains origin) and stops with a clear next step when genuinely behind.
	if command == "push" {
		res, err := safesync.SafePush(context.Background(), safesync.PushOptions{
			Repo:       pathutil.ExpandTilde(*repo),
			Remote:     *remote,
			Branch:     *branch,
			MaxRetries: *retries,
		})
		if err != nil {
			fmt.Fprintf(stderr, "fak sync: %v\n", err)
			return syncExitInternal
		}
		if *asJSON {
			if err := writeIndentedJSON(stdout, res); err != nil {
				fmt.Fprintf(stderr, "fak sync: %v\n", err)
				return syncExitInternal
			}
		} else {
			renderSyncPush(stdout, res)
		}
		if res.Pushed {
			return syncExitOK
		}
		return syncExitRefused
	}

	opts := safesync.Options{
		Repo:   pathutil.ExpandTilde(*repo),
		Remote: *remote,
		Branch: *branch,
		Fetch:  *fetch,
	}
	var (
		info safesync.Assessment
		err  error
	)
	if command == "apply" {
		info, err = safesync.Apply(context.Background(), opts)
	} else {
		info, err = safesync.Assess(context.Background(), opts)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak sync: %v\n", err)
		return syncExitInternal
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, info); err != nil {
			fmt.Fprintf(stderr, "fak sync: %v\n", err)
			return syncExitInternal
		}
	} else {
		renderSync(stdout, command, info)
	}

	if info.State == safesync.StateInSync {
		return syncExitOK
	}
	if command == "apply" {
		if info.Applied {
			return syncExitOK
		}
		return syncExitRefused
	}
	if info.OK {
		return syncExitOK
	}
	return syncExitRefused
}

func syncUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak sync [check] [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]
  fak sync apply   [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]
  fak sync push    [--repo DIR] [--remote origin] [--branch B] [--retries N] [--json]

Safe shared-trunk git for dirty worktrees. check is read-only except for optional
--fetch. apply runs the fast-forward only when every path Git would write is clean at
HEAD or already byte-identical to the remote-tracking version. push pushes the branch
and retries a TRANSIENT non-fast-forward race (a peer landed between fetch and push,
but HEAD already contains origin); on a genuine behind/diverged state it stops with a
clear integrate-then-push next step. None of these run git pull, stash, reset --hard,
clean, add, merge, or --force.
`)
}

// renderSyncPush is the human view of a SafePush outcome.
func renderSyncPush(w io.Writer, res safesync.PushResult) {
	if res.Pushed {
		attempts := "1 attempt"
		if res.Attempts != 1 {
			attempts = fmt.Sprintf("%d attempts", res.Attempts)
		}
		fmt.Fprintf(w, "pushed %s -> %s/%s (%s)\n", res.Branch, res.Remote, res.Branch, attempts)
		return
	}
	fmt.Fprintf(w, "[REFUSED] not pushed (%s): %s\n", res.Reason, res.Detail)
}

func renderSync(w io.Writer, command string, info safesync.Assessment) {
	switch info.State {
	case safesync.StateInSync:
		fmt.Fprintln(w, "in sync: local branch already matches the remote; nothing to do")
	case safesync.StateAhead, safesync.StateDiverged, safesync.StateNoRemoteRef:
		fmt.Fprintf(w, "%s: %s\n", info.State, info.Reason)
	case safesync.StateBehind:
		status := "REFUSED"
		if info.Applied {
			status = "applied"
		} else if info.OK && command == "check" {
			status = "SAFE"
		}
		fmt.Fprintf(w, "[%s] behind %s: %d fast-forward path(s), %d identical, %d divergent\n",
			status, info.TargetRef, info.WriteCount, len(info.Identical), len(info.Divergent))
		if info.Reason != "" {
			fmt.Fprintf(w, "  %s\n", info.Reason)
		}
		for _, e := range info.Divergent {
			fmt.Fprintf(w, "    DIVERGES  %s  %s\n", e.Status, e.Path)
		}
		if info.Applied {
			fmt.Fprintf(w, "  HEAD -> %s (novel local work on other paths preserved)\n", short(info.NewHead))
		}
	default:
		fmt.Fprintf(w, "%s: %s\n", info.State, info.Reason)
	}
}
