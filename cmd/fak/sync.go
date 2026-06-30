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
		case "check", "apply":
			command = argv[0]
			argv = argv[1:]
		case "help", "-h", "--help":
			syncUsage(stdout)
			return syncExitOK
		default:
			if !strings.HasPrefix(argv[0], "-") {
				fmt.Fprintf(stderr, "fak sync: unknown command %q (want check or apply)\n", argv[0])
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
	asJSON := fs.Bool("json", false, "emit the assessment as JSON")
	if err := fs.Parse(argv); err != nil {
		return syncExitUsage
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

Safe fast-forward sync for dirty shared worktrees. check is read-only except for
optional --fetch. apply runs the fast-forward only when every path Git would write
is clean at HEAD or already byte-identical to the remote-tracking version. It never
runs git pull, stash, reset --hard, clean, or add.
`)
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
