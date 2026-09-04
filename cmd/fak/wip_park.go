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
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// runWipPark handles `fak wip park <session> --path <p>... [--target <ref>] [--apply] [--json]`.
func runWipPark(stdout, stderr io.Writer, argv []string) int {
	var sessionArg string
	// Check if the first argument is a positional session id (doesn't start with -)
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		sessionArg = argv[0]
		argv = argv[1:]
	}

	fs := flag.NewFlagSet("wip park", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	sessionFlag := fs.String("session", "", "session id (default: positional arg, else $CLAUDE_CODE_SESSION_ID or $FAK_SESSION_ID)")
	target := fs.String("target", "", "target ref or commit to integrate (default: origin/main or tracking branch)")
	apply := fs.Bool("apply", false, "execute in-place integration and reapply (default: false, dry-run preview)")
	asJSON := fs.Bool("json", false, "emit park receipt as JSON")

	var paths pathList
	fs.Var(&paths, "path", "repo-relative path to suspend and reapply (repeatable)")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	sess := strings.TrimSpace(sessionArg)
	if sess == "" {
		sess = strings.TrimSpace(*sessionFlag)
	}
	if sess == "" && len(fs.Args()) > 0 {
		sess = strings.TrimSpace(fs.Arg(0))
	}
	if sess == "" {
		sess = firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
	}
	if sess == "" {
		fmt.Fprintln(stderr, "fak wip park: session id is required (pass as argument or via --session)")
		return 2
	}
	if !wipref.ValidSession(sess) {
		fmt.Fprintf(stderr, "fak wip park: invalid session id %q\n", sess)
		return 2
	}

	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak wip park: at least one --path is required")
		return 2
	}

	repoPath := pathutil.ExpandTilde(*repo)
	if repoPath == "" {
		repoPath = "."
	}

	opts := safesync.ParkOptions{
		Repo:      repoPath,
		Session:   sess,
		Paths:     paths,
		TargetRef: strings.TrimSpace(*target),
		Apply:     *apply,
	}

	receipt, err := safesync.Park(context.Background(), opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip park: %v\n", err)
		if *asJSON {
			_ = encodeJSONOrFail(stdout, stderr, receipt, "fak wip park")
		}
		return 1
	}

	if *asJSON {
		if err := encodeJSONOrFail(stdout, stderr, receipt, "fak wip park"); err != 0 {
			return err
		}
	} else {
		renderWipPark(stdout, receipt)
	}

	if !receipt.OK || receipt.Status == safesync.ParkStatusConflict {
		return 3
	}
	return 0
}

func renderWipPark(w io.Writer, r safesync.ParkReceipt) {
	fmt.Fprintf(w, "[%s] session=%s checkpoint=%s\n", r.Status, r.Session, r.CheckpointRef)
	fmt.Fprintf(w, "  base: %s  target: %s (%s)", shortWipSHA(r.BaseHEAD), shortWipSHA(r.TargetSHA), r.TargetRef)
	if r.NewHEAD != "" {
		fmt.Fprintf(w, "  new_head: %s", shortWipSHA(r.NewHEAD))
	}
	fmt.Fprintln(w)
	if len(r.Effects) > 0 {
		fmt.Fprintln(w, "  effects:")
		for _, eff := range r.Effects {
			fmt.Fprintf(w, "    %s: %s", eff.Path, eff.Classification)
			if eff.Detail != "" {
				fmt.Fprintf(w, " (%s)", eff.Detail)
			}
			fmt.Fprintln(w)
		}
	}
	if len(r.UnrelatedPathsPreserved) > 0 {
		fmt.Fprintf(w, "  unrelated preserved: %s\n", strings.Join(r.UnrelatedPathsPreserved, ", "))
	}
	if r.Reason != "" {
		fmt.Fprintf(w, "  reason: %s\n", r.Reason)
	}
	if r.Detail != "" {
		fmt.Fprintf(w, "  detail: %s\n", r.Detail)
	}
}
