package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/shadowgit"
)

// cmdShadowGit handles `fak shadowgit <subcommand>` — a non-invasive per-step write
// ledger over a worktree. It drives git through a SEPARATE git dir so the repo's real
// .git is never touched; each snapshot's diff attributes the files that changed since
// the previous step. See internal/shadowgit for the mechanism.
func cmdShadowGit(args []string) {
	if len(args) == 0 {
		shadowGitUsage()
		os.Exit(2)
	}
	switch args[0] {
	case "baseline":
		cmdShadowGitBaseline(args[1:])
	case "snapshot":
		cmdShadowGitSnapshot(args[1:])
	case "status":
		cmdShadowGitStatus(args[1:])
	case "-h", "--help", "help":
		shadowGitUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak shadowgit: unknown subcommand %q\n", args[0])
		shadowGitUsage()
		os.Exit(2)
	}
}

func shadowGitUsage() {
	fmt.Fprintln(os.Stderr, "usage: fak shadowgit <baseline|snapshot|status> [--repo <dir>] [--shadow <dir>]")
	fmt.Fprintln(os.Stderr, "       fak shadowgit baseline                                  (record the reference snapshot)")
	fmt.Fprintln(os.Stderr, "       fak shadowgit snapshot --step N [--label L] [--changelog f]  (attribute writes to step N)")
	fmt.Fprintln(os.Stderr, "       fak shadowgit status                                    (are there un-snapshotted writes?)")
	fmt.Fprintln(os.Stderr, "  The shadow git dir is separate from the repo's .git — the real repo is never modified.")
}

// shadowGitFlags registers the flags common to every subcommand and opens the ledger.
func openShadow(fs *flag.FlagSet, args []string) *shadowgit.ShadowGit {
	repo := fs.String("repo", ".", "worktree to attribute writes in")
	shadow := fs.String("shadow", ".fak/shadow.git", "shadow git dir (kept separate from the repo's .git)")
	includeIgnored := fs.Bool("include-ignored", false, "also attribute writes to .gitignore'd paths")
	_ = fs.Parse(args)
	sg, err := shadowgit.Open(*shadow, *repo, shadowgit.Options{IncludeIgnored: *includeIgnored})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak shadowgit: %v\n", err)
		os.Exit(1)
	}
	return sg
}

func cmdShadowGitBaseline(args []string) {
	fs := flag.NewFlagSet("shadowgit baseline", flag.ExitOnError)
	sg := openShadow(fs, args)
	snap, err := sg.Baseline()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak shadowgit baseline: %v\n", err)
		os.Exit(1)
	}
	emitSnapshot(snap)
}

func cmdShadowGitSnapshot(args []string) {
	fs := flag.NewFlagSet("shadowgit snapshot", flag.ExitOnError)
	step := fs.Int("step", 0, "step number to attribute the writes to")
	label := fs.String("label", "", "human label for this step (e.g. the tool/turn)")
	changelog := fs.String("changelog", "", "append the snapshot to this state_changelog JSONL")
	// openShadow parses the shared flags too; register step/label/changelog on the same set.
	sg := openShadow(fs, args)
	snap, err := sg.Snapshot(*step, *label)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak shadowgit snapshot: %v\n", err)
		os.Exit(1)
	}
	if *changelog != "" {
		f, err := os.OpenFile(*changelog, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak shadowgit snapshot: %v\n", err)
			os.Exit(1)
		}
		if err := shadowgit.WriteChangelogLine(f, snap); err != nil {
			f.Close()
			fmt.Fprintf(os.Stderr, "fak shadowgit snapshot: %v\n", err)
			os.Exit(1)
		}
		f.Close()
	}
	emitSnapshot(snap)
}

func cmdShadowGitStatus(args []string) {
	fs := flag.NewFlagSet("shadowgit status", flag.ExitOnError)
	sg := openShadow(fs, args)
	dirty, err := sg.CheckForWrites()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak shadowgit status: %v\n", err)
		os.Exit(1)
	}
	json.NewEncoder(os.Stdout).Encode(map[string]bool{"pending_writes": dirty})
}

func emitSnapshot(snap shadowgit.Snapshot) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(snap)
}
