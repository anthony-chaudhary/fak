package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/commitlane"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

var commitStatusFn = commitlane.Status

// removeIndexLockFn removes a reclaimed stale .git/index.lock. Indirected so the
// reclaim actuator is testable without touching a real lock file.
var removeIndexLockFn = os.Remove

func runCommitStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit the commit-lane status as JSON")
	reclaim := fs.Bool("reclaim-stale-index-lock", false, "reclaim an orphaned .git/index.lock when the lane evidence proves it stale with no live writer (dry-run unless --apply)")
	apply := fs.Bool("apply", false, "with --reclaim-stale-index-lock, actually remove the lock (default: dry-run)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak commit status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	rep, err := commitStatusFn(context.Background(), commitlane.Options{Dir: pathutil.ExpandTilde(*dir)})
	if err != nil {
		fmt.Fprintf(stderr, "fak commit status: %v\n", err)
		return 1
	}
	if *reclaim {
		return runIndexLockReclaim(stdout, stderr, rep, *apply)
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak commit status: %v\n", err)
			return 1
		}
		return 0
	}
	renderCommitStatus(stdout, rep)
	return 0
}

// runIndexLockReclaim applies the stale-.git/index.lock reclaim decision derived from
// the lane report (#5294). It NEVER removes a lock the decision would keep: only the
// present + probe-ok + no-live-writer + stale-past-grace signature reaps, and even then
// only with --apply — the default is a dry-run that reports what it would do. A lock
// another session already cleared is idempotent success, not an error.
func runIndexLockReclaim(stdout, stderr io.Writer, rep commitlane.Report, apply bool) int {
	d := commitlane.DecideIndexLockReclaim(rep)
	if !d.Reap {
		fmt.Fprintf(stdout, "index.lock: no reclaim (%s) %s\n", d.Reason, d.Path)
		return 0
	}
	if !apply {
		fmt.Fprintf(stdout, "index.lock: WOULD reclaim (%s) %s — re-run with --apply to remove\n", d.Reason, d.Path)
		return 0
	}
	if err := removeIndexLockFn(d.Path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(stdout, "index.lock: already cleared %s\n", d.Path)
			return 0
		}
		fmt.Fprintf(stderr, "fak commit status: reclaim failed: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "index.lock: reclaimed stale orphan (%s) %s\n", d.Reason, d.Path)
	return 0
}

func renderCommitStatus(w io.Writer, rep commitlane.Report) {
	fmt.Fprintf(w, "commit lane: %s", rep.Verdict)
	if rep.Reason != "" {
		fmt.Fprintf(w, " (%s)", rep.Reason)
	}
	fmt.Fprintln(w)
	if rep.RepoRoot != "" {
		fmt.Fprintf(w, "  repo: %s\n", rep.RepoRoot)
	}
	renderCommitLockLine(w, rep.CommitLock)
	renderIndexLockLine(w, rep.IndexLock)
	if rep.Owner != nil {
		fmt.Fprintf(w, "  owner: pid=%d %s\n", rep.Owner.PID, processLabel(*rep.Owner))
	}
	if len(rep.Queue) > 0 {
		fmt.Fprintf(w, "  queue: %d possible fak commit waiter(s)\n", len(rep.Queue))
		for _, q := range rep.Queue {
			fmt.Fprintf(w, "    pid=%d %s\n", q.PID, processLabel(q))
		}
	} else {
		fmt.Fprintln(w, "  queue: none observed")
	}
	if len(rep.LiveWriters) > 0 {
		fmt.Fprintf(w, "  live writers: %d observed\n", len(rep.LiveWriters))
	}
	if rep.ProcessProbe != "" && rep.ProcessProbe != "ok" {
		fmt.Fprintf(w, "  process probe: %s\n", rep.ProcessProbe)
	}
	for _, e := range rep.Errors {
		fmt.Fprintf(w, "  warning: %s\n", e)
	}
	if rep.NextAction != "" {
		fmt.Fprintf(w, "  next: %s\n", rep.NextAction)
	}
}

func renderCommitLockLine(w io.Writer, lock commitlane.CommitLock) {
	if !lock.Present {
		fmt.Fprintf(w, "  fak commit lock: none (%s)\n", lock.Path)
		return
	}
	if lock.Stale {
		fmt.Fprintf(w, "  fak commit lock: STALE pid=%d (%s)\n", lock.HolderPID, lock.Path)
		return
	}
	if lock.HolderPID > 0 {
		live := "dead"
		if lock.HolderAlive {
			live = "live"
		}
		fmt.Fprintf(w, "  fak commit lock: held pid=%d %s (%s)\n", lock.HolderPID, live, lock.Path)
		return
	}
	fmt.Fprintf(w, "  fak commit lock: present, owner unknown (%s)\n", lock.Path)
}

func renderIndexLockLine(w io.Writer, lock commitlane.IndexLock) {
	if !lock.Present {
		fmt.Fprintf(w, "  git index lock: none (%s)\n", lock.Path)
		return
	}
	age := ""
	if lock.AgeSeconds > 0 {
		age = fmt.Sprintf(", age=%ds", lock.AgeSeconds)
	}
	state := "present"
	if lock.StaleHint {
		state = "stale-hint"
	}
	fmt.Fprintf(w, "  git index lock: %s%s (%s)\n", state, age, lock.Path)
	if lock.Detail != "" {
		fmt.Fprintf(w, "    %s\n", lock.Detail)
	}
}

func processLabel(p commitlane.ProcessFact) string {
	parts := []string{}
	if p.Name != "" {
		parts = append(parts, p.Name)
	}
	if p.Match != "" {
		parts = append(parts, p.Match)
	}
	if p.Confidence != "" {
		parts = append(parts, p.Confidence)
	}
	if len(parts) == 0 {
		return strings.TrimSpace(p.Command)
	}
	label := strings.Join(parts, " ")
	if p.Command != "" {
		label += " :: " + p.Command
	}
	return label
}
