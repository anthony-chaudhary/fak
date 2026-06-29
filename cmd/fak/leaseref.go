package main

// fak leaseref -- the operator-facing CLI surface over internal/leaseref, the
// CROSS-MACHINE LEASE VISIBILITY substrate (#825). internal/leaseref persists a
// lease record under refs/fak/locks/<id> so lease state rides ordinary git
// fetch/push between clones; this verb is the READ side that lets that state feed
// an admission decision:
//
//   fak leaseref live [--dir DIR]            -> JSON [{lane,lane_kind,tree}, ...]
//   fak leaseref list [--json] [--dir DIR]   -> the records under refs/fak/locks/*
//   fak leaseref reap [--dir DIR]            -> delete the expired (reapable) records
//
// `live` is the headline: it emits the non-expired records projected into the
// exact live_leases shape a dos_arbitrate-style admission kernel consumes, so an
// arbiter on machine B can SEE a lease machine A pushed (after an ordinary fetch)
// instead of being blind to it. The wiring an operator runs is, e.g.:
//
//   git fetch origin 'refs/fak/locks/*:refs/fak/locks/*'
//   dos arbitrate --lane <l> --tree <t> --leases "$(fak leaseref live)"
//
// HONEST BOUNDARY (kept in lockstep with the package doc): this is DISTRIBUTION /
// VISIBILITY, not atomic acquisition — it lets the arbiter see a cross-machine
// conflict, it does not arbitrate a same-fetch-window race. Documented in
// docs/cli-reference.md.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func cmdLeaseref(argv []string) { os.Exit(runLeaseref(os.Stdout, os.Stderr, argv)) }

func runLeaseref(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, leaserefUsage)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "live":
		return runLeaserefLive(stdout, stderr, rest)
	case "list":
		return runLeaserefList(stdout, stderr, rest)
	case "reap":
		return runLeaserefReap(stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, leaserefUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak leaseref: unknown subcommand %q\n%s\n", sub, leaserefUsage)
		return 2
	}
}

const leaserefUsage = `fak leaseref - cross-machine lease visibility (over internal/leaseref, #825)

  fak leaseref live [--dir DIR]
      Read the NON-EXPIRED records under refs/fak/locks/* and emit them as the
      dos_arbitrate live_leases JSON array [{lane,lane_kind,tree}, ...]. This is
      the source that makes a peer's lease (fetched into the local ref store)
      visible at admission. Pipe it: dos arbitrate ... --leases "$(fak leaseref live)".

  fak leaseref list [--json] [--dir DIR]
      List every record under refs/fak/locks/* (incl. expired), one per line with
      its LIVE/EXPIRED status; --json emits the raw records.

  fak leaseref reap [--dir DIR]
      Delete the expired (reapable) records — BOTH expired lock leases and expired
      guard-session descriptors under refs/fak/locks/*. A crashed holder's lapsed
      lease (or a crashed node's lapsed session) is bounded, not a permanent ghost.
      The delete is an ordinary ref delete that converges across clones the same way
      acquisition does.

This is VISIBILITY, not atomic acquisition: it lets an arbiter SEE a cross-machine
conflict, it does not arbitrate a same-fetch-window race.
Exit: 0 ok, 2 usage/parse error, 1 a git/store failure.`

func runLeaserefLive(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref live", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	store := leaseref.NewInDir(*dir)
	leases, err := store.LiveLeases(context.Background(), time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref live: %v\n", err)
		return 1
	}
	return emitLeaserefJSON(stdout, stderr, leases, "live")
}

func runLeaserefList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit the raw records as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	store := leaseref.NewInDir(*dir)
	recs, err := store.List(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref list: %v\n", err)
		return 1
	}
	if *asJSON {
		if recs == nil {
			recs = []leaseref.Record{}
		}
		return emitLeaserefJSON(stdout, stderr, recs, "list")
	}
	now := time.Now()
	if len(recs) == 0 {
		fmt.Fprintln(stdout, "no leases under refs/fak/locks/*")
		return 0
	}
	for _, r := range recs {
		status := "LIVE"
		if r.Expired(now) {
			status = "EXPIRED"
		}
		fmt.Fprintf(stdout, "%-24s holder=%s tree=%v ttl=%ds %s\n", r.ID, r.Holder, r.TreeGlobs, r.TTLSeconds, status)
	}
	return 0
}

func runLeaserefReap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	store := leaseref.NewInDir(*dir)
	ctx := context.Background()
	now := time.Now()
	rc := 0

	// Reap BOTH ref kinds under refs/fak/locks/*: expired lock leases and expired session
	// descriptors. Each sweep is independent and best-effort — a failure on one kind is
	// reported but never suppresses the other. Store.Reap / ReapSessions delete only their
	// own kind (the namespace split), so a session is never mistaken for a lock lease.
	leases, lerr := store.Reap(ctx, now)
	if lerr != nil {
		fmt.Fprintf(stderr, "fak leaseref reap: leases: %v\n", lerr)
		rc = 1
	}
	sessions, serr := store.ReapSessions(ctx, now)
	if serr != nil {
		fmt.Fprintf(stderr, "fak leaseref reap: sessions: %v\n", serr)
		rc = 1
	}
	fmt.Fprintf(stdout, "reaped %d expired lease(s), %d expired session(s)\n", len(leases), len(sessions))
	return rc
}

func emitLeaserefJSON(stdout, stderr io.Writer, v any, sub string) int {
	if err := writeIndentedJSON(stdout, v); err != nil {
		fmt.Fprintf(stderr, "fak leaseref %s: encode json: %v\n", sub, err)
		return 1
	}
	return 0
}
