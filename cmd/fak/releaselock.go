package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/release"
)

// cmdReleaseLock is the thin CLI surface over internal/release — the single,
// process-safe release lock that every path mutating VERSION/the tag sequence
// must take. It exists so the scheduled auto-cut and the human /release wrap
// their critical section in the SAME lock (issue #1391), and so an operator can
// diagnose a stuck lock.
//
// Wiring residual: register this in cmd/fak/main.go's switch with one line —
//
//	case "release-lock":
//		cmdReleaseLock(os.Args[2:])
//
// (main.go is outside this lane's commit set; the wiring line lands in a
// follow-up once main.go is quiescent.)
func cmdReleaseLock(argv []string) { os.Exit(runReleaseLock(os.Stdout, os.Stderr, argv)) }

func runReleaseLock(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		releaseLockUsage(stderr)
		return 2
	}
	sub := argv[0]
	rest := argv[1:]
	switch sub {
	case "-h", "--help", "help":
		releaseLockUsage(stdout)
		return 0
	case "acquire", "release", "status":
		// handled below
	default:
		fmt.Fprintf(stderr, "fak release-lock: unknown subcommand %q\n", sub)
		releaseLockUsage(stderr)
		return 2
	}

	fs := flag.NewFlagSet("release-lock "+sub, flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		owner   = fs.String("owner", "", "lock owner token (default: $FAK_RELEASE_OWNER, else $CLAUDE_CODE_SESSION_ID, else host+user+pid)")
		ttl     = fs.Duration("ttl", release.DefaultTTL, "lock time-to-live; a crashed holder auto-recovers after this")
		note    = fs.String("note", "", "free-text note recorded in the lock (diagnostics)")
		force   = fs.Bool("force", false, "steal a live lock / release a lock you do not own")
		noSteal = fs.Bool("no-steal-stale", false, "do NOT take over an expired (stale) lock on acquire")
		asJSON  = fs.Bool("json", false, "emit the lock state as JSON")
	)
	if err := fs.Parse(rest); err != nil {
		return 2
	}

	opts := release.Options{Owner: *owner, Root: repoRoot()}

	switch sub {
	case "acquire":
		st, err := release.Acquire(opts, *ttl, *note, !*noSteal, *force)
		return emitLockResult(stdout, stderr, *asJSON, st, err, "acquire")
	case "release":
		st, err := release.Release(opts, *force)
		return emitLockResult(stdout, stderr, *asJSON, st, err, "release")
	case "status":
		st := release.Status(opts)
		return emitLockResult(stdout, stderr, *asJSON, st, nil, "status")
	}
	return 0
}

// emitLockResult prints the state (JSON or human) and maps the verb's outcome to
// an exit code: 0 ok, 3 contended/denied (held by another / not owner). The
// denied code matches tools/release_lock.py's EXIT_DENIED so callers that gate on
// "lock not free" see the same signal from either implementation.
func emitLockResult(stdout, stderr io.Writer, asJSON bool, st *release.State, err error, verb string) int {
	if asJSON {
		payload := map[string]any{"verb": verb, "state": st}
		if err != nil {
			payload["error"] = err.Error()
		}
		b, _ := json.MarshalIndent(payload, "", "  ")
		fmt.Fprintln(stdout, string(b))
	}
	if err != nil {
		if !asJSON {
			fmt.Fprintf(stderr, "fak release-lock %s: %v\n", verb, err)
			if st != nil && st.Lock != nil {
				fmt.Fprintf(stderr, "  held by %s (pid %d on %s)\n", st.Lock.Owner, st.Lock.PID, st.Lock.Host)
			}
		}
		return 3 // EXIT_DENIED, shared with tools/release_lock.py
	}
	if !asJSON {
		fmt.Fprintln(stdout, humanLockLine(st))
	}
	return 0
}

func humanLockLine(st *release.State) string {
	if st == nil {
		return "release-lock: (no state)"
	}
	if st.Lock == nil {
		return "release-lock: free"
	}
	state := "free"
	switch {
	case st.Stale:
		state = "stale"
	case st.Held:
		state = "held"
	}
	rem := ""
	if st.Remaining > 0 {
		rem = fmt.Sprintf(", %s left", time.Duration(st.Remaining*float64(time.Second)).Round(time.Second))
	}
	stolen := ""
	if st.Stolen != nil {
		stolen = fmt.Sprintf(" (stole from %s)", st.Stolen.Owner)
	}
	return fmt.Sprintf("release-lock: %s by %s (pid %d on %s)%s%s",
		state, st.Lock.Owner, st.Lock.PID, st.Lock.Host, rem, stolen)
}

func releaseLockUsage(w io.Writer) {
	fmt.Fprint(w, `fak release-lock — the shared process-safe release lock (VERSION/tag critical section)

Every path that mutates VERSION or the vX.Y.Z tag sequence — the scheduled
auto-cut AND a human /release — takes this SAME lock so they cannot race (#1391).

Usage:
  fak release-lock acquire [--ttl 30m] [--owner TOKEN] [--note TEXT] [--no-steal-stale] [--force] [--json]
  fak release-lock release [--owner TOKEN] [--force] [--json]
  fak release-lock status  [--json]

Owner identity defaults to $FAK_RELEASE_OWNER, else $CLAUDE_CODE_SESSION_ID, else
host+user+pid (stable within a session, so acquire/release auto-share an owner).

Exit codes: 0 ok, 2 usage, 3 contended/denied (held by another owner / not owner).
`)
}
