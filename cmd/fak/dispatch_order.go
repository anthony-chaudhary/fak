package main

// dispatch_order.go — `fak dispatch order`, the operator/integration front door to the
// deterministic dispatch-ORDER decision (internal/dispatchorder). It answers the question the
// issue-dispatch picker cannot today: given a set of candidate work units, which ONE should a
// worker take first, and which are stale duplicates that should be SUPERSEDED (collapsed to the
// freshest) rather than re-attempted?
//
//	# from the dispatcher: emit candidates, get back the freshest-first, supersede-collapsed pick
//	fak dispatch order --in candidates.json --json
//	cat candidates.json | fak dispatch order --cooldown-min 120
//
// The decision is PURE (internal/dispatchorder.Plan): same candidates + clock in, same order
// out — no clock read, no I/O. This shell does only the wire: read the candidate facts (a JSON
// array or {"candidates":[...]}), supply the clock and cooldown from flags, call Plan, and
// render the ordered dispositions as an aligned table or raw JSON. It is the exact leaf/shell
// split resume_scan.go uses — the decision lives in the leaf, the wire lives here.
//
// It is the missing COLLAPSE step over the current SKIP-only picker
// (tools/issue_resolve_dispatch.py: pick_target_issue): the dispatcher builds one Candidate per
// open unit (id = issue number, key = the shared-target identity, updated/created from `gh`,
// last_attempt from the run-dir log mtime, live from the pid sidecar), pipes them here, and
// takes the returned `keep` list — its head is the unit to spawn.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// cmdDispatch is the `fak dispatch` entry point; it maps the testable core's exit code to the
// process exit code, mirroring cmdResume.
func cmdDispatch(argv []string) { os.Exit(runDispatch(os.Stdout, os.Stderr, argv)) }

// runDispatch is the testable core: it returns the process exit code (0 ok, 1 a runtime error,
// 2 a usage error) and takes its streams explicitly so a test drives it without a process.
func runDispatch(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		dispatchUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "order":
		return runDispatchOrder(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		dispatchUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak dispatch: unknown subcommand %q (want order)\n", argv[0])
		dispatchUsage(stderr)
		return 2
	}
}

// runDispatchOrder parses the candidate facts, supplies the clock and cooldown, computes the
// deterministic order, and renders it.
func runDispatchOrder(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch order", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "read candidate units from this JSON file (default: stdin)")
	cooldownMin := fs.Int("cooldown-min", 120, "skip a freshest unit attempted within this many minutes (-1 disables)")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for cooldown math (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the raw Result JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}

	raw, code := readDispatchInput(stderr, *in)
	if code != 0 {
		return code
	}
	cands, err := parseCandidates(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch order: parse candidates: %v\n", err)
		return 1
	}

	now := *nowUnix
	if now == 0 {
		now = time.Now().Unix()
	}
	cooldownSec := int64(*cooldownMin) * 60
	if *cooldownMin < 0 {
		cooldownSec = -1 // disable the hold
	}

	res := dispatchorder.Plan(dispatchorder.Input{
		Candidates:      cands,
		NowUnix:         now,
		CooldownSeconds: cooldownSec,
	})

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(struct {
			dispatchorder.Result
			Pick string `json:"pick"`
		}{res, res.Pick()}); err != nil {
			fmt.Fprintf(stderr, "fak dispatch order: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	renderDispatchOrder(stdout, res, now)
	return 0
}

// readDispatchInput returns the raw JSON bytes from the --in file or stdin, plus an exit code.
func readDispatchInput(stderr io.Writer, path string) ([]byte, int) {
	if path == "" || path == "-" {
		raw, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch order: read stdin: %v\n", err)
			return nil, 1
		}
		return raw, 0
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch order: read %q: %v\n", path, err)
		return nil, 1
	}
	return raw, 0
}

// parseCandidates accepts either a bare JSON array of candidates or an object with a
// "candidates" field, so the dispatcher can pipe whichever shape it has.
func parseCandidates(raw []byte) ([]dispatchorder.Candidate, error) {
	var obj struct {
		Candidates []dispatchorder.Candidate `json:"candidates"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Candidates != nil {
		return obj.Candidates, nil
	}
	var arr []dispatchorder.Candidate
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, err
	}
	return arr, nil
}

// renderDispatchOrder prints the decision as an aligned, scannable table: the freshest-first
// keep list (the pick order), then the superseded duplicates, then the units skipped because a
// worker is live or they are cooling.
func renderDispatchOrder(w io.Writer, r dispatchorder.Result, now int64) {
	fmt.Fprintf(w, "dispatch order — %d candidate(s): %d keep  %d superseded  %d live  %d cooling\n\n",
		len(r.Order), r.KeepCount, r.SupersededCount, r.LiveCount, r.CoolingCount)

	if r.KeepCount > 0 {
		fmt.Fprintf(w, "%-4s %-16s %-14s %10s\n", "rank", "unit", "key", "age")
		for _, x := range r.Order {
			if x.Disposition == dispatchorder.DispKeep {
				fmt.Fprintf(w, "%-4d %-16s %-14s %10s\n", x.Rank, x.ID, x.Key, dispAge(now, x.Recency))
			}
		}
	}
	for _, x := range r.Order {
		switch x.Disposition {
		case dispatchorder.DispSuperseded:
			fmt.Fprintf(w, "  superseded %-16s <- %s (fresher, same key %q)\n", x.ID, x.SupersededBy, x.Key)
		case dispatchorder.DispCooling:
			fmt.Fprintf(w, "  cooling    %-16s (freshest for key %q, attempted recently)\n", x.ID, x.Key)
		case dispatchorder.DispLive:
			fmt.Fprintf(w, "  live       %-16s (worker already running)\n", x.ID)
		}
	}

	if pick := r.Pick(); pick != "" {
		fmt.Fprintf(w, "\npick: %s  (freshest dispatchable unit)\n", pick)
	} else {
		fmt.Fprintln(w, "\npick: none  (every candidate is superseded, live, or cooling)")
	}
}

// dispAge renders how long ago a recency timestamp was, relative to now (compact, unknown for a
// zero/absent timestamp or a future stamp).
func dispAge(now, recency int64) string {
	if recency <= 0 {
		return "unknown"
	}
	s := now - recency
	switch {
	case s < 0:
		return "0s"
	case s < 90:
		return fmt.Sprintf("%ds", s)
	case s < 5400:
		return fmt.Sprintf("%dm", s/60)
	case s < 172800:
		return fmt.Sprintf("%dh", s/3600)
	default:
		return fmt.Sprintf("%dd", s/86400)
	}
}

func dispatchUsage(w io.Writer) {
	fmt.Fprint(w, `fak dispatch — deterministic dispatch helpers

  fak dispatch order [--in FILE] [--cooldown-min N] [--now UNIX] [--json]

order answers "of these candidate work units, which should a worker take FIRST, and which are
stale duplicates?" It collapses units that share a target (the same "key") to the single most
recently UPDATED one (the others are SUPERSEDED, not re-attempted), folds in the live-worker
and cooldown skips, and returns the survivors freshest-first. Pure and deterministic: same
candidates + clock in, same order out.

Candidates are a JSON array (or {"candidates":[...]}), each:
  {"id":"123","key":"<shared-target>","created_unix":N,"updated_unix":N,"last_attempt_unix":N,"live":false}

example (collapse 25 tasks for the same target to the freshest, then pick):
  fak dispatch order --in candidates.json --json | jq .pick
`)
}
