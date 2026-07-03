package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/waiting"
)

func cmdWaiting(argv []string) { os.Exit(runWaiting(os.Stdout, os.Stderr, argv)) }

// runWaiting is the R3 waiting-on-human queue verb (#2272, epic #2269): it folds
// the loop-event ledger `fak loop` already writes into fak.waiting-on-human.v1 —
// every blocked-on-operator notify with no terminal end is a row carrying age,
// held resources, deadline, and the safe default that fires on expiry, ranked by
// cost-of-delay. Babysitting inverted: the fleet files tickets on the human, with
// deadlines, so the operator never scans for them.
//
// Read-only by construction. The verb SURFACES the expired-default prescription;
// executing that default (releasing the held seat/lease, requeuing) goes through
// the normal admission path, NOT this fold. Exit codes: 0 ok, 2 usage/IO error.
func runWaiting(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak waiting", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	ledger := fs.String("ledger", "", "loop ledger path (default: <root>/.fak/loops.jsonl, $FAK_LOOP_LEDGER)")
	deadline := fs.Duration("deadline", waiting.DefaultDeadline, "bounded wait before the safe default fires")
	asJSON := fs.Bool("json", false, "emit the fak.waiting-on-human.v1 queue as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak waiting: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	// Honor the same ledger default `fak loop` / `fak loop-score` use, so the queue
	// folds the ledger the live loops actually write to.
	ledgerPath := *ledger
	if ledgerPath == "" {
		ledgerPath = os.Getenv("FAK_LOOP_LEDGER")
	}
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, ".fak", "loops.jsonl")
	}

	events, err := loopmgr.Load(ledgerPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak waiting: load loop ledger: %v\n", err)
		return 2
	}

	q := waiting.Fold(events, waiting.Params{AsOf: time.Now().UTC(), Deadline: *deadline})

	if *asJSON {
		if err := writeIndentedJSON(stdout, q); err != nil {
			fmt.Fprintf(stderr, "fak waiting: encode json: %v\n", err)
			return 2
		}
		return 0
	}
	fmt.Fprint(stdout, renderWaiting(q))
	return 0
}

// renderWaiting is the human table: active rows oldest-first (cost-of-delay),
// each showing status, key, age, what it holds, and — once expired — the safe
// default that the admission path will fire. The AckClosureNotYet honesty fence
// is surfaced verbatim so the reader sees which closure source is still missing.
func renderWaiting(q waiting.Queue) string {
	var b strings.Builder
	if len(q.Items) == 0 {
		b.WriteString("waiting-on-human queue: empty — nothing is blocked on you.\n")
	} else {
		fmt.Fprintf(&b, "waiting-on-human queue: %d active row(s), %d past deadline (ranked by cost-of-delay)\n\n",
			len(q.Items), q.PastDeadline)
		for _, it := range q.Items {
			age := time.Duration(it.AgeSeconds * float64(time.Second)).Round(time.Second)
			held := "—"
			if it.Held.WorkerSeat {
				held = "worker-seat"
			}
			if len(it.Held.LeaseRefs) > 0 {
				held += " +" + strings.Join(it.Held.LeaseRefs, ",")
			}
			fmt.Fprintf(&b, "  %-16s %-24s age=%-10s held=%s", it.Status, it.Key, age, held)
			if it.Status == waiting.StatusExpiredDefault {
				fmt.Fprintf(&b, "  EXPIRED → %s", it.SafeDefault)
			}
			b.WriteByte('\n')
			if it.Reason != "" {
				fmt.Fprintf(&b, "      reason=%s\n", it.Reason)
			}
		}
	}
	if len(q.Resolved) > 0 {
		fmt.Fprintf(&b, "\nresolved this window: %d (closed on run-end)\n", len(q.Resolved))
	}
	fmt.Fprintf(&b, "\nnote: %s\n", q.AckClosureNotYet)
	return b.String()
}
