package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/taskgraph"
)

func cmdTasks(argv []string) { os.Exit(runTasks(os.Stdout, os.Stderr, argv)) }

// runTasks is the thin shell over internal/taskgraph — the shared task list
// pure-folded to a typed table with lease-gated claims. The leaf is a pure,
// init-free fold, so its verdict vocabulary is registered here, by the consumer
// (the toolproc/egressfloor pattern: internal/abi is human-owned; RegisterReason
// is the sanctioned additive path).
func runTasks(stdout, stderr io.Writer, argv []string) int {
	for _, pr := range taskgraph.ReasonPairs() {
		abi.RegisterReason(pr.Code, pr.Name)
	}
	if len(argv) == 0 {
		tasksUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "table":
		return runTasksTable(stdout, stderr, argv[1:])
	case "sample":
		return runTasksSample(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		tasksUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak tasks: unknown subcommand %q (table | sample)\n", argv[0])
		tasksUsage(stderr)
		return 2
	}
}

func runTasksTable(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tasks table", flag.ContinueOnError)
	fs.SetOutput(stderr)
	eventsPath := fs.String("events", "", "JSONL task journal (created/claimed/blocked/completed/abandoned); required, '-' reads stdin")
	nowMS := fs.Int64("now-unix-ms", 0, "fold instant (default: wall clock; pin it for deterministic fixtures)")
	graceMS := fs.Int64("lease-grace-ms", 0, "extra window a lease stays live past its declared expiry (0 = none)")
	asJSON := fs.Bool("json", false, "emit the table as JSON")
	failOnRefused := fs.Bool("fail-on-refused", false, "exit 3 when any task carries a refusal finding (gate-able)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*eventsPath) == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak tasks table: --events FILE is required ('-' reads stdin)")
		return 2
	}
	var in io.Reader = os.Stdin
	if *eventsPath != "-" {
		f, err := os.Open(*eventsPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak tasks table: %v\n", err)
			return 1
		}
		defer f.Close()
		in = f
	}
	events, err := taskgraph.ParseEvents(in)
	if err != nil {
		fmt.Fprintf(stderr, "fak tasks table: %v\n", err)
		return 1
	}
	now := *nowMS
	if now == 0 {
		now = time.Now().UnixMilli()
	}
	tab, err := taskgraph.Fold(events, now, taskgraph.Config{LeaseGraceMS: *graceMS})
	if err != nil {
		fmt.Fprintf(stderr, "fak tasks table: %v\n", err)
		return 1
	}
	if *asJSON {
		if rc := encodeJSONOrFail(stdout, stderr, tab, "fak tasks table"); rc != 0 {
			return rc
		}
	} else {
		renderTasksTable(stdout, tab)
	}
	if *failOnRefused && tab.AttentionNeeded {
		return 3
	}
	return 0
}

func runTasksSample(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tasks sample", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the folded table as JSON")
	journal := fs.Bool("journal", false, "print the raw sample journal JSONL (pipe it into `fak tasks table --events -`)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak tasks sample: unexpected positional arguments")
		return 2
	}
	events, now, cfg := taskgraph.Sample()
	if *journal {
		for _, ev := range events {
			b, err := json.Marshal(ev)
			if err != nil {
				fmt.Fprintf(stderr, "fak tasks sample: encode: %v\n", err)
				return 1
			}
			fmt.Fprintln(stdout, string(b))
		}
		return 0
	}
	tab, err := taskgraph.Fold(events, now, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "fak tasks sample: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, tab, "fak tasks sample")
	}
	renderTasksTable(stdout, tab)
	fmt.Fprintln(stdout, "sample: a deterministic built-in journal (no key, no model, no GPU) — one row per status class plus a refused claim")
	return 0
}

// renderTasksTable prints the owner-lease, status, and blockedBy columns the
// acceptance names. The render itself lives in the pure leaf (taskgraph.RenderText)
// so it is unit-testable off the poisoned cmd build; this is the thin delegate.
func renderTasksTable(w io.Writer, tab taskgraph.Table) { taskgraph.RenderText(w, tab) }

func tasksUsage(w io.Writer) {
	fmt.Fprint(w, `fak tasks - the shared task list pure-folded to a typed table with lease-gated claims

  fak tasks table --events FILE|- [--now-unix-ms N] [--lease-grace-ms N]
                  [--json] [--fail-on-refused]
  fak tasks sample [--json | --journal]

The shared task-list pattern (blocks/blockedBy, self-claiming, an only-mark-
completed-when-done honor rule) works but rests on file locks and good behavior.
taskgraph folds an append-only event journal (created / claimed / blocked /
completed / abandoned) into the task table at one instant: status, the owner-
lease that holds it, and the blockers still open — from the journal alone. Two
races become closed reasons instead of corrupt state:

  TASK_CLAIM_NO_LIVE_LEASE      a claim under an expired/absent lease is refused
  TASK_CLAIM_TREE_COLLISION     a claim over a live-claimed tree is refused
  TASK_COMPLETE_OPEN_BLOCKERS   completing over an open blocker is refused

table folds the journal and renders the columns; --fail-on-refused makes it exit
3 when any task carries a refusal (gate-able), else it exits 0. sample folds a
deterministic built-in journal exercising every status plus a refused claim (a
demo, not a gate); --journal prints the raw JSONL instead.

The fold is a pure function (same journal + now + config => byte-identical
table, injectable clock) built on the internal/toolproc discipline. Wiring the
unblocked-and-unclaimed evidence into the dispatch tick's pickDispatchLane is
the labeled next step; see the issue #2437 body.
`)
}
