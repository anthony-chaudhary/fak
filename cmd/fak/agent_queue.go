package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentqueue"
)

func cmdAgentQueue(args []string) {
	os.Exit(runAgentQueue(os.Stdout, os.Stderr, args))
}

func runAgentQueue(stdout, stderr io.Writer, args []string) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return runAgentQueueContext(ctx, stdout, stderr, args)
}

func runAgentQueueContext(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		out := stdout
		if len(args) == 0 {
			out = stderr
		}
		fmt.Fprintln(out, "usage: fak agent-queue reconcile --state SNAPSHOT.json [--json]\n       fak agent-queue reconcile-restart --state SNAPSHOT.json [--json] [--apply]\n       fak agent-queue run --state queue.json [--interval 5s] [--fak <path>] [--json] [--once] [--reconcile-restart]")
		if len(args) == 0 {
			return 2
		}
		return 0
	}
	switch args[0] {
	case "reconcile":
		return runAgentQueueReconcile(stdout, stderr, args[1:])
	case "reconcile-restart":
		return runAgentQueueReconcileRestart(stdout, stderr, args[1:])
	case "run":
		return runAgentQueueRun(ctx, stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "agent-queue: unknown subcommand %q\n", args[0])
		return 2
	}
}

func runAgentQueueReconcileRestart(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("agent-queue reconcile-restart", flag.ContinueOnError)
	fs.SetOutput(stderr)
	state := fs.String("state", "", "snapshot JSON (- for stdin)")
	asJSON := fs.Bool("json", false, "emit JSON restart reconciliation receipt")
	apply := fs.Bool("apply", false, "apply changes and save updated snapshot to state file")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *state == "" {
		fmt.Fprintln(stderr, "agent-queue reconcile-restart: --state is required")
		return 2
	}
	var r io.Reader = os.Stdin
	if *state != "-" {
		b, e := os.ReadFile(*state)
		if e != nil {
			fmt.Fprintln(stderr, e)
			return 1
		}
		r = bytes.NewReader(b)
	}
	var s agentqueue.Snapshot
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if e := d.Decode(&s); e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	rec, updated, e := agentqueue.ReconcileRestart(s, nil, agentqueue.RestartOptions{})
	if e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	if *apply && *state != "-" {
		store := agentqueue.FileStore(*state)
		if err := store.Save(updated); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(rec)
		return 0
	}
	fmt.Fprintf(stdout, "generation=%s adopted=%d replaced=%d held=%d\n", rec.Generation, len(rec.Adopted), len(rec.Replaced), len(rec.Held))
	return 0
}

func runAgentQueueReconcile(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("agent-queue reconcile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	state := fs.String("state", "", "snapshot JSON (- for stdin)")
	asJSON := fs.Bool("json", false, "emit JSON receipt")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *state == "" {
		fmt.Fprintln(stderr, "agent-queue reconcile: --state is required")
		return 2
	}
	var r io.Reader = os.Stdin
	if *state != "-" {
		f, e := os.Open(*state)
		if e != nil {
			fmt.Fprintln(stderr, e)
			return 1
		}
		defer f.Close()
		r = f
	}
	var s agentqueue.Snapshot
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if e := d.Decode(&s); e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	receipt, e := agentqueue.Reconcile(s)
	if e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(receipt)
		return 0
	}
	fmt.Fprintf(stdout, "pool=%s desired=%d observed=%d start=%d hold=%v\n", receipt.PoolID, receipt.Desired, receipt.Observed, len(receipt.Start), receipt.Hold)
	return 0
}

func runAgentQueueRun(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("agent-queue run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	state := fs.String("state", "", "path to the JSON queue state file")
	interval := fs.Duration("interval", 5*time.Second, "ticker interval")
	defaultFak := os.Args[0]
	if defaultFak == "" {
		defaultFak = "fak"
	}
	fakPath := fs.String("fak", defaultFak, "executable path to fak")
	asJSON := fs.Bool("json", false, "stream each TickReceipt as JSON to stdout")
	once := fs.Bool("once", false, "runs a single tick and exits 0")
	reconcileRestart := fs.Bool("reconcile-restart", false, "reconcile orphaned attempts and stale leases on start")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *state == "" {
		fmt.Fprintln(stderr, "agent-queue run: --state is required")
		return 2
	}

	store := agentqueue.FileStore(*state)
	controller := agentqueue.Controller{
		Store:            store,
		FakPath:          *fakPath,
		Interval:         *interval,
		ReconcileOnStart: *reconcileRestart,
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var enc *json.Encoder
	if *asJSON {
		enc = json.NewEncoder(stdout)
	}

	err := controller.Run(runCtx, func(receipt agentqueue.TickReceipt) {
		if *asJSON {
			_ = enc.Encode(receipt)
		} else {
			restartInfo := ""
			if receipt.Restart != nil {
				restartInfo = fmt.Sprintf(" restart=[adopted=%d replaced=%d held=%d]",
					len(receipt.Restart.Adopted), len(receipt.Restart.Replaced), len(receipt.Restart.Held))
			}
			fmt.Fprintf(stdout, "generation=%s pool=%s desired=%d observed=%d start=%d launches=%d%s\n",
				receipt.Generation, receipt.Plan.PoolID, receipt.Plan.Desired, receipt.Plan.Observed, len(receipt.Plan.Start), len(receipt.Launches), restartInfo)
		}
		if *once {
			cancel()
		}
	})
	if err != nil {
		fmt.Fprintf(stderr, "agent-queue run: %v\n", err)
		return 1
	}
	return 0
}
