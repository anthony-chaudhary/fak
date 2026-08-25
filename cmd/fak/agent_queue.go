package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/agentqueue"
	"io"
	"os"
)

func cmdAgentQueue(args []string) {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprintln(os.Stdout, "usage: fak agent-queue reconcile --state SNAPSHOT.json [--json]")
		return
	}
	if args[0] != "reconcile" {
		fmt.Fprintf(os.Stderr, "agent-queue: unknown subcommand %q\n", args[0])
		return
	}
	fs := flag.NewFlagSet("agent-queue reconcile", flag.ContinueOnError)
	state := fs.String("state", "", "snapshot JSON (- for stdin)")
	asJSON := fs.Bool("json", false, "emit JSON receipt")
	if fs.Parse(args[1:]) != nil || *state == "" {
		fmt.Fprintln(os.Stderr, "agent-queue reconcile: --state is required")
		return
	}
	var r io.Reader = os.Stdin
	if *state != "-" {
		f, e := os.Open(*state)
		if e != nil {
			fmt.Fprintln(os.Stderr, e)
			return
		}
		defer f.Close()
		r = f
	}
	var s agentqueue.Snapshot
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if e := d.Decode(&s); e != nil {
		fmt.Fprintln(os.Stderr, e)
		return
	}
	receipt, e := agentqueue.Reconcile(s)
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		return
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(receipt)
		return
	}
	fmt.Printf("pool=%s desired=%d observed=%d start=%d hold=%v\n", receipt.PoolID, receipt.Desired, receipt.Observed, len(receipt.Start), receipt.Hold)
}
