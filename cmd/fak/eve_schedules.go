package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/evebridge"
)

func runEveSchedules(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("eve schedules", flag.ContinueOnError)
	fs.SetOutput(stderr)
	host := fs.String("host", "eve-dev", "host projection: eve-dev, eve-start, vercel, or a custom host name")
	jsonOut := fs.Bool("json", false, "emit deterministic JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 1 {
		fmt.Fprintln(stderr, "usage: fak eve schedules [--json] [--host HOST] [ROOT]")
		return 2
	}
	root := "."
	if fs.NArg() == 1 {
		root = fs.Arg(0)
	}
	inventory, err := evebridge.InspectSchedules(os.DirFS(root), *host)
	if err != nil {
		fmt.Fprintf(stderr, "fak eve schedules: %v\n", err)
		return 1
	}
	if *jsonOut {
		_, _ = stdout.Write(inventory.JSON())
	} else {
		renderEveSchedules(stdout, inventory)
	}
	if !inventory.OK {
		return eveInspectFailed
	}
	return 0
}

func renderEveSchedules(w io.Writer, inventory evebridge.ScheduleInventory) {
	fmt.Fprintf(w, "Eve schedules: %d (%s)\n", len(inventory.Schedules), inventory.Host)
	for _, schedule := range inventory.Schedules {
		fmt.Fprintf(w, "  %-24s %-16s %-8s %s\n", schedule.ID, schedule.CronUTC, schedule.Form, schedule.SourcePath)
	}
	for _, diagnostic := range inventory.Diagnostics {
		fmt.Fprintf(w, "%s %s: %s (%s)\n", diagnostic.Severity, diagnostic.Code, diagnostic.Message, diagnostic.EvidencePath)
	}
}

func runEveDispatchReceipt(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("eve dispatch-receipt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "Eve app root")
	schedule := fs.String("schedule", "", "schedule id")
	session := fs.String("session", "", "started Eve session id")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *schedule == "" || *session == "" {
		fmt.Fprintln(stderr, "fak eve dispatch-receipt: --schedule and --session are required")
		return 2
	}
	inventory, err := evebridge.InspectSchedules(os.DirFS(*root), "eve-dev")
	if err != nil || !inventory.OK {
		fmt.Fprintf(stderr, "fak eve dispatch-receipt: schedule inventory failed: %v\n", err)
		return 1
	}
	receipt, err := evebridge.RecordDevDispatch(inventory, *schedule, *session)
	if err != nil {
		fmt.Fprintf(stderr, "fak eve dispatch-receipt: %v\n", err)
		return 1
	}
	b, _ := json.MarshalIndent(receipt, "", "  ")
	_, _ = fmt.Fprintf(stdout, "%s\n", b)
	return 0
}
