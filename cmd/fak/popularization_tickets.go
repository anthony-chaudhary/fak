package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/popularizationtickets"
)

func cmdPopularizationTickets(argv []string) {
	os.Exit(runPopularizationTickets(os.Stdout, os.Stderr, argv))
}

func runPopularizationTickets(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("popularization-tickets", flag.ContinueOnError)
	fs.SetOutput(stderr)
	emitDir := fs.String("emit-files", "", "write one .md body and .title sidecar per ticket into this directory")
	epicRef := fs.String("epic-ref", "the concept-popularization epic", "epic reference string")
	asList := fs.Bool("list", false, "print a dim/title table")
	asJSON := fs.Bool("json", false, "dump tickets as JSON")
	lanesTSV := fs.Bool("lanes-tsv", false, "print title-to-lane TSV for dispatch loops")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	tickets, err := popularizationtickets.Load()
	if err != nil {
		fmt.Fprintf(stderr, "popularization-tickets: %v\n", err)
		return 2
	}
	switch {
	case *asList:
		fmt.Fprintln(stdout, popularizationtickets.List(tickets))
	case *asJSON:
		data, err := popularizationtickets.JSON(tickets)
		if err != nil {
			fmt.Fprintf(stderr, "popularization-tickets: encode json: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
	case *lanesTSV:
		fmt.Fprint(stdout, popularizationtickets.LanesTSV(tickets))
	case *emitDir != "":
		if err := popularizationtickets.EmitFiles(*emitDir, *epicRef, tickets); err != nil {
			fmt.Fprintf(stderr, "popularization-tickets: %v\n", err)
			return 2
		}
		fmt.Fprintf(stdout, "wrote %d ticket bodies to %s\n", len(tickets), *emitDir)
	default:
		fs.Usage()
	}
	return 0
}
