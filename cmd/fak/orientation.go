package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orientation"
)

func cmdOrientation(args []string) {
	if err := runOrientation(args, os.Stdout, os.Stderr, time.Now); err != nil {
		fmt.Fprintf(os.Stderr, "orientation: %v\n", err)
		os.Exit(1)
	}
}

func runOrientation(args []string, stdout, stderr io.Writer, now func() time.Time) error {
	fs := flag.NewFlagSet("orientation", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit the stable JSON orientation snapshot")
	asOf := fs.String("as-of", "", "assess freshness on YYYY-MM-DD (default: today)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected arguments: %v", fs.Args())
	}
	assessmentTime := now()
	if *asOf != "" {
		parsed, err := time.Parse(time.DateOnly, *asOf)
		if err != nil {
			return fmt.Errorf("--as-of must be YYYY-MM-DD: %w", err)
		}
		assessmentTime = parsed
	}
	snapshot, err := orientation.Current()
	if err != nil {
		return err
	}
	view := orientation.Assess(snapshot, assessmentTime)
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(view)
	}
	_, err = io.WriteString(stdout, view.Text())
	return err
}
