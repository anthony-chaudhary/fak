package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/fleettrend"
)

func cmdFleetTrend(argv []string) {
	os.Exit(runFleetTrend(os.Stdout, os.Stderr, os.Stdin, argv))
}

func runFleetTrend(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("fleet-trend", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", fleettrend.DefaultLedger, "JSONL history path")
	appendPath := fs.String("append", "", "append a tick from a fleet_top --json snapshot (file or - for stdin)")
	show := fs.Bool("show", false, "render the trend from the ledger")
	window := fs.Int("window", 24, "how many trailing ticks to fold")
	capRows := fs.Int("cap", fleettrend.DefaultCap, "max ledger rows")
	now := fs.String("now", "", "override the tick timestamp")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *appendPath != "" {
		var raw []byte
		var err error
		if *appendPath == "-" {
			raw, err = io.ReadAll(stdin)
		} else {
			raw, err = os.ReadFile(*appendPath)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fleet-trend: %v\n", err)
			return 2
		}
		var snap map[string]any
		if err := json.Unmarshal(raw, &snap); err != nil {
			fmt.Fprintf(stderr, "fleet-trend: parse snapshot: %v\n", err)
			return 2
		}
		stamp := *now
		if stamp == "" {
			if v, ok := snap["generated_utc"].(string); ok && v != "" {
				stamp = v
			} else {
				stamp = fleettrend.ISONow()
			}
		}
		row, err := fleettrend.Append(*ledger, fleettrend.MetricsOf(snap), stamp, *capRows)
		if err != nil {
			fmt.Fprintf(stderr, "fleet-trend: %v\n", err)
			return 2
		}
		if *asJSON {
			if err := writeIndentedJSONNoEscape(stdout, row); err != nil {
				fmt.Fprintf(stderr, "fleet-trend: encode json: %v\n", err)
				return 2
			}
		} else {
			fmt.Fprintf(stdout, "appended %v\n", row)
		}
		if !*show {
			return 0
		}
	}
	rows := fleettrend.Tail(*ledger, *window)
	if *asJSON {
		payload := map[string]any{"schema": fleettrend.Schema, "ledger": *ledger, "rows": rows}
		if err := writeIndentedJSONNoEscape(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fleet-trend: encode json: %v\n", err)
			return 2
		}
		return 0
	}
	line := fleettrend.RenderLine(rows)
	if line == "" {
		line = "(no history yet)"
	}
	fmt.Fprintln(stdout, line)
	return 0
}
