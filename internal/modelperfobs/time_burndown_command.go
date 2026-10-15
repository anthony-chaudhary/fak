package modelperfobs

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
)

// RunTimeBurndown is the shared offline report command for both observation CLIs.
func RunTimeBurndown(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("time-burndown", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	input := fs.String("input", "model-perf.jsonl", "observation JSONL")
	phases := fs.String("phases", "", "optional request-correlated native phase JSONL")
	format := fs.String("format", "md", "md, json, or prom")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fs.SetOutput(out)
			fmt.Fprintln(out, "usage: time-burndown --input FILE [--phases FILE --format md|json|prom]")
			fs.PrintDefaults()
			return nil
		}
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("time-burndown accepts flags only")
	}
	if *format != "md" && *format != "json" && *format != "prom" {
		return fmt.Errorf("time-burndown format must be md, json, or prom")
	}
	f, err := os.Open(*input)
	if err != nil {
		return fmt.Errorf("open observations: %w", err)
	}
	defer f.Close()
	rows, err := ReadObservations(f)
	if err != nil {
		return err
	}
	var receipts []RequestPhaseReceipt
	if *phases != "" {
		pf, err := os.Open(*phases)
		if err != nil {
			return fmt.Errorf("open native phases: %w", err)
		}
		defer pf.Close()
		receipts, err = ReadRequestPhaseReceipts(pf)
		if err != nil {
			return err
		}
	}
	report, err := ReduceTimeBurndown(rows, receipts)
	if err != nil {
		return err
	}
	switch *format {
	case "json":
		return json.NewEncoder(out).Encode(report)
	case "prom":
		return WriteTimeBurndownPrometheus(out, report)
	default:
		return WriteTimeBurndownMarkdown(out, report)
	}
}
