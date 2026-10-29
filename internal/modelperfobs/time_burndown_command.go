package modelperfobs

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// RunTimeBurndown is the shared offline report command for both observation CLIs.
func RunTimeBurndown(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("time-burndown", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	input := fs.String("input", "model-perf.jsonl", "observation JSONL")
	phases := fs.String("phases", "", "optional request-correlated native phase JSONL")
	format := fs.String("format", "md", "md, json, or prom")
	buckets := fs.Bool("buckets", false, "emit fak-time-burndown/2 campaign buckets (implied by --baseline and --go-runtime-*)")
	baselinePath := fs.String("baseline", "", "prior fak-time-burndown/2 JSON for before -> after -> remaining")
	prefillTarget := fs.Float64("prefill-target-ms", DefaultPrefillTargetMS, "prefill target, ms per prompt token")
	decodeTarget := fs.Float64("decode-target-ms", DefaultDecodeTargetMS, "decode target, ms per completion token")
	rtBefore := fs.String("go-runtime-before", "", "/debug/vars scrape taken before the run")
	rtAfter := fs.String("go-runtime-after", "", "/debug/vars scrape taken after the run")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fs.SetOutput(out)
			fmt.Fprintln(out, "usage: time-burndown --input FILE [--phases FILE --format md|json|prom] [--buckets --baseline FILE --prefill-target-ms N --decode-target-ms N --go-runtime-before FILE --go-runtime-after FILE]")
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
	if (*rtBefore == "") != (*rtAfter == "") {
		return fmt.Errorf("time-burndown --go-runtime-before and --go-runtime-after are a pair")
	}
	if *buckets || *baselinePath != "" || *rtBefore != "" {
		if *format == "prom" {
			return fmt.Errorf("time-burndown buckets render as md or json")
		}
		return runTimeBurndownBuckets(out, *format, rows, receipts, BurndownTargets{PrefillMS: *prefillTarget, DecodeMS: *decodeTarget}, *baselinePath, *rtBefore, *rtAfter)
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

func runTimeBurndownBuckets(out io.Writer, format string, rows []Observation, receipts []RequestPhaseReceipt, targets BurndownTargets, baselinePath, rtBefore, rtAfter string) error {
	var baseline *TimeBurndownV2
	if baselinePath != "" {
		f, err := os.Open(baselinePath)
		if err != nil {
			return fmt.Errorf("open baseline: %w", err)
		}
		defer f.Close()
		if baseline, err = ReadTimeBurndownBaseline(f); err != nil {
			return err
		}
	}
	report, err := ReduceTimeBurndownBuckets(rows, receipts, targets, baseline)
	if err != nil {
		return err
	}
	if rtBefore != "" {
		var scrapes [2]runtimeobs.Receipt
		for i, path := range []string{rtBefore, rtAfter} {
			f, err := os.Open(path)
			if err != nil {
				return fmt.Errorf("open go runtime scrape: %w", err)
			}
			scrapes[i], err = ReadGoRuntimeReceipt(f)
			f.Close()
			if err != nil {
				return err
			}
		}
		if report.GoRuntime, err = GoRuntimeLineFrom(scrapes[0], scrapes[1]); err != nil {
			return err
		}
	}
	if format == "json" {
		return json.NewEncoder(out).Encode(report)
	}
	return WriteTimeBurndownBucketsMarkdown(out, report)
}
