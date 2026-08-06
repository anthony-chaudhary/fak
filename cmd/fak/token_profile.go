package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/tokenprofile"
)

func cmdTokenProfile(argv []string) { os.Exit(runTokenProfile(os.Stdout, os.Stderr, argv)) }

func runTokenProfile(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("token-profile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.Int64("input", 0, "forecast input tokens")
	cached := fs.Int64("cached-input", 0, "input tokens expected to receive cached-input treatment")
	output := fs.Int64("max-output", 0, "reserved maximum output tokens")
	inputPrice := fs.Float64("input-price", 0, "uncached input USD per million tokens")
	cachedPrice := fs.Float64("cached-input-price", 0, "cached input USD per million tokens")
	outputPrice := fs.Float64("output-price", 0, "output USD per million tokens")
	inputWeight := fs.Float64("input-weight", 1, "scheduler units per uncached input token")
	cachedWeight := fs.Float64("cached-input-weight", .25, "scheduler units per cached input token")
	outputWeight := fs.Float64("output-weight", 4, "scheduler units per reserved output token")
	jsonOut := fs.Bool("json", false, "emit the fak-token-profile/1 document")
	halo := fs.Bool("halo", false, "run the cache-heavy shift-left showcase")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak token-profile: no positional arguments")
		return 2
	}
	if *halo {
		*input, *cached, *output = 100000, 90000, 2000
		*inputPrice, *cachedPrice, *outputPrice = 3, .3, 10
	}
	report, err := tokenprofile.Price(tokenprofile.Forecast{
		InputTokens: *input, CachedInputTokens: *cached, MaxOutputTokens: *output,
		Prices:  tokenprofile.Prices{InputUncachedPerMillion: *inputPrice, InputCachedPerMillion: *cachedPrice, OutputPerMillion: *outputPrice},
		Weights: tokenprofile.Weights{InputUncached: *inputWeight, InputCached: *cachedWeight, Output: *outputWeight},
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak token-profile: %v\n", err)
		return 2
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak token-profile: %v\n", err)
			return 1
		}
		return 0
	}
	if *halo {
		fmt.Fprintln(stdout, "HALO: 102k total tokens is not one load: a cache-heavy request has three independently actionable classes.")
	}
	fmt.Fprintf(stdout, "TOKEN PROFILE  phase=%s total=%d worst_case_usd=$%.6f scheduler_units=%.0f\n", report.Phase, report.TotalTokens, report.WorstCaseUSD, report.SchedulerUnits)
	for _, c := range report.Classes {
		fmt.Fprintf(stdout, "  %-16s tokens=%-8d cost=$%.6f load=%.0f certainty=%s\n", c.Kind, c.Tokens, c.WorstCaseUSD, c.SchedulerUnits, c.Certainty)
	}
	fmt.Fprintf(stdout, "DOMINANCE cost=%s load=%s\n", report.DominantCostClass, report.DominantLoadClass)
	fmt.Fprintf(stdout, "SHIFT LEFT: %s\n", report.ShiftLeftAction)
	fmt.Fprintf(stdout, "BOUNDARY: %s\n", report.Boundary)
	return 0
}
