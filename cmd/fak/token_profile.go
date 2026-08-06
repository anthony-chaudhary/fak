package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

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
	jsonOut := fs.Bool("json", false, "emit JSON")
	halo := fs.Bool("halo", false, "compare equal-total cache-heavy and decode-heavy requests")
	selfcheck := fs.Bool("selfcheck", false, "verify the equal-total halo contract")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak token-profile: no positional arguments")
		return 2
	}
	if *halo || *selfcheck {
		return runTokenProfileHalo(stdout, stderr, *jsonOut, *selfcheck)
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

	fmt.Fprintf(stdout, "TOKEN PROFILE  phase=%s total=%d worst_case_usd=$%.6f scheduler_units=%.0f\n", report.Phase, report.TotalTokens, report.WorstCaseUSD, report.SchedulerUnits)
	for _, c := range report.Classes {
		fmt.Fprintf(stdout, "  %-16s tokens=%-8d cost=$%.6f load=%.0f certainty=%s\n", c.Kind, c.Tokens, c.WorstCaseUSD, c.SchedulerUnits, c.Certainty)
	}
	fmt.Fprintf(stdout, "DOMINANCE cost=%s load=%s\n", report.DominantCostClass, report.DominantLoadClass)
	fmt.Fprintf(stdout, "SHIFT LEFT: %s\n", report.ShiftLeftAction)
	fmt.Fprintf(stdout, "BOUNDARY: %s\n", report.Boundary)
	return 0
}

type tokenProfileHalo struct {
	Schema          string              `json:"schema"`
	EqualTotal      int64               `json:"equal_total_tokens"`
	SameTotal       string              `json:"same_total_result"`
	ClassBudget     int64               `json:"class_budget_units"`
	ProfileA        tokenprofile.Report `json:"profile_a"`
	ProfileB        tokenprofile.Report `json:"profile_b"`
	VerdictA        string              `json:"profile_a_result"`
	VerdictB        string              `json:"profile_b_result"`
	ShiftLeftAction string              `json:"shift_left_action"`
}

func runTokenProfileHalo(stdout, stderr io.Writer, asJSON, selfcheck bool) int {
	const classBudget int64 = 50_000
	profileA, err := tokenprofile.Price(tokenprofile.Forecast{InputTokens: 100_000, CachedInputTokens: 90_000, MaxOutputTokens: 2_000, Prices: tokenprofile.Prices{InputUncachedPerMillion: 3, InputCachedPerMillion: .30, OutputPerMillion: 10}, Weights: tokenprofile.DefaultWeights()})
	if err != nil {
		fmt.Fprintf(stderr, "fak token-profile: cache-heavy: %v\n", err)
		return 2
	}
	profileB, err := tokenprofile.Price(tokenprofile.Forecast{InputTokens: 1_000, MaxOutputTokens: 101_000, Prices: tokenprofile.Prices{InputUncachedPerMillion: 3, InputCachedPerMillion: .30, OutputPerMillion: 10}, Weights: tokenprofile.DefaultWeights()})
	if err != nil {
		fmt.Fprintf(stderr, "fak token-profile: decode-heavy: %v\n", err)
		return 2
	}
	gradeUnits := func(units int64) string {
		if units <= classBudget {
			return "ADMIT"
		}
		return "REFUSE_CLASS_LOAD"
	}
	h := tokenProfileHalo{
		Schema:          "fak.token-profile-halo.v1",
		EqualTotal:      profileA.TotalTokens,
		SameTotal:       "ADMIT",
		ClassBudget:     classBudget,
		ProfileA:        profileA,
		ProfileB:        profileB,
		VerdictA:        gradeUnits(int64(profileA.SchedulerUnits)),
		VerdictB:        gradeUnits(int64(profileB.SchedulerUnits)),
		ShiftLeftAction: "preserve cache affinity; cap or route decode-heavy output",
	}
	if selfcheck {
		if h.EqualTotal != profileB.TotalTokens || h.VerdictA != "ADMIT" || h.VerdictB != "REFUSE_CLASS_LOAD" || int64(profileA.WorstCaseUSD*1_000_000) >= int64(profileB.WorstCaseUSD*1_000_000) {
			fmt.Fprintf(stderr, "fak token-profile: selfcheck failed: %#v\n", h)
			return 1
		}
		fmt.Fprintln(stdout, "SELFCHECK OK: equal totals, unequal cost/load, different class outcomes")
		return 0
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, h, "fak token-profile")
	}
	fmt.Fprintf(stdout, "HALO equal_total=%d scalar=%s (both) class_budget=%d\n", h.EqualTotal, h.SameTotal, h.ClassBudget)
	fmt.Fprintf(stdout, "PROFILE_A cache-heavy cost=$%.6f load=%d %s  [cost %s] [load %s]\n", float64(int64(h.ProfileA.WorstCaseUSD*1_000_000))/1_000_000, int64(h.ProfileA.SchedulerUnits), h.VerdictA, haloBar(int64(h.ProfileA.WorstCaseUSD*1_000_000), int64(h.ProfileB.WorstCaseUSD*1_000_000)), haloBar(int64(h.ProfileA.SchedulerUnits), int64(h.ProfileB.SchedulerUnits)))
	fmt.Fprintf(stdout, "PROFILE_B decode-heavy cost=$%.6f load=%d %s  [cost %s] [load %s]\n", float64(int64(h.ProfileB.WorstCaseUSD*1_000_000))/1_000_000, int64(h.ProfileB.SchedulerUnits), h.VerdictB, haloBar(int64(h.ProfileB.WorstCaseUSD*1_000_000), int64(h.ProfileB.WorstCaseUSD*1_000_000)), haloBar(int64(h.ProfileB.SchedulerUnits), int64(h.ProfileB.SchedulerUnits)))
	fmt.Fprintf(stdout, "SHIFT LEFT: %s\n", h.ShiftLeftAction)
	fmt.Fprintln(stdout, "BOUNDARY: forecast only; reconcile provider-observed usage after completion")
	return 0
}

func haloBar(value, max int64) string {
	if value <= 0 || max <= 0 {
		return "----------"
	}
	n := int((value*10 + max - 1) / max)
	if n < 1 {
		n = 1
	}
	if n > 10 {
		n = 10
	}
	return strings.Repeat("#", n) + strings.Repeat("-", 10-n)
}
