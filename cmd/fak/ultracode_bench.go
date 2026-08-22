package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

const ultracodeBenchUsage = `usage: fak ultracode bench --pair PAIR.json [--json]
       fak ultracode bench --selfcheck [--json]

Evaluate identical single-agent and ultracode-fleet runs as one paired artifact.
The pair's cost_comparison requests billed_tokens, spend_usd, or both. The verdict is
GAIN only when witnessed, equal-quality outcomes improve wall efficiency and every
requested provider-billing axis; unavailable, noisy, or unequal pairs ABSTAIN.
Fleet pairs without a complete provider-authoritative aggregate budget receipt ABSTAIN.`

func runUltracodeBench(stdout, stderr io.Writer, args []string) int {
	var pairPath string
	selfcheck, jsonOutput := false, false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-h", "--help", "help":
			fmt.Fprintln(stdout, ultracodeBenchUsage)
			return 0
		case "--selfcheck":
			selfcheck = true
		case "--json":
			jsonOutput = true
		case "--pair":
			if i+1 >= len(args) {
				fmt.Fprintln(stderr, "fak ultracode bench: --pair requires a path")
				return 2
			}
			i++
			pairPath = args[i]
		default:
			fmt.Fprintf(stderr, "fak ultracode bench: unknown argument %q\n", args[i])
			return 2
		}
	}
	if selfcheck == (pairPath != "") {
		fmt.Fprintln(stderr, "fak ultracode bench: choose exactly one of --selfcheck or --pair PATH")
		return 2
	}
	var pair ultracodebench.Pair
	var budgetReceipt ultracodeBenchBudgetReceipt
	if selfcheck {
		pair = ultracodeBenchSelfcheckPair()
		budgetReceipt = ultracodeBenchSelfcheckBudget(pair)
	} else {
		b, err := os.ReadFile(pairPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak ultracode bench: %v\n", err)
			return 1
		}
		if err := json.Unmarshal(b, &pair); err != nil {
			fmt.Fprintf(stderr, "fak ultracode bench: decode pair: %v\n", err)
			return 1
		}
		budgetReceipt, err = decodeUltracodeBenchBudget(b)
		if err != nil {
			fmt.Fprintf(stderr, "fak ultracode bench: decode budget receipt: %v\n", err)
			return 1
		}
	}
	report, err := ultracodebench.Evaluate(pair)
	if err != nil {
		fmt.Fprintf(stderr, "fak ultracode bench: %v\n", err)
		return 1
	}
	report = applyUltracodeBenchBudgetReceipt(report, pair, budgetReceipt)
	if jsonOutput {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(report)
	} else {
		fmt.Fprintf(stdout, "ULTRACODE PAIRED BENCH: %s\n", report.Verdict)
		fmt.Fprintf(stdout, "accepted effects: single=%d fleet=%d | pass rate: %.1f%% / %.1f%% | contradictions: %.1f%% / %.1f%%\n", report.Single.AcceptedEffects, report.Fleet.AcceptedEffects, report.Single.PassRate*100, report.Fleet.PassRate*100, report.Single.ContradictionRate*100, report.Fleet.ContradictionRate*100)
		fmt.Fprintf(stdout, "critical path: single=%dms fleet=%dms (%.2fx) | total worker: %dms / %dms\n", report.Single.CriticalPathMS, report.Fleet.CriticalPathMS, report.Gains.ConcurrencySpeedup, report.Single.TotalWorkerMS, report.Fleet.TotalWorkerMS)
		writeAccountingLines(stdout, report)
		fmt.Fprintf(stdout, "accepted/wall gain: %.1f%% | accepted/billed-token gain: %s | accepted/dollar gain: %s\n", report.Gains.OutcomePerWallGain*100, percentOrUnavailable(report.Gains.OutcomePerBilledTokenGain), percentOrUnavailable(report.Gains.OutcomePerUSDGain))
		for _, reason := range report.Reasons {
			fmt.Fprintf(stdout, "reason: %s\n", reason)
		}
		if selfcheck {
			fmt.Fprintln(stdout, "selfcheck: PASS (offline fixture; not a live model-performance claim)")
		}
	}
	return 0
}

func writeAccountingLines(stdout io.Writer, report ultracodebench.Report) {
	singleBilled, fleetBilled := report.Single.Accounting.BilledTokens, report.Fleet.Accounting.BilledTokens
	if report.Gains.BilledTokenReduction == nil || singleBilled.Value == nil || fleetBilled.Value == nil {
		fmt.Fprintf(stdout, "billed tokens: unavailable (single=%s/%.0f%% fleet=%s/%.0f%%) | fleet cache-read share: %.1f%%\n", singleBilled.Authority, singleBilled.Coverage*100, fleetBilled.Authority, fleetBilled.Coverage*100, report.Gains.CachedInputShare*100)
	} else {
		fmt.Fprintf(stdout, "billed tokens: single=%d fleet=%d (%.1f%% reduction) | fleet cache-read share: %.1f%%\n", *singleBilled.Value, *fleetBilled.Value, *report.Gains.BilledTokenReduction*100, report.Gains.CachedInputShare*100)
	}
	singleSpend, fleetSpend := report.Single.Accounting.SpendUSD, report.Fleet.Accounting.SpendUSD
	if singleSpend.ValueUSD == nil || fleetSpend.ValueUSD == nil {
		fmt.Fprintf(stdout, "spend: unavailable (single=%s/%.0f%% fleet=%s/%.0f%%)\n", singleSpend.Authority, singleSpend.Coverage*100, fleetSpend.Authority, fleetSpend.Coverage*100)
	} else {
		fmt.Fprintf(stdout, "spend: single=$%.4f fleet=$%.4f\n", *singleSpend.ValueUSD, *fleetSpend.ValueUSD)
	}
}

func percentOrUnavailable(value *float64) string {
	if value == nil {
		return "unavailable"
	}
	return fmt.Sprintf("%.1f%%", *value*100)
}

func ultracodeBenchSelfcheckPair() ultracodebench.Pair {
	id := ultracodebench.Identity{Task: "repair three independent fixture defects", TaskDigest: "sha256:selfcheck-task", Model: "fixture-model", Environment: "offline-selfcheck", WallBudgetMS: 30000, TokenBudget: 10000, SpendBudgetUSD: 1}
	control, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "selfcheck-control", ChildID: "control", Harness: "codex", Requested: ultracodebench.SettingOff, Resolved: ultracodebench.SettingOff})
	if err != nil {
		panic(err)
	}
	treatment, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "selfcheck-treatment", ChildID: "treatment", Harness: "codex", Requested: ultracodebench.SettingOn, Resolved: ultracodebench.SettingOn, Injected: true})
	if err != nil {
		panic(err)
	}
	treatment, err = ultracodebench.Acknowledge(treatment, ultracodebench.ObservableActive, ultracodebench.SourceRuntimeObservation)
	if err != nil {
		panic(err)
	}
	return ultracodebench.Pair{Schema: ultracodebench.Schema, CostComparison: ultracodebench.CompareBilledTokensAndSpend,
		Single: ultracodebench.Run{Mode: "single", Identity: id, CriticalPathMS: 9000, TotalWorkerMS: 9000, InputTokens: 6000, OutputTokens: 900, BilledTokens: 6900, SpendUSD: .06, Accounting: ultracodeBenchKnownAccounting(6000, 900, 0, 0, 6900, .06, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:selfcheck-single", Activation: ultracodebench.ActivationCohort{Receipts: []ultracodebench.ActivationReceipt{control}}},
		Fleet:  ultracodebench.Run{Mode: "fleet", Identity: id, CriticalPathMS: 4000, TotalWorkerMS: 10500, InputTokens: 2600, OutputTokens: 800, CacheReadTokens: 4800, CacheWriteTokens: 100, BilledTokens: 3500, SpendUSD: .035, Accounting: ultracodeBenchKnownAccounting(2600, 800, 4800, 100, 3500, .035, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"), ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:selfcheck-fleet", Activation: ultracodebench.ActivationCohort{MinimumActiveRatio: 1, Receipts: []ultracodebench.ActivationReceipt{treatment}}},
	}
}

func ultracodeBenchKnownAccounting(input, output, cacheRead, cacheWrite, billed int64, spend float64, digest string) ultracodebench.AccountingReceipt {
	token := func(value int64, authority ultracodebench.AccountingAuthority) ultracodebench.TokenAccounting {
		return ultracodebench.TokenAccounting{Availability: ultracodebench.AccountingAvailable, Authority: authority, ArtifactDigest: digest, Coverage: 1, Value: &value}
	}
	return ultracodebench.AccountingReceipt{
		Schema:      ultracodebench.AccountingSchema,
		InputTokens: token(input, ultracodebench.AuthorityProviderUsage), OutputTokens: token(output, ultracodebench.AuthorityProviderUsage),
		CacheReadTokens: token(cacheRead, ultracodebench.AuthorityProviderUsage), CacheWriteTokens: token(cacheWrite, ultracodebench.AuthorityProviderUsage),
		BilledTokens: token(billed, ultracodebench.AuthorityProviderBilling),
		SpendUSD:     ultracodebench.SpendAccounting{Availability: ultracodebench.AccountingAvailable, Authority: ultracodebench.AuthorityProviderBilling, ArtifactDigest: digest, Coverage: 1, ValueUSD: &spend},
	}
}
