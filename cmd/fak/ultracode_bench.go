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
The verdict is GAIN only when witnessed, equal-quality outcomes improve both accepted
effects per wall second and per billed token; noisy or unequal pairs ABSTAIN.`

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
	if selfcheck {
		pair = ultracodeBenchSelfcheckPair()
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
	}
	report, err := ultracodebench.Evaluate(pair)
	if err != nil {
		fmt.Fprintf(stderr, "fak ultracode bench: %v\n", err)
		return 1
	}
	if jsonOutput {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(report)
	} else {
		fmt.Fprintf(stdout, "ULTRACODE PAIRED BENCH: %s\n", report.Verdict)
		fmt.Fprintf(stdout, "accepted effects: single=%d fleet=%d | pass rate: %.1f%% / %.1f%% | contradictions: %.1f%% / %.1f%%\n", report.Single.AcceptedEffects, report.Fleet.AcceptedEffects, report.Single.PassRate*100, report.Fleet.PassRate*100, report.Single.ContradictionRate*100, report.Fleet.ContradictionRate*100)
		fmt.Fprintf(stdout, "critical path: single=%dms fleet=%dms (%.2fx) | total worker: %dms / %dms\n", report.Single.CriticalPathMS, report.Fleet.CriticalPathMS, report.Gains.ConcurrencySpeedup, report.Single.TotalWorkerMS, report.Fleet.TotalWorkerMS)
		fmt.Fprintf(stdout, "billed tokens: single=%d fleet=%d (%.1f%% reduction) | fleet cache-read share: %.1f%%\n", report.Single.BilledTokens, report.Fleet.BilledTokens, report.Gains.BilledTokenReduction*100, report.Gains.CachedInputShare*100)
		fmt.Fprintf(stdout, "spend: single=$%.4f fleet=$%.4f | accepted/wall gain: %.1f%% | accepted/billed-token gain: %.1f%%\n", report.Single.SpendUSD, report.Fleet.SpendUSD, report.Gains.OutcomePerWallGain*100, report.Gains.OutcomePerBilledTokenGain*100)
		for _, reason := range report.Reasons {
			fmt.Fprintf(stdout, "reason: %s\n", reason)
		}
		if selfcheck {
			fmt.Fprintln(stdout, "selfcheck: PASS (offline fixture; not a live model-performance claim)")
		}
	}
	return 0
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
	return ultracodebench.Pair{Schema: ultracodebench.Schema,
		Single: ultracodebench.Run{Mode: "single", Identity: id, CriticalPathMS: 9000, TotalWorkerMS: 9000, InputTokens: 6000, OutputTokens: 900, BilledTokens: 6900, SpendUSD: .06, ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:selfcheck-single", Activation: ultracodebench.ActivationCohort{Receipts: []ultracodebench.ActivationReceipt{control}}},
		Fleet:  ultracodebench.Run{Mode: "fleet", Identity: id, CriticalPathMS: 4000, TotalWorkerMS: 10500, InputTokens: 2600, OutputTokens: 800, CacheReadTokens: 4800, CacheWriteTokens: 100, BilledTokens: 3500, SpendUSD: .035, ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:selfcheck-fleet", Activation: ultracodebench.ActivationCohort{MinimumActiveRatio: 1, Receipts: []ultracodebench.ActivationReceipt{treatment}}},
	}
}
