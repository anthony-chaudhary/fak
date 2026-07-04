package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	defaultKVBMReplayArtifact = "internal/compute/testdata/kvbm_agent_replay_issue2666.json"
	defaultKVBMTraceCorpus    = "internal/compute/testdata/kvbm_trace_issue2675_synthetic.json"
)

func cmdKVBM(argv []string) { os.Exit(runKVBM(os.Stdout, os.Stderr, argv)) }

func runKVBM(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		kvbmUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "replay":
		return runKVBMReplay(stdout, stderr, argv[1:])
	case "trace":
		return runKVBMTrace(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		kvbmUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak kvbm: unknown subcommand %q\n", argv[0])
		kvbmUsage(stderr)
		return 2
	}
}

func runKVBMReplay(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak kvbm replay", flag.ContinueOnError)
	fs.SetOutput(stderr)
	artifactPath := fs.String("artifact", filepath.FromSlash(defaultKVBMReplayArtifact), "path to a fak.kvbm.replay/v1 artifact")
	asJSON := fs.Bool("json", false, "emit the replay report as JSON")
	check := fs.Bool("check", false, "exit 1 unless the replay proves the #2666 validation shape")
	fs.Usage = func() { kvbmUsage(stderr) }
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak kvbm replay: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	raw, err := os.ReadFile(*artifactPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak kvbm replay: read %s: %v\n", *artifactPath, err)
		return 1
	}
	artifact, err := compute.ParseKVReplayArtifact(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak kvbm replay: parse %s: %v\n", *artifactPath, err)
		return 2
	}
	report, err := compute.ReplayKVArtifact(artifact)
	if err != nil {
		fmt.Fprintf(stderr, "fak kvbm replay: replay %s: %v\n", *artifactPath, err)
		return 1
	}
	envelope := newKVBMReplayEnvelope(*artifactPath, report)
	if *asJSON {
		if err := writeIndentedJSON(stdout, envelope); err != nil {
			fmt.Fprintf(stderr, "fak kvbm replay: encode json: %v\n", err)
			return 1
		}
	} else {
		renderKVBMReplay(stdout, envelope)
	}
	if *check && !envelope.OK {
		return 1
	}
	return 0
}

func runKVBMTrace(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak kvbm trace", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tracePath := fs.String("trace", filepath.FromSlash(defaultKVBMTraceCorpus), "path to a fak.kvbm.trace/v1 corpus")
	asJSON := fs.Bool("json", false, "emit the trace report as JSON")
	check := fs.Bool("check", false, "exit 1 unless the trace proves the #2675 policy-vs-oracle checks")
	fs.Usage = func() { kvbmUsage(stderr) }
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak kvbm trace: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	raw, err := os.ReadFile(*tracePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak kvbm trace: read %s: %v\n", *tracePath, err)
		return 1
	}
	trace, err := compute.ParseKVReplayTrace(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak kvbm trace: parse %s: %v\n", *tracePath, err)
		return 2
	}
	report, err := compute.ReplayKVTrace(trace, compute.KVEvictLRU, compute.KVEvictCostAware)
	if err != nil {
		fmt.Fprintf(stderr, "fak kvbm trace: replay %s: %v\n", *tracePath, err)
		return 1
	}
	envelope := newKVBMTraceEnvelope(*tracePath, report)
	if *asJSON {
		if err := writeIndentedJSON(stdout, envelope); err != nil {
			fmt.Fprintf(stderr, "fak kvbm trace: encode json: %v\n", err)
			return 1
		}
	} else {
		renderKVBMTrace(stdout, envelope)
	}
	if *check && !envelope.OK {
		return 1
	}
	return 0
}

type kvbmReplayEnvelope struct {
	Schema   string                         `json:"schema"`
	Artifact string                         `json:"artifact"`
	OK       bool                           `json:"ok"`
	Checks   kvbmReplayChecks               `json:"checks"`
	Report   compute.KVReplayArtifactReport `json:"report"`
}

type kvbmReplayChecks struct {
	CostAwareAtLeastLRU  bool `json:"cost_aware_at_least_lru"`
	OracleBoundsPolicies bool `json:"oracle_bounds_policies"`
	PinPressureExercised bool `json:"pin_pressure_exercised"`
	PinsSafe             bool `json:"pins_safe"`
	RestoreExercised     bool `json:"restore_exercised"`
	RestoreBytesStable   bool `json:"restore_bytes_stable"`
}

type kvbmTraceEnvelope struct {
	Schema string                      `json:"schema"`
	Trace  string                      `json:"trace"`
	OK     bool                        `json:"ok"`
	Checks kvbmTraceChecks             `json:"checks"`
	Report compute.KVReplayTraceReport `json:"report"`
}

type kvbmTraceChecks struct {
	CostAwareAtLeastLRU    bool `json:"cost_aware_at_least_lru"`
	CostAwareGDRAtLeastLRU bool `json:"cost_aware_gdr_at_least_lru"`
	OracleBoundsPolicies   bool `json:"oracle_bounds_policies"`
	StabilityNoWorse       bool `json:"stability_no_worse"`
}

func newKVBMReplayEnvelope(path string, report compute.KVReplayArtifactReport) kvbmReplayEnvelope {
	checks := kvbmReplayChecks{
		CostAwareAtLeastLRU: report.CostAwareAtLeastLRU(),
		OracleBoundsPolicies: report.Oracle.Exact &&
			(report.Oracle.HitTokens >= report.LRU.HitTokens && report.Oracle.HitTokens >= report.CostAware.HitTokens),
		PinPressureExercised: report.LRU.PinnedSkips+report.CostAware.PinnedSkips > 0,
		PinsSafe:             report.PinViolations() == 0,
		RestoreExercised:     report.LRU.Restores+report.CostAware.Restores > 0,
		RestoreBytesStable:   report.BitDriftMismatches() == 0,
	}
	return kvbmReplayEnvelope{
		Schema:   "fak.kvbm.replay.report/v1",
		Artifact: path,
		OK: checks.CostAwareAtLeastLRU &&
			checks.OracleBoundsPolicies &&
			checks.PinPressureExercised &&
			checks.PinsSafe &&
			checks.RestoreExercised &&
			checks.RestoreBytesStable,
		Checks: checks,
		Report: report,
	}
}

func newKVBMTraceEnvelope(path string, report compute.KVReplayTraceReport) kvbmTraceEnvelope {
	lru := report.Policies[compute.KVEvictLRU]
	cost := report.Policies[compute.KVEvictCostAware]
	checks := kvbmTraceChecks{
		CostAwareAtLeastLRU:    cost.HitTokens >= lru.HitTokens,
		CostAwareGDRAtLeastLRU: cost.GoodDecisionRatio >= lru.GoodDecisionRatio,
		OracleBoundsPolicies: report.Oracle.Exact &&
			(report.Oracle.HitTokens >= lru.HitTokens && report.Oracle.HitTokens >= cost.HitTokens),
		StabilityNoWorse: cost.EvictionsPerHit <= lru.EvictionsPerHit,
	}
	return kvbmTraceEnvelope{
		Schema: "fak.kvbm.trace.report/v1",
		Trace:  path,
		OK: checks.CostAwareAtLeastLRU &&
			checks.CostAwareGDRAtLeastLRU &&
			checks.OracleBoundsPolicies &&
			checks.StabilityNoWorse,
		Checks: checks,
		Report: report,
	}
}

func renderKVBMReplay(w io.Writer, env kvbmReplayEnvelope) {
	r := env.Report
	fmt.Fprintf(w, "kvbm replay: %s\n", r.Name)
	fmt.Fprintf(w, "artifact: %s\n", env.Artifact)
	fmt.Fprintf(w, "budget_bytes: %d\n", r.BudgetBytes)
	fmt.Fprintf(w, "lru:        hits=%d/%d rate=%.3f evictions=%d restores=%d pinned_skips=%d drift=%d\n",
		r.LRU.HitTokens, r.LRU.AccessTokens, r.LRU.HitRate(), r.LRU.Evictions, r.LRU.Restores, r.LRU.PinnedSkips, r.LRU.BitDriftMismatches)
	fmt.Fprintf(w, "cost-aware: hits=%d/%d rate=%.3f evictions=%d restores=%d pinned_skips=%d drift=%d\n",
		r.CostAware.HitTokens, r.CostAware.AccessTokens, r.CostAware.HitRate(), r.CostAware.Evictions, r.CostAware.Restores, r.CostAware.PinnedSkips, r.CostAware.BitDriftMismatches)
	fmt.Fprintf(w, "oracle:    hits=%d/%d exact=%t\n", r.Oracle.HitTokens, r.Oracle.AccessTokens, r.Oracle.Exact)
	fmt.Fprintf(w, "checks: cost>=lru=%t oracle_bounds=%t pin_pressure=%t pins_safe=%t restore_exercised=%t restore_bytes_stable=%t\n",
		env.Checks.CostAwareAtLeastLRU, env.Checks.OracleBoundsPolicies, env.Checks.PinPressureExercised, env.Checks.PinsSafe, env.Checks.RestoreExercised, env.Checks.RestoreBytesStable)
	verdict := "FAIL"
	if env.OK {
		verdict = "PASS"
	}
	fmt.Fprintf(w, "verdict: %s\n", verdict)
}

func renderKVBMTrace(w io.Writer, env kvbmTraceEnvelope) {
	r := env.Report
	lru := r.Policies[compute.KVEvictLRU]
	cost := r.Policies[compute.KVEvictCostAware]
	fmt.Fprintf(w, "kvbm trace: %s\n", r.Name)
	fmt.Fprintf(w, "trace: %s\n", env.Trace)
	fmt.Fprintf(w, "source: %s\n", r.Source)
	fmt.Fprintf(w, "budget_tokens: %d\n", r.BudgetTokens)
	fmt.Fprintf(w, "lru:        hits=%d/%d rate=%.3f evictions=%d evictions_per_hit=%.6f gdr=%.3f\n",
		lru.HitTokens, lru.AccessTokens, kvbmHitRate(lru.HitTokens, lru.AccessTokens), lru.Evictions, lru.EvictionsPerHit, lru.GoodDecisionRatio)
	fmt.Fprintf(w, "cost-aware: hits=%d/%d rate=%.3f evictions=%d evictions_per_hit=%.6f gdr=%.3f\n",
		cost.HitTokens, cost.AccessTokens, kvbmHitRate(cost.HitTokens, cost.AccessTokens), cost.Evictions, cost.EvictionsPerHit, cost.GoodDecisionRatio)
	fmt.Fprintf(w, "oracle:    hits=%d/%d exact=%t\n", r.Oracle.HitTokens, r.Oracle.AccessTokens, r.Oracle.Exact)
	fmt.Fprintf(w, "checks: cost>=lru=%t gdr>=lru=%t oracle_bounds=%t stability_no_worse=%t\n",
		env.Checks.CostAwareAtLeastLRU, env.Checks.CostAwareGDRAtLeastLRU, env.Checks.OracleBoundsPolicies, env.Checks.StabilityNoWorse)
	verdict := "FAIL"
	if env.OK {
		verdict = "PASS"
	}
	fmt.Fprintf(w, "verdict: %s\n", verdict)
}

func kvbmHitRate(hitTokens, accessTokens int) float64 {
	if accessTokens == 0 {
		return 0
	}
	return float64(hitTokens) / float64(accessTokens)
}

func kvbmUsage(w io.Writer) {
	fmt.Fprintf(w, `fak kvbm - cost-aware KV eviction validation

usage:
  fak kvbm replay [--artifact FILE] [--json] [--check]
  fak kvbm trace  [--trace FILE] [--json] [--check]

The default artifact is %s. --check exits 0 only when the replay proves the #2666
validation shape: cost-aware hit tokens are at least LRU at the same budget, the
offline oracle bounds both policies, pin pressure is exercised without pin violations,
and an evicted/restored span returns with identical bytes.

The default trace corpus is %s. trace --check exits 0 only when the #2675
corpus proves cost-aware >= LRU, improves good-decision ratio against the oracle, and
does not increase evictions-per-hit.
`, filepath.FromSlash(defaultKVBMReplayArtifact), filepath.FromSlash(defaultKVBMTraceCorpus))
}
