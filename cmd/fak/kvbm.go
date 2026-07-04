package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const defaultKVBMReplayArtifact = "internal/compute/testdata/kvbm_agent_replay_issue2666.json"

func cmdKVBM(argv []string) { os.Exit(runKVBM(os.Stdout, os.Stderr, argv)) }

func runKVBM(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		kvbmUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "replay":
		return runKVBMReplay(stdout, stderr, argv[1:])
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

type kvbmReplayEnvelope struct {
	Schema   string                         `json:"schema"`
	Artifact string                         `json:"artifact"`
	OK       bool                           `json:"ok"`
	Checks   kvbmReplayChecks               `json:"checks"`
	Report   compute.KVReplayArtifactReport `json:"report"`
}

type kvbmReplayChecks struct {
	CostAwareAtLeastLRU  bool `json:"cost_aware_at_least_lru"`
	PinPressureExercised bool `json:"pin_pressure_exercised"`
	PinsSafe             bool `json:"pins_safe"`
	RestoreExercised     bool `json:"restore_exercised"`
	RestoreBytesStable   bool `json:"restore_bytes_stable"`
}

func newKVBMReplayEnvelope(path string, report compute.KVReplayArtifactReport) kvbmReplayEnvelope {
	checks := kvbmReplayChecks{
		CostAwareAtLeastLRU:  report.CostAwareAtLeastLRU(),
		PinPressureExercised: report.LRU.PinnedSkips+report.CostAware.PinnedSkips > 0,
		PinsSafe:             report.PinViolations() == 0,
		RestoreExercised:     report.LRU.Restores+report.CostAware.Restores > 0,
		RestoreBytesStable:   report.BitDriftMismatches() == 0,
	}
	return kvbmReplayEnvelope{
		Schema:   "fak.kvbm.replay.report/v1",
		Artifact: path,
		OK: checks.CostAwareAtLeastLRU &&
			checks.PinPressureExercised &&
			checks.PinsSafe &&
			checks.RestoreExercised &&
			checks.RestoreBytesStable,
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
	fmt.Fprintf(w, "checks: cost>=lru=%t pin_pressure=%t pins_safe=%t restore_exercised=%t restore_bytes_stable=%t\n",
		env.Checks.CostAwareAtLeastLRU, env.Checks.PinPressureExercised, env.Checks.PinsSafe, env.Checks.RestoreExercised, env.Checks.RestoreBytesStable)
	verdict := "FAIL"
	if env.OK {
		verdict = "PASS"
	}
	fmt.Fprintf(w, "verdict: %s\n", verdict)
}

func kvbmUsage(w io.Writer) {
	fmt.Fprintf(w, `fak kvbm - cost-aware KV eviction validation

usage:
  fak kvbm replay [--artifact FILE] [--json] [--check]

The default artifact is %s. --check exits 0 only when the replay proves the #2666
validation shape: cost-aware hit tokens are at least LRU at the same budget, pin
pressure is exercised without pin violations, and an evicted/restored span returns
with identical bytes.
`, filepath.FromSlash(defaultKVBMReplayArtifact))
}
