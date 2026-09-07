package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/perfrsiscore"
)

const deterministicFixture = `{"schema":"fak-performance-rsi-evidence/1","snapshot":"fixture","target_multiplier":100,"dimensions":[
{"id":"cycle_time","source":"fixture/cycles","direction":"lower","current":10,"target":5,"unit":"hours/cycle","next_action":"shorten cycle"},
{"id":"improvement_yield","source":"fixture/yield","direction":"higher","current":100,"target":100,"unit":"percent","next_action":"raise yield"},
{"id":"evaluation_latency","source":"fixture/eval","direction":"lower","current":20,"target":5,"unit":"minutes","next_action":"reduce evaluation latency","evidence_kind":"native_benchmark","engine":"fak-native/qwen3.8"},
{"id":"receipt_coverage","source":"fixture/receipts","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"capture receipts"},
{"id":"quality_gate_coverage","source":"fixture/quality","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"add quality gates"},
{"id":"experiment_throughput","source":"fixture/throughput","direction":"higher","current":8,"target":10,"unit":"experiments/day","next_action":"increase safe throughput"},
{"id":"hypothesis_calibration","source":"fixture/calibration","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"calibrate ranking"},
{"id":"discovery_freshness","source":"fixture/discovery","direction":"lower","current":2,"target":1,"unit":"days","next_action":"refresh discovery"},
{"id":"adaptation_speed","source":"fixture/adaptation","direction":"lower","current":2,"target":1,"unit":"days","next_action":"adapt faster"},
{"id":"reuse_ratio","source":"fixture/reuse","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"reuse mechanisms"},
{"id":"learning_retention","source":"fixture/learning","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"retain learning"},
{"id":"production_transfer","source":"fixture/transfer","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"transfer experiments"},
{"id":"hardware_utilization","source":"fixture/hardware","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"use hardware"},
{"id":"attribution_quality","source":"fixture/attribution","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"improve attribution"},
{"id":"automation_coverage","source":"fixture/automation","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"automate loop"},
{"id":"compounding_rate","source":"fixture/compounding","direction":"higher","current":80,"target":100,"unit":"percent","next_action":"compound learning"}
]}`

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("perfrsidemo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfcheck := fs.Bool("selfcheck", false, "run deterministic performance-rsi scorecard selfcheck")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: perfrsidemo [-selfcheck]")
		return 2
	}

	evidence, err := perfrsiscore.Decode(strings.NewReader(deterministicFixture))
	if err != nil {
		fmt.Fprintf(stderr, "perfrsidemo: decode fixture: %v\n", err)
		return 1
	}

	report := perfrsiscore.Score(evidence)

	if *selfcheck {
		if err := validateSelfcheck(report); err != nil {
			fmt.Fprintf(stderr, "selfcheck: FAIL: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, perfrsiscore.RenderHuman(report))
		fmt.Fprintln(stdout, "selfcheck: PASS (deterministic performance-rsi scorecard)")
		return 0
	}

	fmt.Fprintln(stdout, perfrsiscore.RenderHuman(report))
	return 0
}

func validateSelfcheck(report perfrsiscore.Report) error {
	if report.Schema != perfrsiscore.ReportSchema {
		return fmt.Errorf("schema = %q, want %q", report.Schema, perfrsiscore.ReportSchema)
	}

	canon := perfrsiscore.DimensionIDs()
	if len(report.Dimensions) != len(canon) {
		return fmt.Errorf("dimensions count = %d, want %d", len(report.Dimensions), len(canon))
	}

	seen := make(map[string]bool, len(canon))
	for _, d := range report.Dimensions {
		seen[d.ID] = true
	}
	for _, id := range canon {
		if !seen[id] {
			return fmt.Errorf("missing canonical dimension %q", id)
		}
	}

	if report.DominantBottleneck == "" {
		return fmt.Errorf("dominant bottleneck is empty")
	}
	if report.DominantBottleneck != "evaluation_latency" {
		return fmt.Errorf("dominant bottleneck = %q, want %q", report.DominantBottleneck, "evaluation_latency")
	}

	return nil
}
