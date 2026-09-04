package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mtptune"
)

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("tunemtp", flag.ContinueOnError)
	fs.SetOutput(stderr)

	kMin := fs.Int("k-min", 1, "Minimum draft depth K")
	kMax := fs.Int("k-max", 8, "Maximum draft depth K")
	pMin := fs.Float64("p-min", 0.0, "Minimum threshold P")
	pMax := fs.Float64("p-max", 1.0, "Maximum threshold P")
	pStep := fs.Float64("p-step", 0.2, "Threshold step size")
	tasksFlag := fs.String("tasks", "Code,Math,JSON", "Comma-separated task categories to sweep")
	busBandwidth := fs.Float64("bus-bandwidth", 200.0, "Bus bandwidth in GB/s (e.g. 200 for 256-bit LPDDR5X)")
	modelWeight := fs.Float64("model-weight", 13.55, "Model weight in GB (e.g. 13.55 for Qwen 3.8 27B ROCmFP4)")
	mtpHeadWeight := fs.Float64("mtp-head-weight", 0.45, "MTP draft head weights in GB")
	kvTraffic := fs.Float64("kv-traffic", 0.08, "KV cache traffic per token in GB")
	baseCompute := fs.Float64("base-compute", 4.0, "Base verification compute time in ms")
	draftCompute := fs.Float64("draft-compute", 1.5, "Per-draft compute time in ms")
	jsonOutput := fs.Bool("json", false, "Output results in JSON format")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	var tasks []mtptune.TaskCategory
	for _, t := range strings.Split(*tasksFlag, ",") {
		name := strings.TrimSpace(t)
		if name != "" {
			tasks = append(tasks, mtptune.TaskCategory(name))
		}
	}
	if len(tasks) == 0 {
		tasks = mtptune.AllTasks()
	}

	cfg := mtptune.SweepConfig{
		KMin:               *kMin,
		KMax:               *kMax,
		PMin:               *pMin,
		PMax:               *pMax,
		PStep:              *pStep,
		Tasks:              tasks,
		BusBandwidthGBs:    *busBandwidth,
		ModelWeightGB:      *modelWeight,
		MTPHeadWeightGB:    *mtpHeadWeight,
		KVTrafficPerTokGB:  *kvTraffic,
		BaseComputeMs:      *baseCompute,
		DraftStepComputeMs: *draftCompute,
	}

	report, err := mtptune.RunSweep(cfg)
	if err != nil {
		fmt.Fprintf(stderr, "Error running MTP tuning sweep: %v\n", err)
		return 1
	}

	if *jsonOutput {
		data, err := mtptune.FormatReportJSON(report)
		if err != nil {
			fmt.Fprintf(stderr, "Error formatting JSON report: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprint(stdout, mtptune.FormatReportTable(report))
	}
	return 0
}
