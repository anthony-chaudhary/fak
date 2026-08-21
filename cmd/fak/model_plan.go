package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelloadplan"
)

func runModelPlan(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	setup := fs.String("setup", "personal", "harness setup: personal, shared, or batch")
	goal := fs.String("goal", "balanced", "selection goal: balanced, quality, latency, or cost")
	local := fs.String("local", "auto", "local loading policy: auto, require, or disable")
	memory := fs.String("memory", "unified", "memory topology: unified or split")
	deviceGiB := fs.Float64("device-gib", 0, "usable device or unified memory in GiB (0 means unknown)")
	hostGiB := fs.Float64("host-gib", 0, "usable host memory in GiB (0 means unknown)")
	diskGiB := fs.Float64("disk-gib", 0, "usable model-cache disk in GiB (0 means unknown)")
	jsonOut := fs.Bool("json", false, "emit the versioned plan as JSON")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak model plan [qwen38:27b] [--setup personal|shared|batch] [--goal balanced|quality|latency|cost] [--local auto|require|disable] [--memory unified|split] [--device-gib N] [--host-gib N] [--disk-gib N] [--json]")
	}
	modelArg := ""
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		modelArg, args = strings.ToLower(args[0]), args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() > 0 || (modelArg != "" && !oneOfString(modelArg, "qwen38", "qwen38:27b", "qwen/qwen3.8-27b")) {
		fmt.Fprintln(stderr, "fak model plan: this spine supports qwen38:27b; pass no model for that default")
		return 2
	}
	toBytes := func(name string, n float64) (int64, bool) {
		if n < 0 {
			fmt.Fprintf(stderr, "fak model plan: --%s cannot be negative\n", name)
			return 0, false
		}
		return int64(n * (1 << 30)), true
	}
	device, ok := toBytes("device-gib", *deviceGiB)
	if !ok {
		return 2
	}
	host, ok := toBytes("host-gib", *hostGiB)
	if !ok {
		return 2
	}
	disk, ok := toBytes("disk-gib", *diskGiB)
	if !ok {
		return 2
	}
	plan, err := modelloadplan.Build(modelloadplan.Request{Setup: *setup, Goal: *goal, LocalPolicy: *local, Memory: *memory, DeviceBytes: device, HostBytes: host, DiskBytes: disk})
	if err != nil {
		fmt.Fprintf(stderr, "fak model plan: %v\n", err)
		return 2
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(plan); err != nil {
			fmt.Fprintf(stderr, "fak model plan: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "model: %s\nsetup: %s  goal: %s  local: %s  memory: %s\n", plan.Model, plan.Request.Setup, plan.Request.Goal, plan.Request.LocalPolicy, plan.Request.Memory)
	if plan.Selected == nil {
		fmt.Fprintln(stdout, "selected: NONE (no candidate fits the declared policy and capacities)")
	} else {
		fmt.Fprintf(stdout, "selected: %s", plan.Selected.ID)
		if plan.Selected.Quantization != "" {
			fmt.Fprintf(stdout, " (%s)", plan.Selected.Quantization)
		}
		fmt.Fprintf(stdout, "\nuri: %s\nnext: %s\n", plan.Selected.URI, plan.Selected.NextCommand)
	}
	fmt.Fprintln(stdout, "candidates:")
	for _, c := range plan.Candidates {
		verdict := "REJECT"
		if c.Fits {
			verdict = "FIT"
		}
		fmt.Fprintf(stdout, "  %-6s %-22s %s\n", verdict, c.ID, strings.Join(c.Reasons, "; "))
	}
	return 0
}

func oneOfString(v string, values ...string) bool {
	for _, x := range values {
		if v == x {
			return true
		}
	}
	return false
}
