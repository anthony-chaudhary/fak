package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
	"github.com/anthony-chaudhary/fak/internal/containment"
)

func cmdAMDGPU(argv []string) {
	os.Exit(runAMDGPU(os.Stdout, os.Stderr, argv))
}

func runAMDGPU(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "help" || argv[0] == "-h" || argv[0] == "--help" {
		fmt.Fprintln(stdout, "Usage: fak amdgpu <subcommand> [flags]")
		fmt.Fprintln(stdout, "")
		fmt.Fprintln(stdout, "Subcommands:")
		fmt.Fprintln(stdout, "  roofline    Run RDNA 3.5 micro-roofline probe (coherent DRAM/MALL sweep, WMMA matrix)")
		fmt.Fprintln(stdout, "  gotchas     Audit host system against AMD Strix Halo known defects and gotchas")
		fmt.Fprintln(stdout, "  governor    Plan or apply DPM performance governor and TTM memory ceiling")
		fmt.Fprintln(stdout, "  sim         Simulate agentic workloads on AMD Strix Halo APUs")
		fmt.Fprintln(stdout, "  containment Arbitrate APU headroom and evaluate resource containment")
		return 0
	}

	switch argv[0] {
	case "roofline":
		return amdgpu.RunRooflineProbeCLI(stdout, stderr, argv[1:])
	case "gotchas":
		return amdgpu.RunGotchasCLI(stdout, stderr, argv[1:])
	case "governor":
		return amdgpu.RunCLI(stdout, stderr, argv[1:])
	case "sim":
		return amdgpu.RunSimCLI(stdout, stderr, argv[1:])
	case "containment":
		return runContainmentCLI(stdout, stderr, argv[1:])
	default:
		// If flags like --device=gfx1151 or --json are passed directly to `fak amdgpu`, default to roofline probe
		if len(argv[0]) > 0 && argv[0][0] == '-' {
			return amdgpu.RunRooflineProbeCLI(stdout, stderr, argv)
		}
		fmt.Fprintf(stderr, "fak amdgpu: unknown subcommand %q\n", argv[0])
		return 2
	}
}

func runContainmentCLI(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("containment", flag.ContinueOnError)
	fs.SetOutput(stderr)
	apertureGB := fs.Int64("aperture-gb", 120, "total APU unified memory aperture in GiB")
	allocFile := fs.String("allocations", "", "path to JSON file containing APUAllocationRecord array")
	asJSON := fs.Bool("json", false, "output arbitration result as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	var allocations []containment.APUAllocationRecord
	loadAlloc := func(raw string) error {
		var data []byte
		if strings.HasPrefix(strings.TrimSpace(raw), "[") {
			data = []byte(raw)
		} else {
			var err error
			data, err = os.ReadFile(raw)
			if err != nil {
				return err
			}
		}
		return json.Unmarshal(data, &allocations)
	}

	if *allocFile != "" {
		if err := loadAlloc(*allocFile); err != nil {
			fmt.Fprintf(stderr, "fak amdgpu containment: %v\n", err)
			return 1
		}
	} else if fs.NArg() > 0 {
		if err := loadAlloc(fs.Arg(0)); err != nil {
			fmt.Fprintf(stderr, "fak amdgpu containment: %v\n", err)
			return 1
		}
	}

	totalBytes := *apertureGB * 1024 * 1024 * 1024
	consumed, free, hasCap := containment.ArbitrateAPUHeadroom(totalBytes, allocations)

	if *asJSON {
		res := map[string]any{
			"total_aperture_bytes": totalBytes,
			"consumed_bytes":       consumed,
			"free_bytes":           free,
			"has_capacity":         hasCap,
			"allocations_count":    len(allocations),
		}
		b, _ := json.MarshalIndent(res, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintf(stdout, "APU Headroom Arbitration:\n")
		fmt.Fprintf(stdout, "  Total Aperture: %d bytes (%.2f GiB)\n", totalBytes, float64(totalBytes)/(1024*1024*1024))
		fmt.Fprintf(stdout, "  Consumed:       %d bytes (%.2f GiB)\n", consumed, float64(consumed)/(1024*1024*1024))
		fmt.Fprintf(stdout, "  Free:           %d bytes (%.2f GiB)\n", free, float64(free)/(1024*1024*1024))
		fmt.Fprintf(stdout, "  Has Capacity:   %v\n", hasCap)
		fmt.Fprintf(stdout, "  Allocations:    %d\n", len(allocations))
	}
	return 0
}
