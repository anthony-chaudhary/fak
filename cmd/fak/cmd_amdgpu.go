package main

import (
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func cmdAMDGPU(argv []string) {
	os.Exit(runAMDGPU(os.Stdout, os.Stderr, argv))
}

func runAMDGPU(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "help" || argv[0] == "-h" || argv[0] == "--help" {
		fmt.Fprintln(stdout, "Usage: fak amdgpu <subcommand> [flags]")
		fmt.Fprintln(stdout, "")
		fmt.Fprintln(stdout, "Subcommands:")
		fmt.Fprintln(stdout, "  roofline  Run RDNA 3.5 micro-roofline probe (coherent DRAM/MALL sweep, WMMA matrix)")
		fmt.Fprintln(stdout, "  gotchas   Audit host system against AMD Strix Halo known defects and gotchas")
		fmt.Fprintln(stdout, "  governor  Plan or apply DPM performance governor and TTM memory ceiling")
		fmt.Fprintln(stdout, "  sim       Simulate agentic workloads on AMD Strix Halo APUs")
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
	default:
		// If flags like --device=gfx1151 or --json are passed directly to `fak amdgpu`, default to roofline probe
		if len(argv[0]) > 0 && argv[0][0] == '-' {
			return amdgpu.RunRooflineProbeCLI(stdout, stderr, argv)
		}
		fmt.Fprintf(stderr, "fak amdgpu: unknown subcommand %q\n", argv[0])
		return 2
	}
}
