package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func cmdAMDGPUFacts(argv []string) { os.Exit(runAMDGPUFacts(os.Stdout, os.Stderr, argv)) }

func runAMDGPUFacts(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-gpu-facts", flag.ContinueOnError)
	fs.SetOutput(stderr)
	name := fs.String("name", "", "only report a device whose name matches this substring")
	watch := fs.Float64("watch", 0, "sample every SEC seconds until interrupted")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *watch <= 0 {
		facts := amdgpu.Facts(*name, nil)
		data, _ := json.MarshalIndent(facts, "", "  ")
		fmt.Fprintln(stdout, string(data))
		if facts["available"] == true {
			return 0
		}
		return 1
	}
	interval := time.Duration(*watch * float64(time.Second))
	for {
		facts := amdgpu.Facts(*name, nil)
		if facts["available"] == true {
			fmt.Fprintf(stdout, "%s  util(total)=%6v%%  busiest=%v=%v%%  vram=%8v MiB  [%v]\n",
				time.Now().Format("15:04:05"), facts["total_util_pct"], facts["busiest_engine"], facts["busiest_util_pct"], facts["vram_used_mib"], facts["name"])
		} else {
			fmt.Fprintf(stdout, "%s  unavailable: %v\n", time.Now().Format("15:04:05"), facts["error"])
		}
		time.Sleep(interval)
	}
}
