package main

// hwspecs.go — the `fak hwspecs` inventory surface over pkg/hwspecs: list the
// cited hardware-spec catalog, resolve one card by key or human spelling, or
// match a live telemetry device name (nvidia-smi / rocm / benchmark receipt)
// to its cited spec. The facts live in pkg/hwspecs; this file is presentation
// and CLI grammar only.

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/hwspecs"
)

// hwSpecCell renders an optional numeric spec; 0 means "not cited", never zero
// hardware capability, so it prints as "-".
func hwSpecCell(v float64) string {
	if v == 0 {
		return "-"
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// emitHWSpec prints one resolved spec, as JSON or as text detail.
func emitHWSpec(sp hwspecs.DeviceSpec, asJSON bool) {
	if asJSON {
		out := struct {
			Schema string             `json:"schema"`
			Device hwspecs.DeviceSpec `json:"device"`
		}{Schema: hwspecs.Schema, Device: sp}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fmt.Fprintln(os.Stderr, "hwspecs:", err)
			os.Exit(1)
		}
		return
	}
	fmt.Printf("key: %s\nname: %s\nvendor: %s\narchitecture: %s\n", sp.Key, sp.Name, sp.Vendor, sp.Architecture)
	if sp.CUDAComputeMajor != 0 || sp.CUDAComputeMinor != 0 {
		fmt.Printf("cuda compute capability: sm_%d%d\n", sp.CUDAComputeMajor, sp.CUDAComputeMinor)
	}
	if sp.GPUArch != "" {
		fmt.Printf("gpu isa: %s\n", sp.GPUArch)
	}
	fmt.Printf("memory: %g GiB %s @ %g GB/s\n", sp.VRAMGiB, sp.MemoryType, sp.BandwidthGBps)
	fmt.Printf("dense tensor TFLOPS: fp8 %s / fp16 %s / fp4 %s (0/- = not cited)\n",
		hwSpecCell(sp.FP8TFLOPSDense), hwSpecCell(sp.FP16TFLOPSDense), hwSpecCell(sp.FP4TFLOPSDense))
	fmt.Printf("power: %s W   scale-up: %s GB/s\n", hwSpecCell(sp.PowerWatts), hwSpecCell(sp.NVLinkGBps))
	if len(sp.GPUNames) != 0 {
		fmt.Printf("telemetry names: %s\n", strings.Join(sp.GPUNames, "; "))
	}
	fmt.Printf("source: %s\n", sp.Source)
}

func cmdHWSpecs(args []string) {
	fs := flag.NewFlagSet("hwspecs", flag.ExitOnError)
	card := fs.String("card", "", "look up one card by catalog key or human spelling (e.g. h200, 'RTX PRO 6000')")
	match := fs.String("match", "", "resolve a live device name (nvidia-smi/rocm/receipt spelling) to its cited spec")
	jsonOut := fs.Bool("json", false, "emit versioned JSON")
	_ = fs.Parse(args)
	if *card != "" && *match != "" {
		fmt.Fprintln(os.Stderr, "hwspecs: --card and --match are mutually exclusive")
		os.Exit(2)
	}
	if *card != "" || *match != "" {
		query, kind := *card, "card"
		if *match != "" {
			query, kind = *match, "device name"
		}
		sp, ok := hwspecs.Lookup(query)
		if !ok {
			fmt.Fprintf(os.Stderr, "hwspecs: no cited spec for %s %q (catalog: %s)\n", kind, query, strings.Join(hwspecs.Keys(), ", "))
			os.Exit(2)
		}
		emitHWSpec(sp, *jsonOut)
		return
	}
	if *jsonOut {
		out := struct {
			Schema  string               `json:"schema"`
			Devices []hwspecs.DeviceSpec `json:"devices"`
		}{Schema: hwspecs.Schema, Devices: hwspecs.All()}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fmt.Fprintln(os.Stderr, "hwspecs:", err)
			os.Exit(1)
		}
		return
	}
	fmt.Printf("fak hardware-spec inventory (schema %s; cited vendor specs, not measurements)\n", hwspecs.Schema)
	fmt.Printf("%-16s %-52s %8s %9s %9s %9s %9s %6s\n", "KEY", "NAME", "VRAM-GIB", "GB/S", "FP8", "FP16", "FP4", "W")
	for _, sp := range hwspecs.All() {
		fmt.Printf("%-16s %-52s %8s %9s %9s %9s %9s %6s\n",
			sp.Key, sp.Name, hwSpecCell(sp.VRAMGiB), hwSpecCell(sp.BandwidthGBps),
			hwSpecCell(sp.FP8TFLOPSDense), hwSpecCell(sp.FP16TFLOPSDense), hwSpecCell(sp.FP4TFLOPSDense),
			hwSpecCell(sp.PowerWatts))
	}
	fmt.Println("one card: fak hwspecs --card <key|spelling>   telemetry name: --match <name>   machine-readable: --json")
}
