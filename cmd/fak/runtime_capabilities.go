package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/runtimecap"
)

func runRuntimeCapabilities(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("runtime-capabilities", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "runtime-capabilities")
	backend := fs.String("backend", "", "require an exact registered backend name; unknown names never fall back")
	preferBackend := fs.String("prefer-backend", "", "prefer a backend; when it is unavailable, only the explicit local_cpu_degraded policy may select cpu-ref")
	fallbackPolicy := fs.String("fallback-policy", runtimecap.FallbackPolicyPinOrRefuse, "pin_or_refuse or local_cpu_degraded")
	cpuEnvelope := fs.String("cpu-envelope", "", "exact CPU fallback envelope id from supported_cpu_envelopes; evaluated before payload load")
	goos := fs.String("goos", "", "diagnostic override for operating system")
	goarch := fs.String("goarch", "", "diagnostic override for architecture")
	hostTotalRAMBytes := fs.Int64("host-total-ram-bytes", -1, "diagnostic host total RAM override for pre-load CPU-envelope admission")
	hostFreeRAMBytes := fs.Int64("host-free-ram-bytes", -1, "diagnostic host free RAM override for pre-load CPU-envelope admission")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak runtime-capabilities: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	if strings.TrimSpace(*backend) != "" && strings.TrimSpace(*preferBackend) != "" {
		fmt.Fprintln(stderr, "fak runtime-capabilities: --backend cannot be combined with --prefer-backend")
		return 2
	}
	if *fallbackPolicy != runtimecap.FallbackPolicyPinOrRefuse && *fallbackPolicy != runtimecap.FallbackPolicyLocalCPUDegrade {
		fmt.Fprintf(stderr, "fak runtime-capabilities: unsupported --fallback-policy %q\n", *fallbackPolicy)
		return 2
	}
	opts := runtimecap.Options{
		RequestedBackend:  *backend,
		PreferredBackend:  *preferBackend,
		CPUFallbackPolicy: *fallbackPolicy,
		CPUEnvelope:       *cpuEnvelope,
		GOOS:              *goos,
		GOARCH:            *goarch,
	}
	if *hostTotalRAMBytes >= 0 || *hostFreeRAMBytes >= 0 {
		opts.HostMemoryOverride = true
		opts.HostMemory.Known = true
		if *hostTotalRAMBytes >= 0 {
			opts.HostMemory.TotalBytes = *hostTotalRAMBytes
		}
		if *hostFreeRAMBytes >= 0 {
			opts.HostMemory.FreeKnown = true
			opts.HostMemory.FreeBytes = *hostFreeRAMBytes
		}
	}
	report := runtimecap.Probe(opts)
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(report); err != nil {
		fmt.Fprintf(stderr, "fak runtime-capabilities: encode: %v\n", err)
		return 1
	}
	return 0
}
