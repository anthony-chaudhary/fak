package devcmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// RunAMDStrixValidate executes physical sub-kernel validation and ablation sweeps on the Strix Halo appliance.
func RunAMDStrixValidate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-strix-validate", flag.ContinueOnError)
	fs.SetOutput(stderr)

	host := fs.String("host", "", "target Strix Halo host (default: strix1 or FAK_STRIX_HOST)")
	subkernels := fs.String("subkernels", "all", "sub-kernels to test (comma-separated, 'all', or 'none')")
	ablate := fs.String("ablate", "all", "ablation arms to run (comma-separated, 'all', or 'none')")
	asJSON := fs.Bool("json", false, "emit receipt as JSON")
	timeoutSec := fs.Int("timeout", 45, "timeout in seconds")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	runSK := *subkernels != "none" && *subkernels != ""
	runAB := *ablate != "none" && *ablate != ""

	var skList []string
	if runSK && *subkernels != "all" {
		skList = strings.Split(*subkernels, ",")
	}

	var abList []string
	if runAB && *ablate != "all" {
		abList = strings.Split(*ablate, ",")
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutSec)*time.Second)
	defer cancel()

	opts := amdgpu.StrixValidationOpts{
		Host:          *host,
		RunSubkernels: runSK,
		Subkernels:    skList,
		RunAblations:  runAB,
		Ablations:     abList,
		GitRef:        "HEAD",
		Command:       "fak-dev amd-strix-validate " + strings.Join(argv, " "),
		Timeout:       time.Duration(*timeoutSec) * time.Second,
	}

	if !*asJSON {
		fmt.Fprintf(stderr, "==> Probing AMD Strix Halo appliance at %s...\n", *host)
	}

	receipt, err := amdgpu.RunStrixValidation(ctx, opts)
	if err != nil && (receipt == nil || receipt.Verdict == "FAIL") {
		if receipt != nil && *asJSON {
			data, _ := json.MarshalIndent(receipt, "", "  ")
			fmt.Fprintln(stdout, string(data))
		}
		fmt.Fprintf(stderr, "amd-strix-validate: validation failed: %v\n", err)
		return 1
	}

	if *asJSON {
		data, _ := json.MarshalIndent(receipt, "", "  ")
		fmt.Fprintln(stdout, string(data))
		if receipt.Verdict == "PASS" {
			return 0
		}
		return 1
	}

	// Render human-readable summary
	fmt.Fprintf(stdout, "\n================================================================================\n")
	fmt.Fprintf(stdout, "AMD Strix Halo Physical Validation Receipt\n")
	fmt.Fprintf(stdout, "================================================================================\n")
	fmt.Fprintf(stdout, "Verdict:     %s\n", receipt.Verdict)
	fmt.Fprintf(stdout, "Target:      %s (%s)\n", receipt.Target.Host, receipt.Target.Mode)
	fmt.Fprintf(stdout, "CPU Model:   %s\n", receipt.Target.CPUModel)
	fmt.Fprintf(stdout, "GPU Model:   %s (%s, %d CUs)\n", receipt.Target.GPUName, receipt.Target.TargetISA, receipt.Target.ComputeUnits)
	fmt.Fprintf(stdout, "Memory:      %.1f GiB UMA Aperture (Total: %.1f GiB)\n",
		float64(receipt.Target.UMABufferBytes)/(1024*1024*1024),
		float64(receipt.Target.TotalRAMBytes)/(1024*1024*1024))
	fmt.Fprintf(stdout, "DPM Level:   %s (Watchdog: %d)\n", receipt.Target.DPMLevel, receipt.Target.LockupTimeout)
	fmt.Fprintf(stdout, "Digest:      %s\n", receipt.Digest)
	fmt.Fprintf(stdout, "--------------------------------------------------------------------------------\n")
	fmt.Fprintf(stdout, "Sub-Kernels Tested (%d):\n", len(receipt.Subkernels))
	for _, sk := range receipt.Subkernels {
		parity := "PASSED"
		if !sk.Parity.Passed {
			parity = "FAIL"
		}
		fmt.Fprintf(stdout, "  * %-25s [%s] %5d µs  (parity: %s, cosine: %.6f)\n",
			sk.Name, sk.Status, sk.DurationUS, parity, sk.Parity.LogitCosineSimilarity)
	}
	fmt.Fprintf(stdout, "--------------------------------------------------------------------------------\n")
	fmt.Fprintf(stdout, "Ablation Arms Evaluated (%d):\n", len(receipt.Ablations))
	for _, ab := range receipt.Ablations {
		fmt.Fprintf(stdout, "  * %-28s [%s] speedup: %6.1fx (baseline: %d µs, candidate: %d µs)\n",
			ab.Feature, ab.Verdict, ab.Speedup, ab.BaselineArm.LatencyUS, ab.CandidateArm.LatencyUS)
	}
	fmt.Fprintf(stdout, "================================================================================\n\n")

	if receipt.Verdict == "PASS" {
		return 0
	}
	return 1
}

// RunAMDStrixProbe inspects and reports live hardware facts from the Strix Halo appliance.
func RunAMDStrixProbe(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-strix-probe", flag.ContinueOnError)
	fs.SetOutput(stderr)

	host := fs.String("host", "", "target Strix Halo host (default: strix1 or FAK_STRIX_HOST)")
	asJSON := fs.Bool("json", false, "emit facts as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	target, err := amdgpu.DiscoverStrixTarget(ctx, *host)
	if err != nil || target == nil || !target.Reachable {
		fmt.Fprintf(stderr, "amd-strix-probe: cannot reach Strix Halo appliance: %v\n", err)
		return 1
	}

	if *asJSON {
		data, _ := json.MarshalIndent(target, "", "  ")
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	fmt.Fprintf(stdout, "Strix Halo Appliance Status:\n")
	fmt.Fprintf(stdout, "  Host:         %s (%s, latency: %.1f ms)\n", target.Host, target.Mode, target.LatencyMS)
	fmt.Fprintf(stdout, "  CPU:          %s\n", target.CPUModel)
	fmt.Fprintf(stdout, "  GPU:          %s (%s, %d CUs)\n", target.GPUName, target.TargetISA, target.ComputeUnits)
	fmt.Fprintf(stdout, "  RAM:          %.1f GiB UMA (Total: %.1f GiB)\n",
		float64(target.UMABufferBytes)/(1024*1024*1024),
		float64(target.TotalRAMBytes)/(1024*1024*1024))
	fmt.Fprintf(stdout, "  DPM:          %s\n", target.DPMLevel)
	fmt.Fprintf(stdout, "  Lockup TO:    %d\n", target.LockupTimeout)
	fmt.Fprintf(stdout, "  Vulkan ICD:   %s\n", target.VulkanICD)
	return 0
}
