package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hil"
)

func cmdHIL(argv []string) {
	os.Exit(runHIL(os.Stdout, os.Stderr, argv))
}

func runHIL(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hil", flag.ContinueOnError)
	fs.SetOutput(stderr)

	asJSON := fs.Bool("json", false, "emit output as JSON envelope")
	probeOnly := fs.Bool("probe", false, "probe and report physical hardware capabilities only")
	lanProbe := fs.Bool("lan", false, "probe local silicon and LAN appliance node (AMD Strix Halo)")
	inventory := fs.Bool("inventory", false, "emit dynamic hardware fleet inventory (local + LAN appliance)")
	lanHost := fs.String("lan-host", "", "override LAN appliance host alias or address")
	auditComp := fs.Bool("audit-comparison", false, "audit a head-to-head comparison for physical hardware vs simulation")
	candName := fs.String("candidate", "", "candidate system name for comparison audit")
	candVal := fs.Float64("candidate-val", 0, "candidate metric value")
	candPhysical := fs.Bool("candidate-physical", true, "candidate was measured on physical silicon")
	candType := fs.String("candidate-type", "hardware_measurement", "candidate evidence type")
	candTarget := fs.String("candidate-target", "", "candidate hardware target")
	baseName := fs.String("baseline", "", "baseline system name for comparison audit")
	baseVal := fs.Float64("baseline-val", 0, "baseline metric value")
	basePhysical := fs.Bool("baseline-physical", true, "baseline was measured on physical silicon")
	baseType := fs.String("baseline-type", "hardware_measurement", "baseline evidence type")
	baseTarget := fs.String("baseline-target", "", "baseline hardware target")
	metric := fs.String("metric", "tok/s", "comparison metric name")
	lowerBetter := fs.Bool("lower-is-better", false, "lower metric value is better")

	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	// 1. Comparison Audit Mode
	if *auditComp || (*candName != "" && *baseName != "") {
		if *candName == "" || *baseName == "" || *candVal <= 0 || *baseVal <= 0 {
			fmt.Fprintln(stderr, "fak hil: comparison audit requires non-empty --candidate, --baseline, --candidate-val > 0, and --baseline-val > 0")
			return 2
		}

		cTarget := *candTarget
		if cTarget == "" {
			cTarget = "local-silicon"
		}
		bTarget := *baseTarget
		if bTarget == "" {
			bTarget = "reference-silicon"
		}

		cand := hil.ComparisonArm{
			Name:              *candName,
			IsPhysicalSilicon: *candPhysical,
			HardwareTarget:    cTarget,
			EvidenceType:      *candType,
			Metric:            *metric,
			Value:             *candVal,
			Unit:              *metric,
			SampleCount:       10,
		}
		base := hil.ComparisonArm{
			Name:              *baseName,
			IsPhysicalSilicon: *basePhysical,
			HardwareTarget:    bTarget,
			EvidenceType:      *baseType,
			Metric:            *metric,
			Value:             *baseVal,
			Unit:              *metric,
			SampleCount:       10,
		}

		audit := hil.AuditComparison(fmt.Sprintf("%s vs %s", *candName, *baseName), cand, base, *lowerBetter)
		if *asJSON {
			if err := writeIndentedJSON(stdout, audit); err != nil {
				fmt.Fprintf(stderr, "fak hil: write json: %v\n", err)
				return 1
			}
			return 0
		}

		fmt.Fprintf(stdout, "fak hil: Comparison Audit — %s\n", audit.Headline)
		fmt.Fprintf(stdout, "  Verdict:    %s\n", audit.Verdict)
		fmt.Fprintf(stdout, "  Speedup:    %.2fx\n", audit.Speedup)
		fmt.Fprintf(stdout, "  Candidate:  %s = %.2f %s (physical=%t, type=%s, target=%s)\n",
			cand.Name, cand.Value, cand.Unit, cand.IsPhysicalSilicon, cand.EvidenceType, cand.HardwareTarget)
		fmt.Fprintf(stdout, "  Baseline:   %s = %.2f %s (physical=%t, type=%s, target=%s)\n",
			base.Name, base.Value, base.Unit, base.IsPhysicalSilicon, base.EvidenceType, base.HardwareTarget)
		fmt.Fprintf(stdout, "  Reason:     %s\n", audit.Reason)
		fmt.Fprintf(stdout, "  Enforce:    %s\n", audit.EnforcementAction)
		if !audit.AllowedAsAchievedWin {
			return 1
		}
		return 0
	}

	// 2. Dynamic Hardware Inventory Mode
	if *inventory {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		inv := hil.ProbeInventory(ctx, *lanHost)
		if *asJSON {
			if err := writeIndentedJSON(stdout, inv); err != nil {
				fmt.Fprintf(stderr, "fak hil: write json: %v\n", err)
				return 1
			}
			return 0
		}
		fmt.Fprintf(stdout, "fak hil: Dynamic Hardware Inventory\n")
		fmt.Fprintf(stdout, "  Local Silicon:     %s (%s, physical=%t)\n",
			inv.Local.DeviceName, inv.Local.Kind, inv.Local.PhysicalAvailable)
		if inv.Local.MemoryTotalBytes > 0 {
			fmt.Fprintf(stdout, "  Local Memory:      %.2f GB (unified=%t)\n",
				float64(inv.Local.MemoryTotalBytes)/(1024*1024*1024), inv.Local.MemoryUnified)
		}
		fmt.Fprintf(stdout, "  LAN Node (%s):   [%s] reachable=%t transport=%s\n",
			inv.LAN.Host, inv.LAN.Status, inv.LAN.Reachable, inv.LAN.Transport)
		if inv.LAN.Appliance != "" {
			fmt.Fprintf(stdout, "    Appliance:       %s\n", inv.LAN.Appliance)
		}
		if inv.LAN.GPU != "" {
			fmt.Fprintf(stdout, "    GPU:             %s\n", inv.LAN.GPU)
		}
		if inv.LAN.LatencyMicros > 0 {
			fmt.Fprintf(stdout, "    Latency:         %.2f ms\n", float64(inv.LAN.LatencyMicros)/1000.0)
		}
		if inv.LAN.Error != "" {
			fmt.Fprintf(stdout, "    Notice:          %s\n", inv.LAN.Error)
		}
		fmt.Fprintf(stdout, "  Next Action:       %s\n", inv.NextAction)
		return 0
	}

	// 3. Probe Only Mode
	if *probeOnly {
		hw := hil.ProbeHardware()
		if *lanProbe {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			lan := hil.ProbeLANNode(ctx, *lanHost)
			hw.LANNode = &lan
		}
		if *asJSON {
			if err := writeIndentedJSON(stdout, hw); err != nil {
				fmt.Fprintf(stderr, "fak hil: write json: %v\n", err)
				return 1
			}
			return 0
		}
		fmt.Fprintf(stdout, "fak hil: Hardware Probe\n")
		fmt.Fprintf(stdout, "  Kind:              %s\n", hw.Kind)
		fmt.Fprintf(stdout, "  Device Name:       %s\n", hw.DeviceName)
		fmt.Fprintf(stdout, "  Platform/Arch:     %s/%s\n", hw.Platform, hw.Architecture)
		fmt.Fprintf(stdout, "  Physical Silicon:  %t\n", hw.PhysicalAvailable)
		fmt.Fprintf(stdout, "  Unified Memory:    %t\n", hw.MemoryUnified)
		if hw.MemoryTotalBytes > 0 {
			fmt.Fprintf(stdout, "  Total Memory:      %.2f GB\n", float64(hw.MemoryTotalBytes)/(1024*1024*1024))
		}
		for k, v := range hw.Details {
			fmt.Fprintf(stdout, "  %s: %s\n", k, v)
		}
		if hw.LANNode != nil {
			fmt.Fprintf(stdout, "  LAN Node (%s):   [%s] reachable=%t transport=%s\n",
				hw.LANNode.Host, hw.LANNode.Status, hw.LANNode.Reachable, hw.LANNode.Transport)
			if hw.LANNode.Appliance != "" {
				fmt.Fprintf(stdout, "    Appliance:       %s\n", hw.LANNode.Appliance)
			}
			if hw.LANNode.GPU != "" {
				fmt.Fprintf(stdout, "    GPU:             %s\n", hw.LANNode.GPU)
			}
		}
		return 0
	}

	// 4. Default Micro-Dose Execution Mode
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	report := hil.RunMicroDoses(ctx)

	if *asJSON {
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak hil: write json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "fak hil — Hardware-in-the-Loop Micro-Dose Suite\n")
		fmt.Fprintf(stdout, "  Silicon:           %s (%s, physical=%t)\n",
			report.Hardware.DeviceName, report.Hardware.Kind, report.Hardware.PhysicalAvailable)
		fmt.Fprintf(stdout, "  Total Duration:    %d µs (%.2f ms)\n",
			report.TotalDurationMicros, float64(report.TotalDurationMicros)/1000.0)
		fmt.Fprintf(stdout, "  Micro-Doses Ran:   %d\n\n", len(report.MicroDoses))

		for i, dose := range report.MicroDoses {
			status := "PASS"
			if !dose.Passed {
				status = "FAIL"
			}
			fmt.Fprintf(stdout, "  [%d] %-24s [%s] %s\n", i+1, dose.Name, status, dose.DurationFormatted)
			fmt.Fprintf(stdout, "      Detail: %s\n", dose.Detail)
			if dose.ComputeGFLOPS > 0 {
				fmt.Fprintf(stdout, "      Compute: %.2f GFLOP/s\n", dose.ComputeGFLOPS)
			}
			if dose.BandwidthGBs > 0 {
				fmt.Fprintf(stdout, "      Bandwidth: %.2f GB/s\n", dose.BandwidthGBs)
			}
		}

		fmt.Fprintf(stdout, "\nGuidance:\n  %s\n", report.Guidance)
	}

	if !report.AllPassed {
		return 1
	}
	return 0
}

const hilUsage = `fak hil — hardware-in-the-loop micro-dose execution and comparison audit.

usage:
  fak hil [--json]
  fak hil --probe [--lan] [--json]
  fak hil --inventory [--lan-host <host>] [--json]
  fak hil --audit-comparison --candidate <name> --candidate-val <val> --baseline <name> --baseline-val <val>

flags:
  --json                  emit canonical JSON envelope (fak.hil.report.v1, fak.hil.inventory.v1, or fak.hil.comparison.v1)
  --probe                 probe and report physical hardware capabilities only
  --lan                   probe local silicon and LAN appliance node (AMD Strix Halo)
  --inventory             emit dynamic hardware fleet inventory (local silicon + LAN appliance)
  --lan-host <host>       override LAN appliance host alias or address (default: strix1)
  --audit-comparison      audit head-to-head comparison (enforces real HW measurements vs simulations)
  --candidate <name>      candidate system name for comparison audit
  --candidate-val <val>   candidate metric value
  --candidate-physical    candidate measured on physical silicon (default: true)
  --candidate-type <type> candidate evidence type (default: hardware_measurement)
  --candidate-target <tg> candidate hardware target
  --baseline <name>       baseline system name for comparison audit
  --baseline-val <val>    baseline metric value
  --baseline-physical     baseline measured on physical silicon (default: true)
  --baseline-type <type>  baseline evidence type (default: hardware_measurement)
  --baseline-target <tg>  baseline hardware target
  --metric <name>         comparison metric name (default: tok/s)
  --lower-is-better       lower value is better (e.g. latency)
`
