package hil

import (
	"context"
	"fmt"
	"time"
)

// DefaultMicroDoses returns the standard suite of fast micro-dose hardware probes.
func DefaultMicroDoses() []DoseKind {
	return []DoseKind{
		DoseLiveness,
		DoseComputeGEMM,
		DoseBandwidth,
		DoseNumericParity,
	}
}

// RunMicroDoses executes the requested micro-doses using the platform runner.
// A host CPU reference result is not evidence of physical accelerator execution.
func RunMicroDoses(ctx context.Context, kinds ...DoseKind) Report {
	start := time.Now()
	hw := ProbeHardware()

	if len(kinds) == 0 {
		kinds = DefaultMicroDoses()
	}

	report := Report{
		Schema:             ReportSchema,
		Timestamp:          start.UTC(),
		Hardware:           hw,
		BiasTowardHardware: hw.PhysicalAvailable,
		MicroDoses:         make([]MicroDoseResult, 0, len(kinds)),
	}

	allPassed := true
	for _, kind := range kinds {
		if ctx.Err() != nil {
			break
		}
		res := runPlatformMicroDose(kind, hw)
		if !res.Passed {
			allPassed = false
		}
		report.MicroDoses = append(report.MicroDoses, res)
	}

	ctxErr := ctx.Err()
	report.AllPassed = len(report.MicroDoses) > 0 && len(report.MicroDoses) == len(kinds) && ctxErr == nil && allPassed
	report.TotalDurationMicros = time.Since(start).Microseconds()

	if ctxErr != nil {
		report.Guidance = fmt.Sprintf("Micro-dose suite canceled after %d of %d requested doses: %v. No complete passing suite is reported.", len(report.MicroDoses), len(kinds), ctxErr)
	} else if hw.PhysicalAvailable {
		if report.AllPassed {
			report.Guidance = fmt.Sprintf("Physical %s silicon active; all micro-doses passed in %d µs (<100ms). Bias toward hardware testing satisfied.", hw.Kind, report.TotalDurationMicros)
		} else {
			report.Guidance = fmt.Sprintf("Physical %s silicon active, but one or more micro-doses failed. Inspect hardware and driver health.", hw.Kind)
		}
	} else if report.AllPassed {
		report.Guidance = "All micro-doses passed in host CPU reference mode (reference only). No physical accelerator execution was verified; dispatch accelerator-gated work to sanctioned compute."
	} else {
		report.Guidance = "One or more host CPU reference micro-doses failed or did not complete. No physical accelerator execution was verified; inspect the reference results separately from accelerator availability."
	}

	return report
}
