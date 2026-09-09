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

// RunMicroDoses executes the requested micro-dose hardware tests on physical silicon.
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

	report.AllPassed = allPassed
	report.TotalDurationMicros = time.Since(start).Microseconds()

	if hw.PhysicalAvailable {
		if allPassed {
			report.Guidance = fmt.Sprintf("Physical %s silicon active; all micro-doses passed in %d µs (<100ms). Bias toward hardware testing satisfied.", hw.Kind, report.TotalDurationMicros)
		} else {
			report.Guidance = fmt.Sprintf("Physical %s silicon active, but one or more micro-doses failed. Inspect hardware and driver health.", hw.Kind)
		}
	} else {
		report.Guidance = "No local physical accelerator detected. Micro-doses ran in host CPU reference mode; dispatch accelerator-gated work to sanctioned compute."
	}

	return report
}
