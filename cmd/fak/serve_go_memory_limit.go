package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// Without a Go memory limit the GC paces off GOGC=100 alone, so heap_goal reaches about twice
// the live heap (a resident 27B serve: heap_goal 37 GiB over 18.7 GiB live). Serve therefore
// installs a limit: the predicted host load peak plus the admission margin when the loader has
// an estimate, else a fraction of host RAM. An operator GOMEMLIMIT (including GOMEMLIMIT=off)
// and any limit already in force (the streamed-expert route's predicted-peak limit) are kept.
const (
	serveGoMemLimitSourceEnv          = "env"
	serveGoMemLimitSourceFitPlan      = "fitplan"
	serveGoMemLimitSourceHostFraction = "host_fraction"
	serveGoMemLimitSourceNone         = "none"
)

type serveGoMemoryLimitPlan struct {
	Target int64
	Source string
}

// serveConfigureGoMemoryLimitBytes is ggufload.ConfigureHostMemoryLimitBytes, injectable so
// tests never lower the test process's own limit.
var serveConfigureGoMemoryLimitBytes = ggufload.ConfigureHostMemoryLimitBytes

func planServeGoMemoryLimit(peak ggufload.HostLoadPeak, peakKnown bool, hostTotal int64, hostKnown bool, getenv func(string) string) serveGoMemoryLimitPlan {
	if strings.TrimSpace(getenv("GOMEMLIMIT")) != "" {
		return serveGoMemoryLimitPlan{Source: serveGoMemLimitSourceEnv}
	}
	if peakKnown && peak.PeakAnonBytes > 0 {
		target := peak.PeakAnonBytes + serveHostPeakMarginBytes(getenv)
		if target < peak.PeakAnonBytes { // overflow
			target = math.MaxInt64 - 1
		}
		return serveGoMemoryLimitPlan{Target: target, Source: serveGoMemLimitSourceFitPlan}
	}
	if hostKnown && hostTotal > 0 {
		fraction := ggufload.DefaultMemoryLimitFraction
		if v := strings.TrimSpace(getenv("FAK_MEMORY_LIMIT_FRACTION")); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 && f <= 1 {
				fraction = f
			}
		}
		return serveGoMemoryLimitPlan{Target: int64(float64(hostTotal) * fraction), Source: serveGoMemLimitSourceHostFraction}
	}
	return serveGoMemoryLimitPlan{Source: serveGoMemLimitSourceNone}
}

func applyServeGoMemoryLimit(plan serveGoMemoryLimitPlan) gateway.StartupMessage {
	const gib = float64(1 << 30)
	msg := func(level, text string) gateway.StartupMessage {
		return newServeStartupMessage("go-runtime", "go-memory-limit", level, text)
	}
	switch plan.Source {
	case serveGoMemLimitSourceEnv:
		return msg("info", fmt.Sprintf("Go memory limit source=env: operator GOMEMLIMIT kept (%s)", serveGoMemLimitText(serveSetMemoryLimit(-1))))
	case serveGoMemLimitSourceNone:
		return msg("warning", "Go memory limit source=none: host RAM is not probeable and no load-peak estimate exists; heap_goal paces off GOGC alone")
	}
	applied, changed := serveConfigureGoMemoryLimitBytes(plan.Target)
	if !changed {
		return msg("info", fmt.Sprintf("Go memory limit already in force (%s) kept; %s target %.2f GiB not applied",
			serveGoMemLimitText(applied), plan.Source, float64(plan.Target)/gib))
	}
	return msg("info", fmt.Sprintf("Go memory limit source=%s: %.2f GiB", plan.Source, float64(applied)/gib))
}

func serveGoMemLimitText(limit int64) string {
	if limit <= 0 || limit == math.MaxInt64 {
		return "unlimited"
	}
	return fmt.Sprintf("%.2f GiB", float64(limit)/(1<<30))
}

// serveApplyHostFractionGoMemoryLimit is the serve-start fallback for load routes without a
// header peak estimate. It runs after the model load so a fitplan limit installed by the Q4_K
// device arm takes precedence.
func serveApplyHostFractionGoMemoryLimit(getenv func(string) string) gateway.StartupMessage {
	total, _, known := compute.HostSystemMemoryInfo()
	return applyServeGoMemoryLimit(planServeGoMemoryLimit(ggufload.HostLoadPeak{}, false, total, known, getenv))
}
