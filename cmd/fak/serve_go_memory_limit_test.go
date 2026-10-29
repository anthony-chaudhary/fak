package main

import (
	"math"
	"os"
	"runtime/debug"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fak-test:runtime fast est=2ms lane=default
func TestPlanServeGoMemoryLimitSources(t *testing.T) {
	const gib = int64(1 << 30)
	hostTotal := 64 * gib
	defaultFraction := int64(float64(hostTotal) * ggufload.DefaultMemoryLimitFraction)
	peak := ggufload.HostLoadPeak{SteadyAnonBytes: 18 * gib, TransientBytes: 2 * gib, PeakAnonBytes: 20 * gib}
	cases := []struct {
		name      string
		peak      ggufload.HostLoadPeak
		peakKnown bool
		total     int64
		hostKnown bool
		env       map[string]string
		want      serveGoMemoryLimitPlan
	}{
		{"env wins", peak, true, 64 * gib, true, map[string]string{"GOMEMLIMIT": "off"}, serveGoMemoryLimitPlan{Source: serveGoMemLimitSourceEnv}},
		{"fitplan peak plus default margin", peak, true, 64 * gib, true, nil, serveGoMemoryLimitPlan{Target: 20*gib + ggufload.HostLoadPeakDefaultMarginBytes, Source: serveGoMemLimitSourceFitPlan}},
		{"fitplan peak plus override margin", peak, true, 0, false, map[string]string{ggufload.HostLoadPeakMarginEnv: "4"}, serveGoMemoryLimitPlan{Target: 24 * gib, Source: serveGoMemLimitSourceFitPlan}},
		{"host fraction default", ggufload.HostLoadPeak{}, false, 64 * gib, true, nil, serveGoMemoryLimitPlan{Target: defaultFraction, Source: serveGoMemLimitSourceHostFraction}},
		{"host fraction override", ggufload.HostLoadPeak{}, false, 64 * gib, true, map[string]string{"FAK_MEMORY_LIMIT_FRACTION": "0.5"}, serveGoMemoryLimitPlan{Target: 32 * gib, Source: serveGoMemLimitSourceHostFraction}},
		{"host fraction ignores bad override", ggufload.HostLoadPeak{}, false, 64 * gib, true, map[string]string{"FAK_MEMORY_LIMIT_FRACTION": "7"}, serveGoMemoryLimitPlan{Target: defaultFraction, Source: serveGoMemLimitSourceHostFraction}},
		{"unknown peak and host", ggufload.HostLoadPeak{}, false, 0, false, nil, serveGoMemoryLimitPlan{Source: serveGoMemLimitSourceNone}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := planServeGoMemoryLimit(tc.peak, tc.peakKnown, tc.total, tc.hostKnown, hostPeakTestEnv(tc.env))
			if got != tc.want {
				t.Fatalf("plan = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestApplyServeGoMemoryLimitInstallsFitPlanBeforeHostFraction(t *testing.T) {
	const gib = int64(1 << 30)
	before := debug.SetMemoryLimit(-1)
	limit := int64(math.MaxInt64)
	origSet, origConfigure := serveSetMemoryLimit, serveConfigureGoMemoryLimitBytes
	t.Cleanup(func() { serveSetMemoryLimit, serveConfigureGoMemoryLimitBytes = origSet, origConfigure })
	serveSetMemoryLimit = func(v int64) int64 {
		prev := limit
		if v >= 0 {
			limit = v
		}
		return prev
	}
	serveConfigureGoMemoryLimitBytes = func(target int64) (int64, bool) {
		if limit < math.MaxInt64 || target <= 0 {
			return limit, false
		}
		limit = target
		return target, true
	}

	msg := applyServeGoMemoryLimit(serveGoMemoryLimitPlan{Target: 22 * gib, Source: serveGoMemLimitSourceFitPlan})
	if limit != 22*gib || msg.Kind != "go-memory-limit" || msg.Level != "info" {
		t.Fatalf("fitplan apply: limit=%d msg=%+v", limit, msg)
	}
	// The serve-start host-fraction fallback must not raise the fitplan limit.
	msg = applyServeGoMemoryLimit(serveGoMemoryLimitPlan{Target: 54 * gib, Source: serveGoMemLimitSourceHostFraction})
	if limit != 22*gib || msg.Level != "info" {
		t.Fatalf("host-fraction fallback changed limit: limit=%d msg=%+v", limit, msg)
	}
	if msg = applyServeGoMemoryLimit(serveGoMemoryLimitPlan{Source: serveGoMemLimitSourceNone}); msg.Level != "warning" {
		t.Fatalf("none source level = %q", msg.Level)
	}
	if got := debug.SetMemoryLimit(-1); got != before {
		t.Fatalf("test process memory limit changed to %d", got)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestApplyServeGoMemoryLimitRealRuntimeRestores(t *testing.T) {
	restore := ggufload.PreserveMemoryLimit()
	t.Cleanup(restore)
	debug.SetMemoryLimit(math.MaxInt64)
	t.Setenv("GOMEMLIMIT", "")
	os.Unsetenv("GOMEMLIMIT") // LookupEnv treats an empty GOMEMLIMIT as explicit
	const target = int64(48 << 30)
	msg := applyServeGoMemoryLimit(serveGoMemoryLimitPlan{Target: target, Source: serveGoMemLimitSourceHostFraction})
	if got := debug.SetMemoryLimit(-1); got != target {
		t.Fatalf("runtime limit = %d, want %d (msg %+v)", got, target, msg)
	}
}
