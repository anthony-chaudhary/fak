package main

import (
	"math"
	"runtime/debug"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// installFakeGoMemoryLimit swaps the serve memory-limit seams for an in-memory limit, so
// these tests never change the test process's own runtime limit.
func installFakeGoMemoryLimit(t *testing.T) *int64 {
	t.Helper()
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
	return &limit
}

// The fitplan limit bounds the model load's garbage; it is not a serving budget. Measured on
// the q2kxl H200 route, prefix and continuation snapshots grew the live heap to 39.5 GB over a
// 36.29 GiB load-peak limit and the GC ran back to back (C=8 decode 1.40 tok/s aggregate vs
// 40.5 with the limit lifted). The post-load step must retire the load-peak limit.
//
// fak-test:runtime fast est=2ms lane=default
func TestServePostLoadRetiresLoadPeakGoMemoryLimit(t *testing.T) {
	total, _, known := compute.HostSystemMemoryInfo()
	if !known || total <= 2<<30 {
		t.Skip("host RAM not probeable")
	}
	before := debug.SetMemoryLimit(-1)
	limit := installFakeGoMemoryLimit(t)
	env := hostPeakTestEnv(map[string]string{"FAK_MEMORY_LIMIT_FRACTION": "1"})

	const loadPeak = int64(1 << 30)
	applyServeGoMemoryLimit(serveGoMemoryLimitPlan{Target: loadPeak, Source: serveGoMemLimitSourceFitPlan})
	if *limit != loadPeak {
		t.Fatalf("fitplan load limit = %d, want %d", *limit, loadPeak)
	}
	msg := serveApplyHostFractionGoMemoryLimit(env)
	if *limit != total {
		t.Fatalf("post-load limit = %d, want the host-fraction target %d (msg %+v)", *limit, total, msg)
	}
	if msg.Kind != "go-memory-limit" || msg.Level != "info" {
		t.Fatalf("post-load message = %+v", msg)
	}
	if got := debug.SetMemoryLimit(-1); got != before {
		t.Fatalf("test process memory limit changed to %d", got)
	}
}

// A limit someone else lowered after the load (the streamed-expert peak cap, an operator) and
// an operator GOMEMLIMIT stay in force; only the serve's own load-peak limit is retired.
//
// fak-test:runtime fast est=2ms lane=default
func TestServePostLoadKeepsForeignGoMemoryLimit(t *testing.T) {
	total, _, known := compute.HostSystemMemoryInfo()
	if !known || total <= 2<<30 {
		t.Skip("host RAM not probeable")
	}
	limit := installFakeGoMemoryLimit(t)
	env := hostPeakTestEnv(map[string]string{"FAK_MEMORY_LIMIT_FRACTION": "1"})

	applyServeGoMemoryLimit(serveGoMemoryLimitPlan{Target: 1 << 30, Source: serveGoMemLimitSourceFitPlan})
	serveSetMemoryLimit(512 << 20) // a later, lower cap
	serveApplyHostFractionGoMemoryLimit(env)
	if *limit != 512<<20 {
		t.Fatalf("lowered cap replaced: limit = %d", *limit)
	}

	*limit = math.MaxInt64
	applyServeGoMemoryLimit(serveGoMemoryLimitPlan{Target: 1 << 30, Source: serveGoMemLimitSourceFitPlan})
	serveApplyHostFractionGoMemoryLimit(hostPeakTestEnv(map[string]string{"GOMEMLIMIT": "8GiB"}))
	if *limit != 1<<30 {
		t.Fatalf("operator GOMEMLIMIT plan changed the limit to %d", *limit)
	}
}
