package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
)

// serveAdmitInkling2Resident decides whether the device-resident Metal forward can
// serve the checkpoint; the selector has already refused a build without it.
func serveAdmitInkling2Resident(path string, shards int) error {
	if shards > 1 {
		return &inkling2MetalServeRefusal{Path: path,
			Reason: fmt.Sprintf("the %s path binds a single-file GGUF and this one is split into %d shards", inkling2MetalTargetResidentLabel, shards)}
	}
	return nil
}

// inkling2ResidentCheckpointBytes bounds the rewind buffers one resident session
// allocates on its first checkpoint: a conv history copy per layer visit plus a
// zero and a fill row, and K/V side rows for each local visit. A saved span never
// exceeds the sliding window, so the window bounds those rows.
func inkling2ResidentCheckpointBytes(s inkling2ResidentShape) int64 {
	var total int64
	alloc := func(floats int) {
		b := 4 * int64(floats)
		total += (b + inkling2ResidentBufferAlign - 1) / inkling2ResidentBufferAlign * inkling2ResidentBufferAlign
	}
	layers, core := len(s.Local), s.CoreEnd-s.CoreStart
	kvDim := s.KVHeads * s.HeadDim
	hist := (2*kvDim + 2*s.Hidden) * (s.ConvKernel - 1)
	rows := s.SlidingWindow * kvDim
	visits := layers + (s.Passes-1)*core
	for slot := 0; slot < visits; slot++ {
		alloc(hist)
		layer := slot
		if slot >= layers {
			layer = s.CoreStart + (slot-layers)%core
		}
		if s.Local[layer] {
			alloc(rows)
			alloc(rows)
		}
	}
	alloc(hist)
	alloc(hist)
	alloc(rows)
	return total
}

// inkling2ServeResidentSlots is how many resident sessions the serve engine may
// hold: one without reuse, else FAK_INKLING2_PREFIX_SLOTS bounded by how many
// sessions of perSlot bytes fit under FAK_INKLING2_PREFIX_CACHE_MIB, as the
// engine's cap meter admits a second slot only when it fits beside the first.
func inkling2ServeResidentSlots(lookup func(string) (string, bool), perSlot int64) int {
	slots, capBytes := inkling2PrefixCacheFromEnv(lookup)
	if !inkling2PrefixReuseFrom(lookup) || slots <= 1 || perSlot <= 0 {
		return 1
	}
	return int(max(1, min(int64(slots), capBytes/perSlot)))
}

// inkling2ServeResidentPlan prices weights plus the larger of the scalar prefix
// slots and cohort state, since scalar and cohort engines are serialized on the
// single GPU worker and runScalar releases cohort arenas first.
func inkling2ServeResidentPlan(weights localadmission.MemoryPlan, perSlot int64, slots int, cohort int64) (localadmission.MemoryPlan, error) {
	if perSlot < 0 || slots < 1 || cohort < 0 || perSlot > (math.MaxInt64-weights.SteadyBytes)/int64(slots) || cohort > math.MaxInt64-weights.SteadyBytes {
		return localadmission.MemoryPlan{}, fmt.Errorf("Inkling 2: resident serve plan overflow")
	}
	extra := int64(slots) * perSlot
	if cohort > extra {
		extra = cohort
	}
	plan := localadmission.MemoryPlan{SteadyBytes: weights.SteadyBytes + extra}
	plan.StartupPeakBytes, _ = metalServeStartupPeakBytes(serveLoadArmResidentQ4K, plan.SteadyBytes, 0, false)
	return plan, nil
}

// inkling2ServeResidentSizing is what serve admits for the resident path: the
// served context, the prefix slots the engine may hold, the cohort rows it may
// run (zero without cohorts), and their memory plan.
type inkling2ServeResidentSizing struct {
	Ctx, Slots, Rows int
	PerSlot, Cohort  int64
	Plan             localadmission.MemoryPlan
}

// serveInkling2ResidentCtx is the driver's FAK_INKLING2_CTX, refused where the
// admission mirror would clamp it below what the driver allocates.
func serveInkling2ResidentCtx(lookup func(string) (string, bool)) (int, error) {
	v, set := lookup(inkling2ResidentCtxEnv)
	if n, err := strconv.Atoi(strings.TrimSpace(v)); set && err == nil && n > inkling2ResidentCtxMax {
		return 0, fmt.Errorf("fak serve: %s=%d exceeds the %d-token context the %s path is priced for",
			inkling2ResidentCtxEnv, n, inkling2ResidentCtxMax, inkling2MetalTargetResidentLabel)
	}
	return inkling2ResidentCtx(v, set), nil
}

// inkling2ServeResidentRowsDefault is the resident Metal cohort width when
// FAK_INKLING2_SERVE_BATCH_ROWS is unset: every row holds KV at up to the served
// context on unified memory, so the forward's widest cohort would price far past
// what a laptop holds beside the weights.
const inkling2ServeResidentRowsDefault = 4

// serveInkling2ResidentRows is the cohort width the resident engine may run, or
// zero when FAK_INKLING2_SERVE_BATCH is off.
func serveInkling2ResidentRows(lookup func(string) (string, bool)) (int, error) {
	if !inkling2ServeBatchEnabled(lookup) {
		return 0, nil
	}
	if v, ok := lookup(inkling2ServeBatchRowsEnv); !ok || strings.TrimSpace(v) == "" {
		return inkling2ServeResidentRowsDefault, nil
	}
	return inkling2ServeBatchRows(lookup)
}

// inkling2ServeResidentCohortBytes prices the cohort engine beside the scalar
// one: its own weight binding (a resident session's share past its KV, as
// perSlot counts it), the widest state of rows rows at the served context, and
// the opt-in cross-request prefix store, which is host memory and so unified
// memory on Metal.
func inkling2ServeResidentCohortBytes(binding, state int64, lookup func(string) (string, bool)) int64 {
	total := binding + state
	if mib, on, err := ggufload.Inkling2BatchPrefixCacheMiB(nil, lookup); err == nil && on {
		total += int64(mib) << 20
	}
	return total
}

func serveInkling2ResidentSizingOf(path string, lookup func(string) (string, bool)) (inkling2ServeResidentSizing, error) {
	var sz inkling2ServeResidentSizing
	ctx, err := serveInkling2ResidentCtx(lookup)
	if err != nil {
		return sz, err
	}
	rows, err := serveInkling2ResidentRows(lookup)
	if err != nil {
		return sz, err
	}
	arch, err := serveInkling2Arch(path)
	if err != nil {
		return sz, err
	}
	f, err := ggufload.Open(path)
	if err != nil {
		return sz, err
	}
	weights, err := inkling2RunMetalMemory(path, f)
	if err != nil {
		return sz, err
	}
	one, err := inkling2RunResidentMemory(path, f, arch, ctx)
	if err != nil {
		return sz, err
	}
	shape, err := inkling2ResidentShapeOf(f, arch)
	if err != nil {
		return sz, err
	}
	perSlot := one.SteadyBytes - weights.SteadyBytes
	var cohort int64
	if rows > 0 {
		state, err := ggufload.Inkling2MetalCohortStateBytes(f, rows, ctx)
		if err != nil {
			return sz, err
		}
		cohort = inkling2ServeResidentCohortBytes(perSlot-inkling2ResidentSessionBytes(shape, ctx), state, lookup)
	}
	if inkling2PrefixReuseFrom(lookup) {
		perSlot += inkling2ResidentCheckpointBytes(shape)
	}
	slots := inkling2ServeResidentSlots(lookup, perSlot)
	plan, err := inkling2ServeResidentPlan(weights, perSlot, slots, cohort)
	if err != nil {
		return sz, err
	}
	return inkling2ServeResidentSizing{Ctx: ctx, Slots: slots, Rows: rows, PerSlot: perSlot, Cohort: cohort, Plan: plan}, nil
}

// inkling2ServeResidentPlacement names the resident path in the startup message.
func inkling2ServeResidentPlacement(arch string, layers, hidden, passes int, sz inkling2ServeResidentSizing) string {
	out := fmt.Sprintf("%s on metal target=%s: layers=%d hidden=%d passes=%d; prefix slots admitted=%d",
		arch, inkling2MetalTargetResidentLabel, layers, hidden, passes, sz.Slots)
	if sz.Rows > 0 {
		out += fmt.Sprintf("; cohort rows admitted=%d", sz.Rows)
	}
	return out
}

// serveInkling2ResidentLookup is lookup as the resident serve engines read it,
// held to what admission priced: the prefix slots, since a slot may hold less
// than its price when it opens and the cap meter alone could admit more, and the
// cohort width, since the cohort engine would otherwise run its widest.
func serveInkling2ResidentLookup(lookup func(string) (string, bool), sz inkling2ServeResidentSizing) func(string) (string, bool) {
	slots, rows := strconv.Itoa(sz.Slots), strconv.Itoa(sz.Rows)
	return func(key string) (string, bool) {
		switch {
		case key == "FAK_INKLING2_PREFIX_SLOTS":
			return slots, true
		case key == inkling2ServeBatchRowsEnv && sz.Rows > 0:
			return rows, true
		}
		return lookup(key)
	}
}
