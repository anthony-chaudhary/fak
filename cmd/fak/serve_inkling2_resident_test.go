package main

import (
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
)

func inkling2EnvLookup(env map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := env[k]; return v, ok }
}

func TestServeInkling2MetalSelectorAdmitsResident(t *testing.T) {
	path := writeInkling2HeaderGGUF(t, serveInkling2LoopArch)
	kind, err := serveInkling2MetalSelected(path, " Resident ")
	if !inkling2MetalResidentAvailable {
		if err == nil || !strings.Contains(err.Error(), "not available in this build") {
			t.Fatalf("resident on a build without it: %v, want the build refusal", err)
		}
		return
	}
	if err != nil || kind != inkling2MetalResident {
		t.Fatalf("resident: kind=%v err=%v, want the resident path", kind, err)
	}
	for _, selector := range []string{"", "vendored"} {
		if kind, err := serveInkling2MetalSelected(path, selector); err != nil || kind != inkling2MetalVendored {
			t.Fatalf("%q: kind=%v err=%v, want the vendored path unchanged", selector, kind, err)
		}
	}
	t.Setenv(inkling2MetalTargetEnv, "resident")
	routed, kind, err := serveInkling2MetalPathOf(path, nil, true)
	if !routed || kind != inkling2MetalResident || err != nil {
		t.Fatalf("route: routed=%v kind=%v err=%v", routed, kind, err)
	}
	// a header-only GGUF has no shape, so the resident sizing, not the vendored
	// estimate, refuses it
	if plan, err := serveInkling2MetalWeights(path, nil, true); plan != nil || err == nil || !strings.Contains(err.Error(), "lacks") {
		t.Fatalf("resident weights = %v, %v, want the resident shape refusal", plan, err)
	}
	if routed, _, err := serveInkling2MetalPathOf(path, nil, false); routed || err != nil {
		t.Fatalf("resident without Metal: routed=%v err=%v", routed, err)
	}
}

func TestServeInkling2ResidentSplitRefused(t *testing.T) {
	var refusal *inkling2MetalServeRefusal
	if err := serveAdmitInkling2Resident("m.gguf", 2); !errors.As(err, &refusal) || !strings.Contains(err.Error(), inkling2MetalTargetResidentLabel) {
		t.Fatalf("split resident = %v, want the typed refusal naming the resident path", err)
	}
	if err := serveAdmitInkling2Resident("m.gguf", 1); err != nil {
		t.Fatalf("single-file resident refused: %v", err)
	}
}

func TestServeInkling2ResidentSlots(t *testing.T) {
	const perSlot = 300 << 20
	for _, tc := range []struct {
		env  map[string]string
		want int
	}{
		{nil, 1},
		{map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "4"}, 4},
		{map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "8", "FAK_INKLING2_PREFIX_CACHE_MIB": "1000"}, 3},
		{map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "4", "FAK_INKLING2_PREFIX_CACHE_MIB": "100"}, 1},
		{map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "4", "FAK_INKLING2_PREFIX_REUSE": "off"}, 1},
		{map[string]string{"FAK_INKLING2_PREFIX_SLOTS": "99"}, inkling2PrefixSlotsMax},
	} {
		if got := inkling2ServeResidentSlots(inkling2EnvLookup(tc.env), perSlot); got != tc.want {
			t.Errorf("%v: slots %d, want %d", tc.env, got, tc.want)
		}
	}
}

func TestServeInkling2ResidentPlanScalesWithSlots(t *testing.T) {
	weights := localadmission.MemoryPlan{SteadyBytes: 10 << 30}
	const perSlot = 512 << 20
	one, err := inkling2ServeResidentPlan(weights, perSlot, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	three, err := inkling2ServeResidentPlan(weights, perSlot, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	if one.SteadyBytes != weights.SteadyBytes+perSlot || three.SteadyBytes-one.SteadyBytes != 2*perSlot {
		t.Fatalf("steady one=%d three=%d", one.SteadyBytes, three.SteadyBytes)
	}
	if three.StartupPeakBytes < three.SteadyBytes || three.StartupPeakBytes <= one.StartupPeakBytes {
		t.Fatalf("startup peak one=%d three=%d below or flat against steady", one.StartupPeakBytes, three.StartupPeakBytes)
	}
	if _, err := inkling2ServeResidentPlan(weights, 1<<62, 8, 0); err == nil {
		t.Fatal("overflowing plan accepted")
	}
}

func TestServeInkling2ResidentCheckpointBytes(t *testing.T) {
	// Six visits, two of them to the local layer 1 and one to layer 3.
	s := inkling2ResidentShape{Hidden: 8, Heads: 4, KVHeads: 2, HeadDim: 4, SlidingWindow: 16, ConvKernel: 3,
		CoreStart: 1, CoreEnd: 3, Passes: 2, Local: []bool{false, true, false, true}}
	page := func(floats int64) int64 { return (4*floats + 16383) / 16384 * 16384 }
	hist, rows := page((2*8+2*8)*2), page(16*8)
	want := 8*hist + 3*2*rows + rows
	if got := inkling2ResidentCheckpointBytes(s); got != want {
		t.Fatalf("checkpoint bytes %d, want %d", got, want)
	}
}

func TestServeInkling2ResidentCtxRefusesUnpricedContext(t *testing.T) {
	if ctx, err := serveInkling2ResidentCtx(inkling2EnvLookup(nil)); err != nil || ctx != inkling2ResidentCtxDefault {
		t.Fatalf("unset ctx = %d, %v", ctx, err)
	}
	if ctx, err := serveInkling2ResidentCtx(inkling2EnvLookup(map[string]string{inkling2ResidentCtxEnv: "8192"})); err != nil || ctx != 8192 {
		t.Fatalf("8192 ctx = %d, %v", ctx, err)
	}
	atMax := strconv.Itoa(ggufload.Inkling2ResidentCtxMax("metal"))
	if ctx, err := serveInkling2ResidentCtx(inkling2EnvLookup(map[string]string{inkling2ResidentCtxEnv: atMax})); err != nil || ctx != ggufload.Inkling2ResidentCtxMax("metal") {
		t.Fatalf("%s ctx = %d, %v", atMax, ctx, err)
	}
	overMax := strconv.Itoa(ggufload.Inkling2ResidentCtxMax("metal") + 1)
	if _, err := serveInkling2ResidentCtx(inkling2EnvLookup(map[string]string{inkling2ResidentCtxEnv: overMax})); err == nil {
		t.Fatal("a context above the priced maximum was admitted")
	}
}

func TestServeInkling2ResidentPlacementLabel(t *testing.T) {
	got := inkling2ServeResidentPlacement(serveInkling2LoopArch, 24, 2048, 3, inkling2ServeResidentSizing{Slots: 2, Rows: 4})
	if !strings.Contains(got, "target="+inkling2MetalTargetResidentLabel) || strings.Contains(got, inkling2MetalTargetVendoredLabel) ||
		!strings.Contains(got, "prefix slots admitted=2") || !strings.Contains(got, "cohort rows admitted=4") {
		t.Fatalf("placement %q does not name the resident path honestly", got)
	}
}

func TestServeInkling2BatchRowsBound(t *testing.T) {
	if n, err := inkling2ServeBatchRows(inkling2EnvLookup(nil)); err != nil || n != ggufload.Inkling2LoopMaxBatch {
		t.Fatalf("unset rows = %d, %v, want the forward's widest cohort", n, err)
	}
	if n, err := inkling2ServeBatchRows(inkling2EnvLookup(map[string]string{inkling2ServeBatchRowsEnv: " 3 "})); err != nil || n != 3 {
		t.Fatalf("rows 3 = %d, %v", n, err)
	}
	for _, bad := range []string{"0", "-1", "x", strconv.Itoa(ggufload.Inkling2LoopMaxBatch + 1)} {
		if _, err := inkling2ServeBatchRows(inkling2EnvLookup(map[string]string{inkling2ServeBatchRowsEnv: bad})); err == nil {
			t.Errorf("rows %q accepted", bad)
		}
	}
	adm, err := inkling2AdmissionFromEnv(inkling2EnvLookup(nil))
	if err != nil || adm.max != ggufload.Inkling2LoopMaxBatch {
		t.Fatalf("default admission max = %d, %v; the CUDA cohort width must not change", adm.max, err)
	}
}

func TestServeInkling2ResidentRowsDefault(t *testing.T) {
	on := map[string]string{"FAK_INKLING2_SERVE_BATCH": "1"}
	for _, tc := range []struct {
		env  map[string]string
		want int
	}{
		{nil, 0},
		{map[string]string{inkling2ServeBatchRowsEnv: "8"}, 0},
		{on, inkling2ServeResidentRowsDefault},
		{map[string]string{"FAK_INKLING2_SERVE_BATCH": "1", inkling2ServeBatchRowsEnv: "2"}, 2},
	} {
		if got, err := serveInkling2ResidentRows(inkling2EnvLookup(tc.env)); err != nil || got != tc.want {
			t.Errorf("%v: rows %d, %v, want %d", tc.env, got, err, tc.want)
		}
	}
	if _, err := serveInkling2ResidentRows(inkling2EnvLookup(map[string]string{"FAK_INKLING2_SERVE_BATCH": "1", inkling2ServeBatchRowsEnv: "99"})); err == nil {
		t.Fatal("an out-of-range resident cohort width was admitted")
	}
}

func TestServeInkling2ResidentLookupHoldsEngineToPlan(t *testing.T) {
	env := inkling2EnvLookup(map[string]string{"FAK_INKLING2_SERVE_BATCH": "1", "FAK_INKLING2_PREFIX_SLOTS": "8"})
	rows, err := serveInkling2ResidentRows(env)
	if err != nil {
		t.Fatal(err)
	}
	sz := inkling2ServeResidentSizing{Slots: 2, Rows: rows}
	look := serveInkling2ResidentLookup(env, sz)
	adm, err := inkling2AdmissionFromEnv(look)
	if err != nil || adm.max != sz.Rows {
		t.Fatalf("engine cohort width %d (%v), priced %d", adm.max, err, sz.Rows)
	}
	if slots, _ := inkling2PrefixCacheFromEnv(look); slots != sz.Slots {
		t.Fatalf("engine prefix slots %d, priced %d", slots, sz.Slots)
	}
	if v, ok := serveInkling2ResidentLookup(env, inkling2ServeResidentSizing{Slots: 1})(inkling2ServeBatchRowsEnv); ok {
		t.Fatalf("rows %q injected with cohorts off", v)
	}
}

func TestServeInkling2ResidentPlanPricesCohort(t *testing.T) {
	weights := localadmission.MemoryPlan{SteadyBytes: 10 << 30}
	const perSlot, cohort = 512 << 20, 3 << 30
	scalar, err := inkling2ServeResidentPlan(weights, perSlot, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	batched, err := inkling2ServeResidentPlan(weights, perSlot, 2, cohort)
	if err != nil {
		t.Fatal(err)
	}
	// scalar and cohort engines are serialized, so steady bytes take max(slots*perSlot, cohort)
	if batched.SteadyBytes != weights.SteadyBytes+cohort || batched.StartupPeakBytes <= scalar.StartupPeakBytes {
		t.Fatalf("cohort not priced as max(slots*perSlot, cohort): scalar=%+v batched=%+v", scalar, batched)
	}
	// when slots*perSlot exceeds cohort, steady bytes stay at weights + slots*perSlot
	smallerCohort, err := inkling2ServeResidentPlan(weights, perSlot, 2, 256<<20)
	if err != nil {
		t.Fatal(err)
	}
	if smallerCohort.SteadyBytes != scalar.SteadyBytes {
		t.Fatalf("smaller cohort double-priced: scalar=%+v smallerCohort=%+v", scalar, smallerCohort)
	}
	if got := inkling2ServeResidentCohortBytes(100, 200, inkling2EnvLookup(map[string]string{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "64"})); got != 300+64<<20 {
		t.Fatalf("cohort bytes with a prefix store = %d", got)
	}
	if got := inkling2ServeResidentCohortBytes(100, 200, inkling2EnvLookup(nil)); got != 300 {
		t.Fatalf("cohort bytes = %d", got)
	}
}
