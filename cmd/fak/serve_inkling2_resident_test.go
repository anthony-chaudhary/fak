package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/memgate"
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
	if kind, err := serveInkling2MetalSelected(path, ""); err != nil || kind != inkling2MetalResident {
		t.Fatalf("unset: kind=%v err=%v, want the resident path", kind, err)
	}
	if _, err := serveInkling2MetalSelected(path, "vendored"); err == nil || !strings.Contains(err.Error(), "removed") {
		t.Fatalf("vendored: err=%v, want removed refusal", err)
	}
	t.Setenv(inkling2MetalTargetEnv, "resident")
	routed, kind, err := serveInkling2MetalPathOf(path, nil, true)
	if !routed || kind != inkling2MetalResident || err != nil {
		t.Fatalf("route: routed=%v kind=%v err=%v", routed, kind, err)
	}
	// a header-only GGUF has no shape, so the resident sizing refuses it
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
	if !strings.Contains(got, "target="+inkling2MetalTargetResidentLabel) ||
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

func writeInkling2ResidentSizingGGUF(t *testing.T, totalBytes int64) string {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	kvStr := func(k, v string) { str(k); w(uint32(8)); str(v) }
	kvU32 := func(k string, v uint32) { str(k); w(uint32(4)); w(v) }
	kvF32 := func(k string, v float32) { str(k); w(uint32(6)); w(v) }
	kvBoolArr := func(k string, vs []bool) {
		str(k)
		w(uint32(9)) // GGUF_TYPE_ARRAY
		w(uint32(7)) // GGUF_TYPE_BOOL
		w(uint64(len(vs)))
		for _, x := range vs {
			if x {
				b.WriteByte(1)
			} else {
				b.WriteByte(0)
			}
		}
	}

	const arch = serveInkling2LoopArch
	p := arch + "."
	// 24 layers, 20 local + 4 global, core [4,20) x 2 passes = 40 slots total.
	local := make([]bool, 24)
	for i := range local {
		local[i] = i%6 != 5
	}

	b.WriteString("GGUF")
	w(uint32(3))  // version
	w(uint64(1))  // tensor count: output.weight
	w(uint64(15)) // kv count
	kvStr("general.architecture", arch)
	kvU32(p+"block_count", 24)
	kvU32(p+"embedding_length", 3072)
	kvU32(p+"vocab_size_unpadded", 32000)
	kvU32(p+"feed_forward_length", 4096)
	kvU32(p+"expert_count", 128)
	kvU32(p+"expert_used_count", 4)
	kvU32(p+"expert_feed_forward_length", 768)
	kvU32(p+"attention.head_count", 32)
	kvU32(p+"attention.head_count_kv", 4)
	kvU32(p+"attention.key_length", 128)
	kvU32(p+"attention.sliding_window", 1024)
	kvU32(p+"attention.relative_length", 16)
	kvU32(p+"attention.relative_extent", 64)
	kvU32(p+"shortconv.l_cache", 4)
	kvU32(p+"attn_res.block_size", 4)
	kvU32(p+"loop.start", 4)
	kvU32(p+"loop.end", 20)
	kvU32(p+"loop.count", 2)
	kvU32(p+"loop.gate_rank", 128)
	kvU32(p+"attention.log_scale_floor", 256)
	kvF32(p+"attention.layer_norm_rms_epsilon", 1e-6)
	kvF32(p+"attention.qk_clip", 10.0)
	kvF32(p+"attention.relative_clip", 10.0)
	kvF32(p+"attention.log_scale_alpha", 0.1)
	kvF32(p+"activation.situ_beta", 1.0)
	kvF32(p+"activation.situ_linear_beta", 0.5)
	kvBoolArr(p+"attention.sliding_window_pattern", local)
	// Fix kv count to 28.
	raw := b.Bytes()
	binary.LittleEndian.PutUint64(raw[16:24], 28)

	// Tensor info: output.weight [3072, 32000] F32 at offset 0.
	str("output.weight")
	w(uint32(2))
	w(uint64(3072))
	w(uint64(32000))
	w(uint32(0)) // GGML_TYPE_F32
	w(uint64(0))

	path := filepath.Join(t.TempDir(), "inkling2_sizing.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(b.Bytes()); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if totalBytes > int64(b.Len()) {
		if err := f.Truncate(totalBytes); err != nil {
			_ = f.Close()
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestServeInkling2ResidentUnifiedMemorySizing24GBAnd36GB(t *testing.T) {
	// ~15 GiB weights file so 24 GB (18 GiB GPU share) has ~3 GiB headroom
	// and 36 GB (27 GiB GPU share) has ~12 GiB headroom.
	path := writeInkling2ResidentSizingGGUF(t, 15<<30)
	origRead := serveReadMemory
	t.Cleanup(func() { serveReadMemory = origRead })

	// 24 GB Mac: 18 GiB GPU share. 16k context with default f16 KV must step
	// down prefill chunk from 2048 and admit cleanly within 18 GiB.
	serveReadMemory = func() (memgate.Memory, error) {
		return memgate.Memory{TotalBytes: 24 << 30, AvailableBytes: 20 << 30}, nil
	}
	sz24, err := serveInkling2ResidentSizingOf(path, inkling2EnvLookup(map[string]string{
		inkling2ResidentCtxEnv: "16384",
	}))
	if err != nil {
		t.Fatalf("24 GB sizing err: %v", err)
	}
	if sz24.Ctx != 16384 {
		t.Fatalf("24 GB ctx = %d, want 16384", sz24.Ctx)
	}
	if sz24.Chunk >= 2048 || (sz24.Chunk != 1024 && sz24.Chunk != 512) {
		t.Fatalf("24 GB chunk = %d, want stepped-down 1024 or 512", sz24.Chunk)
	}
	if sz24.Plan.SteadyBytes > 18<<30 {
		t.Fatalf("24 GB steady bytes = %d (> 18 GiB GPU share)", sz24.Plan.SteadyBytes)
	}
	look24 := serveInkling2ResidentLookup(inkling2EnvLookup(nil), sz24)
	if v, ok := look24("FAK_INKLING2_PREFILL_CHUNK"); !ok || v != strconv.Itoa(sz24.Chunk) {
		t.Fatalf("lookup chunk = %q (%v), want %d", v, ok, sz24.Chunk)
	}

	// Explicit FAK_INKLING2_KV=f32 must price higher than f16 at the same chunk.
	sz24F32, err := serveInkling2ResidentSizingOf(path, inkling2EnvLookup(map[string]string{
		inkling2ResidentCtxEnv:       "16384",
		"FAK_INKLING2_KV":            "f32",
		"FAK_INKLING2_PREFILL_CHUNK": strconv.Itoa(sz24.Chunk),
	}))
	if err != nil {
		t.Fatalf("24 GB f32 sizing err: %v", err)
	}
	if sz24F32.Chunk != sz24.Chunk {
		t.Fatalf("explicit chunk %d not honored: got %d", sz24.Chunk, sz24F32.Chunk)
	}
	if sz24F32.Plan.SteadyBytes <= sz24.Plan.SteadyBytes {
		t.Fatalf("f32 steady %d <= f16 steady %d", sz24F32.Plan.SteadyBytes, sz24.Plan.SteadyBytes)
	}

	// 36 GB Mac: 27 GiB GPU share. 16k and 32k contexts admit at full chunk=2048.
	serveReadMemory = func() (memgate.Memory, error) {
		return memgate.Memory{TotalBytes: 36 << 30, AvailableBytes: 32 << 30}, nil
	}
	for _, ctx := range []string{"16384", "32768"} {
		sz36, err := serveInkling2ResidentSizingOf(path, inkling2EnvLookup(map[string]string{
			inkling2ResidentCtxEnv: ctx,
		}))
		if err != nil {
			t.Fatalf("36 GB ctx=%s sizing err: %v", ctx, err)
		}
		if sz36.Chunk != 2048 {
			t.Fatalf("36 GB ctx=%s chunk = %d, want 2048", ctx, sz36.Chunk)
		}
		if sz36.Plan.SteadyBytes > 27<<30 {
			t.Fatalf("36 GB ctx=%s steady bytes = %d (> 27 GiB GPU share)", ctx, sz36.Plan.SteadyBytes)
		}
	}
}
