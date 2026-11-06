package compute

import (
	"math"
	"testing"
)

// smolLM2_135M is the decode geometry of SmolLM2-135M (the model modelbench runs live and the ncu
// script profiles): hidden 576 = 9 heads × 64, GQA with 3 KV heads, FFN 1536, vocab 49152 (HF
// config config.json). KVLen is a representative mid-context decode step. These are the numbers the
// A100 measurement is reconciled against.
var smolLM2_135M = DecodeGeometry{
	DModel: 576, NHeads: 9, NKVHeads: 3, HeadDim: 64, DFF: 1536, Vocab: 49152, KVLen: 2048,
}

func decApprox(t *testing.T, name string, got, want, eps float64) {
	t.Helper()
	if math.Abs(got-want) > eps {
		t.Errorf("%s = %g, want %g (±%g)", name, got, want, eps)
	}
}

func TestNVArchTable(t *testing.T) {
	a, ok := LookupNVArch(SM80)
	if !ok {
		t.Fatal("sm_80 must be a known arch")
	}
	if a.Label != "sm_80" || a.RegsPerSM != 65536 || a.MaxWarpsPerSM != 64 || a.MaxBlocksPerSM != 32 {
		t.Errorf("sm_80 row wrong: %+v", a)
	}
	if a.SmemPerSM != 164*1024 {
		t.Errorf("sm_80 SmemPerSM = %d, want 164 KiB", a.SmemPerSM)
	}
	// sm_90/sm_100 grew shared memory to 228 KiB; the per-SM ISA ceilings are otherwise identical.
	for _, cap := range []NVCompute{SM90, SM100} {
		g, ok := LookupNVArch(cap)
		if !ok || g.SmemPerSM != 228*1024 {
			t.Errorf("%v SmemPerSM wrong: %+v (ok=%v)", cap, g, ok)
		}
	}
	// fail-closed: the zero value is not a known arch.
	if _, ok := LookupNVArch(NVUnknown); ok {
		t.Error("NVUnknown must not resolve to an arch")
	}
	if n := len(KnownNVArches()); n != 3 {
		t.Errorf("KnownNVArches len = %d, want 3", n)
	}
}

// TestFlashDecodeUnderfill is the headline: SmolLM2's flash decode launches one block per query head
// (grid=9) onto an A100's 108 SMs, leaving 99 idle EVERY decode step — the exact, register-
// independent underfill that grounds the persistent-kernel verdict.
func TestFlashDecodeUnderfill(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	occ := a.Occupancy(FlashDecodeLaunch(smolLM2_135M, 40), A100SMs)

	if occ.GridBlocks != 9 {
		t.Errorf("GridBlocks = %d, want 9 (one block per query head)", occ.GridBlocks)
	}
	if occ.IdleSMs != 99 {
		t.Errorf("IdleSMs = %d, want 99 (108 SMs − 9 blocks)", occ.IdleSMs)
	}
	if occ.WarpsPerBlock != 4 {
		t.Errorf("WarpsPerBlock = %d, want 4 (128/32)", occ.WarpsPerBlock)
	}
	// per-SM occupancy is actually decent (register-limited to 12 blocks → 75%); the problem is not
	// per-SM, it is that only 9 SMs ever get work.
	if occ.BindingLimiter != "register" {
		t.Errorf("BindingLimiter = %q, want register", occ.BindingLimiter)
	}
	if occ.BlocksPerSM != 12 {
		t.Errorf("BlocksPerSM = %d, want 12", occ.BlocksPerSM)
	}
	decApprox(t, "TheoreticalOcc", occ.TheoreticalOcc, 0.75, 1e-9)
	// achieved DEVICE occupancy is catastrophic: 9 blocks × 4 warps / (108×64) ≈ 0.5%.
	decApprox(t, "DeviceOcc", occ.DeviceOcc, 36.0/6912.0, 1e-9)
}

// TestUnderfillRegisterIndependent proves the doc's load-bearing claim: the grid-underfill verdict
// does NOT depend on the compiler's register count. Whatever ncu reports for launch__registers_per_
// thread, the idle-SM count is unchanged — so the file/no-file decision needs no device measurement.
func TestUnderfillRegisterIndependent(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	for _, regs := range []int{24, 32, 40, 64, 96} {
		occ := a.Occupancy(FlashDecodeLaunch(smolLM2_135M, regs), A100SMs)
		if occ.IdleSMs != 99 {
			t.Errorf("regs=%d: IdleSMs = %d, want 99 (register-independent)", regs, occ.IdleSMs)
		}
	}
}

// TestDecodeGEMVWaveQuant checks the OTHER decode shape: the P=1 GEMV fills the machine (no idle
// SMs) but its grid tiles into scheduling waves whose ragged final wave wastes >5% — the tail the
// CLC tile-scheduling candidate targets.
func TestDecodeGEMVWaveQuant(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	occ := a.Occupancy(Q8GemmDecodeLaunch(smolLM2_135M.DFF, 40), A100SMs)
	if occ.GridBlocks != 1536 {
		t.Errorf("GridBlocks = %d, want 1536 (DFF)", occ.GridBlocks)
	}
	if occ.IdleSMs != 0 {
		t.Errorf("IdleSMs = %d, want 0 (grid ≫ SMs fills the machine)", occ.IdleSMs)
	}
	if occ.WaveQuantWaste <= decodeTailGapFloor {
		t.Errorf("WaveQuantWaste = %g, want > %g (a real tail)", occ.WaveQuantWaste, decodeTailGapFloor)
	}
}

// TestDecodeTraffic checks the HBM-traffic accounting: memory-bound (~0.5 FLOP/byte) and dominated
// by a K/V stream whose GQA re-read is the exact waste the reuse/__ldcs candidates target.
func TestDecodeTraffic(t *testing.T) {
	tr := DecodeHBMTraffic(smolLM2_135M)
	if !tr.MemoryBound() {
		t.Errorf("decode attention must be memory-bound (intensity %g)", tr.Intensity)
	}
	decApprox(t, "Intensity", tr.Intensity, 0.5, 0.01)
	if tr.KVStreamFrac <= 0.5 {
		t.Errorf("KVStreamFrac = %g, want > 0.5 (KV stream dominates)", tr.KVStreamFrac)
	}
	// exact GQA re-read waste: 2·(nH−nKV)·N·hd floats = 2·6·2048·64·4 bytes.
	wantWaste := int64(2) * int64(9-3) * 2048 * 64 * 4
	if tr.KVReuseWaste != wantWaste {
		t.Errorf("KVReuseWaste = %d, want %d", tr.KVReuseWaste, wantWaste)
	}
	if tr.ReuseOptimal >= tr.Streamed {
		t.Errorf("ReuseOptimal (%d) must be < Streamed (%d)", tr.ReuseOptimal, tr.Streamed)
	}
	// degenerate MQA (nKV==nHeads) has no re-read waste.
	mqa := smolLM2_135M
	mqa.NKVHeads = mqa.NHeads
	if w := DecodeHBMTraffic(mqa).KVReuseWaste; w != 0 {
		t.Errorf("MQA KVReuseWaste = %d, want 0", w)
	}
}

// TestDecodeGapReport pins the promotion verdicts: exactly the four A100-measurable candidates with
// a real modeled gap file; tmem (measurable, but grid-bound so no gap) and the three Blackwell/
// Hopper-only mechanisms do not.
func TestDecodeGapReport(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	report := DecodeGapReport(a, A100SMs, smolLM2_135M, 40)
	if len(report) != 8 {
		t.Fatalf("report has %d entries, want 8 (the shortlist)", len(report))
	}
	wantFile := map[string]bool{
		"persistent-kernel-work-stealing-tail-fix": true,
		"l1-cache-hints-decode":                    true,
		"clc-decode-tile-scheduling":               true,
		"moe-launch-fusion-ladder":                 true,
		"tmem-accumulator-migration":               false,
		"clc-try-cancel-speculative":               false,
		"nvfp4-two-level-block-scale":              false,
		"pdl-moe-kernel-overlap":                   false,
	}
	seen := map[string]bool{}
	for _, g := range report {
		seen[g.Candidate] = true
		want, ok := wantFile[g.Candidate]
		if !ok {
			t.Errorf("unexpected candidate %q", g.Candidate)
			continue
		}
		if g.ShouldFile() != want {
			t.Errorf("%s: ShouldFile()=%v, want %v (measurable=%v gap=%v)",
				g.Candidate, g.ShouldFile(), want, g.Measurable, g.PredictedGap)
		}
	}
	for c := range wantFile {
		if !seen[c] {
			t.Errorf("candidate %q missing from report", c)
		}
	}
	// tmem is the honest measured-no-gap case: A100-measurable but grid-bound, so no device gap.
	for _, g := range report {
		if g.Candidate == "tmem-accumulator-migration" {
			if !g.Measurable || g.PredictedGap {
				t.Errorf("tmem should be Measurable && !PredictedGap, got measurable=%v gap=%v",
					g.Measurable, g.PredictedGap)
			}
		}
	}
}

// TestOccupancyDeviceSMsDefault checks the convenience: deviceSMs ≤ 0 falls back to the A100.
func TestOccupancyDeviceSMsDefault(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	def := a.Occupancy(FlashDecodeLaunch(smolLM2_135M, 40), 0)
	if def.DeviceSMs != A100SMs {
		t.Errorf("deviceSMs=0 fallback = %d, want %d", def.DeviceSMs, A100SMs)
	}
}

// TestDecodeOccupancyWitnessReport is the PRINTED witness `make cuda-occupancy` runs (#4188): the
// per-decode-kernel occupancy + HBM-traffic table plus the eight Cluster-G file/no-file verdicts,
// emitted via t.Logf so `go test -v` prints it on any host. Every number here is the exact analytic
// arm (grid-block vs SM counts, operand-byte counts — no timer, no device); the DEVICE corroboration
// (ncu achieved-occupancy / DRAM-%) is the separate GPU-only harness tools/dgx_decode_occupancy_ncu.sh,
// and the report's last line says it was NOT run rather than claiming it — the SKIP-is-not-PASS
// discipline with nothing skipped, because the analytic arm never skips.
func TestDecodeOccupancyWitnessReport(t *testing.T) {
	a, ok := LookupNVArch(SM80)
	if !ok {
		t.Fatal("sm_80 must be a known arch")
	}
	const flashRegs = 40 // placeholder compiler count; ncu launch__registers_per_thread replaces it
	g := smolLM2_135M

	t.Logf("decode occupancy witness — arch=%s deviceSMs=%d geometry=SmolLM2-135M (nH=%d nKV=%d hd=%d dff=%d vocab=%d kvlen=%d)",
		a.Label, A100SMs, g.NHeads, g.NKVHeads, g.HeadDim, g.DFF, g.Vocab, g.KVLen)
	launches := []DecodeLaunch{
		FlashDecodeLaunch(g, flashRegs),
		Q8GemmDecodeLaunch(g.DFF, flashRegs),    // FFN projection width
		AWQGemvDecodeLaunch(g.Vocab, flashRegs), // LM head, the largest decode GEMV
	}
	for _, l := range launches {
		occ := a.Occupancy(l, A100SMs)
		if occ.GridBlocks <= 0 || occ.BlocksPerSM <= 0 {
			t.Fatalf("%s: degenerate occupancy row %+v", l.Kernel, occ)
		}
		t.Logf("kernel=%-17s grid=%5d blocks/SM=%2d limiter=%-8s theoreticalOcc=%.2f deviceOcc=%.4f idleSMs=%3d waves=%d waveQuantWaste=%.3f",
			occ.Kernel, occ.GridBlocks, occ.BlocksPerSM, occ.BindingLimiter,
			occ.TheoreticalOcc, occ.DeviceOcc, occ.IdleSMs, occ.Waves, occ.WaveQuantWaste)
	}

	tr := DecodeHBMTraffic(g)
	t.Logf("traffic: streamed=%dB reuseOptimal=%dB kvReuseWaste=%dB kvStreamFrac=%.4f intensity=%.3f FLOP/B memoryBound=%v",
		tr.Streamed, tr.ReuseOptimal, tr.KVReuseWaste, tr.KVStreamFrac, tr.Intensity, tr.MemoryBound())

	report := DecodeGapReport(a, A100SMs, g, flashRegs)
	if len(report) != 8 {
		t.Fatalf("gap report has %d rows, want 8 (the Cluster-G shortlist)", len(report))
	}
	filed := 0
	for _, gap := range report {
		if gap.Candidate == "" || gap.Seam == "" || gap.Rationale == "" {
			t.Errorf("incomplete verdict row: %+v", gap)
		}
		verdict := "defer(not-measurable-here)"
		switch {
		case gap.ShouldFile():
			verdict = "FILE"
			filed++
		case gap.Measurable:
			verdict = "no-gap(do-not-file)"
		}
		t.Logf("verdict=%-26s candidate=%-42s keysOn=%-32s seam=%s", verdict, gap.Candidate, gap.KeysOn, gap.Seam)
	}
	if filed == 0 {
		t.Error("no candidate files — an all-no-gap report would gate the whole shortlist; check DecodeGapReport")
	}
	t.Logf("device corroboration: NOT run here — exact analytic arm only; GPU harness = tools/dgx_decode_occupancy_ncu.sh; baseline = docs/notes/2026-07-11-decode-occupancy-witness-measurement.md")
}

// fak-test:runtime fast est=1ms lane=default
func TestDecodeTrafficKVWidths(t *testing.T) {
	g := DecodeGeometry{NHeads: 24, NKVHeads: 4, HeadDim: 256, KVLen: 32768}
	for _, tc := range []struct {
		name              string
		element, scale    int
		optimal, streamed int64 // all 16 full-attention layers, including F32 Q/O
	}{
		{"f32", 4, 0, 4295753728, 25770590208},
		{"f16", 2, 0, 2148270080, 12885688320},
		{"bf16", 2, 0, 2148270080, 12885688320},
		{"fp8-e4m3-per-head-scale", 1, 4, 1091305472, 6543900672},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := DecodeHBMTrafficForKV(g, tc.element, tc.scale)
			if !ok || 16*got.ReuseOptimal != tc.optimal || 16*got.Streamed != tc.streamed ||
				got.KVReuseWaste != (tc.streamed-tc.optimal)/16 {
				t.Fatalf("wrong modeled bytes: %+v ok=%v", got, ok)
			}
			if math.IsNaN(got.KVStreamFrac) || math.IsInf(got.KVStreamFrac, 0) ||
				math.IsNaN(got.Intensity) || math.IsInf(got.Intensity, 0) {
				t.Fatalf("non-finite analytical ratio: %+v", got)
			}
			qo := int64(8 * 24 * 256)
			flops := int64(4) * 24 * 32768 * 256
			decApprox(t, "KVStreamFrac", got.KVStreamFrac, float64(got.Streamed-qo)/float64(got.Streamed), 1e-15)
			decApprox(t, "Intensity", got.Intensity, float64(flops)/float64(got.Streamed), 1e-15)
			t.Logf("analytical operand bytes only: layout=%s Q=24 KV=4 hd=256 positions=32768 layers=16 streamed=%d reuseOptimal=%d; device traffic/performance UNRUN",
				tc.name, tc.streamed, tc.optimal)
		})
	}
	// One paired FP8 K/V row contains 512 code bytes and 8 scale bytes.
	one := DecodeGeometry{NHeads: 1, NKVHeads: 1, HeadDim: 256, KVLen: 1}
	got, ok := DecodeHBMTrafficForKV(one, 1, 4)
	if !ok || got.Streamed != 2048+520 || got.ReuseOptimal != got.Streamed || got.KVReuseWaste != 0 {
		t.Fatalf("scale unit or Q/O cost changed: %+v ok=%v", got, ok)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDecodeTrafficKVLegacyParity(t *testing.T) {
	for _, g := range []DecodeGeometry{
		smolLM2_135M,
		{NHeads: 24, NKVHeads: 4, HeadDim: 256, KVLen: 32768},
		{NHeads: 4, NKVHeads: 4, HeadDim: 65, KVLen: 1},
		{NHeads: 1, NKVHeads: 1, HeadDim: 1, KVLen: 1},
	} {
		got, ok := DecodeHBMTrafficForKV(g, 4, 0)
		if !ok || got != DecodeHBMTraffic(g) {
			t.Fatalf("legacy F32 changed for %+v: %+v ok=%v", g, got, ok)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDecodeTrafficKVRefusesInvalid(t *testing.T) {
	valid := DecodeGeometry{NHeads: 24, NKVHeads: 4, HeadDim: 256, KVLen: 32768}
	for _, layout := range [][2]int{{0, 0}, {-1, 0}, {1, 0}, {1, 8}, {2, 4}, {4, 4}, {4, -1}, {8, 0}} {
		got, ok := DecodeHBMTrafficForKV(valid, layout[0], layout[1])
		if ok || got != (DecodeTraffic{}) {
			t.Fatalf("unsupported layout %v: %+v ok=%v", layout, got, ok)
		}
	}
	maxInt := int(^uint(0) >> 1)
	for _, g := range []DecodeGeometry{
		{},
		{NHeads: -1, NKVHeads: 1, HeadDim: 1, KVLen: 1},
		{NHeads: 1, NKVHeads: 0, HeadDim: 1, KVLen: 1},
		{NHeads: 24, NKVHeads: 5, HeadDim: 256, KVLen: 32768},
		{NHeads: 1, NKVHeads: 2, HeadDim: 1, KVLen: 1},
		{NHeads: 1, NKVHeads: 1, HeadDim: 0, KVLen: 1},
		{NHeads: 1, NKVHeads: 1, HeadDim: 1, KVLen: -1},
		{NHeads: maxInt, NKVHeads: 1, HeadDim: maxInt, KVLen: 1},
		{NHeads: 4, NKVHeads: 1, HeadDim: maxInt, KVLen: maxInt},
	} {
		for _, layout := range [][2]int{{4, 0}, {2, 0}, {1, 4}} {
			got, ok := DecodeHBMTrafficForKV(g, layout[0], layout[1])
			if ok || got != (DecodeTraffic{}) {
				t.Fatalf("invalid/overflow geometry %+v layout=%v: %+v ok=%v", g, layout, got, ok)
			}
		}
	}
	// These int64-boundary geometries cannot be represented by a 32-bit int.
	// The ordinary rejection controls above still cover overflow on 32-bit hosts.
	if int64(maxInt) > 2147483647 {
		limit := int64(1<<63 - 1)
		n := int((limit - 8) / 8)
		g := DecodeGeometry{NHeads: 1, NKVHeads: 1, HeadDim: 1, KVLen: n}
		got, ok := DecodeHBMTrafficForKV(g, 4, 0)
		if !ok || got.Streamed != limit-7 || got.ReuseOptimal != got.Streamed {
			t.Fatalf("last representable row rejected: %+v ok=%v", got, ok)
		}
		g.KVLen++
		if got, ok := DecodeHBMTrafficForKV(g, 4, 0); ok || got != (DecodeTraffic{}) {
			t.Fatalf("Q/O addition overflow accepted: %+v ok=%v", got, ok)
		}
		// FP8 byte counts still fit here, but the modeled FLOP count does not.
		g.HeadDim, g.KVLen = 256, int(limit/1024+1)
		if got, ok := DecodeHBMTrafficForKV(g, 1, 4); ok || got != (DecodeTraffic{}) {
			t.Fatalf("FLOP overflow accepted: %+v ok=%v", got, ok)
		}
	}
}
