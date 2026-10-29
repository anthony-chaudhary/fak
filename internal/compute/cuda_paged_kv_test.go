//go:build cuda && cgo

package compute

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
	"time"
)

// pagedKVFill appends n deterministic rows (seeded by seed) to every layer of kv through
// the prompt-panel bulk append, so contiguous and paged stores ingest identical bytes.
func pagedKVFill(t *testing.T, cb *cudaBackend, kv KVStore, cfg KVConfig, start, n int, seed uint64, keep *[][]float32) {
	t.Helper()
	w := cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		for off := 0; off < n; off += 512 {
			rows := min(512, n-off)
			s := lcg(seed*1_000_003 + uint64(l)*7919 + uint64(start+off))
			kr, kk, vv := randVec(&s, rows*w), randVec(&s, rows*w), randVec(&s, rows*w)
			*keep = append(*keep, kr, kk, vv)
			cb.PromptPanelAppendKV(kv, l, mkResident(cb, []int{rows, w}, kr), mkResident(cb, []int{rows, w}, kk), mkResident(cb, []int{rows, w}, vv), start+off, rows)
		}
	}
}

func pagedMaxAbs(a, b []float32) (float64, float64) {
	var d float64
	for i := range a {
		d = math.Max(d, math.Abs(float64(a[i]-b[i])))
	}
	return d, cosine(a, b)
}

func newPagedForTest(t *testing.T, cb *cudaBackend, cfg KVConfig) *cudaPagedKV {
	t.Helper()
	t.Setenv("FAK_CUDA_PAGED_KV", "1")
	pk, ok := cb.NewKV(cfg).(*cudaPagedKV)
	if !ok {
		t.Fatalf("FAK_CUDA_PAGED_KV=1 NewKV(%+v) did not return the paged store", cfg)
	}
	return pk
}

// TestCUDAPagedAttentionMatchesContiguous is the parity gate: decode attention read through
// the block table (vendored vLLM v1 and partitioned v2 kernels) equals the contiguous flash
// kernel over the same rows, across head widths, GQA groups and lengths that end mid-page.
func TestCUDAPagedAttentionMatchesContiguous(t *testing.T) {
	cb := cudaOrSkip(t)
	cases := []struct {
		hd, nKV, grp, n int
	}{
		{128, 2, 4, 37},   // v1, tail page partial
		{64, 4, 1, 1000},  // v1, several warps of pages
		{128, 8, 4, 600},  // v2 (2 partitions, 32 heads)
		{256, 1, 6, 9000}, // v2 past the v1 8192 bound (Qwen3.8 full-attn geometry)
	}
	for _, tc := range cases {
		cfg := KVConfig{NumLayers: 2, NumKVHeads: tc.nKV, HeadDim: tc.hd, RopeTheta: 10000}
		var keep [][]float32
		ref := cb.newContiguousKV(cfg)
		pk := newPagedForTest(t, cb, cfg)
		pagedKVFill(t, cb, ref, cfg, 0, tc.n, 1, &keep)
		pagedKVFill(t, cb, pk, cfg, 0, tc.n, 1, &keep)
		nH := tc.grp * tc.nKV
		s := lcg(uint64(tc.hd*31 + tc.n))
		qh := randVec(&s, nH*tc.hd)
		q := mkResident(cb, []int{nH * tc.hd}, qh)
		scale := float32(1 / math.Sqrt(float64(tc.hd)))
		for l := 0; l < cfg.NumLayers; l++ {
			want := cb.Read(cb.Attention(q, ref, l, true, tc.grp, scale))
			got := cb.Read(cb.Attention(q, pk, l, true, tc.grp, scale))
			d, cos := pagedMaxAbs(want, got)
			if d > 1e-4 || cos < 0.99999 {
				t.Fatalf("hd=%d nKV=%d grp=%d n=%d layer %d: paged vs contiguous max|d|=%.3g cos=%.8f", tc.hd, tc.nKV, tc.grp, tc.n, l, d, cos)
			}
		}
		if pk.Len() != tc.n || len(pk.BlockTable()) != (tc.n+15)/16 {
			t.Fatalf("paged store len %d table %d for %d rows", pk.Len(), len(pk.BlockTable()), tc.n)
		}
		gotK, wantK := cb.Read(pk.KeysView(1)), cb.Read(ref.KeysView(1))
		gotV, wantV := cb.Read(pk.ValuesView(1)), cb.Read(ref.ValuesView(1))
		if d, _ := pagedMaxAbs(wantK, gotK); d != 0 {
			t.Fatalf("hd=%d KeysView gather differs (max|d|=%g)", tc.hd, d)
		}
		if d, _ := pagedMaxAbs(wantV, gotV); d != 0 {
			t.Fatalf("hd=%d ValuesView gather differs (max|d|=%g)", tc.hd, d)
		}
		ref.Free()
		pk.Free()
	}
}

// TestCUDAPagedKVCloneSharesPagesCopyOnWrite proves the prefix-reuse payoff: Clone maps the
// source's pages (no new device page), the first divergent append copies only the shared
// tail page, and both branches still attend exactly like independent contiguous caches.
func TestCUDAPagedKVCloneSharesPagesCopyOnWrite(t *testing.T) {
	cb := cudaOrSkip(t)
	cfg := KVConfig{NumLayers: 3, NumKVHeads: 2, HeadDim: 128, RopeTheta: 10000}
	var keep [][]float32
	a := newPagedForTest(t, cb, cfg)
	base := a.PageStats().LiveBlocks
	pagedKVFill(t, cb, a, cfg, 0, 40, 1, &keep) // 3 pages, tail holds 8
	live0 := a.PageStats().LiveBlocks
	if live0 != base+3 {
		t.Fatalf("40 rows must take 3 pages: live %d -> %d", base, live0)
	}
	b := a.Clone().(*cudaPagedKV)
	if got := a.PageStats().LiveBlocks; got != live0 {
		t.Fatalf("clone allocated pages: live %d -> %d", live0, got)
	}
	if b.SharedBytes() == 0 || a.ResidentBytes() > int64(len(a.Pos()))*8 {
		t.Fatalf("clone must share every page: shared=%d resident=%d", b.SharedBytes(), a.ResidentBytes())
	}
	pagedKVFill(t, cb, a, cfg, 40, 1, 2, &keep)
	pagedKVFill(t, cb, b, cfg, 40, 1, 3, &keep)
	if got := a.PageStats().LiveBlocks; got != live0+1 {
		t.Fatalf("divergent append must copy exactly the shared tail page: live %d -> %d", live0, got)
	}
	if a.BlockTable()[0] != b.BlockTable()[0] || a.BlockTable()[2] == b.BlockTable()[2] {
		t.Fatalf("full prefix pages must stay shared and the tail must diverge: a=%v b=%v", a.BlockTable(), b.BlockTable())
	}
	refA, refB := cb.newContiguousKV(cfg), cb.newContiguousKV(cfg)
	pagedKVFill(t, cb, refA, cfg, 0, 40, 1, &keep)
	pagedKVFill(t, cb, refA, cfg, 40, 1, 2, &keep)
	pagedKVFill(t, cb, refB, cfg, 0, 40, 1, &keep)
	pagedKVFill(t, cb, refB, cfg, 40, 1, 3, &keep)
	s := lcg(5)
	q := mkResident(cb, []int{4 * 2 * 128}, randVec(&s, 4*2*128))
	for _, pair := range []struct{ p, r KVStore }{{a, refA}, {b, refB}} {
		for l := 0; l < cfg.NumLayers; l++ {
			d, cos := pagedMaxAbs(cb.Read(cb.Attention(q, pair.r, l, true, 4, 0.088)), cb.Read(cb.Attention(q, pair.p, l, true, 4, 0.088)))
			if d > 1e-4 || cos < 0.99999 {
				t.Fatalf("branch attention diverged from its contiguous twin at layer %d: max|d|=%g cos=%f", l, d, cos)
			}
		}
	}
	a.Free()
	b.Free()
	refA.Free()
	refB.Free()
	if got := a.PageStats().LiveBlocks; got != base {
		t.Fatalf("freeing both branches must return their pages: live %d (base %d)", got, base)
	}
}

// TestCUDAPagedKVEvictAndHostRoundTrip pins Evict and the host snapshot to the contiguous
// store's semantics.
func TestCUDAPagedKVEvictAndHostRoundTrip(t *testing.T) {
	cb := cudaOrSkip(t)
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 2, HeadDim: 64, RopeTheta: 10000}
	var keep [][]float32
	ref := cb.newContiguousKV(cfg)
	pk := newPagedForTest(t, cb, cfg)
	pagedKVFill(t, cb, ref, cfg, 0, 35, 9, &keep)
	pagedKVFill(t, cb, pk, cfg, 0, 35, 9, &keep)
	if r1, r2 := ref.Evict(5, 13), pk.Evict(5, 13); r1 != r2 || pk.Len() != ref.Len() {
		t.Fatalf("evict removed %d/%d, len %d/%d", r1, r2, ref.Len(), pk.Len())
	}
	want, err := ref.SnapshotToHost()
	if err != nil {
		t.Fatal(err)
	}
	got, err := pk.SnapshotToHost()
	if err != nil {
		t.Fatal(err)
	}
	for l := range want.K {
		for _, pl := range []struct{ a, b []float32 }{{want.K[l], got.K[l]}, {want.KRaw[l], got.KRaw[l]}, {want.V[l], got.V[l]}} {
			if d, _ := pagedMaxAbs(pl.a, pl.b); len(pl.a) != len(pl.b) || d > 1e-5 {
				t.Fatalf("layer %d plane differs after evict (len %d/%d, max|d|=%g)", l, len(pl.a), len(pl.b), d)
			}
		}
	}
	t.Setenv("FAK_CUDA_PAGED_KV", "1")
	back, err := cb.RestoreKVFromHost(got)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := back.(*cudaPagedKV); !ok || back.Len() != pk.Len() {
		t.Fatalf("paged host restore returned %T len %d", back, back.Len())
	}
	if d, _ := pagedMaxAbs(cb.Read(back.KeysView(1)), cb.Read(ref.KeysView(1))); d > 1e-5 {
		t.Fatalf("restored keys differ: %g", d)
	}
	back.Free()
	pk.Free()
	ref.Free()
}

// witnessKVCapacityBytes is the device KV capacity the stores hold live: the allocated
// (not just used) K/Kraw/V capacity of contiguous stores, or the shared page arena once.
func witnessKVCapacityBytes(stores ...KVStore) int64 {
	var total int64
	arenaCounted := false
	for _, kv := range stores {
		switch s := kv.(type) {
		case *cudaKV:
			for l := range s.K {
				for _, d := range []dslice{s.K[l], s.Kraw[l], s.V[l]} {
					total += int64(d.cap) * int64(d.dtype.Bytes())
				}
			}
		case *cudaPagedKV:
			if !arenaCounted {
				total += s.PageStats().ArenaBytes
				arenaCounted = true
			}
		}
	}
	return total
}

// TestCUDAPagedKVSharedPrefixWitness is the HW witness (FAK_CUDA_PAGED_KV_WITNESS=1): two
// requests share a long prefix (request B is restored from A's snapshot with Clone, the
// GPU prefix-hit path), then both decode T tokens with per-layer append + attention. It
// prints one JSON line with device VRAM used and decode ms/token. Run once with
// FAK_CUDA_PAGED_KV unset (contiguous) and once with it set (paged), each in its own
// process, because freed device memory returns to the allocator pool, not the driver.
func TestCUDAPagedKVSharedPrefixWitness(t *testing.T) {
	if os.Getenv("FAK_CUDA_PAGED_KV_WITNESS") != "1" {
		t.Skip("set FAK_CUDA_PAGED_KV_WITNESS=1 to run the shared-prefix HW witness")
	}
	cb := cudaOrSkip(t)
	envInt := func(k string, d int) int {
		if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
			return v
		}
		return d
	}
	cfg := KVConfig{NumLayers: envInt("WITNESS_LAYERS", 36), NumKVHeads: envInt("WITNESS_KV_HEADS", 8), HeadDim: envInt("WITNESS_HEAD_DIM", 128), RopeTheta: 1e6}
	grp := envInt("WITNESS_GROUP", 4)
	prefix, steps := envInt("WITNESS_PREFIX", 8192), envInt("WITNESS_STEPS", 64)
	w := cfg.NumKVHeads * cfg.HeadDim
	_, free0, _ := cb.DeviceMemory()

	var keep [][]float32
	a := cb.NewKV(cfg)
	pagedKVFill(t, cb, a, cfg, 0, prefix, 1, &keep)
	b := a.Clone()
	s := lcg(77)
	row := mkResident(cb, []int{w}, randVec(&s, w))
	q := mkResident(cb, []int{grp * w}, randVec(&s, grp*w))
	cb.Read(row) // drain the stream before sampling free memory
	_, free1, _ := cb.DeviceMemory()
	capPrefix := witnessKVCapacityBytes(a, b)
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	var last Tensor
	step := func(kv KVStore, pos int) {
		for l := 0; l < cfg.NumLayers; l++ {
			kv.AppendKV(l, row, row, row, pos)
			out := cb.Attention(q, kv, l, true, grp, scale)
			if last.buf != nil {
				cb.Free(last)
			}
			last = out
		}
	}
	step(a, prefix) // warm (first divergent write: copy-on-write in paged mode)
	step(b, prefix)
	cb.Read(last)
	start := time.Now()
	for i := 1; i <= steps; i++ {
		step(a, prefix+i)
		step(b, prefix+i)
	}
	cb.Read(last)
	elapsed := time.Since(start)
	_, free2, _ := cb.DeviceMemory()

	mode := "contiguous"
	if _, ok := a.(*cudaPagedKV); ok {
		mode = "paged"
	}
	rec := map[string]any{
		"schema": "fak.cuda-paged-kv-witness/v1", "mode": mode, "device": cb.Name(),
		"layers": cfg.NumLayers, "kv_heads": cfg.NumKVHeads, "head_dim": cfg.HeadDim, "q_heads": grp * cfg.NumKVHeads,
		"prefix_tokens": prefix, "decode_steps": steps, "requests": 2,
		"vram_used_after_prefix_and_clone_bytes":   free0 - free1,
		"vram_used_after_decode_bytes":             free0 - free2,
		"decode_ms_per_token_per_request":          float64(elapsed.Microseconds()) / 1000 / float64(steps*2),
		"decode_ms_per_step_both_requests":         float64(elapsed.Microseconds()) / 1000 / float64(steps),
		"store_resident_bytes_a":                   a.ResidentBytes(),
		"store_resident_bytes_b":                   b.ResidentBytes(),
		"kv_capacity_bytes_after_prefix_and_clone": capPrefix,
		"kv_capacity_bytes_after_decode":           witnessKVCapacityBytes(a, b),
	}
	if st, ok := CUDAPagedKVPageStatsFor(a); ok {
		rec["pool"] = st
	}
	line, _ := json.Marshal(rec)
	t.Logf("WITNESS %s", line)
	if path := os.Getenv("WITNESS_OUT"); path != "" {
		_ = os.WriteFile(path, append(line, '\n'), 0o644)
	}
	a.Free()
	b.Free()
}
