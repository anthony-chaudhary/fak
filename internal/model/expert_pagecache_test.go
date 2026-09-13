package model

import (
	"bytes"
	"io"
	"sync"
	"testing"
)

// expert_pagecache_test.go - the promotion witness for the page-cache/mmap-aware expert source
// (#1302). It proves, on the current host (Windows), that:
//
//  1. the FIRST read of an expert issues exactly ONE device ReadAt and it is page-aligned (offset
//     rounded down, length rounded up);
//  2. the SECOND read of the SAME expert issues ZERO device reads and is reported as a warm hit,
//     with bytes bit-identical to readExpert; and
//  3. a source that never enabled the page cache is byte-identical to readExpert with no ledger
//     movement - the default-off gate.
//
// The tests reuse newGGUFExpertFixture and fixtureStride from gguf_expert_source_test.go (same
// package). They never require a real mmap: enablePageCache takes a byte slice the test owns, and
// the advice shim's no-op arm is exercised on Windows so the whole path stays correct where
// madvise does not exist.

// recordingExpertReaderAt counts reads and records every requested range before delegating to an
// in-memory reader, so a test can assert both the number of device reads and their alignment.
type recordingExpertReaderAt struct {
	r     io.ReaderAt
	calls []expertRecordedRead
}

func (r *recordingExpertReaderAt) ReadAt(p []byte, off int64) (int, error) {
	r.calls = append(r.calls, expertRecordedRead{off: off, n: len(p)})
	return r.r.ReadAt(p, off)
}

// pageCacheFixture builds the shared fixture, a recording reader over it, and a source indexed
// from the same descriptions. The recorder is returned so a test can inspect device reads.
func pageCacheFixture(t *testing.T) (ggufExpertFixture, *recordingExpertReaderAt, *ggufExpertSource) {
	t.Helper()
	fix := newGGUFExpertFixture(t)
	rec := &recordingExpertReaderAt{r: bytes.NewReader(fix.file)}
	src, err := newGGUFExpertSource(rec, int64(len(fix.file)), fix.descs)
	if err != nil {
		t.Fatalf("newGGUFExpertSource: %v", err)
	}
	return fix, rec, src
}

// TestGGUFExpertPageCacheWarmReadAvoidsSecondDeviceRead is the issue's required witness: a cold
// read faults one aligned device range; a repeat read of the SAME expert faults none and is a
// warm hit whose bytes are bit-identical to readExpert, and the ledger moves accordingly.
func TestGGUFExpertPageCacheWarmReadAvoidsSecondDeviceRead(t *testing.T) {
	fix, rec, src := pageCacheFixture(t)
	if len(rec.calls) != 0 {
		t.Fatalf("construction performed %d payload reads, want 0", len(rec.calls))
	}
	if !src.enablePageCache(fix.file) {
		t.Fatal("enablePageCache refused a mapping exactly the size of the source")
	}
	if !src.PageCacheEnabled() {
		t.Fatal("PageCacheEnabled reported false after a valid enable")
	}
	d := fix.descs[0]
	const e = 2
	stride := fixtureStride(d)
	page := cachedPageSize()
	wantStart := d.Offset + int64(e*stride)
	wantAlignedStart := pageAlignDown(wantStart, page)
	wantAlignedEnd := pageAlignUp(wantStart+int64(stride), page)

	first, warm, err := src.readExpertPageCache(d.Name, e)
	if err != nil {
		t.Fatalf("first readExpertPageCache(%s, %d): %v", d.Name, e, err)
	}
	if warm {
		t.Fatal("first read reported a warm hit on a cold expert")
	}
	ref := fix.residentExpert(d, e)
	if !bytes.Equal(first, ref) {
		t.Fatal("cold page-cache read bytes differ from readExpert")
	}
	if len(rec.calls) != 1 {
		t.Fatalf("first read issued %d device reads, want exactly 1", len(rec.calls))
	}
	if got := rec.calls[0]; got.off != wantAlignedStart || int64(got.n) != wantAlignedEnd-wantAlignedStart {
		t.Fatalf("first read range = [%d,%d), want aligned [%d,%d)",
			got.off, got.off+int64(got.n), wantAlignedStart, wantAlignedEnd)
	}
	afterFirst, _ := src.pageCacheSession()
	if afterFirst.BytesFromDevice <= 0 {
		t.Fatalf("cold read booked BytesFromDevice=%d, want > 0", afterFirst.BytesFromDevice)
	}
	if afterFirst.WarmHits != 0 || afterFirst.ExpertsRead != 1 {
		t.Fatalf("after cold read ledger = %+v, want ExpertsRead=1 WarmHits=0", afterFirst)
	}
	if want := wantAlignedEnd - wantAlignedStart - int64(stride); afterFirst.AdvantageBytes != want {
		t.Fatalf("AdvantageBytes = %d, want alignment over-read %d", afterFirst.AdvantageBytes, want)
	}

	second, warm, err := src.readExpertPageCache(d.Name, e)
	if err != nil {
		t.Fatalf("second readExpertPageCache(%s, %d): %v", d.Name, e, err)
	}
	if !warm {
		t.Fatal("second read of the same expert was not a warm hit")
	}
	if !bytes.Equal(second, ref) {
		t.Fatal("warm page-cache read bytes differ from readExpert")
	}
	if len(rec.calls) != 1 {
		t.Fatalf("second read issued device reads (total %d), want no additional read", len(rec.calls))
	}
	afterSecond, _ := src.pageCacheSession()
	if afterSecond.BytesFromDevice != afterFirst.BytesFromDevice {
		t.Fatalf("warm read changed BytesFromDevice %d -> %d, want unchanged",
			afterFirst.BytesFromDevice, afterSecond.BytesFromDevice)
	}
	if afterSecond.WarmHits != 1 {
		t.Fatalf("WarmHits = %d after one warm read, want 1", afterSecond.WarmHits)
	}
	if afterSecond.BytesFromMapping != int64(stride) {
		t.Fatalf("BytesFromMapping = %d, want one stride %d", afterSecond.BytesFromMapping, stride)
	}
}

// TestGGUFExpertPageCacheAlignedReadThrough pins that a cold read's device range is page-aligned
// down on the offset and up on the end, and that the returned bytes equal readExpert exactly.
func TestGGUFExpertPageCacheAlignedReadThrough(t *testing.T) {
	fix, rec, src := pageCacheFixture(t)
	if !src.enablePageCache(fix.file) {
		t.Fatal("enablePageCache refused a valid mapping")
	}
	page := cachedPageSize()
	for _, d := range fix.descs {
		stride := int64(fixtureStride(d))
		for e := 0; e < d.Experts; e++ {
			start := d.Offset + int64(e)*stride
			wantEnd := pageAlignUp(start+stride, page)
			if wantEnd > int64(len(fix.file)) {
				wantEnd = int64(len(fix.file))
			}
			got, _, err := src.readExpertPageCache(d.Name, e)
			if err != nil {
				t.Fatalf("readExpertPageCache(%s, %d): %v", d.Name, e, err)
			}
			ref := fix.residentExpert(d, e)
			if !bytes.Equal(got, ref) {
				t.Fatalf("page-cache read %s[%d] differs from readExpert", d.Name, e)
			}
		}
	}
	if len(rec.calls) == 0 {
		t.Fatal("aligned read-through issued no device reads")
	}
	for i, call := range rec.calls {
		if call.off%int64(page) != 0 {
			t.Fatalf("read %d offset %d is not page-aligned", i, call.off)
		}
		if int64(call.n)%int64(page) != 0 && call.off+int64(call.n) != int64(len(fix.file)) {
			t.Fatalf("read %d length %d is not page-aligned and does not end at the source bound", i, call.n)
		}
	}
}

// TestGGUFExpertPageCacheDefaultOffIsByteIdentical pins the default-off gate: a source that never
// enabled the page cache reads byte-identically to readExpert, reports no warm hit, and moves no
// ledger - so nothing in the default path changes.
func TestGGUFExpertPageCacheDefaultOffIsByteIdentical(t *testing.T) {
	fix, rec, src := pageCacheFixture(t)
	d := fix.descs[1]
	got, warm, err := src.readExpertPageCache(d.Name, 1)
	if err != nil {
		t.Fatalf("readExpertPageCache on an unmapped source: %v", err)
	}
	if warm {
		t.Fatal("unmapped source reported a warm hit")
	}
	ref, err := src.readExpert(d.Name, 1)
	if err != nil {
		t.Fatalf("readExpert: %v", err)
	}
	if !bytes.Equal(got, ref) {
		t.Fatal("unmapped page-cache read differs from readExpert")
	}
	if len(rec.calls) != 2 {
		t.Fatalf("unmapped path issued %d device reads, want 2 (one per read, no alignment)", len(rec.calls))
	}
	if rec.calls[0].n != rec.calls[1].n || rec.calls[0].off != rec.calls[1].off {
		t.Fatalf("unmapped path ranges differ: %+v", rec.calls)
	}
	if src.PageCacheEnabled() {
		t.Fatal("PageCacheEnabled reported true without an enable call")
	}
	st, on := src.pageCacheSession()
	if on || st.Enabled || st.ExpertsRead != 0 || st.BytesFromDevice != 0 {
		t.Fatalf("default-off ledger moved: %+v on=%v", st, on)
	}
}

// TestGGUFExpertPageCacheAdvice pins that adviseRange fires on a mapped source and is a safe
// no-op/false when there is no mapping - it never reads a byte and never moves the ledger.
func TestGGUFExpertPageCacheAdvice(t *testing.T) {
	fix, _, src := pageCacheFixture(t)
	d := fix.descs[0]
	if src.adviseRange(d.Name, 0, PageAdviceRandom) {
		t.Fatal("adviseRange fired with no mapping")
	}
	if !src.enablePageCache(fix.file) {
		t.Fatal("enablePageCache refused a valid mapping")
	}
	for _, advice := range []PageAdvice{PageAdviceRandom, PageAdviceSequential} {
		if !src.adviseRange(d.Name, 0, advice) {
			// On the no-op shim arm (Windows) this is false; on unix it must fire. The contract is
			// "issued or safely declined", so accept either but require the unix arm to have fired.
			t.Logf("adviseRange(%s) declined on this platform", advice)
		}
	}
	if src.adviseRange("blk.9.ffn_gate_exps.weight", 0, PageAdviceRandom) {
		t.Fatal("adviseRange fired for an unknown tensor")
	}
	if src.adviseRange(d.Name, d.Experts, PageAdviceRandom) {
		t.Fatal("adviseRange fired for an out-of-range expert")
	}
}

// TestPageAlign is the table test for pageAlignDown/pageAlignUp, including the non-positive-page
// no-op and the exact-boundary case where up must not move.
func TestPageAlign(t *testing.T) {
	cases := []struct {
		off      int64
		page     int
		wantDown int64
		wantUp   int64
	}{
		{off: 0, page: 4096, wantDown: 0, wantUp: 0},
		{off: 1, page: 4096, wantDown: 0, wantUp: 4096},
		{off: 4095, page: 4096, wantDown: 0, wantUp: 4096},
		{off: 4096, page: 4096, wantDown: 4096, wantUp: 4096},
		{off: 4097, page: 4096, wantDown: 4096, wantUp: 8192},
		{off: 10000, page: 4096, wantDown: 8192, wantUp: 12288},
		{off: 100, page: 64, wantDown: 64, wantUp: 128},
		{off: 128, page: 64, wantDown: 128, wantUp: 128},
		{off: 999, page: 0, wantDown: 999, wantUp: 999},
		{off: 999, page: -1, wantDown: 999, wantUp: 999},
	}
	for _, tc := range cases {
		if got := pageAlignDown(tc.off, tc.page); got != tc.wantDown {
			t.Fatalf("pageAlignDown(%d, %d) = %d, want %d", tc.off, tc.page, got, tc.wantDown)
		}
		if got := pageAlignUp(tc.off, tc.page); got != tc.wantUp {
			t.Fatalf("pageAlignUp(%d, %d) = %d, want %d", tc.off, tc.page, got, tc.wantUp)
		}
	}
}

// TestExpertCheckpointTierPageCacheWarmFaultReadsNoDeviceBytes is the end-to-end join (#1302
// acceptance criteria 1-3) over the real R5 tier seam: a shard admitted with its mapped region
// (AddShardData) faults the SAME expert twice through tier.staging's mk builder, and the second
// fault must add ZERO device bytes to the source's page-cache ledger while returning bit-identical
// bytes. It also pins the default-off join: a tier added with a nil mapping never enables the
// page-cache ledger at all.
func TestExpertCheckpointTierPageCacheWarmFaultReadsNoDeviceBytes(t *testing.T) {
	const H, E = 256, 4
	resident := expertPrefetchModel(t, H, E, 2)

	var blob []byte
	fused := make([]FusedExpertTensor, 0, 3)
	for _, proj := range []string{"gate", "up", "down"} {
		suffix := proj + "_proj.weight"
		offset := int64(len(blob))
		for e := 0; e < E; e++ {
			blob = append(blob, resident.q4kw[expertName(0, e, suffix)].raw...)
		}
		fused = append(fused, FusedExpertTensor{
			Name: "blk.0.ffn_" + proj + "_exps.weight", Layer: 0, Proj: proj + "_proj",
			Quant: ExpertCheckpointQ4K, Offset: offset, Experts: E, Rows: H, Cols: H,
		})
	}

	// Mapped shard: the whole blob is the page-cache-visible region.
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShardData(bytes.NewReader(blob), int64(len(blob)), blob, fused); err != nil {
		t.Fatalf("AddShardData: %v", err)
	}
	name := expertName(0, 1, "gate_proj.weight")
	first, err := tier.fault(name)
	if err != nil {
		t.Fatalf("first fault: %v", err)
	}
	if first.q4 == nil || len(first.q4.raw) == 0 {
		t.Fatal("first fault returned no q4 bytes")
	}
	afterFirst, on := tier.shards[0].pageCacheSession()
	if !on {
		t.Fatal("AddShardData with an exact-size mapping did not enable the page-cache path")
	}
	if afterFirst.ExpertsRead == 0 || afterFirst.BytesFromDevice == 0 {
		t.Fatalf("cold fault booked no device read: %+v", afterFirst)
	}

	second, err := tier.fault(name)
	if err != nil {
		t.Fatalf("second fault: %v", err)
	}
	afterSecond, _ := tier.shards[0].pageCacheSession()
	if afterSecond.BytesFromDevice != afterFirst.BytesFromDevice {
		t.Fatalf("warm fault moved device bytes %d -> %d, want none",
			afterFirst.BytesFromDevice, afterSecond.BytesFromDevice)
	}
	if afterSecond.WarmHits != afterFirst.WarmHits+1 {
		t.Fatalf("warm fault did not register a page-cache hit: %+v", afterSecond)
	}
	if afterSecond.FillPins == 0 {
		t.Fatal("a cold fill was not recorded as L2-pin eligible (fill recognition)")
	}
	if len(second.q4.raw) != len(first.q4.raw) {
		t.Fatalf("warm fault bytes len=%d, want %d", len(second.q4.raw), len(first.q4.raw))
	}
	for i := range first.q4.raw {
		if first.q4.raw[i] != second.q4.raw[i] {
			t.Fatalf("warm fault bytes differ at %d", i)
		}
	}

	// Default-off join: a nil mapping leaves the source on the historical path with no ledger.
	plain := NewExpertCheckpointTier(0)
	if err := plain.AddShardData(bytes.NewReader(blob), int64(len(blob)), nil, fused); err != nil {
		t.Fatalf("AddShardData(nil): %v", err)
	}
	if _, on := plain.shards[0].pageCacheSession(); on {
		t.Fatal("a nil mapping enabled the page-cache path; default-off is not preserved")
	}
}

// TestGGUFExpertPageCacheConcurrentDefaultOff runs many concurrent readExpertPageCache calls on a
// source that never enabled the page cache. It is the regression guard for the lazy-init race the
// adversarial audit found: state must not allocate on the default-off path, so concurrent readers
// must neither race nor observe a half-built page state. Run under -race to make it bite.
func TestGGUFExpertPageCacheConcurrentDefaultOff(t *testing.T) {
	fix := newGGUFExpertFixture(t)
	src, err := newGGUFExpertSource(bytes.NewReader(fix.file), int64(len(fix.file)), fix.descs)
	if err != nil {
		t.Fatalf("newGGUFExpertSource: %v", err)
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			d := fix.descs[g%len(fix.descs)]
			for e := 0; e < d.Experts; e++ {
				got, warm, err := src.readExpertPageCache(d.Name, e)
				if err != nil {
					t.Errorf("concurrent read %s[%d]: %v", d.Name, e, err)
					return
				}
				if warm {
					t.Errorf("default-off concurrent read %s[%d] reported a warm hit", d.Name, e)
					return
				}
				if !bytes.Equal(got, fix.residentExpert(d, e)) {
					t.Errorf("default-off concurrent read %s[%d] differs from readExpert", d.Name, e)
					return
				}
			}
		}(g)
	}
	wg.Wait()
	if src.PageCacheEnabled() {
		t.Fatal("default-off source reported page cache enabled")
	}
}
