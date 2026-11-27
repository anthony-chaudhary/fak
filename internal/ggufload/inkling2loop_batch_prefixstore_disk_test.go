package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"
)

func richCkpt(pos, posFloats, nSlots int, half bool, seed float32) *ink2RowCkpt {
	slots := make([]ink2SlotCkpt, nSlots)
	for s := 0; s < nSlots; s++ {
		lo := 0
		if s%2 == 1 && pos > 2 {
			lo = pos - 2
		}
		nKV := (pos - lo) * posFloats
		k := make([]float32, nKV)
		v := make([]float32, nKV)
		for i := range k {
			k[i] = seed + float32(s)*0.1 + float32(i)*0.01
			v[i] = -seed - float32(s)*0.2 - float32(i)*0.02
		}
		hist := make([]float32, 3)
		for i := range hist {
			hist[i] = seed*2 + float32(s+i)*0.25
		}
		slots[s] = ink2SlotCkpt{lo: lo, k: k, v: v, hist: hist}
	}
	return &ink2RowCkpt{
		pos:       pos,
		half:      half,
		posFloats: posFloats,
		slots:     slots,
	}
}

func assertCkptEqual(t *testing.T, got, want *ink2RowCkpt) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatalf("nil ckpt: got=%v want=%v", got, want)
	}
	if got.pos != want.pos || got.half != want.half || got.posFloats != want.posFloats || len(got.slots) != len(want.slots) {
		t.Fatalf("header mismatch: got=(pos=%d half=%v posFloats=%d slots=%d) want=(pos=%d half=%v posFloats=%d slots=%d)",
			got.pos, got.half, got.posFloats, len(got.slots),
			want.pos, want.half, want.posFloats, len(want.slots))
	}
	for s := range want.slots {
		g, w := got.slots[s], want.slots[s]
		if g.lo != w.lo || len(g.k) != len(w.k) || len(g.v) != len(w.v) || len(g.hist) != len(w.hist) {
			t.Fatalf("slot %d shape mismatch: got=(lo=%d k=%d v=%d h=%d) want=(lo=%d k=%d v=%d h=%d)",
				s, g.lo, len(g.k), len(g.v), len(g.hist), w.lo, len(w.k), len(w.v), len(w.hist))
		}
		for i := range w.k {
			if math.Float32bits(g.k[i]) != math.Float32bits(w.k[i]) {
				t.Fatalf("slot %d k[%d]: got %v, want %v", s, i, g.k[i], w.k[i])
			}
		}
		for i := range w.v {
			if math.Float32bits(g.v[i]) != math.Float32bits(w.v[i]) {
				t.Fatalf("slot %d v[%d]: got %v, want %v", s, i, g.v[i], w.v[i])
			}
		}
		for i := range w.hist {
			if math.Float32bits(g.hist[i]) != math.Float32bits(w.hist[i]) {
				t.Fatalf("slot %d hist[%d]: got %v, want %v", s, i, g.hist[i], w.hist[i])
			}
		}
	}
}

func TestInkling2PrefixStoreDiskEnvConfig(t *testing.T) {
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	// Disk spill stays off when FAK_INKLING2_BATCH_PREFIX_DISK_DIR is unset or empty.
	st, err := inkling2PrefixStoreFromEnv(env(map[string]string{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "64"}))
	if err != nil || st == nil || st.disk != nil {
		t.Fatalf("unset disk dir: st=%+v err=%v, want disk=nil", st, err)
	}
	// Default disk cap is 16384 MiB when dir is set.
	dir := t.TempDir()
	st, err = inkling2PrefixStoreFromEnv(env(map[string]string{
		"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "64",
		"FAK_INKLING2_BATCH_PREFIX_DISK_DIR":  dir,
	}))
	if err != nil || st == nil || st.disk == nil || st.disk.dir != dir || st.disk.capBytes != int64(inkling2PrefixDiskMiBDefault)<<20 {
		t.Fatalf("default disk cap: st=%+v disk=%+v err=%v", st, st.disk, err)
	}
	// Explicit disk cap MiB overrides default.
	st, err = inkling2PrefixStoreFromEnv(env(map[string]string{
		"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "64",
		"FAK_INKLING2_BATCH_PREFIX_DISK_DIR":  dir,
		"FAK_INKLING2_BATCH_PREFIX_DISK_MIB":  "1024",
	}))
	if err != nil || st == nil || st.disk == nil || st.disk.capBytes != 1024<<20 {
		t.Fatalf("explicit disk cap: st=%+v err=%v", st, err)
	}
	// Malformed FAK_INKLING2_BATCH_PREFIX_DISK_MIB returns an error naming the knob.
	for _, bad := range []string{"abc", "0", "-10"} {
		_, err := inkling2PrefixStoreFromEnv(env(map[string]string{
			"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "64",
			"FAK_INKLING2_BATCH_PREFIX_DISK_DIR":  dir,
			"FAK_INKLING2_BATCH_PREFIX_DISK_MIB":  bad,
		}))
		if err == nil || !strings.Contains(err.Error(), "FAK_INKLING2_BATCH_PREFIX_DISK_MIB") {
			t.Errorf("DISK_MIB=%q: err=%v, want refusal naming FAK_INKLING2_BATCH_PREFIX_DISK_MIB", bad, err)
		}
	}
}

func TestInkling2PrefixStoreDiskSpillAndReload(t *testing.T) {
	dir := t.TempDir()
	ck1 := richCkpt(4, 8, 4, true, 1.5)
	ck2 := richCkpt(4, 8, 4, false, 2.5)
	ck3 := richCkpt(4, 8, 4, true, 3.5)
	per := ck1.Bytes() + 8*4

	st := &inkling2PrefixStore{
		capBytes:  2*per + per/2,
		minTokens: 4,
		loopSlots: 4,
		disk:      newInkling2PrefixDisk(dir, 1<<20),
	}
	e1, e2, e3 := seqTokens(4, 100), seqTokens(4, 200), seqTokens(4, 300)
	st.put(e1, ck1)
	st.put(e2, ck2)
	// Inserting e3 evicts e1 from RAM and spills it to disk.
	st.put(e3, ck3)

	s := st.snapshot()
	if s.Entries != 2 || s.Evictions != 1 || s.DiskSpills != 1 || s.DiskEntries != 1 || s.DiskHeldBytes <= 0 {
		t.Fatalf("after spill: stats=%+v", s)
	}

	// Lookup e1: RAM miss -> disk hit -> reload into RAM (evicting e2 to disk).
	got := st.lookup(append(append([]int(nil), e1...), 999))
	if got == nil {
		t.Fatal("expected disk hit for spilled e1, got nil")
	}
	assertCkptEqual(t, got.ck, ck1)

	s = st.snapshot()
	if s.Hits != 1 || s.DiskHits != 1 || s.HitTokens != 4 || s.DiskSpills != 2 || s.DiskEntries != 1 || s.DiskErrors != 0 {
		t.Fatalf("after disk reload: stats=%+v", s)
	}
	// Verify e1 is now back in RAM (second lookup does not increment DiskHits).
	gotAgain := st.lookup(append(append([]int(nil), e1...), 888))
	if gotAgain == nil {
		t.Fatal("expected RAM hit after reload, got nil")
	}
	if s2 := st.snapshot(); s2.Hits != 2 || s2.DiskHits != 1 {
		t.Fatalf("second lookup should hit RAM: stats=%+v", s2)
	}
}

func TestInkling2PrefixStoreDiskLRUCapEviction(t *testing.T) {
	dir := t.TempDir()
	ck := richCkpt(4, 8, 4, false, 1.0)
	perRAM := ck.Bytes() + 8*4
	encoded, err := encodeInk2DiskCkpt(seqTokens(4, 10), 4, ck)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	perDisk := int64(len(encoded))

	// RAM holds 1 entry; disk holds 2 entries.
	st := &inkling2PrefixStore{
		capBytes:  perRAM,
		minTokens: 4,
		loopSlots: 4,
		disk:      newInkling2PrefixDisk(dir, 2*perDisk+perDisk/2),
	}
	e1, e2, e3, e4 := seqTokens(4, 10), seqTokens(4, 20), seqTokens(4, 30), seqTokens(4, 40)
	st.put(e1, richCkpt(4, 8, 4, false, 1.0))
	st.put(e2, richCkpt(4, 8, 4, false, 2.0)) // e1 -> disk
	st.put(e3, richCkpt(4, 8, 4, false, 3.0)) // e2 -> disk
	st.put(e4, richCkpt(4, 8, 4, false, 4.0)) // e3 -> disk; e1 evicted from disk

	s := st.snapshot()
	if s.DiskEntries != 2 || s.DiskEvictions != 1 || s.DiskHeldBytes > st.disk.capBytes {
		t.Fatalf("disk LRU cap: stats=%+v cap=%d", s, st.disk.capBytes)
	}
	// e1 was evicted from disk and must miss.
	if got := st.lookup(append(append([]int(nil), e1...), 1)); got != nil {
		t.Fatalf("evicted disk entry e1 should miss, got %v", got.tokens)
	}
	// e2 and e3 remain on disk and hit.
	if got := st.lookup(append(append([]int(nil), e2...), 1)); got == nil {
		t.Fatal("e2 should still hit on disk")
	}
	// Verify no orphaned temp files remain in dir.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, de := range entries {
		if strings.HasSuffix(de.Name(), ".tmp") {
			t.Fatalf("leftover temp file in disk dir: %s", de.Name())
		}
	}
}

func TestInkling2PrefixStoreDiskFailClosedCorruption(t *testing.T) {
	t.Run("bitflip_checksum_mismatch", func(t *testing.T) {
		dir := t.TempDir()
		ck := richCkpt(4, 4, 2, false, 1.25)
		per := ck.Bytes() + 8*4
		st := &inkling2PrefixStore{
			capBytes:  per,
			minTokens: 4,
			loopSlots: 2,
			disk:      newInkling2PrefixDisk(dir, 1<<20),
		}
		e1, e2 := seqTokens(4, 10), seqTokens(4, 20)
		st.put(e1, ck)
		st.put(e2, richCkpt(4, 4, 2, false, 2.25)) // spills e1

		path := st.disk.entries[0].file
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		raw[len(raw)-1] ^= 0xFF
		if err := os.WriteFile(path, raw, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		if got := st.lookup(append(append([]int(nil), e1...), 9)); got != nil {
			t.Fatalf("corrupt file should fail closed to nil, got %+v", got)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("corrupt file should be deleted, stat err=%v", err)
		}
		if s := st.snapshot(); s.DiskErrors != 1 || s.DiskEntries != 0 || s.DiskHeldBytes != 0 || s.Hits != 0 {
			t.Fatalf("after corruption: stats=%+v", s)
		}
	})

	t.Run("truncated_file", func(t *testing.T) {
		dir := t.TempDir()
		ck := richCkpt(4, 4, 2, false, 1.25)
		per := ck.Bytes() + 8*4
		st := &inkling2PrefixStore{
			capBytes:  per,
			minTokens: 4,
			loopSlots: 2,
			disk:      newInkling2PrefixDisk(dir, 1<<20),
		}
		e1, e2 := seqTokens(4, 10), seqTokens(4, 20)
		st.put(e1, ck)
		st.put(e2, richCkpt(4, 4, 2, false, 2.25))

		path := st.disk.entries[0].file
		if err := os.WriteFile(path, []byte("INK2CKPT"), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		if got := st.lookup(append(append([]int(nil), e1...), 9)); got != nil {
			t.Fatalf("truncated file should fail closed to nil, got %+v", got)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("truncated file should be deleted, stat err=%v", err)
		}
		if s := st.snapshot(); s.DiskErrors != 1 || s.DiskEntries != 0 {
			t.Fatalf("after truncation: stats=%+v", s)
		}
	})

	t.Run("wrong_loopSlots", func(t *testing.T) {
		dir := t.TempDir()
		ck := richCkpt(4, 4, 2, false, 1.25)
		per := ck.Bytes() + 8*4
		st := &inkling2PrefixStore{
			capBytes:  per,
			minTokens: 4,
			loopSlots: 2,
			disk:      newInkling2PrefixDisk(dir, 1<<20),
		}
		e1, e2 := seqTokens(4, 10), seqTokens(4, 20)
		st.put(e1, ck)
		st.put(e2, richCkpt(4, 4, 2, false, 2.25))

		path := st.disk.entries[0].file
		// Simulate store rebound to a different loop depth than the spilled file.
		st.loopSlots = 4
		if got := st.lookup(append(append([]int(nil), e1...), 9)); got != nil {
			t.Fatalf("wrong loopSlots should fail closed to nil, got %+v", got)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("wrong-loopSlots file should be deleted, stat err=%v", err)
		}
		if s := st.snapshot(); s.DiskErrors != 1 || s.DiskEntries != 0 {
			t.Fatalf("after wrong loopSlots: stats=%+v", s)
		}
	})

	t.Run("trailing_bytes_with_valid_header_rejected", func(t *testing.T) {
		ck := richCkpt(4, 4, 2, false, 1.0)
		raw, err := encodeInk2DiskCkpt(seqTokens(4, 10), 2, ck)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		// Append 4 trailing bytes and recompute checksum to ensure strict EOF check catches it.
		var buf bytes.Buffer
		buf.Write(raw)
		var pad [4]byte
		binary.LittleEndian.PutUint32(pad[:], 42)
		buf.Write(pad[:])
		out := buf.Bytes()
		sum := sha256.Sum256(out[ink2DiskHeaderSize:])
		copy(out[16:48], sum[:])
		if _, _, err := decodeInk2DiskCkpt(out, 2); err == nil {
			t.Fatal("expected error for trailing bytes")
		}
	})
}
