//go:build darwin

package compute

import (
	"strconv"
	"testing"
)

func TestHostSystemMemoryDarwin(t *testing.T) {
	total, free, known := hostSystemMemory()
	if !known {
		t.Fatal("hostSystemMemory() returned known == false on darwin")
	}
	if total <= 0 {
		t.Fatalf("hostSystemMemory() returned non-positive total: %d", total)
	}
	if free <= 0 {
		t.Fatalf("hostSystemMemory() returned free <= 0: %d (expected > 0)", free)
	}
	if free == FreeUnknown {
		t.Fatal("hostSystemMemory() returned free == FreeUnknown")
	}
	if free > total {
		t.Fatalf("hostSystemMemory() returned free (%d) > total (%d)", free, total)
	}
}

func TestParseVMStat(t *testing.T) {
	// Genuine 3-field sample: no compressor/purgeable/wired lines, so the
	// pre-existing formula value is preserved by the regression guard.
	sampleAppleSilicon := []byte(`Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   682407.
Pages active:                                 673382.
Pages inactive:                               603389.
Pages speculative:                             68177.
Pages throttled:                                   0.
`)
	free, ok := parseVMStat(sampleAppleSilicon)
	if !ok {
		t.Fatal("parseVMStat failed on valid sample output")
	}
	// (682407 + 68177 + 603389) * 16384 = 1353973 * 16384 = 22182985728
	want := int64(1353973) * 16384
	if free != want {
		t.Fatalf("parseVMStat returned %d, want %d", free, want)
	}

	sampleIntel := []byte(`Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                                    10000.
Pages inactive:                                20000.
Pages speculative:                              5000.
`)
	freeIntel, okIntel := parseVMStat(sampleIntel)
	if !okIntel {
		t.Fatal("parseVMStat failed on Intel sample output")
	}
	wantIntel := int64(35000) * 4096
	if freeIntel != wantIntel {
		t.Fatalf("parseVMStat returned %d, want %d", freeIntel, wantIntel)
	}

	// Missing Pages free:
	if _, ok := parseVMStat([]byte("some random output\nwithout pages free")); ok {
		t.Fatal("parseVMStat should fail when Pages free: is missing")
	}
}

func TestParseVMStatCompressorAndPurgeable(t *testing.T) {
	// Same 3-field base as TestParseVMStat plus compressor + purgeable + wired.
	sample := []byte(`Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   1000.
Pages active:                                 673382.
Pages inactive:                               604000.
Pages speculative:                             31000.
Pages throttled:                                   0.
Pages wired down:                             178183.
Pages purgeable:                                5000.
Pages stored in compressor:                  1690000.
`)
	p := parseVMStatFields(sample)
	base := p.free + p.speculative + p.inactive
	if p.availablePages() <= base {
		t.Fatalf("availablePages()=%d must exceed 3-field base=%d when compressor/purgeable present", p.availablePages(), base)
	}
	got, ok := p.availableBytes(0)
	if !ok {
		t.Fatal("availableBytes(0) not ok")
	}
	wantBytes := int64(base+5000+uint64(1690000*compressorReclaimFactor)) * 16384
	if got != wantBytes {
		t.Fatalf("availableBytes=%d want %d", got, wantBytes)
	}
}

func assertClamp(t *testing.T, total, wired, want int64) {
	t.Helper()
	sample := []byte("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" +
		"Pages free: 999999.\nPages inactive: 0.\nPages speculative: 0.\n" +
		"Pages stored in compressor: 0.\nPages wired down: " + itoa(wired) + ".\n")
	got, ok := parseVMStatFields(sample).availableBytes(total)
	if !ok {
		t.Fatal("availableBytes not ok")
	}
	if got != want {
		t.Fatalf("availableBytes(total=%d, wired=%d)=%d want %d", total, wired, got, want)
	}
}

func itoa(v int64) string {
	return strconv.FormatInt(v, 10)
}

func TestParseVMStatClamp(t *testing.T) {
	// availablePages = 999999, pageSize 16384 -> raw 16383983616 bytes.
	raw := int64(999999) * 16384
	// 8 GiB total: raw (16.38 GiB) exceeds total, so the clamp must bite.
	total := int64(8) << 30
	wiredPages := int64(178183)
	wiredBytes := wiredPages * 16384
	// raw > total-wired -> clamped to total - wired.
	assertClamp(t, total, wiredPages, total-wiredBytes)
	// wired absent (0) -> no clamp, raw stands.
	assertClamp(t, total, 0, raw)
	// total unknown (0) -> no clamp.
	assertClamp(t, 0, wiredPages, raw)
}

func TestParseVMStatZeroAndAbsentFields(t *testing.T) {
	// All zero counts except free -> free only; no panic/overflow.
	zeros := []byte("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" +
		"Pages free: 0.\nPages speculative: 0.\nPages inactive: 0.\n" +
		"Pages purgeable: 0.\nPages stored in compressor: 0.\nPages wired down: 0.\n")
	if _, ok := parseVMStatFields(zeros).availableBytes(0); ok {
		t.Fatal("all-zero sample should not yield ok")
	}
	// Huge fields must saturate, not overflow.
	huge := []byte("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" +
		"Pages free: 18446744073709551615.\nPages stored in compressor: 18446744073709551615.\n")
	p := parseVMStatFields(huge)
	if p.availablePages() != maxInt64Uint64 {
		t.Fatalf("availablePages should saturate at maxInt64Uint64, got %d", p.availablePages())
	}
	if _, ok := p.availableBytes(0); !ok {
		t.Fatal("huge sample should still be ok")
	}
}
