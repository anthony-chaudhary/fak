//go:build darwin

package compute

import (
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
	sampleAppleSilicon := []byte(`Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   682407.
Pages active:                                 673382.
Pages inactive:                               603389.
Pages speculative:                             68177.
Pages throttled:                                   0.
Pages wired down:                             178183.
Pages purgeable:                                6219.
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
