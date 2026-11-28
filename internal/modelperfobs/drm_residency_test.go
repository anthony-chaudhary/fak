package modelperfobs

import (
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestDrmResidentFootprint(t *testing.T) {
	procRoot := t.TempDir()
	fdDir := filepath.Join(procRoot, "4242", "fdinfo")
	if err := os.MkdirAll(fdDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Two FDs for the same DRM client (id=7) must deduplicate; client id=9 adds separately.
	fd3 := "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-resident-memory:\t512 MiB\ndrm-memory-gtt:\t256 MiB\ndrm-memory-vram:\t256 MiB\n"
	fd4 := "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-resident-memory:\t512 MiB\ndrm-memory-gtt:\t256 MiB\ndrm-memory-vram:\t256 MiB\n"
	fd5 := "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t9\ndrm-resident-memory:\t1 GiB\ndrm-memory-gtt:\t1024 KiB\n"
	fdNonDRM := "pos:\t0\nflags:\t0100002\n"

	for name, body := range map[string]string{"3": fd3, "4": fd4, "5": fd5, "6": fdNonDRM} {
		if err := os.WriteFile(filepath.Join(fdDir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// 512 MiB + 1 GiB = 1536 MiB
	wantResident := int64(512<<20) + int64(1<<30)
	snap, err := CollectDRMResidencyForPID(procRoot, 4242, 1<<30)
	if err != nil {
		t.Fatalf("CollectDRMResidencyForPID: %v", err)
	}
	if snap.Clients != 2 {
		t.Fatalf("Clients=%d, want 2 (deduplicated)", snap.Clients)
	}
	if snap.TotalResidentBytes != wantResident {
		t.Fatalf("TotalResidentBytes=%d, want %d", snap.TotalResidentBytes, wantResident)
	}
	if snap.GTTBytes != int64(256<<20)+int64(1024<<10) || snap.VRAMBytes != int64(256<<20) {
		t.Fatalf("GTT=%d VRAM=%d unexpected", snap.GTTBytes, snap.VRAMBytes)
	}
	if !snap.Exceeded {
		t.Fatal("Exceeded=false, want true when resident > 1 GiB limit")
	}
}
