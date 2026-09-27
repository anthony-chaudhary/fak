package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/memgate"
)

// readMetalAdmissionLedger decodes the reservation ledger the Metal admission
// seam persists under dir. The store writes it with json.MarshalIndent, so
// assertions must decode it rather than substring-match compact JSON. A missing
// ledger file is an empty ledger.
func readMetalAdmissionLedger(t *testing.T, dir string) []localadmission.Reservation {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, "reservations.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		t.Fatalf("read reservation ledger: %v", err)
	}
	var ledger struct {
		Reservations []localadmission.Reservation `json:"reservations"`
	}
	if err := json.Unmarshal(b, &ledger); err != nil {
		t.Fatalf("decode reservation ledger: %v\n%s", err, b)
	}
	return ledger.Reservations
}

// acquireMetalLeaseNoWait reports whether the GPU lease at path is free right
// now: it takes the lease without waiting and immediately gives it back.
func acquireMetalLeaseNoWait(path string) error {
	lease, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
	if err != nil {
		return err
	}
	lease.Release()
	return nil
}

// TestMetalAdmissionRefusesUnprobeableHostMemoryBeforeLoader is the #9587
// fail-closed RED/GREEN witness for the launcher seam. When the host memory
// sample cannot be read, reports no physical total, or reports more allocatable
// bytes than the physical pool, the default and aggregate admission modes must
// refuse with a typed reason BEFORE the loader runs: no loader call, no ledger
// row (not even an empty ledger file), and the GPU lease handed back. HEAD
// skipped the reservation on an unreadable sample and ran the loader unreserved
// (fail-open), and admitted an allocatable reading larger than the host.
//
// It deliberately drives only the historical wrapper and the serveReadMemory
// seam so it compiles against the pre-#9587 tree for the RED proof.
func TestMetalAdmissionRefusesUnprobeableHostMemoryBeforeLoader(t *testing.T) {
	const (
		peak   = int64(512 << 20)
		steady = int64(256 << 20)
	)
	hosts := []struct {
		name       string
		read       func() (memgate.Memory, error)
		wantReason string
	}{
		{
			name: "read_error",
			read: func() (memgate.Memory, error) {
				return memgate.Memory{}, errors.New("sysctl hw.memsize: host memory unavailable")
			},
			wantReason: "pressure_unknown",
		},
		{
			name: "zero_total",
			read: func() (memgate.Memory, error) {
				return memgate.Memory{TotalBytes: 0, FreeBytes: 8 << 30, AvailableBytes: 8 << 30}, nil
			},
			wantReason: "pressure_unknown",
		},
		{
			name: "allocatable_exceeds_total",
			read: func() (memgate.Memory, error) {
				return memgate.Memory{TotalBytes: 8 << 30, FreeBytes: 16 << 30, AvailableBytes: 16 << 30}, nil
			},
			wantReason: "capacity_unknown",
		},
	}
	for _, mode := range []string{"", "aggregate"} {
		modeName := mode
		if modeName == "" {
			modeName = "default"
		}
		for _, host := range hosts {
			t.Run(modeName+"/"+host.name, func(t *testing.T) {
				resDir := filepath.Join(t.TempDir(), "reservations")
				leasePath := filepath.Join(t.TempDir(), "gpu.lease")
				t.Setenv("FAK_RESERVATION_DIR", resDir)
				t.Setenv("FAK_GPU_LEASE", leasePath)
				t.Setenv("FAK_NATIVE_ADMISSION", mode)
				t.Setenv("FAK_ADMISSION_POLICY", "")
				t.Setenv("FAK_ADMISSION_STATE_ENVELOPE", "")
				t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", strconv.FormatInt(peak, 10))
				t.Setenv("FAK_TEST_STEADY_BYTES", strconv.FormatInt(steady, 10))
				orig := serveReadMemory
				t.Cleanup(func() { serveReadMemory = orig })
				serveReadMemory = host.read

				loads := 0
				release, err := loadLocalLauncherModelWithMetalLease(true, "unprobeable-host.gguf", gpulease.Options{}, func() {
					loads++
				})
				if release != nil {
					// Only on a fail-open admission does this hold anything; it runs
					// after the lease/ledger assertions below.
					t.Cleanup(release)
				}

				if err == nil {
					t.Errorf("admission accepted an unprobeable host sample (fail-open): loader ran %d time(s) with no reservation gate", loads)
				} else {
					if !strings.Contains(err.Error(), "local memory reservation refused: "+host.wantReason) {
						t.Errorf("refusal %q does not name the typed reason %q", err, host.wantReason)
					}
					if !strings.Contains(err.Error(), "FAK_NATIVE_ADMISSION=exclusive") {
						t.Errorf("refusal %q does not carry the exclusive-lease remedy", err)
					}
				}
				if loads != 0 {
					t.Errorf("loader calls = %d, want 0: an unprobeable host must refuse before any model byte is allocated", loads)
				}
				if rows := readMetalAdmissionLedger(t, resDir); len(rows) != 0 {
					t.Errorf("refusal left reservation rows: %+v", rows)
				}
				if _, statErr := os.Stat(filepath.Join(resDir, "reservations.json")); !errors.Is(statErr, os.ErrNotExist) {
					t.Errorf("refusal must not create a reservation ledger: stat err=%v", statErr)
				}
				// Checked before calling the returned release: the refusing seam
				// itself must hand the lease back.
				if leaseErr := acquireMetalLeaseNoWait(leasePath); leaseErr != nil {
					t.Errorf("GPU lease still held after the admission returned: %v", leaseErr)
				}
			})
		}
	}
}
