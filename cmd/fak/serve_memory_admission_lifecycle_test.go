package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/memgate"
)

// Issue #9587 lifecycle witnesses for admitLocalMetalModel: every host probe,
// the receipt sink, the reservation dir, and the GPU lease path are injected so
// each test decodes the real on-disk ledger and the real receipt lines.

const (
	lifecyclePeak   = int64(512 << 20) // planned startup peak
	lifecycleSteady = int64(256 << 20) // planned steady residency
	// lifecycleMeasuredPeakRSS is the stubbed measured process peak RSS. It is
	// deliberately different from lifecyclePeak so a receipt that stuffs the
	// planned peak into peak_rss_bytes is caught.
	lifecycleMeasuredPeakRSS = int64(777 << 20)
	lifecycleMeasuredRSS     = int64(300 << 20)
	lifecycleMeasuredSwap    = int64(3 << 20)
)

// metalLifecycleProbes is one stubbed answer for the three receipt probes.
type metalLifecycleProbes struct {
	unified, probed   bool
	swap              int64
	haveSwap          bool
	rss, peak         int64
	haveRSS, havePeak bool
}

// measuredLifecycleProbes reports a probed unified-memory device and every
// measurement available.
var measuredLifecycleProbes = metalLifecycleProbes{
	unified: true, probed: true,
	swap: lifecycleMeasuredSwap, haveSwap: true,
	rss: lifecycleMeasuredRSS, peak: lifecycleMeasuredPeakRSS, haveRSS: true, havePeak: true,
}

// stubMetalAdmissionSeams swaps the host memory reader, the three receipt
// probes, and the receipt sink for the duration of the test and returns the
// buffer that captures receipt lines.
func stubMetalAdmissionSeams(t *testing.T, read func() (memgate.Memory, error), p metalLifecycleProbes) *bytes.Buffer {
	t.Helper()
	origRead, origOut := serveReadMemory, serveAdmissionReceiptOut
	origTopo, origSwap, origRSS := serveUnifiedMemoryTopology, serveSwapUsedBytes, serveProcessRSS
	t.Cleanup(func() {
		serveReadMemory, serveAdmissionReceiptOut = origRead, origOut
		serveUnifiedMemoryTopology, serveSwapUsedBytes, serveProcessRSS = origTopo, origSwap, origRSS
	})
	var buf bytes.Buffer
	serveReadMemory = read
	serveAdmissionReceiptOut = &buf
	serveUnifiedMemoryTopology = func() (bool, bool) { return p.unified, p.probed }
	serveSwapUsedBytes = func() (int64, bool) { return p.swap, p.haveSwap }
	serveProcessRSS = func() (int64, int64, bool, bool) { return p.rss, p.peak, p.haveRSS, p.havePeak }
	return &buf
}

// metalLifecycleEnv points the seam at a fresh reservation dir and lease path,
// pins the admission mode, and injects the memory plan. No dev policy and no
// state envelope, so the plan is exactly the injected bytes.
func metalLifecycleEnv(t *testing.T, mode string, peak, steady int64) (resDir, leasePath string) {
	t.Helper()
	resDir = filepath.Join(t.TempDir(), "reservations")
	leasePath = filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_RESERVATION_DIR", resDir)
	t.Setenv("FAK_GPU_LEASE", leasePath)
	t.Setenv("FAK_NATIVE_ADMISSION", mode)
	t.Setenv("FAK_ADMISSION_POLICY", "")
	t.Setenv("FAK_ADMISSION_STATE_ENVELOPE", "")
	t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", strconv.FormatInt(peak, 10))
	t.Setenv("FAK_TEST_STEADY_BYTES", strconv.FormatInt(steady, 10))
	return resDir, leasePath
}

// lifecycleHost returns a host reader with the given total and allocatable
// bytes and no compressor or wired occupancy (normal pressure).
func lifecycleHost(total, avail int64) func() (memgate.Memory, error) {
	return func() (memgate.Memory, error) {
		return memgate.Memory{TotalBytes: total, FreeBytes: avail, AvailableBytes: avail}, nil
	}
}

// decodedMetalReceipt is one receipt line decoded both into the typed receipt
// and into raw keys, so key presence (measured-or-absent) is assertable.
type decodedMetalReceipt struct {
	rc  metalAdmissionReceipt
	raw map[string]json.RawMessage
}

func decodeMetalAdmissionReceipts(t *testing.T, buf *bytes.Buffer) []decodedMetalReceipt {
	t.Helper()
	var out []decodedMetalReceipt
	for _, line := range strings.Split(buf.String(), "\n") {
		if line == "" {
			continue
		}
		body, ok := strings.CutPrefix(line, metalAdmissionReceiptPrefix)
		if !ok {
			t.Fatalf("receipt sink carried a line without the receipt prefix: %q", line)
		}
		var d decodedMetalReceipt
		if err := json.Unmarshal([]byte(body), &d.rc); err != nil {
			t.Fatalf("decode receipt %q: %v", body, err)
		}
		if err := json.Unmarshal([]byte(body), &d.raw); err != nil {
			t.Fatalf("decode receipt keys %q: %v", body, err)
		}
		out = append(out, d)
	}
	return out
}

func metalReceiptStages(rs []decodedMetalReceipt) []string {
	stages := make([]string, len(rs))
	for i, r := range rs {
		stages[i] = r.rc.Stage
	}
	return stages
}

// requireLeaseBusy asserts the GPU lease at path is held by someone right now.
func requireLeaseBusy(t *testing.T, path, when string) {
	t.Helper()
	if err := acquireMetalLeaseNoWait(path); !errors.Is(err, gpulease.ErrBusy) {
		t.Errorf("%s: GPU lease acquire = %v, want ErrBusy (lease held)", when, err)
	}
}

// requireLeaseFree asserts the GPU lease at path can be taken right now.
func requireLeaseFree(t *testing.T, path, when string) {
	t.Helper()
	if err := acquireMetalLeaseNoWait(path); err != nil {
		t.Errorf("%s: GPU lease still held: %v", when, err)
	}
}

// requireCommonReceiptFacts checks the identity facts every receipt stage must
// carry: schema, engine, model, plan, owner, and that a device buffer is never
// reported host-addressable (the key is present and false).
func requireCommonReceiptFacts(t *testing.T, r decodedMetalReceipt, model, mode string) {
	t.Helper()
	rc := r.rc
	if rc.Schema != localadmission.MetalReservationReceiptSchema {
		t.Errorf("%s receipt schema = %q, want %q", rc.Stage, rc.Schema, localadmission.MetalReservationReceiptSchema)
	}
	if rc.Engine != "fak-native" || rc.Model != model {
		t.Errorf("%s receipt engine/model = %q/%q, want fak-native/%q", rc.Stage, rc.Engine, rc.Model, model)
	}
	if rc.AdmissionMode != mode {
		t.Errorf("%s receipt admission_mode = %q, want %q", rc.Stage, rc.AdmissionMode, mode)
	}
	if want := (localadmission.MemoryPlan{StartupPeakBytes: lifecyclePeak, SteadyBytes: lifecycleSteady}); rc.PlannedBytes != want {
		t.Errorf("%s receipt planned_bytes = %+v, want %+v", rc.Stage, rc.PlannedBytes, want)
	}
	if rc.OwnerPID != os.Getpid() {
		t.Errorf("%s receipt owner_pid = %d, want %d", rc.Stage, rc.OwnerPID, os.Getpid())
	}
	if got := string(r.raw["host_addressable"]); got != "false" {
		t.Errorf("%s receipt host_addressable = %q, want an explicit false", rc.Stage, got)
	}
}

// requireMeasured checks the measured list names exactly the enabled probe
// fields, that each named field carries the stubbed measurement, and that a
// field whose probe was unavailable is absent rather than a reported zero.
func requireMeasured(t *testing.T, r decodedMetalReceipt, p metalLifecycleProbes) {
	t.Helper()
	type field struct {
		key  string
		have bool
		want int64
		got  int64
	}
	fields := []field{
		{"swap_bytes", p.haveSwap, p.swap, r.rc.SwapBytes},
		{"rss_bytes", p.haveRSS, p.rss, r.rc.RSSBytes},
		{"peak_rss_bytes", p.havePeak, p.peak, r.rc.PeakRSSBytes},
	}
	var wantMeasured []string
	for _, f := range fields {
		_, present := r.raw[f.key]
		if f.have {
			wantMeasured = append(wantMeasured, f.key)
			if f.got != f.want {
				t.Errorf("%s receipt %s = %d, want the measured %d", r.rc.Stage, f.key, f.got, f.want)
			}
		} else if present || f.got != 0 {
			t.Errorf("%s receipt reports %s=%d although its probe was unavailable; want the key absent", r.rc.Stage, f.key, f.got)
		}
	}
	got := slices.Clone(r.rc.Measured)
	slices.Sort(got)
	slices.Sort(wantMeasured)
	if !slices.Equal(got, wantMeasured) {
		t.Errorf("%s receipt measured = %v, want %v", r.rc.Stage, r.rc.Measured, wantMeasured)
	}
	if _, present := r.raw["measured"]; len(wantMeasured) == 0 && present {
		t.Errorf("%s receipt carries a measured key with no probe available: %s", r.rc.Stage, r.raw["measured"])
	}
}

// TestMetalAdmissionRefusalReceiptNamesTypedReasonBeforeLoader covers the
// refusal half of #9587 through the receipt: an unprobeable host sample
// (unknown pressure) and critical host pressure without the dev policy, in both
// reserving modes, refuse before the loader with a typed reason and a remedy,
// write no ledger, hand the lease back, and emit exactly one "refuse" REJECT
// receipt. The returned release is a no-op that never runs Teardown.
func TestMetalAdmissionRefusalReceiptNamesTypedReasonBeforeLoader(t *testing.T) {
	hosts := []struct {
		name         string
		read         func() (memgate.Memory, error)
		wantReason   string
		wantPressure localadmission.Pressure
	}{
		{
			name:         "read_error",
			read:         func() (memgate.Memory, error) { return memgate.Memory{}, errors.New("host memory unavailable") },
			wantReason:   "pressure_unknown",
			wantPressure: localadmission.PressureUnknown,
		},
		{
			name:         "zero_total",
			read:         lifecycleHost(0, 8<<30),
			wantReason:   "pressure_unknown",
			wantPressure: localadmission.PressureUnknown,
		},
		{
			// 25% of the pool is compressor-held: AdmissionSampleFor classifies
			// the occupancy as critical.
			name: "critical_occupancy",
			read: func() (memgate.Memory, error) {
				return memgate.Memory{TotalBytes: 16 << 30, FreeBytes: 8 << 30, AvailableBytes: 8 << 30, CompressedBytes: 4 << 30}, nil
			},
			wantReason:   "pressure_critical",
			wantPressure: localadmission.PressureCritical,
		},
		{
			// Plenty of allocatable memory, but the OS reports critical pressure.
			name: "critical_os_signal",
			read: func() (memgate.Memory, error) {
				return memgate.Memory{TotalBytes: 16 << 30, FreeBytes: 12 << 30, AvailableBytes: 12 << 30, Pressure: memgate.PressureCritical, PressureKnown: true}, nil
			},
			wantReason:   "pressure_critical",
			wantPressure: localadmission.PressureCritical,
		},
	}
	for _, mode := range []string{"default", "aggregate"} {
		for _, host := range hosts {
			t.Run(mode+"/"+host.name, func(t *testing.T) {
				resDir, leasePath := metalLifecycleEnv(t, mode, lifecyclePeak, lifecycleSteady)
				buf := stubMetalAdmissionSeams(t, host.read, measuredLifecycleProbes)

				loads, teardowns := 0, 0
				release, err := admitLocalMetalModel(true, "refused-model.gguf", gpulease.Options{}, metalAdmissionSpec{
					Load:     func() bool { loads++; return true },
					Teardown: func() error { teardowns++; return nil },
				})
				if err == nil {
					release()
					t.Fatalf("admission accepted host %s; want refusal %s before the loader", host.name, host.wantReason)
				}
				if !strings.Contains(err.Error(), "local memory reservation refused: "+host.wantReason) {
					t.Errorf("refusal %q does not name %q", err, host.wantReason)
				}
				if loads != 0 {
					t.Errorf("loader calls = %d, want 0", loads)
				}
				if _, statErr := os.Stat(filepath.Join(resDir, "reservations.json")); !errors.Is(statErr, os.ErrNotExist) {
					t.Errorf("refusal created a reservation ledger: stat err=%v", statErr)
				}
				requireLeaseFree(t, leasePath, "after refusal")

				receipts := decodeMetalAdmissionReceipts(t, buf)
				if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"refuse"}) {
					t.Fatalf("receipt stages = %v, want [refuse]", stages)
				}
				r := receipts[0]
				requireCommonReceiptFacts(t, r, "refused-model.gguf", mode)
				requireMeasured(t, r, measuredLifecycleProbes)
				if r.rc.Verdict != "REJECT" || r.rc.Reason != host.wantReason {
					t.Errorf("refuse receipt verdict/reason = %s/%s, want REJECT/%s", r.rc.Verdict, r.rc.Reason, host.wantReason)
				}
				if r.rc.RemedyHint == "" || !strings.Contains(err.Error(), r.rc.RemedyHint) {
					t.Errorf("refuse receipt remedy_hint %q is empty or differs from the error %q", r.rc.RemedyHint, err)
				}
				if r.rc.Pressure != host.wantPressure {
					t.Errorf("refuse receipt pressure = %q, want %q", r.rc.Pressure, host.wantPressure)
				}
				if r.rc.ReservationID != "" || r.rc.Cleanup != "none" || r.rc.Phase != "" {
					t.Errorf("refuse receipt reservation_id/cleanup/phase = %q/%q/%q, want none reserved", r.rc.ReservationID, r.rc.Cleanup, r.rc.Phase)
				}

				release()
				if teardowns != 0 {
					t.Errorf("refusal release ran Teardown %d time(s), want 0 (nothing was loaded)", teardowns)
				}
				if extra := decodeMetalAdmissionReceipts(t, buf); len(extra) != 1 {
					t.Errorf("refusal release emitted more receipts: stages=%v", metalReceiptStages(extra))
				}
			})
		}
	}
}

// TestMetalAdmissionExclusiveRollbackLoadsUnprobeableHostUnderHeldLease pins
// the conservative rollback: FAK_NATIVE_ADMISSION=exclusive serializes
// residency with the GPU lease alone, so the same unprobeable host sample the
// reserving modes refuse still loads, writes no ledger row, and keeps the lease
// held until release() (which runs Teardown before handing the lease back).
func TestMetalAdmissionExclusiveRollbackLoadsUnprobeableHostUnderHeldLease(t *testing.T) {
	for _, host := range []struct {
		name string
		read func() (memgate.Memory, error)
	}{
		{"read_error", func() (memgate.Memory, error) { return memgate.Memory{}, errors.New("host memory unavailable") }},
		{"zero_total", lifecycleHost(0, 8<<30)},
	} {
		t.Run(host.name, func(t *testing.T) {
			resDir, leasePath := metalLifecycleEnv(t, "exclusive", lifecyclePeak, lifecycleSteady)
			buf := stubMetalAdmissionSeams(t, host.read, measuredLifecycleProbes)

			loads, teardowns := 0, 0
			leaseHeldInTeardown := false
			release, err := admitLocalMetalModel(true, "exclusive-model.gguf", gpulease.Options{}, metalAdmissionSpec{
				Load: func() bool { loads++; return true },
				Teardown: func() error {
					teardowns++
					leaseHeldInTeardown = errors.Is(acquireMetalLeaseNoWait(leasePath), gpulease.ErrBusy)
					return nil
				},
			})
			if err != nil {
				t.Fatalf("exclusive rollback refused an unprobeable host: %v", err)
			}
			if loads != 1 {
				t.Fatalf("loader calls = %d, want 1", loads)
			}
			if rows := readMetalAdmissionLedger(t, resDir); len(rows) != 0 {
				t.Errorf("exclusive rollback wrote reservation rows: %+v", rows)
			}
			requireLeaseBusy(t, leasePath, "exclusive residency before release")

			release()
			if teardowns != 1 || !leaseHeldInTeardown {
				t.Errorf("teardowns=%d leaseHeldInTeardown=%v, want Teardown once while the lease is still held", teardowns, leaseHeldInTeardown)
			}
			requireLeaseFree(t, leasePath, "after exclusive release")
			if _, statErr := os.Stat(filepath.Join(resDir, "reservations.json")); !errors.Is(statErr, os.ErrNotExist) {
				t.Errorf("exclusive rollback created a reservation ledger: stat err=%v", statErr)
			}

			receipts := decodeMetalAdmissionReceipts(t, buf)
			if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"admit", "release"}) {
				t.Fatalf("receipt stages = %v, want [admit release]", stages)
			}
			for _, r := range receipts {
				requireCommonReceiptFacts(t, r, "exclusive-model.gguf", "exclusive")
				if r.rc.ReservationID != "" {
					t.Errorf("%s receipt reservation_id = %q, want none in exclusive mode", r.rc.Stage, r.rc.ReservationID)
				}
			}
			admit, rel := receipts[0].rc, receipts[1].rc
			if admit.Verdict != "ADMIT" || admit.Reason != "exclusive_lease" || !admit.LeaseRetained || admit.Pressure != localadmission.PressureUnknown {
				t.Errorf("admit receipt verdict/reason/lease/pressure = %s/%s/%v/%s, want ADMIT/exclusive_lease/true/unknown", admit.Verdict, admit.Reason, admit.LeaseRetained, admit.Pressure)
			}
			if rel.Verdict != "RELEASED" || rel.Cleanup != "released" || rel.Phase != "released" {
				t.Errorf("release receipt verdict/cleanup/phase = %s/%s/%s, want RELEASED/released/released", rel.Verdict, rel.Cleanup, rel.Phase)
			}
		})
	}
}

// TestMetalAdmissionFailedLoadReleasesStartupReservationWithoutSteady pins the
// Load()==false path: during the load the ledger holds the startup peak under
// the lease; once Load reports failure the reservation is gone without ever
// being marked steady, the lease is free, the returned release is a no-op
// (no Teardown, no receipt), and the receipts are exactly admit then
// load_failed.
func TestMetalAdmissionFailedLoadReleasesStartupReservationWithoutSteady(t *testing.T) {
	for _, mode := range []string{"default", "aggregate"} {
		t.Run(mode, func(t *testing.T) {
			resDir, leasePath := metalLifecycleEnv(t, mode, lifecyclePeak, lifecycleSteady)
			buf := stubMetalAdmissionSeams(t, lifecycleHost(16<<30, 8<<30), measuredLifecycleProbes)

			var inLoad []localadmission.Reservation
			loads, teardowns := 0, 0
			leaseHeldInLoad := false
			release, err := admitLocalMetalModel(true, "failing-model.gguf", gpulease.Options{}, metalAdmissionSpec{
				Load: func() bool {
					loads++
					inLoad = readMetalAdmissionLedger(t, resDir)
					leaseHeldInLoad = errors.Is(acquireMetalLeaseNoWait(leasePath), gpulease.ErrBusy)
					return false
				},
				Teardown: func() error { teardowns++; return nil },
			})
			if err != nil {
				t.Fatalf("failed load must return a nil error so the caller's own load-failure handling runs: %v", err)
			}
			if release == nil {
				t.Fatal("failed load returned a nil release")
			}
			if loads != 1 {
				t.Fatalf("loader calls = %d, want 1", loads)
			}
			if len(inLoad) != 1 || inLoad[0].Phase != "startup" || inLoad[0].HeldBytes != lifecyclePeak || inLoad[0].OwnerPID != os.Getpid() {
				t.Fatalf("ledger during load = %+v, want one startup row holding the %d-byte peak for pid %d", inLoad, lifecyclePeak, os.Getpid())
			}
			if !leaseHeldInLoad {
				t.Error("GPU lease was not held while the loader ran")
			}
			if rows := readMetalAdmissionLedger(t, resDir); len(rows) != 0 {
				t.Errorf("failed load retained reservation rows (never steady): %+v", rows)
			}
			requireLeaseFree(t, leasePath, "after failed load")

			release()
			release()
			if teardowns != 0 {
				t.Errorf("failed-load release ran Teardown %d time(s), want 0", teardowns)
			}

			receipts := decodeMetalAdmissionReceipts(t, buf)
			if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"admit", "load_failed"}) {
				t.Fatalf("receipt stages = %v, want [admit load_failed] (never steady, no-op release)", stages)
			}
			for _, r := range receipts {
				requireCommonReceiptFacts(t, r, "failing-model.gguf", mode)
				requireMeasured(t, r, measuredLifecycleProbes)
				if r.rc.ReservationID != inLoad[0].ID {
					t.Errorf("%s receipt reservation_id = %q, want ledger id %q", r.rc.Stage, r.rc.ReservationID, inLoad[0].ID)
				}
			}
			failed := receipts[1].rc
			if failed.Verdict != "RELEASED" || failed.Reason != "load_failed" || failed.Cleanup != "released" || failed.Phase != "released" {
				t.Errorf("load_failed receipt verdict/reason/cleanup/phase = %s/%s/%s/%s, want RELEASED/load_failed/released/released",
					failed.Verdict, failed.Reason, failed.Cleanup, failed.Phase)
			}
			if failed.ReservedBytes != 0 {
				t.Errorf("load_failed receipt reserved_bytes = %d, want 0 (no other owner)", failed.ReservedBytes)
			}
		})
	}
}

// TestMetalAdmissionSteadyResidencyTearsDownBeforeReleasingCapacity pins the
// successful lifecycle: startup peak held during Load, steady residency after,
// and a release that runs Teardown exactly once while the reservation row (and
// a retained lease) still exist, then empties the ledger and frees the lease.
// Receipts are admit, steady, release with the probed topology, a false
// host_addressable, the planned bytes, this pid, the ledger's reservation id,
// and measured values that are never the planned peak.
func TestMetalAdmissionSteadyResidencyTearsDownBeforeReleasingCapacity(t *testing.T) {
	const total, avail = int64(16 << 30), int64(8 << 30)
	for _, tc := range []struct {
		mode         string
		wantRetained bool
	}{
		{"default", true},
		// A plan under half the allocatable pool coexists: aggregate drops the
		// lease after the load and keeps only the reservation.
		{"aggregate", false},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			resDir, leasePath := metalLifecycleEnv(t, tc.mode, lifecyclePeak, lifecycleSteady)
			buf := stubMetalAdmissionSeams(t, lifecycleHost(total, avail), measuredLifecycleProbes)

			var inLoad, inTeardown []localadmission.Reservation
			teardowns := 0
			leaseHeldInTeardown := false
			release, err := admitLocalMetalModel(true, "resident-model.gguf", gpulease.Options{}, metalAdmissionSpec{
				Load: func() bool {
					inLoad = readMetalAdmissionLedger(t, resDir)
					return true
				},
				Teardown: func() error {
					teardowns++
					inTeardown = readMetalAdmissionLedger(t, resDir)
					leaseHeldInTeardown = errors.Is(acquireMetalLeaseNoWait(leasePath), gpulease.ErrBusy)
					return nil
				},
			})
			if err != nil {
				t.Fatalf("admission refused a fitting plan: %v", err)
			}
			t.Cleanup(release)

			if len(inLoad) != 1 || inLoad[0].Phase != "startup" || inLoad[0].HeldBytes != lifecyclePeak ||
				inLoad[0].StartupPeakBytes != lifecyclePeak || inLoad[0].SteadyBytes != lifecycleSteady || inLoad[0].OwnerPID != os.Getpid() {
				t.Fatalf("ledger during load = %+v, want one startup row holding the peak", inLoad)
			}
			id := inLoad[0].ID
			if rows := readMetalAdmissionLedger(t, resDir); len(rows) != 1 || rows[0].ID != id || rows[0].Phase != "steady" || rows[0].HeldBytes != lifecycleSteady {
				t.Fatalf("ledger after load = %+v, want %s steady holding %d", rows, id, lifecycleSteady)
			}
			if tc.wantRetained {
				requireLeaseBusy(t, leasePath, "resident default-mode model")
			} else {
				requireLeaseFree(t, leasePath, "resident small aggregate-mode model")
			}

			release()
			release()
			if teardowns != 1 {
				t.Fatalf("Teardown ran %d time(s) across two releases, want exactly 1", teardowns)
			}
			if len(inTeardown) != 1 || inTeardown[0].ID != id || inTeardown[0].Phase != "steady" {
				t.Errorf("ledger during Teardown = %+v, want the steady row still reserved until the weights are freed", inTeardown)
			}
			if leaseHeldInTeardown != tc.wantRetained {
				t.Errorf("lease held during Teardown = %v, want %v", leaseHeldInTeardown, tc.wantRetained)
			}
			if rows := readMetalAdmissionLedger(t, resDir); len(rows) != 0 {
				t.Errorf("ledger after release = %+v, want empty", rows)
			}
			requireLeaseFree(t, leasePath, "after release")

			receipts := decodeMetalAdmissionReceipts(t, buf)
			if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"admit", "steady", "release"}) {
				t.Fatalf("receipt stages = %v, want [admit steady release] (second release must not re-emit)", stages)
			}
			for _, r := range receipts {
				requireCommonReceiptFacts(t, r, "resident-model.gguf", tc.mode)
				requireMeasured(t, r, measuredLifecycleProbes)
				if r.rc.Topology != "apple-unified-memory" || !r.rc.HostUnified || !r.rc.TopologyProbed {
					t.Errorf("%s receipt topology/unified/probed = %q/%v/%v, want the probed apple-unified-memory", r.rc.Stage, r.rc.Topology, r.rc.HostUnified, r.rc.TopologyProbed)
				}
				if r.rc.ReservationID != id {
					t.Errorf("%s receipt reservation_id = %q, want ledger id %q", r.rc.Stage, r.rc.ReservationID, id)
				}
				if r.rc.PeakRSSBytes == lifecyclePeak {
					t.Errorf("%s receipt peak_rss_bytes is the planned peak %d, want the measured %d", r.rc.Stage, lifecyclePeak, lifecycleMeasuredPeakRSS)
				}
				if r.rc.LeaseRetained != tc.wantRetained {
					t.Errorf("%s receipt lease_retained = %v, want %v", r.rc.Stage, r.rc.LeaseRetained, tc.wantRetained)
				}
				if r.rc.TotalBytes != total || r.rc.LiveAllocatableBytes != avail || r.rc.Pressure != localadmission.PressureNormal {
					t.Errorf("%s receipt total/live_allocatable/pressure = %d/%d/%s, want %d/%d/normal", r.rc.Stage, r.rc.TotalBytes, r.rc.LiveAllocatableBytes, r.rc.Pressure, total, avail)
				}
				if _, present := r.raw["teardown_error"]; present {
					t.Errorf("%s receipt carries teardown_error %s for a clean teardown", r.rc.Stage, r.raw["teardown_error"])
				}
			}
			admit, steady, rel := receipts[0].rc, receipts[1].rc, receipts[2].rc
			if admit.Verdict != "ADMIT" || admit.Reason != "reserved" || admit.Phase != "startup" || admit.Cleanup != "active" {
				t.Errorf("admit receipt verdict/reason/phase/cleanup = %s/%s/%s/%s, want ADMIT/reserved/startup/active", admit.Verdict, admit.Reason, admit.Phase, admit.Cleanup)
			}
			if admit.ReservedBytes != lifecyclePeak || admit.AllocatableBytes != avail || admit.AvailableBytes != avail-lifecyclePeak {
				t.Errorf("admit receipt reserved/allocatable/available = %d/%d/%d, want %d/%d/%d",
					admit.ReservedBytes, admit.AllocatableBytes, admit.AvailableBytes, lifecyclePeak, avail, avail-lifecyclePeak)
			}
			if steady.Verdict != "ADMIT" || steady.Reason != "steady" || steady.Phase != "steady" {
				t.Errorf("steady receipt verdict/reason/phase = %s/%s/%s, want ADMIT/steady/steady", steady.Verdict, steady.Reason, steady.Phase)
			}
			if rel.Verdict != "RELEASED" || rel.Phase != "released" || rel.Cleanup != "released" {
				t.Errorf("release receipt verdict/phase/cleanup = %s/%s/%s, want RELEASED/released/released", rel.Verdict, rel.Phase, rel.Cleanup)
			}
			if rel.ReservedBytes != 0 || rel.AvailableBytes != avail {
				t.Errorf("release receipt reserved/available = %d/%d, want 0/%d (no other owner holds capacity)", rel.ReservedBytes, rel.AvailableBytes, avail)
			}
		})
	}
}

// TestMetalAdmissionReleaseReceiptRecordsTeardownError pins that a teardown
// failure (e.g. sessions still attached) is recorded on the release receipt
// while the reservation and lease are still handed back.
func TestMetalAdmissionReleaseReceiptRecordsTeardownError(t *testing.T) {
	resDir, leasePath := metalLifecycleEnv(t, "default", lifecyclePeak, lifecycleSteady)
	buf := stubMetalAdmissionSeams(t, lifecycleHost(16<<30, 8<<30), measuredLifecycleProbes)
	const teardownMsg = "close weights: 2 sessions still attached"

	release, err := admitLocalMetalModel(true, "teardown-error.gguf", gpulease.Options{}, metalAdmissionSpec{
		Load:     func() bool { return true },
		Teardown: func() error { return errors.New(teardownMsg) },
	})
	if err != nil {
		t.Fatalf("admission refused a fitting plan: %v", err)
	}
	if rows := readMetalAdmissionLedger(t, resDir); len(rows) != 1 || rows[0].Phase != "steady" {
		t.Fatalf("ledger after load = %+v, want one steady row", rows)
	}
	release()
	if rows := readMetalAdmissionLedger(t, resDir); len(rows) != 0 {
		t.Errorf("teardown error kept the reservation: %+v", rows)
	}
	requireLeaseFree(t, leasePath, "after release with a teardown error")

	receipts := decodeMetalAdmissionReceipts(t, buf)
	if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"admit", "steady", "release"}) {
		t.Fatalf("receipt stages = %v, want [admit steady release]", stages)
	}
	rel := receipts[2].rc
	if rel.TeardownError != teardownMsg {
		t.Errorf("release receipt teardown_error = %q, want %q", rel.TeardownError, teardownMsg)
	}
	if rel.Verdict != "RELEASED" || rel.Cleanup != "released" {
		t.Errorf("release receipt verdict/cleanup = %s/%s, want RELEASED/released (the reservation is still returned)", rel.Verdict, rel.Cleanup)
	}
	for _, r := range receipts[:2] {
		if r.rc.TeardownError != "" {
			t.Errorf("%s receipt carries teardown_error %q before any teardown", r.rc.Stage, r.rc.TeardownError)
		}
	}
}

// TestMetalAdmissionAggregateCoResidencyAdmitsAfterRelease pins aggregate
// co-residency: two small models reserve side by side (each dropping the
// lease), a third whose startup peak exceeds what remains is refused with
// aggregate_capacity before its loader, and once one resident releases the
// same third admission fits.
func TestMetalAdmissionAggregateCoResidencyAdmitsAfterRelease(t *testing.T) {
	const (
		total  = int64(8 << 30)
		avail  = int64(3 << 30)
		peak   = int64(5 << 28) // 1.25 GiB: under avail/2, so the lease is dropped
		steady = int64(1 << 30)
	)
	resDir, leasePath := metalLifecycleEnv(t, "aggregate", peak, steady)
	buf := stubMetalAdmissionSeams(t, lifecycleHost(total, avail), measuredLifecycleProbes)

	admit := func(name string, loads *int) func() {
		t.Helper()
		release, err := admitLocalMetalModel(true, name, gpulease.Options{}, metalAdmissionSpec{
			Load: func() bool { *loads++; return true },
		})
		if err != nil {
			t.Fatalf("%s: aggregate admission refused: %v", name, err)
		}
		t.Cleanup(release)
		requireLeaseFree(t, leasePath, name+" resident in aggregate mode")
		return release
	}
	ledgerIDs := func() []string {
		var ids []string
		for _, r := range readMetalAdmissionLedger(t, resDir) {
			if r.Phase != "steady" || r.HeldBytes != steady {
				t.Errorf("co-resident row %+v, want steady holding %d", r, steady)
			}
			ids = append(ids, r.ID)
		}
		slices.Sort(ids)
		return ids
	}

	var loads1, loads2, loads3 int
	release1 := admit("small-1.gguf", &loads1)
	firstIDs := ledgerIDs()
	release2 := admit("small-2.gguf", &loads2)
	bothIDs := ledgerIDs()
	if loads1 != 1 || loads2 != 1 || len(firstIDs) != 1 || len(bothIDs) != 2 {
		t.Fatalf("co-residency loads=%d/%d ledger=%v then %v, want two steady rows", loads1, loads2, firstIDs, bothIDs)
	}
	secondID := bothIDs[0]
	if secondID == firstIDs[0] {
		secondID = bothIDs[1]
	}

	// 2 GiB steady is held; 1 GiB remains, below the 1.25 GiB startup peak.
	if _, err := admitLocalMetalModel(true, "small-3.gguf", gpulease.Options{}, metalAdmissionSpec{
		Load: func() bool { loads3++; return true },
	}); err == nil || !strings.Contains(err.Error(), "aggregate_capacity") {
		t.Fatalf("third admission err = %v, want an aggregate_capacity refusal", err)
	}
	if loads3 != 0 {
		t.Fatalf("third loader ran %d time(s) on an aggregate_capacity refusal, want 0", loads3)
	}
	if ids := ledgerIDs(); !slices.Equal(ids, bothIDs) {
		t.Fatalf("refused admission changed the ledger: %v, want %v", ids, bothIDs)
	}
	requireLeaseFree(t, leasePath, "after the aggregate_capacity refusal")

	release1()
	if ids := ledgerIDs(); len(ids) != 1 || ids[0] != secondID {
		t.Fatalf("ledger after releasing the first resident = %v, want only %s", ids, secondID)
	}
	release3 := admit("small-3.gguf", &loads3)
	if loads3 != 1 || len(ledgerIDs()) != 2 {
		t.Fatalf("third admission after a release: loads=%d ledger=%v, want admitted beside the second", loads3, ledgerIDs())
	}
	release2()
	release3()
	if ids := ledgerIDs(); len(ids) != 0 {
		t.Errorf("ledger after releasing every resident = %v, want empty", ids)
	}

	receipts := decodeMetalAdmissionReceipts(t, buf)
	want := []string{"admit", "steady", "admit", "steady", "refuse", "release", "admit", "steady", "release", "release"}
	if stages := metalReceiptStages(receipts); !slices.Equal(stages, want) {
		t.Fatalf("receipt stages = %v, want %v", stages, want)
	}
	if second := receipts[2].rc; second.ReservedBytes != steady+peak || second.LeaseRetained {
		t.Errorf("second admit receipt reserved/lease_retained = %d/%v, want %d (first steady + own peak)/false", second.ReservedBytes, second.LeaseRetained, steady+peak)
	}
	if refuse := receipts[4].rc; refuse.Verdict != "REJECT" || refuse.Reason != "aggregate_capacity" || refuse.ReservedBytes != 2*steady || refuse.AvailableBytes != avail-2*steady || refuse.RemedyHint == "" {
		t.Errorf("refuse receipt verdict/reason/reserved/available/hint = %s/%s/%d/%d/%q, want REJECT/aggregate_capacity/%d/%d/non-empty",
			refuse.Verdict, refuse.Reason, refuse.ReservedBytes, refuse.AvailableBytes, refuse.RemedyHint, 2*steady, avail-2*steady)
	}
	// The first release leaves the second resident's steady bytes reserved.
	if first := receipts[5].rc; first.ReservationID != firstIDs[0] || first.ReservedBytes != steady || first.AvailableBytes != avail-steady {
		t.Errorf("first release receipt id/reserved/available = %s/%d/%d, want %s/%d/%d", first.ReservationID, first.ReservedBytes, first.AvailableBytes, firstIDs[0], steady, avail-steady)
	}
}

// TestMetalAdmissionReceiptTopologyAndMeasurementsAreProbedNotAssumed pins
// that the receipt reports the device topology the probe answered (or
// "unprobed" when none did), never marks a device buffer host-addressable, and
// omits every measurement whose probe was unavailable instead of reporting a
// zero or substituting a planned value.
func TestMetalAdmissionReceiptTopologyAndMeasurementsAreProbedNotAssumed(t *testing.T) {
	for _, tc := range []struct {
		name         string
		probes       metalLifecycleProbes
		wantTopology string
	}{
		{"unified_all_measured", measuredLifecycleProbes, "apple-unified-memory"},
		{"discrete_nothing_measured", metalLifecycleProbes{unified: false, probed: true}, "discrete"},
		{"unprobed_rss_only", metalLifecycleProbes{rss: lifecycleMeasuredRSS, haveRSS: true}, "unprobed"},
		{"unprobed_swap_and_peak", metalLifecycleProbes{swap: lifecycleMeasuredSwap, haveSwap: true, peak: lifecycleMeasuredPeakRSS, havePeak: true}, "unprobed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _ = metalLifecycleEnv(t, "default", lifecyclePeak, lifecycleSteady)
			buf := stubMetalAdmissionSeams(t, lifecycleHost(16<<30, 8<<30), tc.probes)

			release, err := admitLocalMetalModel(true, "probe-model.gguf", gpulease.Options{}, metalAdmissionSpec{Load: func() bool { return true }})
			if err != nil {
				t.Fatalf("admission refused a fitting plan: %v", err)
			}
			release()

			receipts := decodeMetalAdmissionReceipts(t, buf)
			if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"admit", "steady", "release"}) {
				t.Fatalf("receipt stages = %v, want [admit steady release]", stages)
			}
			for _, r := range receipts {
				requireCommonReceiptFacts(t, r, "probe-model.gguf", "default")
				requireMeasured(t, r, tc.probes)
				if r.rc.Topology != tc.wantTopology {
					t.Errorf("%s receipt topology = %q, want %q", r.rc.Stage, r.rc.Topology, tc.wantTopology)
				}
				wantUnified := tc.probes.probed && tc.probes.unified
				if r.rc.HostUnified != wantUnified || r.rc.TopologyProbed != tc.probes.probed {
					t.Errorf("%s receipt host_unified/topology_probed = %v/%v, want %v/%v", r.rc.Stage, r.rc.HostUnified, r.rc.TopologyProbed, wantUnified, tc.probes.probed)
				}
				if _, present := r.raw["topology_probed"]; present != tc.probes.probed {
					t.Errorf("%s receipt topology_probed key present=%v, want %v", r.rc.Stage, present, tc.probes.probed)
				}
			}
		})
	}
}
