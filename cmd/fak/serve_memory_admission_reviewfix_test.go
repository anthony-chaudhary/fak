package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/memgate"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// Issue #9587 review-fix witnesses for admitLocalMetalModel: deferred release
// while weight sessions are still attached, per-spec receipt routing, the
// exclusive rollback never touching the shared ledger, peer-aware early
// refusals, and the steady receipt's downshifted reserved bytes.

// reapedChildPID returns the pid of a child process that has already exited and
// been reaped, so the reservation store's liveness probe reports it dead.
func reapedChildPID(t *testing.T) int {
	t.Helper()
	child := exec.Command(os.Args[0], "-test.run=^$", "-test.count=1")
	if err := child.Run(); err != nil {
		t.Fatalf("run short-lived child: %v", err)
	}
	return child.Process.Pid
}

// seedMetalAdmissionLedger writes rows as a compact reservation ledger (the
// store itself writes MarshalIndent), so any store rewrite changes the bytes.
func seedMetalAdmissionLedger(t *testing.T, dir string, rows ...localadmission.Reservation) []byte {
	t.Helper()
	b, err := json.Marshal(struct {
		Schema       string                       `json:"schema"`
		Reservations []localadmission.Reservation `json:"reservations"`
	}{"fak-local-memory-reservations/2", rows})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "reservations.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	return b
}

// TestMetalAdmissionTeardownWithActiveSessionsDefersReleaseToExit pins the
// deferred release: when Teardown reports that weight sessions are still
// attached (directly or wrapped), the weights stay resident, so release() must
// keep the ledger row and a retained GPU lease, emit a DEFERRED release receipt,
// and a second release() must be a no-op.
func TestMetalAdmissionTeardownWithActiveSessionsDefersReleaseToExit(t *testing.T) {
	for _, tc := range []struct {
		name         string
		mode         string
		wantRetained bool
		err          error
	}{
		{"default/direct", "default", true, &fakmodel.WeightSessionsActiveError{Count: 1}},
		{"default/wrapped", "default", true, fmt.Errorf("close admitted weights: %w", &fakmodel.WeightSessionsActiveError{Count: 1})},
		// A small aggregate load already dropped the lease; the row alone stays.
		{"aggregate/direct", "aggregate", false, &fakmodel.WeightSessionsActiveError{Count: 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resDir, leasePath := metalLifecycleEnv(t, tc.mode, lifecyclePeak, lifecycleSteady)
			buf := stubMetalAdmissionSeams(t, lifecycleHost(16<<30, 8<<30), measuredLifecycleProbes)

			teardowns := 0
			release, err := admitLocalMetalModel(true, "sessions-attached.gguf", gpulease.Options{}, metalAdmissionSpec{
				Load:     func() bool { return true },
				Teardown: func() error { teardowns++; return tc.err },
			})
			if err != nil {
				t.Fatalf("admission refused a fitting plan: %v", err)
			}
			rows := readMetalAdmissionLedger(t, resDir)
			if len(rows) != 1 || rows[0].Phase != "steady" {
				t.Fatalf("ledger after load = %+v, want one steady row", rows)
			}

			for i := 1; i <= 2; i++ {
				release()
				if teardowns != 1 {
					t.Fatalf("after release #%d Teardown ran %d time(s), want exactly 1", i, teardowns)
				}
				after := readMetalAdmissionLedger(t, resDir)
				if len(after) != 1 || after[0].ID != rows[0].ID || after[0].Phase != "steady" || after[0].HeldBytes != lifecycleSteady {
					t.Fatalf("after release #%d ledger = %+v, want the steady row %s kept while sessions hold the weights", i, after, rows[0].ID)
				}
				if tc.wantRetained {
					requireLeaseBusy(t, leasePath, fmt.Sprintf("deferred release #%d", i))
				} else {
					requireLeaseFree(t, leasePath, fmt.Sprintf("deferred release #%d of an aggregate load", i))
				}
			}

			receipts := decodeMetalAdmissionReceipts(t, buf)
			if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"admit", "steady", "release"}) {
				t.Fatalf("receipt stages = %v, want [admit steady release] (the second release must not re-emit)", stages)
			}
			rel := receipts[2].rc
			if rel.Verdict != "DEFERRED" || rel.Reason != "weight_sessions_active" || rel.Cleanup != "deferred_to_exit" {
				t.Errorf("release receipt verdict/reason/cleanup = %s/%s/%s, want DEFERRED/weight_sessions_active/deferred_to_exit", rel.Verdict, rel.Reason, rel.Cleanup)
			}
			if rel.TeardownError != tc.err.Error() {
				t.Errorf("release receipt teardown_error = %q, want %q", rel.TeardownError, tc.err.Error())
			}
			if rel.ReservationID != rows[0].ID || rel.Phase == "released" {
				t.Errorf("release receipt reservation_id/phase = %s/%s, want the kept reservation %s, not released", rel.ReservationID, rel.Phase, rows[0].ID)
			}
		})
	}
}

// TestMetalAdmissionReceiptOutRoutesReceiptsToSpecWriter pins spec.ReceiptOut:
// when set, every lifecycle receipt (and a refusal receipt) goes to that writer
// and none reaches the default serveAdmissionReceiptOut sink.
func TestMetalAdmissionReceiptOutRoutesReceiptsToSpecWriter(t *testing.T) {
	t.Run("admit_steady_release", func(t *testing.T) {
		_, _ = metalLifecycleEnv(t, "default", lifecyclePeak, lifecycleSteady)
		global := stubMetalAdmissionSeams(t, lifecycleHost(16<<30, 8<<30), measuredLifecycleProbes)
		var own bytes.Buffer
		release, err := admitLocalMetalModel(true, "routed.gguf", gpulease.Options{}, metalAdmissionSpec{
			Load:       func() bool { return true },
			ReceiptOut: &own,
		})
		if err != nil {
			t.Fatalf("admission refused a fitting plan: %v", err)
		}
		release()
		if global.Len() != 0 {
			t.Errorf("default receipt sink received output although spec.ReceiptOut was set: %q", global.String())
		}
		if stages := metalReceiptStages(decodeMetalAdmissionReceipts(t, &own)); !slices.Equal(stages, []string{"admit", "steady", "release"}) {
			t.Errorf("spec.ReceiptOut stages = %v, want [admit steady release]", stages)
		}
	})
	t.Run("refuse", func(t *testing.T) {
		_, _ = metalLifecycleEnv(t, "default", lifecyclePeak, lifecycleSteady)
		global := stubMetalAdmissionSeams(t, func() (memgate.Memory, error) { return memgate.Memory{}, errors.New("host memory unavailable") }, measuredLifecycleProbes)
		var own bytes.Buffer
		if _, err := admitLocalMetalModel(true, "routed.gguf", gpulease.Options{}, metalAdmissionSpec{
			Load:       func() bool { t.Error("loader ran on an unprobeable host"); return true },
			ReceiptOut: &own,
		}); err == nil {
			t.Fatal("admission accepted an unprobeable host")
		}
		if global.Len() != 0 {
			t.Errorf("default receipt sink received the refusal although spec.ReceiptOut was set: %q", global.String())
		}
		receipts := decodeMetalAdmissionReceipts(t, &own)
		if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"refuse"}) || receipts[0].rc.Reason != "pressure_unknown" {
			t.Errorf("spec.ReceiptOut stages = %v, want one pressure_unknown refuse receipt", stages)
		}
	})
}

// TestMetalAdmissionExclusiveModeLeavesSeededLedgerByteIdentical pins that the
// exclusive rollback, which writes no reservation, also never reads, reaps, or
// rewrites the shared ledger: a seeded ledger holding a dead owner's row is
// byte-identical after admit+release and after a failed load. The reserving
// control proves the seeded row really is dead to the store (it gets reaped),
// so the byte-identity is not vacuous.
func TestMetalAdmissionExclusiveModeLeavesSeededLedgerByteIdentical(t *testing.T) {
	dead := localadmission.Reservation{
		ID: "dead-owner-row", OwnerPID: reapedChildPID(t),
		StartupPeakBytes: 2 << 30, SteadyBytes: 1 << 30, HeldBytes: 1 << 30, Phase: "steady",
	}
	for _, tc := range []struct {
		name   string
		loadOK bool
	}{
		{"admit_release", true},
		{"load_failed", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resDir, leasePath := metalLifecycleEnv(t, "exclusive", lifecyclePeak, lifecycleSteady)
			buf := stubMetalAdmissionSeams(t, lifecycleHost(16<<30, 8<<30), measuredLifecycleProbes)
			seeded := seedMetalAdmissionLedger(t, resDir, dead)

			release, err := admitLocalMetalModel(true, "exclusive-seeded.gguf", gpulease.Options{}, metalAdmissionSpec{
				Load: func() bool { return tc.loadOK },
			})
			if err != nil {
				t.Fatalf("exclusive admission refused: %v", err)
			}
			release()
			got, err := os.ReadFile(filepath.Join(resDir, "reservations.json"))
			if err != nil {
				t.Fatalf("read seeded ledger: %v", err)
			}
			if !bytes.Equal(got, seeded) {
				t.Errorf("exclusive mode rewrote the shared ledger:\n got %s\nwant %s", got, seeded)
			}
			requireLeaseFree(t, leasePath, "after exclusive "+tc.name)
			want := []string{"admit", "release"}
			if !tc.loadOK {
				want = []string{"admit", "load_failed"}
			}
			if stages := metalReceiptStages(decodeMetalAdmissionReceipts(t, buf)); !slices.Equal(stages, want) {
				t.Errorf("receipt stages = %v, want %v", stages, want)
			}
		})
	}

	t.Run("reserving_control_reaps_the_dead_row", func(t *testing.T) {
		resDir, _ := metalLifecycleEnv(t, "default", lifecyclePeak, lifecycleSteady)
		_ = stubMetalAdmissionSeams(t, lifecycleHost(16<<30, 8<<30), measuredLifecycleProbes)
		_ = seedMetalAdmissionLedger(t, resDir, dead)
		release, err := admitLocalMetalModel(true, "reserving-seeded.gguf", gpulease.Options{}, metalAdmissionSpec{Load: func() bool { return true }})
		if err != nil {
			t.Fatalf("reserving admission refused: %v", err)
		}
		defer release()
		for _, r := range readMetalAdmissionLedger(t, resDir) {
			if r.ID == dead.ID {
				t.Fatalf("the seeded row (owner pid %d) was not reaped by the reserving store; the exclusive byte-identity witness would be vacuous", dead.OwnerPID)
			}
		}
	})
}

// TestMetalAdmissionEarlyRefusalReportsLivePeersHeldBytes pins that a refusal
// the store decides before reading the ledger (unknown or critical pressure)
// still reports on its receipt what live peers actually hold, reaping dead
// owners, instead of the decision's zero.
func TestMetalAdmissionEarlyRefusalReportsLivePeersHeldBytes(t *testing.T) {
	const (
		livePeerHeld = int64(700 << 20)
		avail        = int64(8 << 30)
	)
	live := localadmission.Reservation{
		ID: "live-peer-row", OwnerPID: os.Getpid(),
		StartupPeakBytes: livePeerHeld, SteadyBytes: livePeerHeld, HeldBytes: livePeerHeld, Phase: "steady",
	}
	dead := localadmission.Reservation{
		ID: "dead-peer-row", OwnerPID: reapedChildPID(t),
		StartupPeakBytes: 5 << 30, SteadyBytes: 5 << 30, HeldBytes: 5 << 30, Phase: "steady",
	}
	for _, tc := range []struct {
		name          string
		read          func() (memgate.Memory, error)
		wantReason    string
		wantAvailable int64
	}{
		{
			name:       "pressure_unknown",
			read:       func() (memgate.Memory, error) { return memgate.Memory{}, errors.New("host memory unavailable") },
			wantReason: "pressure_unknown",
			// An unprobeable host has no admitted capacity to subtract from.
			wantAvailable: 0,
		},
		{
			name: "pressure_critical",
			read: func() (memgate.Memory, error) {
				return memgate.Memory{TotalBytes: 16 << 30, FreeBytes: avail, AvailableBytes: avail, Pressure: memgate.PressureCritical, PressureKnown: true}, nil
			},
			wantReason:    "pressure_critical",
			wantAvailable: avail - livePeerHeld,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resDir, _ := metalLifecycleEnv(t, "default", lifecyclePeak, lifecycleSteady)
			buf := stubMetalAdmissionSeams(t, tc.read, measuredLifecycleProbes)
			_ = seedMetalAdmissionLedger(t, resDir, live, dead)

			if _, err := admitLocalMetalModel(true, "early-refusal.gguf", gpulease.Options{}, metalAdmissionSpec{
				Load: func() bool { t.Error("loader ran on an early refusal"); return true },
			}); err == nil {
				t.Fatalf("admission accepted; want %s", tc.wantReason)
			}
			receipts := decodeMetalAdmissionReceipts(t, buf)
			if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"refuse"}) {
				t.Fatalf("receipt stages = %v, want [refuse]", stages)
			}
			rc := receipts[0].rc
			if rc.Reason != tc.wantReason {
				t.Errorf("refuse receipt reason = %q, want %q", rc.Reason, tc.wantReason)
			}
			if rc.ReservedBytes != livePeerHeld {
				t.Errorf("refuse receipt reserved_bytes = %d, want the live peer's %d (dead owner excluded)", rc.ReservedBytes, livePeerHeld)
			}
			if rc.AvailableBytes != tc.wantAvailable {
				t.Errorf("refuse receipt available_bytes = %d, want %d", rc.AvailableBytes, tc.wantAvailable)
			}
			if rows := readMetalAdmissionLedger(t, resDir); len(rows) != 1 || rows[0].ID != live.ID {
				t.Errorf("ledger after the early refusal = %+v, want only the live peer row", rows)
			}
		})
	}
}

// TestMetalAdmissionSteadyReceiptReservedBytesDropToSteady pins the steady
// receipt's accounting: once the load downshifts from the startup peak to the
// steady residency, reserved_bytes drops by exactly peak-steady (single owner:
// reserved == steady) and available_bytes grows by the same amount.
func TestMetalAdmissionSteadyReceiptReservedBytesDropToSteady(t *testing.T) {
	const avail = int64(8 << 30)
	for _, tc := range []struct {
		name     string
		peerHeld int64
	}{
		{"single_owner", 0},
		{"beside_a_live_peer", 1 << 30},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resDir, _ := metalLifecycleEnv(t, "default", lifecyclePeak, lifecycleSteady)
			buf := stubMetalAdmissionSeams(t, lifecycleHost(16<<30, avail), measuredLifecycleProbes)
			if tc.peerHeld > 0 {
				_ = seedMetalAdmissionLedger(t, resDir, localadmission.Reservation{
					ID: "live-peer-row", OwnerPID: os.Getpid(),
					StartupPeakBytes: tc.peerHeld, SteadyBytes: tc.peerHeld, HeldBytes: tc.peerHeld, Phase: "steady",
				})
			}
			release, err := admitLocalMetalModel(true, "steady-accounting.gguf", gpulease.Options{}, metalAdmissionSpec{Load: func() bool { return true }})
			if err != nil {
				t.Fatalf("admission refused a fitting plan: %v", err)
			}
			defer release()

			receipts := decodeMetalAdmissionReceipts(t, buf)
			if stages := metalReceiptStages(receipts); !slices.Equal(stages, []string{"admit", "steady"}) {
				t.Fatalf("receipt stages before release = %v, want [admit steady]", stages)
			}
			admit, steady := receipts[0].rc, receipts[1].rc
			if admit.ReservedBytes != tc.peerHeld+lifecyclePeak || admit.AvailableBytes != avail-tc.peerHeld-lifecyclePeak {
				t.Errorf("admit receipt reserved/available = %d/%d, want %d/%d", admit.ReservedBytes, admit.AvailableBytes, tc.peerHeld+lifecyclePeak, avail-tc.peerHeld-lifecyclePeak)
			}
			if steady.ReservedBytes != tc.peerHeld+lifecycleSteady || steady.AvailableBytes != avail-tc.peerHeld-lifecycleSteady {
				t.Errorf("steady receipt reserved/available = %d/%d, want %d/%d", steady.ReservedBytes, steady.AvailableBytes, tc.peerHeld+lifecycleSteady, avail-tc.peerHeld-lifecycleSteady)
			}
			var ledgerHeld int64
			for _, r := range readMetalAdmissionLedger(t, resDir) {
				ledgerHeld += r.HeldBytes
			}
			if steady.ReservedBytes != ledgerHeld {
				t.Errorf("steady receipt reserved_bytes = %d, want the ledger's held total %d", steady.ReservedBytes, ledgerHeld)
			}
		})
	}
}
