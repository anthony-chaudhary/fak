package localadmission

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"testing"
)

func metalReq(pid int, peak, steady, capacity int64, pressure Pressure) UnifiedMemoryReservationRequest {
	return UnifiedMemoryReservationRequest{
		OwnerPID: pid,
		Plan:     MemoryPlan{StartupPeakBytes: peak, SteadyBytes: steady},
		Host: AdmissionSample{
			TotalBytes:       capacity,
			AllocatableBytes: capacity,
			Pressure:         pressure,
		},
		ModelName:  "qwen38:27b",
		DeviceName: "Apple M3 Pro",
	}
}

func TestMetalReservationAtomicAdmissionConcurrent(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	const callers = 16
	const capacity = int64(100)
	const peak = int64(40)
	const steady = int64(20)

	var wg sync.WaitGroup
	results := make(chan UnifiedMemoryResult, callers)
	errs := make(chan error, callers)

	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			mgr := NewMetalReservationManager(dir)
			mgr.SetAlive(func(int) bool { return true })
			req := metalReq(os.Getpid(), peak, steady, capacity, PressureNormal)
			got, err := mgr.Reserve(ctx, req)
			if err != nil {
				errs <- err
				return
			}
			results <- got
		}(i)
	}

	wg.Wait()
	close(results)
	close(errs)

	for err := range errs {
		t.Fatalf("unexpected concurrent reserve error: %v", err)
	}

	admitted := 0
	rejected := 0
	for got := range results {
		if got.Admit {
			admitted++
			if got.Verdict != "ADMIT" {
				t.Fatalf("expected verdict ADMIT, got %q", got.Verdict)
			}
			if got.Reservation == nil || got.Reservation.HeldBytes != peak {
				t.Fatalf("unexpected reservation in admitted decision: %+v", got)
			}
		} else {
			rejected++
			if got.Verdict != "REJECT" {
				t.Fatalf("expected verdict REJECT, got %q", got.Verdict)
			}
			if got.Reason != "aggregate_capacity" {
				t.Fatalf("expected reason aggregate_capacity, got %q", got.Reason)
			}
		}
	}

	// At 40 bytes startup peak per reservation on a 100-byte budget, exactly 2 can fit (40+40 <= 100; 40*3 = 120 > 100).
	if admitted != 2 {
		t.Fatalf("admitted = %d, want 2", admitted)
	}
	if rejected != callers-2 {
		t.Fatalf("rejected = %d, want %d", rejected, callers-2)
	}

	mgr := NewMetalReservationManager(dir)
	mgr.SetAlive(func(int) bool { return true })
	total, err := mgr.TotalReservedBytes(ctx)
	if err != nil {
		t.Fatalf("TotalReservedBytes error: %v", err)
	}
	if total != peak*2 {
		t.Fatalf("total reserved bytes = %d, want %d", total, peak*2)
	}
}

func TestMetalReservationPromoteToSteadyTransition(t *testing.T) {
	ctx := context.Background()
	mgr := NewMetalReservationManager(t.TempDir())
	mgr.SetAlive(func(int) bool { return true })

	const capacity = int64(100)

	// Model 1: peak = 60, steady = 30 on capacity = 100. Should be admitted.
	dec1, err := mgr.Reserve(ctx, metalReq(101, 60, 30, capacity, PressureNormal))
	if err != nil || !dec1.Admit || dec1.Reservation == nil {
		t.Fatalf("model 1 admission failed: %+v, err: %v", dec1, err)
	}
	if dec1.Reservation.HeldBytes != 60 || dec1.Reservation.Phase != "startup" {
		t.Fatalf("model 1 startup state unexpected: %+v", dec1.Reservation)
	}

	// Model 2: peak = 50, steady = 20. Total needed = 60 + 50 = 110 > 100. Must be rejected.
	dec2, err := mgr.Reserve(ctx, metalReq(102, 50, 20, capacity, PressureNormal))
	if err != nil {
		t.Fatalf("model 2 reserve err: %v", err)
	}
	if dec2.Admit {
		t.Fatalf("model 2 should have been rejected due to startup peak overcommit: %+v", dec2)
	}
	if dec2.Reason != "aggregate_capacity" {
		t.Fatalf("model 2 rejection reason = %q, want aggregate_capacity", dec2.Reason)
	}

	// Model 1 loading succeeds: promote to steady residency (drops held from 60 to 30).
	err = dec1.Reservation.PromoteToSteady(ctx)
	if err != nil {
		t.Fatalf("PromoteToSteady failed: %v", err)
	}
	if dec1.Reservation.HeldBytes != 30 || dec1.Reservation.Phase != "steady" {
		t.Fatalf("model 1 steady state unexpected: %+v", dec1.Reservation)
	}

	reserved, err := mgr.TotalReservedBytes(ctx)
	if err != nil || reserved != 30 {
		t.Fatalf("total reserved after promote = %d, want 30 (err: %v)", reserved, err)
	}

	// Model 2 tries again now that Model 1 is steady: 30 + 50 = 80 <= 100. Should be admitted.
	dec2Retry, err := mgr.Reserve(ctx, metalReq(102, 50, 20, capacity, PressureNormal))
	if err != nil || !dec2Retry.Admit || dec2Retry.Reservation == nil {
		t.Fatalf("model 2 retry failed: %+v, err: %v", dec2Retry, err)
	}
	if dec2Retry.Reservation.HeldBytes != 50 || dec2Retry.Reservation.Phase != "startup" {
		t.Fatalf("model 2 startup state unexpected: %+v", dec2Retry.Reservation)
	}

	// Model 3: peak = 30, steady = 10. Total needed = 30 (m1 steady) + 50 (m2 startup) + 30 = 110 > 100. Must be rejected.
	dec3, err := mgr.Reserve(ctx, metalReq(103, 30, 10, capacity, PressureNormal))
	if err != nil {
		t.Fatalf("model 3 reserve err: %v", err)
	}
	if dec3.Admit {
		t.Fatalf("model 3 should have been rejected: %+v", dec3)
	}

	// Model 2 promotes to steady: drops held from 50 to 20. Total held is now 30 + 20 = 50.
	err = dec2Retry.Reservation.PromoteToSteady(ctx)
	if err != nil {
		t.Fatalf("model 2 PromoteToSteady failed: %v", err)
	}

	reserved, err = mgr.TotalReservedBytes(ctx)
	if err != nil || reserved != 50 {
		t.Fatalf("total reserved after second promote = %d, want 50 (err: %v)", reserved, err)
	}

	// Model 3 tries again: 50 + 30 = 80 <= 100. Now admitted!
	dec3Retry, err := mgr.Reserve(ctx, metalReq(103, 30, 10, capacity, PressureNormal))
	if err != nil || !dec3Retry.Admit {
		t.Fatalf("model 3 retry failed: %+v, err: %v", dec3Retry, err)
	}

	reserved, err = mgr.TotalReservedBytes(ctx)
	if err != nil || reserved != 80 {
		t.Fatalf("total reserved with three models = %d, want 80 (err: %v)", reserved, err)
	}
}

func TestMetalReservationFailClosedPressureAndExhaustion(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	mgr := NewMetalReservationManager(dir)

	// Critical pressure fails closed
	critReq := metalReq(os.Getpid(), 10, 5, 100, PressureCritical)
	critReq.Host.CompressedBytes = 25
	critReq.Host.TotalBytes = 100
	critDec, err := mgr.Reserve(ctx, critReq)
	if err != nil {
		t.Fatalf("critical reserve err: %v", err)
	}
	if critDec.Admit || critDec.Verdict != "REJECT" || critDec.Reason != "pressure_critical" {
		t.Fatalf("expected fail-closed on critical pressure, got: %+v", critDec)
	}
	if critDec.RemedyHint == "" {
		t.Fatalf("expected remedy hint on critical pressure refusal: %+v", critDec)
	}
	if critDec.Receipt == nil || critDec.Receipt.Engine != "fak-native" || critDec.Receipt.Verdict != "REJECT" {
		t.Fatalf("expected evidentiary receipt on critical refusal: %+v", critDec.Receipt)
	}

	// Critical pressure dev policy override succeeds
	critDevReq := critReq
	critDevReq.Policy = "dev"
	critDevDec, err := mgr.Reserve(ctx, critDevReq)
	if err != nil || !critDevDec.Admit || critDevDec.Verdict != "ADMIT" {
		t.Fatalf("expected dev override on critical pressure to admit: %+v, err: %v", critDevDec, err)
	}

	// Warning pressure fails closed by default
	warnReq := metalReq(os.Getpid(), 10, 5, 100, PressureWarning)
	warnDec, err := mgr.Reserve(ctx, warnReq)
	if err != nil {
		t.Fatalf("warning reserve err: %v", err)
	}
	if warnDec.Admit || warnDec.Verdict != "REJECT" || warnDec.Reason != "pressure_warning" {
		t.Fatalf("expected fail-closed on warning pressure by default, got: %+v", warnDec)
	}

	// Warning pressure with AllowWarning: true succeeds
	warnAllowedReq := warnReq
	warnAllowedReq.AllowWarning = true
	warnAllowedDec, err := mgr.Reserve(ctx, warnAllowedReq)
	if err != nil || !warnAllowedDec.Admit || warnAllowedDec.Verdict != "ADMIT" {
		t.Fatalf("expected warning pressure with AllowWarning to admit: %+v, err: %v", warnAllowedDec, err)
	}

	// Unknown pressure fails closed
	unkReq := metalReq(os.Getpid(), 10, 5, 100, PressureUnknown)
	unkDec, err := mgr.Reserve(ctx, unkReq)
	if err != nil {
		t.Fatalf("unknown pressure reserve err: %v", err)
	}
	if unkDec.Admit || unkDec.Reason != "pressure_unknown" {
		t.Fatalf("expected fail-closed on unknown pressure, got: %+v", unkDec)
	}

	// Budget exhaustion fails closed
	hugeReq := metalReq(os.Getpid(), 150, 80, 100, PressureNormal)
	hugeDec, err := mgr.Reserve(ctx, hugeReq)
	if err != nil {
		t.Fatalf("huge request reserve err: %v", err)
	}
	if hugeDec.Admit || hugeDec.Reason != "aggregate_capacity" {
		t.Fatalf("expected aggregate_capacity refusal, got: %+v", hugeDec)
	}

	// Invalid requests fail closed
	invalidReqs := []UnifiedMemoryReservationRequest{
		metalReq(0, 10, 5, 100, PressureNormal),   // invalid PID
		metalReq(101, 0, 5, 100, PressureNormal),  // zero peak
		metalReq(101, 10, 0, 100, PressureNormal), // zero steady
		metalReq(101, 5, 10, 100, PressureNormal), // steady > peak
		{
			OwnerPID: 101,
			Plan:     MemoryPlan{StartupPeakBytes: 10, SteadyBytes: 5},
			Host:     AdmissionSample{AllocatableBytes: 0, Pressure: PressureNormal}, // unprobeable capacity
		},
	}
	for i, req := range invalidReqs {
		dec, err := mgr.Reserve(ctx, req)
		if err != nil {
			t.Fatalf("case %d reserve err: %v", i, err)
		}
		if dec.Admit {
			t.Fatalf("case %d should have failed closed: %+v", i, dec)
		}
	}
}

func TestMetalReservationCleanupReleaseAndStaleOwnerRecovery(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	mgr := NewMetalReservationManager(dir)

	aliveMap := map[int]bool{101: true, 102: true, 103: true}
	mgr.SetAlive(func(pid int) bool { return aliveMap[pid] })

	const capacity = int64(100)

	// Part 1: Lease release on teardown
	dec1, err := mgr.Reserve(ctx, metalReq(101, 70, 35, capacity, PressureNormal))
	if err != nil || !dec1.Admit {
		t.Fatalf("initial reserve failed: %+v, err: %v", dec1, err)
	}

	// Attempting another large reservation is blocked
	dec2, err := mgr.Reserve(ctx, metalReq(102, 50, 25, capacity, PressureNormal))
	if err != nil {
		t.Fatalf("second reserve err: %v", err)
	}
	if dec2.Admit {
		t.Fatalf("expected second reserve to be blocked: %+v", dec2)
	}

	// Release reservation 1
	err = dec1.Reservation.Release(ctx)
	if err != nil {
		t.Fatalf("reservation release failed: %v", err)
	}
	if dec1.Reservation.Phase != "released" || dec1.Reservation.HeldBytes != 0 {
		t.Fatalf("reservation state after release unexpected: %+v", dec1.Reservation)
	}

	// Re-releasing returns ErrReservationNotFound
	err = dec1.Reservation.Release(ctx)
	if !errors.Is(err, ErrReservationNotFound) {
		t.Fatalf("expected ErrReservationNotFound on duplicate release, got: %v", err)
	}

	// Generate release receipt
	releaseReceipt := mgr.GenerateReleaseReceipt(*dec1.Reservation, metalReq(101, 70, 35, capacity, PressureNormal).Host)
	if releaseReceipt.Verdict != "RELEASED" || releaseReceipt.Cleanup != "released" {
		t.Fatalf("unexpected release receipt: %+v", releaseReceipt)
	}

	// Second reservation now succeeds
	dec2Retry, err := mgr.Reserve(ctx, metalReq(102, 50, 25, capacity, PressureNormal))
	if err != nil || !dec2Retry.Admit {
		t.Fatalf("second reserve retry failed after release: %+v, err: %v", dec2Retry, err)
	}

	// Part 2: Stale owner detection and reaping
	// Kill PID 102
	aliveMap[102] = false

	// Third reservation arrives with PID 103 requiring 80 bytes.
	// Without reaping, 50 + 80 = 130 > 100 would fail.
	// With stale-owner reaping of PID 102, 80 <= 100 succeeds!
	dec3, err := mgr.Reserve(ctx, metalReq(103, 80, 40, capacity, PressureNormal))
	if err != nil {
		t.Fatalf("third reserve err: %v", err)
	}
	if !dec3.Admit {
		t.Fatalf("expected third reserve to succeed after reaping dead owner 102: %+v", dec3)
	}
	if dec3.Reaped != 1 {
		t.Fatalf("expected Reaped = 1, got %d", dec3.Reaped)
	}

	// Direct ReapStale invocation when everything is clean returns 0
	reaped, err := mgr.ReapStale(ctx)
	if err != nil || reaped != 0 {
		t.Fatalf("expected 0 reaped when all alive, got %d (err: %v)", reaped, err)
	}

	// Kill PID 103 and call ReapStale directly
	aliveMap[103] = false
	reaped, err = mgr.ReapStale(ctx)
	if err != nil || reaped != 1 {
		t.Fatalf("expected 1 reaped after killing 103, got %d (err: %v)", reaped, err)
	}

	reserved, err := mgr.TotalReservedBytes(ctx)
	if err != nil || reserved != 0 {
		t.Fatalf("expected 0 reserved bytes after reaping all, got %d", reserved)
	}
}

func TestMetalReservationReceiptGeneration(t *testing.T) {
	ctx := context.Background()
	mgr := NewMetalReservationManager(t.TempDir())
	mgr.SetAlive(func(int) bool { return true })

	req := metalReq(os.Getpid(), 60*(1<<20), 30*(1<<20), 100*(1<<20), PressureNormal)
	req.ModelName = "qwen38:27b"
	req.DeviceName = "Apple M3 Pro"

	dec, err := mgr.Reserve(ctx, req)
	if err != nil || !dec.Admit || dec.Receipt == nil {
		t.Fatalf("reserve failed: %+v, err: %v", dec, err)
	}

	receipt := dec.Receipt
	if receipt.Engine != "fak-native" {
		t.Fatalf("receipt engine = %q, want fak-native", receipt.Engine)
	}
	if receipt.Verdict != "ADMIT" {
		t.Fatalf("receipt verdict = %q, want ADMIT", receipt.Verdict)
	}
	if receipt.Model != "qwen38:27b" {
		t.Fatalf("receipt model = %q, want qwen38:27b", receipt.Model)
	}
	if receipt.Device != "Apple M3 Pro" {
		t.Fatalf("receipt device = %q, want Apple M3 Pro", receipt.Device)
	}
	if receipt.Topology != "apple-unified-memory" {
		t.Fatalf("receipt topology = %q, want apple-unified-memory", receipt.Topology)
	}
	if !receipt.HostUnified {
		t.Fatalf("receipt HostUnified must be true on Apple Silicon")
	}
	if receipt.HostAddressable {
		t.Fatalf("receipt HostAddressable must be false (device buffer host dereference forbidden)")
	}
	if receipt.PlannedBytes.StartupPeakBytes != 60*(1<<20) || receipt.PlannedBytes.SteadyBytes != 30*(1<<20) {
		t.Fatalf("planned bytes mismatch: %+v", receipt.PlannedBytes)
	}
	if receipt.AllocatableBytes != 100*(1<<20) {
		t.Fatalf("allocatable bytes = %d, want %d", receipt.AllocatableBytes, 100*(1<<20))
	}
	if receipt.ReservedBytes != 60*(1<<20) {
		t.Fatalf("reserved bytes = %d, want %d", receipt.ReservedBytes, 60*(1<<20))
	}
	if receipt.Cleanup != "active" {
		t.Fatalf("cleanup = %q, want active", receipt.Cleanup)
	}

	// JSON serialization round-trip
	b, err := receipt.JSON()
	if err != nil {
		t.Fatalf("receipt JSON marshal error: %v", err)
	}
	var decoded MetalReservationReceipt
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("receipt JSON unmarshal error: %v", err)
	}
	if decoded.Engine != receipt.Engine || decoded.Verdict != receipt.Verdict || decoded.Topology != receipt.Topology {
		t.Fatalf("decoded receipt mismatch: %+v vs %+v", decoded, receipt)
	}

	// SampleM3ProQwen38Receipt validation
	witness := SampleM3ProQwen38Receipt()
	if witness.Engine != "fak-native" {
		t.Fatalf("witness engine = %q, want fak-native", witness.Engine)
	}
	if witness.Model != "qwen38:27b" {
		t.Fatalf("witness model = %q, want qwen38:27b", witness.Model)
	}
	if witness.Device != "Apple M3 Pro" {
		t.Fatalf("witness device = %q, want Apple M3 Pro", witness.Device)
	}
	if !witness.HostUnified || witness.HostAddressable {
		t.Fatalf("witness topology invariants violated: host_unified=%v, host_addressable=%v", witness.HostUnified, witness.HostAddressable)
	}
	if witness.PlannedBytes.StartupPeakBytes != 20*(1<<30) || witness.PlannedBytes.SteadyBytes != 16*(1<<30) {
		t.Fatalf("witness planned bytes unexpected: %+v", witness.PlannedBytes)
	}
}
