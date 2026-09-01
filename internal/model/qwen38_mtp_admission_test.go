package model

import (
	"math"
	"testing"
)

func TestAdmitQwen38MTPBoundaryOutcomes(t *testing.T) {
	base := Qwen38MTPAdmissionInput{
		TargetTensorBytes:          100,
		RetainedMTPTensorBytes:     40,
		TransactionalStateBytes:    20,
		VerificationWorkspaceBytes: 10,
		AvailableBytes:             200,
		HeadroomBytes:              20,
		ReservedBytes:              10,
		Pressure:                   Qwen38MTPPressureNominal,
		HostAssistedSupported:      true,
	}
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionResident || got.PlannedBytes != 170 || got.DeviceBytes != 170 || got.UsableBytes != 170 {
		t.Fatalf("exact resident boundary=%+v", got)
	}

	oneByteShort := base
	oneByteShort.AvailableBytes--
	if got := AdmitQwen38MTP(oneByteShort); got.Outcome != Qwen38MTPAdmissionHostAssisted || got.DeviceBytes != 130 || got.PlannedBytes != 170 || got.UsableBytes != 169 {
		t.Fatalf("one-byte-short host assistance=%+v", got)
	}

	targetOnly := oneByteShort
	targetOnly.AvailableBytes = 159 // usable 129, one byte below target + transaction + verifier.
	if got := AdmitQwen38MTP(targetOnly); got.Outcome != Qwen38MTPAdmissionTargetOnly || got.DeviceBytes != 0 || got.UsableBytes != 129 {
		t.Fatalf("one-byte-short target-only downgrade=%+v", got)
	}
}

func TestAdmitQwen38MTPAccountsForEveryComponentAndReservation(t *testing.T) {
	base := Qwen38MTPAdmissionInput{
		TargetTensorBytes:          10,
		RetainedMTPTensorBytes:     10,
		TransactionalStateBytes:    10,
		VerificationWorkspaceBytes: 10,
		AvailableBytes:             40,
		Pressure:                   Qwen38MTPPressureNominal,
	}
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionResident {
		t.Fatalf("baseline=%+v", got)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Qwen38MTPAdmissionInput)
	}{
		{"target", func(in *Qwen38MTPAdmissionInput) { in.TargetTensorBytes++ }},
		{"retained_mtp", func(in *Qwen38MTPAdmissionInput) { in.RetainedMTPTensorBytes++ }},
		{"transaction", func(in *Qwen38MTPAdmissionInput) { in.TransactionalStateBytes++ }},
		{"verification", func(in *Qwen38MTPAdmissionInput) { in.VerificationWorkspaceBytes++ }},
		{"headroom", func(in *Qwen38MTPAdmissionInput) { in.HeadroomBytes++ }},
		{"reserved", func(in *Qwen38MTPAdmissionInput) { in.ReservedBytes++ }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := base
			tc.mutate(&in)
			if got := AdmitQwen38MTP(in); got.Outcome == Qwen38MTPAdmissionResident {
				t.Fatalf("component did not alter boundary decision: %+v", got)
			}
		})
	}
}

func TestAdmitQwen38MTPPressureAndOverflowFailClosed(t *testing.T) {
	base := Qwen38MTPAdmissionInput{
		TargetTensorBytes:          10,
		RetainedMTPTensorBytes:     10,
		TransactionalStateBytes:    10,
		VerificationWorkspaceBytes: 10,
		AvailableBytes:             100,
		Pressure:                   Qwen38MTPPressureElevated,
		HostAssistedSupported:      true,
	}
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionResident {
		t.Fatalf("elevated pressure with resident fit=%+v", got)
	}
	base.AvailableBytes = 35
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionTargetOnly {
		t.Fatalf("elevated pressure must not host-assist=%+v", got)
	}
	base.AvailableBytes = 100
	base.Pressure = Qwen38MTPPressureCritical
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionTargetOnly {
		t.Fatalf("critical pressure=%+v", got)
	}

	overflow := base
	overflow.Pressure = Qwen38MTPPressureNominal
	overflow.TargetTensorBytes = math.MaxUint64
	if got := AdmitQwen38MTP(overflow); got.Outcome != Qwen38MTPAdmissionTargetOnly || got.PlannedBytes != 0 {
		t.Fatalf("component overflow=%+v", got)
	}
	overflow = base
	overflow.Pressure = Qwen38MTPPressureNominal
	overflow.HeadroomBytes = math.MaxUint64
	overflow.ReservedBytes = 1
	if got := AdmitQwen38MTP(overflow); got.Outcome != Qwen38MTPAdmissionTargetOnly || got.UsableBytes != 0 {
		t.Fatalf("reservation overflow=%+v", got)
	}
}

func TestQwen38MTPAdmissionValidatesReceiptAccounting(t *testing.T) {
	admission := AdmitQwen38MTP(Qwen38MTPAdmissionInput{
		TargetTensorBytes:          100,
		RetainedMTPTensorBytes:     40,
		TransactionalStateBytes:    20,
		VerificationWorkspaceBytes: 10,
		AvailableBytes:             169,
		Pressure:                   Qwen38MTPPressureNominal,
		HostAssistedSupported:      true,
	})
	if err := admission.validate(); err != nil {
		t.Fatalf("valid admission: %v", err)
	}
	admission.PlannedBytes++
	if err := admission.validate(); err == nil {
		t.Fatal("tampered component accounting validated")
	}
}
