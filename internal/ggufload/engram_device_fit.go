package ggufload

import (
	"fmt"
	"math"
)

// engram_device_fit.go — V41C-19 PHASE 1 (modeling pass only): the pure device-admission
// arithmetic for admitting the DeepSeek-V4.1-Flash packed Engram tables to HBM instead of
// host-gathering them row-wise per token.
//
// Today the loader drops the Engram tables BY DESIGN: the materializing Q4_K load refuses to
// eager-dequantize them (computeQ4KTensorWork, quant_q4k_loader.go), byte accounting moves their
// bytes out of the dense core into RoutedExpertActiveSet.HostGatheredResident (estimate.go), and
// the fit-plan producer (ActivatedExpertFit.MemoryPlan) charges them only as a HOST-scoped page
// reserve capped at HostGatheredPageReserveCap (expert_activated_fit.go). No device-scoped plan
// line ever carries Engram bytes, so no admission path can place them on device. Changing that
// admission is a fit-plan change, not a flag flip — and this file is the modeling half of it.
//
// This is PHASE 1 of the hill: MODELING + TESTS ONLY. Nothing here is wired into the loader or
// the fit plan yet. The follow-up gate: the fit-plan producer (ActivatedExpertFit.MemoryPlan,
// which feeds cmd/fak serve_fit_confirm) must CONSUME this model and emit a typed device-scoped
// plan line for the admitted Engram bytes before any loader arm may allocate them.
//
// The modeled envelope is the measured H200 baseline, rtp-test-h200 run 20261106T023306Z
// (vcruz Q2_K c4a0855..., pin e2a6b10642, --native-context-tokens 16384, explicit 32 GiB ring):
// device total 143,771 MiB; non-Engram dense core 3,230,237,376 B device-resident; 60.083 GiB
// packed Engram tables host-gathered; 32 GiB activated-expert ring; KV pool 16384 x 240 KiB;
// ring per-token working set 2.863 GiB.
//
// Like internal/model/expert_spill_fit.go, this is PURE math: no device probe, no allocation,
// deterministic, fail-closed on negative inputs, monotone in the admitted amount — admitting
// less always fits when admitting more fits.

// EngramAdmissionBudget is the pure input to the Engram device-admission model: the measured
// device total and the five byte quantities that share it with the Engram tables. All fields are
// non-negative counts; a negative field is a miscomputed input and is refused (fail-closed)
// rather than sized against.
type EngramAdmissionBudget struct {
	// DeviceTotalBytes is the measured device HBM total (the H200 baseline reads 143,771 MiB).
	DeviceTotalBytes int64
	// DenseCoreBytes is the non-Engram dense core that stays device-resident under every
	// placement being considered (the fit_confirm weights charge, excluding the Engram band).
	DenseCoreBytes int64
	// EngramBytes is the packed Engram table payload the admission decision prices (the
	// host_gathered band). 0 models a checkpoint without Engram tables.
	EngramBytes int64
	// RingBytes is the activated-expert device ring co-resident with the dense core.
	RingBytes int64
	// KVPoolBytes is the KV pool for the target context (native-context-tokens x per-token KV).
	KVPoolBytes int64
	// WorkingSetBytes is the per-token working set beside ring and KV: activations, dequant
	// scratch, and the transient decode buffers the measured baseline accounts for.
	WorkingSetBytes int64
	// RequiredHeadroom is the reserved slack the placement must leave free (driver, CUDA
	// context, growth headroom). 0 prices raw capacity.
	RequiredHeadroom int64
}

// EngramDeviceFit is the modeled verdict for one EngramAdmissionBudget.
//
// The invariant the whole model exists to keep: whenever Fits=true, the full-placement demand
// (DenseCore + Engram + Ring + KV + WorkingSet + RequiredHeadroom) is within DeviceTotalBytes —
// the model never reports a fit it would over-commit. When even the zero-Engram placement does
// not fit, AdmissibleBytes is 0 and HeadroomBytes goes NEGATIVE to expose the shortfall rather
// than hide it; no admissible placement then exists.
type EngramDeviceFit struct {
	// AdmissibleBytes is the MAXIMAL Engram byte count whose placement fits the budget: the
	// largest e in [0, EngramBytes] with DenseCore + e + Ring + KV + WorkingSet +
	// RequiredHeadroom <= DeviceTotal. Always in [0, EngramBytes]; 0 when nothing fits.
	AdmissibleBytes int64
	// Fits is true when the FULL requested EngramBytes admission fits the budget.
	Fits bool
	// DeviceResidentBytes is the device-resident weight/pool bytes of the placement at
	// AdmissibleBytes: DenseCore + AdmissibleBytes + Ring + KV + WorkingSet (the required
	// headroom is reserved, not resident, so it is excluded here).
	DeviceResidentBytes int64
	// HeadroomBytes is the slack left after committing DeviceResidentBytes AND reserving
	// RequiredHeadroom: DeviceTotal - DeviceResidentBytes - RequiredHeadroom. Non-negative
	// whenever an admissible placement respects the headroom floor; negative only when even
	// the zero-Engram placement over-commits, in which case it names the exact shortfall.
	HeadroomBytes int64
}

// validate refuses a budget with any negative field: these are byte counts, so a negative is a
// miscomputed input, not a configuration to size against (mirrors ExpertSpillBudget.validate).
func (b EngramAdmissionBudget) validate() error {
	if b.DeviceTotalBytes < 0 {
		return fmt.Errorf("gguf: EngramAdmissionBudget DeviceTotalBytes = %d, want >= 0", b.DeviceTotalBytes)
	}
	if b.DenseCoreBytes < 0 {
		return fmt.Errorf("gguf: EngramAdmissionBudget DenseCoreBytes = %d, want >= 0", b.DenseCoreBytes)
	}
	if b.EngramBytes < 0 {
		return fmt.Errorf("gguf: EngramAdmissionBudget EngramBytes = %d, want >= 0", b.EngramBytes)
	}
	if b.RingBytes < 0 {
		return fmt.Errorf("gguf: EngramAdmissionBudget RingBytes = %d, want >= 0", b.RingBytes)
	}
	if b.KVPoolBytes < 0 {
		return fmt.Errorf("gguf: EngramAdmissionBudget KVPoolBytes = %d, want >= 0", b.KVPoolBytes)
	}
	if b.WorkingSetBytes < 0 {
		return fmt.Errorf("gguf: EngramAdmissionBudget WorkingSetBytes = %d, want >= 0", b.WorkingSetBytes)
	}
	if b.RequiredHeadroom < 0 {
		return fmt.Errorf("gguf: EngramAdmissionBudget RequiredHeadroom = %d, want >= 0", b.RequiredHeadroom)
	}
	return nil
}

// ModelEngramDeviceAdmission computes, from the stated budget alone, whether the full Engram
// admission fits and the maximal admissible Engram byte count. It is the V41C-19 phase-1
// deliverable: pure, typed, unit-testable arithmetic with no device or probe call.
//
// The fit condition is linear and monotone in the admitted amount e:
//
//	DenseCore + e + Ring + KV + WorkingSet + RequiredHeadroom <= DeviceTotal
//
// so the admissible set is the prefix [0, AdmissibleBytes] — admitting less always fits when
// admitting more fits — and AdmissibleBytes is the ceiling division of the leftover room,
// clamped to [0, EngramBytes]. Fail-closed: a negative input is refused with an error; a budget
// too small for even the zero-Engram placement returns Fits=false with AdmissibleBytes=0 and a
// negative HeadroomBytes naming the shortfall — never a negative admissible count, never an
// over-committing verdict.
func ModelEngramDeviceAdmission(b EngramAdmissionBudget) (EngramDeviceFit, error) {
	if err := b.validate(); err != nil {
		return EngramDeviceFit{}, err
	}
	// fixed is everything the placement commits besides the admitted Engram slice, INCLUDING
	// the required headroom reservation.
	fixed := b.DenseCoreBytes
	for _, addend := range []int64{b.RingBytes, b.KVPoolBytes, b.WorkingSetBytes, b.RequiredHeadroom} {
		if fixed > math.MaxInt64-addend {
			return EngramDeviceFit{}, fmt.Errorf("gguf: engram admission fixed demand overflows int64")
		}
		fixed += addend
	}
	full := fixed
	if full > math.MaxInt64-b.EngramBytes {
		return EngramDeviceFit{}, fmt.Errorf("gguf: engram admission full demand overflows int64")
	}
	full += b.EngramBytes
	fits := full <= b.DeviceTotalBytes

	// room is the Engram slice the budget leaves after fixed; safe to go negative because both
	// operands are non-negative int64. Clamp to [0, EngramBytes]: the maximal feasible fraction.
	room := b.DeviceTotalBytes - fixed
	var admissible int64
	switch {
	case room < 0:
		admissible = 0
	case room >= b.EngramBytes:
		admissible = b.EngramBytes
	default:
		admissible = room
	}
	// admissible <= max(room, 0) keeps DenseCore + admissible + Ring + KV + WorkingSet +
	// RequiredHeadroom <= DeviceTotal <= MaxInt64, so the sums below cannot overflow.
	resident := b.DenseCoreBytes + admissible + b.RingBytes + b.KVPoolBytes + b.WorkingSetBytes
	return EngramDeviceFit{
		AdmissibleBytes:     admissible,
		Fits:                fits,
		DeviceResidentBytes: resident,
		HeadroomBytes:       b.DeviceTotalBytes - resident - b.RequiredHeadroom,
	}, nil
}
