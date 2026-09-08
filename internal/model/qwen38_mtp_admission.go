package model

import (
	"fmt"
	"math/bits"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// Apple Silicon unified memory capacity thresholds for MTP speculative decoding.
const (
	AppleSiliconUnifiedMemoryFloorBytes uint64 = 16 * 1024 * 1024 * 1024 // 16 GiB
	AppleSilicon36GBUnifiedMemoryBytes  uint64 = 36 * 1024 * 1024 * 1024 // 36 GiB
)

// qwen38MTPCapacityResult reports the outcome of capacity admission on Apple Silicon
// unified memory before loading or allocating MTP draft weights.
type qwen38MTPCapacityResult struct {
	Approved    bool                      `json:"approved"`
	Outcome     Qwen38MTPAdmissionOutcome `json:"outcome"`
	TotalBytes  uint64                    `json:"total_bytes"`
	UsableBytes uint64                    `json:"usable_bytes"`
	FloorBytes  uint64                    `json:"floor_bytes"`
	Reason      string                    `json:"reason"`
	Admission   Qwen38MTPAdmission        `json:"admission"`
}

// Qwen38MTPMemoryPressure is the bounded pressure signal used before any MTP
// allocation. Unknown and critical pressure fail closed to target-only decode.
type Qwen38MTPMemoryPressure string

const (
	Qwen38MTPPressureNominal  Qwen38MTPMemoryPressure = "nominal"
	Qwen38MTPPressureElevated Qwen38MTPMemoryPressure = "elevated"
	Qwen38MTPPressureCritical Qwen38MTPMemoryPressure = "critical"
)

// Qwen38MTPAdmissionOutcome is the selected fak-native memory placement.
type Qwen38MTPAdmissionOutcome string

const (
	Qwen38MTPAdmissionResident     Qwen38MTPAdmissionOutcome = "resident"
	Qwen38MTPAdmissionHostAssisted Qwen38MTPAdmissionOutcome = "host_assisted"
	Qwen38MTPAdmissionTargetOnly   Qwen38MTPAdmissionOutcome = "native_target_only"
)

// Qwen38MTPAdmissionInput contains every byte component used by admission.
// AvailableBytes is the selected device's current allocatable capacity before
// subtracting operator/system headroom and existing reservations.
type Qwen38MTPAdmissionInput struct {
	TargetTensorBytes          uint64                  `json:"target_tensor_bytes"`
	RetainedMTPTensorBytes     uint64                  `json:"retained_mtp_tensor_bytes"`
	TransactionalStateBytes    uint64                  `json:"transactional_state_bytes"`
	VerificationWorkspaceBytes uint64                  `json:"verification_workspace_bytes"`
	AvailableBytes             uint64                  `json:"available_bytes"`
	HeadroomBytes              uint64                  `json:"headroom_bytes"`
	ReservedBytes              uint64                  `json:"reserved_bytes"`
	Pressure                   Qwen38MTPMemoryPressure `json:"pressure"`
	HostAssistedSupported      bool                    `json:"host_assisted_supported"`
}

// Qwen38MTPAdmission is a receipt-ready, allocation-free decision. PlannedBytes
// is the fully resident requirement; DeviceBytes is the selected outcome's
// device requirement. UsableBytes is available minus headroom and reservations.
type Qwen38MTPAdmission struct {
	Outcome                    Qwen38MTPAdmissionOutcome `json:"outcome"`
	TargetTensorBytes          uint64                    `json:"target_tensor_bytes"`
	RetainedMTPTensorBytes     uint64                    `json:"retained_mtp_tensor_bytes"`
	TransactionalStateBytes    uint64                    `json:"transactional_state_bytes"`
	VerificationWorkspaceBytes uint64                    `json:"verification_workspace_bytes"`
	PlannedBytes               uint64                    `json:"planned_bytes"`
	DeviceBytes                uint64                    `json:"device_bytes"`
	AvailableBytes             uint64                    `json:"available_bytes"`
	HeadroomBytes              uint64                    `json:"headroom_bytes"`
	ReservedBytes              uint64                    `json:"reserved_bytes"`
	UsableBytes                uint64                    `json:"usable_bytes"`
	Pressure                   Qwen38MTPMemoryPressure   `json:"pressure"`
}

// AdmitQwen38MTP chooses a placement without probing allocation failure. Fully
// resident MTP is preferred. Nominal pressure may use explicit host assistance
// for retained MTP tensors when the target and transactional verifier working
// set still fit. Every unsafe or unrecognized case stays on fak-native target decode.
func AdmitQwen38MTP(in Qwen38MTPAdmissionInput) Qwen38MTPAdmission {
	out := Qwen38MTPAdmission{
		Outcome:                    Qwen38MTPAdmissionTargetOnly,
		TargetTensorBytes:          in.TargetTensorBytes,
		RetainedMTPTensorBytes:     in.RetainedMTPTensorBytes,
		TransactionalStateBytes:    in.TransactionalStateBytes,
		VerificationWorkspaceBytes: in.VerificationWorkspaceBytes,
		AvailableBytes:             in.AvailableBytes,
		HeadroomBytes:              in.HeadroomBytes,
		ReservedBytes:              in.ReservedBytes,
		Pressure:                   in.Pressure,
	}
	reserved, carry := bits.Add64(in.HeadroomBytes, in.ReservedBytes, 0)
	if carry != 0 || reserved > in.AvailableBytes {
		return out
	}
	out.UsableBytes = in.AvailableBytes - reserved

	working, ok := addQwen38MTPBytes(in.TargetTensorBytes, in.TransactionalStateBytes, in.VerificationWorkspaceBytes)
	if !ok {
		return out
	}
	planned, carry := bits.Add64(working, in.RetainedMTPTensorBytes, 0)
	if carry != 0 {
		return out
	}
	out.PlannedBytes = planned

	switch in.Pressure {
	case Qwen38MTPPressureNominal, Qwen38MTPPressureElevated:
		if planned <= out.UsableBytes {
			out.Outcome = Qwen38MTPAdmissionResident
			out.DeviceBytes = planned
			return out
		}
	case Qwen38MTPPressureCritical:
		return out
	default:
		return out
	}
	if in.Pressure == Qwen38MTPPressureNominal && in.HostAssistedSupported && working <= out.UsableBytes {
		out.Outcome = Qwen38MTPAdmissionHostAssisted
		out.DeviceBytes = working
	}
	return out
}

func addQwen38MTPBytes(values ...uint64) (uint64, bool) {
	var sum uint64
	for _, value := range values {
		var carry uint64
		sum, carry = bits.Add64(sum, value, 0)
		if carry != 0 {
			return 0, false
		}
	}
	return sum, true
}

func validQwen38MTPAdmissionOutcome(outcome Qwen38MTPAdmissionOutcome) bool {
	switch outcome {
	case Qwen38MTPAdmissionResident, Qwen38MTPAdmissionHostAssisted, Qwen38MTPAdmissionTargetOnly:
		return true
	default:
		return false
	}
}

func (a Qwen38MTPAdmission) validate() error {
	if !validQwen38MTPAdmissionOutcome(a.Outcome) {
		return fmt.Errorf("model: qwen3.8 MTP admission unknown outcome %q", a.Outcome)
	}
	if a.Pressure != Qwen38MTPPressureNominal && a.Pressure != Qwen38MTPPressureElevated && a.Pressure != Qwen38MTPPressureCritical {
		return fmt.Errorf("model: qwen3.8 MTP admission unknown pressure %q", a.Pressure)
	}
	reserved, carry := bits.Add64(a.HeadroomBytes, a.ReservedBytes, 0)
	if carry != 0 || reserved > a.AvailableBytes || a.UsableBytes != a.AvailableBytes-reserved {
		return fmt.Errorf("model: qwen3.8 MTP admission invalid usable byte accounting")
	}
	working, ok := addQwen38MTPBytes(a.TargetTensorBytes, a.TransactionalStateBytes, a.VerificationWorkspaceBytes)
	if !ok {
		return fmt.Errorf("model: qwen3.8 MTP admission component byte accounting overflow")
	}
	planned, carry := bits.Add64(working, a.RetainedMTPTensorBytes, 0)
	if carry != 0 || a.PlannedBytes != planned {
		return fmt.Errorf("model: qwen3.8 MTP admission invalid planned byte accounting")
	}
	switch a.Outcome {
	case Qwen38MTPAdmissionResident:
		if a.Pressure == Qwen38MTPPressureCritical || a.DeviceBytes != planned || a.DeviceBytes > a.UsableBytes {
			return fmt.Errorf("model: qwen3.8 MTP resident admission exceeds safe device envelope")
		}
	case Qwen38MTPAdmissionHostAssisted:
		if a.Pressure != Qwen38MTPPressureNominal || a.DeviceBytes != working || a.DeviceBytes > a.UsableBytes || planned <= a.UsableBytes {
			return fmt.Errorf("model: qwen3.8 MTP host-assisted admission is not required or safe")
		}
	case Qwen38MTPAdmissionTargetOnly:
		if a.DeviceBytes != 0 {
			return fmt.Errorf("model: qwen3.8 MTP target-only admission reserves device bytes")
		}
	}
	return nil
}

// admitQwen38MTPAppleSilicon evaluates capacity admission for Apple Silicon unified memory.
// It refuses cleanly when available unified memory is at or below the 16 GiB floor to prevent
// swap thrashing, or when memory pressure is critical.
func admitQwen38MTPAppleSilicon(in Qwen38MTPAdmissionInput) qwen38MTPCapacityResult {
	if in.AvailableBytes <= AppleSiliconUnifiedMemoryFloorBytes {
		admiss := AdmitQwen38MTP(in)
		admiss.Outcome = Qwen38MTPAdmissionTargetOnly
		admiss.DeviceBytes = 0
		return qwen38MTPCapacityResult{
			Approved:    false,
			Outcome:     Qwen38MTPAdmissionTargetOnly,
			TotalBytes:  in.AvailableBytes,
			UsableBytes: admiss.UsableBytes,
			FloorBytes:  AppleSiliconUnifiedMemoryFloorBytes,
			Reason:      "insufficient unified memory: Apple Silicon <=16GB configuration refuses MTP draft residency to prevent swap thrashing",
			Admission:   admiss,
		}
	}

	if in.Pressure == Qwen38MTPPressureCritical {
		admiss := AdmitQwen38MTP(in)
		return qwen38MTPCapacityResult{
			Approved:    false,
			Outcome:     Qwen38MTPAdmissionTargetOnly,
			TotalBytes:  in.AvailableBytes,
			UsableBytes: admiss.UsableBytes,
			FloorBytes:  AppleSiliconUnifiedMemoryFloorBytes,
			Reason:      "memory pressure critical: saturated configuration refuses MTP draft residency",
			Admission:   admiss,
		}
	}

	admiss := AdmitQwen38MTP(in)
	switch admiss.Outcome {
	case Qwen38MTPAdmissionResident:
		return qwen38MTPCapacityResult{
			Approved:    true,
			Outcome:     Qwen38MTPAdmissionResident,
			TotalBytes:  in.AvailableBytes,
			UsableBytes: admiss.UsableBytes,
			FloorBytes:  AppleSiliconUnifiedMemoryFloorBytes,
			Reason:      "approved: resident MTP draft weights fit within Apple Silicon unified memory headroom",
			Admission:   admiss,
		}
	case Qwen38MTPAdmissionHostAssisted:
		return qwen38MTPCapacityResult{
			Approved:    false,
			Outcome:     admiss.Outcome,
			TotalBytes:  in.AvailableBytes,
			UsableBytes: admiss.UsableBytes,
			FloorBytes:  AppleSiliconUnifiedMemoryFloorBytes,
			Reason:      "headroom constrained: host-assisted fallback selected",
			Admission:   admiss,
		}
	case Qwen38MTPAdmissionTargetOnly:
		return qwen38MTPCapacityResult{
			Approved:    false,
			Outcome:     Qwen38MTPAdmissionTargetOnly,
			TotalBytes:  in.AvailableBytes,
			UsableBytes: admiss.UsableBytes,
			FloorBytes:  AppleSiliconUnifiedMemoryFloorBytes,
			Reason:      "memory headroom exceeded: target-only decode fallback",
			Admission:   admiss,
		}
	default:
		return qwen38MTPCapacityResult{
			Approved:    false,
			Outcome:     admiss.Outcome,
			TotalBytes:  in.AvailableBytes,
			UsableBytes: admiss.UsableBytes,
			FloorBytes:  AppleSiliconUnifiedMemoryFloorBytes,
			Reason:      "memory headroom exceeded: target-only decode fallback",
			Admission:   admiss,
		}
	}
}

// admitQwen38MTPDevice evaluates MTP admission on an Apple Silicon device.
// Takes optional physicalMemoryBytes (defaulting to 36GB if not on Darwin or Darwin query).
// Slices headroom (e.g. 4GB) and reservations (e.g. 2GB) and checks admission.
func admitQwen38MTPDevice(targetTensorBytes, mtpTensorBytes uint64, physicalMemoryBytes ...uint64) qwen38MTPCapacityResult {
	var totalMem uint64
	if len(physicalMemoryBytes) > 0 && physicalMemoryBytes[0] > 0 {
		totalMem = physicalMemoryBytes[0]
	} else if runtime.GOOS == "darwin" {
		if mem, err := queryDarwinPhysicalMemory(); err == nil && mem > 0 {
			totalMem = mem
		} else {
			totalMem = AppleSilicon36GBUnifiedMemoryBytes
		}
	} else {
		totalMem = AppleSilicon36GBUnifiedMemoryBytes
	}

	headroom := uint64(4 * 1024 * 1024 * 1024) // 4 GiB
	reserved := uint64(2 * 1024 * 1024 * 1024) // 2 GiB

	in := Qwen38MTPAdmissionInput{
		TargetTensorBytes:      targetTensorBytes,
		RetainedMTPTensorBytes: mtpTensorBytes,
		AvailableBytes:         totalMem,
		HeadroomBytes:          headroom,
		ReservedBytes:          reserved,
		Pressure:               Qwen38MTPPressureNominal,
		HostAssistedSupported:  false,
	}
	return admitQwen38MTPAppleSilicon(in)
}

func queryDarwinPhysicalMemory() (uint64, error) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		out, err = exec.Command("/usr/sbin/sysctl", "-n", "hw.memsize").Output()
		if err != nil {
			return 0, err
		}
	}
	val, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0, err
	}
	return val, nil
}
