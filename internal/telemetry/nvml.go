// pkg/nvml/types_gen.go and pkg/nvml/zz_generated.api.go at
// Copyright (c) 2020, 2024, NVIDIA CORPORATION. All rights reserved.
// Ported: the Return code vocabulary with its error strings, the Memory_v2 and
// Utilization record shapes, and the Init/Shutdown/DeviceGetCount/
// DeviceGetHandleByIndex and device getter method names. Changes were made.
//
// What fak adds: no cgo and no dlopen. Upstream loads libnvidia-ml at runtime;
// here the library is an interface seam, and the default implementation is
// absent, so a host with no driver gets a typed ErrNVMLUnavailable refusal
// instead of an empty device list that reads as "no GPUs, all fine".

package telemetry

import (
	"errors"
	"fmt"
)

// ErrNVMLUnavailable is returned when no NVML library or driver can be reached,
// or the library reports zero devices.
var ErrNVMLUnavailable = errors.New("telemetry: nvml unavailable")

// ErrNVMLReturnUnknown is returned by NVMLReturnByCode for a code outside the
// pinned vocabulary.
var ErrNVMLReturnUnknown = errors.New("telemetry: nvml return code unknown")

// NVMLReturn is an NVML return code (upstream nvml.Return).
type NVMLReturn int32

// The pinned return codes, verbatim from upstream.
const (
	NVMLSuccess                      NVMLReturn = 0
	NVMLErrorUninitialized           NVMLReturn = 1
	NVMLErrorInvalidArgument         NVMLReturn = 2
	NVMLErrorNotSupported            NVMLReturn = 3
	NVMLErrorNoPermission            NVMLReturn = 4
	NVMLErrorAlreadyInitialized      NVMLReturn = 5
	NVMLErrorNotFound                NVMLReturn = 6
	NVMLErrorInsufficientSize        NVMLReturn = 7
	NVMLErrorInsufficientPower       NVMLReturn = 8
	NVMLErrorDriverNotLoaded         NVMLReturn = 9
	NVMLErrorTimeout                 NVMLReturn = 10
	NVMLErrorIRQIssue                NVMLReturn = 11
	NVMLErrorLibraryNotFound         NVMLReturn = 12
	NVMLErrorFunctionNotFound        NVMLReturn = 13
	NVMLErrorCorruptedInforom        NVMLReturn = 14
	NVMLErrorGPUIsLost               NVMLReturn = 15
	NVMLErrorResetRequired           NVMLReturn = 16
	NVMLErrorOperatingSystem         NVMLReturn = 17
	NVMLErrorLibRMVersionMismatch    NVMLReturn = 18
	NVMLErrorInUse                   NVMLReturn = 19
	NVMLErrorMemory                  NVMLReturn = 20
	NVMLErrorNoData                  NVMLReturn = 21
	NVMLErrorVGPUECCNotSupported     NVMLReturn = 22
	NVMLErrorInsufficientResources   NVMLReturn = 23
	NVMLErrorFreqNotSupported        NVMLReturn = 24
	NVMLErrorArgumentVersionMismatch NVMLReturn = 25
	NVMLErrorDeprecated              NVMLReturn = 26
	NVMLErrorNotReady                NVMLReturn = 27
	NVMLErrorGPUNotFound             NVMLReturn = 28
	NVMLErrorInvalidState            NVMLReturn = 29
	NVMLErrorResetTypeNotSupported   NVMLReturn = 30
	NVMLErrorUnknown                 NVMLReturn = 999
)

var nvmlReturnNames = map[NVMLReturn]string{
	NVMLSuccess:                      "SUCCESS",
	NVMLErrorUninitialized:           "ERROR_UNINITIALIZED",
	NVMLErrorInvalidArgument:         "ERROR_INVALID_ARGUMENT",
	NVMLErrorNotSupported:            "ERROR_NOT_SUPPORTED",
	NVMLErrorNoPermission:            "ERROR_NO_PERMISSION",
	NVMLErrorAlreadyInitialized:      "ERROR_ALREADY_INITIALIZED",
	NVMLErrorNotFound:                "ERROR_NOT_FOUND",
	NVMLErrorInsufficientSize:        "ERROR_INSUFFICIENT_SIZE",
	NVMLErrorInsufficientPower:       "ERROR_INSUFFICIENT_POWER",
	NVMLErrorDriverNotLoaded:         "ERROR_DRIVER_NOT_LOADED",
	NVMLErrorTimeout:                 "ERROR_TIMEOUT",
	NVMLErrorIRQIssue:                "ERROR_IRQ_ISSUE",
	NVMLErrorLibraryNotFound:         "ERROR_LIBRARY_NOT_FOUND",
	NVMLErrorFunctionNotFound:        "ERROR_FUNCTION_NOT_FOUND",
	NVMLErrorCorruptedInforom:        "ERROR_CORRUPTED_INFOROM",
	NVMLErrorGPUIsLost:               "ERROR_GPU_IS_LOST",
	NVMLErrorResetRequired:           "ERROR_RESET_REQUIRED",
	NVMLErrorOperatingSystem:         "ERROR_OPERATING_SYSTEM",
	NVMLErrorLibRMVersionMismatch:    "ERROR_LIB_RM_VERSION_MISMATCH",
	NVMLErrorInUse:                   "ERROR_IN_USE",
	NVMLErrorMemory:                  "ERROR_MEMORY",
	NVMLErrorNoData:                  "ERROR_NO_DATA",
	NVMLErrorVGPUECCNotSupported:     "ERROR_VGPU_ECC_NOT_SUPPORTED",
	NVMLErrorInsufficientResources:   "ERROR_INSUFFICIENT_RESOURCES",
	NVMLErrorFreqNotSupported:        "ERROR_FREQ_NOT_SUPPORTED",
	NVMLErrorArgumentVersionMismatch: "ERROR_ARGUMENT_VERSION_MISMATCH",
	NVMLErrorDeprecated:              "ERROR_DEPRECATED",
	NVMLErrorNotReady:                "ERROR_NOT_READY",
	NVMLErrorGPUNotFound:             "ERROR_GPU_NOT_FOUND",
	NVMLErrorInvalidState:            "ERROR_INVALID_STATE",
	NVMLErrorResetTypeNotSupported:   "ERROR_RESET_TYPE_NOT_SUPPORTED",
	NVMLErrorUnknown:                 "ERROR_UNKNOWN",
}

// Error returns the upstream name of the return code, so an NVMLReturn is
// itself an error the way upstream nvml.Return is.
func (r NVMLReturn) Error() string {
	if name, ok := nvmlReturnNames[r]; ok {
		return name
	}
	return fmt.Sprintf("unknown return value: %d", int32(r))
}

// String returns the same text as Error.
func (r NVMLReturn) String() string { return r.Error() }

// Unavailable reports whether the code means no library or driver could be
// reached, which callers must treat as a refusal rather than a device fault.
func (r NVMLReturn) Unavailable() bool {
	return r == NVMLErrorLibraryNotFound || r == NVMLErrorDriverNotLoaded
}

// Err returns nil for success. Codes for which Unavailable is true wrap
// ErrNVMLUnavailable; every other code is returned as itself, so errors.Is
// matches both the sentinel and the specific code.
func (r NVMLReturn) Err() error {
	switch {
	case r == NVMLSuccess:
		return nil
	case r.Unavailable():
		return fmt.Errorf("%w: %w", ErrNVMLUnavailable, r)
	default:
		return r
	}
}

// NVMLReturnByCode maps a raw code to the pinned vocabulary, refusing a code
// upstream does not define.
func NVMLReturnByCode(code int) (NVMLReturn, error) {
	r := NVMLReturn(code)
	if int(r) != code {
		return 0, fmt.Errorf("%w: %d", ErrNVMLReturnUnknown, code)
	}
	if _, ok := nvmlReturnNames[r]; !ok {
		return 0, fmt.Errorf("%w: %d", ErrNVMLReturnUnknown, code)
	}
	return r, nil
}

// NVMLMemory is upstream Memory_v2 without the struct version, in bytes.
type NVMLMemory struct {
	Total    uint64 `json:"total"`
	Reserved uint64 `json:"reserved"`
	Free     uint64 `json:"free"`
	Used     uint64 `json:"used"`
}

// NVMLUtilization is upstream Utilization, in percent over the last sample
// period.
type NVMLUtilization struct {
	GPU    uint32 `json:"gpu"`
	Memory uint32 `json:"memory"`
}

// NVMLDevice is the subset of the upstream nvml.Device interface fak reads.
type NVMLDevice interface {
	GetUUID() (string, NVMLReturn)
	GetName() (string, NVMLReturn)
	GetMemoryInfo_v2() (NVMLMemory, NVMLReturn)
	GetUtilizationRates() (NVMLUtilization, NVMLReturn)
}

// NVMLLibrary is the subset of the upstream nvml.Interface fak reads.
type NVMLLibrary interface {
	Init() NVMLReturn
	Shutdown() NVMLReturn
	DeviceGetCount() (int, NVMLReturn)
	DeviceGetHandleByIndex(index int) (NVMLDevice, NVMLReturn)
}

type absentNVML struct{}

func (absentNVML) Init() NVMLReturn                  { return NVMLErrorLibraryNotFound }
func (absentNVML) Shutdown() NVMLReturn              { return NVMLErrorUninitialized }
func (absentNVML) DeviceGetCount() (int, NVMLReturn) { return 0, NVMLErrorUninitialized }
func (absentNVML) DeviceGetHandleByIndex(int) (NVMLDevice, NVMLReturn) {
	return nil, NVMLErrorUninitialized
}

// DefaultNVMLLibrary returns the library fak uses when no real binding is
// injected. It never loads anything and always refuses with
// ERROR_LIBRARY_NOT_FOUND.
func DefaultNVMLLibrary() NVMLLibrary { return absentNVML{} }

// NVMLDeviceReading is one device's identity, memory and utilization.
type NVMLDeviceReading struct {
	Index       int             `json:"index"`
	UUID        string          `json:"uuid"`
	Name        string          `json:"name"`
	Memory      NVMLMemory      `json:"memory"`
	Utilization NVMLUtilization `json:"utilization"`
}

// DeviceRecord returns the identity half of a DCGM DeviceRecord, so DCGM field
// values join onto the same device without a second device model.
func (d NVMLDeviceReading) DeviceRecord() DeviceRecord {
	return DeviceRecord{Index: d.Index, UUID: d.UUID}
}

// ReadNVML initializes lib, reads every device, and shuts it down. A nil lib
// uses DefaultNVMLLibrary. It never returns an empty slice with a nil error: a
// failed Init or a zero device count is ErrNVMLUnavailable, and any per-device
// failure aborts the read with the device index and the NVMLReturn.
func ReadNVML(lib NVMLLibrary) ([]NVMLDeviceReading, error) {
	if lib == nil {
		lib = DefaultNVMLLibrary()
	}
	if ret := lib.Init(); ret != NVMLSuccess {
		if ret.Unavailable() {
			return nil, fmt.Errorf("nvml init: %w", ret.Err())
		}
		return nil, fmt.Errorf("nvml init: %w: %w", ErrNVMLUnavailable, ret)
	}
	defer lib.Shutdown()

	count, ret := lib.DeviceGetCount()
	if ret != NVMLSuccess {
		return nil, fmt.Errorf("nvml device count: %w", ret.Err())
	}
	if count <= 0 {
		return nil, fmt.Errorf("%w: no devices", ErrNVMLUnavailable)
	}
	out := make([]NVMLDeviceReading, 0, count)
	for i := 0; i < count; i++ {
		d, err := readNVMLDevice(lib, i)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, nil
}

func readNVMLDevice(lib NVMLLibrary, i int) (NVMLDeviceReading, error) {
	fail := func(op string, r NVMLReturn) (NVMLDeviceReading, error) {
		return NVMLDeviceReading{}, fmt.Errorf("nvml device %d %s: %w", i, op, r.Err())
	}
	dev, ret := lib.DeviceGetHandleByIndex(i)
	if ret != NVMLSuccess {
		return fail("handle", ret)
	}
	if dev == nil {
		return fail("handle", NVMLErrorGPUIsLost)
	}
	r := NVMLDeviceReading{Index: i}
	if r.UUID, ret = dev.GetUUID(); ret != NVMLSuccess {
		return fail("uuid", ret)
	}
	if r.Name, ret = dev.GetName(); ret != NVMLSuccess {
		return fail("name", ret)
	}
	if r.Memory, ret = dev.GetMemoryInfo_v2(); ret != NVMLSuccess {
		return fail("memory", ret)
	}
	if r.Utilization, ret = dev.GetUtilizationRates(); ret != NVMLSuccess {
		return fail("utilization", ret)
	}
	return r, nil
}
