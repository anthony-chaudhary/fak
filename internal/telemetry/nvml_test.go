package telemetry

import (
	"errors"
	"testing"
)

type fakeNVMLDevice struct {
	uuid, name string
	mem        NVMLMemory
	util       NVMLUtilization
	utilRet    NVMLReturn
}

func (d fakeNVMLDevice) GetUUID() (string, NVMLReturn)              { return d.uuid, NVMLSuccess }
func (d fakeNVMLDevice) GetName() (string, NVMLReturn)              { return d.name, NVMLSuccess }
func (d fakeNVMLDevice) GetMemoryInfo_v2() (NVMLMemory, NVMLReturn) { return d.mem, NVMLSuccess }
func (d fakeNVMLDevice) GetUtilizationRates() (NVMLUtilization, NVMLReturn) {
	return d.util, d.utilRet
}

type fakeNVMLLibrary struct {
	initRet   NVMLReturn
	countRet  NVMLReturn
	devices   []fakeNVMLDevice
	shutdowns int
}

func (l *fakeNVMLLibrary) Init() NVMLReturn     { return l.initRet }
func (l *fakeNVMLLibrary) Shutdown() NVMLReturn { l.shutdowns++; return NVMLSuccess }
func (l *fakeNVMLLibrary) DeviceGetCount() (int, NVMLReturn) {
	return len(l.devices), l.countRet
}
func (l *fakeNVMLLibrary) DeviceGetHandleByIndex(i int) (NVMLDevice, NVMLReturn) {
	if i < 0 || i >= len(l.devices) {
		return nil, NVMLErrorInvalidArgument
	}
	return l.devices[i], NVMLSuccess
}

// TestNVMLTelemetry pins the go-nvml return vocabulary and exercises the
// fail-closed NVML device seam through a fake library.
func TestNVMLTelemetry(t *testing.T) {
	t.Run("pinned_return_codes", func(t *testing.T) {
		// Code and string verbatim from go-nvml const.go / return.go at the pin.
		want := map[int]string{
			0: "SUCCESS", 1: "ERROR_UNINITIALIZED", 2: "ERROR_INVALID_ARGUMENT",
			3: "ERROR_NOT_SUPPORTED", 4: "ERROR_NO_PERMISSION", 5: "ERROR_ALREADY_INITIALIZED",
			6: "ERROR_NOT_FOUND", 7: "ERROR_INSUFFICIENT_SIZE", 8: "ERROR_INSUFFICIENT_POWER",
			9: "ERROR_DRIVER_NOT_LOADED", 10: "ERROR_TIMEOUT", 11: "ERROR_IRQ_ISSUE",
			12: "ERROR_LIBRARY_NOT_FOUND", 13: "ERROR_FUNCTION_NOT_FOUND", 14: "ERROR_CORRUPTED_INFOROM",
			15: "ERROR_GPU_IS_LOST", 16: "ERROR_RESET_REQUIRED", 17: "ERROR_OPERATING_SYSTEM",
			18: "ERROR_LIB_RM_VERSION_MISMATCH", 19: "ERROR_IN_USE", 20: "ERROR_MEMORY",
			21: "ERROR_NO_DATA", 22: "ERROR_VGPU_ECC_NOT_SUPPORTED", 23: "ERROR_INSUFFICIENT_RESOURCES",
			24: "ERROR_FREQ_NOT_SUPPORTED", 25: "ERROR_ARGUMENT_VERSION_MISMATCH", 26: "ERROR_DEPRECATED",
			27: "ERROR_NOT_READY", 28: "ERROR_GPU_NOT_FOUND", 29: "ERROR_INVALID_STATE",
			30: "ERROR_RESET_TYPE_NOT_SUPPORTED", 999: "ERROR_UNKNOWN",
		}
		if len(nvmlReturnNames) != len(want) {
			t.Fatalf("vocabulary has %d codes, want %d", len(nvmlReturnNames), len(want))
		}
		for code, name := range want {
			r, err := NVMLReturnByCode(code)
			if err != nil {
				t.Fatalf("NVMLReturnByCode(%d): %v", code, err)
			}
			if r.Error() != name || r.String() != name {
				t.Fatalf("code %d = %q, want %q", code, r.Error(), name)
			}
		}
		if NVMLErrorLibraryNotFound != 12 || NVMLErrorDriverNotLoaded != 9 || NVMLErrorUnknown != 999 {
			t.Fatalf("named constants drifted from upstream values")
		}
		for _, bad := range []int{-1, 31, 998, 1000, 1 << 40} {
			if _, err := NVMLReturnByCode(bad); !errors.Is(err, ErrNVMLReturnUnknown) {
				t.Fatalf("NVMLReturnByCode(%d): err = %v, want ErrNVMLReturnUnknown", bad, err)
			}
		}
		if got := NVMLReturn(31).Error(); got != "unknown return value: 31" {
			t.Fatalf("NVMLReturn(31).Error() = %q", got)
		}
	})

	t.Run("err_mapping", func(t *testing.T) {
		if NVMLSuccess.Err() != nil {
			t.Fatalf("SUCCESS.Err() should be nil")
		}
		for _, r := range []NVMLReturn{NVMLErrorLibraryNotFound, NVMLErrorDriverNotLoaded} {
			err := r.Err()
			if !errors.Is(err, ErrNVMLUnavailable) || !errors.Is(err, r) {
				t.Fatalf("%v.Err() = %v, want ErrNVMLUnavailable wrapping the code", r, err)
			}
		}
		if err := NVMLErrorNotSupported.Err(); errors.Is(err, ErrNVMLUnavailable) || !errors.Is(err, NVMLErrorNotSupported) {
			t.Fatalf("NOT_SUPPORTED.Err() = %v, want the code and not unavailable", err)
		}
	})

	t.Run("default_library_refuses", func(t *testing.T) {
		for _, lib := range []NVMLLibrary{nil, DefaultNVMLLibrary()} {
			got, err := ReadNVML(lib)
			if !errors.Is(err, ErrNVMLUnavailable) || !errors.Is(err, NVMLErrorLibraryNotFound) {
				t.Fatalf("ReadNVML(default): err = %v, want unavailable/LIBRARY_NOT_FOUND", err)
			}
			if got != nil {
				t.Fatalf("ReadNVML(default) returned %d devices, want nil", len(got))
			}
		}
	})

	t.Run("no_driver_refuses", func(t *testing.T) {
		lib := &fakeNVMLLibrary{initRet: NVMLErrorDriverNotLoaded}
		got, err := ReadNVML(lib)
		if !errors.Is(err, ErrNVMLUnavailable) || !errors.Is(err, NVMLErrorDriverNotLoaded) || got != nil {
			t.Fatalf("ReadNVML(no driver) = %v, %v", got, err)
		}
		if lib.shutdowns != 0 {
			t.Fatalf("Shutdown called after failed Init")
		}
	})

	t.Run("zero_devices_refuses", func(t *testing.T) {
		lib := &fakeNVMLLibrary{}
		got, err := ReadNVML(lib)
		if !errors.Is(err, ErrNVMLUnavailable) || got != nil {
			t.Fatalf("ReadNVML(zero devices) = %v, %v; want ErrNVMLUnavailable", got, err)
		}
		if lib.shutdowns != 1 {
			t.Fatalf("Shutdown calls = %d, want 1", lib.shutdowns)
		}
	})

	t.Run("reads_devices_in_order", func(t *testing.T) {
		lib := &fakeNVMLLibrary{devices: []fakeNVMLDevice{
			{uuid: "GPU-aaaa", name: "NVIDIA GeForce RTX 5080", mem: NVMLMemory{Total: 16 << 30, Reserved: 1 << 28, Free: 8 << 30, Used: 8<<30 - 1<<28}, util: NVMLUtilization{GPU: 97, Memory: 40}},
			{uuid: "GPU-bbbb", name: "NVIDIA H200", mem: NVMLMemory{Total: 141 << 30}, util: NVMLUtilization{GPU: 0, Memory: 0}},
		}}
		got, err := ReadNVML(lib)
		if err != nil {
			t.Fatalf("ReadNVML: %v", err)
		}
		if len(got) != 2 || lib.shutdowns != 1 {
			t.Fatalf("got %d devices, %d shutdowns; want 2, 1", len(got), lib.shutdowns)
		}
		if got[0].Index != 0 || got[0].UUID != "GPU-aaaa" || got[0].Utilization.GPU != 97 || got[0].Memory.Total != 16<<30 {
			t.Fatalf("device 0 = %+v", got[0])
		}
		if got[1].Index != 1 || got[1].Name != "NVIDIA H200" {
			t.Fatalf("device 1 = %+v", got[1])
		}
		if rec := got[1].DeviceRecord(); rec.Index != 1 || rec.UUID != "GPU-bbbb" || rec.Values != nil {
			t.Fatalf("DeviceRecord bridge = %+v", rec)
		}
	})

	t.Run("device_failure_aborts", func(t *testing.T) {
		lib := &fakeNVMLLibrary{devices: []fakeNVMLDevice{
			{uuid: "GPU-aaaa"},
			{uuid: "GPU-bbbb", utilRet: NVMLErrorGPUIsLost},
		}}
		got, err := ReadNVML(lib)
		if !errors.Is(err, NVMLErrorGPUIsLost) || errors.Is(err, ErrNVMLUnavailable) || got != nil {
			t.Fatalf("ReadNVML(lost gpu) = %v, %v; want GPU_IS_LOST", got, err)
		}
		if lib.shutdowns != 1 {
			t.Fatalf("Shutdown calls = %d, want 1", lib.shutdowns)
		}
	})
}
