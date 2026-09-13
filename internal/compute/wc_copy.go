package compute

import (
	"fmt"
)

// This file is the additive write-combined (WC) copy adapter seam. It sits alongside
// devicecopy.go (which owns the CPU-reference CopyFromWriteCombined implementation) and
// selects a device-DMA transport over that retained CPU memcpy reference when a device
// surface is available and capable. It links no CUDA driver and uses no cgo: the device
// surface is an injected interface, so tests and callers supply either a fake or a real
// out-of-tree DMA engine.
//
// Motivation (fak#12950): a CPU memcpy of WC device memory ran at ~200 MB/s and consumed
// ~64% of the big-gate time; a device DMA while the GPU is parked avoids that load loop.
// This adapter does NOT assert any throughput figure. The measured DMA-vs-memcpy delta is
// [HW-WITNESSED] and is unmeasured here; only path selection and byte accounting are
// verified in the package's pure-Go tests.

// WCCopyPath identifies which mechanism executed a write-combined copy.
type WCCopyPath string

const (
	// WCCopyPathDeviceDMA copies WC device memory via a device DMA engine parked while
	// the GPU is idle, avoiding a CPU load loop over the WC mapping.
	WCCopyPathDeviceDMA WCCopyPath = "device_dma"

	// WCCopyPathCPUMemcpy is the retained CPU-memcpy reference path
	// (CopyFromWriteCombined).
	WCCopyPathCPUMemcpy WCCopyPath = "cpu_memcpy"
)

// WCDeviceDMASurface is the minimal device-side transport the WC-copy adapter needs.
// It models a DMA engine parked while the GPU is idle: CopyDeviceToHost transfers bytes
// from write-combined device memory to host memory without a CPU load loop.
type WCDeviceDMASurface interface {
	// SupportsDeviceDMA reports whether the surface can currently serve a device-DMA copy.
	// A surface that is present but momentarily incapable (e.g. engine not parked) returns
	// false, which selects the CPU reference path.
	SupportsDeviceDMA() bool

	// CopyDeviceToHost transfers min(len(dst), len(src)) bytes from WC device memory (src)
	// to host memory (dst) via device DMA and returns the number of bytes copied. A non-nil
	// error means the DMA failed; callers must surface it rather than masking it.
	CopyDeviceToHost(dst, src []byte) (int, error)
}

// WCDeviceErrorKind classifies a WC-copy device failure.
type WCDeviceErrorKind string

const (
	// WCDeviceErrorKindDMA identifies a failure inside the device-DMA transport itself.
	WCDeviceErrorKindDMA WCDeviceErrorKind = "dma_failed"

	// WCDeviceErrorKindShort identifies a success-path byte count shorter than the
	// requested transfer length (min(len(dst), len(src))).
	WCDeviceErrorKindShort WCDeviceErrorKind = "short_transfer"

	// WCDeviceErrorKindInvalidExtent identifies a malformed transfer extent.
	WCDeviceErrorKindInvalidExtent WCDeviceErrorKind = "invalid_extent"
)

// WCCopyDeviceError is the typed, fail-closed error returned when the device-DMA path is
// selected but cannot complete. It is never swallowed into a CPU fallback: device failure
// must surface, only an ABSENT or INCAPABLE surface selects the CPU reference path.
type WCCopyDeviceError struct {
	Kind     WCDeviceErrorKind
	Copied   int
	Expected int
	Err      error
}

// Error formats the device failure for operator logs and diagnostic returns.
func (e *WCCopyDeviceError) Error() string {
	if e == nil {
		return "compute: nil WC device copy error"
	}
	msg := fmt.Sprintf("compute: write-combined device DMA failed [%s]", e.Kind)
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	if e.Kind == WCDeviceErrorKindShort {
		msg += fmt.Sprintf(" (copied %d of %d bytes)", e.Copied, e.Expected)
	}
	return msg
}

// Unwrap exposes the underlying transport error to errors.Is / errors.As.
func (e *WCCopyDeviceError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// WCDeviceError is retained as the named Kind enum the spec calls for, so callers can
// reference WCDeviceErrorKind values without ambiguity with the error type name.
const (
	// WCDeviceError aliases the DMA failure kind for readability.
	WCDeviceError = WCDeviceErrorKindDMA
)

// WCCopyResult reports which path executed a copy and how many bytes moved.
type WCCopyResult struct {
	Path  WCCopyPath
	Bytes int
	Err   error
}

// WCDeviceCopyAdapter selects a device-DMA path over the retained CPU memcpy reference.
// A nil surface is valid and pins the adapter to the CPU reference path.
type WCDeviceCopyAdapter struct {
	surface WCDeviceDMASurface
}

// NewWCDeviceCopyAdapter creates an adapter over an optional device-DMA surface. A nil
// surface is allowed and means "no device DMA available"; all copies then use the CPU
// reference path.
func NewWCDeviceCopyAdapter(surface WCDeviceDMASurface) *WCDeviceCopyAdapter {
	return &WCDeviceCopyAdapter{surface: surface}
}

// Surface returns the adapter's device surface, or nil when none is attached.
func (a *WCDeviceCopyAdapter) Surface() WCDeviceDMASurface {
	if a == nil {
		return nil
	}
	return a.surface
}

// Copy transfers min(len(dst), len(src)) bytes from src into dst.
//
// Selection rule: if a surface is present AND SupportsDeviceDMA() reports true, the device
// DMA path runs and Path is WCCopyPathDeviceDMA. Otherwise (absent surface, or present but
// incapable) the retained CPU reference CopyFromWriteCombined runs and Path is
// WCCopyPathCPUMemcpy.
//
// Fail-closed: once the device path is selected, a DMA error is surfaced as a typed
// *WCCopyDeviceError with Path left WCCopyPathDeviceDMA; the adapter never flips to the CPU
// path mid-copy to mask a device failure.
func (a *WCDeviceCopyAdapter) Copy(dst, src []byte) (WCCopyResult, error) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}

	if n <= 0 {
		return WCCopyResult{
			Path:  WCCopyPathCPUMemcpy,
			Bytes: 0,
			Err:   nil,
		}, nil
	}

	if a != nil && a.surface != nil && a.surface.SupportsDeviceDMA() {
		copied, err := a.surface.CopyDeviceToHost(dst, src)
		if err != nil {
			devErr := &WCCopyDeviceError{
				Kind:     WCDeviceErrorKindDMA,
				Copied:   copied,
				Expected: n,
				Err:      err,
			}
			return WCCopyResult{
				Path:  WCCopyPathDeviceDMA,
				Bytes: copied,
				Err:   devErr,
			}, devErr
		}
		if copied != n {
			devErr := &WCCopyDeviceError{
				Kind:     WCDeviceErrorKindShort,
				Copied:   copied,
				Expected: n,
			}
			return WCCopyResult{
				Path:  WCCopyPathDeviceDMA,
				Bytes: copied,
				Err:   devErr,
			}, devErr
		}
		return WCCopyResult{
			Path:  WCCopyPathDeviceDMA,
			Bytes: copied,
			Err:   nil,
		}, nil
	}

	bytes := CopyFromWriteCombined(dst, src)
	return WCCopyResult{
		Path:  WCCopyPathCPUMemcpy,
		Bytes: bytes,
		Err:   nil,
	}, nil
}
