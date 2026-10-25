//go:build cuda && cgo

package compute

/*
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAHop is the CUDA runtime HiddenStateHop (device_handoff.go). It stages the hidden
// state on the source device, crosses to the destination device by the selected route
// (cudaMemcpyPeer, or pinned host bounce), and reads it back from the destination. Until
// bands run on-device it is the hardware witness for the boundary copy itself; the bytes
// it moves are exactly the bytes a device-resident band will hand on.
type CUDAHop struct {
	ForceBounce  bool
	StagingBytes int // pinned staging size; 0 selects DefaultHandoffStagingBytes
}

// NewCUDAHopFromEnv honors FAK_CUDA_HANDOFF=bounce.
func NewCUDAHopFromEnv() CUDAHop { return CUDAHop{ForceBounce: HandoffForceBounceFromEnv()} }

func cudaHandoffErr(stage string, rc C.int) error {
	if rc == 0 {
		return nil
	}
	return fmt.Errorf("%w: %s status %d", ErrHandoffFailed, stage, int(rc))
}

// CUDADeviceCount reports the CUDA devices visible to this process.
func CUDADeviceCount() (int, error) {
	var n C.int
	if err := cudaHandoffErr("device count", C.fcuda_handoff_device_count(&n)); err != nil {
		return 0, err
	}
	return int(n), nil
}

// CUDAPeerAccess reports whether device src can copy straight into device dst.
func CUDAPeerAccess(src, dst int) (bool, error) {
	var can C.int
	if err := cudaHandoffErr("peer probe", C.fcuda_handoff_peer_probe(C.int(src), C.int(dst), &can)); err != nil {
		return false, err
	}
	return can != 0, nil
}

// HopF32 implements HiddenStateHop on real devices.
func (h CUDAHop) HopF32(src []float32, srcDev, dstDev int) ([]float32, DeviceHandoffReceipt, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	n, err := CUDADeviceCount()
	if err != nil {
		return nil, DeviceHandoffReceipt{}, err
	}
	if srcDev >= n || dstDev >= n {
		return nil, DeviceHandoffReceipt{}, fmt.Errorf("%w: devices %d->%d with %d visible", ErrHandoffInvalid, srcDev, dstDev, n)
	}
	route, err := SelectHandoffRoute(srcDev, dstDev, CUDAPeerAccess, h.ForceBounce)
	if err != nil {
		return nil, DeviceHandoffReceipt{}, err
	}
	bytes := len(src) * 4
	rec := DeviceHandoffReceipt{Src: srcDev, Dst: dstDev, Route: route, Bytes: int64(bytes), Chunks: 1}
	out := make([]float32, len(src))
	if bytes == 0 {
		return out, rec, nil
	}
	dSrc := C.fcuda_handoff_alloc(C.int(srcDev), C.size_t(bytes))
	if dSrc == nil {
		return nil, DeviceHandoffReceipt{}, fmt.Errorf("%w: alloc %d B on device %d", ErrHandoffFailed, bytes, srcDev)
	}
	defer C.fcuda_handoff_free(C.int(srcDev), dSrc)
	dDst := C.fcuda_handoff_alloc(C.int(dstDev), C.size_t(bytes))
	if dDst == nil {
		return nil, DeviceHandoffReceipt{}, fmt.Errorf("%w: alloc %d B on device %d", ErrHandoffFailed, bytes, dstDev)
	}
	defer C.fcuda_handoff_free(C.int(dstDev), dDst)
	if err := cudaHandoffErr("stage h2d", C.fcuda_handoff_h2d(C.int(srcDev), dSrc, unsafe.Pointer(&src[0]), C.size_t(bytes))); err != nil {
		return nil, DeviceHandoffReceipt{}, err
	}

	switch route {
	case HandoffPeer:
		if err := cudaHandoffErr("peer enable", C.fcuda_handoff_peer_enable(C.int(srcDev), C.int(dstDev))); err != nil {
			return nil, DeviceHandoffReceipt{}, err
		}
		fallthrough
	case HandoffSameDevice:
		if err := cudaHandoffErr("peer copy", C.fcuda_handoff_peer_copy(C.int(dstDev), dDst, C.int(srcDev), dSrc, C.size_t(bytes))); err != nil {
			return nil, DeviceHandoffReceipt{}, err
		}
	case HandoffHostBounce:
		staging := h.StagingBytes
		if staging == 0 {
			staging = DefaultHandoffStagingBytes
		}
		chunks, err := HostBounceChunks(bytes, min(staging, bytes))
		if err != nil {
			return nil, DeviceHandoffReceipt{}, err
		}
		pinned := C.fcuda_handoff_pinned_alloc(C.size_t(chunks[0].Len))
		if pinned == nil {
			return nil, DeviceHandoffReceipt{}, fmt.Errorf("%w: pinned staging %d B", ErrHandoffFailed, chunks[0].Len)
		}
		defer C.fcuda_handoff_pinned_free(pinned)
		for _, c := range chunks {
			if err := cudaHandoffErr("bounce d2h", C.fcuda_handoff_d2h(C.int(srcDev), pinned, unsafe.Add(dSrc, c.Off), C.size_t(c.Len))); err != nil {
				return nil, DeviceHandoffReceipt{}, err
			}
			if err := cudaHandoffErr("bounce h2d", C.fcuda_handoff_h2d(C.int(dstDev), unsafe.Add(dDst, c.Off), pinned, C.size_t(c.Len))); err != nil {
				return nil, DeviceHandoffReceipt{}, err
			}
		}
		rec.Chunks = len(chunks)
	default:
		return nil, DeviceHandoffReceipt{}, fmt.Errorf("%w: route %s", ErrHandoffInvalid, route)
	}
	if err := cudaHandoffErr("readback d2h", C.fcuda_handoff_d2h(C.int(dstDev), unsafe.Pointer(&out[0]), dDst, C.size_t(bytes))); err != nil {
		return nil, DeviceHandoffReceipt{}, err
	}
	return out, rec, nil
}
