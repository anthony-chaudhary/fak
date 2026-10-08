//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include <stdlib.h>
#include "vulkan_backend.h"

int fvk_restore_begin(size_t max_bytes, size_t max_entries);
int fvk_restore_add(void *dst, size_t dst_offset, const void *src, size_t bytes);
int fvk_restore_submit(void);
void fvk_restore_finish(void);
void fvk_restore_abort(void);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const (
	vulkanKVRestoreBatchBytes   = 16 << 20
	vulkanKVRestoreBatchEntries = 64
)

// SnapshotToHost copies the complete F32 Vulkan KV owner into ordinary host
// DRAM. Packed precision is refused because KVHostSnapshot cannot represent its
// native bit layout without a lossy dequantize/requantize round trip.
func (k *vulkanKV) SnapshotToHost() (KVHostSnapshot, error) {
	if k == nil {
		return KVHostSnapshot{}, fmt.Errorf("vulkan: cannot snapshot nil KV store")
	}
	if k.cfg.Precision != KVPrecisionF32 {
		return KVHostSnapshot{}, fmt.Errorf("%w: vulkan KV precision %s cannot be preserved exactly", ErrKVHostSnapshotUnsupported, k.cfg.Precision)
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	d2hBefore := uint64(C.fvk_d2h_bytes())
	out := KVHostSnapshot{
		Config: cloneKVConfig(k.cfg), Pos: append([]int(nil), k.pos...),
		K: make([][]float32, len(k.K)), KRaw: make([][]float32, len(k.Kraw)), V: make([][]float32, len(k.V)),
	}
	for layer := range k.K {
		var err error
		out.K[layer], err = vulkanKVReadHost(&k.K[layer])
		if err != nil {
			return KVHostSnapshot{}, err
		}
		out.KRaw[layer], err = vulkanKVReadHost(&k.Kraw[layer])
		if err != nil {
			return KVHostSnapshot{}, err
		}
		out.V[layer], err = vulkanKVReadHost(&k.V[layer])
		if err != nil {
			return KVHostSnapshot{}, err
		}
	}
	if err := out.Validate(); err != nil {
		return KVHostSnapshot{}, err
	}
	wantBytes := uint64(out.TransferBytes())
	d2hAfter := uint64(C.fvk_d2h_bytes())
	if d2hAfter < d2hBefore || d2hAfter-d2hBefore != wantBytes {
		return KVHostSnapshot{}, vulkanKVTransferError("SnapshotToHost", fmt.Sprintf("device-to-host bytes=%d, want %d", d2hAfter-d2hBefore, wantBytes))
	}
	return out, nil
}

func vulkanKVReadHost(src *vslice) ([]float32, error) {
	out := make([]float32, src.len)
	if len(out) == 0 {
		return out, nil
	}
	if status := int(C.fvk_d2h(unsafe.Pointer(&out[0]), src.ptr, C.size_t(len(out)*F32.Bytes()))); status != 0 {
		return nil, vulkanReadError(status)
	}
	return out, nil
}

// RestoreKVFromHost bulk-copies a complete F32 host image into fresh Vulkan KV
// allocations. It does not run model forward or prefill work.
func (v *vulkanBackend) RestoreKVFromHost(state KVHostSnapshot) (out KVStore, err error) {
	if err := state.Validate(); err != nil {
		return nil, err
	}
	if state.Config.Precision != KVPrecisionF32 {
		return nil, fmt.Errorf("%w: vulkan KV precision %s cannot be preserved exactly", ErrKVHostSnapshotUnsupported, state.Config.Precision)
	}
	if v == nil {
		return nil, fmt.Errorf("vulkan: cannot restore KV without a backend")
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	h2dBefore := uint64(C.fvk_h2d_bytes())
	k := &vulkanKV{
		be: v, cfg: cloneKVConfig(state.Config), pos: append([]int(nil), state.Pos...),
		K: make([]vslice, len(state.K)), Kraw: make([]vslice, len(state.KRaw)), V: make([]vslice, len(state.V)),
	}
	freePartial := func() {
		for layer := range k.K {
			k.K[layer].releaseBacking()
			k.Kraw[layer].releaseBacking()
			k.V[layer].releaseBacking()
		}
		k.pos = nil
	}
	defer func() {
		if r := recover(); r != nil {
			freePartial()
			out = nil
			if caught, ok := r.(error); ok {
				err = fmt.Errorf("vulkan: restore KV from host: %w", caught)
			} else {
				err = fmt.Errorf("vulkan: restore KV from host: %v", r)
			}
		}
	}()
	cleanup := true
	defer func() {
		if cleanup {
			freePartial()
		}
	}()
	wantBytes := uint64(state.TransferBytes())
	if wantBytes == 0 {
		cleanup = false
		return k, nil
	}
	for layer := range state.K {
		for _, row := range []struct {
			dst  *vslice
			src  []float32
			site string
		}{
			{&k.K[layer], state.K[layer], fmt.Sprintf("kv-key-host-restore layer %d", layer)},
			{&k.Kraw[layer], state.KRaw[layer], fmt.Sprintf("kv-pre-rope-key-host-restore layer %d", layer)},
			{&k.V[layer], state.V[layer], fmt.Sprintf("kv-value-host-restore layer %d", layer)},
		} {
			if len(row.src) == 0 {
				continue
			}
			buf := v.dallocKVFor(len(row.src)*F32.Bytes(), row.site)
			row.dst.ptr, row.dst.len, row.dst.cap = buf.ptr, len(row.src), len(row.src)
			row.dst.backing = &vulkanKVBacking{ptr: buf.ptr, cap: len(row.src), refs: 1, highWater: len(row.src)}
		}
	}
	if status := int(C.fvk_restore_begin(C.size_t(vulkanKVRestoreBatchBytes), C.size_t(vulkanKVRestoreBatchEntries))); status != 0 {
		return nil, vulkanKVRestoreStatusError("begin", status)
	}
	defer C.fvk_restore_abort()
	batchBytes, batchEntries := 0, 0
	submit := func() error {
		if batchEntries == 0 {
			return nil
		}
		if status := int(C.fvk_restore_submit()); status != 0 {
			return vulkanKVRestoreStatusError("submit", status)
		}
		batchBytes, batchEntries = 0, 0
		return nil
	}
	for layer := range state.K {
		for _, row := range []struct {
			dst *vslice
			src []float32
		}{
			{&k.K[layer], state.K[layer]}, {&k.Kraw[layer], state.KRaw[layer]}, {&k.V[layer], state.V[layer]},
		} {
			for off := 0; off < len(row.src); {
				if batchEntries == vulkanKVRestoreBatchEntries || batchBytes == vulkanKVRestoreBatchBytes {
					if err := submit(); err != nil {
						return nil, err
					}
				}
				floats := min(len(row.src)-off, (vulkanKVRestoreBatchBytes-batchBytes)/F32.Bytes())
				if floats == 0 {
					if err := submit(); err != nil {
						return nil, err
					}
					continue
				}
				bytes := floats * F32.Bytes()
				status := int(C.fvk_restore_add(row.dst.ptr, C.size_t(off*F32.Bytes()), unsafe.Pointer(&row.src[off]), C.size_t(bytes)))
				if status != 0 {
					return nil, vulkanKVRestoreStatusError("add", status)
				}
				batchBytes += bytes
				batchEntries++
				off += floats
			}
		}
	}
	if err := submit(); err != nil {
		return nil, err
	}
	C.fvk_restore_finish()
	h2dAfter := uint64(C.fvk_h2d_bytes())
	if h2dAfter < h2dBefore || h2dAfter-h2dBefore != wantBytes {
		return nil, vulkanKVTransferError("RestoreKVFromHost", fmt.Sprintf("host-to-device bytes=%d, want %d", h2dAfter-h2dBefore, wantBytes))
	}
	cleanup = false
	return k, nil
}

func vulkanKVRestoreStatusError(stage string, status int) error {
	if status < 0 {
		err := vulkanReadError(status)
		err.Site = "RestoreKVFromHost"
		err.Message = fmt.Sprintf("%s failed with native status %d", stage, status)
		return err
	}
	class, sentinel := VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed
	if stage == "begin" && (status == 2 || status == 3) {
		class, sentinel = VulkanClassAllocationFailed, ErrVulkanAllocationFailed
	}
	return &BackendError{Backend: "vulkan", Class: class, Site: "RestoreKVFromHost", Err: sentinel, Message: fmt.Sprintf("%s failed with native status %d", stage, status)}
}

func vulkanKVTransferError(site, message string) error {
	return &BackendError{Backend: "vulkan", Class: VulkanClassSubmissionFailed, Site: site, Err: ErrVulkanSubmissionFailed, Message: message}
}
