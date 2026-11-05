//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
#include "cuda_kv_precision.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// The retained store roots its pool, page references, table and private buffers.
// Keep the exact gather owner and both kinds of Go backing storage beside it.
type cudaPagedKVSnapshotCopy struct {
	scratch  *cudaBuf
	host     []float32
	hostPin  runtime.Pinner
	tablePin runtime.Pinner
}

func (k *cudaPagedKV) snapshotFailureLocked(site string, status int) error {
	C.fcuda_kv_snapshot_failed(C.int(status))
	k.pool.copyFailure.CompareAndSwap(0, int32(status))
	k.pool.retainCopyOwnerLocked(k)
	k.pool.be.faultLatch.Observe(DeviceFaultExecution, site, status)
	return &CUDAOpError{Op: "SnapshotToHost", Site: site, Code: status, Msg: "paged snapshot failed; owners retained"}
}

// Caller holds cudaMu and has passed pool admission. The table upload performed
// by syncDeviceTableLocked remains a legacy void-H2D dependency; this function
// certifies neither every table transfer nor general snapshot correctness.
func (k *cudaPagedKV) snapshotToHostLocked() (KVHostSnapshot, error) {
	if err := AdmitDevice(k.pool.be, "paged-kv-snapshot"); err != nil {
		return KVHostSnapshot{}, err
	}
	if k.cfg.NumLayers < 0 || len(k.lens) != k.cfg.NumLayers || len(k.pool.K) != k.cfg.NumLayers || len(k.pool.Kraw) != k.cfg.NumLayers || len(k.pool.V) != k.cfg.NumLayers {
		return KVHostSnapshot{}, fmt.Errorf("cuda: inconsistent paged snapshot inventory")
	}
	// Validate before multiplying or narrowing the native gather dimensions.
	const nativeMax = int(2147483647)
	if k.cfg.NumKVHeads <= 0 || k.cfg.HeadDim <= 0 || k.cfg.HeadDim > nativeMax || k.cfg.NumKVHeads > nativeMax/k.cfg.HeadDim || k.pool.blockTokens <= 0 || k.pool.blockTokens > nativeMax {
		return KVHostSnapshot{}, fmt.Errorf("cuda: invalid paged snapshot native geometry")
	}
	w := k.cfg.NumKVHeads * k.cfg.HeadDim
	maxN := 0
	for _, n := range k.lens {
		if n < 0 || n > nativeMax {
			return KVHostSnapshot{}, fmt.Errorf("cuda: negative paged snapshot length")
		}
		maxN = max(maxN, n)
	}
	if maxN > 0 && (w <= 0 || maxN > int(^uint(0)>>1)/4/w) {
		return KVHostSnapshot{}, fmt.Errorf("cuda: invalid paged snapshot extent")
	}
	out := KVHostSnapshot{Config: cloneKVConfig(k.cfg), Pos: append([]int(nil), k.pos...), K: make([][]float32, k.cfg.NumLayers), KRaw: make([][]float32, k.cfg.NumLayers), V: make([][]float32, k.cfg.NumLayers)}
	holder := &cudaPagedKVSnapshotCopy{}
	k.snapshotCopy = holder
	completed := false
	defer func() {
		// No recover: panic(nil) and Goexit preserve their original unwind. Rooting
		// alone cannot stop explicit Free or sibling pool reuse, so close both gates.
		if !completed {
			status := k.pool.copyFailure.Load()
			unreported := status == 0
			if status == 0 {
				status = -75707
				k.pool.copyFailure.CompareAndSwap(0, status)
			}
			C.fcuda_kv_snapshot_failed(C.int(status))
			k.pool.retainCopyOwnerLocked(k)
			if unreported {
				k.pool.be.faultLatch.Observe(DeviceFaultExecution, "paged-kv-snapshot-unwind", int(status))
			}
			return
		}
		holder.hostPin.Unpin()
		holder.tablePin.Unpin()
		k.snapshotCopy = nil
	}()
	// Publish holder and unwind before allocation, pinning or table preparation.
	if maxN > 0 {
		holder.scratch = k.pool.be.dallocKV(maxN*w*4, "kv-paged-snapshot")
	}
	if len(k.table) > 0 {
		holder.tablePin.Pin(&k.table[0])
	}
	read := func(arena unsafe.Pointer, layer int, keyLayout bool) ([]float32, error) {
		n := k.lens[layer]
		rows := make([]float32, n*w)
		if n == 0 {
			return rows, nil
		}
		holder.host = rows
		holder.hostPin.Pin(&rows[0])
		k.syncDeviceTableLocked(n)
		kl := 0
		if keyLayout {
			kl = 1
		}
		status := int(C.fcuda_paged_kv_snapshot_gather_checked((*C.float)(holder.scratch.ptr), (*C.float)(arena), (*C.int)(k.dTable), C.int(n), C.int(k.cfg.NumKVHeads), C.int(k.cfg.HeadDim), C.int(k.pool.blockTokens), C.int(kl)))
		if status != 0 {
			return nil, k.snapshotFailureLocked("paged-kv-snapshot-gather", status)
		}
		status = int(C.fcuda_kv_snapshot_d2h_checked(unsafe.Pointer(&rows[0]), holder.scratch.ptr, C.size_t(len(rows)*4)))
		if status != 0 {
			return nil, k.snapshotFailureLocked("paged-kv-snapshot-d2h", status)
		}
		holder.hostPin.Unpin()
		holder.host = nil
		return rows, nil
	}
	for l := 0; l < k.cfg.NumLayers; l++ {
		var err error
		if out.K[l], err = read(k.pool.K[l], l, true); err != nil {
			return KVHostSnapshot{}, err
		}
		if out.KRaw[l], err = read(k.pool.Kraw[l], l, true); err != nil {
			return KVHostSnapshot{}, err
		}
		if out.V[l], err = read(k.pool.V[l], l, false); err != nil {
			return KVHostSnapshot{}, err
		}
	}
	// Preserve the existing healthy void-free semantics, not a new checked
	// retirement claim. A panic here still retains the exact scratch owner.
	if holder.scratch != nil {
		C.fcuda_free(holder.scratch.ptr)
	}
	completed = true
	return out, out.Validate()
}
