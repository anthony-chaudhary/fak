//go:build linux && rocm && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#include "rocm_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"sync/atomic"
	"unsafe"
)

type rocmRows struct {
	ptr        unsafe.Pointer
	rows, cap  int
	generation uint64
}

type rocmKV struct {
	be         *rocmBackend
	cfg        KVConfig
	K, Kraw, V []rocmRows
	pos        []int
	// failure is written once under rocmMu before poisoned publishes it.
	failure  any
	poisoned uint32
}

func (r *rocmBackend) NewKV(cfg KVConfig) KVStore {
	if cfg.NumLayers <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 || cfg.RopeTheta <= 0 || math.IsNaN(cfg.RopeTheta) || math.IsInf(cfg.RopeTheta, 0) {
		panic(fmt.Errorf("rocm: invalid KV geometry %+v", cfg))
	}
	if _, ok := checkedROCmBytes([]int{cfg.NumKVHeads, cfg.HeadDim}, F32.Bytes()); !ok {
		panic(fmt.Errorf("rocm: KV row geometry overflows allocation capacity"))
	}
	if cfg.Precision != KVPrecisionF32 {
		panic(fmt.Errorf("rocm: KV precision %v is unsupported; require f32", cfg.Precision))
	}
	return &rocmKV{be: r, cfg: cloneKVConfig(cfg), K: make([]rocmRows, cfg.NumLayers), Kraw: make([]rocmRows, cfg.NumLayers), V: make([]rocmRows, cfg.NumLayers)}
}

// ensureUsable rejects logical reuse after ambiguous native retirement. It does
// not acquire rocmMu: native consumers already hold that lock. The atomic flag
// also lets borrowed views reject access without reading the failure payload.
func (k *rocmKV) ensureUsable() {
	if atomic.LoadUint32(&k.poisoned) != 0 {
		panic(k.failure)
	}
}

// poison retains the first cause. Callers hold rocmMu; publication makes the
// immutable cause visible before any later consumer observes the flag.
func (k *rocmKV) poison(primary any) {
	if atomic.LoadUint32(&k.poisoned) == 0 {
		k.failure = primary
		atomic.StoreUint32(&k.poisoned, 1)
	}
}

// Geometry and byte/position queries remain diagnostic metadata after failure.
// In particular, zero logical bytes after Free does not prove physical reclamation.
func (k *rocmKV) KVConfig() KVConfig { return cloneKVConfig(k.cfg) }
func (k *rocmKV) stride() int        { return k.cfg.NumKVHeads * k.cfg.HeadDim }
func (k *rocmKV) Len() int           { return len(k.pos) }
func (k *rocmKV) Pos() []int         { return append([]int(nil), k.pos...) }

func (k *rocmKV) ResidentBytes() int64 {
	var n int64
	for l := range k.K {
		for _, rows := range []int{k.K[l].rows, k.Kraw[l].rows, k.V[l].rows} {
			bytes, ok := checkedROCmBytes([]int{rows, k.stride()}, F32.Bytes())
			if !ok || int64(bytes) > math.MaxInt64-n {
				panic(fmt.Errorf("rocm: KV resident byte count overflow"))
			}
			n += int64(bytes)
		}
	}
	return n
}

func (k *rocmKV) grow(row *rocmRows, need int, site string) {
	k.ensureUsable()
	if need <= row.cap {
		return
	}
	ncap := need
	if row.cap <= (int(^uint(0)>>1)-1)/2 && row.cap*2+1 > ncap {
		ncap = row.cap*2 + 1
	}
	nbytes, ok := checkedROCmBytes([]int{ncap, k.stride()}, F32.Bytes())
	if !ok {
		panic(fmt.Errorf("rocm: %s capacity overflows allocation size", site))
	}
	nb := k.be.alloc(nbytes, MemoryKVCache, site)
	if row.rows > 0 {
		used, _ := checkedROCmBytes([]int{row.rows, k.stride()}, F32.Bytes())
		finishROCmKVGrowthCopy(nb, func() {
			rocmCheck(C.frocm_d2d(nb.ptr, row.ptr, C.size_t(used)), site+" copy")
		}, k.be.freeBuf)
	}
	k.finishGrowthRetirement(row, nb, ncap, func(ptr unsafe.Pointer) {
		rocmCheck(C.frocm_free(ptr), site+" old free")
	}, k.be.freeBuf)
}

// finishGrowthRetirement owns the unpublished replacement after prefix copy.
// Detach the old pointer and invalidate its views BEFORE trying native Free:
// a failed Free does not establish whether that pointer still exists. On failure
// the whole store becomes unusable; only one-attempt best-effort cleanup remains.
// Production grow holds rocmMu; the callbacks expose this same ownership seam.
func (k *rocmKV) finishGrowthRetirement(row *rocmRows, owned *rocmBuf, capacity int, retire func(unsafe.Pointer), releaseNew func(*rocmBuf)) {
	old := row.ptr
	row.ptr = nil
	atomic.AddUint64(&row.generation, 1)
	defer func() {
		if primary := recover(); primary != nil {
			k.poison(primary)
			func() {
				defer func() { _ = recover() }()
				releaseNew(owned)
			}()
			panic(primary)
		}
	}()
	if old != nil {
		retire(old)
	}
	row.ptr, row.cap = owned.ptr, capacity
}

// finishROCmKVGrowthCopy owns only the new buffer during prefix copying, before
// any attempt to retire the old row. grow holds rocmMu. A copy failure releases
// the unpublished destination and leaves row metadata unchanged. This does not
// establish whole-grow atomicity; finishGrowthRetirement owns the later phase.
func finishROCmKVGrowthCopy(owned *rocmBuf, copyPrefix func(), release func(*rocmBuf)) {
	defer func() {
		if primary := recover(); primary != nil {
			func() {
				defer func() { _ = recover() }()
				release(owned)
			}()
			panic(primary)
		}
	}()
	copyPrefix()
}

func (k *rocmKV) AppendKV(layer int, raw, rope, value Tensor, pos int) {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	k.ensureUsable()
	if layer < 0 || layer >= k.cfg.NumLayers || pos < 0 {
		panic(fmt.Errorf("rocm: KV append invalid layer/position"))
	}
	w := k.stride()
	rb, kb, vb := requireF32Owned(k.be, raw, "kv raw key"), requireF32Owned(k.be, rope, "kv key"), requireF32Owned(k.be, value, "kv value")
	rn, _ := checkedTensorNumel(raw.Shape)
	kn, _ := checkedTensorNumel(rope.Shape)
	vn, _ := checkedTensorNumel(value.Shape)
	if rn != w || kn != w || vn != w {
		panic(fmt.Errorf("rocm: KV append row width mismatch"))
	}
	row := k.K[layer].rows
	if k.Kraw[layer].rows != row || k.V[layer].rows != row {
		panic(fmt.Errorf("rocm: KV layer %d is internally ragged", layer))
	}
	if row > len(k.pos) || (row < len(k.pos) && k.pos[row] != pos) {
		panic(fmt.Errorf("rocm: KV layer %d append position %d does not match shared sequence", layer, pos))
	}
	if row == int(^uint(0)>>1) {
		panic(fmt.Errorf("rocm: KV append position count overflows"))
	}
	k.grow(&k.K[layer], row+1, "kv-key-grow")
	k.grow(&k.Kraw[layer], row+1, "kv-raw-grow")
	k.grow(&k.V[layer], row+1, "kv-value-grow")
	rocmCheck(C.frocm_kv_append_f32((*C.float)(k.K[layer].ptr), (*C.float)(k.Kraw[layer].ptr), (*C.float)(k.V[layer].ptr), (*C.float)(kb.ptr), (*C.float)(rb.ptr), (*C.float)(vb.ptr), rocmDim(row, "kv append slot"), rocmDim(w, "kv row width")), "kv append")
	k.K[layer].rows++
	k.Kraw[layer].rows++
	k.V[layer].rows++
	if row == len(k.pos) {
		k.pos = append(k.pos, pos)
	}
}

func (k *rocmKV) view(rows *rocmRows) Tensor {
	k.ensureUsable()
	p := rows.ptr
	gen := atomic.LoadUint64(&rows.generation)
	n, ok := checkedROCmBytes([]int{rows.cap, k.stride()}, F32.Bytes())
	if !ok {
		panic(fmt.Errorf("rocm: KV view capacity overflow"))
	}
	return makeTensor(k.be, F32, RowMajor, []int{rows.rows, k.stride()}, nil, &rocmBuf{
		ptr: p, n: n, class: MemoryKVCache,
		alive: func() bool {
			return atomic.LoadUint32(&k.poisoned) == 0 && p != nil && atomic.LoadUint64(&rows.generation) == gen
		},
	})
}
func (k *rocmKV) KeysView(layer int) Tensor {
	k.ensureUsable()
	if layer < 0 || layer >= len(k.K) {
		panic(fmt.Errorf("rocm: invalid KV layer %d", layer))
	}
	return k.view(&k.K[layer])
}
func (k *rocmKV) ValuesView(layer int) Tensor {
	k.ensureUsable()
	if layer < 0 || layer >= len(k.V) {
		panic(fmt.Errorf("rocm: invalid KV layer %d", layer))
	}
	return k.view(&k.V[layer])
}

func (k *rocmKV) Evict(from, n int) int {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	k.ensureUsable()
	if from < 0 || n <= 0 || from >= len(k.pos) {
		return 0
	}
	if n > len(k.pos)-from {
		n = len(k.pos) - from
	}
	positions := len(k.pos)
	for l := range k.K {
		if k.K[l].rows != positions || k.Kraw[l].rows != positions || k.V[l].rows != positions {
			panic(fmt.Errorf("rocm: KV eviction requires complete, non-ragged layers"))
		}
		rocmCheck(C.frocm_kv_evict_f32((*C.float)(k.K[l].ptr), (*C.float)(k.Kraw[l].ptr), (*C.float)(k.V[l].ptr), rocmDim(positions, "kv positions"), rocmDim(from, "kv evict from"), rocmDim(n, "kv evict count"), rocmDim(k.cfg.NumKVHeads, "kv heads"), rocmDim(k.cfg.HeadDim, "kv head dim"), C.double(k.cfg.RopeTheta)), "kv evict")
		k.K[l].rows -= n
		k.Kraw[l].rows -= n
		k.V[l].rows -= n
	}
	k.pos = append(k.pos[:from], k.pos[from+n:]...)
	for i := range k.pos {
		k.pos[i] = i
	}
	return n
}

func (k *rocmKV) Clone() KVStore {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	return k.cloneWithOperations(k.be.alloc, func(dst, src unsafe.Pointer, bytes int, site string) {
		rocmCheck(C.frocm_d2d(dst, src, C.size_t(bytes)), site)
	}, func(ptr unsafe.Pointer) {
		rocmCheck(C.frocm_free(ptr), "clone KV rollback")
	})
}

// cloneWithOperations owns an unpublished clone. The production caller holds
// rocmMu and supplies the existing native operations; injected operations expose
// failure ownership without a global hook or a change to KV allocation policy.
func (k *rocmKV) cloneWithOperations(
	allocate func(int, MemoryClass, string) *rocmBuf,
	copyDevice func(unsafe.Pointer, unsafe.Pointer, int, string),
	release func(unsafe.Pointer),
) *rocmKV {
	k.ensureUsable()
	n := &rocmKV{be: k.be, cfg: cloneKVConfig(k.cfg), K: make([]rocmRows, len(k.K)), Kraw: make([]rocmRows, len(k.Kraw)), V: make([]rocmRows, len(k.V)), pos: append([]int(nil), k.pos...)}
	defer func() {
		if primary := recover(); primary != nil {
			// This clone has not escaped. Attempt every destination, including
			// the current failed copy, without releasing any source allocation.
			for l := len(n.K) - 1; l >= 0; l-- {
				for _, row := range []*rocmRows{&n.V[l], &n.Kraw[l], &n.K[l]} {
					if row.ptr == nil {
						continue
					}
					func() {
						defer func() { _ = recover() }()
						release(row.ptr)
					}()
				}
			}
			panic(primary)
		}
	}()

	copyRows := func(dst *rocmRows, src rocmRows, site string) {
		if src.rows == 0 {
			return
		}
		bytes, ok := checkedROCmBytes([]int{src.rows, k.stride()}, F32.Bytes())
		if !ok {
			panic(fmt.Errorf("rocm: %s size overflow", site))
		}
		b := allocate(bytes, MemoryKVCache, site)
		// Private clone owns this allocation before the copy can fail.
		dst.ptr = b.ptr
		copyDevice(b.ptr, src.ptr, bytes, site)
		dst.rows, dst.cap = src.rows, src.rows
	}
	for l := range k.K {
		copyRows(&n.K[l], k.K[l], "clone keys")
		copyRows(&n.Kraw[l], k.Kraw[l], "clone raw keys")
		copyRows(&n.V[l], k.V[l], "clone values")
	}
	return n
}

func (k *rocmKV) Free() {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	k.freeWithOperation(func(ptr unsafe.Pointer) {
		rocmCheck(C.frocm_free(ptr), "free KV")
	})
}

// freeWithOperation logically retires every known-owned row before its one
// native release attempt. Continue after failures, never retry an ambiguous
// pointer. An already-poisoned store suppresses secondary cleanup failures so
// deferred KV cleanup cannot replace its primary failure. Normal Free reports
// its first failure after attempting every row. Successful Free keeps the
// existing reusable-empty-store behavior. Physical reclamation is not promised.
// The production caller holds rocmMu.
func (k *rocmKV) freeWithOperation(release func(unsafe.Pointer)) {
	alreadyPoisoned := atomic.LoadUint32(&k.poisoned) != 0
	var first any
	for l := range k.K {
		for _, row := range []*rocmRows{&k.K[l], &k.Kraw[l], &k.V[l]} {
			ptr := row.ptr
			row.ptr, row.rows, row.cap = nil, 0, 0
			atomic.AddUint64(&row.generation, 1)
			if ptr != nil {
				func() {
					defer func() {
						if cause := recover(); cause != nil {
							if first == nil {
								first = cause
							}
							k.poison(cause)
						}
					}()
					release(ptr)
				}()
			}
		}
	}
	k.pos = nil
	if first != nil && !alreadyPoisoned {
		panic(first)
	}
}
