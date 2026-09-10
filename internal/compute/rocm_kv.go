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
		rocmCheck(C.frocm_d2d(nb.ptr, row.ptr, C.size_t(used)), site+" copy")
	}
	if row.ptr != nil {
		rocmCheck(C.frocm_free(row.ptr), site+" old free")
	}
	row.ptr, row.cap = nb.ptr, ncap
	atomic.AddUint64(&row.generation, 1)
}

func (k *rocmKV) AppendKV(layer int, raw, rope, value Tensor, pos int) {
	rocmMu.Lock()
	defer rocmMu.Unlock()
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
	p := rows.ptr
	gen := atomic.LoadUint64(&rows.generation)
	n, ok := checkedROCmBytes([]int{rows.cap, k.stride()}, F32.Bytes())
	if !ok {
		panic(fmt.Errorf("rocm: KV view capacity overflow"))
	}
	return makeTensor(k.be, F32, RowMajor, []int{rows.rows, k.stride()}, nil, &rocmBuf{
		ptr: p, n: n, class: MemoryKVCache,
		alive: func() bool { return p != nil && atomic.LoadUint64(&rows.generation) == gen },
	})
}
func (k *rocmKV) KeysView(layer int) Tensor {
	if layer < 0 || layer >= len(k.K) {
		panic(fmt.Errorf("rocm: invalid KV layer %d", layer))
	}
	return k.view(&k.K[layer])
}
func (k *rocmKV) ValuesView(layer int) Tensor {
	if layer < 0 || layer >= len(k.V) {
		panic(fmt.Errorf("rocm: invalid KV layer %d", layer))
	}
	return k.view(&k.V[layer])
}

func (k *rocmKV) Evict(from, n int) int {
	rocmMu.Lock()
	defer rocmMu.Unlock()
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
	n := &rocmKV{be: k.be, cfg: cloneKVConfig(k.cfg), K: make([]rocmRows, len(k.K)), Kraw: make([]rocmRows, len(k.Kraw)), V: make([]rocmRows, len(k.V)), pos: append([]int(nil), k.pos...)}
	copyRows := func(dst *rocmRows, src rocmRows, site string) {
		if src.rows == 0 {
			return
		}
		bytes, ok := checkedROCmBytes([]int{src.rows, k.stride()}, F32.Bytes())
		if !ok {
			panic(fmt.Errorf("rocm: %s size overflow", site))
		}
		b := k.be.alloc(bytes, MemoryKVCache, site)
		rocmCheck(C.frocm_d2d(b.ptr, src.ptr, C.size_t(bytes)), site)
		dst.ptr, dst.rows, dst.cap = b.ptr, src.rows, src.rows
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
	for l := range k.K {
		for _, row := range []*rocmRows{&k.K[l], &k.Kraw[l], &k.V[l]} {
			if row.ptr != nil {
				rocmCheck(C.frocm_free(row.ptr), "free KV")
			}
			row.ptr, row.rows, row.cap = nil, 0, 0
			atomic.AddUint64(&row.generation, 1)
		}
	}
	k.pos = nil
}
