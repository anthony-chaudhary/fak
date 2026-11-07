//go:build cuda && cgo

package compute

import (
	"sort"
	"unsafe"
)

// Qwen35IndependentDecodeCohortOwned observes the complete offered cohort before
// its caller splits it into bounded submissions. It neither retains nor transfers
// ownership: the caller must serialize session mutation until submissions finish.
// Only concrete contiguous owners with disjoint recorded allocation extents are
// admitted. Paged, shared, graph-owned and otherwise unknown storage declines.
// This is metadata evidence, not a device-completion or numerical witness.
func (c *cudaBackend) Qwen35IndependentDecodeCohortOwned(stores []KVStore, states [][]Qwen35SequenceState) bool {
	if c == nil || len(stores) < 2 || len(stores) != len(states) {
		return false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if c.capturing || c.graphOnce != nil || c.faultLatch.Health() != DeviceHealthy {
		return false
	}
	var ranges, metadata []cudaQwen35CohortRange
	owners := make(map[*cudaKV]bool, len(stores))
	layers := len(states[0])
	if layers == 0 {
		return false
	}
	for lane, raw := range stores {
		k, ok := raw.(*cudaKV)
		if !ok || k == nil || owners[k] || k.be != c || k.releaseRetained || k.releaseNext != nil || k.snapshotCopy != nil ||
			k.cfg.NumLayers <= 0 || len(k.K) != k.cfg.NumLayers || len(k.Kraw) != k.cfg.NumLayers || len(k.V) != k.cfg.NumLayers ||
			k.maxPos < 0 || (k.maxPos > 0 && len(k.pos) > k.maxPos) || len(states[lane]) != layers {
			return false
		}
		owners[k] = true
		// Empty device stores can still share mutable host slice backing. A
		// subgroup's reserve/append would publish pointers or positions into a
		// later subgroup through that backing. Check the complete capacities in
		// a separate address domain from device storage, including spare slots.
		for _, storage := range [][]dslice{k.K, k.Kraw, k.V} {
			bytes, ok := cudaQwen35CohortProduct(cap(storage), int(unsafe.Sizeof(dslice{})))
			if !ok {
				return false
			}
			r, ok := cudaQwen35CohortExtent(unsafe.Pointer(unsafe.SliceData(storage)), bytes, int(unsafe.Alignof(dslice{})))
			if !ok {
				return false
			}
			metadata = append(metadata, r)
		}
		if cap(k.pos) > 0 {
			bytes, ok := cudaQwen35CohortProduct(cap(k.pos), int(unsafe.Sizeof(int(0))))
			if !ok {
				return false
			}
			r, ok := cudaQwen35CohortExtent(unsafe.Pointer(unsafe.SliceData(k.pos)), bytes, int(unsafe.Alignof(int(0))))
			if !ok {
				return false
			}
			metadata = append(metadata, r)
		}
		stride, ok := cudaQwen35CohortProduct(k.cfg.NumKVHeads, k.cfg.HeadDim)
		if !ok {
			return false
		}
		for i, pos := range k.pos {
			if pos != i {
				return false
			}
		}
		populated := false
		for layer, d := range k.K {
			if d.len != k.Kraw[layer].len || d.len != k.V[layer].len ||
				(d.len != 0 && d.len/stride != len(k.pos)) {
				return false
			}
			populated = populated || d.len > 0
		}
		if len(k.pos) > 0 && !populated {
			return false
		}
		dtype := F32
		switch k.cfg.Precision {
		case KVPrecisionF32:
		case KVPrecisionF16:
			dtype = F16
		case KVPrecisionBF16:
			dtype = BF16
		default:
			return false
		}
		for plane, storage := range [][]dslice{k.K, k.Kraw, k.V} {
			want := dtype
			if plane == 1 {
				want = F32
			}
			for _, d := range storage {
				if d.dtype != want || d.len < 0 || d.cap < d.len || d.len%stride != 0 {
					return false
				}
				if d.cap == 0 {
					if d.ptr != nil {
						return false
					}
					continue
				}
				bytes, ok := cudaQwen35CohortProduct(d.cap, want.Bytes())
				if !ok {
					return false
				}
				r, ok := cudaQwen35CohortExtent(d.ptr, bytes, want.Bytes())
				if !ok {
					return false
				}
				ranges = append(ranges, r)
			}
		}
		linear := 0
		for layer, state := range states[lane] {
			empty := cudaQwen35CohortEmpty(state.Conv) && cudaQwen35CohortEmpty(state.Recurrent)
			first := states[0][layer]
			if empty != (cudaQwen35CohortEmpty(first.Conv) && cudaQwen35CohortEmpty(first.Recurrent)) {
				return false
			}
			if empty {
				continue
			}
			linear++
			for _, t := range []Tensor{state.Conv, state.Recurrent} {
				if t.Backend() != c || t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || len(t.Shape) == 0 {
					return false
				}
				n := 4
				for _, dim := range t.Shape {
					var ok bool
					n, ok = cudaQwen35CohortProduct(n, dim)
					if !ok {
						return false
					}
				}
				b, ok := t.buf.(*cudaBuf)
				if !ok || b == nil || b.n < n || b.class != MemoryKVCache || b.device != 0 || b.invalidForSubmit() ||
					(b.be != nil && b.be != c) || b.managed || b.graphOncePinned || b.host != 0 || b.hostKeep != nil ||
					b.scales != nil || b.scalesN != 0 || b.managedWeight || b.budgetedWeightBytes != 0 {
					return false
				}
				r, ok := cudaQwen35CohortExtent(b.ptr, b.n, 4)
				if !ok {
					return false
				}
				ranges = append(ranges, r)
			}
		}
		if linear == 0 {
			return false
		}
	}
	// Use capacity, never active tensor shape or KV length, for alias checks.
	// Interior wrappers and opposite-side subgroup aliases are therefore covered.
	if !cudaQwen35CohortDisjoint(metadata) || !cudaQwen35CohortDisjoint(ranges) {
		return false
	}
	for _, r := range ranges {
		for _, transient := range c.transient {
			if transient == nil || transient.ptr == nil {
				continue
			}
			other, ok := cudaQwen35CohortExtent(transient.ptr, transient.n, 1)
			if !ok || (r.lo < other.hi && other.lo < r.hi) {
				return false
			}
		}
	}
	return true
}

type cudaQwen35CohortRange struct{ lo, hi uintptr }

func cudaQwen35CohortDisjoint(ranges []cudaQwen35CohortRange) bool {
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].lo < ranges[j].lo })
	for i := 1; i < len(ranges); i++ {
		if ranges[i].lo < ranges[i-1].hi {
			return false
		}
	}
	return true
}

func cudaQwen35CohortExtent(ptr unsafe.Pointer, bytes, alignment int) (cudaQwen35CohortRange, bool) {
	lo := uintptr(ptr)
	if lo == 0 || bytes <= 0 || alignment <= 0 || lo%uintptr(alignment) != 0 || bytes%alignment != 0 || uintptr(bytes) > ^uintptr(0)-lo {
		return cudaQwen35CohortRange{}, false
	}
	return cudaQwen35CohortRange{lo, lo + uintptr(bytes)}, true
}

func cudaQwen35CohortProduct(a, b int) (int, bool) {
	if a <= 0 || b <= 0 || a > int(^uint(0)>>1)/b {
		return 0, false
	}
	return a * b, true
}

func cudaQwen35CohortEmpty(t Tensor) bool {
	return t.buf == nil && t.Backend() == nil && len(t.Shape) == 0 && t.Quant == nil && t.Dtype == F32 && t.Layout == RowMajor
}
