package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"math"
)

// v41RoPETableCache owns at most two exact table extents through Session.halW.
// The caller chooses one or two stable, operation-specific keys. A replacement
// stays caller-owned until validation and retirement of the prior slot finish.
// Unknown native completion is quarantined by Backend.Free, never recycled here.
type v41RoPETableCache struct {
	owner   *Session
	backend compute.Backend
	keys    []string
	entries [2]v41RoPETableEntry
	next    int
}
type v41RoPETableEntry struct {
	values []float32
	tensor compute.Tensor
	model  *Model
	epoch  uint64
}

func newV41RoPETableCache(s *Session, b compute.Backend, keys ...string) *v41RoPETableCache {
	if len(keys) < 1 || len(keys) > 2 {
		panic("invalid rotary table cache capacity")
	}
	return &v41RoPETableCache{owner: s, backend: b, keys: append([]string(nil), keys...)}
}
func (c *v41RoPETableCache) hostTable(cos, sin []float32) []float32 {
	return v41TailRoPEHostTable(cos, sin, [2][]float32{c.entries[0].values, c.entries[1].values})
}
func (c *v41RoPETableCache) owns(t compute.Tensor) bool {
	for _, key := range v41RoPETableKeys {
		if table, present := c.owner.halW[key]; present && t.Buf() == table.Buf() {
			return true
		}
	}
	return false
}
func (c *v41RoPETableCache) lookup(table []float32, dim int) compute.Tensor {
	for i, key := range c.keys {
		entry := c.entries[i]
		live, present := c.owner.halW[key]
		if !present || live.Buf() == nil || live.Buf() != entry.tensor.Buf() || live.Backend() != c.backend || c.owner.Backend != c.backend || entry.model != c.owner.M || entry.epoch != c.owner.cacheGeometryEpoch || live.Dtype != compute.F32 || live.Layout != compute.RowMajor || live.Quant != nil || len(live.Shape) != 2 || live.Shape[0] != dim/2 || live.Shape[1] != 2 || len(entry.values) != len(table) {
			continue
		}
		equal := true
		for j, v := range table {
			if math.Float32bits(v) != math.Float32bits(entry.values[j]) {
				equal = false
				break
			}
		}
		if equal {
			return live
		}
	}
	return compute.Tensor{}
}

// install transfers only a validated fresh table; the caller retains its cleanup
// owner until this returns. Detach the old slot before Free so a panic cannot
// cause either old storage to be retried or fresh storage to lose its owner.
func (c *v41RoPETableCache) install(t compute.Tensor, values []float32) {
	if c.owner.halW == nil {
		c.owner.halW = make(map[string]compute.Tensor)
	}
	key := c.keys[c.next]
	if prior, present := c.owner.halW[key]; present {
		delete(c.owner.halW, key)
		c.backend.Free(prior)
	}
	c.owner.halW[key] = t
	c.entries[c.next] = v41RoPETableEntry{values: values, tensor: t, model: c.owner.M, epoch: c.owner.cacheGeometryEpoch}
	c.next = (c.next + 1) % len(c.keys)
}

// The fixed inventory covers both consumers, including callback rebinding.
var v41RoPETableKeys = [...]string{"v41-indexer-query-rope-table:0", "v41-compressed-publication-rope-table:0", "v41-compressed-publication-rope-table:1"}

// retireV41RoPETables follows the completion-aware routed-output release
// protocol: detach every owner first, schedule every attempt before invoking
// Free, and preserve the first cleanup panic including legacy panic(nil).
// Free returning is not a claim of physical retirement: native deferred release
// and uncertain-submission quarantine remain backend-owned.
func (s *Session) retireV41RoPETables() (failure backendCleanupPanic) {
	var owned [len(v41RoPETableKeys)]compute.Tensor
	n := 0
	for _, key := range v41RoPETableKeys {
		if t, present := s.halW[key]; present {
			delete(s.halW, key)
			owned[n] = t
			n++
		}
	}
	completed := false
	defer func() {
		if !completed {
			s.halClosed = true
		}
	}()
	func() {
		// Reverse registration preserves inventory order; Goexit still runs all
		// pending defers and never returns an apparent successful outcome.
		for i := n - 1; i >= 0; i-- {
			defer func(t compute.Tensor) {
				returned := false
				defer func() {
					p := recover()
					if !returned && !failure.present {
						failure.value, failure.present = p, true
					}
				}()
				t.Backend().Free(t)
				returned = true
			}(owned[i])
		}
	}()
	completed = true
	return failure
}
