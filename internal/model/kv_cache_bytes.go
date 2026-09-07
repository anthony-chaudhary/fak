package model

import (
	"math"
	"strconv"
)

// OwnedPayloadBytes counts backing capacities of KV numeric arrays, including
// recurrent state and token lineage. It excludes Go headers, allocator rounding,
// and external session/device buffers; it is not a process RSS estimate.
func (c *KVCache) OwnedPayloadBytes() int64 { return c.cachePayloadBytes(false) }

// ClonePayloadBytes counts numeric backing storage allocated by Clone, before
// allocating it. Clone copies live elements without the source's spare capacity.
func (c *KVCache) ClonePayloadBytes() int64 { return c.cachePayloadBytes(true) }

func (c *KVCache) cachePayloadBytes(clone bool) int64 {
	if c == nil {
		return 0
	}
	var total int64
	add := func(n, width int) {
		if n < 0 || int64(n) > (math.MaxInt64-total)/int64(width) {
			total = math.MaxInt64
			return
		}
		total += int64(n) * int64(width)
	}
	count := func(length, capacity int) int {
		if clone {
			return length
		}
		return capacity
	}
	f32 := func(rows [][]float32) {
		for _, row := range rows {
			add(count(len(row), cap(row)), 4)
		}
	}
	f64 := func(rows [][]float64) {
		for _, row := range rows {
			add(count(len(row), cap(row)), 8)
		}
	}
	f32(c.K)
	f32(c.Kraw)
	f32(c.V)
	add(count(len(c.pos), cap(c.pos)), strconv.IntSize/8)
	add(count(len(c.lineage.ids), cap(c.lineage.ids)), 4)
	if c.linear != nil {
		for _, state := range c.linear.layers {
			f32(state.conv)
			f32(state.recurrent)
		}
	}
	if c.glm != nil {
		f32(c.glm.K)
		f32(c.glm.Kraw)
		f32(c.glm.V)
		f64(c.glm.IndexK)
		f64(c.glm.IndexKraw)
	}
	if c.msa != nil {
		f32(c.msa.IndexK)
		f32(c.msa.IndexKraw)
	}
	return total
}
