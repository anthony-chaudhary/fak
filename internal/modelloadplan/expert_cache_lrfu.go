package modelloadplan

// ExpertCacheLRFU adapts the ClassLRU policy in ggml-org/llama.cpp,
// src/llama-moe-cache.cpp:17-228 at c4fb579695320929b6a453f4f3a56951bda332ee
// (MIT). This is a Go policy implementation, not a
// loader/device integration. Eight frequency classes retain the CRF across
// eviction; each class has an LRU list. Among its oldest unselected candidates,
// eviction chooses the largest age/2^class, not the globally lowest raw CRF.

/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

import (
	"errors"
	"math"
	"sort"
	"sync"
)

const ExpertCacheLRFUMaxEntries = 1 << 20
const expertCacheLRFUClasses = 8
const expertCacheLRFUHalfLife = float32(128)

var (
	ErrExpertCacheLRFUConfig   = errors.New("modelloadplan: invalid expert cache geometry")
	ErrExpertCacheLRFUInput    = errors.New("modelloadplan: invalid expert cache access")
	ErrExpertCacheLRFUCapacity = errors.New("modelloadplan: selected experts exceed resident slots")
	ErrExpertCacheLRFUClock    = errors.New("modelloadplan: expert cache token clock overflow")
)

type ExpertCacheKey struct{ Layer, Expert int }
type ExpertCacheLRFUFill struct {
	Key     ExpertCacheKey
	Slot    int
	Evicted *ExpertCacheKey // nil means an unused slot; always an owned copy
}
type ExpertCacheLRFUPlan struct {
	Hits                    int
	ClockBefore, ClockAfter int64
	Fills                   []ExpertCacheLRFUFill
}
type ExpertCacheLRFUEntry struct {
	Slot, Class        int // -1 when nonresident
	CRF, DecayedCRF    float32
	LastUse, LayerTime int64 // LastUse is -1 before first access
}

// ExpertCacheLRFU owns a bounded logical resident-slot policy and per-layer token
// clocks. Plan advances simulated residency, not actual weight ownership. A
// future loader must separately handle fill failure, request pinning and device
// synchronization; a returned slot is not evidence that weights are loaded.
// Methods serialize access. Do not copy this value after use.
type ExpertCacheLRFU struct {
	mu              sync.Mutex
	layers, experts int
	keys, slots     []int
	prev, next      []int
	classes         []int
	head, tail      [expertCacheLRFUClasses]int
	crf             []float32
	lastUse, clock  []int64
}

func NewExpertCacheLRFU(layers, experts, residentSlots int) (*ExpertCacheLRFU, error) {
	if layers <= 0 || experts <= 0 || residentSlots <= 0 ||
		experts > ExpertCacheLRFUMaxEntries || layers > ExpertCacheLRFUMaxEntries/experts ||
		residentSlots > layers*experts {
		return nil, ErrExpertCacheLRFUConfig
	}
	entries := layers * experts
	c := &ExpertCacheLRFU{layers: layers, experts: experts, keys: make([]int, residentSlots),
		slots: make([]int, entries), prev: make([]int, residentSlots), next: make([]int, residentSlots),
		classes: make([]int, residentSlots), crf: make([]float32, entries), lastUse: make([]int64, entries), clock: make([]int64, layers)}
	for i := range c.slots {
		c.slots[i], c.lastUse[i] = -1, -1
	}
	for i := range c.head {
		c.head[i], c.tail[i] = -1, -1
	}
	for i := range c.keys {
		c.keys[i], c.prev[i], c.next[i] = -1, i-1, i+1
	}
	c.next[residentSlots-1] = -1
	c.head[0], c.tail[0] = 0, residentSlots-1
	return c, nil
}

func (c *ExpertCacheLRFU) decayed(key int) float32 {
	if c.lastUse[key] < 0 {
		return 0
	}
	age := c.clock[key/c.experts] - c.lastUse[key]
	decay := float32(math.Exp2(float64(-float32(age) / expertCacheLRFUHalfLife)))
	return float32(c.crf[key] * decay)
}

func expertCacheLRFUClass(crf float32) int {
	// The donor's std::log2(float) returns float32 before integer truncation.
	// Preserve that precision: near a power of two it can round up to the
	// next bucket even though truncating the double result would not.
	class := int(float32(math.Log2(float64(crf)))) // post-use CRF is finite and >= 1
	if class >= expertCacheLRFUClasses {
		return expertCacheLRFUClasses - 1
	}
	return class
}

func (c *ExpertCacheLRFU) move(slot, class int) {
	from := c.classes[slot]
	if from == class && c.tail[from] == slot {
		return
	}
	if c.prev[slot] >= 0 {
		c.next[c.prev[slot]] = c.next[slot]
	} else {
		c.head[from] = c.next[slot]
	}
	if c.next[slot] >= 0 {
		c.prev[c.next[slot]] = c.prev[slot]
	} else {
		c.tail[from] = c.prev[slot]
	}
	c.prev[slot], c.next[slot] = c.tail[class], -1
	if c.tail[class] >= 0 {
		c.next[c.tail[class]] = slot
	} else {
		c.head[class] = slot
	}
	c.tail[class], c.classes[slot] = slot, class
}

func (c *ExpertCacheLRFU) victim(layer int, selected map[int]bool) int {
	victim, best := -1, float64(-1)
	for class := range c.head {
		for slot := c.head[class]; slot >= 0; slot = c.next[slot] {
			key := c.keys[slot]
			if key >= 0 && key/c.experts == layer && selected[key%c.experts] {
				continue
			}
			if key < 0 {
				return slot
			}
			age := c.clock[key/c.experts] - c.lastUse[key]
			score := math.Ldexp(float64(age), -class)
			if score > best {
				victim, best = slot, score
			}
			break // only the oldest unselected member of each class competes
		}
	}
	return victim
}

// Plan accounts once per distinct selected expert, promotes all hits before
// choosing miss victims, fills misses in expert order, and then advances only
// this layer's token clock. Selected experts (including fills earlier in this
// call) cannot be victims. Invalid/capacity/clock refusals do not change state.
// No wall clock, I/O, weights, backend or environment variables are used.
func (c *ExpertCacheLRFU) Plan(layer int, ids []int, tokens int64) (ExpertCacheLRFUPlan, error) {
	if c == nil {
		return ExpertCacheLRFUPlan{}, ErrExpertCacheLRFUInput
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if layer < 0 || layer >= c.layers || tokens < 0 {
		return ExpertCacheLRFUPlan{}, ErrExpertCacheLRFUInput
	}
	if tokens > math.MaxInt64-c.clock[layer] {
		return ExpertCacheLRFUPlan{}, ErrExpertCacheLRFUClock
	}
	selected := make(map[int]bool)
	uniq := make([]int, 0, min(len(ids), len(c.keys)))
	for _, id := range ids {
		if id < 0 || id >= c.experts {
			return ExpertCacheLRFUPlan{}, ErrExpertCacheLRFUInput
		}
		if !selected[id] {
			selected[id] = true
			uniq = append(uniq, id)
			if len(uniq) > len(c.keys) {
				return ExpertCacheLRFUPlan{}, ErrExpertCacheLRFUCapacity
			}
		}
	}
	result := ExpertCacheLRFUPlan{ClockBefore: c.clock[layer], ClockAfter: c.clock[layer] + tokens}
	for _, id := range uniq {
		key := layer*c.experts + id
		c.crf[key] = c.decayed(key) + 1
		c.lastUse[key] = c.clock[layer]
		if slot := c.slots[key]; slot >= 0 {
			c.move(slot, expertCacheLRFUClass(c.crf[key]))
			result.Hits++
		}
	}
	sort.Ints(uniq)
	for _, id := range uniq {
		key := layer*c.experts + id
		if c.slots[key] >= 0 {
			continue
		}
		// At most residentSlots distinct experts are protected, and this key
		// is not resident yet, so at least one unselected/empty slot exists.
		slot := c.victim(layer, selected)
		fill := ExpertCacheLRFUFill{Key: ExpertCacheKey{layer, id}, Slot: slot}
		if old := c.keys[slot]; old >= 0 {
			fill.Evicted = &ExpertCacheKey{old / c.experts, old % c.experts}
			c.slots[old] = -1
		}
		c.keys[slot], c.slots[key] = key, slot
		c.move(slot, expertCacheLRFUClass(c.crf[key]))
		result.Fills = append(result.Fills, fill)
	}
	c.clock[layer] += tokens
	return result, nil
}

// Inspect returns owned scalar policy observations. Class is the resident class
// as of last use, not a class recomputed as the layer clock ages. CRF persists
// even when Slot is -1. Exp2 is the Go host implementation; this API makes no
// cross-language/libm bit-identity or physical cache-performance promise.
func (c *ExpertCacheLRFU) Inspect(layer, expert int) (ExpertCacheLRFUEntry, error) {
	if c == nil {
		return ExpertCacheLRFUEntry{}, ErrExpertCacheLRFUInput
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if layer < 0 || layer >= c.layers || expert < 0 || expert >= c.experts {
		return ExpertCacheLRFUEntry{}, ErrExpertCacheLRFUInput
	}
	key := layer*c.experts + expert
	slot, class := c.slots[key], -1
	if slot >= 0 {
		class = c.classes[slot]
	}
	return ExpertCacheLRFUEntry{Slot: slot, Class: class, CRF: c.crf[key], DecayedCRF: c.decayed(key), LastUse: c.lastUse[key], LayerTime: c.clock[layer]}, nil
}
