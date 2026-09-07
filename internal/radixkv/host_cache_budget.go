package radixkv

import (
	"github.com/anthony-chaudhary/fak/internal/model"
	"math"
)

// SetCPUCacheByteBudget bounds retained CPU KV numeric backing storage and
// cached logits. Zero preserves unlimited retention. Configure before use.
// Session copies, Go headers and allocator/RSS overhead are separate budgets.
func (t *Tree) SetCPUCacheByteBudget(bytes int64) {
	if bytes < 0 {
		panic("radixkv: negative CPU cache byte budget")
	}
	t.maxCPUCacheBytes = bytes
}

// CPUCacheByteBudget reports the configured immutable retention bound.
func (t *Tree) CPUCacheByteBudget() int64 { return t.maxCPUCacheBytes }

func (t *Tree) cpuCacheCanClone(kv *model.KVCache) bool {
	return t.maxCPUCacheBytes == 0 || t.cpuCacheRoomWithoutEviction(kv.ClonePayloadBytes())
}

func (t *Tree) cpuCacheRoomWithoutEviction(incoming int64) bool {
	if t.maxCPUCacheBytes == 0 {
		return true
	}
	if incoming == math.MaxInt64 || incoming > t.maxCPUCacheBytes || t.cpuCacheBytes() > t.maxCPUCacheBytes-incoming {
		t.cpuCacheBypasses++
		t.cpuCacheLastBypass = "capacity"
		return false
	}
	return true
}

func cacheByteSum(a, b int64) int64 {
	if a < 0 || b < 0 || a > math.MaxInt64-b {
		return math.MaxInt64
	}
	return a + b
}
func cpuLogitsBytes(n int) int64 {
	if int64(n) > math.MaxInt64/4 {
		return math.MaxInt64
	}
	return int64(n) * 4
}
func cpuNodeBytes(n *node) int64 {
	return cacheByteSum(n.kv.OwnedPayloadBytes(), cpuLogitsBytes(cap(n.logits)))
}
func (t *Tree) cpuCacheBytes() int64 {
	var total int64
	var walk func(*node)
	walk = func(n *node) {
		total = cacheByteSum(total, cpuNodeBytes(n))
		for _, c := range n.children {
			walk(c)
		}
	}
	t.forEachRoot(walk)
	return total
}
func copyCPULogits(src []float32) []float32 {
	if src == nil {
		return nil
	}
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}

// Exclusion and live descendant leases protect the entire reusable prefix path.
// Dropping a payload keeps structural nodes valid for outstanding boundaries.
func (t *Tree) cpuCacheVictim(exclude *node) *node {
	var victim *node
	var walk func(*node) bool
	walk = func(n *node) bool {
		pinned := n == exclude || n.refs > 0
		for _, c := range n.children {
			if walk(c) {
				pinned = true
			}
		}
		if !pinned && cpuNodeBytes(n) > 0 && (victim == nil || n.lastUsed < victim.lastUsed) {
			victim = n
		}
		return pinned
	}
	t.forEachRoot(func(n *node) { walk(n) })
	return victim
}
func (t *Tree) makeCPUCacheRoom(incoming, replaced int64, exclude *node) bool {
	if t.maxCPUCacheBytes == 0 {
		return true
	}
	reject := func(reason string) bool { t.cpuCacheBypasses++; t.cpuCacheLastBypass = reason; return false }
	if incoming == math.MaxInt64 || incoming > t.maxCPUCacheBytes {
		return reject("oversized")
	}
	for {
		held := t.cpuCacheBytes()
		if replaced <= held {
			held -= replaced
		}
		if held <= t.maxCPUCacheBytes-incoming {
			return true
		}
		victim := t.cpuCacheVictim(exclude)
		if victim == nil {
			return reject("pinned")
		}
		victim.kv = nil
		victim.logits = nil
	}
}

// InsertCloneWithLogits checks retained storage before cloning the session.
// A rejected cache fill keeps the boundary lease and leaves generation intact.
func (t *Tree) InsertCloneWithLogits(boundary *node, suffix []int, kv *model.KVCache, logits []float32) *node {
	if boundary == nil {
		return nil
	}
	if len(suffix) == 0 && boundary.kv != nil {
		return t.InsertWithLogits(boundary, suffix, nil, logits)
	}
	if t.maxCPUCacheBytes > 0 && !t.makeCPUCacheRoom(cacheByteSum(kv.ClonePayloadBytes(), cpuLogitsBytes(len(logits))), 0, boundary) {
		return boundary
	}
	var cp *model.KVCache
	if kv != nil {
		cp = kv.Clone()
	}
	if len(suffix) == 0 {
		boundary.kv = cp
		if cp != nil {
			boundary.logits = copyCPULogits(logits)
		}
		return boundary
	}
	return t.InsertWithLogits(boundary, suffix, cp, logits)
}
