package radixkv

import "github.com/anthony-chaudhary/fak/internal/model"

// truncatePrefix returns a clone of c covering only its first L positions, bit-identical
// to a prefill that stopped at L. It uses model.KVCache.TryEvict of the tail span [L, Len):
// because no position is cached after the evicted tail, Evict renumbers and re-RoPEs no
// survivor, so the result is an exact prefix. Recurrent caches that cannot be truncated
// return nil so the split remains a structural boundary and callers recompute.
func truncatePrefix(c *model.KVCache, L int) *model.KVCache {
	cp := c.Clone()
	if L < cp.Len() {
		if _, err := cp.TryEvict(L, cp.Len()-L); err != nil {
			return nil
		}
	}
	return cp
}
