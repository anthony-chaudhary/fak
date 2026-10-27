package radixkv

import "github.com/anthony-chaudhary/fak/internal/model"

// truncatePrefix returns an independent exact prefix. Ordinary dense caches copy
// only retained rows instead of first cloning the discarded tail. The model
// helper retains Clone/TryEvict semantics for specialized or irregular state;
// recurrent caches that cannot truncate still return nil and callers recompute.
func truncatePrefix(c *model.KVCache, L int) *model.KVCache {
	return c.ClonePrefixExact(L)
}
