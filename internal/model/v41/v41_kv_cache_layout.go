package v41

import "github.com/anthony-chaudhary/fak/internal/v41kv"

// Compatibility names for the neutral, byte-preserving V4.1 codec.
// This does not select a backend or activate packed attention.
type V41KVQuantFormat = v41kv.V41KVQuantFormat
type V41KVScaleFormat = v41kv.V41KVScaleFormat
type V41KVCacheLayout = v41kv.V41KVCacheLayout
type V41KVCacheRow = v41kv.V41KVCacheRow

const (
	V41KVUnspecified      = v41kv.V41KVUnspecified
	V41KVFP8              = v41kv.V41KVFP8
	V41KVFP4              = v41kv.V41KVFP4
	V41KVScaleUnspecified = v41kv.V41KVScaleUnspecified
	V41KVScaleE8M0        = v41kv.V41KVScaleE8M0
	V41KVScaleE4M3        = v41kv.V41KVScaleE4M3
)

var ErrV41KVCacheLayout = v41kv.ErrV41KVCacheLayout

func V41KVCacheLayoutFor(format V41KVQuantFormat) (V41KVCacheLayout, bool) {
	return v41kv.V41KVCacheLayoutFor(format)
}
func V41KVCacheLayouts() []V41KVCacheLayout { return v41kv.V41KVCacheLayouts() }
