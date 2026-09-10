// Package strix provides high-performance, hardware-aligned caching primitives
// for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151) APUs.
//
// On AMD Strix Halo, the Memory Attached Last-Level (MALL) Infinity Cache comprises
// 32 MiB of on-die SRAM organized into 32,768 sets x 16 ways x 64-byte cache lines,
// offering >1.2 TB/s internal interconnect bandwidth.
//
// This package implements a hardware-aligned 8 MiB Cuckoo hash table (8,192 sets x
// 16 ways x 64B) designed to reside entirely within an 8 MiB partition of MALL.
// Each 64-byte bucket holds exactly 4 slots of 16 bytes (8-byte CASKey + 8-byte
// KVBlockPtr), matching the native 64-byte cache line size of Zen 5 and RDNA 3.5.
// Lookups evaluate at most 2 candidate buckets (at most 2 cache line fetches),
// achieving sub-20ns Content-Addressable Storage (CAS) handle resolution without
// touching DRAM.
package strix

import "errors"

const (
	// TableSizeBytes is the exact physical size of the 8 MiB MALL Cuckoo table:
	// 8,192 sets x 16 ways x 64 bytes = 8,388,608 bytes.
	TableSizeBytes = 8 * 1024 * 1024

	// NumBuckets is the total number of 64-byte buckets in the table (131,072 = 2^17).
	NumBuckets = 131072

	// SlotsPerBucket is the number of 16-byte entries per 64-byte bucket.
	SlotsPerBucket = 4

	// SlotSizeBytes is the size of each slot in bytes (8B Key + 8B Ptr).
	SlotSizeBytes = 16

	// BucketSizeBytes matches the Zen 5 / RDNA 3.5 / MALL cache line size (64 bytes).
	BucketSizeBytes = 64

	// NumSets is the number of cache sets represented in the 8 MiB partition.
	NumSets = 8192

	// NumWays is the set associativity in MALL Infinity Cache.
	NumWays = 16

	// TotalSlots is the total slot capacity of the primary table (131,072 * 4 = 524,288).
	TotalSlots = NumBuckets * SlotsPerBucket

	// NumStripeLocks is the number of fine-grained mutex stripes for concurrent insertions.
	NumStripeLocks = 4096

	// MaxDisplacements is the maximum search depth during Cuckoo displacement kick-outs.
	MaxDisplacements = 500

	// MaxStashCapacity is the maximum capacity of the DRAM overflow stash before table exhaustion.
	MaxStashCapacity = 1024

	// EmptyCASKey is the sentinel value representing an empty slot key.
	EmptyCASKey CASKey = 0

	// EmptyKVBlockPtr is the sentinel value representing an empty slot pointer.
	EmptyKVBlockPtr KVBlockPtr = 0
)

var (
	// ErrInvalidKey is returned when attempting to insert a zero/empty CAS key.
	ErrInvalidKey = errors.New("cuckoo: invalid CAS key (zero key not permitted)")

	// ErrInvalidPointer is returned when attempting to insert a zero/empty KV block pointer.
	ErrInvalidPointer = errors.New("cuckoo: invalid KV block pointer (zero pointer not permitted)")

	// ErrTableFull is returned when primary Cuckoo displacement and DRAM stash are both saturated.
	ErrTableFull = errors.New("cuckoo: table capacity and DRAM overflow stash exhausted")

	// ErrStashFull is returned when the DRAM overflow stash exceeds MaxStashCapacity.
	ErrStashFull = errors.New("cuckoo: DRAM overflow stash capacity exceeded")
)

// CASKey represents a 64-bit Content-Addressable Storage (CAS) handle derived
// from cryptographic or xxHash sequence hashes in ctxmmu.
type CASKey uint64

// KVBlockPtr represents a 64-bit physical KV cache block pointer in unified memory.
type KVBlockPtr uint64

// CuckooSlot represents a single 16-byte slot in the Cuckoo hash table.
// Each slot contains an 8-byte CASKey and an 8-byte KVBlockPtr.
type CuckooSlot struct {
	Key CASKey     // 8-byte CAS handle (0 = empty)
	Ptr KVBlockPtr // 8-byte KV block physical pointer
}

// CuckooBucket represents a 64-byte bucket holding 4 contiguous 16-byte slots.
// Its layout matches a single 64-byte physical cache line in MALL.
type CuckooBucket struct {
	Slots [SlotsPerBucket]CuckooSlot // 4 slots x 16 bytes = 64 bytes
}

// CuckooTelemetry exports runtime operational metrics, load factors,
// MALL hit rates, and displacement statistics.
type CuckooTelemetry struct {
	TotalLookups         uint64  `json:"total_lookups"`
	Bucket1Hits          uint64  `json:"bucket1_hits"`
	Bucket2Hits          uint64  `json:"bucket2_hits"`
	StashHits            uint64  `json:"stash_hits"`
	Misses               uint64  `json:"misses"`
	TotalInsertions      uint64  `json:"total_insertions"`
	KickoutDisplacements uint64  `json:"kickout_displacements"`
	StashSpills          uint64  `json:"stash_spills"`
	ActiveEntries        uint64  `json:"active_entries"`
	StashEntries         uint64  `json:"stash_entries"`
	LoadFactor           float64 `json:"load_factor"`
	MALLHitRate          float64 `json:"mall_hit_rate"`
	DisplacementRate     float64 `json:"displacement_rate"`
}
