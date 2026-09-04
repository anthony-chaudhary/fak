// Package deepseekv4moe provides a pure, weight-free synthetic model of DeepSeek V4's
// MoE architecture, dispatch contracts, cache admission, and multi-node sharding.
//
// This file implements contiguous selection range-based expert sharding for memory-constrained
// multi-node / multi-APU Tensor Parallelism (TP).
//
// Borrowed from wkljohn/ds4-strix-halo-tp-odinlink:
//   - weights_model_map_sharded_spans (ds4.c:7068-7108@48e10779d4723)
//   - ds4_gpu_tp_expert_shard_remap (ds4_rocm.cu:1970-2039@48e10779d4723)
//
// Context (Issue #10755):
// On memory-constrained multi-node or multi-APU setups (such as dual AMD Strix Halo APU
// nodes with fixed 96 GiB GPU carve-outs), loading an entire 256- or 384-expert MoE model
// on each rank and masking unowned weights causes VRAM exhaustion, OS page-ins from disk,
// and catastrophic OOM crashes.
//
// In ds4-strix-halo-tp-odinlink, each rank maps only its owned contiguous expert range
// [Base, Base + Count) (e.g. 128 of 256 experts, taking 80.76 GiB instead of 95.52 GiB).
// Backing storage is allocated ONLY for resident experts (0 bytes for unowned experts).
// During router scoring, unowned experts are remapped to index 0 with weight 0.0f
// (for branchless GPU kernel execution) or marked with negative sentinels (for skip-based
// host dispatch). AllReduce across ranks yields 100% mathematical parity with un-sharded execution.
package deepseekv4moe

import (
	"errors"
	"fmt"
)

// Sharding errors.
var (
	ErrInvalidWorldSize      = errors.New("deepseekv4moe: world size must be positive and <= total experts")
	ErrInvalidExpertSpan     = errors.New("deepseekv4moe: invalid expert span")
	ErrInvalidBytesPerExpert = errors.New("deepseekv4moe: bytes per expert must be positive")
	ErrUnownedExpert         = errors.New("deepseekv4moe: expert is unowned by this rank")
	ErrExpertOutOfRange      = errors.New("deepseekv4moe: expert index out of range")
)

// Sentinels for router remapping.
const (
	// UnownedExpertSentinel is used for host-side dispatch loops to skip unowned experts.
	UnownedExpertSentinel = -1
	// UnownedExpertIndexZero is used for branchless GPU kernels where index 0 is valid memory
	// and multiplied by weight 0.0f to produce a harmless no-op.
	UnownedExpertIndexZero = 0
)

// ShardedExpertSpan defines a contiguous slice of experts owned by a single rank:
// [Base, Base + Count).
type ShardedExpertSpan struct {
	Base         int // 0-based global index of the first expert owned by this rank
	Count        int // number of contiguous experts owned by this rank
	TotalExperts int // total number of routed experts in the model
	Rank         int // rank ID of this span [0, WorldSize)
	WorldSize    int // total number of participating ranks
}

// Validate checks internal consistency and boundary invariants of the span.
func (s ShardedExpertSpan) Validate() error {
	if s.TotalExperts <= 0 {
		return ErrExpertCount
	}
	if s.WorldSize <= 0 || s.WorldSize > s.TotalExperts {
		return ErrInvalidWorldSize
	}
	if s.Rank < 0 || s.Rank >= s.WorldSize {
		return fmt.Errorf("%w: rank %d not in [0, %d)", ErrInvalidExpertSpan, s.Rank, s.WorldSize)
	}
	if s.Base < 0 || s.Count < 0 || s.Base+s.Count > s.TotalExperts {
		return fmt.Errorf("%w: span [%d, %d) exceeds total %d", ErrInvalidExpertSpan, s.Base, s.Base+s.Count, s.TotalExperts)
	}
	return nil
}

// Contains reports whether globalExpert falls within this span's owned contiguous range.
func (s ShardedExpertSpan) Contains(globalExpert int) bool {
	return globalExpert >= s.Base && globalExpert < s.Base+s.Count
}

// LocalIndex converts a global expert index into a 0-based resident index within this rank's span.
// Returns (-1, false) if globalExpert is not owned by this rank.
func (s ShardedExpertSpan) LocalIndex(globalExpert int) (int, bool) {
	if !s.Contains(globalExpert) {
		return -1, false
	}
	return globalExpert - s.Base, true
}

// GlobalIndex converts a 0-based resident index back to the global expert index.
// Returns (-1, false) if localIndex is out of range [0, Count).
func (s ShardedExpertSpan) GlobalIndex(localIndex int) (int, bool) {
	if localIndex < 0 || localIndex >= s.Count {
		return -1, false
	}
	return s.Base + localIndex, true
}

// PartitionContiguousSpans partitions totalExperts across worldSize ranks into contiguous,
// non-overlapping spans [Base, Base + Count) that form an exact, disjoint cover of [0, totalExperts).
// If totalExperts is not evenly divisible by worldSize, the remainder is distributed
// across the lowest-indexed ranks.
func PartitionContiguousSpans(totalExperts, worldSize int) ([]ShardedExpertSpan, error) {
	if totalExperts <= 0 {
		return nil, ErrExpertCount
	}
	if worldSize <= 0 || worldSize > totalExperts {
		return nil, ErrInvalidWorldSize
	}

	spans := make([]ShardedExpertSpan, worldSize)
	perRank := totalExperts / worldSize
	rem := totalExperts % worldSize
	base := 0

	for r := 0; r < worldSize; r++ {
		count := perRank
		if r < rem {
			count++
		}
		spans[r] = ShardedExpertSpan{
			Base:         base,
			Count:        count,
			TotalExperts: totalExperts,
			Rank:         r,
			WorldSize:    worldSize,
		}
		base += count
	}

	if base != totalExperts {
		return nil, fmt.Errorf("deepseekv4moe: partition covered %d of %d experts", base, totalExperts)
	}
	return spans, nil
}

// ExpertStorageLayout models the physical memory layout and backing storage allocation
// for a sharded rank. Backing storage is allocated ONLY for Count resident experts,
// with strictly 0 bytes allocated for unowned experts.
type ExpertStorageLayout struct {
	Span            ShardedExpertSpan
	BytesPerExpert  int64
	ResidentExperts int
	UnownedExperts  int
	ResidentBytes   int64
	UnownedBytes    int64 // strictly 0 bytes
	FullModelBytes  int64
	SavedBytes      int64
	FootprintRatio  float64 // ResidentBytes / FullModelBytes
	SavingsRatio    float64 // SavedBytes / FullModelBytes
}

// LayoutExpertStorage computes the storage layout for a given expert span and byte footprint per expert.
func LayoutExpertStorage(span ShardedExpertSpan, bytesPerExpert int64) (ExpertStorageLayout, error) {
	if err := span.Validate(); err != nil {
		return ExpertStorageLayout{}, err
	}
	if bytesPerExpert <= 0 {
		return ExpertStorageLayout{}, ErrInvalidBytesPerExpert
	}

	residentExperts := span.Count
	unownedExperts := span.TotalExperts - span.Count
	residentBytes := int64(residentExperts) * bytesPerExpert
	unownedBytes := int64(0) // 0 bytes allocated for unowned experts
	fullModelBytes := int64(span.TotalExperts) * bytesPerExpert
	savedBytes := fullModelBytes - residentBytes

	var footprintRatio, savingsRatio float64
	if fullModelBytes > 0 {
		footprintRatio = float64(residentBytes) / float64(fullModelBytes)
		savingsRatio = float64(savedBytes) / float64(fullModelBytes)
	}

	return ExpertStorageLayout{
		Span:            span,
		BytesPerExpert:  bytesPerExpert,
		ResidentExperts: residentExperts,
		UnownedExperts:  unownedExperts,
		ResidentBytes:   residentBytes,
		UnownedBytes:    unownedBytes,
		FullModelBytes:  fullModelBytes,
		SavedBytes:      savedBytes,
		FootprintRatio:  footprintRatio,
		SavingsRatio:    savingsRatio,
	}, nil
}

// NewExpertStorageLayout is an alias for LayoutExpertStorage.
func NewExpertStorageLayout(span ShardedExpertSpan, bytesPerExpert int64) (ExpertStorageLayout, error) {
	return LayoutExpertStorage(span, bytesPerExpert)
}

// AllocatedBytesForExpert returns the allocated backing memory in bytes for the specified expert.
// For resident experts within the span, it returns BytesPerExpert.
// For unowned experts, it returns 0 bytes.
func (p ExpertStorageLayout) AllocatedBytesForExpert(globalExpert int) int64 {
	if globalExpert < 0 || globalExpert >= p.Span.TotalExperts {
		return 0
	}
	if p.Span.Contains(globalExpert) {
		return p.BytesPerExpert
	}
	return 0
}

// ResidentOffset returns the byte offset of the expert within the resident allocation buffer.
// If the expert is not resident on this rank, it returns an error.
func (p ExpertStorageLayout) ResidentOffset(globalExpert int) (int64, error) {
	localIdx, ok := p.Span.LocalIndex(globalExpert)
	if !ok {
		return 0, fmt.Errorf("%w: expert %d not resident on rank %d", ErrUnownedExpert, globalExpert, p.Span.Rank)
	}
	return int64(localIdx) * p.BytesPerExpert, nil
}

// VerifyZeroUnownedAllocation asserts that strictly zero bytes are allocated
// for experts outside this rank's resident expert span.
func (p ExpertStorageLayout) VerifyZeroUnownedAllocation() error {
	for exp := 0; exp < p.Span.TotalExperts; exp++ {
		if !p.Span.Contains(exp) {
			if alloc := p.AllocatedBytesForExpert(exp); alloc != 0 {
				return fmt.Errorf("deepseekv4moe: unowned expert %d has non-zero allocation: %d bytes", exp, alloc)
			}
		}
	}
	if p.UnownedBytes != 0 {
		return fmt.Errorf("deepseekv4moe: storage layout records non-zero unowned bytes: %d", p.UnownedBytes)
	}
	return nil
}

// AllocateResidentBuffer allocates a contiguous byte slice sized strictly to ResidentBytes.
// Unowned experts consume 0 bytes of memory.
func (p ExpertStorageLayout) AllocateResidentBuffer() []byte {
	return make([]byte, p.ResidentBytes)
}

// ExpertSlice returns a sub-slice into residentBuffer corresponding to the resident expert.
// Returns an error if the expert is unowned or buffer size does not match ResidentBytes.
func (p ExpertStorageLayout) ExpertSlice(residentBuffer []byte, globalExpert int) ([]byte, error) {
	offset, err := p.ResidentOffset(globalExpert)
	if err != nil {
		return nil, err
	}
	end := offset + p.BytesPerExpert
	if int64(len(residentBuffer)) < end {
		return nil, fmt.Errorf("resident buffer too short: length %d, need %d", len(residentBuffer), end)
	}
	return residentBuffer[offset:end], nil
}

// RemapRoutedTokens remaps router output for tokens on a sharded rank.
// For resident experts (within span), the local resident index [0, Count) is returned
// with the original routing weight, and isLocal is true.
// For unowned experts, local index is mapped to 0 (valid resident memory index for branchless GPU kernels),
// the weight is zeroed (0.0f), and isLocal is false.
func RemapRoutedTokens(selectedExperts []int, weights []float32, span ShardedExpertSpan) (localIndices []int, localWeights []float32, isLocal []bool) {
	return RemapRoutedTokensWithSentinel(selectedExperts, weights, span, UnownedExpertIndexZero)
}

// RemapRoutedTokensWithSentinel remaps router selections using a custom sentinel index for unowned experts
// (such as UnownedExpertSentinel = -1 for skipping during host dispatch, or UnownedExpertIndexZero = 0).
// Unowned experts always receive a weight of 0.0f and isLocal = false.
func RemapRoutedTokensWithSentinel(selectedExperts []int, weights []float32, span ShardedExpertSpan, unownedSentinel int) (localIndices []int, localWeights []float32, isLocal []bool) {
	n := len(selectedExperts)
	localIndices = make([]int, n)
	localWeights = make([]float32, n)
	isLocal = make([]bool, n)

	for i := 0; i < n; i++ {
		expert := selectedExperts[i]
		if span.Contains(expert) {
			localIndices[i] = expert - span.Base
			if i < len(weights) {
				localWeights[i] = weights[i]
			}
			isLocal[i] = true
		} else {
			localIndices[i] = unownedSentinel
			localWeights[i] = 0.0
			isLocal[i] = false
		}
	}
	return localIndices, localWeights, isLocal
}

// CalculateShardMemorySavings computes the full model expert memory footprint,
// the sharded resident footprint for residentCount experts, and the footprint ratio (sharded / full).
func CalculateShardMemorySavings(totalExperts, residentCount int, bytesPerExpert int64) (fullBytes, shardedBytes int64, ratio float64) {
	if totalExperts <= 0 || residentCount < 0 || bytesPerExpert <= 0 {
		return 0, 0, 0
	}
	if residentCount > totalExperts {
		residentCount = totalExperts
	}
	fullBytes = int64(totalExperts) * bytesPerExpert
	shardedBytes = int64(residentCount) * bytesPerExpert
	if fullBytes > 0 {
		ratio = float64(shardedBytes) / float64(fullBytes)
	}
	return fullBytes, shardedBytes, ratio
}
