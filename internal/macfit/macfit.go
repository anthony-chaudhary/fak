package macfit

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Input describes a modeled unified-memory budget. All byte quantities are
// binary bytes; ContextTokens, SharedPrefixTokens, and TailCapTokens are tokens.
type Input struct {
	MemoryBytes        uint64 `json:"memory_bytes"`
	ReserveBytes       uint64 `json:"reserve_bytes"`
	WeightBytes        uint64 `json:"weight_bytes"`
	ContextTokens      uint64 `json:"context_tokens"`
	Layers             uint64 `json:"layers"`
	KVHeads            uint64 `json:"kv_heads"`
	HeadDim            uint64 `json:"head_dim"`
	KVBytesPerElement  uint64 `json:"kv_bytes_per_element"`
	SharedPrefixTokens uint64 `json:"shared_prefix_tokens"`
	TailCapTokens      uint64 `json:"tail_cap_tokens"`
}

// Result is a modeled capacity comparison, not a hardware measurement.
type Result struct {
	Schema                 string `json:"schema"`
	Provenance             string `json:"provenance"`
	MemoryBytes            uint64 `json:"memory_bytes"`
	ReserveBytes           uint64 `json:"reserve_bytes"`
	WeightBytes            uint64 `json:"weight_bytes"`
	KVPoolBytes            uint64 `json:"kv_pool_bytes"`
	KVBytesPerToken        uint64 `json:"kv_bytes_per_token"`
	OffKVBytesPerAgent     uint64 `json:"off_kv_bytes_per_agent"`
	OnSharedKVBytes        uint64 `json:"on_shared_kv_bytes"`
	OnTailKVBytesPerAgent  uint64 `json:"on_tail_kv_bytes_per_agent"`
	OffAgentsThatFit       uint64 `json:"off_agents_that_fit"`
	OnAgentsThatFit        uint64 `json:"on_agents_that_fit"`
	ExtraAgents            uint64 `json:"extra_agents"`
	CrossoverContextTokens uint64 `json:"crossover_context_tokens,omitempty"`
	CrossoverFound         bool   `json:"crossover_found"`
}

func mul(values ...uint64) (uint64, error) {
	n := uint64(1)
	for _, v := range values {
		if v != 0 && n > math.MaxUint64/v {
			return 0, errors.New("capacity arithmetic overflows uint64")
		}
		n *= v
	}
	return n, nil
}

func fit(pool, fixed, perAgent uint64) uint64 {
	if fixed > pool {
		return 0
	}
	if perAgent == 0 {
		return 0 // no private tail means the requested distinct-agent model is undefined
	}
	return (pool - fixed) / perAgent
}

// Calculate compares independent full-context KV with one shared prefix plus a
// bounded private tail per agent. Model weights are resident once in both cases.
func Calculate(in Input) (Result, error) {
	if in.MemoryBytes == 0 || in.ContextTokens == 0 || in.Layers == 0 || in.KVHeads == 0 || in.HeadDim == 0 || in.KVBytesPerElement == 0 {
		return Result{}, errors.New("memory, context, layers, kv-heads, head-dim, and kv element bytes must be positive")
	}
	if in.TailCapTokens == 0 {
		return Result{}, errors.New("tail cap must be positive")
	}
	if in.SharedPrefixTokens >= in.ContextTokens {
		return Result{}, errors.New("shared prefix must be shorter than the full context")
	}
	if in.ReserveBytes > in.MemoryBytes || in.WeightBytes > in.MemoryBytes-in.ReserveBytes {
		return Result{}, errors.New("reserve plus weights exceed unified memory")
	}
	kvpt, err := mul(2, in.Layers, in.KVHeads, in.HeadDim, in.KVBytesPerElement)
	if err != nil {
		return Result{}, err
	}
	pool := in.MemoryBytes - in.ReserveBytes - in.WeightBytes
	at := func(context uint64) (off, on uint64, offPer, shared, tailPer uint64, err error) {
		offPer, err = mul(kvpt, context)
		if err != nil {
			return
		}
		prefix := min(in.SharedPrefixTokens, context)
		shared, err = mul(kvpt, prefix)
		if err != nil {
			return
		}
		tailTokens := context - prefix
		if tailTokens > in.TailCapTokens {
			tailTokens = in.TailCapTokens
		}
		tailPer, err = mul(kvpt, tailTokens)
		if err != nil {
			return
		}
		off = fit(pool, 0, offPer)
		on = fit(pool, shared, tailPer)
		return
	}
	off, on, offPer, shared, tailPer, err := at(in.ContextTokens)
	if err != nil {
		return Result{}, err
	}

	// A distinct-agent comparison requires at least one private token. Search
	// the requested horizon for the first context where sharing/capping buys a slot.
	start := min(in.SharedPrefixTokens, in.ContextTokens) + 1
	var cross uint64
	found := false
	if start <= in.ContextTokens {
		lo, hi := start, in.ContextTokens
		for lo <= hi {
			mid := lo + (hi-lo)/2
			a, b, _, _, _, e := at(mid)
			if e != nil {
				return Result{}, e
			}
			if b > a {
				cross, found, hi = mid, true, mid-1
			} else {
				lo = mid + 1
			}
		}
	}
	extra := uint64(0)
	if on > off {
		extra = on - off
	}
	return Result{
		Schema: "fak-macfit/1", Provenance: "modeled", MemoryBytes: in.MemoryBytes,
		ReserveBytes: in.ReserveBytes, WeightBytes: in.WeightBytes, KVPoolBytes: pool,
		KVBytesPerToken: kvpt, OffKVBytesPerAgent: offPer, OnSharedKVBytes: shared,
		OnTailKVBytesPerAgent: tailPer, OffAgentsThatFit: off, OnAgentsThatFit: on,
		ExtraAgents: extra, CrossoverContextTokens: cross, CrossoverFound: found,
	}, nil
}

func (r Result) Validate() error {
	if r.Schema != "fak-macfit/1" {
		return fmt.Errorf("unexpected schema %q", r.Schema)
	}
	return nil
}

// GiB is the number of bytes in one gibibyte.
const GiB = uint64(1 << 30)

// DefaultMinHeadroomRatio is the minimum guaranteed headroom ratio to prevent swap (20%).
const DefaultMinHeadroomRatio = 0.20

// ModelTier describes an optimal model configuration for a unified memory tier.
type ModelTier struct {
	Name           string `json:"name"`             // e.g. "7B", "27B", "70B"
	ModelID        string `json:"model_id"`         // e.g. "qwen3.8-7b-q4_k_m"
	QuantTier      string `json:"quant_tier"`       // e.g. "Q4_K_M"
	WeightBytes    uint64 `json:"weight_bytes"`     // resident model weight bytes
	Layers         uint64 `json:"layers"`           // transformer layer count
	KVHeads        uint64 `json:"kv_heads"`         // KV head count
	HeadDim        uint64 `json:"head_dim"`         // dimension per head
	MinMemoryBytes uint64 `json:"min_memory_bytes"` // minimum RAM required for this tier
}

// StandardTiers defines the default Apple Silicon tier hierarchy.
var StandardTiers = []ModelTier{
	{
		Name:           "70B",
		ModelID:        "qwen3.8-70b-q4_k_m",
		QuantTier:      "Q4_K_M",
		WeightBytes:    42 * GiB,
		Layers:         80,
		KVHeads:        8,
		HeadDim:        128,
		MinMemoryBytes: 64 * GiB,
	},
	{
		Name:           "27B",
		ModelID:        "qwen3.8-27b-q4_k_m",
		QuantTier:      "Q4_K_M",
		WeightBytes:    16 * GiB,
		Layers:         64,
		KVHeads:        4,
		HeadDim:        128,
		MinMemoryBytes: 36 * GiB,
	},
	{
		Name:           "7B",
		ModelID:        "qwen3.8-7b-q4_k_m",
		QuantTier:      "Q4_K_M",
		WeightBytes:    9 * GiB / 2, // 4.5 GiB
		Layers:         28,
		KVHeads:        4,
		HeadDim:        128,
		MinMemoryBytes: 8 * GiB,
	},
	{
		Name:           "3B",
		ModelID:        "qwen3.8-3b-q4_k_m",
		QuantTier:      "Q4_K_M",
		WeightBytes:    2 * GiB,
		Layers:         16,
		KVHeads:        2,
		HeadDim:        128,
		MinMemoryBytes: 4 * GiB,
	},
}

// SelectModelTier selects the optimal model tier for the given unified memory bytes.
func SelectModelTier(memoryBytes uint64) ModelTier {
	for _, tier := range StandardTiers {
		if memoryBytes >= tier.MinMemoryBytes {
			return tier
		}
	}
	return StandardTiers[len(StandardTiers)-1]
}

// TurnkeyProfile describes the sizing calculation for turnkey model provisioning.
type TurnkeyProfile struct {
	Schema              string    `json:"schema"`
	MemoryBytes         uint64    `json:"memory_bytes"`
	ReserveBytes        uint64    `json:"reserve_bytes"`
	HeadroomBytes       uint64    `json:"headroom_bytes"`
	HeadroomRatio       float64   `json:"headroom_ratio"`
	Tier                ModelTier `json:"tier"`
	ContextBudgetTokens uint64    `json:"context_budget_tokens"`
	KVBytesPerToken     uint64    `json:"kv_bytes_per_token"`
	KVPoolBytes         uint64    `json:"kv_pool_bytes"`
}

// ConfigureTurnkey auto-selects the optimal model tier and context budget for a given
// memory size, guaranteeing >= 20% memory headroom to prevent swap.
func ConfigureTurnkey(memoryBytes uint64) (TurnkeyProfile, error) {
	if memoryBytes == 0 {
		return TurnkeyProfile{}, errors.New("memory bytes must be positive")
	}

	// 1. Reserve at least 20% of unified memory for OS/WindowServer/headroom to prevent swap.
	reserveBytes := (memoryBytes * 20) / 100
	if reserveBytes == 0 {
		reserveBytes = 1
	}
	usableBytes := memoryBytes - reserveBytes

	// 2. Select the optimal model tier that fits within usableBytes.
	tier := SelectModelTier(memoryBytes)
	if tier.WeightBytes >= usableBytes {
		fitted := false
		for i := len(StandardTiers) - 1; i >= 0; i-- {
			if StandardTiers[i].WeightBytes < usableBytes {
				tier = StandardTiers[i]
				fitted = true
				break
			}
		}
		if !fitted {
			return TurnkeyProfile{}, fmt.Errorf("insufficient memory (%d bytes): cannot guarantee 20%% headroom", memoryBytes)
		}
	}

	// 3. Allocate remaining usable memory for KV cache pool.
	kvPoolBytes := usableBytes - tier.WeightBytes

	// KV bytes per token (FP16 = 2 bytes per element, key + value = 2).
	kvpt, err := mul(2, tier.Layers, tier.KVHeads, tier.HeadDim, 2)
	if err != nil {
		return TurnkeyProfile{}, err
	}

	// 4. Calculate maximum context tokens that fit in the pool.
	maxTokens := kvPoolBytes / kvpt
	if maxTokens == 0 {
		maxTokens = 512
	}

	// Bucket context budget to standard boundaries without exceeding maxTokens.
	var contextBudget uint64
	for _, bucket := range []uint64{65536, 32768, 16384, 8192, 4096, 2048, 1024, 512} {
		if maxTokens >= bucket {
			contextBudget = bucket
			break
		}
	}
	if contextBudget == 0 {
		contextBudget = maxTokens
	}

	// 5. Total used memory = weights + KV cache context budget.
	usedKVBytes := contextBudget * kvpt
	allocatedBytes := tier.WeightBytes + usedKVBytes
	headroomBytes := memoryBytes - allocatedBytes
	headroomRatio := float64(headroomBytes) / float64(memoryBytes)

	return TurnkeyProfile{
		Schema:              "fak-macfit-turnkey/1",
		MemoryBytes:         memoryBytes,
		ReserveBytes:        reserveBytes,
		HeadroomBytes:       headroomBytes,
		HeadroomRatio:       headroomRatio,
		Tier:                tier,
		ContextBudgetTokens: contextBudget,
		KVBytesPerToken:     kvpt,
		KVPoolBytes:         kvPoolBytes,
	}, nil
}

// DetectUnifiedMemory inspects Apple Silicon physical memory.
// It checks FAK_UP_MEMORY_BYTES override, then queries host system memory info.
func DetectUnifiedMemory() (uint64, error) {
	if v := os.Getenv("FAK_UP_MEMORY_BYTES"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil && n > 0 {
			return n, nil
		}
	}
	total, _, known := compute.HostSystemMemoryInfo()
	if known && total > 0 {
		return uint64(total), nil
	}
	return 16 * GiB, nil
}
