package macfit

import (
	"errors"
	"fmt"
	"math"
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
