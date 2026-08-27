package model

import "fmt"

// KVAllocationMode identifies one fak-native KV storage geometry.
type KVAllocationMode string

const (
	KVAllocationContiguous KVAllocationMode = "contiguous-arena"
	KVAllocationPaged      KVAllocationMode = "paged-blocks"
	KVAllocationVirtual    KVAllocationMode = "virtual-growth"
)

// KVAllocationReceipt makes storage geometry and growth-copy costs comparable at
// one accepted-token envelope. BlockTouches is the conservative TLB/cache working-
// set proxy; serving benchmarks can attach latency without changing this accounting.
type KVAllocationReceipt struct {
	Schema                string           `json:"schema"`
	Engine                string           `json:"engine"`
	Mode                  KVAllocationMode `json:"mode"`
	AcceptedTokens        int              `json:"accepted_tokens"`
	AdmittedTokens        int              `json:"admitted_tokens"`
	PhysicalBytes         int64            `json:"physical_bytes"`
	LiveBytes             int64            `json:"live_bytes"`
	GrowthCopyBytes       int64            `json:"growth_copy_bytes"`
	FragmentationBytes    int64            `json:"fragmentation_bytes"`
	BytesPerAcceptedToken float64          `json:"bytes_per_accepted_token"`
	BlockTouches          int              `json:"block_touches"`
	AllocationEvents      int              `json:"allocation_events"`
	QualityConstraint     string           `json:"quality_constraint"`
	Rollback              string           `json:"rollback"`
}

// CompareKVAllocation derives net byte and allocation costs for equal token and
// tensor geometry. rowBytes includes every layer/plane for one token.
func CompareKVAllocation(mode KVAllocationMode, tokens, admittedTokens, rowBytes, blockTokens int) (KVAllocationReceipt, error) {
	if tokens < 0 || admittedTokens < tokens || rowBytes <= 0 || blockTokens <= 0 {
		return KVAllocationReceipt{}, fmt.Errorf("model: invalid KV allocation envelope")
	}
	r := KVAllocationReceipt{
		Schema: "fak-kv-allocation-receipt/1", Engine: "fak-native", Mode: mode,
		AcceptedTokens: tokens, AdmittedTokens: admittedTokens,
		LiveBytes:         int64(tokens) * int64(rowBytes),
		QualityConstraint: "identical logical K/V rows and accepted-token count",
		Rollback:          "retain contiguous arena selector",
	}
	switch mode {
	case KVAllocationContiguous:
		r.PhysicalBytes = int64(admittedTokens) * int64(rowBytes)
		if admittedTokens > 0 {
			r.AllocationEvents = 1
			r.BlockTouches = 1
		}
	case KVAllocationPaged:
		blocks := kvCeilDiv(tokens, blockTokens)
		admittedBlocks := kvCeilDiv(admittedTokens, blockTokens)
		r.PhysicalBytes = int64(admittedBlocks*blockTokens) * int64(rowBytes)
		r.BlockTouches = blocks
		r.AllocationEvents = admittedBlocks
	case KVAllocationVirtual:
		capacity := 0
		for capacity < tokens {
			next := capacity * 2
			if next < blockTokens {
				next = blockTokens
			}
			if next > admittedTokens {
				next = admittedTokens
			}
			if next <= capacity {
				return KVAllocationReceipt{}, fmt.Errorf("model: virtual growth exhausted admitted capacity")
			}
			r.GrowthCopyBytes += int64(capacity) * int64(rowBytes)
			capacity = next
			r.AllocationEvents++
		}
		r.PhysicalBytes = int64(capacity) * int64(rowBytes)
		r.BlockTouches = kvCeilDiv(tokens, blockTokens)
	default:
		return KVAllocationReceipt{}, fmt.Errorf("model: unknown KV allocation mode %q", mode)
	}
	r.FragmentationBytes = r.PhysicalBytes - r.LiveBytes
	if tokens > 0 {
		r.BytesPerAcceptedToken = float64(r.PhysicalBytes+r.GrowthCopyBytes) / float64(tokens)
	}
	return r, nil
}

func kvCeilDiv(n, d int) int {
	if n == 0 {
		return 0
	}
	return (n + d - 1) / d
}
