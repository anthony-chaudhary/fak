package gateway

import (
	"fmt"
)

// KVTPTransferMode defines how KV cache shards map between source and destination tensor parallelism.
type KVTPTransferMode string

const (
	KVTPTransferBijective KVTPTransferMode = "bijective_pairing"
	KVTPTransferFanOut    KVTPTransferMode = "fan_out_broadcast"
	KVTPTransferFanIn     KVTPTransferMode = "fan_in_gather"
	KVTPTransferRefused   KVTPTransferMode = "refused_incompatible"
)

// KVShardMapping defines one explicit source-to-destination rank connection and head ownership.
type KVShardMapping struct {
	SrcRank int   `json:"src_rank"`
	DstRank int   `json:"dst_rank"`
	Heads   []int `json:"heads"`
}

// KVTPTransferStrategyReceipt records the resolved transfer strategy across TP boundaries.
type KVTPTransferStrategyReceipt struct {
	SrcTP        int              `json:"src_tp"`
	DstTP        int              `json:"dst_tp"`
	NumKVHeads   int              `json:"num_kv_heads"`
	Mode         KVTPTransferMode `json:"mode"`
	Mappings     []KVShardMapping `json:"mappings"`
	IsBijective  bool             `json:"is_bijective"`
	IsBroadcast  bool             `json:"is_broadcast"`
	RefusalError string           `json:"refusal_error,omitempty"`
}

// ResolveKVTPTransferStrategy derives explicit shard mapping across changing TP configurations.
// Incompatible configurations are refused fail-closed.
func ResolveKVTPTransferStrategy(srcTP, dstTP, numKVHeads int) (KVTPTransferStrategyReceipt, error) {
	receipt := KVTPTransferStrategyReceipt{
		SrcTP:      srcTP,
		DstTP:      dstTP,
		NumKVHeads: numKVHeads,
		Mode:       KVTPTransferRefused,
	}

	if srcTP <= 0 || dstTP <= 0 || numKVHeads <= 0 {
		receipt.RefusalError = fmt.Sprintf("all parameters must be positive: srcTP=%d, dstTP=%d, heads=%d", srcTP, dstTP, numKVHeads)
		return receipt, fmt.Errorf("%s", receipt.RefusalError)
	}

	if numKVHeads%srcTP != 0 {
		receipt.RefusalError = fmt.Sprintf("numKVHeads %d not divisible by srcTP %d", numKVHeads, srcTP)
		return receipt, fmt.Errorf("%s", receipt.RefusalError)
	}
	if numKVHeads%dstTP != 0 {
		receipt.RefusalError = fmt.Sprintf("numKVHeads %d not divisible by dstTP %d", numKVHeads, dstTP)
		return receipt, fmt.Errorf("%s", receipt.RefusalError)
	}

	headsPerSrc := numKVHeads / srcTP
	headsPerDst := numKVHeads / dstTP

	// 1. Equal-TP: exact 1-to-1 bijective mapping
	if srcTP == dstTP {
		mappings := make([]KVShardMapping, srcTP)
		for r := 0; r < srcTP; r++ {
			heads := make([]int, headsPerSrc)
			for h := 0; h < headsPerSrc; h++ {
				heads[h] = r*headsPerSrc + h
			}
			mappings[r] = KVShardMapping{
				SrcRank: r,
				DstRank: r,
				Heads:   heads,
			}
		}
		receipt.Mode = KVTPTransferBijective
		receipt.Mappings = mappings
		receipt.IsBijective = true
		return receipt, nil
	}

	// 2. Fan-out: dstTP > srcTP (e.g. TP=2 -> TP=4)
	if dstTP > srcTP && dstTP%srcTP == 0 {
		fanFactor := dstTP / srcTP
		var mappings []KVShardMapping
		for s := 0; s < srcTP; s++ {
			for f := 0; f < fanFactor; f++ {
				dstRank := s*fanFactor + f
				heads := make([]int, headsPerDst)
				for h := 0; h < headsPerDst; h++ {
					heads[h] = dstRank*headsPerDst + h
				}
				mappings = append(mappings, KVShardMapping{
					SrcRank: s,
					DstRank: dstRank,
					Heads:   heads,
				})
			}
		}
		receipt.Mode = KVTPTransferFanOut
		receipt.Mappings = mappings
		receipt.IsBroadcast = true
		return receipt, nil
	}

	// 3. Fan-in: srcTP > dstTP (e.g. TP=4 -> TP=2)
	if srcTP > dstTP && srcTP%dstTP == 0 {
		gatherFactor := srcTP / dstTP
		var mappings []KVShardMapping
		for d := 0; d < dstTP; d++ {
			for g := 0; g < gatherFactor; g++ {
				srcRank := d*gatherFactor + g
				heads := make([]int, headsPerSrc)
				for h := 0; h < headsPerSrc; h++ {
					heads[h] = srcRank*headsPerSrc + h
				}
				mappings = append(mappings, KVShardMapping{
					SrcRank: srcRank,
					DstRank: d,
					Heads:   heads,
				})
			}
		}
		receipt.Mode = KVTPTransferFanIn
		receipt.Mappings = mappings
		return receipt, nil
	}

	// 4. Incompatible TP combinations (e.g. TP=3 to TP=5)
	receipt.RefusalError = fmt.Sprintf("incompatible TP layouts: srcTP=%d and dstTP=%d cannot be partitioned cleanly", srcTP, dstTP)
	return receipt, fmt.Errorf("%s", receipt.RefusalError)
}
