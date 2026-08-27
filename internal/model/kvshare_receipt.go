package model

import "time"

// PrefixShareReceipt is the falsifiable accounting boundary for immutable paged-KV reuse.
// It distinguishes logical bytes reused from physical clone traffic and tail fragmentation.
type PrefixShareReceipt struct {
	Schema                  string        `json:"schema"`
	Engine                  string        `json:"engine"`
	PrefixTokens            int           `json:"prefix_tokens"`
	AcceptedTokens          int           `json:"accepted_tokens"`
	SharedBlocks            int           `json:"shared_blocks"`
	LogicalPrefixBytes      int64         `json:"logical_prefix_bytes"`
	PhysicalBytesBefore     int64         `json:"physical_bytes_before"`
	PhysicalBytesAfterFork  int64         `json:"physical_bytes_after_fork"`
	ForkCloneBytes          int64         `json:"fork_clone_bytes"`
	BytesAvoided            int64         `json:"bytes_avoided"`
	BytesAvoidedPerAccepted float64       `json:"bytes_avoided_per_accepted_token"`
	FragmentationRatio      float64       `json:"fragmentation_ratio"`
	ForkLatency             time.Duration `json:"fork_latency"`
	Validation              string        `json:"validation"`
}

// ForkMeasured shares an immutable prefix and returns a receipt proving whether any
// physical KV storage was cloned. acceptedTokens is the quality-complete denominator;
// callers must supply the number of accepted continuation tokens for their envelope.
func (s *PagedKV) ForkMeasured(acceptedTokens int) (*PagedKV, PrefixShareReceipt) {
	beforeBlocks := s.pool.PhysicalBlocks()
	start := time.Now()
	fork := s.Fork()
	latency := time.Since(start)
	afterBlocks := s.pool.PhysicalBlocks()
	blockBytes := int64(s.pool.nLayers * s.pool.planes * s.pool.blockTokens * s.pool.stride * 4)
	logicalBytes := int64(len(s.table)) * blockBytes
	cloneBytes := int64(afterBlocks-beforeBlocks) * blockBytes
	if cloneBytes < 0 {
		cloneBytes = 0
	}
	avoided := logicalBytes - cloneBytes
	if avoided < 0 {
		avoided = 0
	}
	perAccepted := float64(0)
	if acceptedTokens > 0 {
		perAccepted = float64(avoided) / float64(acceptedTokens)
	}
	validation := "shared-zero-clone"
	if cloneBytes != 0 || fork.nTokens != s.nTokens || len(fork.table) != len(s.table) {
		validation = "failed"
	}
	return fork, PrefixShareReceipt{
		Schema: "fak-prefix-share-receipt/1", Engine: "fak-native",
		PrefixTokens: s.nTokens, AcceptedTokens: acceptedTokens, SharedBlocks: len(s.table),
		LogicalPrefixBytes: logicalBytes, PhysicalBytesBefore: int64(beforeBlocks) * blockBytes,
		PhysicalBytesAfterFork: int64(afterBlocks) * blockBytes, ForkCloneBytes: cloneBytes,
		BytesAvoided: avoided, BytesAvoidedPerAccepted: perAccepted,
		FragmentationRatio: s.OverheadRatio(), ForkLatency: latency, Validation: validation,
	}
}
