package agent

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// KVPrefixPressureCandidate is one native in-kernel complete-prefix owner that
// can be staged to host DRAM and released from the hot device tier.
type KVPrefixPressureCandidate struct {
	SpanDigest string
	Tokens     int
	SizeBytes  int64
	ModelID    string
}

// KVPrefixTransfer is the wire-neutral projection of a radixkv host transfer.
type KVPrefixTransfer struct {
	Outcome    string
	SpanDigest string
	Positions  int
	BytesMoved int64
	Reason     string
}

// KVPrefixPressureSource is implemented only by the native in-kernel planner.
// Upstream/proxy planners intentionally do not expose it because they do not own
// the provider's KV payload bytes.
type KVPrefixPressureSource interface {
	KVPrefixPressuredCandidates() (residentBytes int64, candidates []KVPrefixPressureCandidate)
	StageKVPrefixToHost(context.Context, string) KVPrefixTransfer
	RestoreKVPrefixFromHost(context.Context, string) KVPrefixTransfer
	EvictHotKVPrefix(string) int
}

func (p *InKernelPlanner) KVPrefixPressuredCandidates() (int64, []KVPrefixPressureCandidate) {
	if p == nil || p.tree == nil || p.backend == nil || !p.backend.Caps().DeviceMemory {
		return 0, nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	resident, source := p.tree.PressuredSnapshotCandidates()
	out := make([]KVPrefixPressureCandidate, 0, len(source))
	for _, candidate := range source {
		out = append(out, KVPrefixPressureCandidate{
			SpanDigest: candidate.Digest,
			Tokens:     candidate.Tokens,
			SizeBytes:  candidate.DeviceBytes,
			ModelID:    p.modelID,
		})
	}
	return resident, out
}

func (p *InKernelPlanner) StageKVPrefixToHost(ctx context.Context, digest string) KVPrefixTransfer {
	if err := ctx.Err(); err != nil {
		return KVPrefixTransfer{Outcome: radixkv.SnapshotTransferFault, SpanDigest: digest, Reason: err.Error()}
	}
	if p == nil || p.tree == nil {
		return KVPrefixTransfer{Outcome: radixkv.SnapshotTransferMiss, SpanDigest: digest, Reason: "native prefix tree absent"}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return lowerKVPrefixTransfer(p.tree.StageSnapshotToHost(digest))
}

func (p *InKernelPlanner) RestoreKVPrefixFromHost(ctx context.Context, digest string) KVPrefixTransfer {
	if err := ctx.Err(); err != nil {
		return KVPrefixTransfer{Outcome: radixkv.SnapshotTransferFault, SpanDigest: digest, Reason: err.Error()}
	}
	if p == nil || p.tree == nil {
		return KVPrefixTransfer{Outcome: radixkv.SnapshotTransferMiss, SpanDigest: digest, Reason: "native prefix tree absent"}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return lowerKVPrefixTransfer(p.tree.RestoreSnapshotFromHost(digest))
}

func (p *InKernelPlanner) EvictHotKVPrefix(digest string) int {
	if p == nil || p.tree == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tree.EvictHotSnapshot(digest)
}

func lowerKVPrefixTransfer(in radixkv.SnapshotTransfer) KVPrefixTransfer {
	return KVPrefixTransfer{
		Outcome:    in.Outcome,
		SpanDigest: in.Digest,
		Positions:  in.Positions,
		BytesMoved: in.BytesMoved,
		Reason:     in.Reason,
	}
}
