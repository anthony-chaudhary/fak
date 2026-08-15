package agent

import (
	"context"
	"fmt"

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

// KVPrefixRemoteConfigurer is the production boot-time extension implemented by
// the native planner. Keeping it separate leaves the pressure transport contract
// stable for bridges that do not own L3 configuration.
type KVPrefixRemoteConfigurer interface {
	ConfigureKVPrefixRemote(radixkv.SnapshotStore) error
}

// ConfigureKVPrefixRemote installs the l3kv/blobhttp byte owner on the same
// radix tree that owns native L1/L2 snapshots.
func (p *InKernelPlanner) ConfigureKVPrefixRemote(store radixkv.SnapshotStore) error {
	if p == nil || p.tree == nil || p.m == nil || p.backend == nil {
		return fmt.Errorf("agent: native backend prefix cache is unavailable")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tree.ConfigureRemoteSnapshotStore(store, p.modelID, p.backend, p.m.Cfg)
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
	local := p.tree.StageSnapshotToHost(digest)
	if !p.tree.RemoteSnapshotEnabled() {
		return lowerKVPrefixTransfer(local)
	}
	// Remote L3 is independently sufficient to preserve the hot owner. Attempt it
	// even when host L2 is disabled or full; the capacity adapter may evict only
	// after this confirmed Put returns OK.
	remote := p.tree.StageSnapshotToRemote(ctx, digest)
	if remote.Outcome != radixkv.SnapshotTransferOK {
		return lowerKVPrefixTransfer(remote)
	}
	return lowerKVPrefixTransfer(remote)
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
	local := p.tree.RestoreSnapshotFromHost(digest)
	if local.Outcome != radixkv.SnapshotTransferMiss || !p.tree.RemoteSnapshotEnabled() {
		return lowerKVPrefixTransfer(local)
	}
	return lowerKVPrefixTransfer(p.tree.RestoreSnapshotFromRemote(ctx, digest))
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
