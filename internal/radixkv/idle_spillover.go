package radixkv

import (
	"errors"
	"sort"
)

// idle_spillover.go — single-node idle-slot GPU->RAM->local-NVMe KV spillover trigger.
//
// Targets 1-8 GPU single-node deployments with local host DRAM (L2) and local NVMe/SSD
// (L3) tiering (no cross-node RDMA/NIXL). When a cached prefix node with no active
// request leases (refs == 0) and no eviction immunity (not pinned, no active session
// holderRefs, no live Tier 0 retention) sits idle past the configured logical-clock
// threshold, TriggerIdleSpillover demotes it down the tier hierarchy:
//   1. GPU (L1, n.snapshot) -> Host DRAM (L2, n.hostSnapshot) -> release GPU copy.
//   2. Host DRAM (L2, n.hostSnapshot) -> Local NVMe/SSD (L3, n.ssdSnapshot) -> release RAM copy.
//
// Default-off contract: nothing runs unless Enabled is true and TriggerIdleSpillover
// is called.

const (
	// DefaultGPUIdleSpillTicks is the default logical-clock idle age before a GPU L1
	// snapshot is spilled to host DRAM L2.
	DefaultGPUIdleSpillTicks uint64 = 64

	// DefaultRAMIdleSpillTicks is the default logical-clock idle age before a host
	// DRAM L2 snapshot is spilled to local NVMe/SSD L3.
	DefaultRAMIdleSpillTicks uint64 = 256

	// DefaultMaxSpillsPerTick bounds how many snapshots one trigger pass demotes so a
	// single idle sweep cannot stall a latency-sensitive decode step.
	DefaultMaxSpillsPerTick = 4
)

var (
	// ErrIdleSpilloverConfig is returned when an IdleSpilloverConfig has invalid
	// bounds (e.g. negative MaxSpillsPerTick).
	ErrIdleSpilloverConfig = errors.New("radixkv: invalid idle spillover config")
)

// IdleSpilloverConfig configures the single-node idle-slot GPU->RAM->local-NVMe
// spillover trigger.
type IdleSpilloverConfig struct {
	// Enabled activates the trigger. When false, TriggerIdleSpillover is a no-op.
	Enabled bool
	// GPUIdleTicks is the minimum logical-clock idle age (tree.clock - n.lastUsed)
	// required before a GPU L1 snapshot is staged to Host DRAM L2 and evicted from GPU.
	GPUIdleTicks uint64
	// RAMIdleTicks is the minimum logical-clock idle age required before a Host DRAM L2
	// snapshot (with no GPU copy) is spilled to Local NVMe/SSD L3 and evicted from RAM.
	RAMIdleTicks uint64
	// MaxSpillsPerTick caps total demotions per TriggerIdleSpillover call (0 = unlimited).
	MaxSpillsPerTick int
}

// DefaultIdleSpilloverConfig returns the enabled single-node idle spillover defaults.
func DefaultIdleSpilloverConfig() IdleSpilloverConfig {
	return IdleSpilloverConfig{
		Enabled:          true,
		GPUIdleTicks:     DefaultGPUIdleSpillTicks,
		RAMIdleTicks:     DefaultRAMIdleSpillTicks,
		MaxSpillsPerTick: DefaultMaxSpillsPerTick,
	}
}

// Validate checks that cfg is well-formed, failing closed on negative bounds.
func (c IdleSpilloverConfig) Validate() error {
	if c.MaxSpillsPerTick < 0 {
		return ErrIdleSpilloverConfig
	}
	return nil
}

// IdleSpilloverReport records the exact tier transitions performed by one
// TriggerIdleSpillover pass.
type IdleSpilloverReport struct {
	Scanned        int
	StagedToRAM    int
	EvictedFromGPU int
	SpilledToNVMe  int
	EvictedFromRAM int
	FreedGPUBytes  int64
	FreedRAMBytes  int64
	SkippedPinned  int
	Faults         int
}

type idleSpillCandidate struct {
	ns       string
	node     *node
	lastUsed uint64
	plen     int
}

// TriggerIdleSpillover sweeps the tree for idle, unleased, non-immune prefix snapshots
// and demotes them down the single-node hierarchy: GPU (L1) -> Host DRAM (L2) ->
// Local NVMe/SSD (L3). Candidates are processed oldest-idle first, bounded by
// cfg.MaxSpillsPerTick (when > 0).
func (t *Tree) TriggerIdleSpillover(cfg IdleSpilloverConfig) IdleSpilloverReport {
	var rep IdleSpilloverReport
	if t == nil || !cfg.Enabled || cfg.Validate() != nil {
		return rep
	}

	budget := cfg.MaxSpillsPerTick
	unlimited := budget == 0

	// Phase 1: GPU (L1) -> Host DRAM (L2)
	if t.HostL2Enabled() {
		var gpuCands []idleSpillCandidate
		t.forEachNodeNS(func(ns string, n *node) {
			if n.snapshot == nil {
				return
			}
			rep.Scanned++
			if n.refs > 0 || n.IsComputing() || t.isNodePinnedOrImmune(n) {
				if t.isNodePinnedOrImmune(n) {
					rep.SkippedPinned++
				}
				return
			}
			if t.clock >= n.lastUsed && t.clock-n.lastUsed >= cfg.GPUIdleTicks {
				gpuCands = append(gpuCands, idleSpillCandidate{
					ns:       ns,
					node:     n,
					lastUsed: n.lastUsed,
					plen:     n.plen,
				})
			}
		})
		sortIdleCandidates(gpuCands)

		for _, c := range gpuCands {
			if !unlimited && budget <= 0 {
				break
			}
			n := c.node
			if n.snapshot == nil {
				continue
			}
			_, devBytes := n.snapshot.ResidencyBytes()
			digest := t.snapshotDigest(c.ns, n)
			if n.hostSnapshot == nil && n.ssdSnapshot == nil && n.remoteSnapshot == nil {
				tr := t.StageSnapshotToHost(digest)
				if tr.Outcome != SnapshotTransferOK {
					rep.Faults++
					continue
				}
				rep.StagedToRAM++
			}
			if n.hostSnapshot != nil || n.ssdSnapshot != nil || n.remoteSnapshot != nil {
				if evicted := t.EvictHotSnapshot(digest); evicted > 0 {
					rep.EvictedFromGPU++
					rep.FreedGPUBytes += devBytes
					if !unlimited {
						budget--
					}
				}
			}
		}
	}

	// Phase 2: Host DRAM (L2) -> Local NVMe/SSD (L3)
	if t.LocalSSDSnapshotEnabled() {
		var ramCands []idleSpillCandidate
		t.forEachNodeNS(func(ns string, n *node) {
			if n.hostSnapshot == nil || n.snapshot != nil {
				return
			}
			rep.Scanned++
			if n.refs > 0 || n.IsComputing() || t.isNodePinnedOrImmune(n) {
				if t.isNodePinnedOrImmune(n) {
					rep.SkippedPinned++
				}
				return
			}
			if t.clock >= n.lastUsed && t.clock-n.lastUsed >= cfg.RAMIdleTicks {
				ramCands = append(ramCands, idleSpillCandidate{
					ns:       ns,
					node:     n,
					lastUsed: n.lastUsed,
					plen:     n.plen,
				})
			}
		})
		sortIdleCandidates(ramCands)

		for _, c := range ramCands {
			if !unlimited && budget <= 0 {
				break
			}
			n := c.node
			if n.hostSnapshot == nil {
				continue
			}
			ramBytes := hostSnapshotResidentBytes(n.hostSnapshot)
			hadSSD := n.ssdSnapshot != nil
			t.spillHostSnapshotToSSD(c.ns, n)
			if n.ssdSnapshot != nil || n.remoteSnapshot != nil {
				if !hadSSD && n.ssdSnapshot != nil {
					rep.SpilledToNVMe++
				}
				t.releaseHostSnapshot(n)
				rep.EvictedFromRAM++
				rep.FreedRAMBytes += ramBytes
				if !unlimited {
					budget--
				}
			} else {
				rep.Faults++
			}
		}
	}

	return rep
}

func sortIdleCandidates(cands []idleSpillCandidate) {
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].lastUsed != cands[j].lastUsed {
			return cands[i].lastUsed < cands[j].lastUsed
		}
		if cands[i].plen != cands[j].plen {
			return cands[i].plen < cands[j].plen
		}
		return cands[i].ns < cands[j].ns
	})
}
