package hwutil

// Fleet roll-up in the frozen fak.hwutil.fleet.v1 shape. The roll-up rule:
// sum byte rates and iops, max utilization/percent, min free bytes, count
// unread fields; a node whose snapshot carries a non-empty error is listed,
// never merged as zeros.

import (
	"sort"
	"time"
)

// FleetNode is one node's summary row. Every metric pointer is nil when the
// node did not report it — including every metric of an errored node, which
// is listed but never merged as zeros.
type FleetNode struct {
	Host             string   `json:"host"`
	CPUUtilPct       *float64 `json:"cpu_util_pct"`
	DramUsedPct      *float64 `json:"dram_used_pct"`
	TopDiskBps       *float64 `json:"top_disk_bps"`
	TopNetBps        *float64 `json:"top_net_bps"`
	GPUUtilMaxPct    *float64 `json:"gpu_util_max_pct"`
	TierFreeMinBytes *int64   `json:"tier_free_min_bytes"`
	UnreadCount      int      `json:"unread_count"`
}

// FleetRollups is the declarative roll-up policy embedded in every fleet
// document so consumers apply the same merge rule.
type FleetRollups struct {
	Sum   []string `json:"sum"`
	Max   []string `json:"max"`
	Min   []string `json:"min"`
	Count []string `json:"count"`
}

// Fleet is one multi-node roll-up document.
type Fleet struct {
	Schema  string       `json:"schema"`
	At      time.Time    `json:"at"`
	Nodes   []FleetNode  `json:"nodes"`
	Rollups FleetRollups `json:"rollups"`
}

// StandardRollups is the frozen policy descriptor.
func StandardRollups() FleetRollups {
	return FleetRollups{
		Sum:   []string{"*_bytes_per_s", "iops_*"},
		Max:   []string{"util_pct", "*_pct"},
		Min:   []string{"*_free_bytes"},
		Count: []string{"unread"},
	}
}

// Aggregate rolls a set of node snapshots into one fleet document, sorted by
// host for determinism. Errored nodes keep their identity row with all
// metrics nil so downstream merges treat them as absent, not zero.
func Aggregate(at time.Time, snaps []*Snapshot) *Fleet {
	nodes := make([]FleetNode, 0, len(snaps))
	for _, s := range snaps {
		if s == nil {
			continue
		}
		nodes = append(nodes, fleetNode(s))
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].Host < nodes[j].Host })
	return &Fleet{Schema: FleetSchema, At: at.UTC(), Nodes: nodes, Rollups: StandardRollups()}
}

func fleetNode(s *Snapshot) FleetNode {
	n := FleetNode{Host: s.Host, UnreadCount: s.UnreadCount()}
	if s.Error != "" {
		// Listed, never merged as zeros: identity only.
		return n
	}
	if s.CPU != nil && s.CPU.UtilPct != nil {
		n.CPUUtilPct = s.CPU.UtilPct
	}
	if s.DRAM != nil && s.DRAM.TotalBytes != nil && s.DRAM.UsedBytes != nil && *s.DRAM.TotalBytes > 0 {
		pct := round6(100 * float64(*s.DRAM.UsedBytes) / float64(*s.DRAM.TotalBytes))
		n.DramUsedPct = &pct
	}
	var topDisk *float64
	for _, d := range s.Disks {
		r, w := 0.0, 0.0
		if d.ReadBytesPerS != nil {
			r = *d.ReadBytesPerS
		}
		if d.WriteBytesPerS != nil {
			w = *d.WriteBytesPerS
		}
		v := round6(r + w)
		if topDisk == nil || v > *topDisk {
			topDisk = &v
		}
	}
	n.TopDiskBps = topDisk
	var topNet *float64
	for _, x := range s.Net {
		r, t := 0.0, 0.0
		if x.RxBytesPerS != nil {
			r = *x.RxBytesPerS
		}
		if x.TxBytesPerS != nil {
			t = *x.TxBytesPerS
		}
		v := round6(r + t)
		if topNet == nil || v > *topNet {
			topNet = &v
		}
	}
	n.TopNetBps = topNet
	var gpuMax *float64
	for _, g := range s.GPUs {
		if g.UtilPct == nil {
			continue
		}
		if gpuMax == nil || *g.UtilPct > *gpuMax {
			gpuMax = g.UtilPct
		}
	}
	n.GPUUtilMaxPct = gpuMax
	n.TierFreeMinBytes = tierFreeMin(s)
	return n
}

// tierFreeMin returns the minimum free bytes across the node's storage tiers:
// DRAM available plus each disk's free space. Nil when no tier reports free
// capacity.
func tierFreeMin(s *Snapshot) *int64 {
	var min *int64
	consider := func(v int64) {
		if min == nil || v < *min {
			min = i64(v)
		}
	}
	if s.DRAM != nil && s.DRAM.AvailableBytes != nil {
		consider(*s.DRAM.AvailableBytes)
	}
	for _, d := range s.Disks {
		if d.FreeBytes != nil {
			consider(*d.FreeBytes)
		}
	}
	return min
}
