package hwutil

import (
	"reflect"
	"testing"
	"time"
)

func snapForAgg() *Snapshot {
	s := &Snapshot{Schema: Schema, Host: "b-node", Arch: "amd64", Os: "linux", At: time.Now().UTC()}
	util := 10.0
	s.CPU = &CPU{Cores: 4, UtilPct: &util}
	s.DRAM = &DRAM{TotalBytes: i64(100), UsedBytes: i64(40), AvailableBytes: i64(2000)}
	r1, w1 := 100.0, 50.0
	s.Disks = []Disk{
		{Name: "d1", Kind: DiskNVMe, ReadBytesPerS: &r1, WriteBytesPerS: &w1, FreeBytes: i64(1000)},
		{Name: "d2", Kind: DiskUnknown, ReadBytesPerS: f64(300)},
	}
	rx1, tx1 := 10.0, 5.0
	rx2, tx2 := 50.0, 25.0
	s.Net = []NetIfc{
		{Iface: "e0", RxBytesPerS: &rx1, TxBytesPerS: &tx1},
		{Iface: "e1", RxBytesPerS: &rx2, TxBytesPerS: &tx2},
	}
	g1, g2 := 80.0, 30.0
	s.GPUs = []GPU{{UUID: "g1", Name: "G1", UtilPct: &g1}, {UUID: "g2", Name: "G2", UtilPct: &g2}}
	return s
}

func TestAggregateRollupMath(t *testing.T) {
	a := snapForAgg()
	b := &Snapshot{Schema: Schema, Host: "a-node", Arch: "amd64", Os: "linux", Error: string(CodePartial) + ": cpu: no procfs"}
	fleet := Aggregate(time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC), []*Snapshot{a, b})

	if fleet.Schema != FleetSchema {
		t.Fatalf("schema = %q", fleet.Schema)
	}
	if len(fleet.Nodes) != 2 || fleet.Nodes[0].Host != "a-node" || fleet.Nodes[1].Host != "b-node" {
		t.Fatalf("nodes not sorted by host: %+v", fleet.Nodes)
	}
	// Errored node: identity row only, never merged as zeros.
	errNode := fleet.Nodes[0]
	if errNode.Host != "a-node" || errNode.CPUUtilPct != nil || errNode.TopDiskBps != nil ||
		errNode.GPUUtilMaxPct != nil || errNode.TierFreeMinBytes != nil {
		t.Fatalf("errored node must be listed without metrics: %+v", errNode)
	}
	n := fleet.Nodes[1]
	if n.CPUUtilPct == nil || *n.CPUUtilPct != 10 {
		t.Fatalf("cpu util = %v", n.CPUUtilPct)
	}
	if n.DramUsedPct == nil || *n.DramUsedPct != 40 { // 40/100
		t.Fatalf("dram used pct = %v", n.DramUsedPct)
	}
	if n.TopDiskBps == nil || *n.TopDiskBps != 300 { // d2 beats d1's 150
		t.Fatalf("top disk bps = %v", n.TopDiskBps)
	}
	if n.TopNetBps == nil || *n.TopNetBps != 75 { // e1: 50+25
		t.Fatalf("top net bps = %v", n.TopNetBps)
	}
	if n.GPUUtilMaxPct == nil || *n.GPUUtilMaxPct != 80 {
		t.Fatalf("gpu max = %v", n.GPUUtilMaxPct)
	}
	if n.TierFreeMinBytes == nil || *n.TierFreeMinBytes != 1000 { // min(dram avail 2000, d1 free 1000)
		t.Fatalf("tier free min = %v", n.TierFreeMinBytes)
	}
	// Roll-up policy descriptor is frozen.
	if !reflect.DeepEqual(fleet.Rollups, StandardRollups()) {
		t.Fatalf("rollups = %+v", fleet.Rollups)
	}
}

func TestAggregateSkipsNilSnapshotsAndEmptyInput(t *testing.T) {
	fleet := Aggregate(time.Now(), []*Snapshot{nil, snapForAgg(), nil})
	if len(fleet.Nodes) != 1 || fleet.Nodes[0].Host != "b-node" {
		t.Fatalf("nodes = %+v", fleet.Nodes)
	}
	empty := Aggregate(time.Now(), nil)
	if empty.Schema != FleetSchema || len(empty.Nodes) != 0 {
		t.Fatalf("empty fleet = %+v", empty)
	}
}
