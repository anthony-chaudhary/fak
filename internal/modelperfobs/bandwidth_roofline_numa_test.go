package modelperfobs

import (
	"reflect"
	"strings"
	"testing"
)

func testNUMATopology() NUMATopology {
	return NUMATopology{Provenance: "linux-sysfs", NodeRoot: "/sys/devices/system/node", OnlineSource: "/sys/devices/system/node/online", CPUSetSource: "/proc/self/status:Cpus_allowed_list", OnlineNodeIDs: []int{0, 2}, AllowedCPUIds: []int{0, 1, 8, 9}, Nodes: []NUMATopologyNode{{ID: 0, Online: true, MemoryBytes: 8 << 30, CPUIds: []int{0, 1}, AllowedCPUIds: []int{0, 1}, Eligible: true}, {ID: 2, Online: true, MemoryBytes: 8 << 30, CPUIds: []int{8, 9}, AllowedCPUIds: []int{8, 9}, Eligible: true}, {ID: 7, Online: false, MemoryBytes: 0, CPUIds: nil, AllowedCPUIds: nil, OmissionReason: "offline"}}, Omissions: []NUMATopologyOmission{{NodeID: 7, Reason: "offline"}}}
}
func trial(i int, g float64) RooflineTrial {
	return RooflineTrial{Index: i, Iterations: 2, DurationMS: 10, TrafficBytes: 1024, GBS: g}
}
func verified(cpu, mem int) *NUMAVerifiedPlacement {
	return &NUMAVerifiedPlacement{CPUNode: cpu, MemoryNode: mem, CPUVerifier: "sched_getcpu plus sysfs cpu/node", MemoryVerifier: "/proc/self/numa_maps resident-page counts", CPUEvidence: "cpu 8 belongs to node2", MemoryEvidence: "pages=N0:0 N2:16384"}
}

func TestBuildNUMARooflineMatrixSparseNodesAndRatios(t *testing.T) {
	c := NUMARooflineCapture{Schema: NUMARooflineCaptureSchema, MachineClass: "linux/amd64", Topology: testNUMATopology(), WorkingSetBytes: 64 << 20, PeakBufferBytes: 128 << 20, TargetDurationMS: 100, RuntimeBudgetMS: 5000, DRAMIsolation: "not-proven", Pairs: []NUMARooflinePairCapture{{RequestedCPUNode: 0, RequestedMemoryNode: 0, RequestedCommand: []string{"numactl", "--cpunodebind=0", "--membind=0"}, Verified: verified(0, 0), Trials: []RooflineTrial{trial(0, 100), trial(1, 110), trial(2, 90)}}, {RequestedCPUNode: 0, RequestedMemoryNode: 2, RequestedCommand: []string{"numactl", "--cpunodebind=0", "--membind=2"}, Verified: verified(0, 2), Trials: []RooflineTrial{trial(0, 50), trial(1, 55), trial(2, 45)}}}, Omissions: []NUMARooflinePairCapture{{RequestedCPUNode: 2, RequestedMemoryNode: 7, OmissionReason: "memory node offline"}}}
	m, err := BuildNUMARooflineMatrix(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Pairs) != 2 || m.Pairs[0].SustainableGBS != 100 || m.Pairs[1].SustainableGBS != 50 {
		t.Fatalf("pairs=%+v", m.Pairs)
	}
	if m.Pairs[1].RatioToLocal == nil || *m.Pairs[1].RatioToLocal != 0.5 {
		t.Fatalf("ratio=%v", m.Pairs[1].RatioToLocal)
	}
	if m.DRAMIsolation != "not-proven" || m.Scope != "host-memory" {
		t.Fatalf("matrix=%+v", m)
	}
}

func TestBuildNUMARooflineMatrixRejectsUnverifiedAndPretendIsolation(t *testing.T) {
	base := NUMARooflineCapture{Schema: NUMARooflineCaptureSchema, MachineClass: "linux/amd64", Topology: testNUMATopology(), WorkingSetBytes: 64 << 20, PeakBufferBytes: 128 << 20, TargetDurationMS: 100, RuntimeBudgetMS: 5000, DRAMIsolation: "not-proven", Pairs: []NUMARooflinePairCapture{{RequestedCPUNode: 0, RequestedMemoryNode: 0, RequestedCommand: []string{"numactl"}, Trials: []RooflineTrial{trial(0, 1), trial(1, 1), trial(2, 1)}}}}
	if _, err := BuildNUMARooflineMatrix(base); err == nil || !strings.Contains(err.Error(), "lacks independent placement verification") {
		t.Fatalf("err=%v", err)
	}
	base.Pairs[0].Verified = verified(0, 0)
	base.DRAMIsolation = "proven"
	if _, err := BuildNUMARooflineMatrix(base); err == nil || !strings.Contains(err.Error(), "must be not-proven") {
		t.Fatalf("err=%v", err)
	}
}

func TestBuildNUMARooflineMatrixRejectsCpusetIneligiblePairAndBounds(t *testing.T) {
	topo := testNUMATopology()
	topo.Nodes[1].Eligible = false
	topo.Nodes[1].AllowedCPUIds = nil
	topo.Nodes[1].OmissionReason = "excluded-by-process-cpuset"
	c := NUMARooflineCapture{Schema: NUMARooflineCaptureSchema, Topology: topo, WorkingSetBytes: 64 << 20, PeakBufferBytes: 128 << 20, TargetDurationMS: 100, RuntimeBudgetMS: 5000, DRAMIsolation: "not-proven", Pairs: []NUMARooflinePairCapture{{RequestedCPUNode: 2, RequestedMemoryNode: 0, RequestedCommand: []string{"numactl"}, Verified: verified(2, 0), Trials: []RooflineTrial{trial(0, 1), trial(1, 1), trial(2, 1)}}}}
	if _, err := BuildNUMARooflineMatrix(c); err == nil || !strings.Contains(err.Error(), "cpuset-ineligible") {
		t.Fatalf("err=%v", err)
	}
	c.Pairs = nil
	c.Omissions = []NUMARooflinePairCapture{{RequestedCPUNode: 0, RequestedMemoryNode: 2, OmissionReason: "permission denied"}}
	c.PeakBufferBytes = MaxRooflineWorkingSet*2 + 1
	if _, err := BuildNUMARooflineMatrix(c); err == nil || !strings.Contains(err.Error(), "bounded") {
		t.Fatalf("err=%v", err)
	}
}

func TestParseIDListSparse(t *testing.T) {
	got, err := parseIDList("0-2,8,11-12")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 1, 2, 8, 11, 12}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}
