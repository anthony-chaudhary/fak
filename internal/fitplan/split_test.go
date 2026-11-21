package fitplan

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// rtx5080Like is a 16 GB discrete card: ~960 GB/s, a conservative dense FP16 peak.
func rtx5080Like(in Input, hostGiB int64) Input {
	in.Hardware = Hardware{
		Topology:            TopologyDiscrete,
		PoolUsableBytes:     15 * gib,
		HostPoolUsableBytes: hostGiB * gib,
		Peaks:               Peaks{BytesPerSec: 960e9, FLOPsPerSec: 100e12, Source: "test: rtx5080-like"},
		HostBytesPerSec:     96e9,
		NVMeBytesPerSec:     4e9,
	}
	return in
}

func planSplit(t *testing.T, in Input, ic Interconnect) SplitReport {
	t.Helper()
	rep := PlanSplit(in, ic)
	for _, r := range rep.Rows {
		for _, f := range []float64{r.DecodeTokSPerSeq, r.DecodeTokSAggregate} {
			if math.IsNaN(f) || math.IsInf(f, 0) || f < 0 {
				t.Fatalf("%s/%s: non-finite rate %v", r.Mode, r.Offload, f)
			}
		}
		for _, q := range r.Phases {
			if math.IsNaN(q.Seconds) || math.IsInf(q.Seconds, 0) || q.Seconds <= 0 {
				t.Fatalf("%s/%s phase %s: bad seconds %v", r.Mode, r.Offload, q.Name, q.Seconds)
			}
		}
	}
	return rep
}

func rowOf(rep SplitReport, mode SplitMode, off OffloadMode) *SplitRow {
	for i := range rep.Rows {
		if rep.Rows[i].Mode == mode && rep.Rows[i].Offload == off {
			return &rep.Rows[i]
		}
	}
	return nil
}

func pcie() Interconnect {
	return Interconnect{Devices: 2, HostLinkBytesPerSec: 25.6e9, P2PLatencySeconds: 10e-6, HostSyncSeconds: 20e-6}
}

// The operator's question: a DS4.1-Flash-class MoE on a 5080 whose experts spill past a
// 32 GB host to NVMe. A second 5080 over PCIe holds ~15 GB more experts (a modest gain);
// host RAM that holds every expert removes the NVMe term (a large gain).
func TestSplitSecondGPUVersusRAM(t *testing.T) {
	base := planSplit(t, rtx5080Like(flashLikeInput(), 32), Interconnect{Devices: 1})
	if base.Chosen == nil || base.Chosen.Binding != BindNVMe || base.Status != StatusTiered {
		t.Fatalf("1x5080 + 32 GB: want an NVMe-bound tiered choice, got %+v status=%s", base.Chosen, base.Status)
	}
	two := planSplit(t, rtx5080Like(flashLikeInput(), 32), pcie())
	if two.Chosen == nil || two.Chosen.Devices != 2 {
		t.Fatalf("2x5080: want a 2-device executable choice, got %+v", two.Chosen)
	}
	if two.Chosen.Mode != SplitExpert {
		t.Fatalf("2x5080: expert split is the executable 2-device mode, got %s", two.Chosen.Mode)
	}
	gpuGain := two.Chosen.DecodeTokSAggregate / base.Chosen.DecodeTokSAggregate
	ram := planSplit(t, rtx5080Like(flashLikeInput(), 256), Interconnect{Devices: 1})
	ramGain := ram.Chosen.DecodeTokSAggregate / base.Chosen.DecodeTokSAggregate
	if !(gpuGain > 1 && gpuGain < 1.5) {
		t.Fatalf("second GPU gain %.2fx, want a modest (1, 1.5) gain while NVMe still binds", gpuGain)
	}
	if ramGain < 3*gpuGain {
		t.Fatalf("RAM gain %.2fx vs GPU gain %.2fx: holding every expert in RAM should dominate", ramGain, gpuGain)
	}
	if ram.Chosen.NVMeBytes != 0 || ram.Chosen.Binding == BindNVMe {
		t.Fatalf("256 GB host: nothing should page from NVMe, got %+v", ram.Chosen)
	}
}

// Tensor split pays two ring all-reduces per layer: over PCIe latency it loses to one
// device on a model that fits; over an NVLink-class link it wins. It is priced but never
// chosen by default because serve cannot run it in one process.
func TestSplitTensorLatencyCrossover(t *testing.T) {
	in := rtx5080Like(baseInput(15*gib, cand(compute.KVPrecisionF32, 4096, true)), 64)
	in.Calibration = GenericEmpirical()
	in.Model.AttnLayers = 64
	all := []SplitMode{SplitSingle, SplitLayer, SplitTensor, SplitExpert}

	slow := pcie()
	slow.P2PLatencySeconds, slow.Executable = 60e-6, all
	rs := planSplit(t, in, slow)
	single, tensor := rowOf(rs, SplitSingle, ""), rowOf(rs, SplitTensor, "")
	if single == nil || tensor == nil || tensor.DecodeTokSAggregate >= single.DecodeTokSAggregate {
		t.Fatalf("PCIe @60us: tensor should lose to single; single=%+v tensor=%+v", single, tensor)
	}

	fast := Interconnect{Devices: 2, HostLinkBytesPerSec: 25.6e9, P2PBytesPerSec: 450e9, P2PLatencySeconds: 1.5e-6, Executable: all}
	rf := planSplit(t, in, fast)
	if rf.Chosen == nil || rf.Chosen.Mode != SplitTensor {
		t.Fatalf("NVLink-class: tensor should win when executable, got %+v", rf.Chosen)
	}

	fast.Executable = nil
	rd := planSplit(t, in, fast)
	if rd.Chosen == nil || rd.Chosen.Mode == SplitTensor || rd.Best == nil || rd.Best.Mode != SplitTensor || rd.Best.Executable {
		t.Fatalf("default serve support: tensor must be Best-but-not-Chosen, got chosen=%+v best=%+v", rd.Chosen, rd.Best)
	}
}

// Layer split at batch 1 sums its stages: it buys capacity, never a faster step on a
// model one device already holds.
func TestSplitLayerIsCapacityNotSpeed(t *testing.T) {
	in := rtx5080Like(baseInput(15*gib, cand(compute.KVPrecisionF32, 4096, true)), 64)
	in.Calibration = GenericEmpirical()
	ic := pcie()
	ic.Executable = []SplitMode{SplitSingle, SplitLayer}
	rep := planSplit(t, in, ic)
	single, layer := rowOf(rep, SplitSingle, ""), rowOf(rep, SplitLayer, "")
	if single == nil || layer == nil || layer.DecodeTokSAggregate > single.DecodeTokSAggregate {
		t.Fatalf("layer split should not beat one device on a resident model; single=%+v layer=%+v", single, layer)
	}
	if layer.DeviceBytes[0] >= single.DeviceBytes[0] {
		t.Fatalf("layer split should halve device 0's bytes: single=%d layer=%d", single.DeviceBytes[0], layer.DeviceBytes[0])
	}
}

// Streaming host weights over the link is capped at min(link, host RAM): with a weak CPU
// it beats computing in place; with a strong CPU, cpu offload wins.
func TestSplitStreamVersusCPUOffload(t *testing.T) {
	in := rtx5080Like(flashLikeInput(), 256)
	in.Model.ActiveParamsPerToken = 13e9 // DS4.1-Flash-class active set
	weak := Interconnect{Devices: 1, HostLinkBytesPerSec: 50e9, HostFLOPsPerSec: 0.05e12}
	rw := planSplit(t, in, weak)
	cpu, stream := rowOf(rw, SplitSingle, OffloadCPU), rowOf(rw, SplitSingle, OffloadStream)
	if cpu == nil || stream == nil || stream.DecodeTokSAggregate <= cpu.DecodeTokSAggregate {
		t.Fatalf("weak CPU: stream should beat cpu; cpu=%+v stream=%+v", cpu, stream)
	}
	if cpu.Binding != BindHostCompute {
		t.Fatalf("weak CPU: cpu row should be host-compute bound, got %s", cpu.Binding)
	}
	strong := Interconnect{Devices: 1, HostLinkBytesPerSec: 25.6e9, HostFLOPsPerSec: 2e12}
	rs := planSplit(t, in, strong)
	if rs.Chosen == nil || rs.Chosen.Offload != OffloadCPU {
		t.Fatalf("strong CPU + gen4 link: cpu offload should win, got %+v", rs.Chosen)
	}
	if st := rowOf(rs, SplitSingle, OffloadStream); st == nil || st.Binding != BindHostLink {
		t.Fatalf("gen4 stream should be link-bound, got %+v", st)
	}
}

// A batch reads 1-(1-f)^b of the routed experts, f the per-token active share: the
// geometry pins the closed form, and the aggregate rises sublinearly with batch.
func TestSplitBatchUniqueExpertsClosedForm(t *testing.T) {
	in := rtx5080Like(flashLikeInput(), 256)
	g1, _ := newPlanner(in).splitGeometry(in.Candidates[0])
	f := float64(in.Model.ActiveRoutedBytesPerToken) / float64(in.Model.RoutedExpertBytes)
	if want := float64(g1.routed) * f; math.Abs(g1.uniqueRouted-want) > 1 {
		t.Fatalf("b=1 unique routed %.0f, want the declared active bytes %.0f", g1.uniqueRouted, want)
	}
	one := planSplit(t, in, Interconnect{Devices: 1}).Chosen
	in.Workload = Workload{PromptTokens: 2048, Concurrency: 8, BatchedDecode: true}
	g8, _ := newPlanner(in).splitGeometry(in.Candidates[0])
	if want := float64(g8.routed) * (1 - math.Pow(1-f, 8)); math.Abs(g8.uniqueRouted-want) > 1 {
		t.Fatalf("b=8 unique routed %.0f, want 1-(1-f)^8 = %.0f", g8.uniqueRouted, want)
	}
	eight := planSplit(t, in, Interconnect{Devices: 1}).Chosen
	if one == nil || eight == nil {
		t.Fatal("want a choice at both batches")
	}
	if gain := eight.DecodeTokSAggregate / one.DecodeTokSAggregate; !(gain > 1 && gain < 8) {
		t.Fatalf("batch 8 aggregate gain %.2fx, want sublinear in (1, 8)", gain)
	}
}

// Serial decode (BatchedDecode off) shares one step across the streams, as Plan does.
func TestSplitSerialDecodePerSeqMatchesPlan(t *testing.T) {
	in := rtx5080Like(baseInput(15*gib, cand(compute.KVPrecisionF32, 4096, true)), 64)
	in.Calibration = GenericEmpirical()
	in.Workload = Workload{PromptTokens: 2048, Concurrency: 4}
	row := planSplit(t, in, Interconnect{Devices: 1}).Chosen
	if row == nil {
		t.Fatal("want a choice")
	}
	if math.Abs(row.DecodeTokSPerSeq*4-row.DecodeTokSAggregate) > 1e-9*row.DecodeTokSAggregate {
		t.Fatalf("serial decode: per-seq %.3f x 4 != aggregate %.3f", row.DecodeTokSPerSeq, row.DecodeTokSAggregate)
	}
	pr := Plan(in).Candidates[0]
	if rel := math.Abs(row.DecodeTokSAggregate-pr.DecodeTokSAggregate) / pr.DecodeTokSAggregate; rel > 0.02 {
		t.Fatalf("resident single row %.2f tok/s vs Plan %.2f: should agree", row.DecodeTokSAggregate, pr.DecodeTokSAggregate)
	}
}

// Attention runs where the KV lives (on the device): a long context must not inflate the
// host-compute phase of an expert offload.
func TestSplitAttentionStaysOnDevice(t *testing.T) {
	in := rtx5080Like(flashLikeInput(), 256)
	in.Model.ActiveParamsPerToken = 13e9
	in.Model.AttnLayers, in.Model.NumHeads, in.Model.HeadDim = 43, 64, 128
	ic := Interconnect{Devices: 1, HostFLOPsPerSec: 0.05e12}
	hostPhase := func(prompt int) float64 {
		in.Workload.PromptTokens = prompt
		for _, q := range rowOf(planSplit(t, in, ic), SplitSingle, OffloadCPU).Phases {
			if q.Name == "host_compute" {
				return q.Seconds
			}
		}
		t.Fatal("want a host_compute phase")
		return 0
	}
	if short, long := hostPhase(128), hostPhase(4096); math.Abs(short-long) > 1e-12 {
		t.Fatalf("host compute changed with context (%.6f -> %.6f s): attention leaked to the host", short, long)
	}
}

// Under tensor split a GQA KV shards over min(N, kv_heads) ranks.
func TestSplitTensorKVShards(t *testing.T) {
	in := rtx5080Like(baseInput(15*gib, cand(compute.KVPrecisionF32, 32768, true)), 64)
	in.Calibration = GenericEmpirical()
	ic := pcie()
	ic.Executable = []SplitMode{SplitSingle, SplitTensor}
	rep := planSplit(t, in, ic)
	tensor := rowOf(rep, SplitTensor, "")
	if tensor == nil {
		t.Fatalf("want a resident tensor row, got %+v", rep.Rows)
	}
	g, _ := newPlanner(in).splitGeometry(in.Candidates[0])
	sh := g.kvShards(2)
	if sh != 2 {
		t.Fatalf("GQA with %d kv heads should shard over 2 ranks, got %d", g.kvHeads, sh)
	}
	want := g.dense - g.dense/2 + g.kvTotal/int64(sh) + g.scratch
	if got := tensor.DeviceBytes[0]; got != want {
		t.Fatalf("tensor rank 0 bytes %d, want %d (dense/2 + kv/%d + scratch)", got, want, sh)
	}
	g.compressedKV = true
	if g.kvShards(2) != 1 {
		t.Fatal("a compressed KV must replicate across tensor ranks")
	}
}

// Expert split keeps the dense core on device 0, so its spill streams over ONE link.
func TestSplitExpertDenseSpillUsesOneLink(t *testing.T) {
	in := rtx5080Like(flashLikeInput(), 256)
	in.Hardware.PoolUsableBytes = 2 * gib // dense core 3.2 GB overflows device 0
	ic := pcie()
	ic.HostLinkBytesPerSec = 10e9
	rep := planSplit(t, in, ic)
	row := rowOf(rep, SplitExpert, OffloadStream)
	if row == nil {
		t.Fatalf("want an expert/stream row, got %+v", rep.Rows)
	}
	for _, q := range row.Phases {
		if q.Name == "stream_host_dense" {
			g, _ := newPlanner(in).splitGeometry(in.Candidates[0])
			spill := float64(g.dense-(2*gib-g.kvTotal-g.scratch)) / float64(g.dense) * g.denseStream
			if want := spill / 10e9; math.Abs(q.Seconds-want) > 0.01*want {
				t.Fatalf("dense spill streamed in %.6f s, want one 10 GB/s link: %.6f s", q.Seconds, want)
			}
			return
		}
	}
	t.Fatalf("want a stream_host_dense phase, got %+v", row.Phases)
}

func TestSplitOmitsWhenUnpriceable(t *testing.T) {
	in := rtx5080Like(flashLikeInput(), 32)
	in.Hardware.PoolUsableBytes = 0
	if rep := PlanSplit(in, pcie()); rep.Status != StatusUnknown || len(rep.Omitted) == 0 || rep.Omitted[0] != OmitPoolCapacityUnknown {
		t.Fatalf("unknown pool: want unknown + pool_capacity_unknown, got %+v", rep)
	}
	in = rtx5080Like(flashLikeInput(), 32)
	in.Calibration = Calibration{}
	if rep := PlanSplit(in, pcie()); rep.Status != StatusUnknown || rep.Omitted[0] != OmitCalibrationInvalid {
		t.Fatalf("invalid calibration: want calibration_invalid, got %+v", rep)
	}
	// Routed experts with no active-set facts must not be priced as free.
	in = rtx5080Like(flashLikeInput(), 32)
	in.Model.ActiveRoutedBytesPerToken, in.Model.NumExperts, in.Model.ExpertsUsed = 0, 0, 0
	if rep := PlanSplit(in, pcie()); rep.Status != StatusUnknown || rep.Omitted[0] != OmitMoEActiveSetUnknown {
		t.Fatalf("unknown active set: want moe_active_set_unknown, got %+v", rep)
	}
	in = rtx5080Like(flashLikeInput(), 32)
	if rep := PlanSplit(in, Interconnect{Devices: MaxSplitDevices + 1}); rep.Omitted[0] != OmitSplitDevicesOutOfRange || len(rep.Rows) != 0 {
		t.Fatalf("device count over the bound must refuse, got %+v", rep)
	}
	in = rtx5080Like(flashLikeInput(), 32)
	in.Hardware.Topology = TopologyUnified
	rep := planSplit(t, in, pcie())
	multi := 0
	for _, r := range rep.Rows {
		if r.Devices > 1 {
			multi++
			if r.Refusal != SplitRefusalNotDiscrete {
				t.Fatalf("unified pool: a multi-device row must refuse, got %+v", r)
			}
		}
	}
	if multi == 0 {
		t.Fatal("unified pool: want multi-device rows to be produced and refused")
	}
}

// An empty executable list leaves fitting rows unchosen: status unknown with a typed
// omission, never "refused".
func TestSplitNoExecutableIsNotRefused(t *testing.T) {
	ic := pcie()
	ic.Executable = []SplitMode{}
	rep := planSplit(t, rtx5080Like(flashLikeInput(), 32), ic)
	if rep.Chosen != nil || rep.Best == nil || rep.Status != StatusUnknown {
		t.Fatalf("want no choice, a Best and status unknown, got status=%s chosen=%+v", rep.Status, rep.Chosen)
	}
	found := false
	for _, o := range rep.Omitted {
		found = found || o == OmitNoExecutableSplit
	}
	if !found {
		t.Fatalf("want %s, got %v", OmitNoExecutableSplit, rep.Omitted)
	}
}

// Only a rate a phase actually used may label a row generic or demote the tier.
func TestSplitGenericRatesLabelled(t *testing.T) {
	resident := rtx5080Like(baseInput(15*gib, cand(compute.KVPrecisionF32, 4096, true)), 64)
	resident.Calibration = sol()
	rep := planSplit(t, resident, Interconnect{Devices: 1})
	if rep.Tier != TierSOL || rep.Chosen == nil || len(rep.Chosen.GenericRates) != 0 {
		t.Fatalf("resident single device uses no interconnect: want sol and no generic rates, got tier=%s chosen=%+v", rep.Tier, rep.Chosen)
	}
	spilled := rtx5080Like(flashLikeInput(), 32)
	spilled.Calibration = sol()
	rep = planSplit(t, spilled, Interconnect{Devices: 2})
	if rep.Tier != TierEmpirical || rep.Chosen == nil || len(rep.Chosen.GenericRates) == 0 {
		t.Fatalf("expert split on the generic link/sync: want empirical + generic rates, got tier=%s chosen=%+v", rep.Tier, rep.Chosen)
	}
}
