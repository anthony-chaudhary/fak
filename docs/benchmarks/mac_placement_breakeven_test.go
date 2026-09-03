package benchmarks

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/placementtax"
)

const (
	gib = uint64(1 << 30)
)

// MacHostProfile characterizes an Apple Silicon unified memory node.
type MacHostProfile struct {
	Name                string
	UnifiedMemoryBytes  uint64
	MemoryBandwidthGBps float64 // in GB/s
	ComputeUnits        float64 // normalized compute capability (TFLOPS)
	ActiveWatts         float64
	IdleWatts           float64
}

// InterconnectProfile characterizes the inter-host communication link.
type InterconnectProfile struct {
	Name                    string
	BandwidthBytesPerSecond float64
	Latency                 time.Duration
	EnergyJoulesPerByte     float64
}

// WorkloadSpec defines the inference workload envelope.
type WorkloadSpec struct {
	ModelID        string
	Precision      string
	HiddenDim      int
	NumLayers      int
	ParamsTotal    float64 // total parameter count (e.g. 27e9)
	WeightsBytes   uint64
	SequenceLength int
	BatchSize      int
	IsPrefill      bool
	Engine         string
}

// Predefined hardware profiles.
var (
	M3Max128GB = MacHostProfile{
		Name:                "apple-m3-max-128gb",
		UnifiedMemoryBytes:  128 * gib,
		MemoryBandwidthGBps: 400.0,
		ComputeUnits:        40.0, // ~40 TFLOPS FP16
		ActiveWatts:         95.0,
		IdleWatts:           12.0,
	}

	M3Pro48GB = MacHostProfile{
		Name:                "apple-m3-pro-48gb",
		UnifiedMemoryBytes:  48 * gib,
		MemoryBandwidthGBps: 150.0,
		ComputeUnits:        18.0, // ~18 TFLOPS FP16
		ActiveWatts:         45.0,
		IdleWatts:           8.0,
	}

	M3Pro36GB = MacHostProfile{
		Name:                "apple-m3-pro-36gb",
		UnifiedMemoryBytes:  36 * gib,
		MemoryBandwidthGBps: 150.0,
		ComputeUnits:        18.0,
		ActiveWatts:         45.0,
		IdleWatts:           8.0,
	}

	M3Max64GB = MacHostProfile{
		Name:                "apple-m3-max-64gb",
		UnifiedMemoryBytes:  64 * gib,
		MemoryBandwidthGBps: 400.0,
		ComputeUnits:        40.0,
		ActiveWatts:         95.0,
		IdleWatts:           12.0,
	}

	Thunderbolt4Link = InterconnectProfile{
		Name:                    "thunderbolt-4-p2p",
		BandwidthBytesPerSecond: 3.2 * 1e9, // ~25.6 Gbps effective DMA
		Latency:                 20 * time.Microsecond,
		EnergyJoulesPerByte:     8e-9, // ~8 nJ / byte
	}

	RoCE40GLink = InterconnectProfile{
		Name:                    "roce-v2-40g-rdma",
		BandwidthBytesPerSecond: 4.5 * 1e9, // ~36 Gbps effective RDMA
		Latency:                 6 * time.Microsecond,
		EnergyJoulesPerByte:     3e-9, // ~3 nJ / byte
	}

	Ethernet10GLink = InterconnectProfile{
		Name:                    "ethernet-10g",
		BandwidthBytesPerSecond: 1.1 * 1e9, // ~8.8 Gbps effective
		Latency:                 40 * time.Microsecond,
		EnergyJoulesPerByte:     15e-9,
	}
)

// DefaultQwen38_27B returns a representative Qwen3.8-27B workload.
func DefaultQwen38_27B(seqLen, batchSize int, isPrefill bool) WorkloadSpec {
	return WorkloadSpec{
		ModelID:        "Qwen3.8-27B",
		Precision:      "q8_0",
		HiddenDim:      5120,
		NumLayers:      64,
		ParamsTotal:    27e9,
		WeightsBytes:   28 * gib,
		SequenceLength: seqLen,
		BatchSize:      batchSize,
		IsPrefill:      isPrefill,
		Engine:         "fak-native",
	}
}

// computeKVCacheAndActivationBytes calculates memory demands for attention cache and activations.
func computeKVCacheAndActivationBytes(w WorkloadSpec) (uint64, uint64) {
	// GQA 4:1 ratio (e.g. 32 Q heads, 8 KV heads -> hiddenDim / 4)
	kvDim := uint64(w.HiddenDim / 4)
	bytesPerElem := uint64(2) // fp16/bf16 KV cache
	// 2 for Key and Value
	kvCacheBytes := 2 * uint64(w.NumLayers) * uint64(w.SequenceLength) * uint64(w.BatchSize) * kvDim * bytesPerElem
	activationBytes := uint64(w.BatchSize) * uint64(w.SequenceLength) * uint64(w.HiddenDim) * bytesPerElem
	return kvCacheBytes, activationBytes
}

// CalibratePlacement builds a typed placementtax.Comparison for single host vs two-host cluster.
func CalibratePlacement(
	w WorkloadSpec,
	singleHost MacHostProfile,
	clusterHost MacHostProfile,
	link InterconnectProfile,
	candidateIsCluster bool,
) (placementtax.Comparison, error) {
	if w.Engine != "fak-native" {
		return placementtax.Comparison{}, fmt.Errorf("engine %q rejected: fak-native is required for native placement calibration", w.Engine)
	}

	kvCacheBytes, activationBytes := computeKVCacheAndActivationBytes(w)
	singleMemDemand := w.WeightsBytes + kvCacheBytes + activationBytes
	clusterMemDemandPerNode := (w.WeightsBytes + kvCacheBytes + activationBytes) / 2

	// Token volume: for decode step, tokens = batchSize; for prefill, tokens = batchSize * seqLen
	tokens := float64(w.BatchSize)
	if w.IsPrefill {
		tokens = float64(w.BatchSize * w.SequenceLength)
	}

	// Useful compute calculation:
	// Single host:
	singleMemTimeSec := float64(w.WeightsBytes) / (singleHost.MemoryBandwidthGBps * 1e9)
	singleFlops := 2.0 * w.ParamsTotal * tokens
	singleComputeTimeSec := singleFlops / (singleHost.ComputeUnits * 1e12)
	singleUsefulSec := math.Max(singleMemTimeSec, singleComputeTimeSec)
	singleUsefulDur := time.Duration(singleUsefulSec * float64(time.Second))
	singleEnergyJoules := singleUsefulSec * singleHost.ActiveWatts

	// Cluster (2 nodes, TP=2):
	clusterMemTimeSec := float64(w.WeightsBytes/2) / (clusterHost.MemoryBandwidthGBps * 1e9)
	clusterComputeTimeSec := singleFlops / (2.0 * clusterHost.ComputeUnits * 1e12)
	clusterUsefulSec := math.Max(clusterMemTimeSec, clusterComputeTimeSec)
	clusterUsefulDur := time.Duration(clusterUsefulSec * float64(time.Second))
	clusterActiveWatts := 2.0 * clusterHost.ActiveWatts
	clusterEnergyJoules := clusterUsefulSec * clusterActiveWatts

	// Communication calculation:
	// TP=2: 2 All-Reduces per transformer layer (attention out-proj + MLP down-proj)
	messagesPerLink := uint64(w.NumLayers * 2)
	bytesPerMessage := uint64(tokens * float64(w.HiddenDim*2)) // fp16/bf16 activations
	bytesPerLink := messagesPerLink * bytesPerMessage

	// Raw communication duration:
	rawCommSec := float64(messagesPerLink)*link.Latency.Seconds() + float64(bytesPerLink)/link.BandwidthBytesPerSecond

	// Overlap calculation:
	var overlapLat, overlapCyc time.Duration
	if w.IsPrefill && w.SequenceLength >= 512 {
		// Prefill GEMMs allow overlapping communication with subsequent layer compute (up to 25%)
		hiddenSec := math.Min(rawCommSec*0.25, clusterUsefulSec*0.25)
		overlapLat = time.Duration(hiddenSec * float64(time.Second))
		overlapCyc = overlapLat
	}

	// Synchronization & stragglers:
	syncDur := time.Duration(float64(messagesPerLink) * 1.5 * float64(time.Microsecond))
	stragglerDur := time.Duration(clusterUsefulSec * 0.02 * float64(time.Second))

	top := placementtax.Topology{
		Domains: []placementtax.Domain{
			{ID: "single-domain"},
			{ID: "cluster-domain-0"},
			{ID: "cluster-domain-1"},
		},
		Nodes: []placementtax.Node{
			{
				ID:       "single-node",
				DomainID: "single-domain",
				Capacity: placementtax.Capacity{
					MemoryBytes:  singleHost.UnifiedMemoryBytes,
					ComputeUnits: singleHost.ComputeUnits,
				},
			},
			{
				ID:       "cluster-node-0",
				DomainID: "cluster-domain-0",
				Capacity: placementtax.Capacity{
					MemoryBytes:  clusterHost.UnifiedMemoryBytes,
					ComputeUnits: clusterHost.ComputeUnits,
				},
			},
			{
				ID:       "cluster-node-1",
				DomainID: "cluster-domain-1",
				Capacity: placementtax.Capacity{
					MemoryBytes:  clusterHost.UnifiedMemoryBytes,
					ComputeUnits: clusterHost.ComputeUnits,
				},
			},
		},
		Links: []placementtax.Link{
			{
				ID:                      "link-0-to-1",
				FromNode:                "cluster-node-0",
				ToNode:                  "cluster-node-1",
				Latency:                 link.Latency,
				BandwidthBytesPerSecond: link.BandwidthBytesPerSecond,
				MonetaryUSDPerByte:      placementtax.ModeledValue{Value: 0, Modeled: false},
				EnergyJoulesPerByte:     placementtax.ModeledValue{Value: link.EnergyJoulesPerByte, Modeled: true},
			},
			{
				ID:                      "link-1-to-0",
				FromNode:                "cluster-node-1",
				ToNode:                  "cluster-node-0",
				Latency:                 link.Latency,
				BandwidthBytesPerSecond: link.BandwidthBytesPerSecond,
				MonetaryUSDPerByte:      placementtax.ModeledValue{Value: 0, Modeled: false},
				EnergyJoulesPerByte:     placementtax.ModeledValue{Value: link.EnergyJoulesPerByte, Modeled: true},
			},
		},
	}

	singlePlacement := placementtax.Placement{
		Name: singleHost.Name,
		Allocations: []placementtax.Allocation{
			{
				NodeID: "single-node",
				Demand: placementtax.Capacity{
					MemoryBytes:  singleMemDemand,
					ComputeUnits: math.Min(10.0, singleHost.ComputeUnits*0.5),
				},
			},
		},
		UsefulCompute: placementtax.ComponentCost{
			Latency:      singleUsefulDur,
			Cycle:        singleUsefulDur,
			EnergyJoules: placementtax.ModeledValue{Value: singleEnergyJoules, Modeled: true},
			Provenance:   placementtax.ProvenanceEstimated,
		},
	}

	clusterPlacement := placementtax.Placement{
		Name: fmt.Sprintf("cluster-2x-%s", clusterHost.Name),
		Allocations: []placementtax.Allocation{
			{
				NodeID: "cluster-node-0",
				Demand: placementtax.Capacity{
					MemoryBytes:  clusterMemDemandPerNode,
					ComputeUnits: math.Min(5.0, clusterHost.ComputeUnits*0.5),
				},
			},
			{
				NodeID: "cluster-node-1",
				Demand: placementtax.Capacity{
					MemoryBytes:  clusterMemDemandPerNode,
					ComputeUnits: math.Min(5.0, clusterHost.ComputeUnits*0.5),
				},
			},
		},
		UsefulCompute: placementtax.ComponentCost{
			Latency:      clusterUsefulDur,
			Cycle:        clusterUsefulDur,
			EnergyJoules: placementtax.ModeledValue{Value: clusterEnergyJoules, Modeled: true},
			Provenance:   placementtax.ProvenanceEstimated,
		},
		Transfers: []placementtax.Transfer{
			{
				LinkID:   "link-0-to-1",
				Messages: messagesPerLink,
				Bytes:    bytesPerLink,
				Overlap: placementtax.Overlap{
					Latency: overlapLat,
					Cycle:   overlapCyc,
				},
				Provenance: placementtax.ProvenanceEstimated,
			},
			{
				LinkID:   "link-1-to-0",
				Messages: messagesPerLink,
				Bytes:    bytesPerLink,
				Overlap: placementtax.Overlap{
					Latency: overlapLat,
					Cycle:   overlapCyc,
				},
				Provenance: placementtax.ProvenanceEstimated,
			},
		},
		Synchronization: placementtax.ComponentCost{
			Latency:      syncDur,
			Cycle:        syncDur,
			EnergyJoules: placementtax.ModeledValue{Value: 0, Modeled: true},
			Provenance:   placementtax.ProvenanceEstimated,
		},
		ImbalanceStraggler: placementtax.ComponentCost{
			Latency:      stragglerDur,
			Cycle:        stragglerDur,
			EnergyJoules: placementtax.ModeledValue{Value: 0, Modeled: true},
			Provenance:   placementtax.ProvenanceEstimated,
		},
	}

	comp := placementtax.Comparison{
		Workload: placementtax.Workload{
			ID:    fmt.Sprintf("%s-s%d-b%d", w.ModelID, w.SequenceLength, w.BatchSize),
			Units: float64(w.BatchSize),
			Unit:  "tokens",
			Quality: placementtax.QualityEnvelope{
				ModelID:        w.ModelID,
				Precision:      w.Precision,
				SequenceLength: w.SequenceLength,
				BatchSize:      w.BatchSize,
				Target:         fmt.Sprintf("engine=%s;quality=exact;loss_delta=0", w.Engine),
			},
		},
		Topology: top,
	}

	if candidateIsCluster {
		comp.Candidate = clusterPlacement
		comp.Reference = singlePlacement
	} else {
		comp.Candidate = singlePlacement
		comp.Reference = clusterPlacement
	}

	return comp, nil
}

// FindBreakEvenBatch sweeps batch size to find where Candidate latency matches or beats Reference latency.
func FindBreakEvenBatch(
	baseWorkload WorkloadSpec,
	singleHost MacHostProfile,
	clusterHost MacHostProfile,
	link InterconnectProfile,
	maxBatch int,
) (int, bool) {
	for b := 1; b <= maxBatch; b *= 2 {
		w := baseWorkload
		w.BatchSize = b
		comp, err := CalibratePlacement(w, singleHost, clusterHost, link, true)
		if err != nil {
			continue
		}
		report, err := placementtax.Analyze(comp)
		if err != nil || !report.Candidate.Feasibility.Feasible || !report.Reference.Feasibility.Feasible {
			continue
		}
		// Candidate (cluster) beats Reference (single) if Delta.Latency <= 0
		if report.Delta != nil && report.Delta.Latency <= 0 {
			return b, true
		}
	}
	return -1, false
}

// FindBreakEvenSequenceLength sweeps sequence length to find where Candidate beats Reference latency.
func FindBreakEvenSequenceLength(
	baseWorkload WorkloadSpec,
	singleHost MacHostProfile,
	clusterHost MacHostProfile,
	link InterconnectProfile,
	seqLengths []int,
) (int, bool) {
	for _, s := range seqLengths {
		w := baseWorkload
		w.SequenceLength = s
		comp, err := CalibratePlacement(w, singleHost, clusterHost, link, true)
		if err != nil {
			continue
		}
		report, err := placementtax.Analyze(comp)
		if err != nil || !report.Candidate.Feasibility.Feasible || !report.Reference.Feasibility.Feasible {
			continue
		}
		if report.Delta != nil && report.Delta.Latency <= 0 {
			return s, true
		}
	}
	return -1, false
}

func TestMacPlacementBreakeven(t *testing.T) {
	t.Run("EngineBindingFakNative", func(t *testing.T) {
		w := DefaultQwen38_27B(2048, 1, false)
		w.Engine = "llamacpp" // unauthorized external fallback

		_, err := CalibratePlacement(w, M3Max128GB, M3Pro48GB, Thunderbolt4Link, true)
		if err == nil {
			t.Fatal("expected error when engine != fak-native, got nil")
		}
		if !strings.Contains(err.Error(), "fak-native is required") {
			t.Fatalf("unexpected error message: %v", err)
		}
	})

	t.Run("LowBatchDecodeSingleHostWins", func(t *testing.T) {
		// Single M3 Max 128GB (400 GB/s) vs 2x M3 Pro 48GB (150 GB/s each, TB4 link) at batch=1 decode
		w := DefaultQwen38_27B(2048, 1, false)
		comp, err := CalibratePlacement(w, M3Max128GB, M3Pro48GB, Thunderbolt4Link, true)
		if err != nil {
			t.Fatalf("CalibratePlacement failed: %v", err)
		}

		report, err := placementtax.Analyze(comp)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		if !report.Candidate.Feasibility.Feasible || !report.Reference.Feasibility.Feasible {
			t.Fatalf("both placements should be feasible: candidate=%+v reference=%+v",
				report.Candidate.Feasibility, report.Reference.Feasibility)
		}

		// Candidate is 2x M3 Pro, Reference is Single M3 Max 128GB.
		// Delta is Candidate - Reference.
		// Single host wins: Candidate latency > Reference latency -> Delta.Latency > 0
		if report.Delta.Latency <= 0 {
			t.Fatalf("expected candidate (2x M3 Pro) latency > reference (M3 Max) for low-batch decode, got delta=%v", report.Delta.Latency)
		}

		// Penalty ratio > 1, efficiency < 1
		if report.Relative.Latency.PenaltyRatio <= 1.0 {
			t.Fatalf("expected penalty ratio > 1.0 for two-host cluster at batch=1, got %v", report.Relative.Latency.PenaltyRatio)
		}
		if report.Relative.Latency.PlacementEfficiency >= 1.0 {
			t.Fatalf("expected placement efficiency < 1.0, got %v", report.Relative.Latency.PlacementEfficiency)
		}

		// Verify communication tax is properly recorded
		commLedger := report.Candidate.Communication
		if len(commLedger) != 2 {
			t.Fatalf("expected 2 directional communication records, got %d", len(commLedger))
		}
		totalCommExposed := commLedger[0].ExposedLatency + commLedger[1].ExposedLatency
		if totalCommExposed <= 0 {
			t.Fatalf("expected positive exposed communication latency, got %v", totalCommExposed)
		}
	})

	t.Run("TwoHostWinsInHighBatchComputeBound", func(t *testing.T) {
		// Near-matched nodes: 2x M3 Pro 48GB vs 1x M3 Pro 48GB in compute-bound prefill (SeqLen=4096, Batch=8)
		w := DefaultQwen38_27B(4096, 8, true)
		comp, err := CalibratePlacement(w, M3Pro48GB, M3Pro48GB, Thunderbolt4Link, true)
		if err != nil {
			t.Fatalf("CalibratePlacement failed: %v", err)
		}

		report, err := placementtax.Analyze(comp)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		if !report.Candidate.Feasibility.Feasible || !report.Reference.Feasibility.Feasible {
			t.Fatalf("both placements should be feasible: candidate=%+v reference=%+v",
				report.Candidate.Feasibility, report.Reference.Feasibility)
		}

		// In compute-bound prefill with matched nodes, 2x M3 Pro has 2x the compute capability and aggregate memory bandwidth.
		// Candidate (2x M3 Pro) latency < Reference (1x M3 Pro) latency -> Delta.Latency < 0
		if report.Delta.Latency >= 0 {
			t.Fatalf("expected 2x M3 Pro to beat 1x M3 Pro in high-batch prefill, got delta=%v", report.Delta.Latency)
		}
		if report.Relative.Latency.PenaltyRatio >= 1.0 {
			t.Fatalf("expected penalty ratio < 1.0, got %v", report.Relative.Latency.PenaltyRatio)
		}
		if report.Relative.Latency.PlacementEfficiency <= 1.0 {
			t.Fatalf("expected placement efficiency > 1.0, got %v", report.Relative.Latency.PlacementEfficiency)
		}
		if report.Delta.Throughput <= 0 {
			t.Fatalf("expected positive throughput delta for 2-host placement, got %v", report.Delta.Throughput)
		}
	})

	t.Run("BreakEvenThresholdDerivation", func(t *testing.T) {
		// Sweep sequence lengths and determine that break-even batch size exists and behaves consistently.
		seqLengths := []int{512, 1024, 2048, 4096}
		w := DefaultQwen38_27B(2048, 1, true)

		// Compare 2x M3 Pro vs 1x M3 Max 128GB
		breakEvenSeq, foundSeq := FindBreakEvenSequenceLength(w, M3Max128GB, M3Pro48GB, Thunderbolt4Link, seqLengths)
		// Even if 2x M3 Pro doesn't beat M3 Max at batch=1, check 2x M3 Pro vs 1x M3 Pro
		breakEvenProSeq, foundProSeq := FindBreakEvenSequenceLength(w, M3Pro48GB, M3Pro48GB, Thunderbolt4Link, seqLengths)
		if !foundProSeq {
			t.Fatalf("expected break-even sequence length for 2x M3 Pro vs 1x M3 Pro, got none")
		}
		if breakEvenProSeq > 2048 {
			t.Fatalf("expected break-even sequence length <= 2048 for 2x M3 Pro vs 1x M3 Pro, got %d", breakEvenProSeq)
		}

		// Also check batch sweep for prefill
		wPrefill := DefaultQwen38_27B(2048, 1, true)
		breakEvenBatch, foundBatch := FindBreakEvenBatch(wPrefill, M3Pro48GB, M3Pro48GB, Thunderbolt4Link, 32)
		if !foundBatch {
			t.Fatalf("expected break-even batch size for 2x M3 Pro vs 1x M3 Pro, got none")
		}
		if breakEvenBatch < 1 || breakEvenBatch > 16 {
			t.Fatalf("break-even batch %d out of expected range [1, 16]", breakEvenBatch)
		}
		_ = breakEvenSeq
		_ = foundSeq
	})

	t.Run("InterconnectLinkSensitivity", func(t *testing.T) {
		// RoCE-RDMA (6µs latency, 4.5 GB/s) vs TB4 (20µs latency, 3.2 GB/s) vs 10GbE (40µs, 1.1 GB/s)
		w := DefaultQwen38_27B(2048, 4, true)

		compRoCE, err := CalibratePlacement(w, M3Pro48GB, M3Pro48GB, RoCE40GLink, true)
		if err != nil {
			t.Fatal(err)
		}
		reportRoCE, err := placementtax.Analyze(compRoCE)
		if err != nil {
			t.Fatal(err)
		}

		compTB4, err := CalibratePlacement(w, M3Pro48GB, M3Pro48GB, Thunderbolt4Link, true)
		if err != nil {
			t.Fatal(err)
		}
		reportTB4, err := placementtax.Analyze(compTB4)
		if err != nil {
			t.Fatal(err)
		}

		compEth, err := CalibratePlacement(w, M3Pro48GB, M3Pro48GB, Ethernet10GLink, true)
		if err != nil {
			t.Fatal(err)
		}
		reportEth, err := placementtax.Analyze(compEth)
		if err != nil {
			t.Fatal(err)
		}

		roceExposed := reportRoCE.Candidate.Communication[0].ExposedLatency
		tb4Exposed := reportTB4.Candidate.Communication[0].ExposedLatency
		ethExposed := reportEth.Candidate.Communication[0].ExposedLatency

		if roceExposed >= tb4Exposed {
			t.Fatalf("expected RoCE exposed latency (%v) < TB4 exposed latency (%v)", roceExposed, tb4Exposed)
		}
		if tb4Exposed >= ethExposed {
			t.Fatalf("expected TB4 exposed latency (%v) < 10GbE exposed latency (%v)", tb4Exposed, ethExposed)
		}
	})

	t.Run("MemoryInfeasibilityGating", func(t *testing.T) {
		// Scenario 1: Model demand exceeds 1x M3 Pro 36GB capacity, but fits on 2x M3 Pro 36GB (72GB total)
		// Let weights = 40GB (exceeds 36GB)
		wLarge := DefaultQwen38_27B(2048, 1, false)
		wLarge.WeightsBytes = 40 * gib

		comp, err := CalibratePlacement(wLarge, M3Pro36GB, M3Pro36GB, Thunderbolt4Link, true)
		if err != nil {
			t.Fatal(err)
		}
		report, err := placementtax.Analyze(comp)
		if err != nil {
			t.Fatal(err)
		}

		// Candidate (2x M3 Pro 36GB) should be feasible because each node gets 40GB/2 + KV/2 = ~22GB <= 36GB
		if !report.Candidate.Feasibility.Feasible {
			t.Fatalf("candidate should be feasible on 2x M3 Pro 36GB: %+v", report.Candidate.Feasibility)
		}
		// Reference (1x M3 Pro 36GB) must be infeasible (demand > 36GB)
		if report.Reference.Feasibility.Feasible {
			t.Fatalf("reference single host 36GB should be infeasible for 40GB+ demand")
		}
		// Deltas and relative metrics must be nil
		if report.Delta != nil || report.Relative != nil {
			t.Fatalf("infeasible reference must refuse deltas and ratios: delta=%+v relative=%+v", report.Delta, report.Relative)
		}
		refusalReason := strings.Join(report.Reference.Feasibility.Reasons, "; ")
		if !strings.Contains(refusalReason, "demand exceeds capacity") {
			t.Fatalf("expected capacity refusal reason, got: %s", refusalReason)
		}
	})

	t.Run("PlacementTaxComponentLedger", func(t *testing.T) {
		w := DefaultQwen38_27B(2048, 2, false)
		comp, err := CalibratePlacement(w, M3Max128GB, M3Pro48GB, Thunderbolt4Link, true)
		if err != nil {
			t.Fatal(err)
		}
		report, err := placementtax.Analyze(comp)
		if err != nil {
			t.Fatal(err)
		}

		ledger := report.Candidate.Ledger
		if len(ledger) != 6 {
			t.Fatalf("expected 6 ledger entries, got %d", len(ledger))
		}

		expectedKinds := map[placementtax.ComponentKind]bool{
			placementtax.ComponentUsefulCompute:         false,
			placementtax.ComponentExposedCommunication:  false,
			placementtax.ComponentSynchronization:       false,
			placementtax.ComponentImbalanceStraggler:    false,
			placementtax.ComponentDataMovement:          false,
			placementtax.ComponentOrchestrationRecovery: false,
		}

		for _, entry := range ledger {
			if _, ok := expectedKinds[entry.Kind]; !ok {
				t.Fatalf("unexpected ledger entry kind: %v", entry.Kind)
			}
			expectedKinds[entry.Kind] = true
		}

		for kind, seen := range expectedKinds {
			if !seen {
				t.Fatalf("missing expected ledger kind: %v", kind)
			}
		}
	})

	t.Run("PowerAndEnergyAccounting", func(t *testing.T) {
		w := DefaultQwen38_27B(2048, 1, false)
		comp, err := CalibratePlacement(w, M3Max128GB, M3Pro48GB, Thunderbolt4Link, true)
		if err != nil {
			t.Fatal(err)
		}
		report, err := placementtax.Analyze(comp)
		if err != nil {
			t.Fatal(err)
		}

		candEnergy := report.Candidate.Metrics.EnergyJoules
		refEnergy := report.Reference.Metrics.EnergyJoules

		if !candEnergy.Modeled || !refEnergy.Modeled {
			t.Fatalf("energy must be modeled on both sides: cand=%+v ref=%+v", candEnergy, refEnergy)
		}
		if candEnergy.Value <= 0 || refEnergy.Value <= 0 {
			t.Fatalf("energy values must be positive: cand=%v ref=%v", candEnergy.Value, refEnergy.Value)
		}
		if !report.Delta.EnergyJoules.Modeled {
			t.Fatalf("energy delta must be modeled")
		}
	})
}
