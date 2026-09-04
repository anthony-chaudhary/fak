package nativeperf

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
)

// Schema identifier for native byte reconciliation ledger.
const ByteLedgerSchema = "fak-native-byte-reconciliation/v1"

// PhysicalMemoryTier identifies a hardware residency or transport layer.
type PhysicalMemoryTier string

const (
	TierSRAM     PhysicalMemoryTier = "sram"      // On-chip SRAM / L1 / register file / shared memory
	TierL2       PhysicalMemoryTier = "l2_slc"    // On-chip L2 / System Level Cache (SLC)
	TierHBM      PhysicalMemoryTier = "hbm"       // High-Bandwidth Memory (GPU device HBM)
	TierVRAM     PhysicalMemoryTier = "vram"      // Video RAM / unified device memory
	TierHostDRAM PhysicalMemoryTier = "host_dram" // Host system main memory (DDR4/DDR5)
	TierPCIe     PhysicalMemoryTier = "pcie_bus"  // PCIe / NVLink / system interconnect bus
	TierStorage  PhysicalMemoryTier = "storage"   // NVMe / SSD disk offload tier
)

var validPhysicalMemoryTiers = map[PhysicalMemoryTier]bool{
	TierSRAM:     true,
	TierL2:       true,
	TierHBM:      true,
	TierVRAM:     true,
	TierHostDRAM: true,
	TierPCIe:     true,
	TierStorage:  true,
}

// ProvenanceKind identifies the origin and certainty of a byte metric.
type ProvenanceKind string

const (
	ProvenanceMeasured  ProvenanceKind = "measured"  // Hardware performance counter or hardware monitor
	ProvenanceEstimated ProvenanceKind = "estimated" // Analytical model / dimensional derivation
	ProvenanceDeclared  ProvenanceKind = "declared"  // Static parameter / configuration / specification
)

// ByteValue wraps an accounted byte quantity with provenance and measurement error/uncertainty.
type ByteValue struct {
	Bytes        uint64         `json:"bytes"`
	Provenance   ProvenanceKind `json:"provenance"`
	Uncertainty  float64        `json:"uncertainty_percent,omitempty"` // Margin of error (e.g. ±2.5%)
	SourceDetail string         `json:"source_detail,omitempty"`       // Name of counter or estimator formula
}

// LogicalByteBreakdown accounts for the theoretical/algorithmic bytes required
// by an inference pass across weights, quant metadata, KV cache, activations, etc.
type LogicalByteBreakdown struct {
	Weights             ByteValue `json:"weights"`              // Model base parameters
	QuantMetadata       ByteValue `json:"quant_metadata"`       // Scales, zero-points, codebooks, bias
	KVCache             ByteValue `json:"kv_cache"`             // Key-value attention state
	RecurrentState      ByteValue `json:"recurrent_state"`      // SSM/RNN/GDN hybrid state (if present)
	ActivationWorkspace ByteValue `json:"activation_workspace"` // Intermediate activations & workspace
	CopiesStaging       ByteValue `json:"copies_staging"`       // Software staging buffers / host copies
}

// TotalBytes returns the sum of all logical component bytes.
func (l LogicalByteBreakdown) TotalBytes() uint64 {
	return l.Weights.Bytes +
		l.QuantMetadata.Bytes +
		l.KVCache.Bytes +
		l.RecurrentState.Bytes +
		l.ActivationWorkspace.Bytes +
		l.CopiesStaging.Bytes
}

// PhysicalTierTraffic accounts for observed or modeled hardware traffic at one memory tier.
type PhysicalTierTraffic struct {
	Tier       PhysicalMemoryTier `json:"tier"`
	ReadBytes  ByteValue          `json:"read_bytes"`
	WriteBytes ByteValue          `json:"write_bytes"`
	HitBytes   ByteValue          `json:"hit_bytes,omitempty"`  // Cache hit bytes avoided from lower tiers
	MissBytes  ByteValue          `json:"miss_bytes,omitempty"` // Cache miss bytes serviced by lower tiers
	HitRate    *float64           `json:"hit_rate,omitempty"`   // hit_bytes / (hit_bytes + miss_bytes)
}

// TotalBytes returns the sum of physical read and write bytes.
func (p PhysicalTierTraffic) TotalBytes() uint64 {
	return p.ReadBytes.Bytes + p.WriteBytes.Bytes
}

// PhaseByteLedger reconciles logical demand with physical traffic for a single bounded phase.
type PhaseByteLedger struct {
	Phase                Phase                 `json:"phase"`
	Logical              LogicalByteBreakdown  `json:"logical"`
	PhysicalTiers        []PhysicalTierTraffic `json:"physical_tiers"`
	TotalLogicalBytes    uint64                `json:"total_logical_bytes"`
	TotalPhysicalBytes   uint64                `json:"total_physical_bytes"`
	PrimaryTier          PhysicalMemoryTier    `json:"primary_tier"`
	AmplificationFactor  float64               `json:"amplification_factor"`
	ResidualUnknownBytes int64                 `json:"residual_unknown_bytes"`
	UnexplainedPercent   float64               `json:"unexplained_percent"`
	Reconciled           bool                  `json:"reconciled"`
	Notes                []string              `json:"notes,omitempty"`
}

// ByteReconciliationReceipt records the complete cross-phase byte reconciliation.
type ByteReconciliationReceipt struct {
	Schema              string            `json:"schema"`
	Engine              string            `json:"engine"`
	Backend             string            `json:"backend"`
	ForwardPath         string            `json:"forward_path"`
	FallbackActive      bool              `json:"fallback_active"`
	Phases              []PhaseByteLedger `json:"phases"`
	TotalLogicalBytes   uint64            `json:"total_logical_bytes"`
	TotalPhysicalBytes  uint64            `json:"total_physical_bytes"`
	GlobalAmplification float64           `json:"global_amplification"`
	UnexplainedBytes    int64             `json:"unexplained_bytes"`
	ConservationPassed  bool              `json:"conservation_passed"`
}

// ReconcilePhaseBytes computes the reconciliation between logical bytes and physical tier traffic
// for a single phase, evaluating amplification factor, cache service, and residual discrepancy.
//
// amplification_factor = physical_bytes / logical_bytes.
// When logical_bytes == 0 and physical_bytes == 0, amplification_factor is 1.0 (neutral baseline).
// If logical_bytes == 0 and physical_bytes > 0, amplification_factor is +Inf or a bounded ceiling.
//
// primaryTier designates the dominant hardware tier (e.g., TierHBM or TierVRAM for GPU, TierHostDRAM for CPU).
//
// tolerancePercent defines the allowable discrepancy between logical demand (+ expected amplification)
// and physical measurements before flagged as unexplained residual.
func ReconcilePhaseBytes(phase Phase, logical LogicalByteBreakdown, tiers []PhysicalTierTraffic, primaryTier PhysicalMemoryTier, tolerancePercent float64) (PhaseByteLedger, error) {
	if !validPhysicalMemoryTiers[primaryTier] {
		return PhaseByteLedger{}, fmt.Errorf("invalid primary tier: %q", primaryTier)
	}

	totLogical := logical.TotalBytes()
	var totPhysical uint64
	var primaryPhysical uint64
	var notes []string

	sortedTiers := make([]PhysicalTierTraffic, len(tiers))
	copy(sortedTiers, tiers)
	sort.Slice(sortedTiers, func(i, j int) bool {
		return sortedTiers[i].Tier < sortedTiers[j].Tier
	})

	for i := range sortedTiers {
		t := &sortedTiers[i]
		if !validPhysicalMemoryTiers[t.Tier] {
			return PhaseByteLedger{}, fmt.Errorf("invalid tier in traffic: %q", t.Tier)
		}
		// Validate and derive hit rate if hits/misses are provided
		totAccess := t.HitBytes.Bytes + t.MissBytes.Bytes
		if totAccess > 0 {
			rate := float64(t.HitBytes.Bytes) / float64(totAccess)
			t.HitRate = &rate
		}

		tierTot := t.TotalBytes()
		totPhysical += tierTot
		if t.Tier == primaryTier {
			primaryPhysical += tierTot
		}
	}

	// Calculate amplification factor on the primary tier (or total physical if primary has 0)
	comparisonPhysical := primaryPhysical
	if comparisonPhysical == 0 && totPhysical > 0 {
		comparisonPhysical = totPhysical
	}

	var amp float64
	if totLogical == 0 {
		if comparisonPhysical == 0 {
			amp = 1.0
		} else {
			amp = math.Inf(1)
		}
	} else {
		amp = float64(comparisonPhysical) / float64(totLogical)
	}

	// Residual calculation: physical tier bytes minus logical bytes
	// Positive residual means amplification (overfetch, cache line waste, re-reads, bank conflicts).
	// Negative residual means cache filtering / reuse where physical < logical demand.
	residual := int64(comparisonPhysical) - int64(totLogical)
	var unexplainedPct float64
	if totLogical > 0 {
		unexplainedPct = (math.Abs(float64(residual)) / float64(totLogical)) * 100.0
	} else if comparisonPhysical > 0 {
		unexplainedPct = 100.0
	}

	// Check conservation / tolerance
	reconciled := true
	if tolerancePercent > 0 && unexplainedPct > tolerancePercent {
		// If unexplained variation exceeds tolerance, note it
		notes = append(notes, fmt.Sprintf("residual variation %.2f%% exceeds tolerance threshold %.2f%%", unexplainedPct, tolerancePercent))
		reconciled = false
	}

	return PhaseByteLedger{
		Phase:                phase,
		Logical:              logical,
		PhysicalTiers:        sortedTiers,
		TotalLogicalBytes:    totLogical,
		TotalPhysicalBytes:   totPhysical,
		PrimaryTier:          primaryTier,
		AmplificationFactor:  amp,
		ResidualUnknownBytes: residual,
		UnexplainedPercent:   unexplainedPct,
		Reconciled:           reconciled,
		Notes:                notes,
	}, nil
}

// BuildByteReconciliationReceipt aggregates phase-level byte ledgers into an overall receipt.
func BuildByteReconciliationReceipt(engine, backend, forwardPath string, fallbackActive bool, phases []PhaseByteLedger, maxAllowedUnexplainedPercent float64) (ByteReconciliationReceipt, error) {
	if engine != "fak-native" {
		return ByteReconciliationReceipt{}, fmt.Errorf("byte reconciliation requires fak-native engine, got %q", engine)
	}
	if fallbackActive {
		return ByteReconciliationReceipt{}, errors.New("byte reconciliation rejects fallback-active execution")
	}
	if len(phases) == 0 {
		return ByteReconciliationReceipt{}, errors.New("byte reconciliation receipt requires at least one phase ledger")
	}

	var totalLogical uint64
	var totalPhysical uint64
	conservationPassed := true

	for _, p := range phases {
		totalLogical += p.TotalLogicalBytes
		totalPhysical += p.TotalPhysicalBytes
		if !p.Reconciled {
			conservationPassed = false
		}
	}

	var globalAmp float64
	if totalLogical == 0 {
		if totalPhysical == 0 {
			globalAmp = 1.0
		} else {
			globalAmp = math.Inf(1)
		}
	} else {
		globalAmp = float64(totalPhysical) / float64(totalLogical)
	}

	totalResidual := int64(totalPhysical) - int64(totalLogical)
	if maxAllowedUnexplainedPercent > 0 && totalLogical > 0 {
		overallPct := (math.Abs(float64(totalResidual)) / float64(totalLogical)) * 100.0
		if overallPct > maxAllowedUnexplainedPercent {
			conservationPassed = false
		}
	}

	return ByteReconciliationReceipt{
		Schema:              ByteLedgerSchema,
		Engine:              engine,
		Backend:             backend,
		ForwardPath:         forwardPath,
		FallbackActive:      fallbackActive,
		Phases:              phases,
		TotalLogicalBytes:   totalLogical,
		TotalPhysicalBytes:  totalPhysical,
		GlobalAmplification: globalAmp,
		UnexplainedBytes:    totalResidual,
		ConservationPassed:  conservationPassed,
	}, nil
}

// ValidateReconciliationReceipt validates all required fields, non-negative bounds, and conservation invariants.
func (r ByteReconciliationReceipt) Validate(maxDiscrepancyPercent float64) error {
	if r.Schema != ByteLedgerSchema {
		return fmt.Errorf("invalid schema %q, want %q", r.Schema, ByteLedgerSchema)
	}
	if r.Engine != "fak-native" {
		return fmt.Errorf("engine must be fak-native, got %q", r.Engine)
	}
	if r.FallbackActive {
		return errors.New("reconciliation receipt has fallback active")
	}
	if len(r.Phases) == 0 {
		return errors.New("receipt contains no phases")
	}

	var summedLogical uint64
	var summedPhysical uint64

	seenPhases := make(map[Phase]bool)
	for _, p := range r.Phases {
		if seenPhases[p.Phase] {
			return fmt.Errorf("duplicate phase in receipt: %q", p.Phase)
		}
		seenPhases[p.Phase] = true

		if !validPhysicalMemoryTiers[p.PrimaryTier] {
			return fmt.Errorf("phase %q has invalid primary tier %q", p.Phase, p.PrimaryTier)
		}
		if p.TotalLogicalBytes != p.Logical.TotalBytes() {
			return fmt.Errorf("phase %q TotalLogicalBytes %d does not match sum of components %d", p.Phase, p.TotalLogicalBytes, p.Logical.TotalBytes())
		}

		var tierPhysical uint64
		for _, t := range p.PhysicalTiers {
			if !validPhysicalMemoryTiers[t.Tier] {
				return fmt.Errorf("phase %q has invalid tier %q", p.Phase, t.Tier)
			}
			tierPhysical += t.TotalBytes()
		}
		if p.TotalPhysicalBytes != tierPhysical {
			return fmt.Errorf("phase %q TotalPhysicalBytes %d does not match sum of tiers %d", p.Phase, p.TotalPhysicalBytes, tierPhysical)
		}

		summedLogical += p.TotalLogicalBytes
		summedPhysical += p.TotalPhysicalBytes
	}

	if r.TotalLogicalBytes != summedLogical {
		return fmt.Errorf("receipt TotalLogicalBytes %d does not match sum of phases %d", r.TotalLogicalBytes, summedLogical)
	}
	if r.TotalPhysicalBytes != summedPhysical {
		return fmt.Errorf("receipt TotalPhysicalBytes %d does not match sum of phases %d", r.TotalPhysicalBytes, summedPhysical)
	}

	return nil
}

// MarshalJSON encodes ByteReconciliationReceipt to indented JSON bytes.
func (r ByteReconciliationReceipt) MarshalJSON() ([]byte, error) {
	type Alias ByteReconciliationReceipt
	return json.MarshalIndent(Alias(r), "", "  ")
}
