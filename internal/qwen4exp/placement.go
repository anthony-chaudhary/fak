package qwen4exp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// ManagedComponent identifies model state whose placement must be admitted explicitly.
type ManagedComponent string

const (
	NGram3PLEEmbeddings  ManagedComponent = "ngram-3-ple-embeddings"
	SparseAttentionIndex ManagedComponent = "learned-sparse-attention-index"
)

// PlacementTier is the physical tier holding a managed component.
type PlacementTier string

const (
	TierGPU PlacementTier = "gpu"
	TierCPU PlacementTier = "cpu"
	TierSSD PlacementTier = "ssd"
)

const FakNativeEngine = "fak-native"

// TierCapacity is operator-declared capacity. Admission never changes tiers to make a plan fit.
type TierCapacity struct {
	Tier          PlacementTier `json:"tier"`
	CapacityBytes uint64        `json:"capacity_bytes"`
}

// TrafficEvidence binds admission to observed demand and a measured service envelope.
type TrafficEvidence struct {
	Window                         time.Duration `json:"window"`
	ObservedLookups                uint64        `json:"observed_lookups"`
	PeakLookupsPerSecond           uint64        `json:"peak_lookups_per_second"`
	MeasuredSustainableLookupsRate uint64        `json:"measured_sustainable_lookups_per_second"`
}

// MatchedComparison records caller-supplied measurements for the same prompts in memory and offload.
type MatchedComparison struct {
	PromptSetDigest       string        `json:"prompt_set_digest"`
	InMemoryTier          PlacementTier `json:"in_memory_tier"`
	InMemoryLookupLatency time.Duration `json:"in_memory_lookup_latency"`
	OffloadLookupLatency  time.Duration `json:"offload_lookup_latency"`
	InMemoryPeakBytes     uint64        `json:"in_memory_peak_bytes"`
	OffloadPeakBytes      uint64        `json:"offload_peak_bytes"`
}

// PlacementEvidence is measured evidence supplied by the caller. No field is synthesized as a
// hardware measurement by AdmitPlacement.
type PlacementEvidence struct {
	Component ManagedComponent `json:"component"`
	Tier      PlacementTier    `json:"tier"`

	LogicalBytes    uint64 `json:"logical_bytes"`
	PhysicalBytes   uint64 `json:"physical_bytes"`
	PeakMemoryBytes uint64 `json:"peak_memory_bytes"`

	PageBehavior  string `json:"page_behavior"`
	CacheBehavior string `json:"cache_behavior"`

	LoadLatency   time.Duration `json:"load_latency"`
	LookupLatency time.Duration `json:"lookup_latency"`

	QualityMetric     string  `json:"quality_metric"`
	InMemoryQuality   float64 `json:"in_memory_quality"`
	PlacedQuality     float64 `json:"placed_quality"`
	QualityEquivalent bool    `json:"quality_equivalent"`
	QualityTolerance  float64 `json:"quality_tolerance"`

	Traffic TrafficEvidence   `json:"traffic"`
	Matched MatchedComparison `json:"matched_in_memory_offload_comparison"`

	FailureDetectionLatency time.Duration `json:"failure_detection_latency"`
	RecoveryLatency         time.Duration `json:"recovery_latency"`
	RecoveryReadBytes       uint64        `json:"recovery_read_bytes"`
}

// PlacementRequest is the complete, explicit admission request.
type PlacementRequest struct {
	Engine     string
	Capacities []TierCapacity
	Components []PlacementEvidence
}

type receiptDocument struct {
	Schema     string              `json:"schema"`
	Engine     string              `json:"engine"`
	Capacities []TierCapacity      `json:"capacities"`
	Components []PlacementEvidence `json:"components"`
}

// PlacementReceipt is immutable from outside this package. Accessors return copies.
type PlacementReceipt struct {
	document receiptDocument
	encoded  []byte
	digest   string
}

// AdmitPlacement validates declared capacity and measured traffic, then creates a deterministic receipt.
func AdmitPlacement(req PlacementRequest) (PlacementReceipt, error) {
	if req.Engine != FakNativeEngine {
		return PlacementReceipt{}, fmt.Errorf("placement: engine %q rejected; only %q is admitted", req.Engine, FakNativeEngine)
	}

	capacities, capacityByTier, err := validateCapacities(req.Capacities)
	if err != nil {
		return PlacementReceipt{}, err
	}
	components, err := validateComponents(req.Components, capacityByTier)
	if err != nil {
		return PlacementReceipt{}, err
	}

	doc := receiptDocument{
		Schema:     "fak/qwen4exp-managed-placement/v1",
		Engine:     req.Engine,
		Capacities: capacities,
		Components: components,
	}
	encoded, err := json.Marshal(doc)
	if err != nil {
		return PlacementReceipt{}, fmt.Errorf("placement: encode receipt: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return PlacementReceipt{document: doc, encoded: encoded, digest: "sha256:" + hex.EncodeToString(sum[:])}, nil
}

// JSON returns a copy of the canonical deterministic receipt encoding.
func (r PlacementReceipt) JSON() []byte { return append([]byte(nil), r.encoded...) }

// Digest identifies the exact canonical receipt.
func (r PlacementReceipt) Digest() string { return r.digest }

// Components returns a copy in deterministic component order.
func (r PlacementReceipt) Components() []PlacementEvidence {
	return append([]PlacementEvidence(nil), r.document.Components...)
}

func validateCapacities(in []TierCapacity) ([]TierCapacity, map[PlacementTier]uint64, error) {
	if len(in) == 0 {
		return nil, nil, errors.New("placement: declared tier capacity is required")
	}
	byTier := make(map[PlacementTier]uint64, len(in))
	for _, c := range in {
		if !validTier(c.Tier) || c.CapacityBytes == 0 {
			return nil, nil, fmt.Errorf("placement: invalid declared capacity for tier %q", c.Tier)
		}
		if _, exists := byTier[c.Tier]; exists {
			return nil, nil, fmt.Errorf("placement: duplicate capacity for tier %q", c.Tier)
		}
		byTier[c.Tier] = c.CapacityBytes
	}
	out := append([]TierCapacity(nil), in...)
	sort.Slice(out, func(i, j int) bool { return tierOrder(out[i].Tier) < tierOrder(out[j].Tier) })
	return out, byTier, nil
}

func validateComponents(in []PlacementEvidence, capacities map[PlacementTier]uint64) ([]PlacementEvidence, error) {
	if len(in) != 2 {
		return nil, errors.New("placement: ngram-3 PLE embeddings and learned sparse-attention index must both be declared exactly once")
	}
	seen := make(map[ManagedComponent]bool, 2)
	physical := make(map[PlacementTier]uint64, 3)
	peak := make(map[PlacementTier]uint64, 3)
	out := append([]PlacementEvidence(nil), in...)
	for i := range out {
		c := &out[i]
		if c.Component != NGram3PLEEmbeddings && c.Component != SparseAttentionIndex {
			return nil, fmt.Errorf("placement: unknown component %q", c.Component)
		}
		if seen[c.Component] {
			return nil, fmt.Errorf("placement: duplicate component %q", c.Component)
		}
		seen[c.Component] = true
		if !validTier(c.Tier) {
			return nil, fmt.Errorf("placement: component %q has invalid tier %q", c.Component, c.Tier)
		}
		if _, ok := capacities[c.Tier]; !ok {
			return nil, fmt.Errorf("placement: component %q tier %q has no declared capacity", c.Component, c.Tier)
		}
		if err := validateEvidence(*c); err != nil {
			return nil, err
		}
		physical[c.Tier] += c.PhysicalBytes
		peak[c.Tier] += c.PeakMemoryBytes
	}
	for tier, used := range physical {
		if used > capacities[tier] {
			return nil, fmt.Errorf("placement: tier %q physical bytes %d exceed declared capacity %d", tier, used, capacities[tier])
		}
		if peak[tier] > capacities[tier] {
			return nil, fmt.Errorf("placement: tier %q peak memory %d exceeds declared capacity %d", tier, peak[tier], capacities[tier])
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Component < out[j].Component })
	return out, nil
}

func validateEvidence(c PlacementEvidence) error {
	prefix := fmt.Sprintf("placement: component %q", c.Component)
	if c.LogicalBytes == 0 || c.PhysicalBytes == 0 || c.PeakMemoryBytes == 0 {
		return errors.New(prefix + " requires measured logical, physical, and peak-memory bytes")
	}
	if strings.TrimSpace(c.PageBehavior) == "" || strings.Contains(strings.ToLower(c.PageBehavior), "mmap") {
		return errors.New(prefix + " requires explicit non-mmap page behavior")
	}
	if strings.TrimSpace(c.CacheBehavior) == "" || strings.Contains(strings.ToLower(c.CacheBehavior), "fallback") {
		return errors.New(prefix + " requires explicit cache behavior without fallback")
	}
	if c.LoadLatency <= 0 || c.LookupLatency <= 0 {
		return errors.New(prefix + " requires measured load and lookup latency")
	}
	if strings.TrimSpace(c.QualityMetric) == "" || !c.QualityEquivalent || c.QualityTolerance < 0 {
		return errors.New(prefix + " requires measured quality equivalence")
	}
	delta := c.InMemoryQuality - c.PlacedQuality
	if delta < 0 {
		delta = -delta
	}
	if delta > c.QualityTolerance {
		return fmt.Errorf("%s quality delta %g exceeds tolerance %g", prefix, delta, c.QualityTolerance)
	}
	if c.Traffic.Window <= 0 || c.Traffic.ObservedLookups == 0 || c.Traffic.PeakLookupsPerSecond == 0 || c.Traffic.MeasuredSustainableLookupsRate == 0 {
		return errors.New(prefix + " requires measured traffic evidence")
	}
	if c.Traffic.PeakLookupsPerSecond > c.Traffic.MeasuredSustainableLookupsRate {
		return fmt.Errorf("%s measured traffic peak %d/s exceeds measured sustainable rate %d/s", prefix, c.Traffic.PeakLookupsPerSecond, c.Traffic.MeasuredSustainableLookupsRate)
	}
	m := c.Matched
	if strings.TrimSpace(m.PromptSetDigest) == "" || (m.InMemoryTier != TierGPU && m.InMemoryTier != TierCPU) || m.InMemoryLookupLatency <= 0 || m.OffloadLookupLatency <= 0 || m.InMemoryPeakBytes == 0 || m.OffloadPeakBytes == 0 {
		return errors.New(prefix + " requires a measured matched in-memory/offload comparison")
	}
	if c.FailureDetectionLatency <= 0 || c.RecoveryLatency <= 0 || c.RecoveryReadBytes == 0 {
		return errors.New(prefix + " requires measured failure and recovery cost")
	}
	return nil
}

func validTier(t PlacementTier) bool { return t == TierGPU || t == TierCPU || t == TierSSD }

func tierOrder(t PlacementTier) int {
	switch t {
	case TierGPU:
		return 0
	case TierCPU:
		return 1
	default:
		return 2
	}
}
