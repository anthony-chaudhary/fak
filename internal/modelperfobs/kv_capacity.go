package modelperfobs

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"time"
)

const (
	KVCapacitySchema = "fak-kv-capacity/1"

	// EstimatedCapacityCaveat is rendered with every snapshot because geometry-based
	// capacity describes KV storage, not allocator or device memory headroom.
	EstimatedCapacityCaveat = "Estimated KV capacity is modeled from explicit geometry; it is not measured free memory and must not be reported as such."
)

type KVDialect string

const (
	KVDialectBlock  KVDialect = "block-metrics"
	KVDialectDirect KVDialect = "direct-token-byte-metrics"
)

type KVUnit string

const (
	UnitBlocks KVUnit = "blocks"
	UnitTokens KVUnit = "tokens"
	UnitBytes  KVUnit = "bytes"
	UnitRatio  KVUnit = "ratio"
	UnitCount  KVUnit = "count"
)

// KVValueNature says whether a value came from a live observation, declared
// configuration, or an arithmetic estimate. Those authority levels are never
// collapsed into one unlabeled number.
type KVValueNature string

const (
	NatureObserved   KVValueNature = "observed"
	NatureConfigured KVValueNature = "configured"
	NatureEstimated  KVValueNature = "estimated"
)

type KVConfidence string

const (
	ConfidenceExact       KVConfidence = "exact"
	ConfidenceEstimated   KVConfidence = "estimated"
	ConfidenceUnavailable KVConfidence = "unavailable"
)

type KVDerivationMethod string

const (
	MethodDirectObservation      KVDerivationMethod = "direct-observed"
	MethodConfiguredValue        KVDerivationMethod = "configured-value"
	MethodBlocksTimesBlockTokens KVDerivationMethod = "blocks-times-block-tokens"
	MethodBytesDividedByGeometry KVDerivationMethod = "bytes-divided-by-model-geometry"
	MethodModelGeometryEstimate  KVDerivationMethod = "model-geometry-estimate"
	MethodObservedRatio          KVDerivationMethod = "direct-observed-ratio"
	MethodUsedDividedByTotal     KVDerivationMethod = "used-divided-by-total"
	MethodCounterDelta           KVDerivationMethod = "monotonic-counter-delta"
	MethodUnavailable            KVDerivationMethod = "unavailable"
)

type KVValidationCode string

const (
	ValidationInvalidDenominator  KVValidationCode = "invalid-denominator"
	ValidationImpossibleOccupancy KVValidationCode = "impossible-occupancy"
	ValidationInvalidCapacity     KVValidationCode = "invalid-capacity"
	ValidationUnitInvariant       KVValidationCode = "unit-invariant"
	ValidationCounterReset        KVValidationCode = "counter-reset"
	ValidationIdentityChanged     KVValidationCode = "identity-changed"
	ValidationInvalidScrapeOrder  KVValidationCode = "invalid-scrape-order"
)

type KVProvenance struct {
	Nature KVValueNature `json:"nature"`
	Source string        `json:"source"`
	Metric string        `json:"metric"`
}

type KVUnsignedMetric struct {
	Value      uint64       `json:"value"`
	Unit       KVUnit       `json:"unit"`
	Provenance KVProvenance `json:"provenance"`
}

type KVRatioMetric struct {
	Value      float64      `json:"value"`
	Unit       KVUnit       `json:"unit"`
	Provenance KVProvenance `json:"provenance"`
}

// KVNativeMetrics preserves the backend's own vocabulary. Pointers make a
// reported zero distinguishable from a metric the backend never supplied.
type KVNativeMetrics struct {
	UsedBlocks          *KVUnsignedMetric `json:"used_blocks,omitempty"`
	TotalBlocks         *KVUnsignedMetric `json:"total_blocks,omitempty"`
	BlockTokens         *KVUnsignedMetric `json:"block_size_tokens,omitempty"`
	ResidentTokens      *KVUnsignedMetric `json:"resident_tokens,omitempty"`
	TotalTokens         *KVUnsignedMetric `json:"total_tokens,omitempty"`
	ReusableBlocks      *KVUnsignedMetric `json:"reusable_blocks,omitempty"`
	ReusableTokens      *KVUnsignedMetric `json:"reusable_tokens,omitempty"`
	ResidentBytes       *KVUnsignedMetric `json:"resident_bytes,omitempty"`
	ConfiguredBytes     *KVUnsignedMetric `json:"configured_bytes,omitempty"`
	AllocatableBytes    *KVUnsignedMetric `json:"allocatable_bytes,omitempty"`
	Occupancy           *KVRatioMetric    `json:"occupancy,omitempty"`
	HighWaterMarkBlocks *KVUnsignedMetric `json:"high_water_mark_blocks,omitempty"`
	HighWaterMarkTokens *KVUnsignedMetric `json:"high_water_mark_tokens,omitempty"`
	HighWaterMarkBytes  *KVUnsignedMetric `json:"high_water_mark_bytes,omitempty"`
	Evictions           *KVUnsignedMetric `json:"evictions,omitempty"`
	Preemptions         *KVUnsignedMetric `json:"preemptions,omitempty"`
}

// KVGeometry contains only explicit model/runtime configuration. A missing
// dimension disables the corresponding conversion instead of becoming zero.
type KVGeometry struct {
	Layers        *KVUnsignedMetric `json:"layers,omitempty"`
	KVHeads       *KVUnsignedMetric `json:"kv_heads,omitempty"`
	HeadDimension *KVUnsignedMetric `json:"head_dimension,omitempty"`
	DType         string            `json:"dtype,omitempty"`
	DTypeBytes    *KVUnsignedMetric `json:"dtype_bytes,omitempty"`
}

type KVMetricSample struct {
	Dialect     KVDialect                  `json:"dialect"`
	ScrapedAt   time.Time                  `json:"scraped_at"`
	ModelID     string                     `json:"model_id"`
	RuntimeID   string                     `json:"runtime_id"`
	Native      KVNativeMetrics            `json:"native"`
	Geometry    KVGeometry                 `json:"geometry"`
	RawMetrics  map[string]json.RawMessage `json:"raw_metrics"`
	RawGeometry map[string]json.RawMessage `json:"raw_geometry"`
}

// KVDerivedUint64 always carries method and confidence, including when Value
// is unavailable. Sources name every observed/configured input used.
type KVDerivedUint64 struct {
	Value             *uint64            `json:"value,omitempty"`
	Unit              KVUnit             `json:"unit"`
	Method            KVDerivationMethod `json:"method"`
	Confidence        KVConfidence       `json:"confidence"`
	Sources           []KVProvenance     `json:"sources,omitempty"`
	UnavailableReason string             `json:"unavailable_reason,omitempty"`
}

type KVDerivedRatio struct {
	Value             *float64           `json:"value,omitempty"`
	Unit              KVUnit             `json:"unit"`
	Method            KVDerivationMethod `json:"method"`
	Confidence        KVConfidence       `json:"confidence"`
	Sources           []KVProvenance     `json:"sources,omitempty"`
	UnavailableReason string             `json:"unavailable_reason,omitempty"`
}

type KVNormalizedMetrics struct {
	ResidentTokens      KVDerivedUint64 `json:"resident_tokens"`
	TotalTokens         KVDerivedUint64 `json:"total_tokens"`
	ReusableTokens      KVDerivedUint64 `json:"reusable_tokens"`
	ResidentBytes       KVDerivedUint64 `json:"resident_bytes"`
	ConfiguredBytes     KVDerivedUint64 `json:"configured_bytes"`
	AllocatableBytes    KVDerivedUint64 `json:"allocatable_bytes"`
	HighWaterMarkTokens KVDerivedUint64 `json:"high_water_mark_tokens"`
	HighWaterMarkBytes  KVDerivedUint64 `json:"high_water_mark_bytes"`
	Occupancy           KVDerivedRatio  `json:"occupancy"`
	ReusableOccupancy   KVDerivedRatio  `json:"reusable_occupancy"`
}

type KVCounterDeltas struct {
	Evictions   KVDerivedUint64 `json:"evictions"`
	Preemptions KVDerivedUint64 `json:"preemptions"`
}

type KVValidationIssue struct {
	Code    KVValidationCode `json:"code"`
	Field   string           `json:"field"`
	Message string           `json:"message"`
}

type KVValidation struct {
	Valid               bool                `json:"valid"`
	CrossUnitComparable bool                `json:"cross_unit_comparable"`
	TemporalComparable  bool                `json:"temporal_comparable"`
	Issues              []KVValidationIssue `json:"issues,omitempty"`
}

type KVCapacitySnapshot struct {
	Schema        string                     `json:"schema"`
	Dialect       KVDialect                  `json:"dialect"`
	ScrapedAt     time.Time                  `json:"scraped_at"`
	ModelID       string                     `json:"model_id"`
	RuntimeID     string                     `json:"runtime_id"`
	Native        KVNativeMetrics            `json:"native"`
	Geometry      KVGeometry                 `json:"geometry"`
	Normalized    KVNormalizedMetrics        `json:"normalized"`
	CounterDeltas KVCounterDeltas            `json:"counter_deltas"`
	Validation    KVValidation               `json:"validation"`
	RawMetrics    map[string]json.RawMessage `json:"raw_metrics"`
	RawGeometry   map[string]json.RawMessage `json:"raw_geometry"`
	Caveats       []string                   `json:"caveats"`
}

type kvWireSample struct {
	Schema    string                     `json:"schema"`
	ScrapedAt time.Time                  `json:"scraped_at"`
	ModelID   string                     `json:"model_id"`
	RuntimeID string                     `json:"runtime_id"`
	Metrics   map[string]json.RawMessage `json:"metrics"`
	Geometry  map[string]json.RawMessage `json:"geometry"`
}

// DecodeKVMetricSample adapts one pinned metric dialect while retaining the
// exact raw metric payload alongside typed native values.
func DecodeKVMetricSample(data []byte, dialect KVDialect) (KVMetricSample, error) {
	var wire kvWireSample
	if err := json.Unmarshal(data, &wire); err != nil {
		return KVMetricSample{}, fmt.Errorf("decode KV metric sample: %w", err)
	}
	wantSchema := map[KVDialect]string{
		KVDialectBlock:  "fak-kv-block-metrics/1",
		KVDialectDirect: "fak-kv-direct-metrics/1",
	}[dialect]
	if wantSchema == "" {
		return KVMetricSample{}, fmt.Errorf("unsupported KV metric dialect %q", dialect)
	}
	if wire.Schema != wantSchema {
		return KVMetricSample{}, fmt.Errorf("KV metric schema %q does not match dialect %q (%s)", wire.Schema, dialect, wantSchema)
	}
	sample := KVMetricSample{
		Dialect:     dialect,
		ScrapedAt:   wire.ScrapedAt.UTC(),
		ModelID:     wire.ModelID,
		RuntimeID:   wire.RuntimeID,
		RawMetrics:  cloneRawMetrics(wire.Metrics),
		RawGeometry: cloneRawMetrics(wire.Geometry),
	}
	var err error
	if dialect == KVDialectBlock {
		err = decodeBlockNative(&sample.Native, wire.Schema, wire.Metrics)
	} else {
		err = decodeDirectNative(&sample.Native, wire.Schema, wire.Metrics)
	}
	if err != nil {
		return KVMetricSample{}, err
	}
	if err := decodeGeometry(&sample.Geometry, wire.Schema, wire.Geometry); err != nil {
		return KVMetricSample{}, err
	}
	return sample, nil
}

func decodeBlockNative(dst *KVNativeMetrics, source string, metrics map[string]json.RawMessage) error {
	var err error
	if dst.UsedBlocks, err = decodeUint(metrics, "kv_used_blocks", UnitBlocks, NatureObserved, source); err != nil {
		return err
	}
	if dst.TotalBlocks, err = decodeUint(metrics, "kv_total_blocks", UnitBlocks, NatureConfigured, source); err != nil {
		return err
	}
	if dst.BlockTokens, err = decodeUint(metrics, "kv_block_size_tokens", UnitTokens, NatureConfigured, source); err != nil {
		return err
	}
	if dst.ReusableBlocks, err = decodeUint(metrics, "kv_reusable_blocks", UnitBlocks, NatureObserved, source); err != nil {
		return err
	}
	if dst.ConfiguredBytes, err = decodeUint(metrics, "kv_configured_bytes", UnitBytes, NatureConfigured, source); err != nil {
		return err
	}
	if dst.AllocatableBytes, err = decodeUint(metrics, "kv_allocatable_bytes", UnitBytes, NatureConfigured, source); err != nil {
		return err
	}
	if dst.Occupancy, err = decodeRatio(metrics, "kv_occupancy_ratio", NatureObserved, source); err != nil {
		return err
	}
	if dst.HighWaterMarkBlocks, err = decodeUint(metrics, "kv_high_water_blocks", UnitBlocks, NatureObserved, source); err != nil {
		return err
	}
	if dst.Evictions, err = decodeUint(metrics, "kv_evictions_total", UnitCount, NatureObserved, source); err != nil {
		return err
	}
	dst.Preemptions, err = decodeUint(metrics, "kv_preemptions_total", UnitCount, NatureObserved, source)
	return err
}

func decodeDirectNative(dst *KVNativeMetrics, source string, metrics map[string]json.RawMessage) error {
	var err error
	if dst.ResidentTokens, err = decodeUint(metrics, "kv_resident_tokens", UnitTokens, NatureObserved, source); err != nil {
		return err
	}
	if dst.TotalTokens, err = decodeUint(metrics, "kv_total_tokens", UnitTokens, NatureObserved, source); err != nil {
		return err
	}
	if dst.ReusableTokens, err = decodeUint(metrics, "kv_reusable_tokens", UnitTokens, NatureObserved, source); err != nil {
		return err
	}
	if dst.ResidentBytes, err = decodeUint(metrics, "kv_resident_bytes", UnitBytes, NatureObserved, source); err != nil {
		return err
	}
	if dst.ConfiguredBytes, err = decodeUint(metrics, "kv_configured_bytes", UnitBytes, NatureConfigured, source); err != nil {
		return err
	}
	if dst.AllocatableBytes, err = decodeUint(metrics, "kv_allocatable_bytes", UnitBytes, NatureConfigured, source); err != nil {
		return err
	}
	if dst.Occupancy, err = decodeRatio(metrics, "kv_occupancy_ratio", NatureObserved, source); err != nil {
		return err
	}
	if dst.HighWaterMarkTokens, err = decodeUint(metrics, "kv_high_water_tokens", UnitTokens, NatureObserved, source); err != nil {
		return err
	}
	if dst.HighWaterMarkBytes, err = decodeUint(metrics, "kv_high_water_bytes", UnitBytes, NatureObserved, source); err != nil {
		return err
	}
	if dst.Evictions, err = decodeUint(metrics, "kv_evictions_total", UnitCount, NatureObserved, source); err != nil {
		return err
	}
	dst.Preemptions, err = decodeUint(metrics, "kv_preemptions_total", UnitCount, NatureObserved, source)
	return err
}

func decodeGeometry(dst *KVGeometry, source string, geometry map[string]json.RawMessage) error {
	var err error
	if dst.Layers, err = decodeUint(geometry, "layers", UnitCount, NatureConfigured, source); err != nil {
		return err
	}
	if dst.KVHeads, err = decodeUint(geometry, "kv_heads", UnitCount, NatureConfigured, source); err != nil {
		return err
	}
	if dst.HeadDimension, err = decodeUint(geometry, "head_dimension", UnitCount, NatureConfigured, source); err != nil {
		return err
	}
	if dst.DTypeBytes, err = decodeUint(geometry, "dtype_bytes", UnitBytes, NatureConfigured, source); err != nil {
		return err
	}
	if raw, ok := geometry["dtype"]; ok {
		if err := json.Unmarshal(raw, &dst.DType); err != nil {
			return fmt.Errorf("decode geometry dtype: %w", err)
		}
	}
	return nil
}

func decodeUint(values map[string]json.RawMessage, name string, unit KVUnit, nature KVValueNature, source string) (*KVUnsignedMetric, error) {
	raw, ok := values[name]
	if !ok {
		return nil, nil
	}
	var value uint64
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("decode %s as non-negative integer %s: %w", name, unit, err)
	}
	return uintMetric(value, unit, nature, source, name), nil
}

func decodeRatio(values map[string]json.RawMessage, name string, nature KVValueNature, source string) (*KVRatioMetric, error) {
	raw, ok := values[name]
	if !ok {
		return nil, nil
	}
	var value float64
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("decode %s as ratio: %w", name, err)
	}
	return &KVRatioMetric{Value: value, Unit: UnitRatio, Provenance: KVProvenance{Nature: nature, Source: source, Metric: name}}, nil
}

func uintMetric(value uint64, unit KVUnit, nature KVValueNature, source, metric string) *KVUnsignedMetric {
	return &KVUnsignedMetric{Value: value, Unit: unit, Provenance: KVProvenance{Nature: nature, Source: source, Metric: metric}}
}

func cloneRawMetrics(in map[string]json.RawMessage) map[string]json.RawMessage {
	out := make(map[string]json.RawMessage, len(in))
	for name, value := range in {
		out[name] = append(json.RawMessage(nil), value...)
	}
	return out
}

// NormalizeKVCapacity derives comparable units only from explicit native
// metrics and geometry. Previous may be nil; when present it enables temporal
// identity and monotonic-counter validation.
func normalizeKVCapacity(current KVMetricSample, previous *KVMetricSample) KVCapacitySnapshot {
	snapshot := KVCapacitySnapshot{
		Schema:      KVCapacitySchema,
		Dialect:     current.Dialect,
		ScrapedAt:   current.ScrapedAt,
		ModelID:     current.ModelID,
		RuntimeID:   current.RuntimeID,
		Native:      current.Native,
		Geometry:    current.Geometry,
		RawMetrics:  cloneRawMetrics(current.RawMetrics),
		RawGeometry: cloneRawMetrics(current.RawGeometry),
		Caveats:     []string{EstimatedCapacityCaveat},
		Validation:  KVValidation{Valid: true, TemporalComparable: true},
	}
	n := &snapshot.Normalized
	n.ResidentTokens = unavailableUint(UnitTokens, "resident tokens require a direct token metric, or used blocks plus block size in tokens")
	n.TotalTokens = unavailableUint(UnitTokens, "total tokens require a direct token metric, total blocks plus block size in tokens, or allocatable bytes plus complete model geometry")
	n.ReusableTokens = unavailableUint(UnitTokens, "reusable tokens require a direct token metric, or reusable blocks plus block size in tokens")
	n.ResidentBytes = unavailableUint(UnitBytes, "resident bytes require a direct byte metric, or resident tokens plus complete model geometry")
	n.ConfiguredBytes = unavailableUint(UnitBytes, "configured bytes were not supplied")
	n.AllocatableBytes = unavailableUint(UnitBytes, "allocatable bytes were not supplied")
	n.HighWaterMarkTokens = unavailableUint(UnitTokens, "high-water tokens require a direct token metric, or high-water blocks plus block size in tokens")
	n.HighWaterMarkBytes = unavailableUint(UnitBytes, "high-water bytes require a direct byte metric, or high-water tokens plus complete model geometry")
	n.Occupancy = unavailableRatio("occupancy requires a valid observed ratio or valid used and total capacity")
	n.ReusableOccupancy = unavailableRatio("reusable occupancy requires reusable and total tokens")
	snapshot.CounterDeltas.Evictions = unavailableUint(UnitCount, "a previous eviction counter is required")
	snapshot.CounterDeltas.Preemptions = unavailableUint(UnitCount, "a previous preemption counter is required")

	validateCurrent(&snapshot)
	deriveTokens(&snapshot)
	deriveBytes(&snapshot)
	deriveOccupancy(&snapshot)
	validateUnitAgreement(&snapshot)
	validatePrevious(&snapshot, current, previous)
	snapshot.Validation.CrossUnitComparable = snapshot.Normalized.ResidentTokens.Value != nil
	return snapshot
}

func validateCurrent(snapshot *KVCapacitySnapshot) {
	native := snapshot.Native
	invalid := func(code KVValidationCode, field, message string) {
		snapshot.Validation.Valid = false
		snapshot.Validation.Issues = append(snapshot.Validation.Issues, KVValidationIssue{Code: code, Field: field, Message: message})
	}
	if native.TotalBlocks != nil && native.TotalBlocks.Value == 0 {
		invalid(ValidationInvalidDenominator, "total_blocks", "total blocks must be greater than zero when reported")
	}
	if native.BlockTokens != nil && native.BlockTokens.Value == 0 {
		invalid(ValidationInvalidDenominator, "block_size_tokens", "block size in tokens must be greater than zero when reported")
	}
	if native.TotalTokens != nil && native.TotalTokens.Value == 0 {
		invalid(ValidationInvalidDenominator, "total_tokens", "total tokens must be greater than zero when reported")
	}
	if native.UsedBlocks != nil && native.TotalBlocks != nil && native.UsedBlocks.Value > native.TotalBlocks.Value {
		invalid(ValidationImpossibleOccupancy, "used_blocks", "used blocks exceed total blocks")
	}
	if native.ResidentTokens != nil && native.TotalTokens != nil && native.ResidentTokens.Value > native.TotalTokens.Value {
		invalid(ValidationImpossibleOccupancy, "resident_tokens", "resident tokens exceed total tokens")
	}
	if native.ReusableBlocks != nil && native.UsedBlocks != nil && native.ReusableBlocks.Value > native.UsedBlocks.Value {
		invalid(ValidationImpossibleOccupancy, "reusable_blocks", "reusable blocks exceed resident blocks")
	}
	if native.ReusableTokens != nil && native.ResidentTokens != nil && native.ReusableTokens.Value > native.ResidentTokens.Value {
		invalid(ValidationImpossibleOccupancy, "reusable_tokens", "reusable tokens exceed resident tokens")
	}
	if native.Occupancy != nil && (native.Occupancy.Value < 0 || native.Occupancy.Value > 1 || math.IsNaN(native.Occupancy.Value) || math.IsInf(native.Occupancy.Value, 0)) {
		invalid(ValidationImpossibleOccupancy, "occupancy", "occupancy must be within [0,1]")
	}
	if native.AllocatableBytes != nil && native.ConfiguredBytes != nil && native.AllocatableBytes.Value > native.ConfiguredBytes.Value {
		invalid(ValidationInvalidCapacity, "allocatable_bytes", "allocatable bytes exceed configured bytes")
	}
	if native.HighWaterMarkBlocks != nil && native.UsedBlocks != nil && native.HighWaterMarkBlocks.Value < native.UsedBlocks.Value {
		invalid(ValidationImpossibleOccupancy, "high_water_mark_blocks", "high-water blocks are below current used blocks")
	}
	if native.HighWaterMarkTokens != nil && native.ResidentTokens != nil && native.HighWaterMarkTokens.Value < native.ResidentTokens.Value {
		invalid(ValidationImpossibleOccupancy, "high_water_mark_tokens", "high-water tokens are below current resident tokens")
	}
	for field, metric := range map[string]*KVUnsignedMetric{
		"layers": snapshot.Geometry.Layers, "kv_heads": snapshot.Geometry.KVHeads,
		"head_dimension": snapshot.Geometry.HeadDimension, "dtype_bytes": snapshot.Geometry.DTypeBytes,
	} {
		if metric != nil && metric.Value == 0 {
			invalid(ValidationInvalidDenominator, field, field+" must be greater than zero when reported")
		}
	}
}

func deriveTokens(snapshot *KVCapacitySnapshot) {
	native := snapshot.Native
	n := &snapshot.Normalized
	if native.ResidentTokens != nil {
		n.ResidentTokens = fromUintMetric(native.ResidentTokens)
	} else if value, sources, ok := multiplyMetrics(native.UsedBlocks, native.BlockTokens); ok {
		n.ResidentTokens = exactUint(value, UnitTokens, MethodBlocksTimesBlockTokens, sources)
	}
	if native.TotalTokens != nil {
		n.TotalTokens = fromUintMetric(native.TotalTokens)
	} else if value, sources, ok := multiplyMetrics(native.TotalBlocks, native.BlockTokens); ok {
		n.TotalTokens = exactUint(value, UnitTokens, MethodBlocksTimesBlockTokens, sources)
	} else if bytesPerToken, sources, ok := geometryBytesPerToken(snapshot.Geometry); ok && native.AllocatableBytes != nil && bytesPerToken > 0 {
		value := native.AllocatableBytes.Value / bytesPerToken
		n.TotalTokens = derivedUint(value, UnitTokens, MethodBytesDividedByGeometry, ConfidenceEstimated, append([]KVProvenance{native.AllocatableBytes.Provenance}, sources...))
	}
	if native.ReusableTokens != nil {
		n.ReusableTokens = fromUintMetric(native.ReusableTokens)
	} else if value, sources, ok := multiplyMetrics(native.ReusableBlocks, native.BlockTokens); ok {
		n.ReusableTokens = exactUint(value, UnitTokens, MethodBlocksTimesBlockTokens, sources)
	}
	if native.HighWaterMarkTokens != nil {
		n.HighWaterMarkTokens = fromUintMetric(native.HighWaterMarkTokens)
	} else if value, sources, ok := multiplyMetrics(native.HighWaterMarkBlocks, native.BlockTokens); ok {
		n.HighWaterMarkTokens = exactUint(value, UnitTokens, MethodBlocksTimesBlockTokens, sources)
	}
	if n.ResidentTokens.Value == nil && native.ResidentBytes != nil {
		if bytesPerToken, sources, ok := geometryBytesPerToken(snapshot.Geometry); ok && bytesPerToken > 0 {
			value := native.ResidentBytes.Value / bytesPerToken
			n.ResidentTokens = derivedUint(value, UnitTokens, MethodBytesDividedByGeometry, ConfidenceEstimated, append([]KVProvenance{native.ResidentBytes.Provenance}, sources...))
		}
	}
}

func deriveBytes(snapshot *KVCapacitySnapshot) {
	native := snapshot.Native
	n := &snapshot.Normalized
	if native.ConfiguredBytes != nil {
		n.ConfiguredBytes = fromUintMetric(native.ConfiguredBytes)
	}
	if native.AllocatableBytes != nil {
		n.AllocatableBytes = fromUintMetric(native.AllocatableBytes)
	}
	if native.ResidentBytes != nil {
		n.ResidentBytes = fromUintMetric(native.ResidentBytes)
	} else if n.ResidentTokens.Value != nil {
		if bytesPerToken, sources, ok := geometryBytesPerToken(snapshot.Geometry); ok {
			if value, ok := checkedMul(*n.ResidentTokens.Value, bytesPerToken); ok {
				n.ResidentBytes = derivedUint(value, UnitBytes, MethodModelGeometryEstimate, ConfidenceEstimated, append(append([]KVProvenance(nil), n.ResidentTokens.Sources...), sources...))
			}
		}
	}
	if native.HighWaterMarkBytes != nil {
		n.HighWaterMarkBytes = fromUintMetric(native.HighWaterMarkBytes)
	} else if n.HighWaterMarkTokens.Value != nil {
		if bytesPerToken, sources, ok := geometryBytesPerToken(snapshot.Geometry); ok {
			if value, ok := checkedMul(*n.HighWaterMarkTokens.Value, bytesPerToken); ok {
				n.HighWaterMarkBytes = derivedUint(value, UnitBytes, MethodModelGeometryEstimate, ConfidenceEstimated, append(append([]KVProvenance(nil), n.HighWaterMarkTokens.Sources...), sources...))
			}
		}
	}
}

func deriveOccupancy(snapshot *KVCapacitySnapshot) {
	native := snapshot.Native
	n := &snapshot.Normalized
	if native.Occupancy != nil && native.Occupancy.Value >= 0 && native.Occupancy.Value <= 1 {
		n.Occupancy = derivedRatio(native.Occupancy.Value, MethodObservedRatio, ConfidenceExact, []KVProvenance{native.Occupancy.Provenance})
	} else if n.ResidentTokens.Value != nil && n.TotalTokens.Value != nil && *n.TotalTokens.Value > 0 && *n.ResidentTokens.Value <= *n.TotalTokens.Value {
		n.Occupancy = derivedRatio(float64(*n.ResidentTokens.Value)/float64(*n.TotalTokens.Value), MethodUsedDividedByTotal, ConfidenceExact, append(append([]KVProvenance(nil), n.ResidentTokens.Sources...), n.TotalTokens.Sources...))
	}
	if n.ReusableTokens.Value != nil && n.TotalTokens.Value != nil && *n.TotalTokens.Value > 0 && *n.ReusableTokens.Value <= *n.TotalTokens.Value {
		n.ReusableOccupancy = derivedRatio(float64(*n.ReusableTokens.Value)/float64(*n.TotalTokens.Value), MethodUsedDividedByTotal, ConfidenceExact, append(append([]KVProvenance(nil), n.ReusableTokens.Sources...), n.TotalTokens.Sources...))
	}
}

func validateUnitAgreement(snapshot *KVCapacitySnapshot) {
	invalid := func(field, message string) {
		snapshot.Validation.Valid = false
		snapshot.Validation.Issues = append(snapshot.Validation.Issues, KVValidationIssue{Code: ValidationUnitInvariant, Field: field, Message: message})
	}
	native := snapshot.Native
	if native.ResidentTokens != nil {
		if blockValue, _, ok := multiplyMetrics(native.UsedBlocks, native.BlockTokens); ok && blockValue != native.ResidentTokens.Value {
			invalid("resident_tokens", "direct resident tokens disagree with used blocks times block size")
		}
	}
	if native.TotalTokens != nil {
		if blockValue, _, ok := multiplyMetrics(native.TotalBlocks, native.BlockTokens); ok && blockValue != native.TotalTokens.Value {
			invalid("total_tokens", "direct total tokens disagree with total blocks times block size")
		}
	}
	if native.Occupancy != nil && native.Occupancy.Value >= 0 && native.Occupancy.Value <= 1 && native.UsedBlocks != nil && native.TotalBlocks != nil && native.TotalBlocks.Value > 0 {
		derived := float64(native.UsedBlocks.Value) / float64(native.TotalBlocks.Value)
		if math.Abs(derived-native.Occupancy.Value) > 1e-9 {
			invalid("occupancy", "reported occupancy disagrees with used blocks divided by total blocks")
		}
	}
}

func validatePrevious(snapshot *KVCapacitySnapshot, current KVMetricSample, previous *KVMetricSample) {
	if previous == nil {
		return
	}
	identityChanged := previous.ModelID != current.ModelID || previous.RuntimeID != current.RuntimeID
	if identityChanged {
		snapshot.Validation.TemporalComparable = false
		snapshot.Validation.Issues = append(snapshot.Validation.Issues, KVValidationIssue{
			Code: ValidationIdentityChanged, Field: "model_id,runtime_id",
			Message: fmt.Sprintf("model/runtime identity changed from %q/%q to %q/%q", previous.ModelID, previous.RuntimeID, current.ModelID, current.RuntimeID),
		})
	}
	invalidOrder := current.ScrapedAt.IsZero() || previous.ScrapedAt.IsZero() || !current.ScrapedAt.After(previous.ScrapedAt)
	if invalidOrder {
		snapshot.Validation.TemporalComparable = false
		snapshot.Validation.Issues = append(snapshot.Validation.Issues, KVValidationIssue{
			Code: ValidationInvalidScrapeOrder, Field: "scraped_at",
			Message: "current scrape must be later than the previous scrape",
		})
	}
	temporalInvalid := identityChanged || invalidOrder
	snapshot.CounterDeltas.Evictions = counterDelta("evictions", current.Native.Evictions, previous.Native.Evictions, temporalInvalid, &snapshot.Validation)
	snapshot.CounterDeltas.Preemptions = counterDelta("preemptions", current.Native.Preemptions, previous.Native.Preemptions, temporalInvalid, &snapshot.Validation)
}

func counterDelta(field string, current, previous *KVUnsignedMetric, temporalInvalid bool, validation *KVValidation) KVDerivedUint64 {
	if current == nil || previous == nil {
		return unavailableUint(UnitCount, "both current and previous counters are required")
	}
	if current.Value < previous.Value {
		validation.TemporalComparable = false
		validation.Issues = append(validation.Issues, KVValidationIssue{Code: ValidationCounterReset, Field: field, Message: field + " counter decreased between scrapes"})
		return unavailableUint(UnitCount, field+" counter reset between scrapes")
	}
	if temporalInvalid {
		return unavailableUint(UnitCount, "scrape order or model/runtime identity changed between scrapes")
	}
	return exactUint(current.Value-previous.Value, UnitCount, MethodCounterDelta, []KVProvenance{previous.Provenance, current.Provenance})
}

func fromUintMetric(metric *KVUnsignedMetric) KVDerivedUint64 {
	method := MethodDirectObservation
	if metric.Provenance.Nature == NatureConfigured {
		method = MethodConfiguredValue
	}
	return exactUint(metric.Value, metric.Unit, method, []KVProvenance{metric.Provenance})
}

func unavailableUint(unit KVUnit, reason string) KVDerivedUint64 {
	return KVDerivedUint64{Unit: unit, Method: MethodUnavailable, Confidence: ConfidenceUnavailable, UnavailableReason: reason}
}

func unavailableRatio(reason string) KVDerivedRatio {
	return KVDerivedRatio{Unit: UnitRatio, Method: MethodUnavailable, Confidence: ConfidenceUnavailable, UnavailableReason: reason}
}

func exactUint(value uint64, unit KVUnit, method KVDerivationMethod, sources []KVProvenance) KVDerivedUint64 {
	return derivedUint(value, unit, method, ConfidenceExact, sources)
}

func derivedUint(value uint64, unit KVUnit, method KVDerivationMethod, confidence KVConfidence, sources []KVProvenance) KVDerivedUint64 {
	v := value
	return KVDerivedUint64{Value: &v, Unit: unit, Method: method, Confidence: confidence, Sources: sources}
}

func derivedRatio(value float64, method KVDerivationMethod, confidence KVConfidence, sources []KVProvenance) KVDerivedRatio {
	v := value
	return KVDerivedRatio{Value: &v, Unit: UnitRatio, Method: method, Confidence: confidence, Sources: sources}
}

func multiplyMetrics(a, b *KVUnsignedMetric) (uint64, []KVProvenance, bool) {
	if a == nil || b == nil || b.Value == 0 {
		return 0, nil, false
	}
	value, ok := checkedMul(a.Value, b.Value)
	return value, []KVProvenance{a.Provenance, b.Provenance}, ok
}

func geometryBytesPerToken(geometry KVGeometry) (uint64, []KVProvenance, bool) {
	metrics := []*KVUnsignedMetric{geometry.Layers, geometry.KVHeads, geometry.HeadDimension, geometry.DTypeBytes}
	value := uint64(2) // key and value vectors
	sources := make([]KVProvenance, 0, len(metrics))
	for _, metric := range metrics {
		if metric == nil || metric.Value == 0 {
			return 0, nil, false
		}
		var ok bool
		value, ok = checkedMul(value, metric.Value)
		if !ok {
			return 0, nil, false
		}
		sources = append(sources, metric.Provenance)
	}
	return value, sources, true
}

func checkedMul(a, b uint64) (uint64, bool) {
	if a != 0 && b > math.MaxUint64/a {
		return 0, false
	}
	return a * b, true
}

// WriteKVCapacityMarkdown renders native values before normalized values so a
// comparable view never erases the backend-specific evidence that produced it.
func WriteKVCapacityMarkdown(w io.Writer, snapshot KVCapacitySnapshot) error {
	if _, err := fmt.Fprintf(w, "# KV capacity snapshot\n\n- Dialect: **%s**\n- Model/runtime: **%s / %s**\n- Valid: **%t**; cross-unit comparable: **%t**; temporal comparable: **%t**\n\n## Native values\n\n", snapshot.Dialect, snapshot.ModelID, snapshot.RuntimeID, snapshot.Validation.Valid, snapshot.Validation.CrossUnitComparable, snapshot.Validation.TemporalComparable); err != nil {
		return err
	}
	for _, row := range nativeMarkdownRows(snapshot.Native) {
		if row.unsigned != nil {
			if _, err := fmt.Fprintf(w, "- %s: %d %s (%s; source=%s)\n", row.name, row.unsigned.Value, row.unsigned.Unit, row.unsigned.Provenance.Nature, row.unsigned.Provenance.Source); err != nil {
				return err
			}
		} else if row.ratio != nil {
			if _, err := fmt.Fprintf(w, "- %s: %.6g %s (%s; source=%s)\n", row.name, row.ratio.Value, row.ratio.Unit, row.ratio.Provenance.Nature, row.ratio.Provenance.Source); err != nil {
				return err
			}
		}
	}
	if _, err := fmt.Fprint(w, "\n## Normalized values\n\n"); err != nil {
		return err
	}
	for _, row := range derivedMarkdownRows(snapshot.Normalized, snapshot.CounterDeltas) {
		if row.unsigned != nil {
			if err := writeDerivedUint(w, row.name, *row.unsigned); err != nil {
				return err
			}
		} else if row.ratio != nil {
			if err := writeDerivedRatio(w, row.name, *row.ratio); err != nil {
				return err
			}
		}
	}
	if err := writeRawMetrics(w, "Raw adapter metrics", snapshot.RawMetrics); err != nil {
		return err
	}
	if err := writeRawMetrics(w, "Raw configured geometry", snapshot.RawGeometry); err != nil {
		return err
	}
	if len(snapshot.Validation.Issues) > 0 {
		if _, err := fmt.Fprint(w, "\n## Validation\n\n"); err != nil {
			return err
		}
		for _, issue := range snapshot.Validation.Issues {
			if _, err := fmt.Fprintf(w, "- %s (%s): %s\n", issue.Code, issue.Field, issue.Message); err != nil {
				return err
			}
		}
	}
	_, err := fmt.Fprintf(w, "\n## Provenance boundary\n\n- Observed values come from backend counters; configured values come from explicit runtime/model configuration; estimated values are arithmetic over named sources.\n- %s\n", EstimatedCapacityCaveat)
	return err
}

type nativeMarkdownRow struct {
	name     string
	unsigned *KVUnsignedMetric
	ratio    *KVRatioMetric
}

func nativeMarkdownRows(native KVNativeMetrics) []nativeMarkdownRow {
	rows := []nativeMarkdownRow{
		{"used_blocks", native.UsedBlocks, nil},
		{"total_blocks", native.TotalBlocks, nil},
		{"block_size_tokens", native.BlockTokens, nil},
		{"resident_tokens", native.ResidentTokens, nil},
		{"total_tokens", native.TotalTokens, nil},
		{"reusable_blocks", native.ReusableBlocks, nil},
		{"reusable_tokens", native.ReusableTokens, nil},
		{"resident_bytes", native.ResidentBytes, nil},
		{"configured_bytes", native.ConfiguredBytes, nil},
		{"allocatable_bytes", native.AllocatableBytes, nil},
		{"occupancy", nil, native.Occupancy},
		{"high_water_mark_blocks", native.HighWaterMarkBlocks, nil},
		{"high_water_mark_tokens", native.HighWaterMarkTokens, nil},
		{"high_water_mark_bytes", native.HighWaterMarkBytes, nil},
		{"evictions", native.Evictions, nil},
		{"preemptions", native.Preemptions, nil},
	}
	for i := range rows {
		if rows[i].unsigned != nil {
			rows[i].name = rows[i].unsigned.Provenance.Metric
		} else if rows[i].ratio != nil {
			rows[i].name = rows[i].ratio.Provenance.Metric
		}
	}
	return rows
}

type derivedMarkdownRow struct {
	name     string
	unsigned *KVDerivedUint64
	ratio    *KVDerivedRatio
}

func derivedMarkdownRows(normalized KVNormalizedMetrics, counters KVCounterDeltas) []derivedMarkdownRow {
	return []derivedMarkdownRow{
		{"resident_tokens", &normalized.ResidentTokens, nil},
		{"total_tokens", &normalized.TotalTokens, nil},
		{"reusable_tokens", &normalized.ReusableTokens, nil},
		{"resident_bytes", &normalized.ResidentBytes, nil},
		{"configured_bytes", &normalized.ConfiguredBytes, nil},
		{"allocatable_bytes", &normalized.AllocatableBytes, nil},
		{"high_water_mark_tokens", &normalized.HighWaterMarkTokens, nil},
		{"high_water_mark_bytes", &normalized.HighWaterMarkBytes, nil},
		{"occupancy", nil, &normalized.Occupancy},
		{"reusable_occupancy", nil, &normalized.ReusableOccupancy},
		{"evictions_delta", &counters.Evictions, nil},
		{"preemptions_delta", &counters.Preemptions, nil},
	}
}

func writeRawMetrics(w io.Writer, heading string, metrics map[string]json.RawMessage) error {
	if len(metrics) == 0 {
		return nil
	}
	if _, err := fmt.Fprintf(w, "\n## %s\n\n", heading); err != nil {
		return err
	}
	names := make([]string, 0, len(metrics))
	for name := range metrics {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if _, err := fmt.Fprintf(w, "- %s: `%s`\n", name, strings.TrimSpace(string(metrics[name]))); err != nil {
			return err
		}
	}
	return nil
}

func writeDerivedUint(w io.Writer, name string, value KVDerivedUint64) error {
	return writeDerived(w, name, value.Value, value.Unit, value.Method, value.Confidence, value.UnavailableReason, "%d")
}

func writeDerivedRatio(w io.Writer, name string, value KVDerivedRatio) error {
	return writeDerived(w, name, value.Value, value.Unit, value.Method, value.Confidence, value.UnavailableReason, "%.6g")
}

func writeDerived[T uint64 | float64](w io.Writer, name string, value *T, unit KVUnit, method KVDerivationMethod, confidence KVConfidence, unavailableReason, format string) error {
	if value == nil {
		_, err := fmt.Fprintf(w, "- %s: unavailable (%s, %s; %s)\n", name, method, confidence, unavailableReason)
		return err
	}
	formatted := fmt.Sprintf(format, *value)
	_, err := fmt.Fprintf(w, "- %s: %s %s (%s, %s)\n", name, formatted, unit, method, confidence)
	return err
}
