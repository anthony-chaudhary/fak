package engine

import (
	"fmt"
	"time"
)

// KVPrecision is the storage format of a resident key/value span.
type KVPrecision string

const (
	KVPrecisionFP16 KVPrecision = "fp16"
	KVPrecisionINT8 KVPrecision = "int8"
	KVPrecisionFP8  KVPrecision = "fp8"
	KVPrecisionINT4 KVPrecision = "int4"
)

// KVQuantizationState is the controller-visible state for one resident span.
// EstimatedError is a normalized, backend-measured quality-loss estimate. A span
// is quantizable only when the backend explicitly marks it Eligible.
type KVQuantizationState struct {
	Precision       KVPrecision
	IndexKPrecision KVPrecision // independent auxiliary sparse IndexK precision
	Eligible        bool
	EstimatedError  float64
	LastTransition  time.Time
}

// KVQuantizationThresholds controls the pressure ladder. PromotePressure must be
// lower than DemotePressure, making the gap an explicit hysteresis band.
type KVQuantizationThresholds struct {
	DemotePressure  float64
	PromotePressure float64
	AccuracyBudget  float64
	MinDwell        time.Duration
}

// KVQuantizationTransition is a pure, auditable transition decision.
type KVQuantizationTransition struct {
	From   KVPrecision
	To     KVPrecision
	Change bool
	Reason string
}

func (p KVQuantizationThresholds) normalized() KVQuantizationThresholds {
	if p.DemotePressure <= 0 || p.DemotePressure > 1 {
		p.DemotePressure = 0.90
	}
	if p.PromotePressure < 0 || p.PromotePressure >= p.DemotePressure {
		p.PromotePressure = 0.70
	}
	if p.AccuracyBudget < 0 {
		p.AccuracyBudget = 0
	}
	if p.MinDwell < 0 {
		p.MinDwell = 0
	}
	return p
}

// ChooseKVQuantization chooses at most one adjacent ladder step. Under pressure it
// walks FP16 -> INT8 -> FP8; after pressure clears it walks the reverse direction.
// Ineligible spans, unknown formats, accuracy-budget violations, and dwell-time
// violations are stable no-ops rather than implicit evictions.
func ChooseKVQuantization(now time.Time, pressure float64, state KVQuantizationState, policy KVQuantizationThresholds) KVQuantizationTransition {
	p := policy.normalized()
	d := KVQuantizationTransition{From: state.Precision, To: state.Precision}
	if !state.Eligible {
		d.Reason = "ineligible"
		return d
	}
	if state.Precision != KVPrecisionFP16 && state.Precision != KVPrecisionINT8 && state.Precision != KVPrecisionFP8 {
		d.Reason = "unknown-precision"
		return d
	}
	if !state.LastTransition.IsZero() && now.Sub(state.LastTransition) < p.MinDwell {
		d.Reason = "dwell"
		return d
	}
	if pressure >= p.DemotePressure {
		if state.EstimatedError > p.AccuracyBudget {
			d.Reason = "accuracy-budget"
			return d
		}
		switch state.Precision {
		case KVPrecisionFP16:
			d.To = KVPrecisionINT8
		case KVPrecisionINT8:
			d.To = KVPrecisionFP8
		default:
			d.Reason = "lowest-precision"
			return d
		}
		d.Change, d.Reason = true, "pressure-demote"
		return d
	}
	if pressure <= p.PromotePressure {
		switch state.Precision {
		case KVPrecisionFP8:
			d.To = KVPrecisionINT8
		case KVPrecisionINT8:
			d.To = KVPrecisionFP16
		default:
			d.Reason = "highest-precision"
			return d
		}
		d.Change, d.Reason = true, "pressure-cleared-promote"
		return d
	}
	d.Reason = "hysteresis"
	return d
}

// KVQuantizedBytes returns the expected resident bytes after an adjacent ladder
// transition. The 8-bit rungs occupy half the FP16 bytes; INT8 and FP8 are equal
// sized and differ in backend accuracy/performance characteristics.
func KVQuantizedBytes(fp16Bytes int64, precision KVPrecision) (int64, error) {
	if fp16Bytes < 0 {
		return 0, fmt.Errorf("fp16 bytes must be non-negative")
	}
	switch precision {
	case KVPrecisionFP16:
		return fp16Bytes, nil
	case KVPrecisionINT8, KVPrecisionFP8:
		return (fp16Bytes + 1) / 2, nil
	case KVPrecisionINT4:
		return (fp16Bytes + 3) / 4, nil
	default:
		return 0, fmt.Errorf("unknown KV precision %q", precision)
	}
}

// CalculateDualKVBytes computes main KV bytes and auxiliary IndexK bytes independently,
// verifying that changing auxiliary IndexK precision does not silently mutate main KV bytes.
func CalculateDualKVBytes(span KVQuantizationSpan) (mainBytes, indexKBytes, totalBytes int64, err error) {
	mainBytes, err = KVQuantizedBytes(span.FP16Bytes, span.State.Precision)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("main KV: %w", err)
	}

	indexPrec := span.State.IndexKPrecision
	if indexPrec == "" {
		indexPrec = span.State.Precision
	}
	indexKBytes, err = KVQuantizedBytes(span.IndexKFP16Bytes, indexPrec)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("auxiliary IndexK: %w", err)
	}

	return mainBytes, indexKBytes, mainBytes + indexKBytes, nil
}

// KVColdCodec is the optional cold-tier representation selected after the
// precision ladder. It is separate from KVPrecision: cold-sequence compression is a sequence
// codec, not another arithmetic dtype.
type KVColdCodec string

const (
	KVColdCodecNone     KVColdCodec = "none"
	KVColdCodecSequence KVColdCodec = "sequence"
)

type KVQuantizationSpan struct {
	ID              string
	FP16Bytes       int64
	IndexKFP16Bytes int64 // modeled auxiliary sparse index footprint
	LastAccess      time.Time
	State           KVQuantizationState
	ColdCodec       KVColdCodec
}

type KVQuantizationOptions struct {
	Policy                KVQuantizationThresholds
	EnableColdCompression bool
	ColdAfter             time.Duration
}

type KVQuantizationBackend interface {
	QuantizeKV(id string, from, to KVPrecision) error
	CompressColdKV(id string, codec KVColdCodec) error
}

type KVQuantizationCounters struct {
	Demotions          int64
	Promotions         int64
	ColdCompressions   int64
	TransitionFailures int64
	BytesBefore        int64
	BytesAfter         int64
}

type KVQuantizationOutcome struct {
	Candidate KVQuantizationSpan
	Decision  KVQuantizationTransition
	Metrics   KVQuantizationCounters
	Fallback  bool
	Error     string
}

// ApplyKVQuantization applies one adjacent precision transition, then optionally
// applies cold-sequence compression to an idle low-precision span. Backend errors retain the
// original state and report a fallback rather than fabricating residency gains.
func ApplyKVQuantization(now time.Time, pressure float64, candidate KVQuantizationSpan, cfg KVQuantizationOptions, executor KVQuantizationBackend) KVQuantizationOutcome {
	out := KVQuantizationOutcome{Candidate: candidate}
	before, err := KVQuantizedBytes(candidate.FP16Bytes, candidate.State.Precision)
	if err != nil {
		out.Error = err.Error()
		return out
	}
	out.Metrics.BytesBefore, out.Metrics.BytesAfter = before, before
	decision := ChooseKVQuantization(now, pressure, candidate.State, cfg.Policy)
	out.Decision = decision
	if decision.Change {
		if executor == nil {
			out.Fallback, out.Error, out.Metrics.TransitionFailures = true, "quantization executor unavailable", 1
			return out
		}
		if err := executor.QuantizeKV(candidate.ID, decision.From, decision.To); err != nil {
			out.Fallback, out.Error, out.Metrics.TransitionFailures = true, err.Error(), 1
			return out
		}
		out.Candidate.State.Precision = decision.To
		out.Candidate.State.LastTransition = now
		out.Metrics.BytesAfter, _ = KVQuantizedBytes(candidate.FP16Bytes, decision.To)
		if decision.Reason == "pressure-demote" {
			out.Metrics.Demotions = 1
		} else {
			out.Metrics.Promotions = 1
		}
	}
	coldAfter := cfg.ColdAfter
	if coldAfter <= 0 {
		coldAfter = 5 * time.Minute
	}
	if out.Candidate.ColdCodec == "" {
		out.Candidate.ColdCodec = KVColdCodecNone
	}
	if !cfg.EnableColdCompression || out.Candidate.ColdCodec != KVColdCodecNone || out.Candidate.State.Precision == KVPrecisionFP16 || now.Sub(out.Candidate.LastAccess) < coldAfter {
		return out
	}
	if executor == nil {
		out.Fallback, out.Error, out.Metrics.TransitionFailures = true, "compression executor unavailable", 1
		return out
	}
	if err := executor.CompressColdKV(candidate.ID, KVColdCodecSequence); err != nil {
		out.Fallback, out.Error, out.Metrics.TransitionFailures = true, err.Error(), 1
		return out
	}
	out.Candidate.ColdCodec = KVColdCodecSequence
	out.Metrics.ColdCompressions = 1
	return out
}
