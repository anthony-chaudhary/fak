package qevicteval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"strings"
)

const (
	ContractVersion = "qevicteval/v1"
	RecipeID        = "qevict/arxiv-2608.05326v1"
	RuntimeID       = "qevicteval/trace-replay-v1"
)

type Outcome string

const (
	OutcomeSupported   Outcome = "supported"
	OutcomeUnsupported Outcome = "unsupported"
	OutcomeDelegate    Outcome = "delegate"
)

type Decision string

const (
	DecisionIntegrate Decision = "integrate"
	DecisionAbstain   Decision = "abstain"
)

type EvidenceKind string

const (
	EvidenceModeled  EvidenceKind = "modeled"
	EvidenceObserved EvidenceKind = "observed"
)

type ReasonCode string

const (
	ReasonEvaluated               ReasonCode = "QEVICT_EVALUATED"
	ReasonUnknownContract         ReasonCode = "QEVICT_UNKNOWN_CONTRACT"
	ReasonUnknownRecipe           ReasonCode = "QEVICT_UNKNOWN_RECIPE"
	ReasonInvalidArtifact         ReasonCode = "QEVICT_INVALID_ARTIFACT"
	ReasonInvalidTrace            ReasonCode = "QEVICT_INVALID_TRACE"
	ReasonUnknownRuntime          ReasonCode = "QEVICT_UNKNOWN_RUNTIME"
	ReasonRuntimeEvidenceRequired ReasonCode = "QEVICT_RUNTIME_EVIDENCE_REQUIRED"
	ReasonNoRecoveryBenefit       ReasonCode = "QEVICT_NO_RECOVERY_BENEFIT"
)

type Provenance struct {
	ArtifactID      string `json:"artifact_id"`
	ArtifactVersion string `json:"artifact_version"`
	ArtifactSHA256  string `json:"artifact_sha256"`
	RecipeID        string `json:"recipe_id"`
	RecipeRevision  string `json:"recipe_revision"`
	RecipeSource    string `json:"recipe_source"`
	RuntimeID       string `json:"runtime_id"`
	RuntimeVersion  string `json:"runtime_version"`
}

type WindowEvent struct {
	Step                int     `json:"step"`
	WindowID            string  `json:"window_id"`
	FullBytes           uint64  `json:"full_bytes"`
	QuantizedBytes      uint64  `json:"quantized_bytes"`
	FutureAttentionMass float64 `json:"future_attention_mass"`
	OrdinaryEvicted     bool    `json:"ordinary_evicted"`
	QEvictTier          string  `json:"qevict_tier"` // full, recoverable, or deleted
	Reactivated         bool    `json:"reactivated"`
}

type RuntimeObservation struct {
	Evidence              EvidenceKind `json:"evidence"`
	Platform              string       `json:"platform"`
	Device                string       `json:"device"`
	Command               string       `json:"command"`
	CapturedAt            string       `json:"captured_at"`
	RuntimeArtifactSHA256 string       `json:"runtime_artifact_sha256"`
	OrdinaryLatencyNS     float64      `json:"ordinary_latency_ns"`
	QEvictLatencyNS       float64      `json:"qevict_latency_ns"`
	RecoveryLatencyNS     float64      `json:"recovery_latency_ns"`
}

type Request struct {
	ContractVersion string              `json:"contract_version"`
	Provenance      Provenance          `json:"provenance"`
	Trace           []WindowEvent       `json:"trace"`
	Runtime         *RuntimeObservation `json:"runtime,omitempty"`
}

type Metrics struct {
	OrdinaryPeakBytes        uint64  `json:"ordinary_peak_bytes"`
	QEvictPeakBytes          uint64  `json:"qevict_peak_bytes"`
	RecoverableTierBytes     uint64  `json:"recoverable_tier_bytes"`
	RecoveryReadBytes        uint64  `json:"recovery_read_bytes"`
	RecoveryEvents           int     `json:"recovery_events"`
	OrdinaryFutureMissedMass float64 `json:"ordinary_future_missed_mass"`
	QEvictFutureMissedMass   float64 `json:"qevict_future_missed_mass"`
	AvoidedAttentionDrift    float64 `json:"avoided_attention_drift"`
	OrdinaryLatencyNS        float64 `json:"ordinary_latency_ns"`
	QEvictLatencyNS          float64 `json:"qevict_latency_ns"`
	RecoveryLatencyNS        float64 `json:"recovery_latency_ns"`
	LatencyOverheadNS        float64 `json:"latency_overhead_ns"`
}

type EvidenceSummary struct {
	CapacityAndDrift EvidenceKind `json:"capacity_and_drift"`
	Latency          EvidenceKind `json:"latency"`
	Envelope         string       `json:"envelope"`
}

type Result struct {
	Outcome    Outcome         `json:"outcome"`
	Reason     ReasonCode      `json:"reason"`
	Decision   Decision        `json:"decision"`
	Metrics    Metrics         `json:"metrics"`
	Evidence   EvidenceSummary `json:"evidence"`
	Provenance Provenance      `json:"provenance"`
}

func TraceDigest(trace []WindowEvent) string {
	b, _ := json.Marshal(trace)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func reject(out Outcome, reason ReasonCode, p Provenance) Result {
	return Result{Outcome: out, Reason: reason, Decision: DecisionAbstain, Provenance: p,
		Evidence: EvidenceSummary{CapacityAndDrift: EvidenceModeled, Latency: EvidenceModeled, Envelope: "not evaluated"}}
}

// Invariant: Q-eviction evaluation is fail-closed and deterministic across all evaluation traces.
// Guard: Requests with unknown contract versions, mismatched recipes, or invalid artifact digests are rejected with DecisionAbstain.
// Precondition: Window events in a trace must have strictly monotonic step indices and valid byte allocations.
func Evaluate(req Request) Result {
	p := req.Provenance
	if req.ContractVersion != ContractVersion {
		return reject(OutcomeUnsupported, ReasonUnknownContract, p)
	}
	if p.RecipeID != RecipeID || p.RecipeRevision != "v1" || p.RecipeSource != "https://arxiv.org/abs/2608.05326v1" {
		return reject(OutcomeUnsupported, ReasonUnknownRecipe, p)
	}
	if p.ArtifactID == "" || p.ArtifactVersion == "" || !strings.EqualFold(p.ArtifactSHA256, TraceDigest(req.Trace)) {
		return reject(OutcomeUnsupported, ReasonInvalidArtifact, p)
	}
	var m Metrics
	for i, e := range req.Trace {
		if e.Step != i || e.WindowID == "" || e.FullBytes == 0 || e.QuantizedBytes == 0 || e.QuantizedBytes > e.FullBytes || e.FutureAttentionMass < 0 || e.FutureAttentionMass > 1 || (e.QEvictTier != "full" && e.QEvictTier != "recoverable" && e.QEvictTier != "deleted") || (e.Reactivated && e.QEvictTier != "recoverable") {
			return reject(OutcomeUnsupported, ReasonInvalidTrace, p)
		}
		if !e.OrdinaryEvicted {
			m.OrdinaryPeakBytes += e.FullBytes
		}
		switch e.QEvictTier {
		case "full":
			m.QEvictPeakBytes += e.FullBytes
		case "recoverable":
			m.QEvictPeakBytes += e.QuantizedBytes
			m.RecoverableTierBytes += e.QuantizedBytes
		}
		if e.OrdinaryEvicted {
			m.OrdinaryFutureMissedMass += e.FutureAttentionMass
		}
		if e.QEvictTier == "deleted" {
			m.QEvictFutureMissedMass += e.FutureAttentionMass
		}
		if e.Reactivated {
			m.RecoveryEvents++
			m.RecoveryReadBytes += e.QuantizedBytes
		}
	}
	m.AvoidedAttentionDrift = m.OrdinaryFutureMissedMass - m.QEvictFutureMissedMass
	m.OrdinaryFutureMissedMass = round(m.OrdinaryFutureMissedMass)
	m.QEvictFutureMissedMass = round(m.QEvictFutureMissedMass)
	m.AvoidedAttentionDrift = round(m.AvoidedAttentionDrift)
	if p.RuntimeID != RuntimeID || p.RuntimeVersion == "" {
		return reject(OutcomeDelegate, ReasonUnknownRuntime, p)
	}
	if req.Runtime == nil || req.Runtime.Evidence != EvidenceObserved || req.Runtime.Platform == "" || req.Runtime.Device == "" || req.Runtime.Command == "" || req.Runtime.CapturedAt == "" || req.Runtime.RuntimeArtifactSHA256 == "" {
		r := reject(OutcomeDelegate, ReasonRuntimeEvidenceRequired, p)
		r.Metrics = m
		r.Evidence.Envelope = "bounded trace modeled; latency requires a pinned runtime observation"
		return r
	}
	ro := req.Runtime
	m.OrdinaryLatencyNS, m.QEvictLatencyNS, m.RecoveryLatencyNS = ro.OrdinaryLatencyNS, ro.QEvictLatencyNS, ro.RecoveryLatencyNS
	m.LatencyOverheadNS = round(ro.QEvictLatencyNS - ro.OrdinaryLatencyNS)
	decision, reason := DecisionIntegrate, ReasonEvaluated
	if m.RecoveryEvents == 0 || m.AvoidedAttentionDrift <= 0 {
		decision, reason = DecisionAbstain, ReasonNoRecoveryBenefit
	}
	return Result{Outcome: OutcomeSupported, Reason: reason, Decision: decision, Metrics: m, Provenance: p,
		Evidence: EvidenceSummary{CapacityAndDrift: EvidenceModeled, Latency: EvidenceObserved, Envelope: ro.Platform + "; " + ro.Device + "; bounded fixture only"}}
}

func round(v float64) float64 { return math.Round(v*1e9) / 1e9 }
