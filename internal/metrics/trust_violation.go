package metrics

import (
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// trust_violation.go — mechanical boundary dimensions for TRUST_VIOLATION metrics (#11486).
//
// Historically, TRUST_VIOLATION functioned as a coarse catch-all refusal across
// orthogonal subsystems (prompt-injection quarantine, IFC taint egress sinks,
// integrity witness refutations, and compute residency boundaries). In Prometheus
// and Grafana, this collapsed distinct security-relevant events into a single
// opaque counter.
//
// This file disambiguates TRUST_VIOLATION occurrences into a closed vocabulary of
// mechanical boundary dimensions (refusal_subtype label) based on Verdict.Meta,
// decider (By), Reason, and refusal detail attributes, enforcing strict label
// cardinality control and fail-safe generic fallback.

// TrustViolationSubtype represents the mechanical boundary dimension for a
// TRUST_VIOLATION occurrence.
type TrustViolationSubtype string

const (
	// SubtypeInjectionQuarantine represents prompt-injection or untrusted tool results
	// quarantined by the context-MMU, normgate, or wirescreen filters.
	SubtypeInjectionQuarantine TrustViolationSubtype = "injection_quarantine"

	// SubtypeIFCSink represents sensitive egress sinks fed tainted session data
	// blocked by Information Flow Control (IFC).
	SubtypeIFCSink TrustViolationSubtype = "ifc_sink"

	// SubtypeWitnessRefuted represents integrity verification failures where an
	// external or declared witness actively refuted the agent's claim or effect.
	SubtypeWitnessRefuted TrustViolationSubtype = "witness_refuted"

	// SubtypeResidencyMismatch represents tenant-isolated or sensitivity-tagged
	// payloads routed to unauthorized remote or off-box compute engines.
	SubtypeResidencyMismatch TrustViolationSubtype = "residency_mismatch"

	// SubtypeGeneric is the bounded fallback categorization for unattributed,
	// legacy, or out-of-vocabulary trust violations.
	SubtypeGeneric TrustViolationSubtype = "generic"
)

// Canonical metric and label identifiers.
const (
	// TrustViolationMetricName is the Prometheus counter family name for
	// disambiguated trust violations.
	TrustViolationMetricName = "fak_trust_violations_total"

	// RefusalSubtypeLabel is the Prometheus label key for the mechanical boundary dimension.
	RefusalSubtypeLabel = "refusal_subtype"

	// ReasonLabel is the Prometheus label key for the refusal reason.
	ReasonLabel = "reason"
)

var trustViolationSubtypeOrder = map[TrustViolationSubtype]int{
	SubtypeInjectionQuarantine: 0,
	SubtypeIFCSink:             1,
	SubtypeWitnessRefuted:      2,
	SubtypeResidencyMismatch:   3,
	SubtypeGeneric:             4,
}

// Valid reports whether s is a recognized member of the closed subtype vocabulary.
func (s TrustViolationSubtype) Valid() bool {
	_, ok := trustViolationSubtypeOrder[s]
	return ok
}

// Subtypes returns the closed vocabulary of subtypes in deterministic canonical order.
func Subtypes() []TrustViolationSubtype {
	return []TrustViolationSubtype{
		SubtypeInjectionQuarantine,
		SubtypeIFCSink,
		SubtypeWitnessRefuted,
		SubtypeResidencyMismatch,
		SubtypeGeneric,
	}
}

// NormalizeTrustViolationSubtype maps any input string to the closed set,
// enforcing strict label cardinality control. Any unrecognized, malformed,
// or empty value strictly folds to SubtypeGeneric.
func NormalizeTrustViolationSubtype(s string) TrustViolationSubtype {
	trimmed := strings.TrimSpace(strings.ToLower(s))
	sub := TrustViolationSubtype(trimmed)
	if sub.Valid() {
		return sub
	}
	return SubtypeGeneric
}

// ClassifyTrustViolation maps an abi.Verdict into its mechanical boundary refusal subtype.
func ClassifyTrustViolation(v abi.Verdict) TrustViolationSubtype {
	// 1. Explicit subtype declared in Verdict.Meta takes precedence if valid.
	if len(v.Meta) > 0 {
		for _, key := range []string{"refusal_subtype", "subtype"} {
			if val, ok := v.Meta[key]; ok && strings.TrimSpace(val) != "" {
				sub := NormalizeTrustViolationSubtype(val)
				if sub != SubtypeGeneric {
					return sub
				}
			}
		}
	}

	var claim string
	if wp, ok := v.Payload.(abi.WitnessPayload); ok {
		claim = wp.Claim
	}

	return classifyMechanicalBoundary(v.By, v.Kind, v.Reason, v.Meta, claim, v.Disposition)
}

// ClassifyTrustViolationDetails classifies refusal detail attributes into a TrustViolationSubtype.
func ClassifyTrustViolationDetails(by string, meta map[string]string, detail string) TrustViolationSubtype {
	if len(meta) > 0 {
		for _, key := range []string{"refusal_subtype", "subtype"} {
			if val, ok := meta[key]; ok && strings.TrimSpace(val) != "" {
				sub := NormalizeTrustViolationSubtype(val)
				if sub != SubtypeGeneric {
					return sub
				}
			}
		}
	}
	return classifyMechanicalBoundary(by, abi.VerdictDeny, abi.ReasonTrustViolation, meta, detail, "")
}

// ClassifyTrustViolationAttributes extracts subtype from an arbitrary attribute map.
func ClassifyTrustViolationAttributes(attrs map[string]string) TrustViolationSubtype {
	if attrs == nil {
		return SubtypeGeneric
	}
	by := attrs["by"]
	detail := attrs["detail"]
	if detail == "" {
		detail = attrs["claim"]
	}
	return ClassifyTrustViolationDetails(by, attrs, detail)
}

func classifyMechanicalBoundary(by string, kind abi.VerdictKind, reason abi.ReasonCode, meta map[string]string, detail string, disp string) TrustViolationSubtype {
	byLower := strings.ToLower(strings.TrimSpace(by))
	detailLower := strings.ToLower(strings.TrimSpace(detail))

	// 1. Injection Quarantine:
	// - VerdictKind is Quarantine
	// - Reason is PromptInjection
	// - Decider is context-MMU, normgate, wirescreen, or semantic screen
	// - Meta contains quarantine ID, admit=quarantined, or injection/threat markers
	if kind == abi.VerdictQuarantine || reason == abi.ReasonPromptInjection {
		return SubtypeInjectionQuarantine
	}
	if byLower == "ctxmmu" || byLower == "normgate" || byLower == "wirescreen" ||
		strings.HasPrefix(byLower, "semantic:") || strings.Contains(byLower, "semantic_screen") {
		return SubtypeInjectionQuarantine
	}
	if len(meta) > 0 {
		if meta["quarantine_id"] != "" || meta["admit"] == "quarantined" ||
			meta["threat"] != "" || meta["injection"] != "" || meta["prompt_injection"] != "" {
			return SubtypeInjectionQuarantine
		}
		if det := strings.ToLower(meta["detector"]); strings.Contains(det, "injection") || strings.Contains(det, "normgate") || strings.Contains(det, "wirescreen") {
			return SubtypeInjectionQuarantine
		}
		if clm := strings.ToLower(meta["claim"]); strings.Contains(clm, "injection") || strings.Contains(clm, "quarantine") {
			return SubtypeInjectionQuarantine
		}
	}
	if strings.Contains(detailLower, "injection") || strings.Contains(detailLower, "quarantine") || strings.Contains(detailLower, "poison") {
		return SubtypeInjectionQuarantine
	}

	// 2. IFC Sink (Taint Egress):
	// - Reason is TaintEgress
	// - Decider is ifc-sink, ifc-stamp, ifc-scope-ceiling
	// - Meta contains ifc_sink, ifc_taint, taint, sink, flow
	if reason == abi.ReasonTaintEgress {
		return SubtypeIFCSink
	}
	if strings.HasPrefix(byLower, "ifc") || byLower == "ifc-sink" || byLower == "ifc-stamp" || byLower == "ifc-scope-ceiling" {
		return SubtypeIFCSink
	}
	if len(meta) > 0 {
		if meta["ifc_sink"] != "" || meta["ifc_taint"] != "" || meta["ifc_taint_ceiling"] != "" ||
			meta["taint_source_tool"] != "" || meta["sink"] != "" || meta["flow"] != "" || meta["taint"] != "" {
			return SubtypeIFCSink
		}
		if fix := strings.ToLower(meta["fix"]); strings.Contains(fix, "ifc") {
			return SubtypeIFCSink
		}
	}
	if strings.Contains(detailLower, "ifc") || strings.Contains(detailLower, "taint") || strings.Contains(detailLower, "sink fed") {
		return SubtypeIFCSink
	}

	// 3. Witness Refuted (Integrity Refuted):
	// - Reason is ReasonIntegrityRefuted
	// - Decider is require-witness, witness-gate, assumecheck
	// - Detail/claim or meta mentions refutation
	if reason == abi.ReasonIntegrityRefuted {
		return SubtypeWitnessRefuted
	}
	if byLower == "require-witness" || byLower == "witness-gate" || byLower == "assumecheck" || strings.Contains(byLower, "witness") {
		return SubtypeWitnessRefuted
	}
	if len(meta) > 0 {
		if meta["witness_refuted"] != "" || meta["refuted"] == "true" || meta["witness_verdict"] == "refuted" {
			return SubtypeWitnessRefuted
		}
		if strings.Contains(strings.ToLower(meta["witness"]), "refut") ||
			strings.Contains(strings.ToLower(meta["reason"]), "refut") ||
			strings.Contains(strings.ToLower(meta["detail"]), "refut") {
			return SubtypeWitnessRefuted
		}
	}
	if strings.Contains(detailLower, "refut") {
		return SubtypeWitnessRefuted
	}

	// 4. Residency Mismatch (Scope Crossing):
	// - Reason is ReasonScopeCrossing
	// - Decider is engine-residency, residencyGate
	// - Meta contains engine_route, residency_mismatch, residency_fault, scope=tenant
	if reason == abi.ReasonScopeCrossing {
		return SubtypeResidencyMismatch
	}
	if byLower == "engine-residency" || byLower == "residencygate" || strings.Contains(byLower, "residency") {
		return SubtypeResidencyMismatch
	}
	if len(meta) > 0 {
		if meta["residency_mismatch"] != "" || meta["residency_fault"] != "" || meta["residency"] != "" {
			return SubtypeResidencyMismatch
		}
		if meta["engine_route"] != "" || (meta["scope"] == "tenant" && meta["remote"] == "true") {
			return SubtypeResidencyMismatch
		}
	}
	if strings.Contains(detailLower, "residency") || strings.Contains(detailLower, "remote engine") {
		return SubtypeResidencyMismatch
	}

	// 5. Fallback:
	return SubtypeGeneric
}

// TrustViolationEvent represents one witnessed trust violation occurrence.
type TrustViolationEvent struct {
	Subtype TrustViolationSubtype `json:"refusal_subtype"`
	Count   uint64                `json:"count"`
}

// TrustViolationSubtypeTally records the count for one mechanical subtype.
type TrustViolationSubtypeTally struct {
	Subtype TrustViolationSubtype `json:"refusal_subtype"`
	Count   uint64                `json:"count"`
}

// TrustViolationReport is the aggregated breakdown of trust violations by mechanical boundary dimension.
type TrustViolationReport struct {
	Total     uint64                       `json:"total"`
	BySubtype []TrustViolationSubtypeTally `json:"by_subtype"`
}

// Tally returns the count for a given subtype.
func (r TrustViolationReport) Tally(subtype TrustViolationSubtype) uint64 {
	sub := NormalizeTrustViolationSubtype(string(subtype))
	for _, t := range r.BySubtype {
		if t.Subtype == sub {
			return t.Count
		}
	}
	return 0
}

// OpenMetricFamilies lowers the report onto Prometheus as a counter family
// with granular refusal_subtype label dimensions.
func (r TrustViolationReport) OpenMetricFamilies() []OpenMetricFamily {
	samples := make([]OpenMetricSample, 0, len(r.BySubtype))
	for _, t := range r.BySubtype {
		labels := []OpenMetricLabel{
			{Name: RefusalSubtypeLabel, Value: string(t.Subtype)},
		}
		samples = append(samples, OpenMetricSample{
			Labels: labels,
			Value:  float64(t.Count),
		})
	}
	return []OpenMetricFamily{
		{
			Name:    TrustViolationMetricName,
			Help:    "Trust violation occurrences disambiguated by mechanical boundary refusal subtype (injection_quarantine, ifc_sink, witness_refuted, residency_mismatch, generic).",
			Type:    OpenMetricCounter,
			Samples: samples,
		},
	}
}

// OpenMetricFamiliesWithReason lowers the report onto Prometheus including both
// reason="TRUST_VIOLATION" and refusal_subtype labels.
func (r TrustViolationReport) OpenMetricFamiliesWithReason() []OpenMetricFamily {
	samples := make([]OpenMetricSample, 0, len(r.BySubtype))
	for _, t := range r.BySubtype {
		labels := []OpenMetricLabel{
			{Name: ReasonLabel, Value: "TRUST_VIOLATION"},
			{Name: RefusalSubtypeLabel, Value: string(t.Subtype)},
		}
		samples = append(samples, OpenMetricSample{
			Labels: labels,
			Value:  float64(t.Count),
		})
	}
	return []OpenMetricFamily{
		{
			Name:    TrustViolationMetricName,
			Help:    "Trust violation occurrences disambiguated by mechanical boundary refusal subtype (injection_quarantine, ifc_sink, witness_refuted, residency_mismatch, generic).",
			Type:    OpenMetricCounter,
			Samples: samples,
		},
	}
}

// Prometheus renders the report as OpenMetrics/Prometheus text exposition.
func (r TrustViolationReport) Prometheus() string {
	if len(r.BySubtype) == 0 {
		return ""
	}
	var b strings.Builder
	for _, t := range r.BySubtype {
		b.WriteString(TrustViolationMetricName)
		b.WriteString("{")
		b.WriteString(RefusalSubtypeLabel)
		b.WriteString("=\"")
		b.WriteString(string(t.Subtype))
		b.WriteString("\"} ")
		b.WriteString(strconv.FormatUint(t.Count, 10))
		b.WriteByte('\n')
	}
	return b.String()
}

// FoldTrustViolations folds a slice of events into a deterministic report in canonical order.
func FoldTrustViolations(events []TrustViolationEvent) TrustViolationReport {
	tally := make(map[TrustViolationSubtype]uint64)
	var total uint64
	for _, e := range events {
		sub := NormalizeTrustViolationSubtype(string(e.Subtype))
		cnt := e.Count
		if cnt == 0 {
			cnt = 1
		}
		total += cnt
		tally[sub] += cnt
	}
	subtypes := Subtypes()
	bySubtype := make([]TrustViolationSubtypeTally, 0, len(subtypes))
	for _, s := range subtypes {
		if c, ok := tally[s]; ok && c > 0 {
			bySubtype = append(bySubtype, TrustViolationSubtypeTally{
				Subtype: s,
				Count:   c,
			})
		}
	}
	return TrustViolationReport{
		Total:     total,
		BySubtype: bySubtype,
	}
}

// TrustViolationRecorder tracks trust violation occurrences. It is safe for concurrent use.
type TrustViolationRecorder struct {
	mu     sync.Mutex
	counts map[TrustViolationSubtype]uint64
	total  uint64
}

// NewTrustViolationRecorder creates an initialized, thread-safe recorder.
func NewTrustViolationRecorder() *TrustViolationRecorder {
	return &TrustViolationRecorder{
		counts: make(map[TrustViolationSubtype]uint64),
	}
}

// Record observes an abi.Verdict. If it is a non-allow verdict under TRUST_VIOLATION
// (or its sub-cases), it classifies the subtype, increments the tally, and returns the subtype.
// If the verdict is Allow or Defer, it returns an empty string without recording.
func (r *TrustViolationRecorder) Record(v abi.Verdict) TrustViolationSubtype {
	if v.Kind == abi.VerdictAllow || v.Kind == abi.VerdictDefer {
		return ""
	}
	sub := ClassifyTrustViolation(v)
	r.RecordSubtype(sub, 1)
	return sub
}

// RecordDetails observes refusal detail attributes, increments the tally, and returns the subtype.
func (r *TrustViolationRecorder) RecordDetails(by string, meta map[string]string, detail string) TrustViolationSubtype {
	sub := ClassifyTrustViolationDetails(by, meta, detail)
	r.RecordSubtype(sub, 1)
	return sub
}

// RecordSubtype increments the tally for a specific subtype by n.
func (r *TrustViolationRecorder) RecordSubtype(subtype TrustViolationSubtype, n uint64) {
	if n == 0 {
		n = 1
	}
	sub := NormalizeTrustViolationSubtype(string(subtype))
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.counts == nil {
		r.counts = make(map[TrustViolationSubtype]uint64)
	}
	r.counts[sub] += n
	r.total += n
}

// Counts returns a copy of the current subtype counts.
func (r *TrustViolationRecorder) Counts() map[TrustViolationSubtype]uint64 {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[TrustViolationSubtype]uint64, len(r.counts))
	for k, v := range r.counts {
		out[k] = v
	}
	return out
}

// Report produces a deterministic snapshot report ordered canonically.
func (r *TrustViolationRecorder) Report() TrustViolationReport {
	if r == nil {
		return TrustViolationReport{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	subtypes := Subtypes()
	bySubtype := make([]TrustViolationSubtypeTally, 0, len(r.counts))
	for _, s := range subtypes {
		if c, ok := r.counts[s]; ok && c > 0 {
			bySubtype = append(bySubtype, TrustViolationSubtypeTally{
				Subtype: s,
				Count:   c,
			})
		}
	}
	return TrustViolationReport{
		Total:     r.total,
		BySubtype: bySubtype,
	}
}

// Reset clears all recorded counts.
func (r *TrustViolationRecorder) Reset() {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.counts = make(map[TrustViolationSubtype]uint64)
	r.total = 0
}

// TrustViolationLabels returns the OpenMetricLabel slice for a subtype.
func TrustViolationLabels(subtype TrustViolationSubtype) []OpenMetricLabel {
	sub := NormalizeTrustViolationSubtype(string(subtype))
	return []OpenMetricLabel{
		{Name: RefusalSubtypeLabel, Value: string(sub)},
	}
}

// TrustViolationMetricLabels returns a map of Prometheus labels for the given refusal subtype.
func TrustViolationMetricLabels(subtype TrustViolationSubtype) map[string]string {
	sub := NormalizeTrustViolationSubtype(string(subtype))
	return map[string]string{
		ReasonLabel:         "TRUST_VIOLATION",
		RefusalSubtypeLabel: string(sub),
	}
}
