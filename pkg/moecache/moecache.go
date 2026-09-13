// Package moecache defines fak's shared routed-expert cache telemetry contract:
// one typed record whose metrics are either MEASURED or UNKNOWN-with-reason, and
// NEVER a fabricated zero.
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/moecache) on
// purpose: the private serving repo may import fak/pkg/* only and never
// fak/internal/* (Core Import Invariant), yet an expert-cache hit/recall record
// must span BOTH repos - the public engine measures the ring, the private
// gateway exposes the metric. Placing the record here lets the same value type,
// tier vocabulary, and reason vocabulary be produced and joined on either side.
//
// This is the LIVE-ENGINE telemetry contract, deliberately distinct from the
// benchmark receipt `fak.moe-expert-cache/v1` (internal/model/expert_cache_receipt.go):
// that schema pins one offline benchmark measurement, this one is the shape a
// running serve reports. They share ONE hit/recall vocabulary - hit rate,
// miss bytes, per-tier bytes read - rather than inventing a second dialect.
//
// The load-bearing invariant (P4, zero hardcoded fabrication): every numeric
// field is a Metric whose zero value is UNKNOWN, not zero. A ring that was never
// built, a window that saw no access, a backend that cannot report overlap, and
// a regret replay that was never asked for each report `known:false` with a
// closed reason string, so a consumer can tell "not measured" from "measured as
// zero" and a phantom 0% can never be surfaced as if it were a reading.
//
// The package has no external dependencies and holds no global state.
package moecache

import (
	"encoding/json"
	"fmt"
)

// Schema identifies the live-engine expert-cache telemetry wire format. It is
// distinct from the benchmark receipt `fak.moe-expert-cache/v1` but shares its
// hit/recall metric vocabulary.
const Schema = "fak.moe-expert-cache-telemetry/v1"

// Tier is one rung of the expert memory waterfall: where a routed expert's
// weights live when the activated set does not fit the fastest tier.
type Tier string

const (
	// TierDRAM is the device/DRAM resident tier - the bounded expert ring.
	TierDRAM Tier = "dram"
	// TierCheckpoint is the host checkpoint tier one rung below the ring.
	TierCheckpoint Tier = "checkpoint"
	// TierNVMe is the streamed backing-store tier the checkpoint reads from.
	TierNVMe Tier = "nvme"
)

// Tiers is the closed, ordered tier vocabulary, fastest first.
var Tiers = [...]Tier{TierDRAM, TierCheckpoint, TierNVMe}

// Valid reports whether t is a member of the closed tier vocabulary.
func (t Tier) Valid() bool {
	for _, known := range Tiers {
		if t == known {
			return true
		}
	}
	return false
}

// The closed UNKNOWN-reason vocabulary. A metric that could not be measured
// names WHY, from this small set, so an absent value is never prose and a
// consumer can branch on the reason.
const (
	// ReasonNoRing: this session has no routed-expert ring at all (the unbounded
	// halW path), so every ring-derived metric is absent.
	ReasonNoRing = "no-ring"
	// ReasonNoMeasuredAccess: a rate whose denominator is zero - nothing was
	// looked up in the window, so there is no hit rate to report.
	ReasonNoMeasuredAccess = "no-measured-access"
	// ReasonBackendNotAsync: the backend advertises no async uploader, so overlap
	// is not a thing it can report (as opposed to "no overlap was achieved").
	ReasonBackendNotAsync = "backend-not-async"
	// ReasonNoReplayTrace: no eviction-regret replay was run - either the caller
	// did not ask, or the window produced no replayable trace.
	ReasonNoReplayTrace = "no-replay-trace"
	// ReasonNoCheckpoint: the model has no checkpoint tier; experts are fully
	// resident, so there is no next-tier IO to account.
	ReasonNoCheckpoint = "no-checkpoint"
	// ReasonNotDerivable: the counters a metric would be derived from do not
	// exist on the live surface, so inventing a value would be fabrication.
	ReasonNotDerivable = "not-derivable"
	// ReasonNoPlan: there is no residency plan to score (no pin-set, no ring).
	ReasonNoPlan = "no-plan"
)

// Metric is one numeric telemetry value that is either KNOWN (a measured
// number) or UNKNOWN with a reason. The zero value is UNKNOWN - a producer
// cannot forget the discipline, because not setting a metric leaves it
// unreported rather than a phantom 0.
//
// It marshals honestly: an unknown metric carries `known:false` and omits the
// number, while a known metric carries `known:true` and its value. Because the
// zero value is unknown, a partially-populated record can never emit a
// fabricated zero for a metric nobody measured.
type Metric[V any] struct {
	Known  bool
	Value  V
	Reason string
}

// Known returns a KNOWN metric carrying v - the constructor a producer uses for
// a value it actually measured.
func Known[V any](v V) Metric[V] {
	return Metric[V]{Known: true, Value: v}
}

// Unknown returns an UNKNOWN metric naming the closed reason it could not be
// measured. An empty reason is normalized to ReasonNotDerivable so a consumer
// never has to handle a reasonless unknown.
func Unknown[V any](reason string) Metric[V] {
	if reason == "" {
		reason = ReasonNotDerivable
	}
	return Metric[V]{Reason: reason}
}

// Get returns the measured value and whether it is known.
func (m Metric[V]) Get() (V, bool) {
	return m.Value, m.Known
}

// ReasonOrEmpty returns the unknown reason, or "" when the metric is known.
func (m Metric[V]) ReasonOrEmpty() string {
	if m.Known {
		return ""
	}
	return m.Reason
}

// metricJSON is the wire shape of every Metric: the number appears only when the
// value is known, so an unknown metric can never serialize a fabricated 0.
type metricJSON struct {
	Known  bool   `json:"known"`
	Value  any    `json:"value,omitempty"`
	Reason string `json:"reason,omitempty"`
}

// MarshalJSON emits `known:false` + reason (and NO value) for an unknown metric,
// and `known:true` + value for a known one. It is the structural guarantee that
// absent is never written as zero.
func (m Metric[V]) MarshalJSON() ([]byte, error) {
	out := metricJSON{Known: m.Known}
	if m.Known {
		out.Value = m.Value
	} else {
		out.Reason = m.Reason
		if out.Reason == "" {
			out.Reason = ReasonNotDerivable
		}
	}
	return json.Marshal(out)
}

// UnmarshalJSON reads the metric back, preserving the known/unknown boundary: an
// object with known:false and no value round-trips to an UNKNOWN metric, never
// to a known zero.
func (m *Metric[V]) UnmarshalJSON(b []byte) error {
	var raw struct {
		Known  bool            `json:"known"`
		Value  json.RawMessage `json:"value"`
		Reason string          `json:"reason"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Known = raw.Known
	m.Value = *new(V)
	m.Reason = ""
	if raw.Known {
		if len(raw.Value) == 0 {
			return fmt.Errorf("moecache: known metric carries no value")
		}
		return json.Unmarshal(raw.Value, &m.Value)
	}
	m.Reason = raw.Reason
	if m.Reason == "" {
		m.Reason = ReasonNotDerivable
	}
	return nil
}

// ModelShape is the static routing shape a rate is read against: a 60% hit rate
// is excellent at k/E = 8/256 and unremarkable at 8/8, so no rate travels
// without it.
type ModelShape struct {
	Experts int `json:"experts"`
	TopK    int `json:"top_k"`
	Layers  int `json:"layers"`
	// ActivatedFraction is TopK/Experts, or 0 when the model declares no expert
	// count. It is the ladder's premise: modern MoE fires a few percent of its
	// parameters per token.
	ActivatedFraction float64 `json:"activated_fraction"`
}

// TierTelemetry is one tier's #1305 axes. Every metric is a Metric, so a tier
// that was not exercised reports unknowns rather than zeros.
type TierTelemetry struct {
	Tier Tier `json:"tier"`
	// CapacityBytes is the tier's byte ceiling; ResidentBytes its instantaneous
	// footprint.
	CapacityBytes Metric[int64] `json:"capacity_bytes"`
	ResidentBytes Metric[int64] `json:"resident_bytes"`
	// HitRate is Hits/(Hits+Misses) over the tier's access counters.
	HitRate Metric[float64] `json:"hit_rate"`
	// ByteWeightedHitRate is HitBytes/(HitBytes+MissBytes) - the hit rate weighted
	// by the bytes each outcome moved rather than by count.
	ByteWeightedHitRate Metric[float64] `json:"byte_weighted_hit_rate"`
	// HitBytes / MissBytes are the numerator and denominator of the byte-weighted
	// rate; MissBytes is the bytes streamed on a miss (the tier's page-in bytes).
	HitBytes  Metric[int64] `json:"hit_bytes"`
	MissBytes Metric[int64] `json:"miss_bytes"`
	// NVMeBytes is the bytes read from the backing store on this tier's behalf
	// (#1305's per-tier bytes read), named consistently with the benchmark
	// receipt's bytes_read_nvme.
	NVMeBytes Metric[int64] `json:"nvme_bytes"`
	// PrefetchPrecision is the share of prefetched weights that were actually
	// activated - the R3 signal a hit rate alone cannot give.
	PrefetchPrecision Metric[float64] `json:"prefetch_precision"`
	// EvictionCount is how many residents this tier paged out.
	EvictionCount Metric[int64] `json:"eviction_count"`
	// BeladyRegret is the eviction regret against the offline Belady oracle
	// (#4233), present only when a replay was run over a real trace.
	BeladyRegret Metric[float64] `json:"belady_regret"`
	// ResidentCount is how many experts (or entries) are resident now.
	ResidentCount Metric[int] `json:"resident_count"`
}

// Coverage carries the coverage/refusal signals already present on
// ExpertRingStats: of the experts the router activated, how many the budget
// could hold, and how many stagings it refused.
type Coverage struct {
	// ActivatedExperts / ActivatedCovered are the R3 coverage meter; Covered <
	// Activated is the direct read on an undersized ring.
	ActivatedExperts Metric[int] `json:"activated_experts"`
	ActivatedCovered Metric[int] `json:"activated_covered"`
	// Refusals / Lookups are the reconciliation pair: a refused staging falls
	// back to permanent halW residency, so a non-zero count means the budget is
	// being quietly abandoned.
	Refusals Metric[int] `json:"refusals"`
	Lookups  Metric[int] `json:"lookups"`
	// AsyncOverlap is the share of fenced transfers already landed when demanded
	// (#5627); absent on a synchronous backend, never a phantom 0.
	AsyncOverlap Metric[float64] `json:"async_overlap"`
}

// ExpertCacheTelemetry is the live-engine expert-cache telemetry record: the
// model shape, the coverage signals, and one TierTelemetry per tier of the
// waterfall.
type ExpertCacheTelemetry struct {
	Schema string     `json:"schema"`
	Shape  ModelShape `json:"shape"`
	// Coverage is the ring-level coverage/refusal signals (shared across tiers).
	Coverage Coverage `json:"coverage"`
	// Tiers carries one entry per tier that the waterfall has, in canonical
	// order. A tier that does not exist is not fabricated: the ring tier is
	// present whenever a ring is enabled, and the checkpoint tier only when the
	// model has one.
	Tiers []TierTelemetry `json:"tiers"`
}

// New returns an empty telemetry record stamped with the schema.
func New() ExpertCacheTelemetry {
	return ExpertCacheTelemetry{Schema: Schema}
}

// Tier returns a pointer to the telemetry for t, creating the slot if absent.
// The returned pointer is inside the slice, so writes through it persist.
func (e *ExpertCacheTelemetry) Tier(t Tier) *TierTelemetry {
	for i := range e.Tiers {
		if e.Tiers[i].Tier == t {
			return &e.Tiers[i]
		}
	}
	e.Tiers = append(e.Tiers, TierTelemetry{Tier: t})
	return &e.Tiers[len(e.Tiers)-1]
}

// Find returns the telemetry for t and whether the tier is present.
func (e ExpertCacheTelemetry) Find(t Tier) (TierTelemetry, bool) {
	for _, tt := range e.Tiers {
		if tt.Tier == t {
			return tt, true
		}
	}
	return TierTelemetry{}, false
}
