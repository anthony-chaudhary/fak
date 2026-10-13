package model

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
)

// expert_cpu_miss.go — CPU-compute-on-miss with an explicit quality gate (ticket #3343,
// DS4.1 P7). The MoE bandwidth partition solver (moe_bandwidth_partition.go, #11038)
// already answers "given a measured PCIe/CPU profile, how many of the K missing experts
// should execute on the host CPU concurrently with the PCIe stream?". This file turns
// that solver into an operational, default-off seam and places two fences around it:
//
//   - a closed QUALITY vocabulary. The shipped CPU-miss path is EXACT: the host reads the
//     exact routed-expert bytes from the checkpoint/page-cache and runs the SAME host
//     k-quant GEMV the resident CPU path runs, so the token is bit-identical to a resident
//     compute. A plan built by that path MUST label itself ExpertMissQualityExact. A lossy
//     variant — Strata's routing-bias approximation, the #1299 shadow fallback, or any
//     sub-precision substitute — is NOT this path, MUST be labelled ExpertMissQualityNonExact,
//     and MUST stay opt-in behind ExpertCPUMissPolicy.NonExact. Nothing in this file selects a
//     lossy arm on its own; the field exists only so a caller that has already chosen to pay
//     the quality delta has a place to say so honestly.
//
//   - a reconciliation LEDGER. Every lookup lands in exactly one of Hits, CPUComputed,
//     Streamed, or Refused, so Hits+CPUComputed+Streamed+Refused == Lookups is an identity a
//     consumer can check rather than trust. The miss-path milliseconds-per-token reading is
//     only ever reported when both a token count and at least one lookup were observed; an
//     unmeasured rate is `known=false`, never a fabricated zero.
//
// UMA BANDWIDTH SHARING. The ticket's first DoD item is the observation that on a single
// unified-memory APU (Strix Halo) the host CPU GEMV and the device DMA draw from the SAME
// physical memory, so the partition solver's assumption of independent PCIe and host-memory
// channels does not hold. MeasureUMABandwidthSharing derives the contention ratios from
// paired solo/concurrent samples. Its provenance follows the ExpertCacheReceipt rule: samples
// captured from a synthetic/replayed trace produce ExpertCacheMeasurementSimulated
// ([SW-VERIFIED]), while only samples captured on a physical appliance may be relabelled
// ExpertCacheMeasurementHardware ([HW-WITNESSED]); a simulated receipt can never masquerade as
// a hardware one.
//
// DEFAULT OFF, BYTE-FOR-BYTE. The zero ExpertCPUMissMode is Off and the zero
// ExpertCPUMissController is disabled: PlanExpertCPUMiss then returns every miss on the
// stream side with no CPU expert, and the unchanged path runs. Fail closed: an enabled policy
// with a malformed profile is an error, never a partially-planned miss.

// ExpertCPUMissMode selects the miss-compute route for one MoE layer. The zero value
// ExpertCPUMissOff is the exact default: the CPU-miss path is never selected and every miss
// is streamed, byte-for-byte the pre-#3343 behavior.
type ExpertCPUMissMode int

const (
	// ExpertCPUMissOff is the zero value — the default. No routed expert is computed on the
	// host CPU; every ring miss streams. Nothing about the shipped path changes.
	ExpertCPUMissOff ExpertCPUMissMode = iota
	// ExpertCPUMissExact enables the exact host CPU-miss path: the solver partitions misses
	// between the stream and the host, and the CPU arm runs the exact routed-expert bytes
	// through the existing host k-quant GEMV. Its answer is bit-identical to a resident
	// compute, so a plan it produces reports ExpertMissQualityExact.
	ExpertCPUMissExact
)

// String names the mode for a report, a flag, or a log line. An unrecognized value is
// reported as "off" rather than panicking, because a diagnostic formatter must not be the
// thing that kills a forward.
func (m ExpertCPUMissMode) String() string {
	switch m {
	case ExpertCPUMissExact:
		return "exact"
	default:
		return "off"
	}
}

// ParseExpertCPUMissMode parses a case-insensitive mode token. It accepts "off", "",
// and "exact" and REFUSES anything else (ok=false) — there is no silent fallback to the
// enabled or disabled arm, because either guess could change what a serve does.
func ParseExpertCPUMissMode(s string) (ExpertCPUMissMode, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "off":
		return ExpertCPUMissOff, true
	case "exact":
		return ExpertCPUMissExact, true
	default:
		return ExpertCPUMissOff, false
	}
}

// ExpertCPUMissEnv is the environment variable that gates the CPU-miss path. It is unset by
// default, and unset, empty, or "off" is the exact default: the CPU-miss path is never
// selected. "exact" enables the exact host CPU-miss path. Any other value is NOT silently
// ignored — ExpertCPUMissModeFromEnv reports ok=false so a caller can refuse the launch rather
// than guess whether a misspelled knob meant to enable or disable the path.
const ExpertCPUMissEnv = "FAK_EXPERT_CPU_MISS"

// ExpertCPUMissModeFromEnv reads ExpertCPUMissEnv and parses it. It returns (Off, true) when
// the variable is unset or empty (the default) and (mode, true) when the value parses. It
// returns (ExpertCPUMissOff, false) ONLY when the value is present AND unparseable, so an
// invalid setting is distinguishable from a deliberate off.
func ExpertCPUMissModeFromEnv() (ExpertCPUMissMode, bool) {
	return ParseExpertCPUMissMode(os.Getenv(ExpertCPUMissEnv))
}

// ExpertCPUMissPolicyFromEnv builds a policy from the ExpertCPUMissEnv flag (default Off) and
// the provided measured profile. An off flag yields a disabled policy and a nil error. An
// unparseable flag value yields an error naming ExpertCPUMissEnv and no policy: it FAILS
// CLOSED rather than falling back, so a misspelled knob can never silently change a serve.
func ExpertCPUMissPolicyFromEnv(profile MoEBandwidthProfile) (ExpertCPUMissPolicy, error) {
	raw := os.Getenv(ExpertCPUMissEnv)
	mode, ok := ParseExpertCPUMissMode(raw)
	if !ok {
		return ExpertCPUMissPolicy{}, fmt.Errorf(
			"model: invalid %s=%q: want unset, \"off\", or \"exact\"",
			ExpertCPUMissEnv, raw)
	}
	if mode == ExpertCPUMissOff {
		return ExpertCPUMissPolicy{}, nil
	}
	return ExpertCPUMissPolicy{Mode: mode, Profile: profile}, nil
}

// DefaultExpertCPUMissPerLayer is the shipped cap on how many of a layer's ring misses may be
// computed on the host CPU at once — the MoE4All cpu_miss.rs:20 bound the ticket ports. A
// policy that leaves MaxCPUMissesPerLayer at 0 (its zero value) gets this cap. The cap exists
// because the unconstrained solver will happily hand the whole miss set to the CPU when the
// measured host bandwidth looks large, which would starve the stream for several tokens and
// is precisely the serial-wait pathology this leaf exists to remove.
const DefaultExpertCPUMissPerLayer = 3

// ExpertCPUMissPolicy is the per-layer miss partition policy: whether the CPU-miss path is
// enabled, how many misses it may take, and the measured profile the solver partitions
// against. The zero value is disabled (ExpertCPUMissOff) and therefore inert; every knob is
// only consulted when Enabled() is true.
type ExpertCPUMissPolicy struct {
	// Mode selects the miss route. ExpertCPUMissOff (zero) disables the path entirely.
	Mode ExpertCPUMissMode
	// MaxCPUMissesPerLayer caps how many misses may run on the host CPU. 0 means
	// DefaultExpertCPUMissPerLayer; a negative value is treated as 0, i.e. the default.
	MaxCPUMissesPerLayer int
	// Profile is the measured hardware transfer/execution rates the partition is solved
	// against. It is validated (MoEBandwidthProfile.Validate) before the solver runs, so a
	// negative rate is an error rather than an unbounded CPU share.
	Profile MoEBandwidthProfile
	// NonExact, when true, asks the caller's own lossy arm to be recorded as
	// ExpertMissQualityNonExact. It does NOT make this file compute a lossy result — this
	// file only labels. A CPU-miss implementation that reads the exact checkpoint bytes and
	// runs the exact host GEMV must leave this false; only a routing-bias approximation
	// (Strata) or a sub-precision substitute (the #1299 shadow fallback) may set it, opting
	// in explicitly to the measured quality delta.
	NonExact bool
}

// Enabled reports whether the CPU-miss path is selected. The zero value returns false, so an
// unset policy is byte-for-byte the pre-#3343 path.
func (p ExpertCPUMissPolicy) Enabled() bool { return p.Mode != ExpertCPUMissOff }

// maxCPUMisses resolves the effective per-layer CPU cap. A non-positive value is the
// documented default; it is never "unbounded".
func (p ExpertCPUMissPolicy) maxCPUMisses() int {
	if p.MaxCPUMissesPerLayer > 0 {
		return p.MaxCPUMissesPerLayer
	}
	return DefaultExpertCPUMissPerLayer
}

// ExpertMissQuality is the closed quality vocabulary of a miss plan. The zero value is Exact:
// the fail-closed default for a plan nobody has explicitly labelled otherwise.
type ExpertMissQuality int

const (
	// ExpertMissQualityExact is the exact CPU-miss host kernel: the checkpoint/page-cache bytes
	// are read verbatim and run through the existing host k-quant GEMV, so the result is
	// bit-identical to the resident compute.
	ExpertMissQualityExact ExpertMissQuality = iota
	// ExpertMissQualityNonExact is any lossy substitute (Strata routing-bias, shadow fallback,
	// sub-precision copy). It must only ever be reported by an explicitly opted-in caller.
	ExpertMissQualityNonExact
)

// String names the quality rung: "exact" or "non-exact".
func (q ExpertMissQuality) String() string {
	switch q {
	case ExpertMissQualityNonExact:
		return "non-exact"
	case ExpertMissQualityExact:
		return "exact"
	}
	// Fail closed: an unknown rung must never render as the strongest claim.
	return "unknown"
}

// ExpertCPUMissPlan is the sealed partition of one layer's ring misses. CPUExperts run on the
// host CPU concurrently with StreamExperts moving over the link; the two slices partition the
// input miss set, and EstimatedCompletionSec is max(CPU, stream) — the bottleneck.
type ExpertCPUMissPlan struct {
	Layer int               `json:"layer"`
	Mode  ExpertCPUMissMode `json:"mode"`
	// CPUExperts are the misses assigned to host CPU execution. Empty when the policy is off.
	CPUExperts []MoEMissCandidate `json:"cpu_experts"`
	// StreamExperts are the misses assigned to the PCIe stream. When the policy is off this is
	// every miss, in input order.
	StreamExperts []MoEMissCandidate `json:"stream_experts"`
	// QStar is the solver's continuous optimal stream fraction, PCIe_BW/(PCIe_BW+CPU_BW).
	QStar float64 `json:"q_star"`
	// EstimatedCPUTimeSec / EstimatedStreamTimeSec are the channel durations at the realized
	// integer split; EstimatedCompletionSec is their max.
	EstimatedCPUTimeSec    float64 `json:"estimated_cpu_time_sec"`
	EstimatedStreamTimeSec float64 `json:"estimated_stream_time_sec"`
	EstimatedCompletionSec float64 `json:"estimated_completion_sec"`
	// NonExact is true only when the caller explicitly selected a lossy arm. The exact
	// CPU-miss host kernel leaves it false.
	NonExact bool `json:"non_exact"`
}

// QualityLabel reports the plan's quality rung. It is Exact unless the plan was explicitly
// built as lossy, so the default CPU-miss path can never be mistaken for an approximation.
func (p ExpertCPUMissPlan) QualityLabel() ExpertMissQuality {
	if p.NonExact {
		return ExpertMissQualityNonExact
	}
	return ExpertMissQualityExact
}

// PlanExpertCPUMiss partitions one layer's ring misses.
//
// Disabled (the zero policy): it returns a plan with Mode Off, CPUExperts empty,
// StreamExperts equal to the input misses, and QualityLabel Exact. The profile is NOT
// validated on this arm — a session that never enabled the path must not start failing on a
// profile it never uses.
//
// Enabled: the profile is validated, SolveBandwidthPartition computes the integer split, and
// the CPU arm is then CAPPED at MaxCPUMissesPerLayer. The solver already sorts misses by
// Recency descending, so capping keeps the highest-Recency CPU experts and moves the
// deterministic lowest-Recency overflow back to the stream side; the channel times are
// recomputed at the realized split. A malformed profile is returned as an error, never
// partially planned.
func (p ExpertCPUMissPolicy) PlanExpertCPUMiss(layer int, misses []MoEMissCandidate) (ExpertCPUMissPlan, error) {
	if !p.Enabled() {
		stream := make([]MoEMissCandidate, len(misses))
		copy(stream, misses)
		return ExpertCPUMissPlan{
			Layer:         layer,
			Mode:          ExpertCPUMissOff,
			CPUExperts:    []MoEMissCandidate{},
			StreamExperts: stream,
		}, nil
	}
	split, err := SolveBandwidthPartition(p.Profile, misses)
	if err != nil {
		return ExpertCPUMissPlan{}, err
	}

	cpu := split.CPUExperts
	stream := split.PCIeExperts
	if limit := p.maxCPUMisses(); len(cpu) > limit {
		// Keep the highest-Recency CPU experts (the front of the solver's descending sort) and
		// return the overflow to the stream. Rebuild rather than mutate the solver's slices.
		keep := make([]MoEMissCandidate, limit)
		copy(keep, cpu[:limit])
		moved := make([]MoEMissCandidate, 0, len(stream)+len(cpu)-limit)
		moved = append(moved, stream...)
		moved = append(moved, cpu[limit:]...)
		cpu = keep
		stream = moved
	}

	// EstimateChannelTimes returns (pcieTime, cpuTime) in that order; bind the stream
	// channel from the PCIe slot, never the other way round.
	streamTime, cpuTime := p.Profile.EstimateChannelTimes(len(stream), len(cpu))
	return ExpertCPUMissPlan{
		Layer:                  layer,
		Mode:                   ExpertCPUMissExact,
		CPUExperts:             cpu,
		StreamExperts:          stream,
		QStar:                  split.QStar,
		EstimatedCPUTimeSec:    cpuTime,
		EstimatedStreamTimeSec: streamTime,
		EstimatedCompletionSec: math.Max(cpuTime, streamTime),
		NonExact:               p.NonExact,
	}, nil
}

// ExpertCPUMissLedger is the miss-path reconciliation accounting. Every lookup is folded into
// exactly one outcome counter, so Hits+CPUComputed+Streamed+Refused == Lookups is an identity
// a consumer can assert instead of assuming. The zero ledger is valid and empty.
type ExpertCPUMissLedger struct {
	// Lookups is every Observe call. The other four counters partition it.
	Lookups     int `json:"lookups"`
	Hits        int `json:"hits"`
	CPUComputed int `json:"cpu_computed"`
	Streamed    int `json:"streamed"`
	Refused     int `json:"refused"`
	// CPUNanos / StreamNanos are the measured channel durations. They are diagnostic sums, not
	// a partition: a single lookup contributes only to the channel that served it.
	CPUNanos    int64 `json:"cpu_nanos"`
	StreamNanos int64 `json:"stream_nanos"`
}

// Observe folds one lookup into the ledger, incrementing EXACTLY one outcome. Classification
// is deterministic and fails closed:
//
//   - Refused: an invalid observation — an unknown mode, a negative duration, or a lookup that
//     reported BOTH a CPU compute and a stream transfer (one miss cannot have been served by
//     both). Invalid input is counted and never mixed into a clean outcome.
//   - Hits: neither channel ran (both durations zero), i.e. the expert was already resident.
//   - CPUComputed: the exact CPU arm ran (mode Exact, CPU duration, no stream duration).
//   - Streamed: the miss went over the link (mode Off, or a stream duration with no CPU work).
func (l *ExpertCPUMissLedger) Observe(mode ExpertCPUMissMode, cpuNanos, streamNanos int64) {
	if l == nil {
		return
	}
	l.Lookups++
	refusedBefore := l.Refused
	switch {
	case mode != ExpertCPUMissOff && mode != ExpertCPUMissExact:
		l.Refused++
	case cpuNanos < 0 || streamNanos < 0:
		l.Refused++
	case cpuNanos == 0 && streamNanos == 0:
		l.Hits++
	case mode == ExpertCPUMissExact && cpuNanos > 0 && streamNanos == 0:
		l.CPUComputed++
	case cpuNanos == 0 && streamNanos > 0:
		l.Streamed++
	default:
		// Both channels reported work for one miss, or the CPU arm ran under the disabled mode:
		// an inconsistent observation, refused rather than guessed.
		l.Refused++
	}
	// Nanos accumulate ONLY for an accepted outcome. A refused row contributes to Refused and
	// nothing else, so invalid input can never inflate the ms/token numerator below.
	if l.Refused == refusedBefore {
		if cpuNanos > 0 {
			l.CPUNanos += cpuNanos
		}
		if streamNanos > 0 {
			l.StreamNanos += streamNanos
		}
	}
}

// Reconciles reports whether the four outcome counters exactly partition Lookups. It is the
// ledger's self-check and must never be false for a ledger written only through Observe.
func (l ExpertCPUMissLedger) Reconciles() bool {
	return l.Hits+l.CPUComputed+l.Streamed+l.Refused == l.Lookups
}

// MissPathMillisPerToken is the miss-path cost per token: (CPUNanos+StreamNanos)/1e6/tokens.
// It returns known=false unless a positive token count AND at least one ACCEPTED outcome
// (CPUComputed or Streamed) was observed. Requiring an accepted outcome rather than a bare
// Lookups>0 is what keeps a ledger of only refusals from reporting a rate: a refused row
// contributes no nanos, so a nonzero Lookups count alone is not evidence of measured work.
func (l ExpertCPUMissLedger) MissPathMillisPerToken(tokens int64) (float64, bool) {
	if tokens <= 0 || l.CPUComputed+l.Streamed <= 0 {
		return 0, false
	}
	return float64(l.CPUNanos+l.StreamNanos) / 1e6 / float64(tokens), true
}

// ExpertCPUMissStats is the ledger's observable snapshot. QualityLabel names the quality rung
// the recorded path ran at; MillisPerToken is populated only by a caller that knows its token
// count and supplies it (ExpertCPUMissController.Stats leaves it unknown and documents the
// separate MissPathMillisPerToken call).
type ExpertCPUMissStats struct {
	Lookups             int     `json:"lookups"`
	Hits                int     `json:"hits"`
	CPUComputed         int     `json:"cpu_computed"`
	Streamed            int     `json:"streamed"`
	Refused             int     `json:"refused"`
	CPUNanos            int64   `json:"cpu_nanos"`
	StreamNanos         int64   `json:"stream_nanos"`
	QualityLabel        string  `json:"quality_label"`
	MillisPerToken      float64 `json:"millis_per_token"`
	MillisPerTokenKnown bool    `json:"millis_per_token_known"`
}

// Stats returns the ledger's counters with the exact quality label. The ledger records the
// exact CPU-miss path; a lossy arm must carry its own label, and a caller that does must
// override QualityLabel rather than inherit this one. MillisPerToken is deliberately unknown
// here: Stats has no token count.
func (l ExpertCPUMissLedger) Stats() ExpertCPUMissStats {
	return ExpertCPUMissStats{
		Lookups:     l.Lookups,
		Hits:        l.Hits,
		CPUComputed: l.CPUComputed,
		Streamed:    l.Streamed,
		Refused:     l.Refused,
		CPUNanos:    l.CPUNanos,
		StreamNanos: l.StreamNanos,
		// The bare ledger cannot know which arm drove it, so it must not overclaim
		// exactness. ExpertCPUMissController.Stats overrides this with the policy's label.
		QualityLabel: "unknown",
	}
}

// ExpertCPUMissController is the standalone, concurrency-safe home for a policy plus its
// ledger. It exists so the seam needs no new field on Session (and therefore no edit to kv.go)
// and can be shared by several goroutines without a session lock. The zero value is a disabled
// controller: Policy.Mode is Off, Enabled() is false, and no miss is ever planned onto the CPU.
type ExpertCPUMissController struct {
	mu     sync.Mutex
	Policy ExpertCPUMissPolicy
	Ledger ExpertCPUMissLedger
}

// Enabled reports whether the controller's policy selects the CPU-miss path.
func (c *ExpertCPUMissController) Enabled() bool {
	if c == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.Policy.Enabled()
}

// Plan partitions one layer's misses under the controller's policy. The policy is read under
// the mutex; the solver itself is pure and allocates its own result.
func (c *ExpertCPUMissController) Plan(layer int, misses []MoEMissCandidate) (ExpertCPUMissPlan, error) {
	if c == nil {
		return ExpertCPUMissPolicy{}.PlanExpertCPUMiss(layer, misses)
	}
	c.mu.Lock()
	policy := c.Policy
	c.mu.Unlock()
	return policy.PlanExpertCPUMiss(layer, misses)
}

// Observe folds one lookup into the controller's ledger.
func (c *ExpertCPUMissController) Observe(mode ExpertCPUMissMode, cpuNanos, streamNanos int64) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.Ledger.Observe(mode, cpuNanos, streamNanos)
	c.mu.Unlock()
}

// Stats returns the controller's ledger snapshot with the quality label implied by its policy.
// A non-exact policy is reported as non-exact even though the ledger itself only records
// channel outcomes, so a lossy controller can never surface an exact label by default.
func (c *ExpertCPUMissController) Stats() ExpertCPUMissStats {
	if c == nil {
		return ExpertCPUMissStats{QualityLabel: "unknown"}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	stats := c.Ledger.Stats()
	// The controller knows its policy's arm, so it can label the ledger honestly in either
	// direction; the bare ledger leaves this "unknown".
	if c.Policy.NonExact {
		stats.QualityLabel = ExpertMissQualityNonExact.String()
	} else {
		stats.QualityLabel = ExpertMissQualityExact.String()
	}
	return stats
}

// ---- UMA bandwidth-sharing measurement (ticket #3343 DoD item 1) ------------------------

// UMABandwidthSharingSchema identifies the UMA contention measurement wire format.
const UMABandwidthSharingSchema = "fak.uma-bandwidth-sharing/v1"

// UMABandwidthSharingSample is one measured interval on a shared-memory APU. A SOLO sample
// (Concurrent=false) ran one engine at a time and supplies the uncontended rate; a CONCURRENT
// sample ran the host CPU GEMV and the device DMA together and supplies the contended rate.
// Durations must be positive for any channel that moved bytes, so a rate is never derived from
// a channel that did not run.
type UMABandwidthSharingSample struct {
	Label    string `json:"label"`
	CPUBytes int64  `json:"cpu_bytes"`
	CPUNanos int64  `json:"cpu_nanos"`
	GPUBytes int64  `json:"gpu_bytes"`
	GPUNanos int64  `json:"gpu_nanos"`
	// Concurrent is true when both engines were active in the same interval.
	Concurrent bool `json:"concurrent"`
}

// UMABandwidthSharingReceipt is the derived contention picture. Rates are GB/s (bytes per
// nanosecond); a contention ratio below 1 is the shared-memory penalty. Measurement follows
// the ExpertCacheReceipt provenance rule: ExpertCacheMeasurementSimulated for a synthetic or
// replayed trace ([SW-VERIFIED]) and ExpertCacheMeasurementHardware only for a physical
// appliance capture ([HW-WITNESSED]). The distinction is structural, never prose.
type UMABandwidthSharingReceipt struct {
	Schema              string  `json:"schema"`
	Samples             int     `json:"samples"`
	SoloCPUGbps         float64 `json:"solo_cpu_gbps"`
	SoloGPUGbps         float64 `json:"solo_gpu_gbps"`
	ConcurrentCPUGbps   float64 `json:"concurrent_cpu_gbps"`
	ConcurrentGPUGbps   float64 `json:"concurrent_gpu_gbps"`
	CPUContentionRatio  float64 `json:"cpu_contention_ratio"`
	GPUContentionRatio  float64 `json:"gpu_contention_ratio"`
	SharedBandwidthGbps float64 `json:"shared_bandwidth_gbps"`
	Measurement         string  `json:"measurement"`
	Notes               string  `json:"notes"`
}

// MeasureUMABandwidthSharing derives the contention receipt from paired solo/concurrent
// samples. It FAILS CLOSED: fewer than two samples, any negative or inconsistent sample, or a
// non-finite derived rate returns an error and no partial receipt.
//
// Derivation: solo rates are the maximum observed bytes/nanos over Concurrent=false samples;
// concurrent rates the maximum over Concurrent=true samples; each contention ratio is
// concurrent/solo and 0 when the solo rate is 0 (unmeasured, never an implied infinity).
// SharedBandwidthGbps is the total concurrent CPU+GPU bytes over the summed wall time of the
// concurrent intervals. Measurement is ExpertCacheMeasurementSimulated ([SW-VERIFIED]); a caller with a
// physical capture may relabel it ExpertCacheMeasurementHardware ([HW-WITNESSED]) explicitly,
// but this pure derivation never asserts hardware it did not witness.
func MeasureUMABandwidthSharing(samples []UMABandwidthSharingSample) (UMABandwidthSharingReceipt, error) {
	if len(samples) < 2 {
		return UMABandwidthSharingReceipt{}, fmt.Errorf(
			"model: UMA bandwidth sharing needs >= 2 samples, got %d", len(samples))
	}
	for i, s := range samples {
		if s.CPUBytes < 0 || s.GPUBytes < 0 || s.CPUNanos < 0 || s.GPUNanos < 0 {
			return UMABandwidthSharingReceipt{}, fmt.Errorf(
				"model: UMA sample %d (%q) has negative bytes or nanos", i, s.Label)
		}
		if s.CPUBytes > 0 && s.CPUNanos == 0 {
			return UMABandwidthSharingReceipt{}, fmt.Errorf(
				"model: UMA sample %d (%q) moved %d CPU bytes with zero CPU nanos", i, s.Label, s.CPUBytes)
		}
		if s.GPUBytes > 0 && s.GPUNanos == 0 {
			return UMABandwidthSharingReceipt{}, fmt.Errorf(
				"model: UMA sample %d (%q) moved %d GPU bytes with zero GPU nanos", i, s.Label, s.GPUBytes)
		}
		if s.CPUNanos == 0 && s.GPUNanos == 0 {
			return UMABandwidthSharingReceipt{}, fmt.Errorf(
				"model: UMA sample %d (%q) measured no duration", i, s.Label)
		}
	}

	var (
		soloCPU, soloGPU  float64
		concCPU, concGPU  float64
		concBytes         int64
		totalConcNanos    int64
		concurrentSamples int
	)
	for _, s := range samples {
		if !s.Concurrent {
			if s.CPUNanos > 0 {
				if r := float64(s.CPUBytes) / float64(s.CPUNanos); r > soloCPU {
					soloCPU = r
				}
			}
			if s.GPUNanos > 0 {
				if r := float64(s.GPUBytes) / float64(s.GPUNanos); r > soloGPU {
					soloGPU = r
				}
			}
			continue
		}
		concurrentSamples++
		concBytes += s.CPUBytes + s.GPUBytes
		if s.CPUNanos > 0 {
			if r := float64(s.CPUBytes) / float64(s.CPUNanos); r > concCPU {
				concCPU = r
			}
		}
		if s.GPUNanos > 0 {
			if r := float64(s.GPUBytes) / float64(s.GPUNanos); r > concGPU {
				concGPU = r
			}
		}
		if d := s.CPUNanos; s.GPUNanos > d {
			totalConcNanos += s.GPUNanos
		} else {
			totalConcNanos += d
		}
	}

	var cpuRatio, gpuRatio, shared float64
	if soloCPU > 0 {
		cpuRatio = concCPU / soloCPU
	}
	if soloGPU > 0 {
		gpuRatio = concGPU / soloGPU
	}
	// Shared bandwidth is the total concurrent CPU+GPU bytes over the summed wall time of the
	// concurrent intervals -- the aggregate rate the two engines together sustained.
	if totalConcNanos > 0 {
		shared = float64(concBytes) / float64(totalConcNanos)
	}
	for _, v := range []float64{soloCPU, soloGPU, concCPU, concGPU, cpuRatio, gpuRatio, shared} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return UMABandwidthSharingReceipt{}, fmt.Errorf(
				"model: UMA bandwidth sharing derived a non-finite rate")
		}
	}

	notes := fmt.Sprintf("%d samples; %d concurrent", len(samples), concurrentSamples)
	if concurrentSamples == 0 {
		notes = fmt.Sprintf("%d samples; no concurrent sample, contention unmeasured", len(samples))
	}
	return UMABandwidthSharingReceipt{
		Schema:              UMABandwidthSharingSchema,
		Samples:             len(samples),
		SoloCPUGbps:         soloCPU,
		SoloGPUGbps:         soloGPU,
		ConcurrentCPUGbps:   concCPU,
		ConcurrentGPUGbps:   concGPU,
		CPUContentionRatio:  cpuRatio,
		GPUContentionRatio:  gpuRatio,
		SharedBandwidthGbps: shared,
		Measurement:         ExpertCacheMeasurementSimulated,
		Notes:               notes,
	}, nil
}

// ---- Quality panel: NLL / perplexity delta, exact vs non-exact (ticket #3343 DoD item 3) --
//
// The quality vocabulary above labels a miss plan; this panel carries the MEASURED quality
// delta that justifies the label, so a report states both the rung and the number rather than
// asserting exactness. Every derivation fails closed: a delta that cannot be formed from a
// positive token count and finite means reports known=false, never a NaN or a fabricated zero.
//
// THE STRATA WARNING. Strata-DSV4.1 measured 26 tok/s at perplexity 2.38->2.87 -- a
// routing-bias arm that is emphatically NOT the exact CPU-miss kernel. Any such arm MUST carry
// ExpertMissQualityNonExact and this panel MUST NOT be used to promote it to a default: a
// panel whose Label is non-exact is inadmissible as exact evidence even when its numbers look
// small, and NewMissQualityPanel can never upgrade the label it is handed. The warning is
// ticket #3343's guard against citing a routing-bias speed/quality point as the shipped path.

// MissQualityPanel is the measured baseline-vs-candidate quality delta of one miss path. NLL
// values are negative summed log-probabilities; token counts are the denominators behind them.
// The zero value is unmeasured: every derivation returns known=false because both token counts
// are 0.
type MissQualityPanel struct {
	BaselineNLL     float64 `json:"baseline_nll"`
	CandidateNLL    float64 `json:"candidate_nll"`
	BaselineTokens  int     `json:"baseline_tokens"`
	CandidateTokens int     `json:"candidate_tokens"`
}

// NLLDeltaPerToken is the length-normalized NLL delta per token:
// (CandidateNLL/CandidateTokens) - (BaselineNLL/BaselineTokens). It returns known=false unless
// BOTH token counts are positive, so unequal or unmeasured corpora can never masquerade as a
// per-token comparison. A non-finite result is likewise reported as known=false, never NaN.
func (p MissQualityPanel) NLLDeltaPerToken() (float64, bool) {
	if p.BaselineTokens <= 0 || p.CandidateTokens <= 0 {
		return 0, false
	}
	delta := p.CandidateNLL/float64(p.CandidateTokens) - p.BaselineNLL/float64(p.BaselineTokens)
	if math.IsNaN(delta) || math.IsInf(delta, 0) {
		return 0, false
	}
	return delta, true
}

// PerplexityDelta is exp(candidate mean NLL) - exp(baseline mean NLL), the perplexity move the
// Strata reading reports as 2.38->2.87. It returns known=false unless BOTH token counts are
// positive and BOTH means are finite; a non-finite mean, a non-finite exponentiation, or a
// non-finite difference is rejected as known=false rather than surfaced as NaN.
func (p MissQualityPanel) PerplexityDelta() (float64, bool) {
	if p.BaselineTokens <= 0 || p.CandidateTokens <= 0 {
		return 0, false
	}
	baselineMean := p.BaselineNLL / float64(p.BaselineTokens)
	candidateMean := p.CandidateNLL / float64(p.CandidateTokens)
	if math.IsNaN(baselineMean) || math.IsInf(baselineMean, 0) ||
		math.IsNaN(candidateMean) || math.IsInf(candidateMean, 0) {
		return 0, false
	}
	delta := math.Exp(candidateMean) - math.Exp(baselineMean)
	if math.IsNaN(delta) || math.IsInf(delta, 0) {
		return 0, false
	}
	return delta, true
}

// QualityPanel binds a plan's quality rung to its measured delta. The Label is the plan's own
// ExpertMissQuality; the Panel is the measurement. Verdict is the admissibility rule: the panel
// is only admissible as EXACT evidence when Label == ExpertMissQualityExact. A non-exact label
// makes the whole panel inadmissible, no matter how favorable its numbers, which is what keeps
// a Strata routing-bias arm from being promoted to a default on the strength of a small delta.
type QualityPanel struct {
	Label ExpertMissQuality `json:"label"`
	Panel MissQualityPanel  `json:"panel"`
}

// Verdict reports whether the panel is admissible as exact evidence. It requires BOTH an exact
// label AND an actually measured panel: the zero QualityPanel is labelled Exact (iota 0), so a
// label-only check would admit an unmeasured panel as proof of exactness. Both token counts
// must be positive and the NLL delta must be known and finite. A lossy arm -- Strata
// routing-bias, shadow fallback, sub-precision substitute -- must stay
// ExpertMissQualityNonExact and therefore returns false; the rule is about the label AND the
// measurement, not the magnitude, so a small non-exact delta is still non-exact.
func (q QualityPanel) Verdict() (ok bool) {
	if !q.Exact() {
		return false
	}
	delta, known := q.Panel.NLLDeltaPerToken()
	return known && !math.IsNaN(delta) && !math.IsInf(delta, 0)
}

// Exact reports whether the panel carries the exact quality label. It is the predicate behind
// Verdict, exposed so a caller can branch without restating the comparison.
func (q QualityPanel) Exact() bool {
	return q.Label == ExpertMissQualityExact
}

// NewMissQualityPanel folds summed log-probabilities and token counts into a QualityPanel. NLL
// is the negated summed log-probability, so a caller hands over the log-likelihood it already
// accumulated. Negative token counts clamp to 0 (i.e. unmeasured), and the Label is carried
// VERBATIM: the caller owns the honesty of the label, but this helper can NEVER upgrade a
// non-exact arm to expert. Pair it with Verdict before any promotion claim.
func NewMissQualityPanel(baselineSumLogprob float64, baselineTokens int, candidateSumLogprob float64, candidateTokens int, label ExpertMissQuality) QualityPanel {
	if baselineTokens < 0 {
		baselineTokens = 0
	}
	if candidateTokens < 0 {
		candidateTokens = 0
	}
	return QualityPanel{
		Label: label,
		Panel: MissQualityPanel{
			BaselineNLL:     -baselineSumLogprob,
			CandidateNLL:    -candidateSumLogprob,
			BaselineTokens:  baselineTokens,
			CandidateTokens: candidateTokens,
		},
	}
}
