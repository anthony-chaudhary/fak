package fitplan

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/compute/strix"
)

// Topology says how the device pool relates to host RAM, which decides the weight rows the
// pool is charged.
type Topology string

const (
	TopologyDiscrete Topology = "discrete"  // device pool independent of host RAM
	TopologyUnified  Topology = "unified"   // device and host share ONE physical pool (APU)
	TopologyHostOnly Topology = "host_only" // no device backend; host RAM is the pool
)

// sharesHostPool reports whether host-scoped rows draw on the same pool as the KV store.
func (t Topology) sharesHostPool() bool { return t != TopologyDiscrete }

// Bound names the resource that sets a prediction's rate.
type Bound string

const (
	BoundMemory  Bound = "memory"
	BoundCompute Bound = "compute"
	BoundLink    Bound = "link"
)

// Omission is the typed reason a number is absent from a report.
type Omission string

const (
	OmitCalibrationInvalid      Omission = "calibration_invalid"
	OmitMoEActiveSetUnknown     Omission = "moe_active_set_unknown"
	OmitModelStreamUnknown      Omission = "model_stream_unknown"
	OmitStreamBandwidthUnknown  Omission = "stream_bandwidth_unknown"
	OmitOffloadBandwidthUnknown Omission = "offload_bandwidth_unknown"
	OmitPeakBandwidthUnknown    Omission = "peak_bandwidth_unknown"
	OmitPeakComputeUnknown      Omission = "peak_compute_unknown"
	OmitPoolCapacityUnknown     Omission = "pool_capacity_unknown"
	OmitHostPoolCapacityUnknown Omission = "host_pool_capacity_unknown"
	OmitModelFLOPsUnknown       Omission = "model_flops_unknown"
	OmitKVGeometryUnknown       Omission = "kv_geometry_unknown"
	OmitModelFactsUnavailable   Omission = "model_facts_unavailable"
	OmitContextUnknown          Omission = "context_unknown"
)

// Tier is the evidence tier of a report's efficiencies.
type Tier string

const (
	TierSOL        Tier = "sol"
	TierEmpirical  Tier = "empirical"
	TierCalibrated Tier = "calibrated" // fit from this host's receipts, keyed by hardware epoch
	// TierSilicon marks a row fitted from an on-silicon decode confirm (the first-N-token
	// probe a serve runs after load): the measurement ran on this machine's own engine,
	// not a foreign bench receipt ingested for calibration.
	TierSilicon Tier = "silicon"
)

// Status is the planner's overall verdict.
type Status string

const (
	StatusFits        Status = "fits"
	StatusRefused     Status = "refused"
	StatusUnknown     Status = "unknown"
	StatusUnavailable Status = "unavailable"
)

// RefusalReason is the closed vocabulary of why no candidate fits.
type RefusalReason string

const (
	ReasonWeightsExceedPool      RefusalReason = "weights_exceed_pool"
	ReasonContextExceedsPool     RefusalReason = "context_exceeds_pool"
	ReasonConcurrencyExceedsPool RefusalReason = "concurrency_exceeds_pool"
	ReasonPinUnmatched           RefusalReason = "pin_unmatched" // no candidate carries the pinned config
)

var (
	ErrNoFit                  = errors.New("fitplan: no candidate fits the workload")
	ErrWeightsExceedPool      = errors.New("fitplan: weights_exceed_pool")
	ErrContextExceedsPool     = errors.New("fitplan: context_exceeds_pool")
	ErrConcurrencyExceedsPool = errors.New("fitplan: concurrency_exceeds_pool")
	ErrPinUnmatched           = errors.New("fitplan: pin_unmatched")
)

// Sentinel is the per-reason sentinel a Refusal unwraps to; an unknown reason is ErrNoFit.
func (r RefusalReason) Sentinel() error {
	switch r {
	case ReasonWeightsExceedPool:
		return ErrWeightsExceedPool
	case ReasonContextExceedsPool:
		return ErrContextExceedsPool
	case ReasonConcurrencyExceedsPool:
		return ErrConcurrencyExceedsPool
	case ReasonPinUnmatched:
		return ErrPinUnmatched
	case ReasonFitsOnlyAtCtxLT:
		return ErrFitsOnlyAtCtxLT
	case ReasonNeedsRing:
		return ErrNeedsRing
	case ReasonBelowSLO:
		return ErrBelowSLO
	case ReasonPinInfeasible:
		return ErrPinInfeasible
	case ReasonNoRankableCalibration:
		return ErrNoRankableCalibration
	default:
		return ErrNoFit
	}
}

// Model is the header facts the planner prices, in PAYLOAD bytes and weight ELEMENTS, plus
// GatherResidentBytes, the one fact in the selected arm's resident encoding.
type Model struct {
	Arch                      string `json:"arch,omitempty"`
	MaxContextTokens          int    `json:"max_context_tokens,omitempty"`
	PayloadWeightBytes        int64  `json:"payload_weight_bytes,omitempty"`   // every LOADED tensor
	StreamBytesPerToken       int64  `json:"stream_bytes_per_token,omitempty"` // swept at batch 1
	GatherPayloadBytes        int64  `json:"gather_payload_bytes,omitempty"`   // loaded but only row-gathered (untied input embedding)
	GatherResidentBytes       int64  `json:"gather_resident_bytes,omitempty"`  // resident bytes decode only row-gathers, selected arm
	ActiveParamsPerToken      int64  `json:"active_params_per_token,omitempty"`
	HeadParams                int64  `json:"head_params,omitempty"` // LM head elements, part of ActiveParamsPerToken
	RoutedExpertBytes         int64  `json:"routed_expert_bytes,omitempty"`
	ActiveRoutedBytesPerToken int64  `json:"active_routed_bytes_per_token,omitempty"`
	NumExperts                int    `json:"num_experts,omitempty"`
	ExpertsUsed               int    `json:"experts_used,omitempty"`
	ActiveSetUnknown          bool   `json:"active_set_unknown,omitempty"`
	AttnLayers                int    `json:"attn_layers,omitempty"` // KV-bearing attention layers, sliding-window included
	NumHeads                  int    `json:"num_heads,omitempty"`
	HeadDim                   int    `json:"head_dim,omitempty"`
	QuantFamily               string `json:"quant_family,omitempty"` // QuantFamily of the type carrying the most payload bytes
}

// Hardware is the pool and peaks a plan is priced against.
type Hardware struct {
	Topology        Topology
	PoolUsableBytes int64 // headroom-adjusted; <=0 unknown
	// HostPoolUsableBytes is the host RAM a discrete topology's host-scoped weight rows draw
	// on; <=0 unknown. Shared topologies already charge those rows to PoolUsableBytes.
	HostPoolUsableBytes int64
	Peaks               Peaks
}

// DefaultPromptTokens is a typical chat/agent turn prefix.
const DefaultPromptTokens = 2048

// Workload is the traffic a plan is priced for.
type Workload struct {
	PromptTokens  int  `json:"prompt_tokens"`
	Concurrency   int  `json:"concurrency"`
	BatchedDecode bool `json:"batched_decode"`
}

// DefaultWorkload is a single-operator local serve: a DefaultPromptTokens prompt (TTFT is
// taken at min(prompt, ctx)), one stream, and serial decode, because the native serve decodes
// concurrent sequences one at a time unless its batch coalescer applies, which is not assumed.
func DefaultWorkload() Workload {
	return Workload{PromptTokens: DefaultPromptTokens, Concurrency: 1, BatchedDecode: false}
}

func (w Workload) normalized() Workload {
	if w.PromptTokens <= 0 {
		w.PromptTokens = DefaultPromptTokens
	}
	if w.Concurrency <= 0 {
		w.Concurrency = 1
	}
	return w
}

// Pins are operator choices the planner never overrides; zero values are unpinned.
type Pins struct {
	KVPrecision   string `json:"kv_precision,omitempty"`
	ContextTokens int    `json:"context_tokens,omitempty"`
}

// Input is everything one Plan call prices.
type Input struct {
	Model       Model
	Weights     compute.MemoryPlan // the selected arm's weights plan, byte-identical to the sizer input
	Hardware    Hardware
	Workload    Workload
	Calibration Calibration
	Pins        Pins
	Candidates  []Candidate
}

// Prediction is one candidate's capacity and nominal throughput.
type Prediction struct {
	Name                string     `json:"name"`
	KVPrecision         string     `json:"kv_precision"`
	ContextTokens       int        `json:"context_tokens"`
	Current             bool       `json:"current,omitempty"`
	Fit                 string     `json:"fit"`                        // compute.FitVerdict.String()
	MaxConcurrency      int        `json:"max_concurrency,omitempty"`  // absent when unknown or none fit; Fit says which
	PerStreamBytes      int64      `json:"per_stream_bytes,omitempty"` // absent when the KV is not sizable
	DecodeBytesPerStep  int64      `json:"decode_bytes_per_step,omitempty"`
	DecodeTokSPerSeq    float64    `json:"decode_tok_s_per_seq,omitempty"`
	DecodeTokSAggregate float64    `json:"decode_tok_s_aggregate,omitempty"`
	DecodeTokSCeiling   float64    `json:"decode_tok_s_ceiling,omitempty"` // SOL, batch 1
	PrefillTokS         float64    `json:"prefill_tok_s,omitempty"`
	TTFTSeconds         float64    `json:"ttft_s,omitempty"`
	Bound               Bound      `json:"bound,omitempty"`
	Omitted             []Omission `json:"omitted,omitempty"`
}

// Refusal is the typed no-fit verdict; errors.Is matches ErrNoFit and the reason's sentinel.
type Refusal struct {
	Reason           RefusalReason `json:"reason"`
	KVPrecision      string        `json:"kv_precision,omitempty"`
	MaxContextTokens int           `json:"max_context_tokens,omitempty"`
	MaxConcurrency   int           `json:"max_concurrency,omitempty"`
	ShortfallBytes   int64         `json:"shortfall_bytes,omitempty"`
	HostPool         bool          `json:"host_pool,omitempty"` // a discrete topology's host pool, not the device pool, is short
	// T04 joint-search detail (additive; zero values stay absent in JSON).
	RequestedContext int     `json:"requested_context,omitempty"` // the context the ask wanted (fits_only_at_ctx_lt)
	Pin              string  `json:"pin,omitempty"`               // the unsatisfiable pin, e.g. "context_tokens=32768+kv_precision=q8"
	SLOBound         string  `json:"slo_bound,omitempty"`         // "tpot" | "ttft" (below_slo / pin_infeasible)
	SLOLimitMS       float64 `json:"slo_limit_ms,omitempty"`
	PredictedMS      float64 `json:"predicted_ms,omitempty"`
}

func (r *Refusal) Error() string   { return "fitplan: " + string(r.Reason) }
func (r *Refusal) Unwrap() []error { return []error{ErrNoFit, r.Reason.Sentinel()} }

// Evidence is what the numbers were priced against.
type Evidence struct {
	Calibration     Calibration `json:"calibration"`
	Peaks           Peaks       `json:"peaks"`
	Topology        Topology    `json:"topology"`
	PoolUsableBytes int64       `json:"pool_usable_bytes"` // max(avail,0); 0 = unknown
	// HostPoolUsableBytes is set for a discrete topology only; 0 = unknown.
	HostPoolUsableBytes int64   `json:"host_pool_usable_bytes,omitempty"`
	ResidentScale       float64 `json:"resident_scale,omitempty"` // swept resident weight bytes / swept payload bytes
}

// Report is the planner's answer.
type Report struct {
	Status           Status       `json:"status"`
	MaxContextTokens int          `json:"max_context_tokens,omitempty"` // one stream, reference tier
	Workload         Workload     `json:"workload"`
	Pins             Pins         `json:"pins"`
	Evidence         Evidence     `json:"evidence"`
	Model            *Model       `json:"model,omitempty"`
	Chosen           *Prediction  `json:"chosen,omitempty"`
	Refusal          *Refusal     `json:"refusal,omitempty"`
	Candidates       []Prediction `json:"candidates"`
	Omitted          []Omission   `json:"omitted,omitempty"`
}

// Unavailable is the report for a plan that could not be priced at all.
func Unavailable(reason Omission) Report {
	return Report{Status: StatusUnavailable, Candidates: []Prediction{}, Omitted: []Omission{reason}}
}

type planner struct {
	in          Input
	wl          Workload
	avail       int64
	weights     int64 // weight bytes the pool is charged
	hostWeights int64 // discrete topology: host-scoped weight bytes the host pool is charged
	hostShort   bool  // hostWeights exceed a known host pool, so nothing can boot
	hostUnknown bool  // hostWeights > 0 against an unknown host pool
	rho         float64
	rhoOK       bool
	gate        Omission // first reason throughput is not stated; "" when decode is stated
	gateBound   Bound
}

// Plan prices every candidate in input order and chooses one, or refuses with a typed reason.
// It never overrides a pin, and an unknown pool fails open (StatusUnknown, no Refusal).
func Plan(in Input) Report {
	p := newPlanner(in)

	model := in.Model
	rep := Report{
		Workload: p.wl,
		Pins:     in.Pins,
		Evidence: Evidence{
			Calibration:     in.Calibration.finite(in.Hardware.Peaks.BytesPerSec),
			Peaks:           in.Hardware.Peaks.finite(),
			Topology:        in.Hardware.Topology,
			PoolUsableBytes: max(p.avail, 0),
		},
		Model:      &model,
		Candidates: make([]Prediction, 0, len(in.Candidates)),
	}
	if !in.Hardware.Topology.sharesHostPool() {
		rep.Evidence.HostPoolUsableBytes = max(in.Hardware.HostPoolUsableBytes, 0)
	}
	if p.rhoOK {
		rep.Evidence.ResidentScale = finiteOrZero(p.rho)
	}
	for _, c := range in.Candidates {
		rep.Candidates = append(rep.Candidates, p.predict(c))
	}
	if len(in.Candidates) == 0 {
		rep.Status = StatusUnknown
		rep.Omitted = []Omission{OmitContextUnknown}
		return rep
	}
	if p.hostUnknown {
		rep.Omitted = append(rep.Omitted, OmitHostPoolCapacityUnknown)
	}
	ref := referenceIndex(in.Candidates)
	if refC := in.Candidates[ref]; p.avail > 0 {
		if kvSizable(refC.Context) {
			rep.MaxContextTokens = p.maxStreamContext(refC)
		} else {
			rep.Omitted = append(rep.Omitted, OmitKVGeometryUnknown)
		}
	}
	p.choose(&rep, ref)
	return rep
}

func referenceIndex(cs []Candidate) int {
	for i, c := range cs {
		if c.Current {
			return i
		}
	}
	return 0
}

// weightsInPool mirrors the live sizer's PoolSharedWithHost rule: a shared pool is charged
// host-scoped rows once, alongside the device-scoped ones, never as a second disjoint pool.
func weightsInPool(topo Topology, w compute.MemoryPlan) int64 {
	if topo.sharesHostPool() {
		return w.Total()
	}
	return w.DeviceTotal()
}

func scratchBytes(c Candidate) int64 {
	return compute.EstimateHALTransientMemoryPlan(c.Context.Scratch).Total()
}

func perStreamBytes(c Candidate, tokens int) int64 {
	return addSat(c.Context.KVBytes(tokens), c.Context.SessionState.Total())
}

func (p *planner) room(c Candidate) int64 {
	return subSat(subSat(p.avail, p.weights), scratchBytes(c))
}

func (p *planner) predict(c Candidate) Prediction {
	pr := Prediction{
		Name:           fmt.Sprintf("kv=%s ctx=%d", c.KVPrecision, c.ContextTokens),
		KVPrecision:    c.KVPrecision,
		ContextTokens:  c.ContextTokens,
		Current:        c.Current,
		PerStreamBytes: perStreamBytes(c, c.ContextTokens),
	}
	switch {
	case p.hostShort:
		pr.Fit = compute.FitTooBig.String()
	case p.avail <= 0:
		pr.Fit = compute.FitUnknown.String()
		pr.Omitted = append(pr.Omitted, OmitPoolCapacityUnknown)
	case c.Context.KVBytes(c.ContextTokens) <= 0:
		pr.Fit = compute.FitUnknown.String()
		pr.Omitted = append(pr.Omitted, OmitKVGeometryUnknown)
	default:
		room := p.room(c)
		if room >= pr.PerStreamBytes {
			pr.MaxConcurrency = clampInt(room / pr.PerStreamBytes)
		}
		pr.Fit = compute.FitTooBig.String()
		if pr.MaxConcurrency >= p.wl.Concurrency {
			pr.Fit = compute.FitOK.String()
		}
	}
	p.priceThroughput(&pr, c)
	return pr
}

func kvSizable(csc compute.ContextSizeConfig) bool {
	return csc.MaxContext > 0 && csc.KVBytes(csc.MaxContext) > 0
}

// maxStreamContext is the largest context one stream of c's tier fits, by the same monotone
// KVBytes each candidate is judged with. The live sizer is not reused: its flat per-token slope
// under-reports windowed KV, and it clamps to a floor.
func (p *planner) maxStreamContext(c Candidate) int {
	csc := c.Context
	if p.hostShort || p.avail <= 0 || !kvSizable(csc) {
		return 0
	}
	r := subSat(p.room(c), csc.SessionState.Total())
	if r <= 0 {
		return 0
	}
	return sort.Search(csc.MaxContext, func(i int) bool { return csc.KVBytes(i+1) > r })
}

func (p *planner) eligible(pr Prediction) bool {
	if pin := strings.TrimSpace(p.in.Pins.KVPrecision); pin != "" && !strings.EqualFold(pin, pr.KVPrecision) {
		return false
	}
	return p.in.Pins.ContextTokens <= 0 || pr.ContextTokens == p.in.Pins.ContextTokens
}

// firstEligible is ref when it honors the pins, else the first candidate that does; -1 if none.
func (p *planner) firstEligible(preds []Prediction, ref int) int {
	if p.eligible(preds[ref]) {
		return ref
	}
	for i, pr := range preds {
		if p.eligible(pr) {
			return i
		}
	}
	return -1
}

func (p *planner) choose(rep *Report, ref int) {
	preds := rep.Candidates
	first := p.firstEligible(preds, ref)
	switch {
	case first < 0:
		rep.Status = StatusRefused
		rep.Refusal = &Refusal{Reason: ReasonPinUnmatched}
		return
	case p.hostShort:
		rep.Status = StatusRefused
		rep.Refusal = p.refuse(preds, ref)
		return
	case p.avail <= 0 || preds[ref].Fit == compute.FitUnknown.String():
		rep.Status = StatusUnknown
		rep.Chosen = clonePrediction(preds[first])
		return
	}
	ok := compute.FitOK.String()
	best := -1
	if p.eligible(preds[ref]) && preds[ref].Fit == ok {
		best = ref
	} else {
		for i, pr := range preds {
			if !p.eligible(pr) || pr.Fit != ok {
				continue
			}
			if best < 0 || pr.ContextTokens > preds[best].ContextTokens {
				best = i
			}
		}
	}
	if best < 0 {
		rep.Status = StatusRefused
		rep.Refusal = p.refuse(preds, ref)
		return
	}
	rep.Status = StatusFits
	if p.hostUnknown {
		rep.Status = StatusUnknown
	}
	rep.Chosen = clonePrediction(preds[best])
}

func (p *planner) need(c Candidate) int64 {
	return addSat(addSat(p.weights, scratchBytes(c)), mulSat(int64(p.wl.Concurrency), perStreamBytes(c, c.ContextTokens)))
}

// refuse names why no eligible candidate fits. The shortfall is priced on ref when ref honors
// the pins, else on the eligible candidate closest to fitting.
func (p *planner) refuse(preds []Prediction, ref int) *Refusal {
	if p.hostShort {
		return &Refusal{Reason: ReasonWeightsExceedPool, HostPool: true, ShortfallBytes: p.hostWeights - p.in.Hardware.HostPoolUsableBytes}
	}
	if !p.eligible(preds[ref]) {
		ref = -1
		for i, pr := range preds {
			if p.eligible(pr) && (ref < 0 || p.need(p.in.Candidates[i]) < p.need(p.in.Candidates[ref])) {
				ref = i
			}
		}
	}
	r := &Refusal{ShortfallBytes: max(p.need(p.in.Candidates[ref])-p.avail, 0)}
	allZero, maxConc := true, 0
	for _, pr := range preds {
		if !p.eligible(pr) {
			continue
		}
		if pr.MaxConcurrency > 0 {
			allZero = false
		}
		maxConc = max(maxConc, pr.MaxConcurrency)
	}
	switch {
	case subSat(p.avail, p.weights) < 0:
		r.Reason = ReasonWeightsExceedPool
	case allZero:
		r.Reason = ReasonContextExceedsPool
		for i, pr := range preds {
			if !p.eligible(pr) {
				continue
			}
			if l := p.maxStreamContext(p.in.Candidates[i]); l > r.MaxContextTokens {
				r.MaxContextTokens, r.KVPrecision = l, pr.KVPrecision
			}
		}
	default:
		r.Reason = ReasonConcurrencyExceedsPool
		r.MaxConcurrency = maxConc
	}
	return r
}

func clonePrediction(pr Prediction) *Prediction {
	if pr.Omitted != nil {
		pr.Omitted = append([]Omission(nil), pr.Omitted...)
	}
	return &pr
}

func addSat(a, b int64) int64 {
	if b > 0 && a > math.MaxInt64-b {
		return math.MaxInt64
	}
	if b < 0 && a < math.MinInt64-b {
		return math.MinInt64
	}
	return a + b
}

func subSat(a, b int64) int64 {
	if b == math.MinInt64 {
		return addSat(addSat(a, math.MaxInt64), 1)
	}
	return addSat(a, -b)
}

func mulSat(a, b int64) int64 {
	if a <= 0 || b <= 0 {
		return 0
	}
	if a > math.MaxInt64/b {
		return math.MaxInt64
	}
	return a * b
}

func clampInt(v int64) int {
	if v > int64(math.MaxInt) {
		return math.MaxInt
	}
	if v < 0 {
		return 0
	}
	return int(v)
}

// StreamedResidentDType is the MemoryDemand.DType ggufload stamps on per-step streamed working-set rows.
const StreamedResidentDType = "streamed-resident"

// StreamBytes is the payload a decode step over `batch` sequences streams. A routed MoE step
// reads the union of the experts its sequences pick, modelled as independent uniform top-k:
// each expert is untouched with probability (1-K/E)^batch. Batch 1 is the exact header sweep.
func (m Model) StreamBytes(batch int) int64 {
	if batch < 1 {
		batch = 1
	}
	if m.NumExperts <= 0 || m.ExpertsUsed <= 0 || m.RoutedExpertBytes <= 0 || batch == 1 {
		return m.StreamBytesPerToken
	}
	nonRouted := max(m.StreamBytesPerToken-m.ActiveRoutedBytesPerToken, 0)
	pick := math.Min(float64(m.ExpertsUsed)/float64(m.NumExperts), 1)
	u := 1 - math.Pow(1-pick, float64(batch))
	routed := max(m.ActiveRoutedBytesPerToken, sat(float64(m.RoutedExpertBytes)*u))
	return addSat(nonRouted, routed)
}

// residentScale is the selected arm's resident encoding over the header payload, taken over the
// swept weights only: decode streams the resident form, not the file's, and an arm may keep the
// row-gathered input embedding far wider (f32) than the rest. Streamed working-set rows are not
// resident.
func residentScale(m Model, weights compute.MemoryPlan) (float64, bool) {
	var resident int64
	for _, d := range weights {
		if d.Bytes <= 0 || d.DType == StreamedResidentDType {
			continue
		}
		if d.Class != compute.MemoryWeights && d.Class != compute.MemoryOffload {
			continue
		}
		resident = addSat(resident, d.Bytes)
	}
	swept := subSat(resident, max(m.GatherResidentBytes, 0))
	payload := subSat(m.PayloadWeightBytes, max(m.GatherPayloadBytes, 0))
	if swept <= 0 || payload <= 0 {
		return 0, false
	}
	return float64(swept) / float64(payload), true
}

// attendedKVBytes is the KV a decode step reads at L cached positions: the f32 pre-RoPE K row
// is kept only for exact re-positioning on evict and is never attended.
func attendedKVBytes(csc compute.ContextSizeConfig, L int) int64 {
	kv := csc.KVBytes(L)
	if len(csc.CompressedKV) > 0 {
		return kv
	}
	perTok := csc.KVBytesPerToken()
	raw := int64(csc.KV.NumLayers) * int64(csc.KV.NumKVHeads) * int64(csc.KV.HeadDim) * 4 // f32 pre-RoPE K row, both tiers
	if perTok <= 0 || raw <= 0 || raw >= perTok {
		return kv
	}
	return int64(float64(kv) * float64(perTok-raw) / float64(perTok))
}

func (p *planner) throughputGate() (Omission, Bound) {
	in := p.in
	switch {
	case !in.Calibration.valid():
		return OmitCalibrationInvalid, ""
	case in.Model.ActiveSetUnknown:
		return OmitMoEActiveSetUnknown, ""
	case in.Model.StreamBytesPerToken <= 0 || !p.rhoOK:
		return OmitModelStreamUnknown, ""
	case hasStreamedRow(in.Weights):
		return OmitStreamBandwidthUnknown, BoundLink
	case in.Hardware.Topology == TopologyDiscrete && hasHostWeightRow(in.Weights):
		// On a shared pool spilled rows are pool bytes at pool bandwidth; only a discrete
		// device pays a host link for them.
		return OmitOffloadBandwidthUnknown, BoundLink
	case !positiveFinite(in.Hardware.Peaks.BytesPerSec) && !positiveFinite(in.Calibration.AchievedBytesPerSec):
		return OmitPeakBandwidthUnknown, ""
	}
	return "", ""
}

func hasStreamedRow(w compute.MemoryPlan) bool {
	for _, d := range w {
		if d.DType == StreamedResidentDType {
			return true
		}
	}
	return false
}

func hasHostWeightRow(w compute.MemoryPlan) bool {
	for _, d := range w {
		if d.Bytes > 0 && d.ScopeOrDefault() == compute.MemoryScopeHost &&
			(d.Class == compute.MemoryWeights || d.Class == compute.MemoryOffload) {
			return true
		}
	}
	return false
}

func (p *planner) priceThroughput(pr *Prediction, c Candidate) {
	if p.gate != "" {
		pr.Bound = p.gateBound
		pr.Omitted = append(pr.Omitted, p.gate)
		return
	}
	cal, peaks, m := p.in.Calibration, p.in.Hardware.Peaks, p.in.Model
	// Without the model's FLOPs a known FLOP peak cannot bind, so decode is the bandwidth bound.
	flopsKnown := m.ActiveParamsPerToken > 0
	peakBW, peakF := peaks.BytesPerSec, 0.0
	if flopsKnown && positiveFinite(peaks.FLOPsPerSec) {
		peakF = peaks.FLOPsPerSec
	}
	bw, f := p.effectiveRates()

	b := 1
	if p.wl.BatchedDecode {
		b = p.wl.Concurrency
		if pr.MaxConcurrency > 0 && b > pr.MaxConcurrency {
			b = pr.MaxConcurrency
		}
	}
	lkv := min(p.wl.PromptTokens, c.ContextTokens)
	kvRead := p.decodeKVRead(c, lkv)

	r := p.decodeStep(b, lkv, kvRead)
	step := finiteOrZero(withOverhead(r.RooflineTokPerSecCeiling(f, bw), cal.OverheadSeconds))
	pr.Bound = BoundMemory
	if f > 0 && r.FLOPBoundTokPerSecCeiling(f) < r.TokPerSecCeiling(bw) {
		pr.Bound = BoundCompute
	}
	if p.wl.BatchedDecode {
		pr.DecodeTokSAggregate = finiteOrZero(float64(b) * step)
		pr.DecodeTokSPerSeq = step
	} else {
		pr.DecodeTokSAggregate = step
		pr.DecodeTokSPerSeq = step / float64(p.wl.Concurrency)
	}
	pr.DecodeBytesPerStep = r.WeightBytesPerToken
	if positiveFinite(peakBW) {
		pr.DecodeTokSCeiling = finiteOrZero(p.decodeStep(1, lkv, kvRead).RooflineTokPerSecCeiling(peakF, peakBW))
	} else {
		// The achieved bandwidth stated decode; the SOL ceiling still needs the spec peak.
		pr.Omitted = append(pr.Omitted, OmitPeakBandwidthUnknown)
	}

	if !flopsKnown {
		pr.Omitted = append(pr.Omitted, OmitModelFLOPsUnknown)
	}
	if !positiveFinite(peaks.FLOPsPerSec) {
		pr.Omitted = append(pr.Omitted, OmitPeakComputeUnknown)
	}
	if peakF <= 0 || lkv <= 0 {
		return
	}
	// Prefill/TTFT assume an empty queue, no prefix-cache hit, warm weights, unchunked prefill,
	// and the first token sampled from the prefill's last logits: the LM head runs once.
	t := float64(lkv)
	active := nonneg(m.ActiveParamsPerToken)
	head := math.Min(nonneg(m.HeadParams), active)
	flopsPre := 2*(active-head)*t + 2*head + 2*p.attnWidth()*t*t // causal QK^T and AV
	bytesPre := p.rho*float64(m.StreamBytes(lkv)) + float64(c.Context.KVBytes(lkv))
	tPre := compute.StageCost{Name: "prefill", FLOPs: sat(flopsPre), WeightBytes: sat(bytesPre)}.RooflineSeconds(f, bw)
	if tPre > 0 && !math.IsInf(tPre, 0) {
		pr.TTFTSeconds = tPre
		pr.PrefillTokS = finiteOrZero(t / tPre)
	}
}

// attnWidth is KV-bearing attention layers x heads x head dim, the per-position attention width.
// Sliding-window layers are priced as full attention, so its terms are an upper bound there.
func (p *planner) attnWidth() float64 {
	m := p.in.Model
	return nonneg(int64(m.AttnLayers)) * nonneg(int64(m.NumHeads)) * nonneg(int64(m.HeadDim))
}

func (p *planner) decodeStep(b, lkv int, kvRead int64) compute.DecodeRoofline {
	m := p.in.Model
	bytes := p.rho*float64(m.StreamBytes(b)) + float64(b)*float64(kvRead)
	flops := float64(b) * (2*nonneg(m.ActiveParamsPerToken) + 4*p.attnWidth()*nonneg(int64(lkv)))
	return compute.DecodeRoofline{WeightBytesPerToken: sat(bytes), FLOPsPerToken: sat(flops)}
}

// sat clamps a float64 to [0, MaxInt64]; NaN reads as 0.
func sat(f float64) int64 {
	if !(f > 0) {
		return 0
	}
	if f >= math.MaxInt64 {
		return math.MaxInt64
	}
	return int64(f)
}

func nonneg(v int64) float64 { return float64(max(v, 0)) }

func positiveFinite(f float64) bool { return f > 0 && !math.IsInf(f, 0) }

func finiteOrZero(f float64) float64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0
	}
	return f
}

// Calibration is the fraction of each peak the engine is assumed to sustain. A positive
// AchievedBytesPerSec (a calibrated row's effective bandwidth) overrides BandwidthEfficiency x
// peak, the llm-analysis achieved-bandwidth shape, so a host with no spec peak still prices
// decode. OverheadSeconds is t_ovh, the fixed per-step cost a decode step adds.
type Calibration struct {
	Tier                Tier    `json:"tier"`
	BandwidthEfficiency float64 `json:"bandwidth_efficiency"`
	ComputeEfficiency   float64 `json:"compute_efficiency"`
	Source              string  `json:"source"`
	AchievedBytesPerSec float64 `json:"achieved_bytes_per_s,omitempty"`
	OverheadSeconds     float64 `json:"overhead_s,omitempty"`
	// ComputeTier and ComputeSource label ComputeEfficiency when its evidence differs from Tier.
	ComputeTier   Tier            `json:"compute_tier,omitempty"`
	ComputeSource string          `json:"compute_source,omitempty"`
	State         RowState        `json:"state,omitempty"`  // a calibrated row's state; drift = not fit to rank
	Row           *CalibrationRow `json:"row,omitempty"`    // the store row: key, fit, residuals, receipt provenance
	Lookup        *CalibrationKey `json:"lookup,omitempty"` // the key the plan resolved against, row or not
}

func (c Calibration) valid() bool {
	in := func(e float64) bool { return e > 0 && e <= 1 }
	bw := in(c.BandwidthEfficiency) || positiveFinite(c.AchievedBytesPerSec)
	ovh := c.OverheadSeconds >= 0 && !math.IsInf(c.OverheadSeconds, 0)
	return bw && in(c.ComputeEfficiency) && ovh
}

// finite is the calibration as evidence: non-finite numbers zeroed, and an achieved bandwidth
// restated as its efficiency against a known spec peak, so bandwidth_efficiency keeps its (0,1]
// meaning wherever a peak exists.
func (c Calibration) finite(peak float64) Calibration {
	if positiveFinite(c.AchievedBytesPerSec) && positiveFinite(peak) {
		c.BandwidthEfficiency = math.Min(c.AchievedBytesPerSec/peak, 1)
	}
	c.BandwidthEfficiency = finiteOrZero(c.BandwidthEfficiency)
	c.ComputeEfficiency = finiteOrZero(c.ComputeEfficiency)
	c.AchievedBytesPerSec = finiteOrZero(c.AchievedBytesPerSec)
	c.OverheadSeconds = finiteOrZero(c.OverheadSeconds)
	return c
}

// effectiveBandwidth is the achieved bandwidth when stated, capped at a known spec peak (no
// prediction beats the SOL ceiling), else efficiency x peak.
func (c Calibration) effectiveBandwidth(peak float64) float64 {
	if positiveFinite(c.AchievedBytesPerSec) {
		if positiveFinite(peak) {
			return math.Min(c.AchievedBytesPerSec, peak)
		}
		return c.AchievedBytesPerSec
	}
	return c.BandwidthEfficiency * peak
}

// withOverhead adds t_ovh to a roofline rate's step time: 1 / (1/rate + t_ovh).
func withOverhead(rate, overhead float64) float64 {
	if !positiveFinite(rate) || !(overhead > 0) {
		return rate
	}
	return 1 / (1/rate + overhead)
}

// generic, uncalibrated: bandwidth 0.55 (llmfit estimate_tps, MIT, AlexsJones/llmfit fit.rs), compute 0.5 (low end of the 0.5-0.7 FLOP-efficiency band analytic LLM models assume, cf. cli99/llm-analysis, Apache-2.0)
const (
	genericBandwidthEfficiency = 0.55
	genericComputeEfficiency   = 0.5
)

// GenericEmpirical is the uncalibrated tier used until device receipts supply efficiencies.
func GenericEmpirical() Calibration {
	return Calibration{
		Tier:                TierEmpirical,
		BandwidthEfficiency: genericBandwidthEfficiency,
		ComputeEfficiency:   genericComputeEfficiency,
		Source:              "generic, uncalibrated: bandwidth 0.55 (llmfit estimate_tps, MIT, AlexsJones/llmfit fit.rs), compute 0.5 (low end of the 0.5-0.7 FLOP-efficiency band analytic LLM models assume, cf. cli99/llm-analysis, Apache-2.0)",
	}
}

// Peaks are a device's peak memory bandwidth and FLOP rate; a non-positive value is unknown.
type Peaks struct {
	BytesPerSec float64 `json:"bytes_per_s,omitempty"`
	FLOPsPerSec float64 `json:"flops_per_s,omitempty"`
	Source      string  `json:"source,omitempty"`
}

func (p Peaks) finite() Peaks {
	p.BytesPerSec = finiteOrZero(p.BytesPerSec)
	p.FLOPsPerSec = finiteOrZero(p.FLOPsPerSec)
	return p
}

// DevicePeaks resolves an architecture token to the public target table's peaks. FP32 is
// used because the kernels accumulate f32; the WMMA peak is not assumed.
func DevicePeaks(arch string) (Peaks, bool) {
	a := strings.ToLower(strings.TrimSpace(arch))
	cfg, err := strix.GetTargetConfig(strix.RDNAArch(a))
	if err != nil || cfg.PeakBandwidthGBps <= 0 {
		return Peaks{}, false
	}
	return Peaks{
		BytesPerSec: cfg.PeakBandwidthGBps * 1e9,
		FLOPsPerSec: cfg.PeakFP32TFLOPs * 1e12,
		Source:      "strix.TargetConfig " + string(cfg.Arch) + " (" + cfg.ModelName + ")",
	}, true
}

// ContextLadder is the resolved context, any extra (e.g. pinned) contexts, common served windows
// (4K chat, 32K agent, 128K long context) and the model's full window, ascending and bounded by it.
func ContextLadder(resolved, maxContext int, extra ...int) []int {
	ns := append([]int{resolved}, extra...)
	if maxContext > 0 {
		ns = append(ns, 4096, 32768, 131072, maxContext)
	}
	out := make([]int, 0, len(ns))
	for _, n := range ns {
		if n > 0 && (maxContext <= 0 || n <= maxContext) {
			out = append(out, n)
		}
	}
	sort.Ints(out)
	k := 0
	for i, n := range out {
		if i == 0 || n != out[k-1] {
			out[k] = n
			k++
		}
	}
	return out[:k]
}

// KVTier is one KV storage tier's context geometry.
type KVTier struct {
	Precision string
	Context   compute.ContextSizeConfig
}

// Candidate is one (KV tier, context) configuration to price.
type Candidate struct {
	KVPrecision   string // compute.KVPrecision.String(): "f32" | "q8"
	Context       compute.ContextSizeConfig
	ContextTokens int
	Current       bool // the config a live boot takes
}

// BuildCandidates expands tiers over ContextLadder; tiers[0] at the resolved context is the
// current config. A tier whose KV charge equals an earlier tier's (a model that forces one
// storage tier) is skipped.
func BuildCandidates(tiers []KVTier, resolved int, extra ...int) []Candidate {
	var out []Candidate
	for ti, tier := range tiers {
		if sameKVCharge(tiers[:ti], tier) {
			continue
		}
		for _, l := range ContextLadder(resolved, tier.Context.MaxContext, extra...) {
			out = append(out, Candidate{
				KVPrecision:   tier.Precision,
				Context:       tier.Context,
				ContextTokens: l,
				Current:       ti == 0 && l == resolved,
			})
		}
	}
	return out
}

func sameKVCharge(earlier []KVTier, t KVTier) bool {
	for _, e := range earlier {
		if e.Context.KVBytesPerToken() == t.Context.KVBytesPerToken() &&
			e.Context.FixedKVBytesPerStream() == t.Context.FixedKVBytesPerStream() &&
			e.Context.SessionState.Total() == t.Context.SessionState.Total() {
			return true
		}
	}
	return false
}
