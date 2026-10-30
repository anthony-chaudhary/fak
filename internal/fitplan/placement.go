package fitplan

import (
	"math"
	"sort"
)

// Every model fits through device -> host RAM -> NVMe offload, so an oversize model is a cost
// (decode tok/s, context, batch), not a refusal. When the weights exceed the pool the planner
// answers with a TieredPlacement built dense-core-first: the dense core (every token streams
// it) takes the fastest pool, the routed experts fill what is left of it as the expert ring,
// then host RAM, then an NVMe mmap of the file, which always has room. A typed refusal stays
// only for an operator-pinned placement over its budget (Pins.Placement).

// StatusTiered is a plan that fits only by spilling weights below the fastest pool.
const StatusTiered Status = "tiered"

// PlacementDeviceResident pins every weight byte resident in the priced pool: no offload.
// It is the one placement whose over-budget verdict is still a typed refusal.
const PlacementDeviceResident = "device_resident"

// PlacementStrategyDenseCoreFirst names the fill order TieredPlacement rows follow.
const PlacementStrategyDenseCoreFirst = "dense_core_first"

// PlacementPool is one tier a weight segment can live in, fastest first.
type PlacementPool string

const (
	PlacementPoolDevice  PlacementPool = "device"
	PlacementPoolHostRAM PlacementPool = "host_ram"
	PlacementPoolNVMe    PlacementPool = "nvme_mmap"
)

// PlacementSegment is a weight segment the placement moves as a unit. routed_experts on the
// device pool is the expert ring.
type PlacementSegment string

const (
	SegmentDenseCore     PlacementSegment = "dense_core"
	SegmentRoutedExperts PlacementSegment = "routed_experts"
)

// Generic effective bandwidths for the tiers below the device, used until Hardware states them:
// dual-channel DDR5-class host RAM at ~0.55 of ~110 GB/s, and a PCIe 4.0 x4 NVMe sequential read.
const (
	genericHostBytesPerSec = 60e9
	genericNVMeBytesPerSec = 3.5e9
)

// PlacementRow is one segment's bytes resident in one pool, and the bandwidth it is read at.
type PlacementRow struct {
	Segment          PlacementSegment `json:"segment"`
	Pool             PlacementPool    `json:"pool"`
	Bytes            int64            `json:"bytes"`
	BytesPerSec      float64          `json:"bytes_per_s,omitempty"`
	GenericBandwidth bool             `json:"generic_bandwidth,omitempty"` // priced at the generic default, not a stated rate
}

// TieredPlacement is one offload candidate: where each segment lives, and what that costs.
// MaxContextTokens and MaxBatch let the expert ring yield its device bytes to KV; DecodeTokS
// is one stream at ContextTokens with the ring as placed.
type TieredPlacement struct {
	Strategy         string         `json:"strategy"`
	Rows             []PlacementRow `json:"rows"`
	KVPool           PlacementPool  `json:"kv_pool,omitempty"`
	KVPrecision      string         `json:"kv_precision,omitempty"`
	ContextTokens    int            `json:"context_tokens,omitempty"`
	MaxContextTokens int            `json:"max_context_tokens,omitempty"`
	MaxBatch         int            `json:"max_batch,omitempty"`
	DecodeTokS       float64        `json:"decode_tok_s,omitempty"`
	Bound            Bound          `json:"bound,omitempty"`
	Tier             Tier           `json:"tier"`
	Omitted          []Omission     `json:"omitted,omitempty"`
}

// pinnedResident reports whether the operator pinned the all-resident placement.
func pinnedResident(placement string) bool { return placement == PlacementDeviceResident }

// weightsOverflow reports whether the all-resident placement exceeds a known pool.
func (p *planner) weightsOverflow() bool {
	return p.hostShort || (p.avail > 0 && subSat(p.avail, p.weights) < 0)
}

type placementTier struct {
	pool    PlacementPool
	cap     int64
	bw      float64
	generic bool
}

// placementTiers are the pools a segment may fill, fastest first; NVMe is unbounded (the file).
func (p *planner) placementTiers() []placementTier {
	hw := p.in.Hardware
	devBW, _ := p.effectiveRates()
	nvmeBW, nvmeGeneric := hw.NVMeBytesPerSec, false
	if !positiveFinite(nvmeBW) {
		nvmeBW, nvmeGeneric = genericNVMeBytesPerSec, true
	}
	nvme := placementTier{pool: PlacementPoolNVMe, cap: math.MaxInt64, bw: nvmeBW, generic: nvmeGeneric}
	switch hw.Topology {
	case TopologyHostOnly:
		return []placementTier{{pool: PlacementPoolHostRAM, cap: p.avail, bw: devBW}, nvme}
	case TopologyUnified:
		return []placementTier{{pool: PlacementPoolDevice, cap: p.avail, bw: devBW}, nvme}
	case TopologyDiscrete:
		// device pool and host RAM are separate pools; both can absorb segments,
		// so use the shared three-tier list below
	}
	hostBW, hostGeneric := hw.HostBytesPerSec*p.in.Calibration.BandwidthEfficiency, false
	if !positiveFinite(hostBW) {
		hostBW, hostGeneric = genericHostBytesPerSec, true
	}
	return []placementTier{
		{pool: PlacementPoolDevice, cap: p.avail, bw: devBW},
		{pool: PlacementPoolHostRAM, cap: max(hw.HostPoolUsableBytes, 0), bw: hostBW, generic: hostGeneric},
		nvme,
	}
}

// segmentBytes splits the resident weights into the dense core and the routed experts.
func (p *planner) segmentBytes() (dense, routed int64) {
	m := p.in.Model
	if m.PayloadWeightBytes <= 0 {
		return addSat(p.weights, p.hostWeights), 0
	}
	routed = min(max(m.RoutedExpertBytes, 0), m.PayloadWeightBytes)
	dense = m.PayloadWeightBytes - routed
	if p.rhoOK {
		dense, routed = sat(float64(dense)*p.rho), sat(float64(routed)*p.rho)
	}
	return dense, routed
}

// placeTiered builds the dense-core-first placement, pricing KV with kv when it is sizable.
// It returns nil only when the pool capacity is unknown or the model carries no weight bytes.
func (p *planner) placeTiered(kv *Candidate) *TieredPlacement {
	dense, routed := p.segmentBytes()
	if p.avail <= 0 || addSat(dense, routed) <= 0 {
		return nil
	}
	tiers := p.placementTiers()
	tp := &TieredPlacement{Strategy: PlacementStrategyDenseCoreFirst, Rows: []PlacementRow{}, Tier: p.in.Calibration.Tier}
	if p.in.Hardware.Topology == TopologyDiscrete && p.in.Hardware.HostPoolUsableBytes <= 0 {
		tp.Omitted = append(tp.Omitted, OmitHostPoolCapacityUnknown)
	}

	// KV and scratch live in the fastest RAM pool that holds them beside its share of the core.
	var scratch int64
	ctx, kvIdx := 0, 0
	sizable := kv != nil && kvSizable(kv.Context) && kv.ContextTokens > 0
	if sizable {
		scratch = scratchBytes(*kv)
		ctx = kv.ContextTokens
		reserve := addSat(scratch, perStreamBytes(*kv, ctx))
		if subSat(tiers[0].cap, dense) < reserve {
			for i := 1; i < len(tiers)-1; i++ {
				if tiers[i].cap >= reserve {
					kvIdx = i
					break
				}
			}
		}
		if limit := p.contextIn(*kv, subSat(tiers[kvIdx].cap, scratch)); limit < ctx {
			ctx = limit
		}
	}
	left := make([]int64, len(tiers))
	for i, t := range tiers {
		left[i] = t.cap
	}
	if sizable && ctx > 0 {
		left[kvIdx] = max(subSat(left[kvIdx], addSat(scratch, perStreamBytes(*kv, ctx))), 0)
	}
	denseOn := fill(left, dense)
	routedOn := fill(left, routed)
	for _, seg := range []struct {
		name PlacementSegment
		on   []int64
	}{{SegmentDenseCore, denseOn}, {SegmentRoutedExperts, routedOn}} {
		for i, b := range seg.on {
			if b > 0 {
				tp.Rows = append(tp.Rows, PlacementRow{Segment: seg.name, Pool: tiers[i].pool, Bytes: b, BytesPerSec: finiteOrZero(tiers[i].bw), GenericBandwidth: tiers[i].generic})
				if tiers[i].generic {
					tp.Tier = TierEmpirical
				}
				if i > 0 {
					tp.Bound = BoundLink
				}
			}
		}
	}
	if tp.Bound == "" {
		tp.Bound = BoundMemory
	}

	if !sizable {
		tp.Omitted = append(tp.Omitted, OmitKVGeometryUnknown)
	} else {
		tp.KVPool, tp.KVPrecision, tp.ContextTokens = tiers[kvIdx].pool, kv.KVPrecision, ctx
		room := subSat(subSat(tiers[kvIdx].cap, denseOn[kvIdx]), scratch)
		tp.MaxContextTokens = p.contextIn(*kv, room)
		if ctx > 0 {
			if per := perStreamBytes(*kv, ctx); per > 0 && room >= per {
				tp.MaxBatch = clampInt(room / per)
			}
		}
	}
	p.priceTiered(tp, tiers, kv, ctx, dense, routed, denseOn, routedOn, kvIdx)
	return tp
}

// contextIn is the largest context whose one-stream KV and session state fit in room.
func (p *planner) contextIn(c Candidate, room int64) int {
	csc := c.Context
	r := subSat(room, csc.SessionState.Total())
	if r <= 0 || csc.MaxContext <= 0 {
		return 0
	}
	return sort.Search(csc.MaxContext, func(i int) bool { return csc.KVBytes(i+1) > r })
}

// fill places want bytes into left, fastest first, and returns the bytes landed per pool.
func fill(left []int64, want int64) []int64 {
	on := make([]int64, len(left))
	for i := range left {
		if want <= 0 {
			break
		}
		take := min(want, max(left[i], 0))
		on[i], left[i], want = take, left[i]-take, want-take
	}
	return on
}

// priceTiered states one stream's decode rate: each pool's share of the per-token bytes over
// that pool's bandwidth, plus the attended KV, plus the calibrated per-step overhead. Routed
// bytes are split by residency, the uniform-routing expectation.
func (p *planner) priceTiered(tp *TieredPlacement, tiers []placementTier, kv *Candidate, ctx int, dense, routed int64, denseOn, routedOn []int64, kvIdx int) {
	m, cal := p.in.Model, p.in.Calibration
	scale := 1.0
	if p.rhoOK {
		scale = p.rho
	}
	denseStream := float64(dense)
	activeRouted := 0.0
	if m.StreamBytesPerToken > 0 {
		denseStream = scale * float64(max(m.StreamBytesPerToken-max(m.ActiveRoutedBytesPerToken, 0), 0))
	}
	switch {
	case m.ActiveRoutedBytesPerToken > 0:
		activeRouted = scale * float64(m.ActiveRoutedBytesPerToken)
	case routed > 0 && m.NumExperts > 0 && m.ExpertsUsed > 0:
		activeRouted = float64(routed) * math.Min(float64(m.ExpertsUsed)/float64(m.NumExperts), 1)
	}
	switch {
	case !cal.valid():
		tp.Omitted = append(tp.Omitted, OmitCalibrationInvalid)
		return
	case m.ActiveSetUnknown || (routed > 0 && activeRouted <= 0):
		tp.Omitted = append(tp.Omitted, OmitMoEActiveSetUnknown)
		return
	case !positiveFinite(tiers[0].bw):
		tp.Omitted = append(tp.Omitted, OmitPeakBandwidthUnknown)
		return
	}
	seconds := 0.0
	for i, t := range tiers {
		bytes := 0.0
		if dense > 0 {
			bytes += denseStream * float64(denseOn[i]) / float64(dense)
		}
		if routed > 0 {
			bytes += activeRouted * float64(routedOn[i]) / float64(routed)
		}
		if bytes > 0 {
			seconds += bytes / t.bw
		}
	}
	if kv != nil && ctx > 0 {
		seconds += float64(p.decodeKVRead(*kv, min(p.wl.PromptTokens, ctx))) / tiers[kvIdx].bw
	}
	if seconds <= 0 {
		return
	}
	if cal.OverheadSeconds > 0 {
		seconds += cal.OverheadSeconds
	}
	tp.DecodeTokS = finiteOrZero(1 / seconds)
}
