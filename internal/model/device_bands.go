package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// device_bands.go — the VRAM-aware planner that splits one model's transformer layers
// into contiguous bands, one per CUDA device in a single process. It sits on top of the
// pipeline contract (pipeline.go): every plan converts to a validated PartitionPlan, so
// the band boundaries it picks obey the same rules ForwardBand and RunPipeline enforce.
//
// Ownership rule: a layer's state lives on the device that runs the layer. Each band is
// charged its own weights, the KV of its full-attention layers at the planned context,
// and the fixed Gated-DeltaNet conv+recurrent state of its linear-attention layers. No
// state is shared across a boundary; only the residual hidden state crosses it. For a
// Qwen3.5-family hybrid the planner also cuts only right after a full-attention layer, so
// every band holds whole [linear..., full] periods and its KV store has full-attention rows.
//
// Objective: minimize the largest per-device utilization (band bytes / usable bytes) over
// contiguous bands in the caller's device order. Device order is pipeline order: the first
// device embeds and the last runs the final norm and LM head. A plan that cannot fit fails
// closed with ErrDeviceBandsDoNotFit before any weight is uploaded.

// Sentinel errors for callers and tests (errors.Is).
var (
	ErrDeviceBandsInvalid   = errors.New("model: invalid device band request")
	ErrDeviceBandsDoNotFit  = errors.New("model: layers do not fit the device budgets")
	ErrDeviceBandsTooFewCut = errors.New("model: not enough legal cut points for the device count")
)

// DefaultDeviceBandHeadroom is the fraction of free device memory a band may use. It is
// the same 15% headroom the single-device context auto-sizer keeps (cmd/fak serve fit).
const DefaultDeviceBandHeadroom = 0.85

// DeviceBudget is one CUDA device available to a band: its ordinal in the process and the
// free bytes measured before load (cudaMemGetInfo or nvidia-smi memory.total-memory.used).
type DeviceBudget struct {
	Device    int    `json:"device"`
	Name      string `json:"name,omitempty"`
	FreeBytes int64  `json:"free_bytes"`
}

// BandLayerCost is the resident memory one transformer layer pins on its band's device.
type BandLayerCost struct {
	WeightBytes   int64 `json:"weight_bytes"`
	KVBytes       int64 `json:"kv_bytes"`    // full-attention KV at the planned context
	StateBytes    int64 `json:"state_bytes"` // fixed recurrent state (GDN conv + recurrent)
	FullAttention bool  `json:"full_attention"`
}

// DeviceBandCosts is the per-layer cost table plus the layer-agnostic charges.
type DeviceBandCosts struct {
	Layers []BandLayerCost `json:"layers"`
	// EmbedBytes is charged to the first band (token embedding lookup).
	EmbedBytes int64 `json:"embed_bytes"`
	// HeadBytes is charged to the last band (final norm + LM head; a tied head repeats the
	// embedding bytes because the last device must hold its own copy).
	HeadBytes int64 `json:"head_bytes"`
	// FixedBytesPerDevice is charged to every band (cuBLAS workspace, HAL transients,
	// handoff staging).
	FixedBytesPerDevice int64 `json:"fixed_bytes_per_device"`
	ContextTokens       int   `json:"context_tokens"`
}

// DeviceBandOptions tunes the planner. The zero value is the default policy.
type DeviceBandOptions struct {
	// Headroom is the usable fraction of DeviceBudget.FreeBytes; 0 selects
	// DefaultDeviceBandHeadroom. It must be in (0, 1].
	Headroom float64
}

// DeviceBand is one planned band: the layers it owns, its device, and its byte charges.
type DeviceBand struct {
	Stage               StageSpec `json:"stage"`
	Device              int       `json:"device"`
	WeightBytes         int64     `json:"weight_bytes"`
	KVBytes             int64     `json:"kv_bytes"`
	StateBytes          int64     `json:"state_bytes"`
	FixedBytes          int64     `json:"fixed_bytes"` // embed/head/per-device fixed
	TotalBytes          int64     `json:"total_bytes"`
	UsableBytes         int64     `json:"usable_bytes"`
	FullAttentionLayers int       `json:"full_attention_layers"`
	LinearLayers        int       `json:"linear_layers"`
}

// Utilization is TotalBytes / UsableBytes.
func (b DeviceBand) Utilization() float64 {
	if b.UsableBytes <= 0 {
		return math.Inf(1)
	}
	return float64(b.TotalBytes) / float64(b.UsableBytes)
}

// DeviceBandPlan is an ordered set of bands that tiles [0, NumLayers).
type DeviceBandPlan struct {
	NumLayers     int          `json:"num_layers"`
	ContextTokens int          `json:"context_tokens"`
	Bands         []DeviceBand `json:"bands"`
}

// PartitionPlan returns the pipeline partition for this plan, re-validated against cfg.
func (p DeviceBandPlan) PartitionPlan(cfg Config) (PartitionPlan, error) {
	specs := make([]StageSpec, len(p.Bands))
	for i, b := range p.Bands {
		specs[i] = b.Stage
	}
	plan := PartitionPlan{NumLayers: p.NumLayers, Stages: specs}
	if err := plan.Validate(cfg); err != nil {
		return PartitionPlan{}, err
	}
	return plan, nil
}

// MaxUtilization is the largest band utilization in the plan.
func (p DeviceBandPlan) MaxUtilization() float64 {
	worst := 0.0
	for _, b := range p.Bands {
		worst = math.Max(worst, b.Utilization())
	}
	return worst
}

// HybridDeviceBandCosts builds the cost table for a Config from per-layer weight bytes
// (for a GGUF, ggufload.LayerWeightCensus). KV is charged only to layers that append to
// the HAL KV store (every layer for a dense model, the full-attention layers for a
// Qwen3.5-family hybrid) at the store's precision; GDN conv+recurrent state is charged
// to each linear-attention layer with the shapes initQwen35HALState allocates.
func HybridDeviceBandCosts(cfg Config, layerWeightBytes []int64, embedBytes, headBytes, fixedPerDevice int64, contextTokens int, prec compute.KVPrecision) (DeviceBandCosts, error) {
	if cfg.NumLayers <= 0 || len(layerWeightBytes) != cfg.NumLayers {
		return DeviceBandCosts{}, fmt.Errorf("%w: %d layer weight entries for %d layers", ErrDeviceBandsInvalid, len(layerWeightBytes), cfg.NumLayers)
	}
	if contextTokens < 0 || embedBytes < 0 || headBytes < 0 || fixedPerDevice < 0 {
		return DeviceBandCosts{}, fmt.Errorf("%w: negative context or byte charge", ErrDeviceBandsInvalid)
	}
	kvPerLayer := compute.EstimateKVStoreBytes(compute.KVConfig{NumLayers: 1, NumKVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim, Precision: prec}, contextTokens)
	var gdnState int64
	if cfg.IsQwen35Hybrid() {
		_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
		conv, ok1 := checkedMul(int64(max(0, cfg.LinearConvKernelDim-1)), int64(convDim), 4)
		rec, ok2 := checkedMul(int64(nV), int64(kHd), int64(vHd), 4)
		sum, ok3 := checkedAdd(conv, rec)
		if !ok1 || !ok2 || !ok3 {
			return DeviceBandCosts{}, fmt.Errorf("%w: recurrent state geometry overflows int64", ErrDeviceBandsInvalid)
		}
		gdnState = sum
	}
	layers := make([]BandLayerCost, cfg.NumLayers)
	for l := range layers {
		if layerWeightBytes[l] < 0 {
			return DeviceBandCosts{}, fmt.Errorf("%w: layer %d weight bytes %d", ErrDeviceBandsInvalid, l, layerWeightBytes[l])
		}
		c := BandLayerCost{WeightBytes: layerWeightBytes[l]}
		if cfg.isLinearAttnLayer(l) {
			c.StateBytes = gdnState
		} else {
			c.FullAttention = true
			c.KVBytes = kvPerLayer
		}
		layers[l] = c
	}
	return DeviceBandCosts{Layers: layers, EmbedBytes: embedBytes, HeadBytes: headBytes, FixedBytesPerDevice: fixedPerDevice, ContextTokens: contextTokens}, nil
}

// legalBandCuts lists the interior boundaries a band may start at. A hybrid cuts only
// right after a full-attention layer; a GLM-MoE-DSA model only where ForwardBand admits a
// band start; anything else may cut at every layer boundary.
func legalBandCuts(cfg Config, costs DeviceBandCosts) []int {
	n := len(costs.Layers)
	var cuts []int
	for k := 1; k < n; k++ {
		if cfg.IsQwen35Hybrid() && !costs.Layers[k-1].FullAttention {
			continue
		}
		if cfg.isGLMMoeDsa() && !glmDsaBandStartOK(cfg, k) {
			continue
		}
		cuts = append(cuts, k)
	}
	return cuts
}

// PlanDeviceBands assigns one contiguous band to each device, in order, minimizing the
// largest utilization. Every device gets at least one layer; pass only the devices the
// plan should use. It fails closed on an invalid request, on too few legal cut points, and
// with ErrDeviceBandsDoNotFit when even the best split overfills a device.
func PlanDeviceBands(cfg Config, costs DeviceBandCosts, devices []DeviceBudget, opts DeviceBandOptions) (DeviceBandPlan, error) {
	n := len(costs.Layers)
	if n == 0 || n != cfg.NumLayers {
		return DeviceBandPlan{}, fmt.Errorf("%w: cost table has %d layers, config %d", ErrDeviceBandsInvalid, n, cfg.NumLayers)
	}
	if len(devices) == 0 {
		return DeviceBandPlan{}, fmt.Errorf("%w: no devices", ErrDeviceBandsInvalid)
	}
	headroom := opts.Headroom
	if headroom == 0 {
		headroom = DefaultDeviceBandHeadroom
	}
	if !(headroom > 0 && headroom <= 1) {
		return DeviceBandPlan{}, fmt.Errorf("%w: headroom %v outside (0,1]", ErrDeviceBandsInvalid, headroom)
	}
	seen := make(map[int]bool, len(devices))
	usable := make([]int64, len(devices))
	for i, d := range devices {
		if d.Device < 0 || seen[d.Device] || d.FreeBytes <= 0 {
			return DeviceBandPlan{}, fmt.Errorf("%w: device entry %d (ordinal %d, free %d)", ErrDeviceBandsInvalid, i, d.Device, d.FreeBytes)
		}
		seen[d.Device] = true
		usable[i] = int64(float64(d.FreeBytes) * headroom)
	}

	// prefix[k] = layer bytes of layers [0,k).
	prefix := make([]int64, n+1)
	for l, c := range costs.Layers {
		if c.WeightBytes < 0 || c.KVBytes < 0 || c.StateBytes < 0 {
			return DeviceBandPlan{}, fmt.Errorf("%w: layer %d has a negative charge", ErrDeviceBandsInvalid, l)
		}
		sum, ok := checkedAdd(prefix[l], c.WeightBytes, c.KVBytes, c.StateBytes)
		if !ok {
			return DeviceBandPlan{}, fmt.Errorf("%w: layer bytes overflow int64", ErrDeviceBandsInvalid)
		}
		prefix[l+1] = sum
	}

	// bounds are the band-start candidates: 0, every legal interior cut, then n.
	bounds := append([]int{0}, legalBandCuts(cfg, costs)...)
	bounds = append(bounds, n)
	D := len(devices)
	if len(bounds)-1 < D {
		return DeviceBandPlan{}, fmt.Errorf("%w: %d devices but only %d legal bands", ErrDeviceBandsTooFewCut, D, len(bounds)-1)
	}
	bandBytes := func(dev, lo, hi int) int64 {
		b := prefix[hi] - prefix[lo] + costs.FixedBytesPerDevice
		if dev == 0 {
			b += costs.EmbedBytes
		}
		if dev == D-1 {
			b += costs.HeadBytes
		}
		return b
	}
	util := func(dev, lo, hi int) float64 { return float64(bandBytes(dev, lo, hi)) / float64(usable[dev]) }

	// best[d][j]: min over splits of the max utilization when devices [0,d] cover layers
	// [0, bounds[j]). from[d][j] is the bound index the band on device d starts at.
	B := len(bounds)
	inf := math.Inf(1)
	best := make([][]float64, D)
	from := make([][]int, D)
	for d := range best {
		best[d] = make([]float64, B)
		from[d] = make([]int, B)
		for j := range best[d] {
			best[d][j] = inf
			from[d][j] = -1
		}
	}
	for j := 1; j < B; j++ {
		best[0][j] = util(0, 0, bounds[j])
		from[0][j] = 0
	}
	for d := 1; d < D; d++ {
		for j := d + 1; j < B; j++ {
			for i := d; i < j; i++ {
				if best[d-1][i] == inf {
					continue
				}
				v := math.Max(best[d-1][i], util(d, bounds[i], bounds[j]))
				if v < best[d][j] {
					best[d][j] = v
					from[d][j] = i
				}
			}
		}
	}
	if best[D-1][B-1] == inf {
		return DeviceBandPlan{}, fmt.Errorf("%w: no split covers %d layers", ErrDeviceBandsTooFewCut, n)
	}

	starts := make([]int, D+1)
	starts[D] = B - 1
	for d, j := D-1, B-1; d >= 0; d-- {
		i := from[d][j]
		starts[d] = i
		j = i
	}
	plan := DeviceBandPlan{NumLayers: n, ContextTokens: costs.ContextTokens, Bands: make([]DeviceBand, D)}
	for d := 0; d < D; d++ {
		lo, hi := bounds[starts[d]], bounds[starts[d+1]]
		b := DeviceBand{
			Stage:       StageSpec{Lo: lo, Hi: hi, First: d == 0, Last: d == D-1},
			Device:      devices[d].Device,
			FixedBytes:  bandBytes(d, lo, hi) - (prefix[hi] - prefix[lo]),
			TotalBytes:  bandBytes(d, lo, hi),
			UsableBytes: usable[d],
		}
		for l := lo; l < hi; l++ {
			c := costs.Layers[l]
			b.WeightBytes += c.WeightBytes
			b.KVBytes += c.KVBytes
			b.StateBytes += c.StateBytes
			if c.FullAttention {
				b.FullAttentionLayers++
			} else {
				b.LinearLayers++
			}
		}
		plan.Bands[d] = b
	}
	if _, err := plan.PartitionPlan(cfg); err != nil {
		return DeviceBandPlan{}, err
	}
	for _, b := range plan.Bands {
		if b.TotalBytes > b.UsableBytes {
			return plan, fmt.Errorf("%w: device %d band [%d,%d) needs %d B, usable %d B (short %d B)",
				ErrDeviceBandsDoNotFit, b.Device, b.Stage.Lo, b.Stage.Hi, b.TotalBytes, b.UsableBytes, b.TotalBytes-b.UsableBytes)
		}
	}
	return plan, nil
}
