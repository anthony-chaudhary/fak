package macfit

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Input describes a modeled unified-memory budget. All byte quantities are
// binary bytes; ContextTokens, SharedPrefixTokens, and TailCapTokens are tokens.
type Input struct {
	MemoryBytes        uint64 `json:"memory_bytes"`
	ReserveBytes       uint64 `json:"reserve_bytes"`
	WeightBytes        uint64 `json:"weight_bytes"`
	ContextTokens      uint64 `json:"context_tokens"`
	Layers             uint64 `json:"layers"`
	KVHeads            uint64 `json:"kv_heads"`
	HeadDim            uint64 `json:"head_dim"`
	KVBytesPerElement  uint64 `json:"kv_bytes_per_element"`
	SharedPrefixTokens uint64 `json:"shared_prefix_tokens"`
	TailCapTokens      uint64 `json:"tail_cap_tokens"`
}

// Result is a modeled capacity comparison, not a hardware measurement.
type Result struct {
	Schema                 string `json:"schema"`
	Provenance             string `json:"provenance"`
	MemoryBytes            uint64 `json:"memory_bytes"`
	ReserveBytes           uint64 `json:"reserve_bytes"`
	WeightBytes            uint64 `json:"weight_bytes"`
	KVPoolBytes            uint64 `json:"kv_pool_bytes"`
	KVBytesPerToken        uint64 `json:"kv_bytes_per_token"`
	OffKVBytesPerAgent     uint64 `json:"off_kv_bytes_per_agent"`
	OnSharedKVBytes        uint64 `json:"on_shared_kv_bytes"`
	OnTailKVBytesPerAgent  uint64 `json:"on_tail_kv_bytes_per_agent"`
	OffAgentsThatFit       uint64 `json:"off_agents_that_fit"`
	OnAgentsThatFit        uint64 `json:"on_agents_that_fit"`
	ExtraAgents            uint64 `json:"extra_agents"`
	CrossoverContextTokens uint64 `json:"crossover_context_tokens,omitempty"`
	CrossoverFound         bool   `json:"crossover_found"`
}

func mul(values ...uint64) (uint64, error) {
	n := uint64(1)
	for _, v := range values {
		if v != 0 && n > math.MaxUint64/v {
			return 0, errors.New("capacity arithmetic overflows uint64")
		}
		n *= v
	}
	return n, nil
}

// kvQuantized reports whether a KV precision tier stores the attended rows at a
// reduced width (q8_0/q4_0) rather than full FP16/FP32. It gates the memory-pressure
// context ceiling: a quantized tier earns a proportionally larger context budget.
func kvQuantized(prec model.KVPrecision) bool {
	switch prec {
	case model.KVPrecisionQ8_0, model.KVPrecisionQ4_0:
		return true
	default:
		return false
	}
}

// kvBytesPerToken returns the resident KV bytes for one token of one layer's K and V
// vectors, each of length KVHeads*HeadDim, packed at prec. The model.KVVectorBytes
// helper already accounts for per-group scale/min metadata, so the quantized result is
// honest rather than a naive bit-width division. FP16 reproduces the historical
// 2*KVHeads*HeadDim*2 bytes per token per layer exactly.
func kvBytesPerToken(tier ModelTier, prec model.KVPrecision) (uint64, error) {
	dim, err := mul(tier.KVHeads, tier.HeadDim)
	if err != nil {
		return 0, err
	}
	if dim == 0 {
		return 0, nil
	}
	perSide := uint64(model.KVVectorBytes(int(dim), prec))
	if perSide == 0 {
		return 0, nil
	}
	// K and V are the same shape, so the per-token per-layer cost is two sides.
	return mul(2, tier.Layers, perSide)
}

func fit(pool, fixed, perAgent uint64) uint64 {
	if fixed > pool {
		return 0
	}
	if perAgent == 0 {
		return 0 // no private tail means the requested distinct-agent model is undefined
	}
	return (pool - fixed) / perAgent
}

// Calculate compares independent full-context KV with one shared prefix plus a
// bounded private tail per agent. Model weights are resident once in both cases.
func Calculate(in Input) (Result, error) {
	if in.MemoryBytes == 0 || in.ContextTokens == 0 || in.Layers == 0 || in.KVHeads == 0 || in.HeadDim == 0 || in.KVBytesPerElement == 0 {
		return Result{}, errors.New("memory, context, layers, kv-heads, head-dim, and kv element bytes must be positive")
	}
	if in.TailCapTokens == 0 {
		return Result{}, errors.New("tail cap must be positive")
	}
	if in.SharedPrefixTokens >= in.ContextTokens {
		return Result{}, errors.New("shared prefix must be shorter than the full context")
	}
	if in.ReserveBytes > in.MemoryBytes || in.WeightBytes > in.MemoryBytes-in.ReserveBytes {
		return Result{}, errors.New("reserve plus weights exceed unified memory")
	}
	kvpt, err := mul(2, in.Layers, in.KVHeads, in.HeadDim, in.KVBytesPerElement)
	if err != nil {
		return Result{}, err
	}
	pool := in.MemoryBytes - in.ReserveBytes - in.WeightBytes
	at := func(context uint64) (off, on uint64, offPer, shared, tailPer uint64, err error) {
		offPer, err = mul(kvpt, context)
		if err != nil {
			return
		}
		prefix := min(in.SharedPrefixTokens, context)
		shared, err = mul(kvpt, prefix)
		if err != nil {
			return
		}
		tailTokens := context - prefix
		if tailTokens > in.TailCapTokens {
			tailTokens = in.TailCapTokens
		}
		tailPer, err = mul(kvpt, tailTokens)
		if err != nil {
			return
		}
		off = fit(pool, 0, offPer)
		on = fit(pool, shared, tailPer)
		return
	}
	off, on, offPer, shared, tailPer, err := at(in.ContextTokens)
	if err != nil {
		return Result{}, err
	}

	// A distinct-agent comparison requires at least one private token. Search
	// the requested horizon for the first context where sharing/capping buys a slot.
	start := min(in.SharedPrefixTokens, in.ContextTokens) + 1
	var cross uint64
	found := false
	if start <= in.ContextTokens {
		lo, hi := start, in.ContextTokens
		for lo <= hi {
			mid := lo + (hi-lo)/2
			a, b, _, _, _, e := at(mid)
			if e != nil {
				return Result{}, e
			}
			if b > a {
				cross, found, hi = mid, true, mid-1
			} else {
				lo = mid + 1
			}
		}
	}
	extra := uint64(0)
	if on > off {
		extra = on - off
	}
	return Result{
		Schema: "fak-macfit/1", Provenance: "modeled", MemoryBytes: in.MemoryBytes,
		ReserveBytes: in.ReserveBytes, WeightBytes: in.WeightBytes, KVPoolBytes: pool,
		KVBytesPerToken: kvpt, OffKVBytesPerAgent: offPer, OnSharedKVBytes: shared,
		OnTailKVBytesPerAgent: tailPer, OffAgentsThatFit: off, OnAgentsThatFit: on,
		ExtraAgents: extra, CrossoverContextTokens: cross, CrossoverFound: found,
	}, nil
}

func (r Result) Validate() error {
	if r.Schema != "fak-macfit/1" {
		return fmt.Errorf("unexpected schema %q", r.Schema)
	}
	return nil
}

// GiB is the number of bytes in one gibibyte.
const GiB = uint64(1 << 30)

// DefaultMinHeadroomRatio is the minimum guaranteed headroom ratio to prevent swap (20%).
const DefaultMinHeadroomRatio = 0.20

// TurnkeySLOContextTokens is the out-of-the-box context floor turnkey provisioning
// targets on the flagship Apple-Silicon serving tier (Qwen3.8-27B Q4_K_M on a 36 GiB
// M3 Pro). It is a deliberate SLO default, not a hard ceiling: when the tier's STABLE
// (reserve-based) KV pool holds this many tokens the plan serves exactly this many,
// instead of the larger standard bucket that sits at the edge of the memory envelope
// and spuriously refuses under a momentary live-memory dip. A box that genuinely
// cannot hold it still clamps lower, and an explicit --context still wins.
const TurnkeySLOContextTokens = 20480

// turnkeySLOTier reports whether a tier carries the turnkey context SLO. Only the
// 27B flagship tier does; every other tier keeps the historical bucket selection.
func turnkeySLOTier(tierName string) bool {
	return strings.EqualFold(strings.TrimSpace(tierName), "27B")
}

// TurnkeySLODefault returns the SLO context to serve for a tier when the STABLE KV
// pool (usableBytes - weights, before any transient available-memory clamp) can hold
// it at the planned per-token cost. It returns 0 when the SLO does not apply or the
// box genuinely cannot hold it, preserving the conservative bucket/clamp result.
// Stability is the point: a momentary dip in live free memory must not shrink a
// turnkey default below the SLO on a box whose reserve-based envelope admits it.
func TurnkeySLODefault(tier ModelTier, kvpt, staticKVBytes uint64) uint64 {
	if !turnkeySLOTier(tier.Name) || kvpt == 0 {
		return 0
	}
	if staticKVBytes/kvpt < TurnkeySLOContextTokens {
		return 0
	}
	return TurnkeySLOContextTokens
}

// ModelTier describes an optimal model configuration for a unified memory tier.
type ModelTier struct {
	Name           string `json:"name"`             // e.g. "7B", "27B", "70B"
	ModelID        string `json:"model_id"`         // e.g. "qwen3.8-7b-q4_k_m"
	QuantTier      string `json:"quant_tier"`       // e.g. "Q4_K_M"
	WeightBytes    uint64 `json:"weight_bytes"`     // resident model weight bytes
	Layers         uint64 `json:"layers"`           // transformer layer count
	KVHeads        uint64 `json:"kv_heads"`         // KV head count
	HeadDim        uint64 `json:"head_dim"`         // dimension per head
	MinMemoryBytes uint64 `json:"min_memory_bytes"` // minimum RAM required for this tier
}

// StandardTiers defines the default Apple Silicon tier hierarchy.
var StandardTiers = []ModelTier{
	{
		Name:           "70B",
		ModelID:        "qwen3.8-70b-q4_k_m",
		QuantTier:      "Q4_K_M",
		WeightBytes:    42 * GiB,
		Layers:         80,
		KVHeads:        8,
		HeadDim:        128,
		MinMemoryBytes: 64 * GiB,
	},
	{
		Name:           "27B",
		ModelID:        "qwen3.8-27b-q4_k_m",
		QuantTier:      "Q4_K_M",
		WeightBytes:    16 * GiB,
		Layers:         64,
		KVHeads:        4,
		HeadDim:        128,
		MinMemoryBytes: 36 * GiB,
	},
	{
		Name:           "7B",
		ModelID:        "qwen3.8-7b-q4_k_m",
		QuantTier:      "Q4_K_M",
		WeightBytes:    9 * GiB / 2, // 4.5 GiB
		Layers:         28,
		KVHeads:        4,
		HeadDim:        128,
		MinMemoryBytes: 8 * GiB,
	},
	{
		Name:           "3B",
		ModelID:        "qwen3.8-3b-q4_k_m",
		QuantTier:      "Q4_K_M",
		WeightBytes:    2 * GiB,
		Layers:         16,
		KVHeads:        2,
		HeadDim:        128,
		MinMemoryBytes: 4 * GiB,
	},
}

// SelectModelTier selects the optimal model tier for the given unified memory bytes.
func SelectModelTier(memoryBytes uint64) ModelTier {
	for _, tier := range StandardTiers {
		if memoryBytes >= tier.MinMemoryBytes {
			return tier
		}
	}
	return StandardTiers[len(StandardTiers)-1]
}

// LookupModelTier resolves a friendly tier name (e.g. "3B", "27B") to its
// StandardTiers geometry. It is case-insensitive on the Name field and
// returns false when no tier matches.
func LookupModelTier(name string) (ModelTier, bool) {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return ModelTier{}, false
	}
	for _, tier := range StandardTiers {
		if strings.EqualFold(tier.Name, trimmed) {
			return tier, true
		}
	}
	return ModelTier{}, false
}

// KVBytesPerTokenForTier exposes the per-token KV resident cost for a tier's
// geometry at the given precision, so a caller that swaps in a replacement tier
// can recompute a plan's KV accounting instead of carrying a stale value. An
// empty precision is treated as FP16, matching ConfigureTurnkeyWithOptions.
func KVBytesPerTokenForTier(tier ModelTier, prec model.KVPrecision) (uint64, error) {
	if prec == "" {
		prec = model.KVPrecisionFP16
	}
	return kvBytesPerToken(tier, prec)
}

// TurnkeyOptions configures turnkey model sizing, supporting dynamic memory pressure and display buffer accounting.
type TurnkeyOptions struct {
	AvailableBytes     uint64 // live available memory in bytes (0 = auto-detect via compute.HostSystemMemoryInfo)
	DisplayBufferBytes uint64 // display buffer reservation in bytes (0 = auto-detect)
	StaticOnly         bool   // disable dynamic memory clamping (preserves standard static sizing)

	// KVPrecision optionally selects a quantized KV storage tier (q8_0/q4_0) for the
	// KV-bytes-per-token estimate below. The zero value ("" / fp16) preserves the
	// historical FP16 arithmetic byte-for-byte, so no existing plan changes.
	KVPrecision model.KVPrecision
}

// TurnkeyProfile describes the sizing calculation for turnkey model provisioning.
type TurnkeyProfile struct {
	Schema              string            `json:"schema"`
	MemoryBytes         uint64            `json:"memory_bytes"`
	ReserveBytes        uint64            `json:"reserve_bytes"`
	HeadroomBytes       uint64            `json:"headroom_bytes"`
	HeadroomRatio       float64           `json:"headroom_ratio"`
	Tier                ModelTier         `json:"tier"`
	ContextBudgetTokens uint64            `json:"context_budget_tokens"`
	KVBytesPerToken     uint64            `json:"kv_bytes_per_token"`
	KVPrecision         model.KVPrecision `json:"kv_precision,omitempty"`
	KVPoolBytes         uint64            `json:"kv_pool_bytes"`
	AvailableBytes      uint64            `json:"available_bytes,omitempty"`
	DisplayBufferBytes  uint64            `json:"display_buffer_bytes,omitempty"`
	MemoryPressure      bool              `json:"memory_pressure,omitempty"`
}

// DefaultDisplayBufferReservePerDisplay is the estimated memory reserved by WindowServer
// for high-DPI display compositing and surface backing stores per attached active display.
const DefaultDisplayBufferReservePerDisplay = 1 * GiB

// MinDisplayBufferReserve is the baseline display compositor reservation for the primary display.
const MinDisplayBufferReserve = 512 * 1024 * 1024 // 512 MiB

// DetectDisplayBufferBytes estimates the memory reserved by macOS WindowServer and display
// compositors for active displays.
//
// Precedence:
// 1. FAK_UP_DISPLAY_BUFFER_BYTES environment variable override.
// 2. FAK_UP_DISPLAYS environment variable override (count * DefaultDisplayBufferReservePerDisplay).
// 3. Platform inspection: on macOS, inspect active framebuffer displays via ioreg.
// 4. Fallback default: MinDisplayBufferReserve on darwin, 0 elsewhere.
func DetectDisplayBufferBytes() uint64 {
	if v := os.Getenv("FAK_UP_DISPLAY_BUFFER_BYTES"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			return n
		}
	}
	if v := os.Getenv("FAK_UP_DISPLAYS"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil && n > 0 {
			return n * DefaultDisplayBufferReservePerDisplay
		}
	}
	return detectDisplayBuffersPlatform()
}

func detectDisplayBuffersPlatform() uint64 {
	if runtime.GOOS != "darwin" {
		return 0
	}
	cmd := exec.Command("/usr/sbin/ioreg", "-c", "IOMobileFramebufferShim", "-r", "-k", "DisplayWidth")
	out, err := cmd.Output()
	if err != nil {
		cmd = exec.Command("ioreg", "-c", "IOMobileFramebufferShim", "-r", "-k", "DisplayWidth")
		out, err = cmd.Output()
	}
	if err == nil && len(out) > 0 {
		if total := parseIOMobileFramebufferOutput(out); total > 0 {
			return total
		}
	}
	return MinDisplayBufferReserve
}

func parseIOMobileFramebufferOutput(out []byte) uint64 {
	var totalReservation uint64
	var currentWidth, currentHeight uint64
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "+-o IOMobileFramebufferShim") {
			if currentWidth > 0 && currentHeight > 0 {
				totalReservation += displayReservation(currentWidth, currentHeight)
				currentWidth, currentHeight = 0, 0
			}
		}
		if val, ok := parseDisplayDimension(line, "\"DisplayWidth\" = "); ok {
			currentWidth = val
		} else if val, ok := parseDisplayDimension(line, "\"DisplayHeight\" = "); ok {
			currentHeight = val
		}
	}
	if currentWidth > 0 && currentHeight > 0 {
		totalReservation += displayReservation(currentWidth, currentHeight)
	}
	return totalReservation
}

func parseDisplayDimension(line, prefix string) (uint64, bool) {
	idx := strings.Index(line, prefix)
	if idx == -1 {
		return 0, false
	}
	rest := strings.TrimSpace(line[idx+len(prefix):])
	val, err := strconv.ParseUint(rest, 10, 64)
	if err != nil {
		return 0, false
	}
	return val, true
}

func displayReservation(width, height uint64) uint64 {
	pixels := width * height
	bytesPerFrame := pixels * 4
	res := (bytesPerFrame * 8) + (256 * 1024 * 1024)
	if res < MinDisplayBufferReserve {
		res = MinDisplayBufferReserve
	}
	return res
}

// ConfigureTurnkey auto-selects the optimal model tier and context budget for a given
// memory size, guaranteeing >= 20% memory headroom to prevent swap. It inspects dynamic
// available memory and display buffer reservations to prevent swap thrashing under active memory pressure.
func ConfigureTurnkey(memoryBytes uint64) (TurnkeyProfile, error) {
	return ConfigureTurnkeyWithOptions(memoryBytes, TurnkeyOptions{})
}

// ConfigureTurnkeyWithOptions selects the optimal model tier and context budget with explicit or auto-detected options.
func ConfigureTurnkeyWithOptions(memoryBytes uint64, opts TurnkeyOptions) (TurnkeyProfile, error) {
	if memoryBytes == 0 {
		return TurnkeyProfile{}, errors.New("memory bytes must be positive")
	}

	// 1. Reserve at least 20% of unified memory for OS/WindowServer/headroom to prevent swap.
	reserveBytes := (memoryBytes * 20) / 100
	if reserveBytes == 0 {
		reserveBytes = 1
	}
	usableBytes := memoryBytes - reserveBytes

	// Check if dynamic adjustment is disabled.
	isStatic := opts.StaticOnly || os.Getenv("FAK_UP_STATIC_ONLY") == "1"

	var availableBytes uint64
	var displayBufferBytes uint64
	var memoryPressure bool

	if !isStatic {
		if opts.AvailableBytes > 0 {
			availableBytes = opts.AvailableBytes
		} else if v := os.Getenv("FAK_UP_AVAILABLE_BYTES"); v != "" {
			if n, err := strconv.ParseUint(v, 10, 64); err == nil && n > 0 {
				availableBytes = n
			}
		}

		if opts.DisplayBufferBytes > 0 {
			displayBufferBytes = opts.DisplayBufferBytes
		} else if v := os.Getenv("FAK_UP_DISPLAY_BUFFER_BYTES"); v != "" {
			if n, err := strconv.ParseUint(v, 10, 64); err == nil {
				displayBufferBytes = n
			}
		}

		// Auto-detection when running live without explicit overrides:
		if availableBytes == 0 {
			if os.Getenv("FAK_UP_MEMORY_BYTES") == "" {
				hostTotal, hostFree, known := compute.HostSystemMemoryInfo()
				if known && hostTotal > 0 && memoryBytes == uint64(hostTotal) && hostFree > 0 {
					availableBytes = uint64(hostFree)
					if displayBufferBytes == 0 {
						displayBufferBytes = DetectDisplayBufferBytes()
					}
				}
			}
		}
	}

	// 2. Select the optimal model tier that fits within usableBytes.
	tier := SelectModelTier(memoryBytes)
	if tier.WeightBytes >= usableBytes {
		fitted := false
		for i := len(StandardTiers) - 1; i >= 0; i-- {
			if StandardTiers[i].WeightBytes < usableBytes {
				tier = StandardTiers[i]
				fitted = true
				break
			}
		}
		if !fitted {
			return TurnkeyProfile{}, fmt.Errorf("insufficient memory (%d bytes): cannot guarantee 20%% headroom", memoryBytes)
		}
	}

	// 3. Allocate remaining usable memory for KV cache pool.
	staticKVBytes := usableBytes - tier.WeightBytes

	// KV bytes per token. The default (zero-value) precision is FP16 over a K and a V
	// vector of KVHeads*HeadDim elements, which is byte-identical to the historical
	// 2*Layers*KVHeads*HeadDim*2 estimate. A quantized tier (q8_0/q4_0) packs the same
	// two vectors denser, so the same KV pool holds proportionally more context.
	kvPrecision := opts.KVPrecision
	if kvPrecision == "" {
		kvPrecision = model.KVPrecisionFP16
	}
	kvpt, err := kvBytesPerToken(tier, kvPrecision)
	if err != nil {
		return TurnkeyProfile{}, err
	}

	kvPoolBytes := staticKVBytes

	// Dynamic capacity accounting & KV cache clamping:
	// Working spine: compute.HostSystemMemoryInfo -> dynamic available memory read -> display buffer reservation subtraction -> tier and context sizing -> safe execution plan.
	if availableBytes > 0 {
		var netAvailable uint64
		if availableBytes > displayBufferBytes {
			netAvailable = availableBytes - displayBufferBytes
		}

		if netAvailable > tier.WeightBytes {
			headroomForKV := netAvailable - tier.WeightBytes
			if headroomForKV < staticKVBytes {
				memoryPressure = true
				safetyMargin := (headroomForKV * 15) / 100
				safeKVBytes := headroomForKV - safetyMargin
				if safeKVBytes < kvPoolBytes {
					kvPoolBytes = safeKVBytes
				}
			}
		} else {
			// Severe memory pressure: weights consume net available memory.
			// Clamp KV pool to minimal floor to prevent runaway allocation.
			memoryPressure = true
			kvPoolBytes = 512 * kvpt
		}
	}

	// 4. Calculate maximum context tokens that fit in the pool.
	maxTokens := kvPoolBytes / kvpt
	if maxTokens == 0 {
		maxTokens = 512
	}

	// Bucket context budget to standard boundaries without exceeding maxTokens.
	// TurnkeySLOContextTokens (20480) is a first-class bucket so the flagship tier
	// can select the SLO instead of the larger bucket above it.
	var contextBudget uint64
	for _, bucket := range []uint64{65536, 32768, TurnkeySLOContextTokens, 16384, 8192, 4096, 2048, 1024, 512} {
		if maxTokens >= bucket {
			contextBudget = bucket
			break
		}
	}
	if contextBudget == 0 {
		contextBudget = maxTokens
	}

	// Memory pressure signals trigger conservative context budget reductions instead of swap thrashing.
	// A quantized KV tier makes the same KV pool hold far more tokens, so the FP16
	// 8192-token ceiling is only applied when KV is stored at full FP16 precision;
	// the smaller hard floors below still apply under severe pressure.
	if memoryPressure {
		if !kvQuantized(kvPrecision) && contextBudget > 8192 {
			contextBudget = 8192
		}
		if kvPoolBytes < 2*GiB && contextBudget > 2048 {
			contextBudget = 2048
		}
		if kvPoolBytes < 1*GiB && contextBudget > 1024 {
			contextBudget = 1024
		}
		if contextBudget == 0 {
			contextBudget = 512
		}
	}

	// SLO default for the flagship tier: a momentary live-memory dip must not shrink
	// the out-of-the-box context below the 20k SLO when the STABLE, reserve-based pool
	// (staticKVBytes = usableBytes - weights) can hold it. The 20% reserve already
	// absorbs OS/other-app pressure, so the transient clamp above must not double-count
	// it and silently under-serve. When the box genuinely cannot hold the SLO the floor
	// is 0 and the clamped bucket stands; an explicit --context still wins downstream.
	if slo := TurnkeySLODefault(tier, kvpt, staticKVBytes); slo > 0 {
		contextBudget = slo
		if used := contextBudget * kvpt; kvPoolBytes < used {
			// Keep the reported pool self-consistent with the served context. This is
			// still >= the context's own resident bytes, so HeadroomBytes never reports
			// a negative (over-committed) allocation.
			kvPoolBytes = used
		}
	}

	// 5. Total used memory = weights + KV cache context budget.
	usedKVBytes := contextBudget * kvpt
	allocatedBytes := tier.WeightBytes + usedKVBytes
	headroomBytes := memoryBytes - allocatedBytes
	headroomRatio := float64(headroomBytes) / float64(memoryBytes)

	return TurnkeyProfile{
		Schema:              "fak-macfit-turnkey/1",
		MemoryBytes:         memoryBytes,
		ReserveBytes:        reserveBytes,
		HeadroomBytes:       headroomBytes,
		HeadroomRatio:       headroomRatio,
		Tier:                tier,
		ContextBudgetTokens: contextBudget,
		KVBytesPerToken:     kvpt,
		KVPrecision:         kvPrecision,
		KVPoolBytes:         kvPoolBytes,
		AvailableBytes:      availableBytes,
		DisplayBufferBytes:  displayBufferBytes,
		MemoryPressure:      memoryPressure,
	}, nil
}

// DetectUnifiedMemoryInfo reports both physical total and live available memory on Apple Silicon.
//
// Precedence:
// 1. FAK_UP_MEMORY_BYTES override for total memory.
// 2. FAK_UP_AVAILABLE_BYTES override for available memory.
// 3. Live memory inspection via compute.HostSystemMemoryInfo().
// 4. Default fallback: 16 GiB total, 80% available.
func DetectUnifiedMemoryInfo() (total, available uint64, err error) {
	if v := os.Getenv("FAK_UP_MEMORY_BYTES"); v != "" {
		if n, parseErr := strconv.ParseUint(v, 10, 64); parseErr == nil && n > 0 {
			total = n
		}
	}
	if v := os.Getenv("FAK_UP_AVAILABLE_BYTES"); v != "" {
		if n, parseErr := strconv.ParseUint(v, 10, 64); parseErr == nil && n > 0 {
			available = n
		}
	}

	hostTotal, hostFree, known := compute.HostSystemMemoryInfo()
	if total == 0 {
		if known && hostTotal > 0 {
			total = uint64(hostTotal)
		} else {
			total = 16 * GiB
		}
	}

	if available == 0 {
		if v := os.Getenv("FAK_UP_MEMORY_BYTES"); v != "" {
			// When total memory is explicitly overridden in tests without FAK_UP_AVAILABLE_BYTES,
			// assume standard unconstrained 80% usable capacity.
			available = (total * 80) / 100
		} else if known && hostFree > 0 {
			available = uint64(hostFree)
			if available > total {
				available = total
			}
		} else {
			available = (total * 80) / 100
		}
	}

	return total, available, nil
}

// DetectUnifiedMemory inspects Apple Silicon physical memory.
// It reports physical total memory for backwards-compatibility with callers expecting (uint64, error).
// Call DetectUnifiedMemoryInfo to inspect both total and live available memory.
func DetectUnifiedMemory() (uint64, error) {
	total, _, err := DetectUnifiedMemoryInfo()
	return total, err
}
