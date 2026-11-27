// Package modelloadplan selects a model artifact before any download or allocation.
//
// Context-aware fit contract (usable context, requested/effective/clamp ladder,
// multi-user inverse, and graded headroom ratios) adapted from AlexsJones/llmfit
// (MIT, pinned at 7f47fca32c27230ccfbb012994a0d0bff03bc283; see
// THIRD_PARTY_NOTICES.md). Uses Fak's own kvbudget geometry and fixed
// per-session state, keeps admission and grade thresholds aligned, and always
// propagates clamped contexts into runtime sizing.
package modelloadplan

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

const (
	Schema       = "fak.model-load-plan/v1alpha1"
	ModelID      = "Qwen/Qwen3.8-27B"
	GGUFRevision = "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"
	OpenRouterID = "qwen/qwen3.8-27b"

	DefaultNativeContextTokens    = 262144
	DefaultEstimatedContextTokens = 8192

	// Qwen3.8-27B hybrid GDN KV defaults: 16 full-attention layers (out of 64)
	// scale with context at 4 KV heads x 128 head_dim x F16 (32 KiB/token),
	// while the 48 linear-attention GDN layers contribute fixed per-session state.
	DefaultFullAttnLayers          = 16
	DefaultKVHeads                 = 4
	DefaultHeadDim                 = 128
	DefaultFixedSessionBytes int64 = 3355443

	FitGradePerfect  = "perfect"
	FitGradeGood     = "good"
	FitGradeMarginal = "marginal"
	FitGradeTooTight = "too_tight"
	FitGradeUnknown  = "unknown"

	PerfectMaxUtilization  = 0.60
	GoodMaxUtilization     = 0.85
	MarginalMaxUtilization = 0.98
)

// DefaultKVShape returns Qwen3.8-27B's full-attention KV cache geometry.
func DefaultKVShape() kvbudget.Shape {
	return kvbudget.Shape{
		Kind:       kvbudget.MHA,
		Layers:     DefaultFullAttnLayers,
		NumKVHeads: DefaultKVHeads,
		HeadDim:    DefaultHeadDim,
		VHeadDim:   DefaultHeadDim,
	}
}

// DefaultKVQuant returns the default KV precision (F16) for context sizing.
func DefaultKVQuant() kvbudget.Quant {
	return kvbudget.F16
}

type Request struct {
	Setup               string          `json:"setup"`
	Goal                string          `json:"goal"`
	LocalPolicy         string          `json:"local_policy"`
	Memory              string          `json:"memory_topology"`
	DeviceBytes         int64           `json:"device_bytes"`
	HostBytes           int64           `json:"host_bytes"`
	DiskBytes           int64           `json:"disk_bytes"`
	ContextTokens       int             `json:"context_tokens,omitempty"`
	Users               int             `json:"users,omitempty"`
	AllowContextClamp   bool            `json:"allow_context_clamp,omitempty"`
	NativeContextTokens int             `json:"native_context_tokens,omitempty"`
	KVShape             *kvbudget.Shape `json:"kv_shape,omitempty"`
	KVQuant             *kvbudget.Quant `json:"kv_quant,omitempty"`
	FixedSessionBytes   *int64          `json:"fixed_session_bytes,omitempty"`
}

// ContextRung reports memory sizing and fit grade at one context length.
type ContextRung struct {
	ContextTokens   int     `json:"context_tokens"`
	KVBytes         int64   `json:"kv_bytes"`
	RuntimeBytes    int64   `json:"runtime_bytes"`
	HeadroomBytes   int64   `json:"headroom_bytes"`
	PoolUtilization float64 `json:"pool_utilization,omitempty"`
	Fits            bool    `json:"fits"`
	FitGrade        string  `json:"fit_grade"`
	MaxUsers        int     `json:"max_users"`
}

// ContextSizing captures the model and pool parameters required to answer
// MaxContextFor and MaxContextForUsers independently of the initial request.
type ContextSizing struct {
	MemoryTopology      string         `json:"memory_topology"`
	PoolBytes           int64          `json:"pool_bytes"`
	DeviceBytes         int64          `json:"device_bytes,omitempty"`
	BaseBytes           int64          `json:"base_bytes"`
	FixedSessionBytes   int64          `json:"fixed_session_bytes,omitempty"`
	NativeContextTokens int            `json:"native_context_tokens"`
	KVShape             kvbudget.Shape `json:"kv_shape"`
	KVQuant             kvbudget.Quant `json:"kv_quant"`
}

// UserContextBudget is the max-context-for-users inverse across headroom grades.
type UserContextBudget struct {
	Users                 int     `json:"users"`
	MaxContextTokens      int     `json:"max_context_tokens"`
	MarginalContextTokens int     `json:"marginal_context_tokens"`
	GoodContextTokens     int     `json:"good_context_tokens"`
	PerfectContextTokens  int     `json:"perfect_context_tokens"`
	AvailableKVBytes      int64   `json:"available_kv_bytes"`
	HeadroomBytes         int64   `json:"headroom_bytes"`
	HeadroomRatio         float64 `json:"headroom_ratio"`
	FitGrade              string  `json:"fit_grade"`
}

type Candidate struct {
	ID                     string         `json:"id"`
	Kind                   string         `json:"kind"`
	Quantization           string         `json:"quantization,omitempty"`
	URI                    string         `json:"uri"`
	ArtifactBytes          int64          `json:"artifact_bytes,omitempty"`
	WeightsBytes           int64          `json:"weights_bytes,omitempty"`
	OverheadBytes          int64          `json:"overhead_bytes,omitempty"`
	FixedSessionBytes      int64          `json:"fixed_session_bytes,omitempty"`
	KVBytes                int64          `json:"kv_bytes,omitempty"`
	KVBytesPerToken        float64        `json:"kv_bytes_per_token,omitempty"`
	RuntimeBytes           int64          `json:"runtime_bytes,omitempty"`
	DeviceBytes            int64          `json:"device_bytes,omitempty"`
	HostBytes              int64          `json:"host_bytes,omitempty"`
	PoolBytes              int64          `json:"pool_bytes,omitempty"`
	HeadroomBytes          int64          `json:"headroom_bytes,omitempty"`
	PoolUtilization        float64        `json:"pool_utilization,omitempty"`
	FitGrade               string         `json:"fit_grade,omitempty"`
	RequestedContextTokens int            `json:"requested_context_tokens,omitempty"`
	EffectiveContextTokens int            `json:"effective_context_tokens,omitempty"`
	NativeContextTokens    int            `json:"native_context_tokens,omitempty"`
	UsableContextTokens    int            `json:"usable_context_tokens,omitempty"`
	ContextClamped         bool           `json:"context_clamped,omitempty"`
	ContextClampReason     string         `json:"context_clamp_reason,omitempty"`
	ContextLadder          []ContextRung  `json:"context_ladder,omitempty"`
	Users                  int            `json:"users,omitempty"`
	MaxUsers               int            `json:"max_users,omitempty"`
	ContextSizing          *ContextSizing `json:"context_sizing,omitempty"`
	Fits                   bool           `json:"fits"`
	Reasons                []string       `json:"reasons"`
	NextCommand            string         `json:"next_command,omitempty"`
}

// MaxContextFor returns the maximum context length in tokens that fits for the
// given concurrent user count within the Marginal headroom ceiling, capped at
// NativeContextTokens. Returns 0 when capacity is unknown or exhausted.
func (c Candidate) MaxContextFor(users int) int {
	if c.ContextSizing == nil {
		return 0
	}
	return MaxContextForUsers(*c.ContextSizing, users).MaxContextTokens
}

// MaxContextForUsers returns the max-context-for-users inverse across all
// headroom grades (Perfect, Good, Marginal) for this candidate.
func (c Candidate) MaxContextForUsers(users int) UserContextBudget {
	if c.ContextSizing == nil {
		return UserContextBudget{Users: users, FitGrade: FitGradeUnknown}
	}
	return MaxContextForUsers(*c.ContextSizing, users)
}

type Plan struct {
	Schema     string      `json:"schema"`
	Model      string      `json:"model"`
	Request    Request     `json:"request"`
	Selected   *Candidate  `json:"selected,omitempty"`
	Candidates []Candidate `json:"candidates"`
	Provenance []string    `json:"provenance"`
}

type variant struct {
	quant string
	file  string
	bytes int64
}

var variants = []variant{
	{"IQ2_S", "Qwen3.8-27B-UD-IQ2_S.gguf", 8371970048},
	{"Q3_K_XL", "Qwen3.8-27B-UD-Q3_K_XL.gguf", 13146393504},
	{"Q4_K_M", "Qwen3.8-27B-UD-Q4_K_M.gguf", 16464440224},
	{"Q5_K_M", "Qwen3.8-27B-UD-Q5_K_M.gguf", 19771509664},
	{"Q6_K", "Qwen3.8-27B-UD-Q6_K.gguf", 21983677344},
	{"Q8_0", "Qwen3.8-27B-Q8_0.gguf", 29047086048},
}

func Normalize(r Request) (Request, error) {
	r.Setup = strings.ToLower(strings.TrimSpace(r.Setup))
	r.Goal = strings.ToLower(strings.TrimSpace(r.Goal))
	r.LocalPolicy = strings.ToLower(strings.TrimSpace(r.LocalPolicy))
	r.Memory = strings.ToLower(strings.TrimSpace(r.Memory))
	if r.Setup == "" {
		r.Setup = "personal"
	}
	if r.Goal == "" {
		r.Goal = "balanced"
	}
	if r.LocalPolicy == "" {
		r.LocalPolicy = "auto"
	}
	if r.Memory == "" {
		r.Memory = "unified"
	}
	if !oneOf(r.Setup, "personal", "shared", "batch") {
		return r, fmt.Errorf("setup must be personal, shared, or batch")
	}
	if !oneOf(r.Goal, "balanced", "quality", "latency", "cost") {
		return r, fmt.Errorf("goal must be balanced, quality, latency, or cost")
	}
	if !oneOf(r.LocalPolicy, "auto", "require", "disable") {
		return r, fmt.Errorf("local policy must be auto, require, or disable")
	}
	if !oneOf(r.Memory, "unified", "split") {
		return r, fmt.Errorf("memory topology must be unified or split")
	}
	if r.DeviceBytes < 0 || r.HostBytes < 0 || r.DiskBytes < 0 {
		return r, fmt.Errorf("capacities cannot be negative")
	}
	if r.ContextTokens < 0 || r.Users < 0 || r.NativeContextTokens < 0 {
		return r, fmt.Errorf("context tokens, users, and native context cannot be negative")
	}
	if r.FixedSessionBytes != nil && *r.FixedSessionBytes < 0 {
		return r, fmt.Errorf("fixed session bytes cannot be negative")
	}
	if r.KVQuant != nil && (math.IsNaN(r.KVQuant.BytesPerElem) || math.IsInf(r.KVQuant.BytesPerElem, 0) || r.KVQuant.BytesPerElem <= 0) {
		return r, fmt.Errorf("kv quant bytes per element must be positive and finite")
	}
	if r.KVShape != nil && (r.KVShape.Layers <= 0 || r.KVShape.KVElemsPerStream(1) <= 0) {
		return r, fmt.Errorf("kv shape must have positive layers and positive KV elements per token")
	}
	if r.Users == 0 {
		r.Users = 1
	}
	if r.NativeContextTokens == 0 {
		r.NativeContextTokens = DefaultNativeContextTokens
	}
	if r.ContextTokens == 0 {
		r.ContextTokens = minInt(r.NativeContextTokens, DefaultEstimatedContextTokens)
	}
	return r, nil
}

func resolveContextGeometry(r Request) (kvbudget.Shape, kvbudget.Quant, int64) {
	shape := DefaultKVShape()
	fixedSession := DefaultFixedSessionBytes
	if r.KVShape != nil {
		shape = *r.KVShape
		// A caller supplying an explicit model KV shape owns any non-KV session
		// state too; avoid charging Qwen3.8-27B's GDN state to unrelated shapes.
		fixedSession = 0
	}
	quant := DefaultKVQuant()
	if r.KVQuant != nil {
		quant = *r.KVQuant
	}
	if r.FixedSessionBytes != nil {
		fixedSession = *r.FixedSessionBytes
	}
	return shape, quant, fixedSession
}

func declaredPoolBytes(r Request) int64 {
	if r.Memory == "unified" {
		if r.DeviceBytes > 0 {
			return r.DeviceBytes
		}
		return r.HostBytes
	}
	if r.DeviceBytes == 0 && r.HostBytes == 0 {
		return 0
	}
	return addSat(r.DeviceBytes, r.HostBytes)
}

func Build(req Request) (Plan, error) {
	r, err := Normalize(req)
	if err != nil {
		return Plan{}, err
	}
	p := Plan{Schema: Schema, Model: ModelID, Request: r, Provenance: []string{
		"Hugging Face unsloth/Qwen3.8-27B-GGUF file metadata retrieved 2026-08-19",
		"OpenRouter qwen/qwen3.8-27b catalog metadata retrieved 2026-08-19",
	}}
	reserve := map[string]int64{"personal": 2 << 30, "shared": 6 << 30, "batch": 10 << 30}[r.Setup]
	shape, quant, fixedSession := resolveContextGeometry(r)
	pool := declaredPoolBytes(r)
	native := r.NativeContextTokens
	requestedCtx := r.ContextTokens
	targetCtx := minInt(requestedCtx, native)

	for _, v := range variants {
		baseBytes := addSat(addSat(v.bytes, v.bytes/12), reserve)
		sizing := ContextSizing{
			MemoryTopology:      r.Memory,
			PoolBytes:           pool,
			DeviceBytes:         r.DeviceBytes,
			BaseBytes:           baseBytes,
			FixedSessionBytes:   fixedSession,
			NativeContextTokens: native,
			KVShape:             shape,
			KVQuant:             quant,
		}
		usableCtx := maxContextForPool(sizing, r.Users, MarginalMaxUtilization, false)
		effectiveCtx := targetCtx
		clamped := requestedCtx > native
		clampReason := ""
		if clamped {
			clampReason = "clamped_to_native"
		}
		if r.AllowContextClamp && pool > 0 && usableCtx > 0 && targetCtx > usableCtx {
			effectiveCtx = usableCtx
			clamped = true
			if clampReason != "" {
				clampReason = "clamped_to_native_and_usable_context"
			} else {
				clampReason = "clamped_to_usable_context"
			}
		}

		kvBytes, runtimeBytes, ok := computeRuntime(baseBytes, fixedSession, shape, quant, effectiveCtx, r.Users)
		if !ok {
			runtimeBytes = math.MaxInt64
		}
		sessionBytes := mulSat(int64(r.Users), fixedSession)
		deviceBytes, hostBytes := placement(r, runtimeBytes)

		var kvPerTok float64
		if effectiveCtx > 0 {
			kvPerTok = shape.KVBytesPerStream(effectiveCtx, quant) / float64(effectiveCtx)
		}
		headroomBytes := int64(0)
		poolUtil := 0.0
		fitGrade := FitGradeUnknown
		if pool > 0 {
			headroomBytes = pool - runtimeBytes
			poolUtil = float64(runtimeBytes) / float64(pool)
			fitGrade = gradeFit(pool, runtimeBytes, r.DeviceBytes, r.Memory)
		}

		c := Candidate{
			ID:                     "local/" + strings.ToLower(v.quant),
			Kind:                   "local",
			Quantization:           v.quant,
			URI:                    "hf://unsloth/Qwen3.8-27B-GGUF@" + GGUFRevision + "/" + v.file,
			ArtifactBytes:          v.bytes,
			WeightsBytes:           v.bytes,
			OverheadBytes:          addSat(v.bytes/12+reserve, sessionBytes),
			FixedSessionBytes:      sessionBytes,
			KVBytes:                kvBytes,
			KVBytesPerToken:        kvPerTok,
			RuntimeBytes:           runtimeBytes,
			DeviceBytes:            deviceBytes,
			HostBytes:              hostBytes,
			PoolBytes:              pool,
			HeadroomBytes:          headroomBytes,
			PoolUtilization:        poolUtil,
			FitGrade:               fitGrade,
			RequestedContextTokens: requestedCtx,
			EffectiveContextTokens: effectiveCtx,
			NativeContextTokens:    native,
			UsableContextTokens:    usableCtx,
			ContextClamped:         clamped,
			ContextClampReason:     clampReason,
			ContextLadder:          buildContextLadder(sizing, r.Users, targetCtx, effectiveCtx, usableCtx),
			Users:                  r.Users,
			MaxUsers:               maxUsersForContext(sizing, effectiveCtx, MarginalMaxUtilization),
			ContextSizing:          &sizing,
		}
		c.Fits, c.Reasons = localFit(r, c)
		c.NextCommand = "fak serve --gguf " + c.URI + " --model qwen38:27b"
		p.Candidates = append(p.Candidates, c)
	}
	hosted := Candidate{ID: "hosted/openrouter", Kind: "hosted", URI: OpenRouterID, Fits: r.LocalPolicy != "require",
		Reasons:     []string{"hosted route avoids local weight residency and disk use"},
		NextCommand: "fak serve --provider openai --base-url https://openrouter.ai/api/v1 --model " + OpenRouterID + " --api-key-env OPENROUTER_API_KEY"}
	if !hosted.Fits {
		hosted.Reasons = []string{"rejected because local policy is require"}
	}
	p.Candidates = append(p.Candidates, hosted)
	rank(p.Candidates, r)
	for i := range p.Candidates {
		if p.Candidates[i].Fits {
			chosen := p.Candidates[i]
			p.Selected = &chosen
			break
		}
	}
	return p, nil
}

func placement(r Request, runtime int64) (device, host int64) {
	if r.Memory == "unified" {
		return runtime, 0
	}
	device = min64(runtime, r.DeviceBytes)
	return device, runtime - device
}

func localFit(r Request, c Candidate) (bool, []string) {
	if r.LocalPolicy == "disable" {
		return false, []string{"rejected because local policy is disable"}
	}
	var reasons []string
	if r.DiskBytes == 0 {
		reasons = append(reasons, "disk capacity is unknown")
	} else if c.ArtifactBytes > r.DiskBytes {
		reasons = append(reasons, "artifact exceeds disk capacity")
	}
	if r.Memory == "unified" {
		cap := r.DeviceBytes
		if cap == 0 {
			cap = r.HostBytes
		}
		if cap == 0 {
			reasons = append(reasons, "unified memory capacity is unknown")
		} else if c.RuntimeBytes > cap {
			reasons = append(reasons, "runtime estimate exceeds unified memory capacity")
		} else if c.FitGrade == FitGradeTooTight {
			reasons = append(reasons, "runtime estimate exceeds 98% unified memory headroom threshold")
		}
	} else {
		if r.DeviceBytes == 0 && r.HostBytes == 0 {
			reasons = append(reasons, "device and host memory capacities are unknown")
		} else if c.RuntimeBytes > addSat(r.DeviceBytes, r.HostBytes) {
			reasons = append(reasons, "runtime estimate exceeds combined device and host memory")
		} else if c.FitGrade == FitGradeTooTight {
			reasons = append(reasons, "runtime estimate exceeds 98% combined memory headroom threshold")
		} else if c.HostBytes > 0 {
			reasons = append(reasons, fmt.Sprintf("plans %d bytes of host offload", c.HostBytes))
		}
	}
	if c.ContextClamped {
		reasons = append(reasons, fmt.Sprintf("context clamped from %d to %d tokens (%s)", c.RequestedContextTokens, c.EffectiveContextTokens, c.ContextClampReason))
	}
	fit := true
	for _, reason := range reasons {
		if strings.Contains(reason, "unknown") || strings.Contains(reason, "exceeds") || strings.Contains(reason, "rejected") {
			fit = false
		}
	}
	if fit {
		reasons = append(reasons, "fits declared memory and disk capacities")
	}
	return fit, reasons
}

func rank(cs []Candidate, r Request) {
	preferred := map[string][]string{
		"balanced": {"Q4_K_M", "Q5_K_M", "Q3_K_XL", "Q6_K", "IQ2_S", "Q8_0"},
		"quality":  {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_XL", "IQ2_S"},
		"latency":  {"Q4_K_M", "Q3_K_XL", "IQ2_S", "Q5_K_M", "Q6_K", "Q8_0"},
		"cost":     {"Q4_K_M", "Q3_K_XL", "IQ2_S", "Q5_K_M", "Q6_K", "Q8_0"},
	}[r.Goal]
	pos := map[string]int{}
	for i, q := range preferred {
		pos[q] = i
	}
	sort.SliceStable(cs, func(i, j int) bool {
		a, b := cs[i], cs[j]
		if a.Fits != b.Fits {
			return a.Fits
		}
		if a.Kind != b.Kind {
			if r.LocalPolicy == "disable" || r.Goal == "cost" || (r.Setup == "shared" && r.LocalPolicy == "auto") {
				return a.Kind == "hosted"
			}
			return a.Kind == "local"
		}
		if a.Kind == "local" {
			aMemClamped := a.ContextClampReason == "clamped_to_usable_context" || a.ContextClampReason == "clamped_to_native_and_usable_context"
			bMemClamped := b.ContextClampReason == "clamped_to_usable_context" || b.ContextClampReason == "clamped_to_native_and_usable_context"
			if aMemClamped != bMemClamped {
				return !aMemClamped
			}
			return pos[a.Quantization] < pos[b.Quantization]
		}
		return a.ID < b.ID
	})
}

// MaxContextForUsers computes the max context tokens for users across Perfect,
// Good, and Marginal headroom bands for the given ContextSizing.
func MaxContextForUsers(sizing ContextSizing, users int) UserContextBudget {
	if sizing.PoolBytes <= 0 || sizing.NativeContextTokens <= 0 || users <= 0 {
		return UserContextBudget{Users: users, FitGrade: FitGradeUnknown}
	}
	marginalCtx := maxContextForPool(sizing, users, MarginalMaxUtilization, false)
	goodCtx := maxContextForPool(sizing, users, GoodMaxUtilization, false)
	perfectCtx := maxContextForPool(sizing, users, PerfectMaxUtilization, true)

	admitted := admittedPoolBytes(sizing.PoolBytes, MarginalMaxUtilization)
	nonKV := addSat(sizing.BaseBytes, mulSat(int64(users), sizing.FixedSessionBytes))
	availKV := admitted - nonKV
	if availKV < 0 {
		availKV = 0
	}

	_, rtAtMax, ok := computeRuntime(sizing.BaseBytes, sizing.FixedSessionBytes, sizing.KVShape, sizing.KVQuant, marginalCtx, users)
	if !ok {
		rtAtMax = math.MaxInt64
	}
	var headroomBytes int64
	var headroomRatio float64
	grade := FitGradeTooTight
	if marginalCtx > 0 {
		headroomBytes = sizing.PoolBytes - rtAtMax
		headroomRatio = float64(headroomBytes) / float64(sizing.PoolBytes)
		grade = gradeFit(sizing.PoolBytes, rtAtMax, sizing.DeviceBytes, sizing.MemoryTopology)
	}

	return UserContextBudget{
		Users:                 users,
		MaxContextTokens:      marginalCtx,
		MarginalContextTokens: marginalCtx,
		GoodContextTokens:     goodCtx,
		PerfectContextTokens:  perfectCtx,
		AvailableKVBytes:      availKV,
		HeadroomBytes:         headroomBytes,
		HeadroomRatio:         headroomRatio,
		FitGrade:              grade,
	}
}

func admittedPoolBytes(pool int64, maxUtilization float64) int64 {
	if pool <= 0 || math.IsNaN(maxUtilization) || math.IsInf(maxUtilization, 0) || maxUtilization <= 0 {
		return 0
	}
	if maxUtilization >= 1 {
		return pool
	}
	return int64(math.Floor(float64(pool) * maxUtilization))
}

func maxContextForPool(sizing ContextSizing, users int, maxUtilization float64, disallowHostOffload bool) int {
	if sizing.PoolBytes <= 0 || sizing.NativeContextTokens <= 0 || users <= 0 {
		return 0
	}
	if sizing.KVShape.Layers <= 0 || sizing.KVShape.KVElemsPerStream(1) <= 0 || sizing.KVQuant.BytesPerElem <= 0 {
		return 0
	}
	admitted := admittedPoolBytes(sizing.PoolBytes, maxUtilization)
	if disallowHostOffload && sizing.MemoryTopology == "split" && admitted > sizing.DeviceBytes {
		admitted = sizing.DeviceBytes
	}
	nonKV := addSat(sizing.BaseBytes, mulSat(int64(users), sizing.FixedSessionBytes))
	availKV := admitted - nonKV
	if availKV <= 0 {
		return 0
	}
	if sizing.KVShape.PerLayer == nil {
		slopePerUser := float64(sizing.KVShape.KVElemsPerToken()) * sizing.KVQuant.BytesPerElem * float64(users)
		if slopePerUser <= 0 || math.IsNaN(slopePerUser) || math.IsInf(slopePerUser, 0) {
			return 0
		}
		est := minInt(sizing.NativeContextTokens, int(math.Floor(float64(availKV)/slopePerUser)))
		for est > 0 {
			kv, ok := kvBytesForUsers(sizing.KVShape, sizing.KVQuant, est, users)
			if ok && kv <= availKV {
				break
			}
			est--
		}
		for est < sizing.NativeContextTokens {
			kv, ok := kvBytesForUsers(sizing.KVShape, sizing.KVQuant, est+1, users)
			if !ok || kv > availKV {
				break
			}
			est++
		}
		return est
	}
	return sort.Search(sizing.NativeContextTokens, func(i int) bool {
		kv, ok := kvBytesForUsers(sizing.KVShape, sizing.KVQuant, i+1, users)
		return !ok || kv > availKV
	})
}

func maxUsersForContext(sizing ContextSizing, ctx int, maxUtilization float64) int {
	if sizing.PoolBytes <= 0 || ctx <= 0 {
		return 0
	}
	admitted := admittedPoolBytes(sizing.PoolBytes, maxUtilization)
	availForUsers := admitted - sizing.BaseBytes
	if availForUsers <= 0 {
		return 0
	}
	elems := sizing.KVShape.KVElemsPerStream(ctx)
	if elems <= 0 || sizing.KVQuant.BytesPerElem <= 0 {
		return 0
	}
	perUserApprox := float64(sizing.FixedSessionBytes) + float64(elems)*sizing.KVQuant.BytesPerElem
	if perUserApprox <= 0 || math.IsNaN(perUserApprox) || math.IsInf(perUserApprox, 0) {
		return 0
	}
	est := int(math.Min(float64(math.MaxInt32), math.Floor(float64(availForUsers)/perUserApprox)))
	fitsUsers := func(u int) bool {
		if u <= 0 {
			return true
		}
		kv, ok := kvBytesForUsers(sizing.KVShape, sizing.KVQuant, ctx, u)
		if !ok {
			return false
		}
		totalUsers := addSat(mulSat(int64(u), sizing.FixedSessionBytes), kv)
		return totalUsers <= availForUsers
	}
	for est > 0 && !fitsUsers(est) {
		est--
	}
	for est < math.MaxInt32 && fitsUsers(est+1) {
		est++
	}
	return est
}

func buildContextLadder(sizing ContextSizing, users, targetCtx, effectiveCtx, usableCtx int) []ContextRung {
	baseRungs := []int{512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144}
	seen := make(map[int]struct{}, len(baseRungs)+4)
	rungs := make([]int, 0, len(baseRungs)+4)
	addRung := func(ctx int) {
		if ctx <= 0 || ctx > sizing.NativeContextTokens {
			return
		}
		if _, ok := seen[ctx]; ok {
			return
		}
		seen[ctx] = struct{}{}
		rungs = append(rungs, ctx)
	}
	for _, ctx := range baseRungs {
		addRung(ctx)
	}
	addRung(targetCtx)
	if targetCtx/2 > 0 {
		addRung(targetCtx / 2)
	}
	addRung(effectiveCtx)
	addRung(usableCtx)
	addRung(sizing.NativeContextTokens)
	sort.Ints(rungs)

	out := make([]ContextRung, 0, len(rungs))
	for _, ctx := range rungs {
		kv, rt, ok := computeRuntime(sizing.BaseBytes, sizing.FixedSessionBytes, sizing.KVShape, sizing.KVQuant, ctx, users)
		if !ok {
			rt = math.MaxInt64
		}
		grade := FitGradeUnknown
		var headroom int64
		var util float64
		if sizing.PoolBytes > 0 {
			headroom = sizing.PoolBytes - rt
			util = float64(rt) / float64(sizing.PoolBytes)
			grade = gradeFit(sizing.PoolBytes, rt, sizing.DeviceBytes, sizing.MemoryTopology)
		}
		out = append(out, ContextRung{
			ContextTokens:   ctx,
			KVBytes:         kv,
			RuntimeBytes:    rt,
			HeadroomBytes:   headroom,
			PoolUtilization: util,
			Fits:            sizing.PoolBytes > 0 && grade != FitGradeTooTight && grade != FitGradeUnknown,
			FitGrade:        grade,
			MaxUsers:        maxUsersForContext(sizing, ctx, MarginalMaxUtilization),
		})
	}
	return out
}

func gradeFit(pool, runtime, deviceBytes int64, memoryTopology string) string {
	if pool <= 0 {
		return FitGradeUnknown
	}
	ratio := float64(runtime) / float64(pool)
	if math.IsNaN(ratio) || math.IsInf(ratio, 0) || ratio < 0 {
		return FitGradeTooTight
	}
	if ratio > MarginalMaxUtilization {
		return FitGradeTooTight
	}
	offloaded := memoryTopology == "split" && runtime > deviceBytes
	if ratio <= PerfectMaxUtilization && !offloaded {
		return FitGradePerfect
	}
	if ratio <= GoodMaxUtilization {
		return FitGradeGood
	}
	return FitGradeMarginal
}

func kvBytesForUsers(shape kvbudget.Shape, quant kvbudget.Quant, ctx, users int) (int64, bool) {
	if ctx <= 0 || users <= 0 {
		return 0, true
	}
	if shape.Layers <= 0 || quant.BytesPerElem <= 0 || math.IsNaN(quant.BytesPerElem) || math.IsInf(quant.BytesPerElem, 0) {
		return 0, false
	}
	elems := shape.KVElemsPerStream(ctx)
	if elems <= 0 {
		return 0, false
	}
	bytesFloat := float64(elems) * quant.BytesPerElem * float64(users)
	if math.IsNaN(bytesFloat) || math.IsInf(bytesFloat, 0) || bytesFloat > float64(math.MaxInt64-1) {
		return math.MaxInt64, true
	}
	return int64(math.Ceil(bytesFloat)), true
}

func computeRuntime(baseBytes, fixedSessionBytes int64, shape kvbudget.Shape, quant kvbudget.Quant, ctx, users int) (kvBytes, runtimeBytes int64, ok bool) {
	kvBytes, ok = kvBytesForUsers(shape, quant, ctx, users)
	if !ok {
		return 0, 0, false
	}
	sessionBytes := mulSat(int64(users), fixedSessionBytes)
	return kvBytes, addSat(addSat(baseBytes, sessionBytes), kvBytes), true
}

func oneOf(v string, values ...string) bool {
	for _, x := range values {
		if v == x {
			return true
		}
	}
	return false
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
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

func mulSat(a, b int64) int64 {
	if a <= 0 || b <= 0 {
		return 0
	}
	if a > math.MaxInt64/b {
		return math.MaxInt64
	}
	return a * b
}
