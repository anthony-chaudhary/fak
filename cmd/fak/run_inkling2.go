package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/inklingmetal"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// The looped forward owns independent KV/conv slots for each layer visit.
func openInkling2Run(path, name, backend string, metalFlag bool, cfg nativeControlConfig) (agent.Planner, func(), error) {
	arch, err := serveInkling2Arch(path)
	if err != nil || arch == "" {
		return nil, nil, nil // Other loaders also admit directories and other architectures.
	}
	shards := serveInkling2ShardSet(path)
	f, err := ggufload.Open(shards[0]) // only the first split shard carries the tokenizer
	if err != nil {
		return nil, nil, err
	}
	be, metal, err := inkling2RunBackend(backend, metalFlag)
	if err != nil {
		return nil, nil, err
	}
	if err = applyNativeControls(be, cfg); err != nil {
		return nil, nil, err
	}
	tok, eos, err := runInkling2Tokenizer(f)
	if err != nil {
		return nil, nil, err
	}
	if metal && len(shards) > 1 {
		return nil, nil, fmt.Errorf("Inkling 2: the Metal path binds a single-file GGUF; %s is split into %d shards", path, len(shards))
	}
	var m *ggufload.Inkling2LoopModel
	var target *inklingmetal.Model
	var release func()
	residentCtx := 0
	backendName := "cpu"
	if metal {
		backendName = "metal"
		kind, e := inkling2MetalSelect(os.Getenv(inkling2MetalTargetEnv), arch == serveInkling2LoopArch, inklingmetal.Available(), inkling2MetalResidentAvailable)
		if e != nil {
			return nil, nil, e
		}
		var weights localadmission.MemoryPlan
		switch kind {
		case inkling2MetalVendored:
			// Match the launcher's default single-session context budget, including
			// the direct target's packed images, verification scratch and KV extents.
			planned, e := inklingmetal.Estimate(path, gateway.DefaultAdmissionPolicy().TokenBudget)
			if e != nil {
				return nil, nil, e
			}
			if planned.Bytes() > math.MaxInt64 {
				return nil, nil, fmt.Errorf("Inkling 2: Metal plan exceeds address space")
			}
			weights.SteadyBytes = int64(planned.Bytes())
			weights.StartupPeakBytes, _ = metalServeStartupPeakBytes(serveLoadArmResidentQ4K, weights.SteadyBytes, 0, false)
		case inkling2MetalResident:
			residentCtx = inkling2ResidentCtx(os.LookupEnv(inkling2ResidentCtxEnv))
			weights, err = inkling2RunResidentMemory(path, f, arch, residentCtx)
		default:
			weights, err = inkling2RunMetalMemory(path, f)
		}
		if err != nil {
			return nil, nil, err
		}
		var loadErr error
		release, err = admitLocalMetalModel(true, path, gpulease.Options{}, metalAdmissionSpec{
			Weights: &weights,
			Load: func() bool {
				switch kind {
				case inkling2MetalVendored:
					target, loadErr = inklingmetal.Open(path)
				case inkling2MetalResident:
					m, be, loadErr = openInkling2MetalResident(path)
				default:
					m, loadErr = ggufload.OpenInkling2LoopMetal(path)
				}
				return loadErr == nil
			},
			Teardown: func() error {
				if target != nil {
					return target.Close()
				}
				return m.Close()
			},
		})
		if err == nil {
			err = loadErr
		}
	} else {
		// CUDA binds under the free-VRAM budget and spills cold experts to host;
		// with no device backend the host forward runs over the mapped shards.
		var l *serveInkling2Loaded
		if l, err = loadServeInkling2(path, be, os.Stderr); err == nil {
			m = l.Model
			release = func() { _ = l.Close() }
			if be != nil {
				backendName = be.Name()
			}
		}
	}
	if err != nil {
		return nil, nil, err
	}
	if target != nil {
		fmt.Fprintf(os.Stderr, "fak run: inkling2_loop backend=metal target=%s layers=24 passes=3\n", inkling2MetalTargetVendoredLabel)
		return agent.NewInkling2MetalRunPlanner(name, target, tok, eos), release, nil
	}
	if residentCtx > 0 {
		fmt.Fprintf(os.Stderr, "fak run: %s backend=metal target=%s layers=%d passes=%d ctx=%d\n", arch, inkling2MetalTargetResidentLabel, m.HP.Layers, m.HP.Passes, residentCtx)
		return inkling2ResidentPlanner{agent.NewInkling2DeviceRunPlanner(name, m, be, tok, eos), residentCtx}, release, nil
	}
	if metal {
		backendName += " target=" + inkling2MetalTargetNativeLabel
	}
	fmt.Fprintf(os.Stderr, "fak run: %s backend=%s layers=%d passes=%d\n", arch, backendName, m.HP.Layers, m.HP.Passes)
	if be != nil {
		return agent.NewInkling2DeviceRunPlanner(name, m, be, tok, eos), release, nil
	}
	return agent.NewInkling2RunPlanner(name, m, tok, eos), release, nil
}

const (
	inkling2MetalTargetEnv           = "FAK_INKLING_METAL_TARGET"
	inkling2MetalTargetVendoredLabel = "vendored-splash"
	inkling2MetalTargetNativeLabel   = "fak-metalgemm"
	inkling2MetalTargetResidentLabel = "fak-metal-resident"
)

type inkling2MetalPath int

const (
	inkling2MetalNative inkling2MetalPath = iota
	inkling2MetalVendored
	inkling2MetalResident
)

// inkling2MetalSelect picks between the vendored direct target, fak's
// host-driven packed-kernel forward and fak's device-resident forward. Only an
// unset selector may fall back: an explicit request that cannot be honoured is
// refused so a measurement never reports one path under another's name.
func inkling2MetalSelect(selector string, loopArch, vendoredAvailable, residentAvailable bool) (inkling2MetalPath, error) {
	switch strings.ToLower(strings.TrimSpace(selector)) {
	case "":
		if loopArch && vendoredAvailable {
			return inkling2MetalVendored, nil
		}
		return inkling2MetalNative, nil
	case "vendored":
		if !loopArch {
			return 0, fmt.Errorf("Inkling 2: %s=vendored needs the %s architecture", inkling2MetalTargetEnv, serveInkling2LoopArch)
		}
		if !vendoredAvailable {
			return 0, fmt.Errorf("Inkling 2: %s=vendored but the vendored Metal target is not available in this build or on this device", inkling2MetalTargetEnv)
		}
		return inkling2MetalVendored, nil
	case "native":
		return inkling2MetalNative, nil
	case "resident":
		if !residentAvailable {
			return 0, fmt.Errorf("Inkling 2: %s=resident but the device-resident Metal forward is not available in this build", inkling2MetalTargetEnv)
		}
		return inkling2MetalResident, nil
	}
	return 0, fmt.Errorf("Inkling 2: unknown %s=%q (want vendored, native or resident)", inkling2MetalTargetEnv, selector)
}

// These mirror the resident driver's unexported sizing in internal/ggufload
// (inkling2ResidentCtxFrom, inkling2ArenaFloats); admission has to price the
// session before the driver exists.
const (
	inkling2ResidentCtxEnv      = "FAK_INKLING2_CTX"
	inkling2ResidentCtxDefault  = 4096
	inkling2ResidentCtxMin      = 512
	inkling2ResidentCtxMax      = 23552
	inkling2ResidentArenaFloats = 16 << 20
	// Metal rounds a shared buffer up to whole pages.
	inkling2ResidentBufferAlign = 16 << 10
)

func inkling2ResidentCtx(v string, set bool) int {
	if !set {
		return inkling2ResidentCtxDefault
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil || n < inkling2ResidentCtxMin {
		return inkling2ResidentCtxDefault
	}
	return min(n, inkling2ResidentCtxMax)
}

// inkling2ResidentShape is the part of the hyperparameters that sizes a
// resident session's device buffers.
type inkling2ResidentShape struct {
	Hidden, Heads, KVHeads, HeadDim int
	SlidingWindow, ConvKernel       int
	CoreStart, CoreEnd, Passes      int
	Local                           []bool
}

func inkling2ResidentShapeOf(f *ggufload.File, arch string) (inkling2ResidentShape, error) {
	var s inkling2ResidentShape
	var err error
	p := arch + "."
	u := func(key string, dst *int) {
		v, ok := f.Uint64(p + key)
		if !ok && err == nil {
			err = fmt.Errorf("Inkling 2: %s lacks %s%s", arch, p, key)
		}
		*dst = int(v)
	}
	// A head count is a scalar or a per-layer array; the binder requires the array uniform.
	heads := func(key string, dst *int) {
		if v, ok := f.IntArray(p + key); ok && len(v) > 0 {
			*dst = v[0]
			return
		}
		u(key, dst)
	}
	u("embedding_length", &s.Hidden)
	u("attention.key_length", &s.HeadDim)
	u("attention.sliding_window", &s.SlidingWindow)
	u("shortconv.l_cache", &s.ConvKernel)
	heads("attention.head_count", &s.Heads)
	heads("attention.head_count_kv", &s.KVHeads)
	s.Passes = 1
	if arch == serveInkling2LoopArch {
		u("loop.start", &s.CoreStart)
		u("loop.end", &s.CoreEnd)
		u("loop.count", &s.Passes)
	}
	s.Local, _ = f.BoolArray(p + "attention.sliding_window_pattern")
	if err == nil && (len(s.Local) == 0 || s.CoreStart < 0 || s.CoreEnd < s.CoreStart || s.CoreEnd > len(s.Local) || s.Passes < 1) {
		err = fmt.Errorf("Inkling 2: %s has an invalid loop core [%d,%d) x%d over %d layers", arch, s.CoreStart, s.CoreEnd, s.Passes, len(s.Local))
	}
	return s, err
}

// inkling2ResidentSessionBytes replays the allocations of the resident
// driver's session constructor for one unbatched session: K and V caches and a
// convolution history per layer visit, the attention score row, the token
// input, the per-token arena and the argmax index.
func inkling2ResidentSessionBytes(s inkling2ResidentShape, ctx int) int64 {
	var total int64
	alloc := func(floats int) {
		b := 4 * int64(floats)
		total += (b + inkling2ResidentBufferAlign - 1) / inkling2ResidentBufferAlign * inkling2ResidentBufferAlign
	}
	layers, core := len(s.Local), s.CoreEnd-s.CoreStart
	kvDim := s.KVHeads * s.HeadDim
	channels := 2*kvDim + 2*s.Hidden
	maxCap := s.SlidingWindow
	for slot := 0; slot < layers+(s.Passes-1)*core; slot++ {
		layer := slot
		if slot >= layers {
			// Slots past the first pass belong to the revisited core.
			layer = s.CoreStart + (slot-layers)%core
		}
		cap := ctx
		if s.Local[layer] {
			cap = s.SlidingWindow
		}
		maxCap = max(maxCap, cap)
		alloc(cap * kvDim)
		alloc(cap * kvDim)
		alloc(channels * (s.ConvKernel - 1))
	}
	alloc(2 * s.Heads * maxCap)
	alloc(s.Hidden + 4)
	alloc(inkling2ResidentArenaFloats)
	alloc(4)
	return total
}

// inkling2RunResidentMemory is the weight plan plus what the resident forward
// holds on the device: the session buffers and an F32 copy of each loop
// controller projection, which that path uploads instead of reading in place.
func inkling2RunResidentMemory(path string, f *ggufload.File, arch string, ctx int) (localadmission.MemoryPlan, error) {
	plan, err := inkling2RunMetalMemory(path, f)
	if err != nil {
		return plan, err
	}
	shape, err := inkling2ResidentShapeOf(f, arch)
	if err != nil {
		return localadmission.MemoryPlan{}, err
	}
	extra := inkling2ResidentSessionBytes(shape, ctx)
	for _, t := range f.Tensors {
		if !inkling2RunControllerWeight(t.Name) {
			continue
		}
		// inkling2RunMetalMemory has already bounded every dimension product.
		var bytes int64 = 4
		for _, n := range t.Dims {
			bytes *= int64(n)
		}
		extra += bytes
	}
	if extra > math.MaxInt64-plan.SteadyBytes {
		return localadmission.MemoryPlan{}, fmt.Errorf("Inkling 2: resident session plan overflow")
	}
	plan.SteadyBytes += extra
	plan.StartupPeakBytes, _ = metalServeStartupPeakBytes(serveLoadArmResidentQ4K, plan.SteadyBytes, 0, false)
	return plan, nil
}

// inkling2ResidentPlanner names the context limit when the driver runs out of it.
type inkling2ResidentPlanner struct {
	agent.Planner
	ctx int
}

func (p inkling2ResidentPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	c, err := p.Planner.Complete(ctx, messages, tools, opts...)
	return c, inkling2ResidentCtxError(err, p.ctx)
}

// The driver reports an exhausted cache only as text.
func inkling2ResidentCtxError(err error, ctx int) error {
	if err == nil || !strings.Contains(err.Error(), "global cache") {
		return err
	}
	return fmt.Errorf("Inkling 2: prompt plus output exceeds the %d-token context of the %s Metal path (%s; default %d, max %d): %w",
		ctx, inkling2MetalTargetResidentLabel, inkling2ResidentCtxEnv, inkling2ResidentCtxDefault, inkling2ResidentCtxMax, err)
}

// runInkling2Tokenizer is a seam so tests can bind a toy vocab to a synthetic GGUF.
var runInkling2Tokenizer = func(f *ggufload.File) (*tokenizer.Tokenizer, int, error) {
	table, ok := f.GGMLTokenizer()
	if !ok || table.EOS < 0 {
		return nil, 0, fmt.Errorf("Inkling 2: checkpoint needs an embedded BPE tokenizer and EOS")
	}
	tok, err := tokenizer.FromGGML(table.Tokens, table.Merges, table.TokenTypes, table.Pre)
	if err != nil {
		return nil, 0, err
	}
	for _, marker := range []string{"<|message_system|>", "<|message_user|>", "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>", "<|content_model_end_sampling|>"} {
		if _, ok := tok.SpecialID(marker); !ok {
			return nil, 0, fmt.Errorf("Inkling 2: checkpoint tokenizer lacks %s", marker)
		}
	}
	return tok, table.EOS, nil
}

// Inkling uses packed MSL kernels directly, including on devices where the
// general MPS compute backend does not register. CUDA takes the budgeted
// device bind; every other device backend is refused.
func inkling2RunBackend(name string, metalFlag bool) (compute.Backend, bool, error) {
	selector := strings.TrimSpace(name)
	if selector == "" {
		selector = strings.TrimSpace(os.Getenv("FAK_BACKEND"))
	}
	if strings.EqualFold(selector, "metal") {
		if !metalgemm.Available() {
			return nil, false, metalgemm.ErrMetalUnavailable
		}
		return nil, true, nil
	}
	be, err := resolveServeChatBackend(name)
	if err != nil {
		return nil, false, err
	}
	if be != nil {
		if be.Name() != "cuda" {
			return nil, false, fmt.Errorf("Inkling 2 run: unsupported backend %q", be.Name())
		}
		return be, false, nil
	}
	metal, err := resolveRunMetal(metalFlag, os.Getenv("FAK_METAL") != "", name)
	return be, metal, err
}

// Weight views map packed projections directly. Count an additional F32 copy
// for decoded vectors, convolution taps, controller queries and unknown tensors.
func inkling2RunMetalMemory(path string, f *ggufload.File) (localadmission.MemoryPlan, error) {
	st, err := os.Stat(path)
	if err != nil {
		return localadmission.MemoryPlan{}, err
	}
	steady := st.Size()
	for _, t := range f.Tensors {
		if inkling2RunMappedWeight(t.Name) {
			continue
		}
		var bytes int64 = 4
		for _, n := range t.Dims {
			if n == 0 || n > uint64(math.MaxInt64/bytes) {
				return localadmission.MemoryPlan{}, fmt.Errorf("Inkling 2: invalid dimensions for %s", t.Name)
			}
			bytes *= int64(n)
		}
		if bytes > math.MaxInt64-steady {
			return localadmission.MemoryPlan{}, fmt.Errorf("Inkling 2: weight plan overflow at %s", t.Name)
		}
		steady += bytes
	}
	peak, _ := metalServeStartupPeakBytes(serveLoadArmResidentQ4K, steady, 0, false)
	return localadmission.MemoryPlan{SteadyBytes: steady, StartupPeakBytes: peak}, nil
}

func inkling2RunMappedWeight(name string) bool {
	if name == "token_embd.weight" || name == "output.weight" {
		return true
	}
	block, ok := strings.CutPrefix(name, "blk.")
	if !ok {
		return false
	}
	_, field, ok := strings.Cut(block, ".")
	if !ok {
		return false
	}
	switch field {
	case "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_gate.weight", "attn_rel.weight", "attn_sink_proj.weight",
		"attn_k_conv_proj.weight", "attn_v_conv_proj.weight", "ffn_gate_inp.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight":
		return true
	}
	return inkling2RunControllerField(field)
}

func inkling2RunControllerWeight(name string) bool {
	block, _ := strings.CutPrefix(name, "blk.")
	_, field, _ := strings.Cut(block, ".")
	return inkling2RunControllerField(field)
}

func inkling2RunControllerField(field string) bool {
	controller, ok := strings.CutPrefix(field, "loop_")
	if !ok {
		return false
	}
	channel, projection, ok := strings.Cut(controller, "_ctrl_")
	if !ok {
		return false
	}
	switch channel {
	case "q", "k", "v", "ffn":
		return projection == "down.weight" || projection == "gain.weight" || projection == "forget.weight"
	}
	return false
}
