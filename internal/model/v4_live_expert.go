package model

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var ErrV4LiveExpert = errors.New("model: DeepSeek V4 live expert forward failed")

// ErrV4ExpertOwnerClosed means the (model, backend) V4 runtime owner has been torn down by
// weight close, so no session may attach a new one to it.
var ErrV4ExpertOwnerClosed = errors.New("model: V4 expert runtime owner is closed")

const (
	defaultV4ExpertRingBytes int64 = 480 << 30
	defaultV4ExpertOpenFiles       = 8
)

type v4LiveExpertRuntime interface {
	Close() error
	forward(layer, tokenID int, x, logits, correctionBias []float32) ([]float32, error)
	Stats() v4ExpertRuntimeStats
}

// v4LiveExpertRuntimeBuilder is the construction seam for the shared V4 runtime. The shipped
// implementation builds the concrete paged-ring runtime; tests inject a counting fake so the
// lifetime contract (one build across sessions, one free at teardown) is measurable without
// touching real weights.
var v4LiveExpertRuntimeBuilder = func(dir string, cfg Config, be compute.Backend) (v4LiveExpertRuntime, error) {
	return newV4LiveExpert(dir, cfg, be)
}

type v4LiveExpert struct {
	runtime *v4ExpertRuntime
}

// v4ExpertOwner is the single bounded V4 routed-expert runtime for ONE (Model, Backend) pair.
// It is the missing mechanism under perf(model) fak#13479: before it, each Session built its own
// runtime (newV4LiveExpert) and Session.Close freed the ring, so sequential requests paid the
// expert page-in again and concurrent requests could hold duplicate rings under separate budgets.
//
// Sessions keep their own logits/KV/router state; only the model-constant routed-expert bytes and
// their ring are shared. attach() refuses a backend whose identity differs from the owner's,
// because a device handle is only valid on the device that produced it.
//
// Lifetime: attach() refcounts; detach() releases the session without freeing. free() runs once,
// from Model.CloseWeights, after the last detach. The mutex guards clients/build/closed.
type v4ExpertOwner struct {
	mu      sync.Mutex
	m       *Model
	be      compute.Backend
	clients int
	build   v4LiveExpertRuntime
	closed  bool
}

func (o *v4ExpertOwner) attach(m *Model, be compute.Backend) (v4LiveExpertRuntime, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return nil, fmt.Errorf("%w: %v", ErrV4LiveExpert, ErrV4ExpertOwnerClosed)
	}
	if o.build == nil {
		if m == nil {
			return nil, fmt.Errorf("%w: nil model", ErrV4LiveExpert)
		}
		if m.sourceDir == "" {
			return nil, fmt.Errorf("%w: model was not loaded from an indexed safetensors directory", ErrV4LiveExpert)
		}
		// Fail fast on invalid ring/open caps before constructing anything, so a
		// misconfiguration is reported as a typed V4 error rather than a partial build.
		if _, _, err := v4RuntimeLimits(); err != nil {
			return nil, err
		}
		built, err := v4LiveExpertRuntimeBuilder(m.sourceDir, m.Cfg, be)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrV4LiveExpert, err)
		}
		o.build = built
	}
	o.clients++
	return o.build, nil
}

// detach releases one client WITHOUT freeing resident pages: the bytes belong to the (model,
// device) pair, so a conversation ending must leave them for the sessions still using them.
func (o *v4ExpertOwner) detach() {
	if o == nil {
		return
	}
	o.mu.Lock()
	if o.clients > 0 {
		o.clients--
	}
	o.mu.Unlock()
}

// free releases the ring exactly once, at model-weight teardown, after the last session has
// detached. It is a no-op on a second call and on an owner that never built a runtime.
func (o *v4ExpertOwner) free() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return nil
	}
	o.closed = true
	if o.build == nil {
		return nil
	}
	err := o.build.Close()
	o.build = nil
	return err
}

// v4ExpertOwnerFor returns the model's sole V4 runtime owner, building the registry lazily. The
// owner is NOT freed here; Model.CloseWeights calls each owner's free once.
func (m *Model) v4ExpertOwnerFor(be compute.Backend) *v4ExpertOwner {
	v4ExpertOwnerInitMu.Lock()
	defer v4ExpertOwnerInitMu.Unlock()
	if m.v4ExpertOwners == nil {
		m.v4ExpertOwners = map[compute.Backend]*v4ExpertOwner{}
	}
	o := m.v4ExpertOwners[be]
	if o == nil {
		o = &v4ExpertOwner{m: m, be: be}
		m.v4ExpertOwners[be] = o
	}
	return o
}

var v4ExpertOwnerInitMu sync.Mutex

// freeV4ExpertOwners frees every (model, backend) V4 runtime owner exactly once, at
// model-weight teardown. It is the single owner-side free the issue's acceptance gate names.
func (m *Model) freeV4ExpertOwners() {
	if m == nil {
		return
	}
	v4ExpertOwnerInitMu.Lock()
	owners := m.v4ExpertOwners
	m.v4ExpertOwners = nil
	v4ExpertOwnerInitMu.Unlock()
	for _, o := range owners {
		_ = o.free()
	}
}

func v4RuntimeLimits() (int64, int, error) {
	capBytes := defaultV4ExpertRingBytes
	maxOpen := defaultV4ExpertOpenFiles
	if raw := os.Getenv("FAK_V4_EXPERT_RING_BYTES"); raw != "" {
		v, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || v <= 0 {
			return 0, 0, fmt.Errorf("%w: invalid FAK_V4_EXPERT_RING_BYTES=%q", ErrV4LiveExpert, raw)
		}
		capBytes = v
	}
	if raw := os.Getenv("FAK_V4_EXPERT_OPEN_FILES"); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v <= 0 {
			return 0, 0, fmt.Errorf("%w: invalid FAK_V4_EXPERT_OPEN_FILES=%q", ErrV4LiveExpert, raw)
		}
		maxOpen = v
	}
	return capBytes, maxOpen, nil
}

func newV4LiveExpert(dir string, cfg Config, be compute.Backend) (*v4LiveExpert, error) {
	capBytes, maxOpen, err := v4RuntimeLimits()
	if err != nil {
		return nil, err
	}
	rt, err := newV4ExpertRuntime(dir, cfg, be, capBytes, maxOpen)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrV4LiveExpert, err)
	}
	return &v4LiveExpert{runtime: rt}, nil
}

func (v *v4LiveExpert) Close() error {
	if v == nil || v.runtime == nil {
		return nil
	}
	return v.runtime.Close()
}

func (v *v4LiveExpert) forward(layer, tokenID int, x, logits, correctionBias []float32) ([]float32, error) {
	if v == nil || v.runtime == nil {
		return nil, fmt.Errorf("%w: runtime is not initialized", ErrV4LiveExpert)
	}
	var (
		out []float32
		err error
	)
	if layer < 3 {
		out, err = v.runtime.forwardHash(layer, tokenID, logits, compute.NewF32(compute.Default(), []int{len(x)}, x))
	} else {
		out, err = v.runtime.forwardScored(layer, logits, correctionBias, compute.NewF32(compute.Default(), []int{len(x)}, x))
	}
	if err != nil {
		return nil, fmt.Errorf("%w: layer %d: %v", ErrV4LiveExpert, layer, err)
	}
	return out, nil
}

func (v *v4LiveExpert) Stats() v4ExpertRuntimeStats {
	if v == nil || v.runtime == nil {
		return v4ExpertRuntimeStats{}
	}
	return v.runtime.Stats()
}

func (s *Session) ensureV4LiveExpert() (v4LiveExpertRuntime, error) {
	if s == nil || s.M == nil || !s.M.Cfg.IsDeepSeekV4() {
		return nil, fmt.Errorf("%w: session architecture is not deepseek_v4", ErrV4LiveExpert)
	}
	if s.M.sourceDir == "" {
		return nil, fmt.Errorf("%w: model was not loaded from an indexed safetensors directory", ErrV4LiveExpert)
	}
	if s.Backend == nil {
		return nil, fmt.Errorf("%w: session has no compute backend", ErrV4LiveExpert)
	}
	if s.v4Expert != nil {
		return s.v4Expert, nil
	}
	// Attach to the (Model, Backend)-scoped owner rather than constructing a per-session
	// runtime (fak#13479). The bytes and ring are model-constant; only session state stays
	// per-session. The owner is freed once, by Model.CloseWeights.
	owner := s.M.v4ExpertOwnerFor(s.Backend)
	rt, err := owner.attach(s.M, s.Backend)
	if err != nil {
		return nil, err
	}
	s.v4Expert = rt
	s.v4ExpertOwner = owner
	return rt, nil
}

// applyV4ExpertHAL is the live HAL seam for the V4 routed FFN. It keeps the
// residual tensor resident while the bounded expert runtime operates on one
// normalized token and returns the selected-expert contribution.
func (s *Session) applyV4ExpertHAL(layer, tokenID int, residual, postAttnNorm compute.Tensor, eps float32) error {
	normalized := s.Backend.RMSNorm(residual, postAttnNorm, eps)
	out, err := s.v4ExpertForward(layer, tokenID, normalized)
	if err != nil {
		return err
	}
	s.Backend.AddInPlace(residual, out)
	s.Backend.Free(out)
	return nil
}

func (s *Session) v4ExpertForward(layer, tokenID int, normalized compute.Tensor) (compute.Tensor, error) {
	v, err := s.ensureV4LiveExpert()
	if err != nil {
		return compute.Tensor{}, err
	}
	x := s.Backend.Read(normalized)
	if len(x) != s.M.Cfg.HiddenSize {
		return compute.Tensor{}, fmt.Errorf("%w: normalized width=%d want %d", ErrV4LiveExpert, len(x), s.M.Cfg.HiddenSize)
	}
	prefix := layerName(layer, "mlp.")
	gate := s.matWeightHAL(prefix + "gate.weight")
	logitsTensor := s.Backend.MatMul(gate, normalized)
	logits := s.Backend.Read(logitsTensor)
	s.Backend.Free(logitsTensor)
	if len(logits) != s.M.Cfg.NumExperts {
		return compute.Tensor{}, fmt.Errorf("%w: router width=%d want %d", ErrV4LiveExpert, len(logits), s.M.Cfg.NumExperts)
	}
	var bias []float32
	if layer >= 3 {
		biasName := prefix + "gate.e_score_correction_bias"
		if _, ok := s.M.manifest[biasName]; !ok {
			return compute.Tensor{}, fmt.Errorf("%w: missing %s", ErrV4LiveExpert, biasName)
		}
		bias = append([]float32(nil), s.M.tensor(biasName)...)
	}
	out, err := v.forward(layer, tokenID, x, logits, bias)
	if err != nil {
		return compute.Tensor{}, err
	}
	if len(out) != s.M.Cfg.HiddenSize {
		return compute.Tensor{}, fmt.Errorf("%w: expert output width=%d want %d", ErrV4LiveExpert, len(out), s.M.Cfg.HiddenSize)
	}
	// The always-on shared expert runs on every token and its contribution is
	// added to the routed sum. Compute routed first, then shared, then combine
	// as routed+shared: float addition order is observable, so the order is
	// pinned here and the combine reuses the single pinned shared-add helper
	// (v41_router.go). Scope: the Flash profile (epic #12638) is the admitted
	// model whose shared expert must execute; the Pro profile keeps its prior
	// routed-only behavior, and a shared-less profile (NSharedExperts==0)
	// likewise stays routed-only.
	if s.M.Cfg.NSharedExperts > 0 && isDeepSeekV4FlashProfile(s.M.Cfg) {
		shared, err := s.v4SharedExpertForward(layer, x, float32(s.M.Cfg.SwigluLimit), s.M.Cfg.HiddenSize, s.M.Cfg.MoEIntermediateSize)
		if err != nil {
			return compute.Tensor{}, err
		}
		combined, err := v41SharedExpertAdd(out, shared, v41RouterConfig{SharedCount: s.M.Cfg.NSharedExperts})
		if err != nil {
			return compute.Tensor{}, fmt.Errorf("%w: shared add: %v", ErrV4SharedExpert, err)
		}
		out = combined
	}
	if len(out) != s.M.Cfg.HiddenSize {
		return compute.Tensor{}, fmt.Errorf("%w: combined output width=%d want %d", ErrV4LiveExpert, len(out), s.M.Cfg.HiddenSize)
	}
	return s.uploadHostF32([]int{s.M.Cfg.HiddenSize}, out, compute.MemoryActivation, "v4-expert-output"), nil
}
