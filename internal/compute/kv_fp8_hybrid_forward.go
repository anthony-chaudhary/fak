package compute

import (
	"errors"
	"math"
	"sync/atomic"
)

var errKVFP8Hybrid = errors.New("compute: FP8 hybrid forward refused")

// This is a request snapshot, not a policy or advertised backend capability.
// Future model binding must derive flags from the actual selected configuration.
type kvFP8HybridContract struct {
	first, tokens                                         int
	linear                                                []bool
	theta                                                 []uint64
	qkNorm                                                bool
	scaledRoPE, longRoPE, fusedQK, sparseAttention, graph bool
}

type kvFP8HybridState struct {
	backend                 *kvFP8BackendIdentity
	device, mainLayer, kind int // kind0 convolution, kind1 recurrent
	storage                 kvFP8Allocation
	invalid                 *uint32
	root                    any
}

type kvFP8HybridFlight struct {
	owner    *kvFP8Owner
	states   []kvFP8HybridState
	retained bool
	next     *kvFP8HybridFlight
}

// Accessed only under the existing ForwardExecutionOwner/cudaMu domain.
var kvFP8HybridRetained *kvFP8HybridFlight

func (f *kvFP8HybridFlight) poison() {
	f.owner.poisonNumerical()
	for _, s := range f.states {
		atomic.StoreUint32(s.invalid, 1)
	}
}
func (f *kvFP8HybridFlight) healthy() bool {
	for _, s := range f.states {
		if atomic.LoadUint32(s.invalid) != 0 {
			return false
		}
	}
	return true
}
func (f *kvFP8HybridFlight) retain() {
	if !f.retained {
		f.retained = true
		f.next = kvFP8HybridRetained
		kvFP8HybridRetained = f
	}
}
func kvFP8HybridDisjoint(a, b kvFP8Allocation) bool {
	x, y := uintptr(a.ptr), uintptr(b.ptr)
	return x+uintptr(a.bytes) <= y || y+uintptr(b.bytes) <= x
}

func kvFP8HybridPreflight(o *kvFP8Owner, c kvFP8HybridContract, states []kvFP8HybridState) error {
	if err := o.usable(); err != nil {
		return err
	}
	if o.layers > int(^uint(0)>>1)/6 || len(c.linear) != 4*o.layers ||
		len(c.theta) != len(c.linear) || len(states) != 6*o.layers || c.first != o.image.positions ||
		c.tokens <= 0 || c.tokens > 65535 || c.first < 0 || c.tokens > math.MaxInt32-c.first ||
		!c.qkNorm || c.scaledRoPE || c.longRoPE || c.fusedQK || c.sparseAttention || c.graph {
		return errKVFP8Hybrid
	}
	if _, _, err := o.planeBytes(c.first + c.tokens); err != nil {
		return err
	}
	for main, linear := range c.linear {
		theta := math.Float64frombits(c.theta[main])
		if linear != ((main+1)%4 != 0) || !(theta > 0) || math.IsInf(theta, 0) {
			return errKVFP8Hybrid
		}
		if !linear {
			bound := o.rotary[main/4]
			if bound.producer != kvFP8QwenPartialPanelV1 || bound.mainLayer != main ||
				bound.headDim != 256 || bound.rotaryDim != 64 || bound.thetaBits != c.theta[main] ||
				o.heads != 4 || o.headDim != 256 {
				return errKVFP8Hybrid
			}
		}
	}
	for i, s := range states {
		wantMain := (i/6)*4 + (i%6)/2
		if s.backend != o.backend || s.device != o.device || s.mainLayer != wantMain || s.kind != i%2 ||
			s.root == nil || s.invalid == nil || atomic.LoadUint32(s.invalid) != 0 ||
			s.storage.ptr == nil || s.storage.bytes <= 0 ||
			uintptr(s.storage.bytes) > ^uintptr(0)-uintptr(s.storage.ptr) {
			return errKVFP8Hybrid
		}
		for _, prior := range states[:i] {
			if prior.invalid == s.invalid || !kvFP8HybridDisjoint(prior.storage, s.storage) {
				return errKVFP8Hybrid
			}
		}
		for _, planes := range o.image.layers {
			for _, a := range planes {
				if a.bytes > 0 && !kvFP8HybridDisjoint(a, s.storage) {
					return errKVFP8Hybrid
				}
			}
		}
	}
	return nil
}

// This private guard is NOT a model implementation. Each callback must be a
// separately reviewed locked main-layer body; all producer/consumer owners and
// native ingress must already be bound. No callback may publish results, free
// uncertain owners, acquire cudaMu, or invoke a lock-taking public operation.
// GDN callbacks may update only their exact two bound state owners. Attention
// callbacks must consume exactly one compact layer using the Qwen ingress leaf.
// Each layer and the final tail is checked before the FP8 image can publish.
//
// admit is query-only and must include Qwen timing/terminal refusal BEFORE the
// outer append allocates anything. Refusal here leaves a healthy owner/states
// unchanged. Once admitted, any abort invalidates BOTH FP8 and all bound GDN
// states; known numerical rejection remains owner-local, not device loss.
// No graph owner, prefix reuse or public result is made admissible by this guard.
func kvFP8HybridForwardEager(o *kvFP8Owner, c kvFP8HybridContract, states []kvFP8HybridState,
	peakBudget int64, ops kvFP8AppendOps, admit func() error,
	steps []func(*kvFP8AppendTransaction, int) error, tail func(*kvFP8AppendTransaction) error) (err error) {
	if !ops.complete() || admit == nil || tail == nil || len(steps) != len(c.linear) {
		return errKVFP8Hybrid
	}
	for _, step := range steps {
		if step == nil {
			return errKVFP8Hybrid
		}
	}
	if err := kvFP8HybridPreflight(o, c, states); err != nil {
		return err
	}
	flight := &kvFP8HybridFlight{owner: o, states: append([]kvFP8HybridState(nil), states...)}
	if err := admit(); err != nil {
		return err
	}
	// The admission callback is query-only; reject illicit mutation before effects.
	if err := kvFP8HybridPreflight(o, c, flight.states); err != nil {
		return err
	}
	normalReturn := false
	defer func() {
		if !normalReturn || err != nil {
			flight.poison()
			if !normalReturn || o.uncertain || o.backend.stopped.Load() {
				o.retainUncertain()
				flight.retain()
				if !normalReturn {
					ops.stop(kvFP8AppendFailure)
				}
			}
		}
	}()
	err = kvFP8AppendEager(o, c.first, c.tokens, peakBudget, ops, func(tx *kvFP8AppendTransaction) error {
		for main, step := range steps {
			if !flight.healthy() {
				return tx.uncertain(errKVFP8Hybrid, 0)
			}
			if e := step(tx, main); e != nil {
				if tx.failure != nil {
					return tx.failure
				}
				return tx.uncertain(e, 0)
			}
			if !flight.healthy() || tx.busy || tx.ready || tx.next != (main+1)/4 {
				return tx.uncertain(errKVFP8Hybrid, 0)
			}
			if e := tx.checked(); e != nil {
				return e
			}
		}
		if e := tail(tx); e != nil {
			if tx.failure != nil {
				return tx.failure
			}
			return tx.uncertain(e, 0)
		}
		if !flight.healthy() || tx.busy || tx.ready || tx.next != o.layers {
			return tx.uncertain(errKVFP8Hybrid, 0)
		}
		return tx.checked()
	})
	normalReturn = true
	return err
}
