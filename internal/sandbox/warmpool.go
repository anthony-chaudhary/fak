// Adapted from kubernetes-sigs/agent-sandbox
// extensions/controllers/sandboxwarmpool_controller.go at
// 890de53a4dd18e35313173e6dc34d9559b1a9733 (Apache-2.0). This is a native
// reimplementation of the SandboxWarmPool reconcile semantics (template-hash
// staleness, rate-limited refill, surplus deletion), not a copy.

package sandbox

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

// ProvisionFunc creates one sandbox instance for spec.
type ProvisionFunc func(ctx context.Context, spec Spec) (Instance, error)

// RegistryProvisioner provisions through the provider that r resolves for spec.Tier.
func RegistryProvisioner(r *Registry) ProvisionFunc {
	return func(ctx context.Context, spec Spec) (Instance, error) {
		p, err := r.ResolveTier(spec.Tier)
		if err != nil {
			return nil, err
		}
		return p.Create(ctx, spec)
	}
}

// TemplateHash returns a stable digest of spec used to detect stale warm instances.
func TemplateHash(spec Spec) (string, error) {
	raw, err := json.Marshal(spec)
	if err != nil {
		return "", fmt.Errorf("hash sandbox template: %w", err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:8]), nil
}

// WarmPoolConfig configures a WarmPool.
type WarmPoolConfig struct {
	Template  Spec
	Replicas  int
	Provision ProvisionFunc
	// MaxBatch caps creates and surplus deletes per Reconcile; zero means unlimited.
	MaxBatch int
	// MaxRefillRate caps creates per second through a token bucket holding at
	// most max(1, rate) tokens; zero leaves refill unshaped.
	MaxRefillRate float64
	Now           func() time.Time
}

// WarmPoolStatus is a point-in-time view of a WarmPool.
type WarmPoolStatus struct {
	TemplateHash string
	Desired      int
	Ready        int
	Stale        int
}

// WarmPoolReconcileResult reports what one Reconcile pass did.
type WarmPoolReconcileResult struct {
	Created      int
	Failed       int
	StaleRemoved int
	Deleted      int
	// Paced counts deficit the token bucket withheld this pass.
	Paced int
	// RequeueAfter is when the next refill token accrues; zero when nothing was paced.
	RequeueAfter time.Duration
}

type warmMember struct {
	inst Instance
	hash string
}

// WarmPool keeps pre-provisioned sandbox instances ready for Claim.
// Reconcile is caller-driven; the pool runs no background goroutines.
type WarmPool struct {
	provision ProvisionFunc
	maxBatch  int
	rate      float64
	now       func() time.Time

	// reconcileMu serializes Reconcile so provisioning can run without mu held.
	reconcileMu sync.Mutex

	mu       sync.Mutex
	template Spec
	hash     string
	desired  int
	// members stays in creation order: Claim takes the oldest, surplus deletion the newest.
	members  []warmMember
	tokens   float64
	tokensAt time.Time
	bucketOK bool
	closed   bool
}

// NewWarmPool validates cfg and returns an empty pool; call Reconcile to fill it.
func NewWarmPool(cfg WarmPoolConfig) (*WarmPool, error) {
	if cfg.Provision == nil {
		return nil, errors.New("warm pool: provision func is required")
	}
	if cfg.Replicas < 0 || cfg.MaxBatch < 0 {
		return nil, fmt.Errorf("warm pool: replicas (%d) and max batch (%d) must be non-negative", cfg.Replicas, cfg.MaxBatch)
	}
	if cfg.MaxRefillRate < 0 || math.IsNaN(cfg.MaxRefillRate) || math.IsInf(cfg.MaxRefillRate, 0) {
		return nil, fmt.Errorf("warm pool: max refill rate must be finite and non-negative: %v", cfg.MaxRefillRate)
	}
	if err := cfg.Template.Validate(); err != nil {
		return nil, err
	}
	hash, err := TemplateHash(cfg.Template)
	if err != nil {
		return nil, err
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	return &WarmPool{
		provision: cfg.Provision,
		maxBatch:  cfg.MaxBatch,
		rate:      cfg.MaxRefillRate,
		now:       now,
		template:  cfg.Template,
		hash:      hash,
		desired:   cfg.Replicas,
	}, nil
}

// SetReplicas changes the desired number of warm instances.
func (p *WarmPool) SetReplicas(n int) error {
	if n < 0 {
		return fmt.Errorf("warm pool: replicas must be non-negative: %d", n)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return NewSandboxError(ErrSandboxUnavailable, "warm pool is closed")
	}
	p.desired = n
	return nil
}

// SetTemplate replaces the template; members built from an older template become stale.
func (p *WarmPool) SetTemplate(spec Spec) error {
	if err := spec.Validate(); err != nil {
		return err
	}
	hash, err := TemplateHash(spec)
	if err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return NewSandboxError(ErrSandboxUnavailable, "warm pool is closed")
	}
	p.template = spec
	p.hash = hash
	return nil
}

// Status reports the pool's current counts.
func (p *WarmPool) Status() WarmPoolStatus {
	p.mu.Lock()
	defer p.mu.Unlock()
	st := WarmPoolStatus{TemplateHash: p.hash, Desired: p.desired}
	for _, m := range p.members {
		if m.hash == p.hash {
			st.Ready++
		} else {
			st.Stale++
		}
	}
	return st
}

// Claim removes and returns the oldest current-template instance. The caller
// owns the instance; the freed slot becomes deficit for the next Reconcile.
func (p *WarmPool) Claim() (Instance, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, NewSandboxError(ErrSandboxUnavailable, "warm pool is closed")
	}
	for i, m := range p.members {
		if m.hash != p.hash {
			continue
		}
		p.members = append(p.members[:i], p.members[i+1:]...)
		return m.inst, nil
	}
	return nil, NewSandboxError(ErrSandboxUnavailable, "warm pool has no ready instance")
}

// Reconcile removes stale members, refills any deficit within the batch and
// rate limits, and deletes surplus members, newest first.
func (p *WarmPool) Reconcile(ctx context.Context) (WarmPoolReconcileResult, error) {
	p.reconcileMu.Lock()
	defer p.reconcileMu.Unlock()

	var res WarmPoolReconcileResult
	var errs []error

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return res, NewSandboxError(ErrSandboxUnavailable, "warm pool is closed")
	}
	stale := p.takeStaleLocked()
	template, hash := p.template, p.hash
	want := p.desired - len(p.members)
	if p.maxBatch > 0 && want > p.maxBatch {
		want = p.maxBatch
	}
	toCreate := 0
	if want > 0 {
		toCreate, res.RequeueAfter = p.takeTokensLocked(want, p.now())
		res.Paced = want - toCreate
	}
	p.mu.Unlock()

	res.StaleRemoved = len(stale)
	errs = append(errs, closeMembers(stale)...)

	// Failed creates keep their tokens: the rate bounds provisioning attempts, not successes.
	for i := 0; i < toCreate; i++ {
		inst, err := p.provision(ctx, template)
		if err != nil {
			res.Failed++
			errs = append(errs, err)
			if inst != nil {
				errs = append(errs, inst.Close())
			}
			continue
		}
		if inst == nil {
			res.Failed++
			errs = append(errs, errors.New("warm pool: provision returned no instance"))
			continue
		}
		if !p.admit(inst, hash) {
			errs = append(errs, inst.Close())
			continue
		}
		res.Created++
	}

	surplus := p.takeSurplus()
	res.Deleted = len(surplus)
	errs = append(errs, closeMembers(surplus)...)

	return res, errors.Join(errs...)
}

// Close closes every pooled instance and refuses further Claim and Reconcile.
func (p *WarmPool) Close() error {
	p.reconcileMu.Lock()
	defer p.reconcileMu.Unlock()
	p.mu.Lock()
	members := p.members
	p.members = nil
	p.closed = true
	p.mu.Unlock()
	return errors.Join(closeMembers(members)...)
}

func (p *WarmPool) takeStaleLocked() []warmMember {
	var stale []warmMember
	kept := p.members[:0]
	for _, m := range p.members {
		if m.hash == p.hash {
			kept = append(kept, m)
		} else {
			stale = append(stale, m)
		}
	}
	p.members = kept
	return stale
}

// admit adds inst unless the template changed while it was being provisioned.
func (p *WarmPool) admit(inst Instance, hash string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed || hash != p.hash {
		return false
	}
	p.members = append(p.members, warmMember{inst: inst, hash: hash})
	return true
}

func (p *WarmPool) takeSurplus() []warmMember {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := len(p.members) - p.desired
	if n <= 0 {
		return nil
	}
	if p.maxBatch > 0 && n > p.maxBatch {
		n = p.maxBatch
	}
	cut := len(p.members) - n
	surplus := append([]warmMember(nil), p.members[cut:]...)
	p.members = p.members[:cut]
	return surplus
}

// takeTokensLocked grants up to want creates and, when short, the wait until
// the next whole token. A first call starts with a full bucket.
func (p *WarmPool) takeTokensLocked(want int, now time.Time) (int, time.Duration) {
	if p.rate <= 0 {
		return want, 0
	}
	capacity := math.Max(1, p.rate)
	if !p.bucketOK {
		p.tokens, p.tokensAt, p.bucketOK = capacity, now, true
	} else if elapsed := now.Sub(p.tokensAt); elapsed > 0 {
		p.tokens = math.Min(capacity, p.tokens+p.rate*elapsed.Seconds())
		p.tokensAt = now
	}
	granted := want
	if p.tokens < float64(want) {
		granted = int(p.tokens)
	}
	p.tokens -= float64(granted)
	if granted >= want {
		return granted, 0
	}
	wait := time.Duration(math.Ceil((1 - p.tokens) / p.rate * float64(time.Second)))
	return granted, wait
}

func closeMembers(ms []warmMember) []error {
	var errs []error
	for _, m := range ms {
		if err := m.inst.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errs
}
