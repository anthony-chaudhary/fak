package agent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	ErrDFlash2Identity    = errors.New("dflash2: invalid trusted identity")
	ErrDFlash2Busy        = errors.New("dflash2: serving domain is busy or reentrant")
	ErrDFlash2Quarantined = errors.New("dflash2: serving domain is quarantined")
	dflash2Domains        sync.Map // stable *sync.Mutex -> *dflash2Domain
	dflash2BackendDomains sync.Map // comparable backend identity -> *dflash2Domain
)

type dflash2Domain struct {
	owner    *sync.Mutex
	slot     chan struct{}
	mu       sync.Mutex
	failure  error
	retained []any
}

func dflash2Owner(be compute.Backend) (*sync.Mutex, error) {
	if be == nil || !reflect.TypeOf(be).Comparable() || (reflect.ValueOf(be).Kind() == reflect.Pointer && reflect.ValueOf(be).IsNil()) {
		return nil, errors.New("dflash2: comparable owned HAL backend required")
	}
	provider, ok := be.(compute.ForwardExecutionOwnerBackend)
	if !ok {
		return nil, errors.New("dflash2: stable execution owner required")
	}
	owner := provider.ForwardExecutionOwner()
	if owner == nil {
		return nil, errors.New("dflash2: nil execution owner")
	}
	return owner, nil
}

func registerDFlash2Domain(be compute.Backend) (*dflash2Domain, error) {
	owner, err := dflash2Owner(be)
	if err != nil {
		return nil, err
	}
	if prior, ok := dflash2BackendDomains.Load(be); ok {
		d := prior.(*dflash2Domain)
		if d.owner != owner {
			return nil, errors.New("dflash2: backend execution owner changed")
		}
		return d, nil
	}
	value, _ := dflash2Domains.LoadOrStore(owner, &dflash2Domain{owner: owner, slot: make(chan struct{}, 1)})
	d := value.(*dflash2Domain)
	prior, _ := dflash2BackendDomains.LoadOrStore(be, d)
	if prior.(*dflash2Domain) != d {
		return nil, errors.New("dflash2: backend domain changed during registration")
	}
	return d, nil
}

// Registration is startup-only. Every Complete using a registered recording
// domain, including an ordinary planner or wrapper alias, takes this separate
// request gate. The backend's own execution mutex is never borrowed here.
func registeredDFlash2Domain(be compute.Backend) (*dflash2Domain, error) {
	if be == nil {
		return nil, nil
	}
	if !reflect.TypeOf(be).Comparable() {
		if _, owned := be.(compute.ForwardExecutionOwnerBackend); owned {
			return nil, errors.New("dflash2: non-comparable execution-owner alias refused")
		}
		return nil, nil
	}
	if prior, ok := dflash2BackendDomains.Load(be); ok {
		d := prior.(*dflash2Domain)
		owner, err := dflash2Owner(be)
		if err != nil || owner != d.owner {
			return nil, errors.New("dflash2: registered backend owner changed")
		}
		return d, nil
	}
	provider, ok := be.(compute.ForwardExecutionOwnerBackend)
	if !ok {
		return nil, nil
	}
	owner := provider.ForwardExecutionOwner()
	if value, ok := dflash2Domains.Load(owner); ok {
		d := value.(*dflash2Domain)
		dflash2BackendDomains.Store(be, d)
		return d, nil
	}
	return nil, nil
}

func (d *dflash2Domain) acquire(ctx context.Context) (func(), error) {
	if ctx == nil {
		return nil, errors.New("dflash2: context required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case d.slot <- struct{}{}:
	default:
		return nil, ErrDFlash2Busy
	}
	release := func() { <-d.slot }
	d.mu.Lock()
	err := d.failure
	d.mu.Unlock()
	if err != nil {
		release()
		return nil, errors.Join(ErrDFlash2Quarantined, err)
	}
	if err := ctx.Err(); err != nil {
		release()
		return nil, err
	}
	return release, nil
}

func (d *dflash2Domain) quarantine(err error, owner any) {
	if err == nil {
		err = errors.New("dflash2: incomplete retirement")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.failure = errors.Join(d.failure, err)
	d.retained = append(d.retained, owner)
}

func (p *InKernelPlanner) acquireDFlash2Domain(ctx context.Context) (func(), error) {
	d, err := registeredDFlash2Domain(p.backend)
	if err != nil {
		return nil, err
	}
	if d == nil {
		return func() {}, nil
	}
	return d.acquire(ctx)
}

type dflash2Prepared interface {
	Tokens() []int
	Commit(context.Context, int) ([]int, error)
	Abort() error
}
type dflash2RequestOwner interface {
	Bootstrap(context.Context, []int) error
	Prepare(context.Context, int) (dflash2Prepared, error)
	Close() error
}
type dflash2NativeRequest struct{ *model.Qwen38DFlash2Request }

func (r dflash2NativeRequest) Prepare(ctx context.Context, n int) (dflash2Prepared, error) {
	return r.Qwen38DFlash2Request.Prepare(ctx, n)
}

// EnableDFlash2 is a startup-only binding to an authenticated target owner.
// It does not alter any MTP/ngram setting. Conflicting routes are refused.
func (p *InKernelPlanner) EnableDFlash2(target *DFlash2Target) error {
	if p == nil || target == nil {
		return errors.New("dflash2: planner and authenticated target required")
	}
	if p.speculativeEngine != nil || p.vulkanMTP || p.vulkanMTPDraftFactory != nil || p.MetalMTPCoordinator() != nil {
		return errors.New("dflash2: another speculative route is already configured")
	}
	if err := target.ValidateBinding(p.m, p.tok, p.backend, p.q4k, p.metal); err != nil {
		return err
	}
	p.dflash2 = target
	return nil
}

func (p *InKernelPlanner) validateDFlash2Request(sp SampleParams, temp, frequency, presence float64) error {
	if p.dflash2 == nil {
		return nil
	}
	if sp.Seed != nil {
		return ErrSeedUnsupported
	}
	if err := validateDFlash2Stops(sp.Stop); err != nil {
		return err
	}
	if err := p.dflash2.ValidateBinding(p.m, p.tok, p.backend, p.q4k, p.metal); err != nil {
		return err
	}
	if (sp.TopP != nil && *sp.TopP != 0) || (sp.TopK != nil && *sp.TopK > 0) || temp != 0 || frequency != 0 || presence != 0 || len(sp.LogitBias) != 0 ||
		len(sp.ResponseFormat) != 0 || len(sp.ToolChoice) != 0 || sp.MinP != nil ||
		len(sp.ChatTemplateKwargs) != 0 || sp.SpeculativeNMax != nil || sp.RawPrompt ||
		sp.NativeInferenceReceipt || sp.NativeDecodeTokenIDs ||
		(sp.DecodeTrace && sp.DecodeTokenObserver == nil) {
		return errors.New("dflash2: selected diagnostic route requires unmodified greedy decoding without unsupported receipts or constraints")
	}
	if inKernelPromptRendererID(p.m.Cfg, sp) != p.dflash2.spec.RendererID {
		return errors.New("dflash2: request renderer identity mismatch")
	}
	return nil
}

func (p *InKernelPlanner) generateDFlash2(ctx context.Context, ids []int, maxNew int,
	preview func([]int) (dflash2StopPreview, error), publish func(dflash2StopPreview) error) (res inKernelGenerateResult, err error) {
	if p.dflash2 == nil || maxNew < 1 || len(ids) == 0 || preview == nil || publish == nil {
		return res, errors.New("dflash2: ready target, prompt and publication callbacks required")
	}
	if err = p.checkExecutionDeadline(ctx, len(ids), 0, maxNew); err != nil {
		return res, err
	}
	lease, err := p.dflash2.newRequest(ctx)
	if err != nil {
		return res, err
	}
	bodyComplete := false
	defer func() {
		cleanupErr := dflash2CleanupCall("request close", lease.close)
		if bodyComplete {
			err = errors.Join(err, cleanupErr)
		}
	}()
	res, err = p.runDFlash2Request(ctx, ids, maxNew, lease, preview, publish)
	bodyComplete = true
	return res, err
}

func (p *InKernelPlanner) runDFlash2Request(ctx context.Context, ids []int, maxNew int, lease *dflash2Lease,
	preview func([]int) (dflash2StopPreview, error), publish func(dflash2StopPreview) error) (res inKernelGenerateResult, err error) {
	if err = lease.request.Bootstrap(ctx, ids); err != nil {
		return res, err
	}
	res.promptTok = len(ids)
	for res.gen < maxNew {
		if err = p.checkExecutionDeadline(ctx, len(ids), 0, maxNew); err != nil {
			return res, err
		}
		if err = ctx.Err(); err != nil {
			return res, err
		}
		round, prepareErr := lease.request.Prepare(ctx, min(8, maxNew-res.gen))
		if prepareErr != nil {
			return res, prepareErr
		}
		if round == nil {
			return res, errors.New("dflash2: no prepared round")
		}
		// Abort is registered before inspecting any prepared data. A successfully
		// committed round is terminal; an interrupted callback never reuses it.
		var plan dflash2StopPreview
		var committed []int
		err = func() (err error) {
			done := false
			defer func() {
				if !done {
					cleanupErr := dflash2CleanupCall("round abort", round.Abort)
					if cleanupErr != nil {
						lease.target.domain.quarantine(cleanupErr, lease)
					}
					err = errors.Join(err, cleanupErr)
				}
			}()
			tokens := round.Tokens()
			if len(tokens) == 0 || len(tokens) > min(8, maxNew-res.gen) {
				return errors.New("dflash2: invalid verified token count")
			}
			plan, err = preview(tokens)
			if err != nil {
				return err
			}
			committed, err = round.Commit(ctx, len(plan.tokens))
			if err != nil {
				return err
			}
			done = true
			if !slices.Equal(committed, plan.tokens) {
				return errors.New("dflash2: committed prefix differs from preview")
			}
			return nil
		}()
		if err != nil {
			return res, err
		}
		res.gen += len(committed)
		res.stopped = plan.stopped
		if err = ctx.Err(); err != nil {
			return res, err
		}
		if err = publish(plan); err != nil {
			return res, err
		}
		if plan.stopped {
			return res, nil
		}
		if len(committed) == 0 {
			return res, fmt.Errorf("dflash2: no progress without a stop")
		}
	}
	return res, nil
}

// validateDFlash2Stops runs before either streaming projector or scanner allocation.
func validateDFlash2Stops(stops []string) error {
	if len(stops) > 1024 {
		return errors.New("dflash2: stop count budget exceeded")
	}
	for _, stop := range stops {
		if len(stop) > 64<<10 {
			return errors.New("dflash2: stop byte budget exceeded")
		}
	}
	return nil
}

// dflash2CleanupCall converts secondary cleanup panics into retained errors.
// The completion bit distinguishes a normal nil return from legacy panic(nil).
// A cleanup callback's own Goexit continues unwinding naturally; it is not
// claimed to preserve an earlier returned error or panic.
func dflash2CleanupCall(site string, fn func() error) (err error) {
	complete := false
	defer func() {
		value := recover()
		if !complete {
			err = fmt.Errorf("dflash2: %s did not return normally: %v", site, value)
		}
	}()
	err = fn()
	complete = true
	return err
}
