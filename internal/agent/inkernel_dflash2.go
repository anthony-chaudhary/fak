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

// Sentinel errors for the per-stream DFlash2 tree contract. Callers match
// these with errors.Is; wrapped details name the offending value.
var (
	// ErrDFlash2TreesNotBuilt: a tree request arrived before the stream built its trees.
	ErrDFlash2TreesNotBuilt = errors.New("dflash2: tree stream has no built trees")
	// ErrDFlash2NotDrafting: kept-row absorption or node offers arrived outside the drafting window.
	ErrDFlash2NotDrafting = errors.New("dflash2: tree stream is not drafting")
	// ErrDFlash2TreeWidthExceeded: the requested tree exceeds the target width bound.
	ErrDFlash2TreeWidthExceeded = errors.New("dflash2: tree node request exceeds target width")
)

// DFlash2TreeSpec is the fixed per-stream DFlash2 tree shape. Attaching a tree
// is not a generic arbitrary-drafter load: only this exact shape may bind to an
// authenticated target, and every tap is checked against the target geometry
// before the target learns them.
type DFlash2TreeSpec struct {
	Taps []int // target layer indices whose feature rows feed the tree
}

// dflash2TreeStream owns one per-stream DFlash2 tree drafter attached to a
// registered serving domain. The target learns the drafter's taps at attach,
// before any forward; kept target rows are absorbed into drafter state;
// offered chain/tree nodes are bounded by the target width. Unsupported tree
// requests refuse with the closed sentinels above.
type dflash2TreeStream struct {
	mu       sync.Mutex
	target   *DFlash2Target
	taps     []int
	width    int
	built    bool
	drafting bool
	kept     []int
	retired  bool
	closeErr error
}

// AttachDFlash2Tree binds one per-stream DFlash2 tree drafter to the planner's
// authenticated target. Startup-only: the target learns the drafter's taps
// while no lease is live, so no forward can precede the tap declaration. A
// serial target panel verifies one row at a time, offers no parallel tree
// width, and refuses the attach.
func (p *InKernelPlanner) AttachDFlash2Tree(spec DFlash2TreeSpec) (*dflash2TreeStream, error) {
	if p == nil || p.dflash2 == nil {
		return nil, errors.New("dflash2: authenticated target required")
	}
	if err := p.dflash2.ValidateBinding(p.m, p.tok, p.backend, p.q4k, p.metal); err != nil {
		return nil, err
	}
	t := p.dflash2
	t.mu.Lock()
	busy := len(t.leases) != 0
	t.mu.Unlock()
	if busy {
		return nil, ErrDFlash2Busy
	}
	layers := p.m.Cfg.NumLayers
	if len(spec.Taps) == 0 || len(spec.Taps) > layers {
		return nil, errors.New("dflash2: tree taps must cover one to all target layers")
	}
	seen := make(map[int]bool, len(spec.Taps))
	for _, tap := range spec.Taps {
		if tap < 0 || tap >= layers || seen[tap] {
			return nil, errors.New("dflash2: tree tap outside target layers or repeated")
		}
		seen[tap] = true
	}
	width, err := dflash2TreeTargetWidth(t.options)
	if err != nil {
		return nil, err
	}
	if width == 0 {
		return nil, errors.New("dflash2: serial target panel offers no tree width")
	}
	return &dflash2TreeStream{target: t, taps: slices.Clone(spec.Taps), width: width}, nil
}

// dflash2TreeTargetWidth maps the authenticated target's verification panel to
// the node width bound offered to attached tree streams. It mirrors the model
// panel widths; serial verification offers no parallel tree width.
func dflash2TreeTargetWidth(o DFlash2TargetOptions) (int, error) {
	switch o.Request.TargetPanel {
	case model.DFlash2TargetSerial:
		return 0, nil
	case model.DFlash2TargetPanel4:
		return 4, nil
	case model.DFlash2TargetPanel8:
		return 8, nil
	default:
		return 0, errors.New("dflash2: unknown target panel selection")
	}
}

// usableLocked reports whether the stream still belongs to an open target.
// Callers hold s.mu; the target lock nests inside it and no other path takes
// the pair in reverse order.
func (s *dflash2TreeStream) usableLocked() error {
	if s.retired {
		return errors.New("dflash2: tree stream retired")
	}
	s.target.mu.Lock()
	defer s.target.mu.Unlock()
	if s.target.closed {
		return errors.New("dflash2: target closed")
	}
	return nil
}

// Build materializes the tree node table once per stream. The target already
// learned the taps at attach, before any forward.
func (s *dflash2TreeStream) Build(ctx context.Context) error {
	if ctx == nil {
		return errors.New("dflash2: context required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.usableLocked(); err != nil {
		return err
	}
	if s.built {
		return errors.New("dflash2: tree stream already built")
	}
	s.built = true
	return nil
}

// Begin opens the drafting window. Node offers and kept-row absorption refuse
// while the window is closed.
func (s *dflash2TreeStream) Begin(ctx context.Context) error {
	if ctx == nil {
		return errors.New("dflash2: context required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.usableLocked(); err != nil {
		return err
	}
	if !s.built {
		return ErrDFlash2TreesNotBuilt
	}
	if s.drafting {
		return errors.New("dflash2: tree stream already drafting")
	}
	s.drafting = true
	return nil
}

// Finish closes the drafting window. Absorbed kept rows remain drafter state.
func (s *dflash2TreeStream) Finish() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.usableLocked(); err != nil {
		return err
	}
	if !s.drafting {
		return ErrDFlash2NotDrafting
	}
	s.drafting = false
	return nil
}

// Nodes offers chain/tree node IDs bounded by the target width: exactly the
// first requested node IDs, never more than width. A request past the bound is
// unsupported and refuses instead of silently shrinking.
func (s *dflash2TreeStream) Nodes(ctx context.Context, requested int) ([]int, error) {
	if ctx == nil {
		return nil, errors.New("dflash2: context required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.usableLocked(); err != nil {
		return nil, err
	}
	if !s.built {
		return nil, ErrDFlash2TreesNotBuilt
	}
	if !s.drafting {
		return nil, ErrDFlash2NotDrafting
	}
	if requested < 1 {
		return nil, errors.New("dflash2: tree node request must be positive")
	}
	if requested > s.width {
		return nil, fmt.Errorf("%w: requested=%d width=%d", ErrDFlash2TreeWidthExceeded, requested, s.width)
	}
	nodes := make([]int, requested)
	for i := range nodes {
		nodes[i] = i
	}
	return nodes, nil
}

// AbsorbKept folds committed kept target rows into drafter state. Rows are the
// token IDs a committed round kept; they persist across Finish until Close.
func (s *dflash2TreeStream) AbsorbKept(rows []int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.usableLocked(); err != nil {
		return err
	}
	if !s.drafting {
		return ErrDFlash2NotDrafting
	}
	if len(rows) == 0 {
		return errors.New("dflash2: kept rows required")
	}
	for _, row := range rows {
		if row < 0 {
			return errors.New("dflash2: kept row outside vocabulary")
		}
	}
	s.kept = append(s.kept, rows...)
	return nil
}

// KeptRows returns an owned copy of the absorbed kept target rows.
func (s *dflash2TreeStream) KeptRows() []int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.kept)
}

// Taps returns the tap set the target learned before any forward.
func (s *dflash2TreeStream) Taps() []int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.taps)
}

// Width reports the target width bound on offered tree nodes.
func (s *dflash2TreeStream) Width() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.width
}

// Close retires the stream exactly once and ends the drafter stream lifecycle.
// Repeated Close returns the stored result; a clean retirement stores nil.
func (s *dflash2TreeStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.retired {
		return s.closeErr
	}
	s.retired = true
	s.drafting = false
	s.closeErr = nil
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
		sp.NativeInferenceReceipt || sp.NativeDecodeTokenIDs || sp.NativeDecodeTopLogprobs > 0 ||
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
