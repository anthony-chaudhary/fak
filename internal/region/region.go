package region

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

const (
	ToolPut        = "region.put"
	ToolGet        = "region.get"
	ToolAccumulate = "region.accumulate"
)

// AccumulateOp names the deterministic fold Accumulate applies.
type AccumulateOp string

const (
	Sum    AccumulateOp = "sum"
	Max    AccumulateOp = "max"
	Concat AccumulateOp = "concat"
)

var (
	ErrDenied      = errors.New("region: access denied")
	ErrEmpty       = errors.New("region: empty window")
	ErrNoKernel    = errors.New("region: nil kernel")
	ErrNoResolver  = errors.New("region: nil resolver")
	ErrScopeWiden  = errors.New("region: write would widen scope")
	ErrUnknownOp   = errors.New("region: unknown accumulate op")
	ErrNilTarget   = errors.New("region: nil target ref")
	accumulateLock sync.Mutex
)

// Coherence observes successful write-shaped completions. *vdso.VDSO implements it.
type Coherence interface {
	Emit(abi.Event)
}

// Option configures a Window.
type Option func(*Window)

// WithResolver overrides the kernel's active Resolver. Tests and custom stores use
// this when the kernel is only an adjudication fence.
func WithResolver(r abi.Resolver) Option {
	return func(w *Window) { w.resolver = r }
}

// WithCoherence overrides the write-completion observer. Passing nil disables the
// epoch bump, which is useful only for isolated tests.
func WithCoherence(c Coherence) Option {
	return func(w *Window) { w.coherence = c }
}

// Window is a one-sided shared region containing the current Ref and a scope
// ceiling. Its mutex is the lost-update-safe linearization point for Accumulate.
type Window struct {
	mu        sync.Mutex
	kernel    abi.Kernel
	resolver  abi.Resolver
	coherence Coherence
	scope     abi.ShareScope
	ref       abi.Ref
	hasRef    bool
}

// New builds a region window admitted by k. scope is the maximum share scope this
// window may write; ScopeTenant is rejected because region writes cap at ScopeFleet.
func New(k abi.Kernel, scope abi.ShareScope, opts ...Option) (*Window, error) {
	if k == nil {
		return nil, ErrNoKernel
	}
	if wider(scope, abi.ScopeFleet) {
		return nil, ErrScopeWiden
	}
	w := &Window{
		kernel:    k,
		resolver:  k.Resolver(),
		coherence: vdso.Default,
		scope:     scope,
	}
	for _, opt := range opts {
		opt(w)
	}
	if w.resolver == nil {
		return nil, ErrNoResolver
	}
	return w, nil
}

// Ref returns the current window Ref, if one has been written.
func (w *Window) Ref() (abi.Ref, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.ref, w.hasRef
}

// Put adjudicates and stores b, then makes the returned Ref the current window
// value. The requested scope must not widen the window or the existing Ref.
func (w *Window) Put(ctx context.Context, b []byte, scope abi.ShareScope) (abi.Ref, abi.Verdict, error) {
	return w.PutTainted(ctx, b, scope, abi.TaintTainted)
}

// PutTainted is Put with an explicit taint label.
func (w *Window) PutTainted(ctx context.Context, b []byte, scope abi.ShareScope, taint abi.TaintLabel) (abi.Ref, abi.Verdict, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.checkWriteScopeLocked(scope); err != nil {
		return abi.Ref{}, abi.Verdict{}, err
	}
	call, verdict, err := adjudicate(ctx, w.kernel, ToolPut, b, scope, taint, "")
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	ref, err := putRef(ctx, w.resolver, b, scope, taint)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	w.ref, w.hasRef = ref, true
	emitWrite(w.coherence, call, ref)
	return ref, verdict, nil
}

// Get adjudicates a read and resolves the current Ref.
func (w *Window) Get(ctx context.Context) ([]byte, abi.Ref, abi.Verdict, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.hasRef {
		return nil, abi.Ref{}, abi.Verdict{}, ErrEmpty
	}
	args := refArgs(w.ref, "", nil)
	_, verdict, err := submitCall(ctx, w.kernel, ToolGet, args, false)
	if err != nil {
		return nil, abi.Ref{}, verdict, err
	}
	b, err := w.resolver.Resolve(ctx, w.ref)
	if err != nil {
		return nil, abi.Ref{}, verdict, err
	}
	return b, w.ref, verdict, nil
}

// Accumulate adjudicates and applies op to the current Ref and delta. A missing
// window value folds from the operation identity: 0 for sum, delta for max, and
// empty bytes for concat.
func (w *Window) Accumulate(ctx context.Context, op AccumulateOp, delta []byte) (abi.Ref, abi.Verdict, error) {
	return w.AccumulateTainted(ctx, op, delta, abi.TaintTainted)
}

// AccumulateTainted is Accumulate with an explicit contribution taint.
func (w *Window) AccumulateTainted(ctx context.Context, op AccumulateOp, delta []byte, taint abi.TaintLabel) (abi.Ref, abi.Verdict, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	scope := w.scope
	currentTaint := taint
	var current []byte
	if w.hasRef {
		scope = w.ref.Scope
		currentTaint = joinTaint(w.ref.Taint, taint)
		var err error
		current, err = w.resolver.Resolve(ctx, w.ref)
		if err != nil {
			return abi.Ref{}, abi.Verdict{}, err
		}
	}
	if err := w.checkWriteScopeLocked(scope); err != nil {
		return abi.Ref{}, abi.Verdict{}, err
	}
	call, verdict, err := adjudicate(ctx, w.kernel, ToolAccumulate, delta, scope, currentTaint, string(op))
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	next, err := fold(op, current, delta)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	ref, err := putRef(ctx, w.resolver, next, scope, currentTaint)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	w.ref, w.hasRef = ref, true
	emitWrite(w.coherence, call, ref)
	return ref, verdict, nil
}

func (w *Window) checkWriteScopeLocked(scope abi.ShareScope) error {
	if wider(scope, abi.ScopeFleet) || wider(scope, w.scope) {
		return ErrScopeWiden
	}
	if w.hasRef && wider(scope, w.ref.Scope) {
		return ErrScopeWiden
	}
	return nil
}

// Put adjudicates and stores b through k's Resolver. It is the stateless helper
// for callers that already hold their own Ref slot.
func Put(ctx context.Context, k abi.Kernel, b []byte, scope abi.ShareScope) (abi.Ref, abi.Verdict, error) {
	return PutTainted(ctx, k, b, scope, abi.TaintTainted)
}

// PutTainted is Put with an explicit taint label.
func PutTainted(ctx context.Context, k abi.Kernel, b []byte, scope abi.ShareScope, taint abi.TaintLabel) (abi.Ref, abi.Verdict, error) {
	if k == nil {
		return abi.Ref{}, abi.Verdict{}, ErrNoKernel
	}
	if wider(scope, abi.ScopeFleet) {
		return abi.Ref{}, abi.Verdict{}, ErrScopeWiden
	}
	r := k.Resolver()
	if r == nil {
		return abi.Ref{}, abi.Verdict{}, ErrNoResolver
	}
	call, verdict, err := adjudicate(ctx, k, ToolPut, b, scope, taint, "")
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	ref, err := putRef(ctx, r, b, scope, taint)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	emitWrite(vdso.Default, call, ref)
	return ref, verdict, nil
}

// Get adjudicates and resolves ref through k's Resolver.
func Get(ctx context.Context, k abi.Kernel, ref abi.Ref) ([]byte, abi.Verdict, error) {
	if k == nil {
		return nil, abi.Verdict{}, ErrNoKernel
	}
	r := k.Resolver()
	if r == nil {
		return nil, abi.Verdict{}, ErrNoResolver
	}
	args := refArgs(ref, "", nil)
	_, verdict, err := submitCall(ctx, k, ToolGet, args, false)
	if err != nil {
		return nil, verdict, err
	}
	b, err := r.Resolve(ctx, ref)
	if err != nil {
		return nil, verdict, err
	}
	return b, verdict, nil
}

// Accumulate adjudicates and applies op to *target under a package-wide
// linearization lock. Window.Accumulate gives narrower locking when callers can
// own a Window value; this helper exists for a shared Ref slot.
func Accumulate(ctx context.Context, k abi.Kernel, target *abi.Ref, op AccumulateOp, delta []byte) (abi.Ref, abi.Verdict, error) {
	if target == nil {
		return abi.Ref{}, abi.Verdict{}, ErrNilTarget
	}
	accumulateLock.Lock()
	defer accumulateLock.Unlock()
	if k == nil {
		return abi.Ref{}, abi.Verdict{}, ErrNoKernel
	}
	r := k.Resolver()
	if r == nil {
		return abi.Ref{}, abi.Verdict{}, ErrNoResolver
	}
	scope := target.Scope
	taint := target.Taint
	var current []byte
	if !zeroRef(*target) {
		var err error
		current, err = r.Resolve(ctx, *target)
		if err != nil {
			return abi.Ref{}, abi.Verdict{}, err
		}
	} else {
		taint = abi.TaintTainted
	}
	if wider(scope, abi.ScopeFleet) {
		return abi.Ref{}, abi.Verdict{}, ErrScopeWiden
	}
	call, verdict, err := adjudicate(ctx, k, ToolAccumulate, delta, scope, taint, string(op))
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	next, err := fold(op, current, delta)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	ref, err := putRef(ctx, r, next, scope, taint)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	*target = ref
	emitWrite(vdso.Default, call, ref)
	return ref, verdict, nil
}

func adjudicate(ctx context.Context, k abi.Kernel, tool string, b []byte, scope abi.ShareScope, taint abi.TaintLabel, op string) (*abi.ToolCall, abi.Verdict, error) {
	if k == nil {
		return nil, abi.Verdict{}, ErrNoKernel
	}
	args := writeArgs(b, scope, taint, op)
	return submitCall(ctx, k, tool, args, true)
}

func submitCall(ctx context.Context, k abi.Kernel, tool string, args abi.Ref, destructive bool) (*abi.ToolCall, abi.Verdict, error) {
	meta := map[string]string{"region": "true"}
	if destructive {
		meta["destructive"] = "true"
	} else {
		meta["readOnlyHint"] = "true"
		meta["idempotentHint"] = "true"
	}
	call := &abi.ToolCall{Tool: tool, Args: args, Meta: meta}
	_, verdict := k.Submit(ctx, call)
	if verdict.Kind != abi.VerdictAllow {
		return call, verdict, fmt.Errorf("%w: %s by %s", ErrDenied, abi.ReasonName(verdict.Reason), verdict.By)
	}
	return call, verdict, nil
}

func putRef(ctx context.Context, r abi.Resolver, b []byte, scope abi.ShareScope, taint abi.TaintLabel) (abi.Ref, error) {
	ref, err := r.Put(ctx, b)
	if err != nil {
		return abi.Ref{}, err
	}
	ref.Scope = scope
	ref.Taint = taint
	ref.Len = int64(len(b))
	return ref, nil
}

func emitWrite(c Coherence, call *abi.ToolCall, ref abi.Ref) {
	if c == nil || call == nil {
		return
	}
	c.Emit(abi.Event{
		Kind: abi.EvComplete,
		Call: call,
		Result: &abi.Result{
			Call:    call,
			Status:  abi.StatusOK,
			Outcome: abi.OutcomeCommitted,
			Payload: ref,
		},
	})
}

func fold(op AccumulateOp, current, delta []byte) ([]byte, error) {
	switch op {
	case Sum:
		a, err := numberOrZero(current)
		if err != nil {
			return nil, err
		}
		b, err := number(delta)
		if err != nil {
			return nil, err
		}
		return []byte(strconv.FormatFloat(a+b, 'g', -1, 64)), nil
	case Max:
		b, err := number(delta)
		if err != nil {
			return nil, err
		}
		if len(strings.TrimSpace(string(current))) == 0 {
			return []byte(strconv.FormatFloat(b, 'g', -1, 64)), nil
		}
		a, err := number(current)
		if err != nil {
			return nil, err
		}
		if b > a {
			a = b
		}
		return []byte(strconv.FormatFloat(a, 'g', -1, 64)), nil
	case Concat:
		out := make([]byte, 0, len(current)+len(delta))
		out = append(out, current...)
		out = append(out, delta...)
		return out, nil
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnknownOp, op)
	}
}

func numberOrZero(b []byte) (float64, error) {
	if len(strings.TrimSpace(string(b))) == 0 {
		return 0, nil
	}
	return number(b)
}

func number(b []byte) (float64, error) {
	n, err := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
	if err != nil {
		return 0, fmt.Errorf("region: parse accumulate number %q: %w", string(b), err)
	}
	return n, nil
}

func writeArgs(b []byte, scope abi.ShareScope, taint abi.TaintLabel, op string) abi.Ref {
	body := map[string]any{
		"len":   len(b),
		"scope": int(scope),
		"taint": int(taint),
	}
	if op != "" {
		body["op"] = op
	}
	encoded, _ := json.Marshal(body)
	return abi.Ref{Kind: abi.RefInline, Inline: encoded, Len: int64(len(encoded)), Scope: scope, Taint: taint}
}

func refArgs(ref abi.Ref, op string, extra map[string]any) abi.Ref {
	body := map[string]any{
		"kind":   int(ref.Kind),
		"digest": ref.Digest,
		"len":    ref.Len,
		"scope":  int(ref.Scope),
		"taint":  int(ref.Taint),
	}
	for k, v := range extra {
		body[k] = v
	}
	if op != "" {
		body["op"] = op
	}
	encoded, _ := json.Marshal(body)
	return abi.Ref{Kind: abi.RefInline, Inline: encoded, Len: int64(len(encoded)), Scope: ref.Scope, Taint: ref.Taint}
}

func joinTaint(a, b abi.TaintLabel) abi.TaintLabel {
	if a == abi.TaintQuarantined || b == abi.TaintQuarantined {
		return abi.TaintQuarantined
	}
	if a == abi.TaintTainted || b == abi.TaintTainted {
		return abi.TaintTainted
	}
	return abi.TaintTrusted
}

func wider(a, b abi.ShareScope) bool { return a > b }

func zeroRef(r abi.Ref) bool {
	return r.Kind == 0 && r.Digest == "" && len(r.Inline) == 0 && r.Handle == 0 && r.Len == 0
}
