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

// ToolPut is the synthetic tool identifier submitted to the kernel for region store calls.
// Guard: Kernel admission check must succeed before storage mutation occurs.
// Invariant: Synthetic tool call is constructed with region and destructive metadata.
const ToolPut = "region.put"

// ToolGet is the synthetic tool identifier submitted to the kernel for region fetch calls.
// Guard: Evaluated as read-only and idempotent prior to resolver lookup.
// Invariant: Never executed against empty windows without stored refs.
const ToolGet = "region.get"

// ToolAccumulate is the synthetic tool identifier submitted to the kernel for region fold operations.
// Guard: Kernel admission check must succeed before fold execution and persistence.
// Invariant: Dispatched with destructive metadata marking state mutation.
const ToolAccumulate = "region.accumulate"

// AccumulateOp names the deterministic fold Accumulate applies.
// Invariant: Must be one of Sum, Max, or Concat; invalid operators return ErrUnknownOp.
type AccumulateOp string

const (
	// Sum performs deterministic numeric addition over float64 representations.
	// Guard: Both current and delta values must parse as valid floats.
	Sum AccumulateOp = "sum"

	// Max computes the numeric maximum over float64 representations.
	// Guard: Delta must parse as a valid float; empty current values fold directly to delta.
	Max AccumulateOp = "max"

	// Concat appends raw byte slices deterministically.
	// Invariant: Preserves exact byte ordering without insertion of delimiters.
	Concat AccumulateOp = "concat"
)

var (
	// ErrDenied indicates kernel admission refused the synthetic tool call.
	// Guard: Fail-closed; write and resolver mutations are skipped.
	ErrDenied = errors.New("region: access denied")

	// ErrEmpty indicates an uninitialized window holding no Ref was read.
	// Guard: Fail-closed; returns before kernel submission or resolver fetch.
	ErrEmpty = errors.New("region: empty window")

	// ErrNoKernel indicates a nil abi.Kernel was passed.
	// Guard: Non-nil admission authority required; fails closed immediately.
	ErrNoKernel = errors.New("region: nil kernel")

	// ErrNoResolver indicates no non-nil abi.Resolver is available.
	// Guard: Storage backend required; fails closed if neither kernel nor options provide one.
	ErrNoResolver = errors.New("region: nil resolver")

	// ErrScopeWiden indicates an attempted scope widening or an attempt to exceed ScopeFleet.
	// Guard: Monotonic scope enforcement; writes cannot widen beyond current scope or ScopeFleet.
	// Invariant: ScopeTenant is refused at region boundary.
	ErrScopeWiden = errors.New("region: write would widen scope")

	// ErrUnknownOp indicates an unrecognised AccumulateOp was supplied.
	// Guard: Closed operation set (Sum, Max, Concat); invalid operators fail closed.
	ErrUnknownOp = errors.New("region: unknown accumulate op")

	// ErrNilTarget indicates a nil target Ref pointer was passed to Accumulate.
	// Guard: Target pointer must be non-nil to receive updated Ref.
	ErrNilTarget = errors.New("region: nil target ref")

	accumulateLock sync.Mutex
)

// Coherence observes successful write-shaped completions. *vdso.VDSO implements it.
// Guard: Nil observer suppresses event dispatch without failing the write.
// Invariant: Completion notifications occur only after verified storage commit.
type Coherence interface {
	Emit(abi.Event)
}

// Option configures a Window during construction.
// Invariant: Evaluated in order prior to resolver validation.
type Option func(*Window)

// WithResolver overrides the kernel's active Resolver. Tests and custom stores use
// this when the kernel is only an adjudication fence.
// Guard: Overridden resolver must be non-nil at completion of New or ErrNoResolver is returned.
// Invariant: Explicit option takes precedence over k.Resolver().
func WithResolver(r abi.Resolver) Option {
	return func(w *Window) { w.resolver = r }
}

// WithCoherence overrides the write-completion observer. Passing nil disables the
// epoch bump, which is useful only for isolated tests.
// Guard: Optional observer; nil disables coherence emission without error.
// Invariant: Non-nil observer receives EvComplete events on successful mutations.
func WithCoherence(c Coherence) Option {
	return func(w *Window) { w.coherence = c }
}

// Window is a one-sided shared region containing the current Ref and a scope
// ceiling. Its mutex is the lost-update-safe linearization point for Accumulate.
// Guard: Concurrent reads and mutations linearize under internal mutex lock.
// Invariant: Window scope ceiling is immutable and bounded by ScopeFleet.
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
// Guard: k must be non-nil; scope must not exceed ScopeFleet.
// Invariant: Returns non-nil Window with non-nil resolver, or fail-closed error.
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
// Guard: Thread-safe read protected by window mutex lock.
// Invariant: Returns false when no Ref has been successfully written.
func (w *Window) Ref() (abi.Ref, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.ref, w.hasRef
}

// Put adjudicates and stores b, then makes the returned Ref the current window
// value. The requested scope must not widen the window or the existing Ref.
// Guard: Requires kernel admission; scope must not widen window.scope or current ref.Scope.
// Invariant: On rejection or error, existing window Ref remains unchanged. Defaults taint to TaintTainted.
func (w *Window) Put(ctx context.Context, b []byte, scope abi.ShareScope) (abi.Ref, abi.Verdict, error) {
	return w.PutTainted(ctx, b, scope, abi.TaintTainted)
}

// PutTainted is Put with an explicit taint label.
// Guard: Scope ceiling checks and kernel admission must pass before storage write.
// Invariant: Atomically advances window Ref upon successful adjudication and storage.
func (w *Window) PutTainted(ctx context.Context, b []byte, scope abi.ShareScope, taint abi.TaintLabel) (abi.Ref, abi.Verdict, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.commitWriteLocked(ctx, ToolPut, "", b, scope, taint, func() ([]byte, error) { return b, nil })
}

// commitWriteLocked enforces the scope ceiling, then adjudicates and commits a
// write through the window's own kernel, resolver, and coherence plane, and
// advances the window value. The caller must hold w.mu. See commitWrite for the
// store/derivation contract.
func (w *Window) commitWriteLocked(ctx context.Context, tool, op string, adjudicationPayload []byte, scope abi.ShareScope, taint abi.TaintLabel, store func() ([]byte, error)) (abi.Ref, abi.Verdict, error) {
	if err := w.checkWriteScopeLocked(scope); err != nil {
		return abi.Ref{}, abi.Verdict{}, err
	}
	ref, verdict, err := commitWrite(ctx, w.kernel, w.resolver, w.coherence, tool, op, adjudicationPayload, scope, taint, store)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	w.ref, w.hasRef = ref, true
	return ref, verdict, nil
}

// Get adjudicates a read and resolves the current Ref.
// Guard: Window must hold a valid Ref; kernel admission must allow the read.
// Invariant: Returns ErrEmpty without kernel dispatch if no Ref has been written.
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
// Guard: Kernel admission and valid numeric/byte format required.
// Invariant: Lost-update-safe via window mutex linearization. Defaults taint to TaintTainted.
func (w *Window) Accumulate(ctx context.Context, op AccumulateOp, delta []byte) (abi.Ref, abi.Verdict, error) {
	return w.AccumulateTainted(ctx, op, delta, abi.TaintTainted)
}

// AccumulateTainted is Accumulate with an explicit contribution taint.
// Guard: Kernel admission required; input numbers must parse if op is Sum or Max.
// Invariant: Monotonically joins existing Ref taint with contribution taint.
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
	return w.commitWriteLocked(ctx, ToolAccumulate, string(op), delta, scope, currentTaint, func() ([]byte, error) {
		return fold(op, current, delta)
	})
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
// Guard: k and k.Resolver() must be non-nil; scope must not exceed ScopeFleet.
// Invariant: Fails closed with ErrNoKernel, ErrNoResolver, or ErrScopeWiden before write. Defaults taint to TaintTainted.
func Put(ctx context.Context, k abi.Kernel, b []byte, scope abi.ShareScope) (abi.Ref, abi.Verdict, error) {
	return PutTainted(ctx, k, b, scope, abi.TaintTainted)
}

// PutTainted is Put with an explicit taint label.
// Guard: k and k.Resolver() must be non-nil; scope must not exceed ScopeFleet; kernel must admit call.
// Invariant: Adjudicated write with explicit taint; fails closed without storage on admission denial.
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
	return commitWrite(ctx, k, r, vdso.Default, ToolPut, "", b, scope, taint, func() ([]byte, error) { return b, nil })
}

// commitWrite adjudicates an adjudicationPayload write through k, derives the
// bytes to store via store (run only after a successful adjudication so a
// derivation error carries the adjudicated verdict), stores them through r, and
// emits the write-completion event on c. The stored bytes may differ from
// adjudicationPayload (e.g. Accumulate adjudicates the delta but stores the
// folded result). It is the shared write path behind both the Window methods
// and the stateless package functions.
func commitWrite(ctx context.Context, k abi.Kernel, r abi.Resolver, c Coherence, tool, op string, adjudicationPayload []byte, scope abi.ShareScope, taint abi.TaintLabel, store func() ([]byte, error)) (abi.Ref, abi.Verdict, error) {
	call, verdict, err := adjudicate(ctx, k, tool, adjudicationPayload, scope, taint, op)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	stored, err := store()
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	ref, err := putRef(ctx, r, stored, scope, taint)
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	emitWrite(c, call, ref)
	return ref, verdict, nil
}

// Get adjudicates and resolves ref through k's Resolver.
// Guard: k and k.Resolver() must be non-nil; kernel admission required.
// Invariant: Read-only call submitted to kernel; fails closed on denial or resolve error.
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
// Guard: target, k, and k.Resolver() must be non-nil; scope cannot widen beyond ScopeFleet.
// Invariant: Thread-safe through package accumulateLock; target updated only on successful commit.
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
	ref, verdict, err := commitWrite(ctx, k, r, vdso.Default, ToolAccumulate, string(op), delta, scope, taint, func() ([]byte, error) {
		return fold(op, current, delta)
	})
	if err != nil {
		return abi.Ref{}, verdict, err
	}
	*target = ref
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
	return inlineRef(body, scope, taint)
}

// inlineRef JSON-encodes body into an inline Ref tagged with scope and taint.
func inlineRef(body map[string]any, scope abi.ShareScope, taint abi.TaintLabel) abi.Ref {
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
	return inlineRef(body, ref.Scope, ref.Taint)
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
