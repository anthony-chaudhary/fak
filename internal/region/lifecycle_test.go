package region

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

type nilResolverKernel struct {
	verdict abi.Verdict
}

func (n *nilResolverKernel) Submit(ctx context.Context, c *abi.ToolCall) (abi.SubmissionHandle, abi.Verdict) {
	return abi.SubmissionHandle{Seq: 1}, n.verdict
}

func (n *nilResolverKernel) Reap(ctx context.Context, h abi.SubmissionHandle) (*abi.Result, error) {
	return nil, nil
}

func (n *nilResolverKernel) Syscall(ctx context.Context, c *abi.ToolCall) (*abi.Result, abi.Verdict) {
	return &abi.Result{Call: c, Status: abi.StatusOK}, n.verdict
}

func (n *nilResolverKernel) Resolver() abi.Resolver {
	return nil
}

func (n *nilResolverKernel) Negotiate(caps []abi.Capability) []abi.Capability {
	return caps
}

type recordCoherence struct {
	mu     sync.Mutex
	events []abi.Event
}

func (r *recordCoherence) Emit(ev abi.Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, ev)
}

func (r *recordCoherence) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.events)
}

func TestWindowCreationValidation(t *testing.T) {
	k := allowKernel()

	if _, err := New(nil, abi.ScopeFleet); !errors.Is(err, ErrNoKernel) {
		t.Fatalf("New(nil, ScopeFleet) err = %v, want ErrNoKernel", err)
	}

	if _, err := New(k, abi.ScopeTenant); !errors.Is(err, ErrScopeWiden) {
		t.Fatalf("New(k, ScopeTenant) err = %v, want ErrScopeWiden", err)
	}

	nilK := &nilResolverKernel{verdict: abi.Verdict{Kind: abi.VerdictAllow}}
	if _, err := New(nilK, abi.ScopeFleet); !errors.Is(err, ErrNoResolver) {
		t.Fatalf("New(nilK, ScopeFleet) err = %v, want ErrNoResolver", err)
	}

	overrideRes := blob.New()
	w, err := New(nilK, abi.ScopeFleet, WithResolver(overrideRes), WithCoherence(nil))
	if err != nil {
		t.Fatalf("New with WithResolver failed: %v", err)
	}
	if w == nil {
		t.Fatal("New returned nil Window")
	}
}

func TestWindowEmptyState(t *testing.T) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}

	ref, ok := w.Ref()
	if ok || ref.Digest != "" {
		t.Fatalf("Ref() on empty window = %+v, %v; want false and zero ref", ref, ok)
	}

	if _, _, _, err := w.Get(context.Background()); !errors.Is(err, ErrEmpty) {
		t.Fatalf("Get() on empty window err = %v, want ErrEmpty", err)
	}
}

func TestWindowLifecycleAndTaintProgression(t *testing.T) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()

	// 1. Initial Put with Trusted taint
	ref1, verdict1, err := w.PutTainted(ctx, []byte("10"), abi.ScopeAgent, abi.TaintTrusted)
	if err != nil {
		t.Fatalf("PutTainted: %v", err)
	}
	if verdict1.Kind != abi.VerdictAllow {
		t.Fatalf("verdict = %v, want Allow", verdict1.Kind)
	}
	if ref1.Taint != abi.TaintTrusted || ref1.Scope != abi.ScopeAgent {
		t.Fatalf("ref1 taint/scope = %v/%v, want Trusted/Agent", ref1.Taint, ref1.Scope)
	}

	// 2. Read back
	val, refGet, _, err := w.Get(ctx)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(val) != "10" || refGet.Digest != ref1.Digest {
		t.Fatalf("Get = %q, ref = %+v; want 10, %+v", string(val), refGet, ref1)
	}

	// 3. Accumulate with Tainted taint -> joins Trusted + Tainted => Tainted
	ref2, _, err := w.AccumulateTainted(ctx, Sum, []byte("5"), abi.TaintTainted)
	if err != nil {
		t.Fatalf("AccumulateTainted(Sum, 5): %v", err)
	}
	if ref2.Taint != abi.TaintTainted {
		t.Fatalf("ref2 taint = %v, want Tainted", ref2.Taint)
	}

	val, _, _, err = w.Get(ctx)
	if err != nil {
		t.Fatalf("Get after sum: %v", err)
	}
	if string(val) != "15" {
		t.Fatalf("val after sum = %q, want 15", string(val))
	}

	// 4. Accumulate with Quarantined taint -> joins Tainted + Quarantined => Quarantined
	ref3, _, err := w.AccumulateTainted(ctx, Sum, []byte("10"), abi.TaintQuarantined)
	if err != nil {
		t.Fatalf("AccumulateTainted(Sum, 10): %v", err)
	}
	if ref3.Taint != abi.TaintQuarantined {
		t.Fatalf("ref3 taint = %v, want Quarantined", ref3.Taint)
	}

	val, _, _, err = w.Get(ctx)
	if err != nil {
		t.Fatalf("Get after second sum: %v", err)
	}
	if string(val) != "25" {
		t.Fatalf("val after second sum = %q, want 25", string(val))
	}
}

func TestWindowScopeMonotonicityRules(t *testing.T) {
	k := allowKernel()
	ctx := context.Background()

	// Window with ScopeAgent ceiling
	wAgent, err := New(k, abi.ScopeAgent, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := wAgent.Put(ctx, []byte("val"), abi.ScopeFleet); !errors.Is(err, ErrScopeWiden) {
		t.Fatalf("Put ScopeFleet on ScopeAgent window err = %v, want ErrScopeWiden", err)
	}

	// Window with ScopeFleet ceiling, but current Ref has ScopeAgent
	wFleet, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := wFleet.Put(ctx, []byte("val"), abi.ScopeAgent); err != nil {
		t.Fatalf("Put ScopeAgent: %v", err)
	}
	if _, _, err := wFleet.Put(ctx, []byte("val2"), abi.ScopeFleet); !errors.Is(err, ErrScopeWiden) {
		t.Fatalf("Put ScopeFleet after ScopeAgent ref err = %v, want ErrScopeWiden", err)
	}
}

func TestWindowAccumulateOperations(t *testing.T) {
	k := allowKernel()
	ctx := context.Background()

	t.Run("SumEmptyInitial", func(t *testing.T) {
		w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := w.Accumulate(ctx, Sum, []byte("42")); err != nil {
			t.Fatalf("Accumulate Sum from empty: %v", err)
		}
		b, _, _, err := w.Get(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != "42" {
			t.Fatalf("got %q, want 42", string(b))
		}
	})

	t.Run("SumInvalidNumber", func(t *testing.T) {
		w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := w.Accumulate(ctx, Sum, []byte("not-a-number")); err == nil {
			t.Fatal("expected error on invalid number for sum, got nil")
		}
	})

	t.Run("MaxBehavior", func(t *testing.T) {
		w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
		if err != nil {
			t.Fatal(err)
		}
		// First fold from empty sets value to delta
		if _, _, err := w.Accumulate(ctx, Max, []byte("50")); err != nil {
			t.Fatal(err)
		}
		// Smaller delta retains 50
		if _, _, err := w.Accumulate(ctx, Max, []byte("20")); err != nil {
			t.Fatal(err)
		}
		b, _, _, err := w.Get(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != "50" {
			t.Fatalf("got %q, want 50", string(b))
		}
		// Larger delta updates to 100
		if _, _, err := w.Accumulate(ctx, Max, []byte("100")); err != nil {
			t.Fatal(err)
		}
		b, _, _, err = w.Get(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != "100" {
			t.Fatalf("got %q, want 100", string(b))
		}
	})

	t.Run("MaxInvalidNumber", func(t *testing.T) {
		w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := w.Accumulate(ctx, Max, []byte("abc")); err == nil {
			t.Fatal("expected error on invalid number for max, got nil")
		}
	})

	t.Run("ConcatBehavior", func(t *testing.T) {
		w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := w.Accumulate(ctx, Concat, []byte("foo")); err != nil {
			t.Fatal(err)
		}
		if _, _, err := w.Accumulate(ctx, Concat, []byte("bar")); err != nil {
			t.Fatal(err)
		}
		b, _, _, err := w.Get(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != "foobar" {
			t.Fatalf("got %q, want foobar", string(b))
		}
	})

	t.Run("UnknownOp", func(t *testing.T) {
		w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := w.Accumulate(ctx, AccumulateOp("unsupported"), []byte("1")); !errors.Is(err, ErrUnknownOp) {
			t.Fatalf("Accumulate unsupported op err = %v, want ErrUnknownOp", err)
		}
	})
}

func TestWindowDeniedKernelResponses(t *testing.T) {
	ctx := context.Background()
	kDeny := newFakeKernel(abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"})

	w, err := New(kDeny, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}

	// Denied Put
	if _, _, err := w.Put(ctx, []byte("secret"), abi.ScopeFleet); !errors.Is(err, ErrDenied) {
		t.Fatalf("Put denied err = %v, want ErrDenied", err)
	}
	if _, ok := w.Ref(); ok {
		t.Fatal("Ref() returned ok after denied Put")
	}

	// Denied Accumulate
	if _, _, err := w.Accumulate(ctx, Sum, []byte("10")); !errors.Is(err, ErrDenied) {
		t.Fatalf("Accumulate denied err = %v, want ErrDenied", err)
	}
	if _, ok := w.Ref(); ok {
		t.Fatal("Ref() returned ok after denied Accumulate")
	}

	// Pre-populate with allowKernel, then swap kernel to deny for Get
	kAllow := allowKernel()
	wAllow, err := New(kAllow, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := wAllow.Put(ctx, []byte("sample"), abi.ScopeFleet); err != nil {
		t.Fatal(err)
	}
	// Switch kernel to deny kernel
	wAllow.kernel = kDeny
	if _, _, _, err := wAllow.Get(ctx); !errors.Is(err, ErrDenied) {
		t.Fatalf("Get denied err = %v, want ErrDenied", err)
	}
}

func TestStatelessHelpersValidationAndLifecycle(t *testing.T) {
	ctx := context.Background()
	k := allowKernel()

	// 1. Put validations
	if _, _, err := Put(ctx, nil, []byte("val"), abi.ScopeFleet); !errors.Is(err, ErrNoKernel) {
		t.Fatalf("Put(nil) err = %v, want ErrNoKernel", err)
	}
	if _, _, err := Put(ctx, k, []byte("val"), abi.ScopeTenant); !errors.Is(err, ErrScopeWiden) {
		t.Fatalf("Put(ScopeTenant) err = %v, want ErrScopeWiden", err)
	}
	nilK := &nilResolverKernel{verdict: abi.Verdict{Kind: abi.VerdictAllow}}
	if _, _, err := Put(ctx, nilK, []byte("val"), abi.ScopeFleet); !errors.Is(err, ErrNoResolver) {
		t.Fatalf("Put(nilK) err = %v, want ErrNoResolver", err)
	}

	// 2. Put valid and Get valid
	ref, verdict, err := Put(ctx, k, []byte("stateless_data"), abi.ScopeFleet)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if verdict.Kind != abi.VerdictAllow {
		t.Fatalf("verdict = %v, want Allow", verdict.Kind)
	}

	gotBytes, verdictGet, err := Get(ctx, k, ref)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if verdictGet.Kind != abi.VerdictAllow || string(gotBytes) != "stateless_data" {
		t.Fatalf("Get payload = %q, verdict = %v", string(gotBytes), verdictGet)
	}

	// 3. Get validations
	if _, _, err := Get(ctx, nil, ref); !errors.Is(err, ErrNoKernel) {
		t.Fatalf("Get(nil) err = %v, want ErrNoKernel", err)
	}
	if _, _, err := Get(ctx, nilK, ref); !errors.Is(err, ErrNoResolver) {
		t.Fatalf("Get(nilK) err = %v, want ErrNoResolver", err)
	}

	// 4. Accumulate validations
	if _, _, err := Accumulate(ctx, k, nil, Sum, []byte("1")); !errors.Is(err, ErrNilTarget) {
		t.Fatalf("Accumulate(nil target) err = %v, want ErrNilTarget", err)
	}
	targetRef := abi.Ref{Scope: abi.ScopeFleet}
	if _, _, err := Accumulate(ctx, nil, &targetRef, Sum, []byte("1")); !errors.Is(err, ErrNoKernel) {
		t.Fatalf("Accumulate(nil kernel) err = %v, want ErrNoKernel", err)
	}
	if _, _, err := Accumulate(ctx, nilK, &targetRef, Sum, []byte("1")); !errors.Is(err, ErrNoResolver) {
		t.Fatalf("Accumulate(nilK) err = %v, want ErrNoResolver", err)
	}
	tenantTarget := abi.Ref{Scope: abi.ScopeTenant}
	if _, _, err := Accumulate(ctx, k, &tenantTarget, Sum, []byte("1")); !errors.Is(err, ErrScopeWiden) {
		t.Fatalf("Accumulate(tenantTarget) err = %v, want ErrScopeWiden", err)
	}

	// 5. Accumulate lifecycle on target Ref
	var sharedTarget abi.Ref
	sharedTarget.Scope = abi.ScopeFleet
	refAcc, _, err := Accumulate(ctx, k, &sharedTarget, Sum, []byte("100"))
	if err != nil {
		t.Fatalf("Accumulate on zero target: %v", err)
	}
	if sharedTarget.Digest != refAcc.Digest {
		t.Fatalf("target not updated to returned ref: target=%+v refAcc=%+v", sharedTarget, refAcc)
	}

	val, _, err := Get(ctx, k, sharedTarget)
	if err != nil {
		t.Fatalf("Get after accumulate: %v", err)
	}
	if string(val) != "100" {
		t.Fatalf("Get = %q, want 100", string(val))
	}
}

func TestCoherenceObserverEmission(t *testing.T) {
	ctx := context.Background()
	k := allowKernel()
	recorder := &recordCoherence{}

	w, err := New(k, abi.ScopeFleet, WithCoherence(recorder))
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err := w.Put(ctx, []byte("event1"), abi.ScopeFleet); err != nil {
		t.Fatal(err)
	}
	if recorder.count() != 1 {
		t.Fatalf("coherence count = %d, want 1", recorder.count())
	}

	if _, _, err := w.Accumulate(ctx, Concat, []byte("_event2")); err != nil {
		t.Fatal(err)
	}
	if recorder.count() != 2 {
		t.Fatalf("coherence count = %d, want 2", recorder.count())
	}
}

func BenchmarkRegion(b *testing.B) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	payload := []byte("benchmark-payload")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref, _, err := w.Put(ctx, payload, abi.ScopeFleet)
		if err != nil {
			b.Fatal(err)
		}
		data, _, _, err := w.Get(ctx)
		if err != nil {
			b.Fatal(err)
		}
		if len(data) == 0 || ref.Digest == "" {
			b.Fatal("unexpected empty benchmark result")
		}
	}
}

func BenchmarkRegionPut(b *testing.B) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	payload := []byte("bench-value")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := w.Put(ctx, payload, abi.ScopeFleet); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRegionGet(b *testing.B) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	if _, _, err := w.Put(ctx, []byte("bench-value"), abi.ScopeFleet); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := w.Get(ctx); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRegionAccumulateSum(b *testing.B) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	if _, _, err := w.Put(ctx, []byte("0"), abi.ScopeFleet); err != nil {
		b.Fatal(err)
	}
	delta := []byte("1")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := w.Accumulate(ctx, Sum, delta); err != nil {
			b.Fatal(err)
		}
	}
}
