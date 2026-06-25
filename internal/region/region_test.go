package region

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

type fakeKernel struct {
	res     abi.Resolver
	verdict abi.Verdict
	submits int64
}

func newFakeKernel(v abi.Verdict) *fakeKernel {
	return &fakeKernel{res: blob.New(), verdict: v}
}

func (f *fakeKernel) Submit(ctx context.Context, c *abi.ToolCall) (abi.SubmissionHandle, abi.Verdict) {
	seq := uint64(atomic.AddInt64(&f.submits, 1))
	return abi.SubmissionHandle{Seq: seq}, f.verdict
}

func (f *fakeKernel) Reap(ctx context.Context, h abi.SubmissionHandle) (*abi.Result, error) {
	return nil, nil
}

func (f *fakeKernel) Syscall(ctx context.Context, c *abi.ToolCall) (*abi.Result, abi.Verdict) {
	_, v := f.Submit(ctx, c)
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}, v
}

func (f *fakeKernel) Resolver() abi.Resolver { return f.res }
func (f *fakeKernel) Negotiate(caps []abi.Capability) []abi.Capability {
	return caps
}

func allowKernel() *fakeKernel {
	return newFakeKernel(abi.Verdict{Kind: abi.VerdictAllow, By: "test"})
}

func TestPutGetRoundTrip(t *testing.T) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}
	ref, verdict, err := w.Put(context.Background(), []byte("hello"), abi.ScopeFleet)
	if err != nil {
		t.Fatal(err)
	}
	if verdict.Kind != abi.VerdictAllow {
		t.Fatalf("Put verdict = %v, want Allow", verdict.Kind)
	}
	if ref.Scope != abi.ScopeFleet || ref.Taint != abi.TaintTainted {
		t.Fatalf("ref scope/taint = %v/%v, want fleet/tainted", ref.Scope, ref.Taint)
	}
	got, gotRef, _, err := w.Get(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello" || gotRef.Digest != ref.Digest {
		t.Fatalf("Get = %q ref=%+v, want hello ref digest %q", got, gotRef, ref.Digest)
	}
	if atomic.LoadInt64(&k.submits) != 2 {
		t.Fatalf("submits = %d, want Put+Get adjudicated", k.submits)
	}
}

func TestAccumulateConcurrentSumLostUpdateSafe(t *testing.T) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := w.Put(context.Background(), []byte("0"), abi.ScopeFleet); err != nil {
		t.Fatal(err)
	}
	const n = 128
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			if _, _, err := w.Accumulate(context.Background(), Sum, []byte("1")); err != nil {
				t.Errorf("Accumulate: %v", err)
			}
		}()
	}
	wg.Wait()
	got, _, _, err := w.Get(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "128" {
		t.Fatalf("sum = %q, want 128", got)
	}
}

func TestScopeCannotWidenOnWrite(t *testing.T) {
	k := allowKernel()
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}
	orig, _, err := w.Put(context.Background(), []byte("private"), abi.ScopeAgent)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := w.Put(context.Background(), []byte("fleet"), abi.ScopeFleet); !errors.Is(err, ErrScopeWiden) {
		t.Fatalf("widen private->fleet error = %v, want ErrScopeWiden", err)
	}
	ref, ok := w.Ref()
	if !ok || ref.Digest != orig.Digest || ref.Scope != abi.ScopeAgent {
		t.Fatalf("widening write mutated ref: got %+v ok=%v, want original %+v", ref, ok, orig)
	}
	if _, _, err := Put(context.Background(), k, []byte("tenant"), abi.ScopeTenant); !errors.Is(err, ErrScopeWiden) {
		t.Fatalf("tenant Put error = %v, want ErrScopeWiden", err)
	}
}

func TestVDSOEpochBumpsOnWrites(t *testing.T) {
	k := allowKernel()
	v := vdso.New(8)
	w, err := New(k, abi.ScopeFleet, WithCoherence(v))
	if err != nil {
		t.Fatal(err)
	}
	before := v.WorldVersion()
	if _, _, err := w.Put(context.Background(), []byte("1"), abi.ScopeFleet); err != nil {
		t.Fatal(err)
	}
	afterPut := v.WorldVersion()
	if afterPut != before+1 {
		t.Fatalf("WorldVersion after Put = %d, want %d", afterPut, before+1)
	}
	if _, _, err := w.Accumulate(context.Background(), Sum, []byte("1")); err != nil {
		t.Fatal(err)
	}
	if got := v.WorldVersion(); got != afterPut+1 {
		t.Fatalf("WorldVersion after Accumulate = %d, want %d", got, afterPut+1)
	}
}

func TestDeniedPutDoesNotWrite(t *testing.T) {
	k := newFakeKernel(abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"})
	w, err := New(k, abi.ScopeFleet, WithCoherence(nil))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := w.Put(context.Background(), []byte("blocked"), abi.ScopeFleet); !errors.Is(err, ErrDenied) {
		t.Fatalf("Put denied error = %v, want ErrDenied", err)
	}
	if _, ok := w.Ref(); ok {
		t.Fatal("denied Put stored a Ref")
	}
}
