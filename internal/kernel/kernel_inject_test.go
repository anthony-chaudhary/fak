package kernel

import (
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestWithAdjudicatorsFoldsExplicitChain proves the per-kernel adjudicator injection
// (issue #500): a kernel built with WithAdjudicators folds the SUPPLIED chain, not the
// process-global registry. Here the global registry would ALLOW the call (its only rung
// is a blanket allow), but the injected chain DENIES it — so the injected kernel must
// deny while a default kernel on the same registry allows.
func TestWithAdjudicatorsFoldsExplicitChain(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}) // global: allow
	abi.RegisterEngine("", &countEngine{})

	injected := New("", WithAdjudicators([]abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock}},
	}))
	if _, v := injected.Syscall(context.Background(), call("x", "{}")); v.Kind != abi.VerdictDeny {
		t.Fatalf("injected kernel must fold its OWN deny chain, got %v", v.Kind)
	}

	// A default kernel on the SAME global registry is unaffected — it allows.
	def := New("")
	if _, v := def.Syscall(context.Background(), call("x", "{}")); v.Kind != abi.VerdictAllow {
		t.Fatalf("default kernel must read the global allow registry, got %v", v.Kind)
	}
}

// TestNewWithoutOptionReadsGlobalRegistry is the BACK-COMPAT proof: New("...") with no
// option behaves EXACTLY as before — it folds the process-global adjudicator registry.
// WithAdjudicators(nil) and WithAdjudicators(empty) are no-ops that fall back to the same
// global path, so an injection can never SILENTLY install an empty (deny-everything)
// policy. The verdict for every form must match the registry's verdict.
func TestNewWithoutOptionReadsGlobalRegistry(t *testing.T) {
	setup()
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("", &countEngine{})

	want := abi.VerdictAllow
	cases := map[string]*Kernel{
		"no-option":      New(""),
		"with-adj-nil":   New("", WithAdjudicators(nil)),
		"with-adj-empty": New("", WithAdjudicators([]abi.Adjudicator{})),
	}
	for name, k := range cases {
		if _, v := k.Syscall(context.Background(), call("x", "{}")); v.Kind != want {
			t.Errorf("%s: verdict=%v, want %v (must read the global registry unchanged)", name, v.Kind, want)
		}
	}
}

// TestInjectedChainPreservesToolScoping proves the injected path applies the SAME
// CallScope tool-scoping abi.AdjudicatorsFor applies to the global registry (via
// abi.ScopedFor): a rung scoped to "write" with a deny must route to a write call and
// must NOT leak to a read call, exactly as a registered scoped rung does. The injected
// chain is therefore verdict-equivalent to the same chain registered globally.
func TestInjectedChainPreservesToolScoping(t *testing.T) {
	setup()
	abi.RegisterEngine("", &countEngine{})
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}, // unconditional allow
		scopedAdj{
			v:     abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock},
			tools: []string{"write"},
		},
	}
	k := New("", WithAdjudicators(chain))

	if _, v := k.Syscall(context.Background(), call("write", "{}")); v.Kind != abi.VerdictDeny {
		t.Fatalf("write: injected scoped deny must route to its tool, got %v", v.Kind)
	}
	if _, v := k.Syscall(context.Background(), call("read", "{}")); v.Kind != abi.VerdictAllow {
		t.Fatalf("read: injected scoped deny must NOT leak to other tools, got %v", v.Kind)
	}
}

// TestConcurrentInjectedKernelsAreIndependent is the concurrency-isolation proof: two
// kernels carrying DIFFERENT injected policies (one allow, one deny) running the SAME
// call concurrently never see each other's verdict. This is the property that lets K
// policy-replay arms fan out across goroutines without colliding on a process-global
// monitor — run under -race it is also the data-race gate.
func TestConcurrentInjectedKernelsAreIndependent(t *testing.T) {
	setup()
	// The global registry would ALLOW (so a leak toward the global path is observable as
	// an unexpected allow on the deny kernel), but neither injected kernel reads it.
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	abi.RegisterEngine("", &countEngine{})

	allowK := New("", WithAdjudicators([]abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}))
	denyK := New("", WithAdjudicators([]abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock}}}))

	const iters = 200
	var wg sync.WaitGroup
	var allowBad, denyBad int64
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < iters; i++ {
			if _, v := allowK.Syscall(context.Background(), call("x", "{}")); v.Kind != abi.VerdictAllow {
				allowBad++
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < iters; i++ {
			if _, v := denyK.Syscall(context.Background(), call("x", "{}")); v.Kind != abi.VerdictDeny {
				denyBad++
			}
		}
	}()
	wg.Wait()
	if allowBad != 0 || denyBad != 0 {
		t.Fatalf("concurrent injected kernels collided: allow-arm wrong=%d deny-arm wrong=%d", allowBad, denyBad)
	}
}
