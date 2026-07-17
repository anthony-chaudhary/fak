package adjudicator

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"regexp"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// BenchmarkSetPolicyScaling measures the complete policy-swap path. Its ns/op
// bounds the write-lock stall window paid by concurrent Adjudicate readers.
func BenchmarkSetPolicyScaling(b *testing.B) {
	for _, predicates := range []int{0, 100, 2_000, 10_000} {
		b.Run(fmt.Sprintf("predicates=%d", predicates), func(b *testing.B) {
			p := manyPreds(predicates)
			a := New(Policy{})
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				a.SetPolicy(p)
			}
		})
	}
}

func TestSetPolicySwapsPolicyAndPredicateIndexAtomically(t *testing.T) {
	byPredicate := Policy{
		Allow:         map[string]bool{"probe": true},
		ArgPredicates: []ArgPredicate{{Tool: "probe", Arg: "value", Kind: ArgDenyRegex, Re: regexp.MustCompile("^always$"), Reason: abi.ReasonPolicyBlock}},
	}
	byName := Policy{
		Deny:          map[string]abi.ReasonCode{"probe": abi.ReasonPolicyBlock},
		ArgPredicates: []ArgPredicate{{Tool: "probe", Arg: "value", Kind: ArgDenyRegex, Re: regexp.MustCompile("^never$"), Reason: abi.ReasonPolicyBlock}},
	}
	a := New(byPredicate)
	call := &abi.ToolCall{Tool: "probe", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"value":"always"}`)}}
	var stop atomic.Bool
	var bad atomic.Uint64
	done := make(chan struct{})
	go func() {
		defer close(done)
		for !stop.Load() {
			if v := a.Adjudicate(context.Background(), call); v.Kind != abi.VerdictDeny {
				bad.Add(1) // byPredicate+byName-index is the otherwise-impossible torn ALLOW.
			}
		}
	}()
	for i := 0; i < 2_000; i++ {
		if i%2 == 0 {
			a.SetPolicy(byName)
		} else {
			a.SetPolicy(byPredicate)
		}
	}
	stop.Store(true)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("concurrent adjudicator did not stop")
	}
	if n := bad.Load(); n != 0 {
		t.Fatalf("observed %d torn policy/index verdicts", n)
	}
}
func TestSetPolicyBuildsLargeIndexBeforeAtomicSwap(t *testing.T) {
	// This is an architecture witness rather than a wall-clock assertion. A
	// reader's elapsed time includes scheduler delay, so a millisecond ceiling
	// flakes on loaded CI runners without detecting lock contention. The
	// performance invariant is exact: SetPolicy must finish the O(predicate-count)
	// index build before its single atomic state publication.
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate test source")
	}
	file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(filepath.Dir(source), "decide.go"), nil, 0)
	if err != nil {
		t.Fatalf("parse decide.go: %v", err)
	}

	var buildPos, storePos token.Pos
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "SetPolicy" || fn.Recv == nil {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch fun := call.Fun.(type) {
			case *ast.Ident:
				if fun.Name == "indexArgPredicates" {
					buildPos = call.Pos()
				}
			case *ast.SelectorExpr:
				if fun.Sel.Name == "Store" {
					storePos = call.Pos()
				}
			}
			return true
		})
	}
	if !buildPos.IsValid() || !storePos.IsValid() {
		t.Fatalf("SetPolicy architecture incomplete: index build=%v atomic store=%v", buildPos.IsValid(), storePos.IsValid())
	}
	if buildPos >= storePos {
		t.Fatal("SetPolicy must build the predicate index before publishing the immutable state")
	}
}

func BenchmarkAdjudicateParallel(b *testing.B) {
	a := New(reloadLatencyBenchmarkPolicy())
	call := &abi.ToolCall{Tool: "read", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"x":"safe"}`)}}
	ctx := context.Background()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			a.Adjudicate(ctx, call)
		}
	})
}

func BenchmarkNeverAdmitsParallel(b *testing.B) {
	a := New(reloadLatencyBenchmarkPolicy())
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			a.NeverAdmits("unknown_tool")
		}
	})
}

func reloadLatencyBenchmarkPolicy() Policy {
	return Policy{
		Posture:       PostureFailClosed,
		Allow:         map[string]bool{"read": true},
		ArgPredicates: []ArgPredicate{{Tool: "read", Arg: "x", Kind: ArgDenyRegex, Re: regexp.MustCompile("^blocked$"), Reason: abi.ReasonPolicyBlock}},
	}
}
