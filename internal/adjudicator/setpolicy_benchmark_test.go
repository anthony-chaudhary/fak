package adjudicator

import (
	"context"
	"fmt"
	"regexp"
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
func TestSetPolicyLargeIndexDoesNotBlockReadersForBuildDuration(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"probe": true}})
	large := manyPreds(10_000)
	large.Allow["probe"] = true
	call := &abi.ToolCall{Tool: "probe", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
	started := make(chan struct{})
	done := make(chan struct{})
	go func() {
		close(started)
		a.SetPolicy(large)
		close(done)
	}()
	<-started

	deadline := time.Now().Add(2 * time.Second)
	max := time.Duration(0)
	for {
		start := time.Now()
		if v := a.Adjudicate(context.Background(), call); v.Kind != abi.VerdictAllow {
			t.Fatalf("probe verdict=%+v", v)
		}
		if d := time.Since(start); d > max {
			max = d
		}
		select {
		case <-done:
			if max > 2*time.Millisecond {
				t.Fatalf("reader blocked %s while index built outside lock", max)
			}
			return
		default:
			if time.Now().After(deadline) {
				t.Fatal("SetPolicy did not finish")
			}
		}
	}
}
