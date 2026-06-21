package abi

import (
	"context"
	"fmt"
	"testing"
)

// This file is the GROUND-TRUTH proof of the registry's scaling contract: the
// read accessors the kernel walks on every syscall must be O(1) and
// allocation-free no matter how many ideas are registered. TestRegistryReadsZeroAlloc
// is the durable guard (it FAILS if a future change reintroduces a per-call
// allocation or lock-copy on a read path); BenchmarkRegistryReadScaling lets a
// human see the per-call cost stay flat as the feature count grows 1 -> 1000.

// Minimal driver stubs — they do nothing; we are timing the registry machinery,
// not the drivers.
type benchAdj struct{ v Verdict }

func (b benchAdj) Adjudicate(context.Context, *ToolCall) Verdict { return b.v }
func (benchAdj) Caps() []Capability                              { return nil }

type benchEmitter struct{}

func (benchEmitter) Emit(Event) {}

type benchFP struct{}

func (benchFP) Lookup(context.Context, *ToolCall) (*Result, bool) { return nil, false }
func (benchFP) Caps() []Capability                                { return nil }

type benchRA struct{}

func (benchRA) Admit(context.Context, *ToolCall, *Result) Verdict {
	return Verdict{Kind: VerdictAllow}
}
func (benchRA) Caps() []Capability { return nil }

type benchEngine struct{}

func (benchEngine) Complete(context.Context, *ToolCall) (*Result, error) { return &Result{}, nil }
func (benchEngine) Caps() []Capability                                   { return nil }

// benchSink defeats dead-code elimination of the accessor calls.
var benchSink int

// registerN populates the registry with n of every kind of read-walked driver.
func registerN(n int) {
	ResetForTest()
	for i := 0; i < n; i++ {
		RegisterAdjudicator(i, benchAdj{Verdict{Kind: VerdictDefer}})
		RegisterEmitter(benchEmitter{})
		RegisterFastPath(i, benchFP{})
		RegisterResultAdmitter(i, benchRA{})
	}
	RegisterEngine("e", benchEngine{})
	RegisterCapability("cap")
}

// TestRegistryReadsZeroAlloc is the O(1)-read contract as a machine-checked test:
// with 256 drivers of every kind registered, EVERY read accessor the kernel walks
// on the hot path must perform ZERO allocations per call. The pre-snapshot
// implementation allocated a fresh slice (and took a mutex) on every call, so this
// test is what stops that regression from ever coming back.
func TestRegistryReadsZeroAlloc(t *testing.T) {
	registerN(256)
	defer ResetForTest()

	checks := []struct {
		name string
		f    func()
	}{
		{"Adjudicators", func() { benchSink += len(Adjudicators()) }},
		{"FastPaths", func() { benchSink += len(FastPaths()) }},
		{"ResultAdmitters", func() { benchSink += len(ResultAdmitters()) }},
		{"Emitters", func() { benchSink += len(Emitters()) }},
		{"Witnesses", func() { benchSink += len(Witnesses()) }},
		{"Stewards", func() { benchSink += len(Stewards()) }},
		{"FoldRank", func() { benchSink += FoldRank(VerdictDeny) }},
		{"Supported", func() {
			if Supported("cap") {
				benchSink++
			}
		}},
		{"Engine", func() {
			if Engine("") != nil {
				benchSink++
			}
		}},
		{"LookupOp", func() {
			if _, ok := LookupOp(0); ok {
				benchSink++
			}
		}},
	}
	for _, c := range checks {
		if a := testing.AllocsPerRun(200, c.f); a != 0 {
			t.Errorf("%s: %.2f allocs/op on the hot path; want 0 (the O(1) read contract)", c.name, a)
		}
	}
}

// BenchmarkRegistryReadScaling prints the per-call cost of reading the accessors
// the kernel walks, at 1/10/100/1000 registered drivers. With the atomic snapshot
// the ns/op and allocs/op are FLAT across N — adding the 1000th idea costs the
// read path nothing. (Pre-snapshot, ns/op and B/op grew linearly with N.)
func BenchmarkRegistryReadScaling(b *testing.B) {
	for _, n := range []int{1, 10, 100, 1000} {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			registerN(n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSink += len(Adjudicators()) + len(Emitters()) +
					len(FastPaths()) + len(ResultAdmitters())
			}
		})
	}
	ResetForTest()
}
