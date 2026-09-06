package kernel

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// benchNoopEngine is a lightweight, zero-allocation completion engine for benchmark harnesses.
type benchNoopEngine struct{}

func (benchNoopEngine) Complete(_ context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}, nil
}
func (benchNoopEngine) Caps() []abi.Capability { return nil }

// =============================================================================
// 1. Benchmark Session Creation
// =============================================================================

func BenchmarkSessionCreation_Default(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k := New("")
		if k == nil {
			b.Fatal("nil kernel")
		}
	}
}

func BenchmarkSessionCreation_WithEngine(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k := New("bench-engine")
		if k == nil {
			b.Fatal("nil kernel")
		}
	}
}

func BenchmarkSessionCreation_WithAdjudicators(b *testing.B) {
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k := New("bench-engine", WithAdjudicators(chain))
		if k == nil {
			b.Fatal("nil kernel")
		}
	}
}

func BenchmarkSessionCreation_Parallel(b *testing.B) {
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			k := New("bench-engine", WithAdjudicators(chain))
			if k == nil {
				b.Fatal("nil kernel")
			}
		}
	})
}

// =============================================================================
// 2. Benchmark Step Execution
// =============================================================================

func BenchmarkStepExecution_SubmitReap(b *testing.B) {
	setup()
	const engineID = "bench-step-submit-reap"
	abi.RegisterEngine(engineID, benchNoopEngine{})
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}
	k := New(engineID, WithAdjudicators(chain))
	ctx := context.Background()
	tc := call("tool_step", `{"param":1}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc.SeqNo = 0
		h, v := k.Submit(ctx, tc)
		if v.Kind != abi.VerdictAllow {
			b.Fatalf("verdict = %v, want allow", v.Kind)
		}
		r, err := k.Reap(ctx, h)
		if err != nil || r.Status != abi.StatusOK {
			b.Fatalf("reap failed: err=%v, r=%v", err, r)
		}
	}
}

func BenchmarkStepExecution_VDSOHit(b *testing.B) {
	setup()
	abi.RegisterFastPath(1, fakeFP{hit: true})
	k := New("")
	ctx := context.Background()
	tc := call("tool_vdso", `{"query":"cached"}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc.SeqNo = 0
		h, v := k.Submit(ctx, tc)
		if v.Kind != abi.VerdictAllow {
			b.Fatalf("verdict = %v, want allow", v.Kind)
		}
		r, err := k.Reap(ctx, h)
		if err != nil || r == nil || r.Status != abi.StatusOK {
			b.Fatalf("reap failed: err=%v, r=%v", err, r)
		}
	}
}

func BenchmarkStepExecution_MultiStep(b *testing.B) {
	setup()
	const engineID = "bench-multistep-engine"
	abi.RegisterEngine(engineID, benchNoopEngine{})
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k := New(engineID, WithAdjudicators(chain))
		for step := 0; step < 5; step++ {
			tc := call("step_tool", `{"step":true}`)
			h, v := k.Submit(ctx, tc)
			if v.Kind != abi.VerdictAllow {
				b.Fatalf("step %d verdict = %v", step, v.Kind)
			}
			r, err := k.Reap(ctx, h)
			if err != nil || r.Status != abi.StatusOK {
				b.Fatalf("step %d reap failed: err=%v", step, err)
			}
		}
	}
}

func BenchmarkStepExecution_BatchReapAll(b *testing.B) {
	setup()
	const engineID = "bench-batch-reapall"
	abi.RegisterEngine(engineID, benchNoopEngine{})
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}
	k := New(engineID, WithAdjudicators(chain))
	ctx := context.Background()
	const batchSize = 8

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handles := make([]abi.SubmissionHandle, batchSize)
		for j := 0; j < batchSize; j++ {
			tc := call("batch_tool", `{"idx":1}`)
			h, _ := k.Submit(ctx, tc)
			handles[j] = h
		}
		results, err := k.ReapAll(ctx, handles)
		if err != nil || len(results) != batchSize {
			b.Fatalf("ReapAll failed: %v", err)
		}
	}
}

func BenchmarkStepExecution_ReapAny(b *testing.B) {
	setup()
	const engineID = "bench-reapany-engine"
	abi.RegisterEngine(engineID, benchNoopEngine{})
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}
	k := New(engineID, WithAdjudicators(chain))
	ctx := context.Background()
	const batchSize = 4

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handles := make([]abi.SubmissionHandle, batchSize)
		for j := 0; j < batchSize; j++ {
			tc := call("tool", `{"idx":1}`)
			h, _ := k.Submit(ctx, tc)
			handles[j] = h
		}
		for len(handles) > 0 {
			h, r, err := k.ReapAny(ctx, handles)
			if err != nil || r == nil {
				b.Fatalf("ReapAny failed: %v", err)
			}
			for idx, item := range handles {
				if item.Seq == h.Seq {
					handles = append(handles[:idx], handles[idx+1:]...)
					break
				}
			}
		}
	}
}

// =============================================================================
// 3. Benchmark Adjudication
// =============================================================================

func BenchmarkAdjudication_Decide_Allow(b *testing.B) {
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}
	k := New("", WithAdjudicators(chain))
	ctx := context.Background()
	tc := call("tool_allow", `{"action":"read"}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := k.Decide(ctx, tc)
		if v.Kind != abi.VerdictAllow {
			b.Fatalf("got %v, want allow", v.Kind)
		}
	}
}

func BenchmarkAdjudication_Decide_Deny(b *testing.B) {
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock}}}
	k := New("", WithAdjudicators(chain))
	ctx := context.Background()
	tc := call("tool_deny", `{"action":"shell"}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := k.Decide(ctx, tc)
		if v.Kind != abi.VerdictDeny {
			b.Fatalf("got %v, want deny", v.Kind)
		}
	}
}

func BenchmarkAdjudication_Decide_Transform(b *testing.B) {
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{
		Kind:    abi.VerdictTransform,
		Payload: abi.TransformPayload{NewArgs: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"transformed":true}`)}},
	}}}
	k := New("", WithAdjudicators(chain))
	ctx := context.Background()
	tc := call("tool_transform", `{"orig":true}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := k.Decide(ctx, tc)
		if v.Kind != abi.VerdictTransform {
			b.Fatalf("got %v, want transform", v.Kind)
		}
	}
}

func BenchmarkAdjudication_Fold_MultiRung(b *testing.B) {
	chain := []abi.Adjudicator{
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictAllow, By: "allow-rung"}},
		fakeAdj{abi.Verdict{Kind: abi.VerdictDefer}},
	}
	ctx := context.Background()
	tc := call("tool_fold", `{}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := Fold(ctx, chain, tc)
		if v.Kind != abi.VerdictAllow {
			b.Fatalf("got %v, want allow", v.Kind)
		}
	}
}

func BenchmarkAdjudication_BatchDecide(b *testing.B) {
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}
	k := New("", WithAdjudicators(chain))
	ctx := context.Background()
	const batchSize = 16
	calls := make([]*abi.ToolCall, batchSize)
	for j := 0; j < batchSize; j++ {
		calls[j] = call("batch_call", `{"i":0}`)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		verdicts := k.BatchDecide(ctx, calls)
		if len(verdicts) != batchSize {
			b.Fatalf("BatchDecide got %d verdicts", len(verdicts))
		}
	}
}

// =============================================================================
// 4. Benchmark Syscall Admission
// =============================================================================

func BenchmarkSyscallAdmission_Syscall_Sync(b *testing.B) {
	setup()
	const engineID = "bench-syscall-sync"
	abi.RegisterEngine(engineID, benchNoopEngine{})
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}
	k := New(engineID, WithAdjudicators(chain))
	ctx := context.Background()
	tc := call("tool_syscall", `{"exec":true}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc.SeqNo = 0
		r, v := k.Syscall(ctx, tc)
		if v.Kind != abi.VerdictAllow || r.Status != abi.StatusOK {
			b.Fatalf("Syscall failed: v=%v, r=%v", v, r)
		}
	}
}

func BenchmarkSyscallAdmission_Syscall_Parallel(b *testing.B) {
	setup()
	const engineID = "bench-syscall-parallel"
	abi.RegisterEngine(engineID, benchNoopEngine{})
	chain := []abi.Adjudicator{fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}}}
	k := New(engineID, WithAdjudicators(chain))
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			tc := call("tool_parallel", `{"exec":true}`)
			r, v := k.Syscall(ctx, tc)
			if v.Kind != abi.VerdictAllow || r.Status != abi.StatusOK {
				b.Fatalf("Syscall failed: v=%v, r=%v", v, r)
			}
		}
	})
}

func BenchmarkSyscallAdmission_AdmitResult_Allow(b *testing.B) {
	setup()
	abi.RegisterResultAdmitter(0, verdictAdmitter{v: abi.Verdict{Kind: abi.VerdictAllow, By: "allow-admit"}})
	k := New("")
	ctx := context.Background()
	tc := call("tool_admit", `{}`)
	r := &abi.Result{Call: tc, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := k.AdmitResult(ctx, tc, r)
		if v.Kind != abi.VerdictAllow {
			b.Fatalf("got %v, want allow", v.Kind)
		}
	}
}

func BenchmarkSyscallAdmission_AdmitResult_Quarantine(b *testing.B) {
	setup()
	abi.RegisterResultAdmitter(0, quarantineAdmitter{})
	k := New("")
	ctx := context.Background()
	tc := call("tool_quarantine", `{}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := &abi.Result{Call: tc, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted}
		v := k.AdmitResult(ctx, tc, r)
		if v.Kind != abi.VerdictQuarantine {
			b.Fatalf("got %v, want quarantine", v.Kind)
		}
	}
}

func BenchmarkSyscallAdmission_AdmitResult_Transform(b *testing.B) {
	setup()
	transformedArgs := abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"sanitized":true}`)}
	abi.RegisterResultAdmitter(0, verdictAdmitter{v: abi.Verdict{
		Kind:    abi.VerdictTransform,
		By:      "transform-admit",
		Payload: abi.TransformPayload{NewArgs: transformedArgs},
	}})
	k := New("")
	ctx := context.Background()
	tc := call("tool_transform", `{}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := &abi.Result{Call: tc, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted}
		v := k.AdmitResult(ctx, tc, r)
		if v.Kind != abi.VerdictTransform {
			b.Fatalf("got %v, want transform", v.Kind)
		}
	}
}

func BenchmarkSyscallAdmission_AdmitResult_Deny(b *testing.B) {
	setup()
	abi.RegisterResultAdmitter(0, verdictAdmitter{v: abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "deny-admit"}})
	k := New("")
	ctx := context.Background()
	tc := call("tool_deny", `{}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := &abi.Result{Call: tc, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted}
		v := k.AdmitResult(ctx, tc, r)
		if v.Kind != abi.VerdictDeny {
			b.Fatalf("got %v, want deny", v.Kind)
		}
	}
}
