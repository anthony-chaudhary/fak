package microagent

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

var (
	benchCompSink    *agent.Completion
	benchReceiptSink WorkerReceipt
	benchBytesSink   []byte
	benchTaskSink    CoordinatorTask
	benchResultSink  ToolResult
	benchTurnSink    TenantTask
	benchChildSink   Task
)

type benchGateway struct{}

func (benchGateway) Model() string { return "bench-model" }

func (benchGateway) Complete(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{
		Message: agent.Message{
			Role:    agent.RoleAssistant,
			Content: "step completed",
		},
		Usage: agent.Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}, nil
}

type benchAgent struct {
	turns int
	took  int
	done  chan struct{}
}

func (a *benchAgent) Step(ctx context.Context, gw Gateway) (bool, error) {
	a.took++
	if gw != nil {
		_, _ = gw.Complete(ctx, nil, nil)
	}
	if a.done != nil {
		close(a.done)
	}
	return true, nil
}

type benchFloor struct{}

func (benchFloor) Decide(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "bench-floor"}
}

// BenchmarkHostAgentLifecycle measures end-to-end microagent execution in Host:
// Spawn admission, worker Step execution through shared Gateway, audit event
// sink recording, session table state transitions (Running -> Stopped), and
// retire cleanup.
func BenchmarkHostAgentLifecycle(b *testing.B) {
	gw := benchGateway{}
	h, err := NewHost(gw, Config{
		Workers: 8,
		Queue:   512,
	})
	if err != nil {
		b.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		done := make(chan struct{})
		id := fmt.Sprintf("bench-agent-%d", i)
		if err := h.Spawn(id, &benchAgent{done: done}); err != nil {
			b.Fatalf("Spawn(%s): %v", id, err)
		}
		<-done
	}
	_ = h.Reap()
}

// BenchmarkHostConcurrentBatchLifecycle measures Host concurrent throughput
// when driving batches of concurrent microagents across worker Step goroutines.
func BenchmarkHostConcurrentBatchLifecycle(b *testing.B) {
	gw := benchGateway{}
	h, err := NewHost(gw, Config{
		Workers: 16,
		Queue:   1024,
	})
	if err != nil {
		b.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	const batchSize = 16
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		doneChans := make([]chan struct{}, batchSize)
		for j := 0; j < batchSize; j++ {
			done := make(chan struct{})
			doneChans[j] = done
			id := fmt.Sprintf("batch-%d-%d", i, j)
			if err := h.Spawn(id, &benchAgent{done: done}); err != nil {
				b.Fatalf("Spawn(%s): %v", id, err)
			}
		}
		for j := 0; j < batchSize; j++ {
			<-doneChans[j]
		}
	}
	_ = h.Reap()
}

// BenchmarkSlotSchedulerAcquireRelease measures single-goroutine slot pool
// acquisition and release with priority heap ordering.
func BenchmarkSlotSchedulerAcquireRelease(b *testing.B) {
	sched := NewScheduler(8)
	defer sched.Close()
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rel, err := sched.Acquire(ctx, i%10)
		if err != nil {
			b.Fatalf("Acquire: %v", err)
		}
		rel()
	}
}

// BenchmarkSlotSchedulerParallel measures slot scheduler throughput under
// parallel multi-goroutine contention for K slots.
func BenchmarkSlotSchedulerParallel(b *testing.B) {
	sched := NewScheduler(16)
	defer sched.Close()
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		p := 0
		for pb.Next() {
			rel, err := sched.Acquire(ctx, p%5)
			if err != nil {
				return
			}
			rel()
			p++
		}
	})
}

// BenchmarkBudgetQueueAdmitRelease measures token budget queue admission and
// release with token accounting and in-order FIFO waiter management.
func BenchmarkBudgetQueueAdmitRelease(b *testing.B) {
	bq := NewBudgetQueueDepth(100000, 1024)
	defer bq.Close()
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rel, err := bq.Admit(ctx, 100, i%10)
		if err != nil {
			b.Fatalf("Admit: %v", err)
		}
		rel()
	}
}

// BenchmarkBudgetQueueParallel measures token budget queue throughput under
// parallel multi-goroutine reservation and release.
func BenchmarkBudgetQueueParallel(b *testing.B) {
	bq := NewBudgetQueueDepth(1000000, 4096)
	defer bq.Close()
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		p := 0
		for pb.Next() {
			rel, err := bq.Admit(ctx, 50, p%5)
			if err != nil {
				return
			}
			rel()
			p++
		}
	})
}

// BenchmarkTenantQueueSchedule measures multi-tenant fair scheduling,
// submitting tasks across weighted envelopes and computing Next dispatch decisions.
func BenchmarkTenantQueueSchedule(b *testing.B) {
	envelopes := []TenantEnvelope{
		{Tenant: "t1", Weight: 10, MaxQueued: 100000, MaxConcurrent: 10},
		{Tenant: "t2", Weight: 5, MaxQueued: 100000, MaxConcurrent: 10},
		{Tenant: "t3", Weight: 1, MaxQueued: 100000, MaxConcurrent: 10},
	}
	tq, err := NewTenantQueue(envelopes)
	if err != nil {
		b.Fatalf("NewTenantQueue: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	now := time.Now()
	for i := 0; i < b.N; i++ {
		tenant := "t1"
		if i%3 == 1 {
			tenant = "t2"
		} else if i%3 == 2 {
			tenant = "t3"
		}
		task := TenantTask{
			ID:          fmt.Sprintf("task-%d", i),
			Tenant:      tenant,
			CostMicros:  10,
			Interactive: i%2 == 0,
		}
		if err := tq.Submit(task); err != nil {
			b.Fatalf("Submit: %v", err)
		}
		t, ok := tq.Next(now)
		if !ok {
			b.Fatal("Next returned false")
		}
		benchTurnSink = t
	}
}

// BenchmarkSessionGatewayComplete measures direct-call session gateway
// adjudication: session.Table.Decide per turn, model invocation, and DebitUsage.
func BenchmarkSessionGatewayComplete(b *testing.B) {
	gw := benchGateway{}
	tbl := session.NewTable()
	sgw := NewSessionGateway(gw, tbl)
	ctx := WithTrace(context.Background(), "")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		comp, err := sgw.Complete(ctx, nil, nil)
		if err != nil {
			b.Fatalf("Complete: %v", err)
		}
		benchCompSink = comp
	}
}

// BenchmarkCoordinatorTaskRegistration measures coordinator atomic S0/S1
// task admission, validating bounds (1 deliverable, 1-3 files, 1 witness command)
// and checking high-risk architectural boundaries.
func BenchmarkCoordinatorTaskRegistration(b *testing.B) {
	c := NewCoordinator(CoordinatorConfig{})
	task := CoordinatorTask{
		ID:             "task-s0-1",
		Deliverable:    "Add microagent benchmark tests",
		TargetFiles:    []string{"internal/microagent/benchmark_test.go"},
		WitnessCommand: "go test -v ./internal/microagent",
		Subsystem:      "microagent",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := task
		t.ID = fmt.Sprintf("task-%d", i)
		if err := c.RegisterTask(t); err != nil {
			b.Fatalf("RegisterTask: %v", err)
		}
		benchTaskSink = t
	}
}

// BenchmarkCoordinatorReceiptIngestion measures coordinator worker receipt
// validation, compact serialization, context token estimation, and context updates.
func BenchmarkCoordinatorReceiptIngestion(b *testing.B) {
	c := NewCoordinator(CoordinatorConfig{TokenCap: 1000000})
	task := CoordinatorTask{
		ID:             "task-bench",
		Deliverable:    "Deliverable benchmark",
		TargetFiles:    []string{"internal/microagent/microagent.go"},
		WitnessCommand: "go test -v ./internal/microagent",
	}
	if err := c.RegisterTask(task); err != nil {
		b.Fatalf("RegisterTask: %v", err)
	}

	receipt := WorkerReceipt{
		TaskID:          "task-bench",
		Status:          StatusCompleted,
		TouchedFiles:    []string{"internal/microagent/microagent.go"},
		WitnessCommand:  "go test -v ./internal/microagent",
		WitnessExitCode: 0,
		TokensUsed:      450,
		Summary:         "Successfully executed microagent production benchmark",
	}
	rawTranscript := "=== RUN TestBenchmark\nPASS\nok internal/microagent 0.05s\n"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := c.IngestReceipt(receipt, rawTranscript); err != nil {
			b.Fatalf("IngestReceipt: %v", err)
		}
	}
}

// BenchmarkWorkerReceiptValidateAndCompactJSON measures WorkerReceipt validation
// and compact JSON marshaling for coordinator inclusion.
func BenchmarkWorkerReceiptValidateAndCompactJSON(b *testing.B) {
	receipt := WorkerReceipt{
		TaskID:          "task-bench-42",
		Status:          StatusCompleted,
		TouchedFiles:    []string{"internal/microagent/coordinator.go", "internal/microagent/coordinator_test.go"},
		WitnessCommand:  "go test -v ./internal/microagent",
		WitnessExitCode: 0,
		GitSHA:          "abcdef0123456789",
		TokensUsed:      320,
		Summary:         "Bounded worker completed S0 leaf with witness",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := receipt.Validate(); err != nil {
			b.Fatalf("Validate: %v", err)
		}
		raw := receipt.CompactJSON()
		benchBytesSink = []byte(raw)
	}
}

// BenchmarkCompletionReceiptLifecycle measures CompletionReceipt validation,
// size bounding, JSON encoding, and independent verification admission via FoldVerifiedReceipt.
func BenchmarkCompletionReceiptLifecycle(b *testing.B) {
	ctx := context.Background()
	rootCtx := NewContext(100000)
	receipt := CompletionReceipt{
		Schema:  CompletionReceiptSchema,
		Child:   "child-agent-42",
		Summary: "Autonomous worker completed verified subtask within bounded turns",
		Provenance: []EvidenceRef{
			{Kind: "journal-row", Ref: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
		},
		Allowed: 1,
		Denied:  0,
		Errored: 0,
	}
	verifier := ReceiptVerifierFunc(func(context.Context, CompletionReceipt) ReceiptReview {
		return AcceptReceipt()
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := FoldVerifiedReceipt(ctx, rootCtx, receipt, verifier, nil); err != nil {
			b.Fatalf("FoldVerifiedReceipt: %v", err)
		}
	}
}

// BenchmarkToolExecFloorAdjudication measures tool execution through the in-process
// KernelFloor adjudication gate and GoroutineBackend dispatch.
func BenchmarkToolExecFloorAdjudication(b *testing.B) {
	gb := NewGoroutineBackend()
	if err := gb.Register("echo_tool", func(ctx context.Context, act ToolAction) (ToolResult, error) {
		return ToolResult{ExitCode: 0, Stdout: act.Stdin}, nil
	}); err != nil {
		b.Fatalf("Register: %v", err)
	}

	te, err := NewToolExecBackend(benchFloor{}, gb)
	if err != nil {
		b.Fatalf("NewToolExecBackend: %v", err)
	}

	ctx := context.Background()
	act := ToolAction{
		Tool:  "echo_tool",
		Stdin: []byte(`{"message":"hello"}`),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := te.Run(ctx, act)
		if err != nil {
			b.Fatalf("Run: %v", err)
		}
		benchResultSink = res
	}
}

// BenchmarkContextAppendWithEviction measures linear Context append, token estimation,
// and whole-message FIFO eviction when exceeding the hard token ceiling.
func BenchmarkContextAppendWithEviction(b *testing.B) {
	c := NewContext(500)
	msgContent := "user turn message payload for microagent context token estimation and eviction"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Append("user", msgContent)
	}
}

// BenchmarkManagedContextCompaction measures ManagedContext turn append with
// stale turn compaction and durable artifact pointer retention.
func BenchmarkManagedContextCompaction(b *testing.B) {
	mc := NewManagedContext(500)
	ptr := ArtifactPointer{Kind: "git-sha", URI: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mc.Append("user", "step content with pointer info", ptr)
	}
}

// BenchmarkDecodeChildContractJSON measures strict JSON decoding and validation
// of child Task envelopes.
func BenchmarkDecodeChildContractJSON(b *testing.B) {
	task := Task{
		Goal:         "Implement microagent benchmark functions",
		ArtifactRefs: []string{"file://internal/microagent/microagent.go"},
		Authority:    []string{"file_read", "file_write"},
		Budget:       TaskBudget{MaxTurns: 3},
	}
	raw, err := json.Marshal(task)
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decoded, err := DecodeTaskJSON(raw)
		if err != nil {
			b.Fatalf("DecodeTaskJSON: %v", err)
		}
		benchChildSink = decoded
	}
}
