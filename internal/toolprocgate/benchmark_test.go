package toolprocgate

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

var (
	benchVerdictSink      abi.Verdict
	benchReuseVerdictSink ReuseVerdict
	benchGrantSink        SpawnGrant
	benchFSDecisionSink   FSDecision
	benchOutputSink       OutputAdmission
	benchContainmentSink  ContainmentDecision
	benchConsoleFaultSink ConsoleFaultClass
	benchSink             any
)

func BenchmarkGateAdmitClean(b *testing.B) {
	Reset()
	b.Cleanup(Reset)
	g := Gate{}
	ctx := context.Background()
	c := &abi.ToolCall{Tool: "bg_dump", TraceID: "t-bench-clean"}
	r := &abi.Result{
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"status":"ok"}`), Len: 15},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := g.Admit(ctx, c, r)
		if v.Kind != abi.VerdictDefer {
			b.Fatalf("unexpected verdict kind: %v", v.Kind)
		}
		benchVerdictSink = v
	}
}

func BenchmarkGateAdmitQuarantine(b *testing.B) {
	Reset()
	b.Cleanup(Reset)
	Kill("t-bench-quarantine", toolproc.ReasonToolDeadlineExceededName)
	g := Gate{}
	ctx := context.Background()
	c := &abi.ToolCall{Tool: "bg_dump", TraceID: "t-bench-quarantine"}
	rawBody := []byte(`{"stdout":"sensitive output from revoked tool execution"}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := &abi.Result{
			Status:  abi.StatusOK,
			Payload: abi.Ref{Kind: abi.RefInline, Inline: rawBody, Len: int64(len(rawBody))},
		}
		v := g.Admit(ctx, c, r)
		if v.Kind != abi.VerdictQuarantine {
			b.Fatalf("unexpected verdict kind: %v", v.Kind)
		}
		benchVerdictSink = v
	}
}

func BenchmarkKillTable(b *testing.B) {
	Reset()
	b.Cleanup(Reset)
	keyRange := 2 * maxKills
	ids := make([]string, keyRange)
	for i := 0; i < keyRange; i++ {
		ids[i] = fmt.Sprintf("call-%d", i)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := ids[i%keyRange]
		Kill(id, toolproc.ReasonToolOrphanedName)
		reason, ok := KilledReason(id)
		if !ok {
			b.Fatal("expected killed reason in table")
		}
		benchSink = reason
	}
}

func BenchmarkSpawnBrokerAdmit(b *testing.B) {
	broker := NewSpawnBroker()
	attempt := SpawnAttempt{
		AgentRunID:   "agent-run-bench",
		ParentRunID:  "agent-parent-bench",
		ToolCallID:   "call-bench-1",
		PolicyDigest: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		Argv:         []string{"tool", "--flag=value"},
		Env:          []EnvVar{{Name: "FAK_TEST", Value: "1"}},
		CWD:          b.TempDir(),
		Backend:      "local",
		Envelope: CapabilityEnvelope{
			Capabilities: []abi.Capability{CapAgentRunSpawn},
			LaneTree:     []string{"internal/toolprocgate/**"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grant, err := broker.Admit(attempt)
		if err != nil {
			b.Fatalf("unexpected spawn denial: %v", err)
		}
		if grant.GrantID == "" {
			b.Fatal("expected non-empty grant id")
		}
		benchGrantSink = grant
		broker.audits = broker.audits[:0]
		broker.leaks = broker.leaks[:0]
	}
}

func BenchmarkLaneFloorAdmitTouch(b *testing.B) {
	grant := SpawnGrant{
		Envelope: CapabilityEnvelope{
			LaneTree: []string{"internal/toolprocgate/**", "cmd/fak/**"},
		},
	}
	floor := grant.LaneFloor()
	touch := FSTouch{
		ToolCallID: "call-bench",
		Path:       "internal/toolprocgate/lanefloor.go",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dec := floor.AdmitTouch(touch)
		if !dec.Allowed() {
			b.Fatalf("expected allowed touch, got %s", dec.Reason)
		}
		benchFSDecisionSink = dec
	}
}

func BenchmarkLaneFloorAdmitTouchDenied(b *testing.B) {
	grant := SpawnGrant{
		Envelope: CapabilityEnvelope{
			LaneTree: []string{"internal/toolprocgate/**"},
		},
	}
	floor := grant.LaneFloor()
	touch := FSTouch{
		ToolCallID: "call-bench",
		Path:       "internal/secretgate/secretgate.go",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dec := floor.AdmitTouch(touch)
		if dec.Allowed() {
			b.Fatal("expected denied touch")
		}
		benchFSDecisionSink = dec
	}
}

func BenchmarkAdmitChildOutput(b *testing.B) {
	ctx := context.Background()
	in := ChildOutput{
		AgentRunID:   "agent-run-1",
		ToolCallID:   "call-1",
		TraceID:      "trace-1",
		PolicyDigest: "sha256:abcd",
		Backend:      "local",
		Channel:      ChannelStdout,
		Bytes:        []byte("standard worker output line for benchmarking admission\n"),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adm := AdmitChildOutput(ctx, in)
		if adm.Verdict.Kind != abi.VerdictDefer && adm.Verdict.Kind != abi.VerdictAllow {
			b.Fatalf("unexpected output verdict: %v", adm.Verdict.Kind)
		}
		benchOutputSink = adm
	}
}

func BenchmarkReuseArmServeHit(b *testing.B) {
	arm := NewReuseArm(toolproc.ArmedConfig{})
	body := []byte("cached response content for tool repetition")
	rec := toolproc.CallRecord{
		Tool:        "shell_command",
		Raw:         "cat SKILL.md",
		AtMS:        0,
		OutputBytes: int64(len(body)),
		Digest:      "d1",
	}
	_ = arm.Serve(context.Background(), nil, rec)
	if !arm.Offer(rec, body) {
		b.Fatal("offer must be retained")
	}

	call := &abi.ToolCall{TraceID: "call-repeat-bench"}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hit := arm.Serve(ctx, call, rec)
		if !hit.Served() {
			b.Fatal("expected served cache hit")
		}
		benchReuseVerdictSink = hit
	}
}

func BenchmarkDecideContainment(b *testing.B) {
	pol := DefaultContainmentPolicy()
	req := ContainmentRequest{
		Surface:       "conhost",
		LiveOnSurface: 1,
		NowMS:         1000000,
	}
	faults := []ConsoleFaultEvent{
		{
			Session: "session-1",
			Surface: "conhost",
			AtMS:    999000,
			Class:   ConsoleHostFailFast,
		},
		{
			Session: "session-2",
			Surface: "pty",
			AtMS:    998000,
			Class:   ConsolePipeLost,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dec := DecideContainment(pol, faults, req)
		if !dec.Admit {
			b.Fatalf("expected containment admit, got %s", dec.Verdict)
		}
		benchContainmentSink = dec
	}
}

func BenchmarkClassifyConsoleFault(b *testing.B) {
	errText := "System.Management.Automation.Host.HostException: No process is on the other end of the pipe."

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		class, ok := ClassifyConsoleFault(errText)
		if !ok || class != ConsoleHostFailFast {
			b.Fatalf("expected ConsoleHostFailFast, got %q, %v", class, ok)
		}
		benchConsoleFaultSink = class
	}
}

func TestBenchmarkKillTableEviction(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	keyRange := 2 * maxKills
	ids := make([]string, keyRange)
	for i := 0; i < keyRange; i++ {
		ids[i] = fmt.Sprintf("call-%d", i)
	}

	for i := 0; i < keyRange; i++ {
		Kill(ids[i], toolproc.ReasonToolOrphanedName)
	}

	// The first maxKills keys must have been evicted.
	for i := 0; i < maxKills; i++ {
		if _, ok := KilledReason(ids[i]); ok {
			t.Fatalf("expected key %s to be evicted after inserting %d keys into table of capacity %d", ids[i], keyRange, maxKills)
		}
	}

	// The last maxKills keys must remain in the table.
	for i := maxKills; i < keyRange; i++ {
		if _, ok := KilledReason(ids[i]); !ok {
			t.Fatalf("expected key %s to be present in table", ids[i])
		}
	}
}

func TestBenchmarkOperationsSanity(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	benchmarks := []struct {
		name string
		fn   func(b *testing.B)
	}{
		{"GateAdmitClean", BenchmarkGateAdmitClean},
		{"GateAdmitQuarantine", BenchmarkGateAdmitQuarantine},
		{"KillTable", BenchmarkKillTable},
		{"SpawnBrokerAdmit", BenchmarkSpawnBrokerAdmit},
		{"LaneFloorAdmitTouch", BenchmarkLaneFloorAdmitTouch},
		{"LaneFloorAdmitTouchDenied", BenchmarkLaneFloorAdmitTouchDenied},
		{"AdmitChildOutput", BenchmarkAdmitChildOutput},
		{"ReuseArmServeHit", BenchmarkReuseArmServeHit},
		{"DecideContainment", BenchmarkDecideContainment},
		{"ClassifyConsoleFault", BenchmarkClassifyConsoleFault},
	}

	for _, bm := range benchmarks {
		res := testing.Benchmark(bm.fn)
		if res.N == 0 {
			t.Errorf("benchmark %s did not run", bm.name)
		}
	}
}
