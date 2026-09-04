package tracesink

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

func TestTraceSinkWriteAndFlush(t *testing.T) {
	sink := NewTraceSink(Options{Clock: fixedClock()})
	ev := abi.Event{
		Kind: abi.EvSubmit,
		Call: &abi.ToolCall{
			Tool: "read_file",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"main.go"}`)},
		},
	}

	for i := 0; i < 10; i++ {
		sink.Emit(ev)
	}

	tr := sink.Trace()
	if len(tr.Calls) != 10 {
		t.Fatalf("calls = %d, want 10", len(tr.Calls))
	}
	if !sink.Complete() {
		t.Fatal("expected sink to be complete")
	}
}

func BenchmarkTraceSinkWrite(b *testing.B) {
	ev := abi.Event{
		Kind: abi.EvSubmit,
		Call: &abi.ToolCall{
			Tool: "read_file",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"main.go"}`)},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	sink := NewTraceSink(Options{Clock: fixedClock()})
	for i := 0; i < b.N; i++ {
		sink.Emit(ev)
		if (i & 0x3f) == 0x3f {
			_ = sink.Trace()
			sink = NewTraceSink(Options{Clock: fixedClock()})
		}
	}
	_ = sink.Trace()
}

func BenchmarkTraceSinkFlush(b *testing.B) {
	sink := NewTraceSink(Options{Clock: fixedClock()})
	ev := abi.Event{
		Kind: abi.EvSubmit,
		Call: &abi.ToolCall{
			Tool: "calculate",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"a":1,"b":2}`)},
		},
	}
	for i := 0; i < 100; i++ {
		sink.Emit(ev)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sink.Trace()
	}
}

func BenchmarkTraceSinkEgressFloor(b *testing.B) {
	ledger := ifc.NewLedger()
	ledger.Raise("bench-trace", abi.TaintTainted)
	sink := NewTraceSink(Options{Ledger: ledger, Clock: fixedClock()})
	ev := abi.Event{
		Kind: abi.EvSubmit,
		Call: &abi.ToolCall{
			TraceID: "bench-trace",
			Tool:    "send_webhook",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":"https://example.com","data":"secret"}`)},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.Emit(ev)
	}
	_ = sink.Trace()
}
