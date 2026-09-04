package harnessprotocol

import (
	"bytes"
	"context"
	"io"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

var (
	benchEnvelopeSink   harnesskit.Envelope
	benchEnvelopesSink  []harnesskit.Envelope
	benchProjectionSink Projection
	benchStringSink     string
	benchCursorSink     harnesskit.Cursor
)

func sampleEvents(n int) []harnesskit.Envelope {
	p := NewProducer("bench-sample-run", []byte("secret-key"))
	_, _ = p.Append(harnesskit.EventRunStarted, "c1", "", harnesskit.SensitivityPublic, harnesskit.RunPayload{Status: "running"})
	for i := 1; i < n-1; i++ {
		if i%3 == 0 {
			_, _ = p.Append(harnesskit.EventToolStarted, "c1", "c1", harnesskit.SensitivityPrivate, harnesskit.ToolPayload{CallID: "call-1", Name: "search", Status: "running"})
		} else if i%3 == 1 {
			_, _ = p.Append(harnesskit.EventToolCompleted, "c1", "c1", harnesskit.SensitivityPrivate, harnesskit.ToolPayload{CallID: "call-1", Name: "search", Status: "completed"})
		} else {
			_, _ = p.Append(harnesskit.EventMessageDelta, "c1", "c1", harnesskit.SensitivityPublic, harnesskit.MessagePayload{MessageID: "m1", Text: "chunk "})
		}
	}
	_, _ = p.Append(harnesskit.EventRunCompleted, "c1", "c1", harnesskit.SensitivityPublic, harnesskit.RunPayload{Status: "completed"})
	evs, _, _ := p.Resume(context.Background(), harnesskit.Cursor{}, uint32(n))
	return evs
}

func BenchmarkHarnessProtocol(b *testing.B) {
	secret := []byte("bench-hmac-secret-run-6789")
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		p := NewProducer("bench-run-pipeline", secret)

		_, _ = p.Append(harnesskit.EventRunStarted, "c0", "", harnesskit.SensitivityPublic, harnesskit.RunPayload{Status: "running"})
		_, _ = p.Append(harnesskit.EventMessageStarted, "c0", "c0", harnesskit.SensitivityPublic, harnesskit.MessagePayload{MessageID: "m1", Role: "assistant", Text: "starting"})
		_, _ = p.Append(harnesskit.EventMessageDelta, "c0", "c0", harnesskit.SensitivityPublic, harnesskit.MessagePayload{MessageID: "m1", Text: " work"})
		_, _ = p.Append(harnesskit.EventToolStarted, "c0", "c0", harnesskit.SensitivityPrivate, harnesskit.ToolPayload{CallID: "t1", Name: "eval", Status: "running"})
		_, _ = p.Append(harnesskit.EventToolCompleted, "c0", "c0", harnesskit.SensitivityPrivate, harnesskit.ToolPayload{CallID: "t1", Name: "eval", Status: "completed"})
		_, _ = p.Append(harnesskit.EventUsage, "c0", "c0", harnesskit.SensitivityPublic, harnesskit.UsagePayload{InputTokens: 100, OutputTokens: 50, CostMicros: 200})
		last, err := p.Append(harnesskit.EventRunCompleted, "c0", "c0", harnesskit.SensitivityPublic, harnesskit.RunPayload{Status: "completed"})
		if err != nil {
			b.Fatal(err)
		}
		benchEnvelopeSink = last

		c := p.Cursor()
		benchCursorSink = c

		replayed, nextCursor, err := p.Resume(ctx, harnesskit.Cursor{}, 7)
		if err != nil {
			b.Fatalf("Resume failed: %v", err)
		}
		if len(replayed) != 7 || nextCursor.Sequence != 7 {
			b.Fatalf("unexpected resume output: count=%d seq=%d", len(replayed), nextCursor.Sequence)
		}
		benchEnvelopesSink = replayed

		redacted := Redact(replayed, false)
		if len(redacted) != 7 {
			b.Fatalf("unexpected redacted count: %d", len(redacted))
		}

		proj, err := Project(replayed)
		if err != nil {
			b.Fatalf("Project failed: %v", err)
		}
		benchProjectionSink = proj

		cli := CLIText(proj)
		tui := TUIText(proj)
		benchStringSink = cli + tui
	}
}

func BenchmarkProducerAppend(b *testing.B) {
	p := NewProducer("bench-append-run", []byte("secret"))
	payload := harnesskit.MessagePayload{MessageID: "m1", Role: "assistant", Text: "stream token"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e, err := p.Append(harnesskit.EventMessageDelta, "c1", "c1", harnesskit.SensitivityPublic, payload)
		if err != nil {
			b.Fatal(err)
		}
		benchEnvelopeSink = e
	}
}

func BenchmarkProducerResume(b *testing.B) {
	p := NewProducer("bench-resume-run", []byte("secret"))
	for i := 0; i < 64; i++ {
		_, _ = p.Append(harnesskit.EventMessageDelta, "c1", "c1", harnesskit.SensitivityPublic, harnesskit.MessagePayload{MessageID: "m1", Text: "text"})
	}
	ctx := context.Background()
	cursor := harnesskit.Cursor{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evs, cur, err := p.Resume(ctx, cursor, 16)
		if err != nil {
			b.Fatal(err)
		}
		benchEnvelopesSink = evs
		benchCursorSink = cur
	}
}

func BenchmarkProject(b *testing.B) {
	events := sampleEvents(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proj, err := Project(events)
		if err != nil {
			b.Fatal(err)
		}
		benchProjectionSink = proj
	}
}

func BenchmarkRedact(b *testing.B) {
	events := sampleEvents(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchEnvelopesSink = Redact(events, false)
	}
}

func BenchmarkCLIText(b *testing.B) {
	events := sampleEvents(16)
	proj, _ := Project(events)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = CLIText(proj)
	}
}

func BenchmarkTUIText(b *testing.B) {
	events := sampleEvents(16)
	proj, _ := Project(events)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = TUIText(proj)
	}
}

func BenchmarkWriteJSONL(b *testing.B) {
	events := sampleEvents(16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := WriteJSONL(io.Discard, events); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReadJSONL(b *testing.B) {
	events := sampleEvents(16)
	var buf bytes.Buffer
	if err := WriteJSONL(&buf, events); err != nil {
		b.Fatal(err)
	}
	data := buf.Bytes()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := ReadJSONL(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
		benchEnvelopesSink = out
	}
}

func TestBenchmarkHarnessProtocolExecution(t *testing.T) {
	res := testing.Benchmark(BenchmarkHarnessProtocol)
	if res.N <= 0 {
		t.Fatalf("expected BenchmarkHarnessProtocol iterations > 0, got %d", res.N)
	}
}
