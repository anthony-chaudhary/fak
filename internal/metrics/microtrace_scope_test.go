package metrics

import (
	"sync"
	"testing"
	"time"
)

func TestMicroSpanScopeRecordsOnePairedDuration(t *testing.T) {
	tr := newMicroTracerWithClock(sequenceClock(
		time.Unix(0, 100),
		time.Unix(0, 107),
	))

	scope := tr.Scope("micro-000", MicroSpan{Kind: SpanStep, Label: "prefill"})
	if scope.ID() == 0 {
		t.Fatal("Scope returned the reserved zero span id")
	}
	scope.End()
	scope.End() // cleanup is idempotent even when a caller closes explicitly and via defer

	got, ok := tr.Trace("micro-000")
	if !ok {
		t.Fatal("scope trace missing")
	}
	if len(got.Spans) != 2 {
		t.Fatalf("spans=%+v, want exactly one start and one terminal record", got.Spans)
	}
	start, end := got.Spans[0], got.Spans[1]
	if start.Event != MicroSpanStart || end.Event != MicroSpanEnd {
		t.Fatalf("events=(%q,%q), want (%q,%q)", start.Event, end.Event, MicroSpanStart, MicroSpanEnd)
	}
	if start.SpanID != scope.ID() || end.SpanID != scope.ID() {
		t.Fatalf("span ids=(%d,%d), want shared id %d", start.SpanID, end.SpanID, scope.ID())
	}
	if start.Dur != 0 {
		t.Fatalf("start duration=%s, want 0", start.Dur)
	}
	if end.Dur != 7*time.Nanosecond {
		t.Fatalf("end duration=%s, want 7ns", end.Dur)
	}
	if start.Kind != SpanStep || end.Kind != SpanStep || start.Label != "prefill" || end.Label != "prefill" {
		t.Fatalf("scope metadata changed across pair: start=%+v end=%+v", start, end)
	}
}

func TestMicroSpanScopeDeferClosesEveryExitPath(t *testing.T) {
	tr := newMicroTracerWithClock(stepClock(time.Unix(0, 1_000), time.Nanosecond))

	run := func(traceID string, early bool) {
		scope := tr.Scope(traceID, MicroSpan{Kind: SpanTool, Label: "dispatch"})
		defer scope.End()
		if early {
			return
		}
	}
	run("normal", false)
	run("early", true)

	func() {
		defer func() {
			if got := recover(); got != "boom" {
				t.Fatalf("panic=%v, want boom", got)
			}
		}()
		func() {
			scope := tr.Scope("panic", MicroSpan{Kind: SpanTool, Label: "dispatch"})
			defer scope.End()
			panic("boom")
		}()
	}()

	for _, traceID := range []string{"normal", "early", "panic"} {
		got, ok := tr.Trace(traceID)
		if !ok {
			t.Fatalf("%s trace missing", traceID)
		}
		if len(got.Spans) != 2 {
			t.Fatalf("%s spans=%+v, want paired lifecycle", traceID, got.Spans)
		}
		if got.Spans[0].Event != MicroSpanStart || got.Spans[1].Event != MicroSpanEnd {
			t.Fatalf("%s events=(%q,%q), want start/end", traceID, got.Spans[0].Event, got.Spans[1].Event)
		}
		if got.Spans[0].SpanID != got.Spans[1].SpanID {
			t.Fatalf("%s span ids=(%d,%d), want one paired id", traceID, got.Spans[0].SpanID, got.Spans[1].SpanID)
		}
		if got.Spans[1].Dur < 0 {
			t.Fatalf("%s duration=%s, want nonnegative", traceID, got.Spans[1].Dur)
		}
	}
}

func TestMicroSpanScopeClampsBackwardClock(t *testing.T) {
	tr := newMicroTracerWithClock(sequenceClock(
		time.Unix(0, 20),
		time.Unix(0, 10),
	))
	scope := tr.Scope("micro-000", MicroSpan{Kind: SpanStep})
	scope.End()

	got, _ := tr.Trace("micro-000")
	if got.Spans[1].Dur != 0 {
		t.Fatalf("duration=%s, want 0 when a test clock moves backward", got.Spans[1].Dur)
	}
}

func TestMicroSpanScopeConcurrentIDsDoNotCollide(t *testing.T) {
	tr := newMicroTracerWithClock(func() time.Time { return time.Unix(0, 1) })
	const workers = 32
	const scopesPerWorker = 64

	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < scopesPerWorker; i++ {
				scope := tr.Scope("fleet", MicroSpan{Kind: SpanStep, Label: "decode"})
				scope.End()
			}
		}()
	}
	wg.Wait()

	got, ok := tr.Trace("fleet")
	if !ok {
		t.Fatal("fleet trace missing")
	}
	wantPairs := workers * scopesPerWorker
	if len(got.Spans) != wantPairs*2 {
		t.Fatalf("span records=%d, want %d", len(got.Spans), wantPairs*2)
	}
	type pair struct {
		start int
		end   int
	}
	ids := make(map[uint64]pair, wantPairs)
	for _, span := range got.Spans {
		if span.SpanID == 0 {
			t.Fatal("concurrent scope used reserved zero span id")
		}
		p := ids[span.SpanID]
		switch span.Event {
		case MicroSpanStart:
			p.start++
		case MicroSpanEnd:
			p.end++
		default:
			t.Fatalf("span id %d has unexpected event %q", span.SpanID, span.Event)
		}
		ids[span.SpanID] = p
	}
	if len(ids) != wantPairs {
		t.Fatalf("unique span ids=%d, want %d", len(ids), wantPairs)
	}
	for id, p := range ids {
		if p.start != 1 || p.end != 1 {
			t.Fatalf("span id %d records=%+v, want one start and one end", id, p)
		}
	}
}

func sequenceClock(times ...time.Time) func() time.Time {
	var mu sync.Mutex
	next := 0
	return func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		if next >= len(times) {
			panic("sequenceClock exhausted")
		}
		now := times[next]
		next++
		return now
	}
}

func stepClock(now time.Time, step time.Duration) func() time.Time {
	var mu sync.Mutex
	return func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		got := now
		now = now.Add(step)
		return got
	}
}
