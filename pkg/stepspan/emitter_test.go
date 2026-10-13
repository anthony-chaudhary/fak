package stepspan

import (
	"bytes"
	"errors"
	"sync"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
var errSink = errors.New("sink failure")

type testSink struct {
	entered chan struct{}
	release chan struct{}
	once    sync.Once
	spans   []Span
	err     error
	closed  bool
}

func (s *testSink) WriteSpan(v Span) error {
	if s.entered != nil {
		s.once.Do(func() { close(s.entered) })
		<-s.release
	}
	s.spans = append(s.spans, v)
	return s.err
}
func (s *testSink) Close() error { s.closed = true; return s.err }
func TestEmitterOverflowNeverWaitsForSink(t *testing.T) {
	sink := &testSink{entered: make(chan struct{}), release: make(chan struct{})}
	e, err := NewEmitter(2, sink)
	if err != nil {
		t.Fatal(err)
	}
	if !e.Emit(fixture()) {
		t.Fatal("first enqueue")
	}
	<-sink.entered
	s := fixture()
	s.IDs.SeqIDs = []string{"original"}
	s.Links = []Link{{s.TraceID, s.SpanID}}
	if !e.Emit(s) || !e.Emit(fixture()) {
		t.Fatal("ring should have two free slots")
	}
	s.IDs.SeqIDs[0] = "mutated"
	s.Links[0].SpanID = "mutated"
	done := make(chan struct{})
	go func() {
		for i := 0; i < 10000; i++ {
			if e.Emit(fixture()) {
				t.Error("overflow accepted")
			}
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Emit blocked behind sink")
	}
	if e.Dropped(Engine) != 10000 {
		t.Fatalf("drops=%d", e.Dropped(Engine))
	}
	var metrics bytes.Buffer
	if err := e.WritePrometheus(&metrics); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(metrics.Bytes(), []byte("fak_stepspan_dropped_total{plane=\"engine\"} 10000")) {
		t.Fatal(metrics.String())
	}
	close(sink.release)
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if len(sink.spans) != 3 || !sink.closed || sink.spans[1].IDs.SeqIDs[0] != "original" || sink.spans[1].Links[0].SpanID == "mutated" {
		t.Fatalf("drain ownership: %#v", sink.spans)
	}
	if e.Emit(fixture()) {
		t.Fatal("accepted after close")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}
func TestEmitterConcurrentCloseAndFailures(t *testing.T) {
	sink := &testSink{err: errSink}
	e, err := NewEmitter(32, sink)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				e.Emit(fixture())
			}
		}()
	}
	if err := e.Close(); !errors.Is(err, errSink) {
		t.Fatal(err)
	}
	wg.Wait()
	if e.SinkFailures() != uint64(len(sink.spans)) {
		t.Fatal("failure counter")
	}
	total := uint64(len(sink.spans)) + e.Dropped(Engine)
	if total != 800 {
		t.Fatalf("accounted=%d", total)
	}
}
