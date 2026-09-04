package framebus

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewBusDefaults(t *testing.T) {
	bus := NewBus(DefaultConfig())
	if bus == nil {
		t.Fatal("expected non-nil bus")
	}
	if bus.IsClosed() {
		t.Fatal("expected bus to be open")
	}
	if count := bus.SubscriberCount(); count != 0 {
		t.Fatalf("expected 0 subscribers, got %d", count)
	}
	if err := bus.Close(); err != nil {
		t.Fatalf("expected clean close, got %v", err)
	}
	if !bus.IsClosed() {
		t.Fatal("expected bus to be closed")
	}
}

func TestFrameValidation(t *testing.T) {
	var nilFrame *Frame
	if err := nilFrame.Validate(); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("expected ErrInvalidFrame for nil frame, got %v", err)
	}

	invalid := &Frame{ID: "", Type: FrameTypeEvent, Topic: "test"}
	if err := invalid.Validate(); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("expected ErrInvalidFrame for empty ID, got %v", err)
	}

	invalidType := &Frame{ID: "1", Type: "", Topic: "test"}
	if err := invalidType.Validate(); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("expected ErrInvalidFrame for empty Type, got %v", err)
	}

	invalidTopic := &Frame{ID: "1", Type: FrameTypeEvent, Topic: ""}
	if err := invalidTopic.Validate(); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("expected ErrInvalidFrame for empty Topic, got %v", err)
	}

	valid := &Frame{ID: "1", Type: FrameTypeEvent, Topic: "test"}
	if err := valid.Validate(); err != nil {
		t.Fatalf("expected valid frame, got %v", err)
	}

	_, err := NewFrame("", "topic", nil)
	if !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("expected ErrInvalidFrame for empty type in NewFrame, got %v", err)
	}

	_, err = NewFrame(FrameTypeEvent, "", nil)
	if !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("expected ErrInvalidFrame for empty topic in NewFrame, got %v", err)
	}

	f, err := NewFrame(FrameTypeEvent, "topic.sub", []byte("payload"))
	if err != nil {
		t.Fatalf("expected successful NewFrame, got %v", err)
	}
	if f.ID == "" || f.Type != FrameTypeEvent || f.Topic != "topic.sub" || string(f.Payload) != "payload" {
		t.Fatalf("unexpected frame fields: %+v", f)
	}
	if f.Timestamp.IsZero() {
		t.Fatal("expected non-zero timestamp")
	}
}

func TestPublishSubscribeExactTopic(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	sub, err := bus.Subscribe("events.audit")
	if err != nil {
		t.Fatalf("unexpected subscribe error: %v", err)
	}
	defer sub.Close()

	f1, _ := NewFrame(FrameTypeEvent, "events.audit", []byte("user-login"))
	f1.Metadata["user"] = "alice"
	f2, _ := NewFrame(FrameTypeEvent, "events.other", []byte("other-event"))

	if err := bus.Publish(f1); err != nil {
		t.Fatalf("publish f1 failed: %v", err)
	}
	if err := bus.Publish(f2); err != nil {
		t.Fatalf("publish f2 failed: %v", err)
	}

	select {
	case received := <-sub.Channel():
		if received.ID != f1.ID {
			t.Fatalf("expected frame ID %s, got %s", f1.ID, received.ID)
		}
		if string(received.Payload) != "user-login" {
			t.Fatalf("expected payload 'user-login', got %s", string(received.Payload))
		}
		if received.Metadata["user"] != "alice" {
			t.Fatalf("expected metadata user alice, got %s", received.Metadata["user"])
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("timed out waiting for f1")
	}

	select {
	case unexpected := <-sub.Channel():
		t.Fatalf("unexpected frame received: %+v", unexpected)
	case <-time.After(30 * time.Millisecond):
		// Expected: no more frames
	}
}

func TestPublishSubscribeWildcardTopic(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	sub, err := bus.Subscribe("*")
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	f1, _ := NewFrame(FrameTypeEvent, "a.b", []byte("1"))
	f2, _ := NewFrame(FrameTypeTelemetry, "x.y.z", []byte("2"))

	_ = bus.Publish(f1)
	_ = bus.Publish(f2)

	for _, expected := range []*Frame{f1, f2} {
		select {
		case r := <-sub.Channel():
			if r.ID != expected.ID {
				t.Fatalf("expected ID %s, got %s", expected.ID, r.ID)
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatalf("timed out waiting for frame %s", expected.ID)
		}
	}
}

func TestPublishSubscribePrefixTopic(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	sub, err := bus.Subscribe("telemetry.*")
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	f1, _ := NewFrame(FrameTypeTelemetry, "telemetry.cpu", []byte("cpu"))
	f2, _ := NewFrame(FrameTypeTelemetry, "telemetry.memory", []byte("mem"))
	f3, _ := NewFrame(FrameTypeEvent, "audit.event", []byte("audit"))

	_ = bus.Publish(f1)
	_ = bus.Publish(f2)
	_ = bus.Publish(f3)

	for _, expected := range []*Frame{f1, f2} {
		select {
		case r := <-sub.Channel():
			if r.ID != expected.ID {
				t.Fatalf("expected ID %s, got %s", expected.ID, r.ID)
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatalf("timed out waiting for %s", expected.ID)
		}
	}

	select {
	case unexpected := <-sub.Channel():
		t.Fatalf("received unexpected frame: %+v", unexpected)
	case <-time.After(30 * time.Millisecond):
	}
}

func TestFrameTypeFilter(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	sub, err := bus.Subscribe(
		"events",
		WithFrameTypeFilter(FrameTypeTelemetry, FrameTypeEvent),
	)
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	fEvent, _ := NewFrame(FrameTypeEvent, "events", []byte("event"))
	fControl, _ := NewFrame(FrameTypeControl, "events", []byte("control"))
	fTelem, _ := NewFrame(FrameTypeTelemetry, "events", []byte("telem"))

	_ = bus.Publish(fEvent)
	_ = bus.Publish(fControl)
	_ = bus.Publish(fTelem)

	for _, expected := range []*Frame{fEvent, fTelem} {
		select {
		case r := <-sub.Channel():
			if r.ID != expected.ID {
				t.Fatalf("expected %s, got %s", expected.ID, r.ID)
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatalf("timed out waiting for %s", expected.ID)
		}
	}

	select {
	case unexpected := <-sub.Channel():
		t.Fatalf("received unexpected frame: %+v", unexpected)
	case <-time.After(30 * time.Millisecond):
	}
}

func TestCustomFilter(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	sub, err := bus.Subscribe(
		"app",
		WithFilter(func(f *Frame) bool {
			return f.Metadata["env"] == "production"
		}),
	)
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	f1, _ := NewFrame(FrameTypeMessage, "app", []byte("prod"))
	f1.Metadata["env"] = "production"

	f2, _ := NewFrame(FrameTypeMessage, "app", []byte("dev"))
	f2.Metadata["env"] = "development"

	_ = bus.Publish(f1)
	_ = bus.Publish(f2)

	select {
	case r := <-sub.Channel():
		if r.ID != f1.ID {
			t.Fatalf("expected f1, got %s", r.ID)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("timed out waiting for f1")
	}

	select {
	case unexpected := <-sub.Channel():
		t.Fatalf("unexpected frame received: %+v", unexpected)
	case <-time.After(30 * time.Millisecond):
	}
}

func TestDropPolicyFailClosed(t *testing.T) {
	cfg := BusConfig{
		DefaultBufferSize: 2,
		DefaultDropPolicy: DropPolicyFailClosed,
		PublishTimeout:    50 * time.Millisecond,
	}
	bus := NewBus(cfg)
	defer bus.Close()

	sub, err := bus.Subscribe("queue")
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	f1, _ := NewFrame(FrameTypeEvent, "queue", []byte("1"))
	f2, _ := NewFrame(FrameTypeEvent, "queue", []byte("2"))
	f3, _ := NewFrame(FrameTypeEvent, "queue", []byte("3"))

	if err := bus.Publish(f1); err != nil {
		t.Fatalf("publish 1 failed: %v", err)
	}
	if err := bus.Publish(f2); err != nil {
		t.Fatalf("publish 2 failed: %v", err)
	}

	err = bus.Publish(f3)
	if !errors.Is(err, ErrBufferFull) {
		t.Fatalf("expected ErrBufferFull on saturated queue, got %v", err)
	}

	if drops := sub.DroppedCount(); drops != 1 {
		t.Fatalf("expected 1 dropped frame, got %d", drops)
	}
}

func TestDropPolicyDropNewest(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	sub, err := bus.Subscribe(
		"queue",
		WithBufferSize(2),
		WithDropPolicy(DropPolicyDropNewest),
	)
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	f1, _ := NewFrame(FrameTypeEvent, "queue", []byte("1"))
	f2, _ := NewFrame(FrameTypeEvent, "queue", []byte("2"))
	f3, _ := NewFrame(FrameTypeEvent, "queue", []byte("3"))

	if err := bus.Publish(f1); err != nil {
		t.Fatalf("publish 1 failed: %v", err)
	}
	if err := bus.Publish(f2); err != nil {
		t.Fatalf("publish 2 failed: %v", err)
	}
	if err := bus.Publish(f3); err != nil {
		t.Fatalf("publish 3 with DropNewest should not error, got %v", err)
	}

	if drops := sub.DroppedCount(); drops != 1 {
		t.Fatalf("expected 1 dropped frame, got %d", drops)
	}

	r1 := <-sub.Channel()
	r2 := <-sub.Channel()
	if string(r1.Payload) != "1" || string(r2.Payload) != "2" {
		t.Fatalf("expected 1 and 2 in buffer, got %s and %s", string(r1.Payload), string(r2.Payload))
	}

	select {
	case unexpected := <-sub.Channel():
		t.Fatalf("expected no 3rd frame, got %+v", unexpected)
	case <-time.After(20 * time.Millisecond):
	}
}

func TestDropPolicyDropOldest(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	sub, err := bus.Subscribe(
		"queue",
		WithBufferSize(2),
		WithDropPolicy(DropPolicyDropOldest),
	)
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	f1, _ := NewFrame(FrameTypeEvent, "queue", []byte("1"))
	f2, _ := NewFrame(FrameTypeEvent, "queue", []byte("2"))
	f3, _ := NewFrame(FrameTypeEvent, "queue", []byte("3"))

	if err := bus.Publish(f1); err != nil {
		t.Fatalf("publish 1 failed: %v", err)
	}
	if err := bus.Publish(f2); err != nil {
		t.Fatalf("publish 2 failed: %v", err)
	}
	if err := bus.Publish(f3); err != nil {
		t.Fatalf("publish 3 failed: %v", err)
	}

	if drops := sub.DroppedCount(); drops != 1 {
		t.Fatalf("expected 1 dropped frame, got %d", drops)
	}

	r1 := <-sub.Channel()
	r2 := <-sub.Channel()
	if string(r1.Payload) != "2" || string(r2.Payload) != "3" {
		t.Fatalf("expected 2 and 3 in buffer (1 evicted), got %s and %s", string(r1.Payload), string(r2.Payload))
	}
}

func TestDropPolicyBlock(t *testing.T) {
	cfg := BusConfig{
		DefaultBufferSize: 1,
		DefaultDropPolicy: DropPolicyBlock,
		PublishTimeout:    20 * time.Millisecond,
	}
	bus := NewBus(cfg)
	defer bus.Close()

	sub, err := bus.Subscribe("blocking")
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	f1, _ := NewFrame(FrameTypeEvent, "blocking", []byte("1"))
	f2, _ := NewFrame(FrameTypeEvent, "blocking", []byte("2"))

	if err := bus.Publish(f1); err != nil {
		t.Fatalf("publish 1 failed: %v", err)
	}

	// Saturated buffer: publish should block and time out
	start := time.Now()
	err = bus.Publish(f2)
	elapsed := time.Since(start)
	if !errors.Is(err, ErrBufferFull) {
		t.Fatalf("expected ErrBufferFull on block timeout, got %v", err)
	}
	if elapsed < 15*time.Millisecond {
		t.Fatalf("expected timeout delay >= 15ms, got %v", elapsed)
	}

	// Unblock via concurrent consumer
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(10 * time.Millisecond)
		<-sub.Channel() // Read f1
	}()

	f3, _ := NewFrame(FrameTypeEvent, "blocking", []byte("3"))
	if err := bus.Publish(f3); err != nil {
		t.Fatalf("publish 3 should unblock and succeed, got %v", err)
	}
	wg.Wait()
}

func TestPublishSyncContextCancelled(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	f, _ := NewFrame(FrameTypeEvent, "test", []byte("data"))
	err := bus.PublishSync(ctx, f)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestUnsubscribe(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	sub, err := bus.Subscribe("test.topic")
	if err != nil {
		t.Fatalf("subscribe failed: %v", err)
	}
	if bus.SubscriberCount() != 1 {
		t.Fatalf("expected 1 subscriber, got %d", bus.SubscriberCount())
	}

	if err := bus.Unsubscribe(sub.ID()); err != nil {
		t.Fatalf("unsubscribe failed: %v", err)
	}
	if bus.SubscriberCount() != 0 {
		t.Fatalf("expected 0 subscribers, got %d", bus.SubscriberCount())
	}

	// Verify channel was closed
	select {
	case _, ok := <-sub.Channel():
		if ok {
			t.Fatal("expected closed channel")
		}
	default:
		t.Fatal("channel should have closed")
	}

	// Unsubscribe again returns ErrSubscriberNotFound
	if err := bus.Unsubscribe(sub.ID()); !errors.Is(err, ErrSubscriberNotFound) {
		t.Fatalf("expected ErrSubscriberNotFound, got %v", err)
	}

	// Subscription.Close() is idempotent
	if err := sub.Close(); err != nil {
		t.Fatalf("expected nil error on sub.Close(), got %v", err)
	}
}

func TestCloseBus(t *testing.T) {
	bus := NewBus(DefaultConfig())
	sub1, _ := bus.Subscribe("topic1")
	sub2, _ := bus.Subscribe("topic2")

	if err := bus.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	// Repeated close returns ErrBusClosed
	if err := bus.Close(); !errors.Is(err, ErrBusClosed) {
		t.Fatalf("expected ErrBusClosed on duplicate close, got %v", err)
	}

	// New subscribe returns ErrBusClosed
	_, err := bus.Subscribe("topic3")
	if !errors.Is(err, ErrBusClosed) {
		t.Fatalf("expected ErrBusClosed on subscribe after close, got %v", err)
	}

	// Publish returns ErrBusClosed
	f, _ := NewFrame(FrameTypeEvent, "topic1", []byte("data"))
	if err := bus.Publish(f); !errors.Is(err, ErrBusClosed) {
		t.Fatalf("expected ErrBusClosed on publish after close, got %v", err)
	}

	// Subscription channels should be closed
	for i, sub := range []*Subscription{sub1, sub2} {
		select {
		case _, ok := <-sub.Channel():
			if ok {
				t.Fatalf("sub %d channel was not closed", i)
			}
		default:
			t.Fatalf("sub %d channel was not closed", i)
		}
	}
}

func BenchmarkFramePublish(b *testing.B) {
	cfg := BusConfig{
		DefaultBufferSize: 1024,
		DefaultDropPolicy: DropPolicyDropNewest,
		PublishTimeout:    10 * time.Millisecond,
	}
	bus := NewBus(cfg)
	defer bus.Close()

	sub, err := bus.Subscribe("bench.topic")
	if err != nil {
		b.Fatalf("subscribe failed: %v", err)
	}
	defer sub.Close()

	var done atomic.Bool
	go func() {
		for !done.Load() {
			select {
			case <-sub.Channel():
			default:
				time.Sleep(time.Microsecond)
			}
		}
	}()
	defer done.Store(true)

	f, _ := NewFrame(FrameTypeTelemetry, "bench.topic", []byte("benchmark-payload"))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if err := bus.Publish(f); err != nil {
			b.Fatalf("publish failed at %d: %v", i, err)
		}
	}
}
