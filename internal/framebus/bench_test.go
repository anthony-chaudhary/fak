package framebus

import (
	"context"
	"testing"
)

func BenchmarkBus_PublishSingleSubscriber(b *testing.B) {
	bus := NewBus(BusConfig{
		DefaultBufferSize: 100000,
		DefaultDropPolicy: DropPolicyDropNewest,
	})
	defer bus.Close()

	sub, err := bus.Subscribe("telemetry.bench")
	if err != nil {
		b.Fatalf("Subscribe failed: %v", err)
	}
	defer sub.Close()

	// Drain frames in background to prevent buffer buildup
	go func() {
		for range sub.Channel() {
		}
	}()

	frame, err := NewFrame(FrameTypeTelemetry, "telemetry.bench", []byte(`{"metric":"cpu","val":42}`))
	if err != nil {
		b.Fatalf("NewFrame failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if err := bus.Publish(frame); err != nil {
			b.Fatalf("Publish failed: %v", err)
		}
	}
}

func BenchmarkBus_PublishFanoutMultipleSubscribers(b *testing.B) {
	bus := NewBus(BusConfig{
		DefaultBufferSize: 100000,
		DefaultDropPolicy: DropPolicyDropNewest,
	})
	defer bus.Close()

	const subCount = 8
	for i := 0; i < subCount; i++ {
		sub, err := bus.Subscribe("events.*")
		if err != nil {
			b.Fatalf("Subscribe failed: %v", err)
		}
		defer sub.Close()
		go func(s *Subscription) {
			for range s.Channel() {
			}
		}(sub)
	}

	frame, err := NewFrame(FrameTypeEvent, "events.user_login", []byte(`{"user":"alice"}`))
	if err != nil {
		b.Fatalf("NewFrame failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if err := bus.Publish(frame); err != nil {
			b.Fatalf("Publish failed: %v", err)
		}
	}
}

func BenchmarkFrame_CreationAndValidation(b *testing.B) {
	payload := []byte("bench payload data")

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		f, err := NewFrame(FrameTypeMessage, "chat.room1", payload)
		if err != nil {
			b.Fatalf("NewFrame failed: %v", err)
		}
		if err := f.Validate(); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

func BenchmarkTopicMatching(b *testing.B) {
	topics := []struct {
		pattern string
		target  string
	}{
		{"*", "events.action"},
		{"events.*", "events.action"},
		{"events.*", "metrics.cpu"},
		{"exact.topic", "exact.topic"},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		t := topics[i%len(topics)]
		_ = topicMatches(t.pattern, t.target)
	}
}

func BenchmarkPublishSyncWithFilter(b *testing.B) {
	bus := NewBus(BusConfig{
		DefaultBufferSize: 50000,
		DefaultDropPolicy: DropPolicyDropNewest,
	})
	defer bus.Close()

	sub, err := bus.Subscribe("data.*", WithFilter(func(f *Frame) bool {
		return len(f.Payload) > 0
	}))
	if err != nil {
		b.Fatalf("Subscribe failed: %v", err)
	}
	defer sub.Close()

	go func() {
		for range sub.Channel() {
		}
	}()

	ctx := context.Background()
	frame, _ := NewFrame(FrameTypeEvent, "data.point", []byte("valid payload"))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if err := bus.PublishSync(ctx, frame); err != nil {
			b.Fatalf("PublishSync failed: %v", err)
		}
	}
}
