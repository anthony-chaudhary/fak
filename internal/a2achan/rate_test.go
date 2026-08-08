package a2achan

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestSendRateLimitsPerSenderTargetWhileReceiverDrains(t *testing.T) {
	now := time.Unix(100, 0)
	b := NewBus()
	b.rateBurst, b.rateWindow, b.now = 2, time.Second, func() time.Time { return now }
	caps := []abi.Capability{CapA2ASend, CapA2ARecv}
	for i := 0; i < 2; i++ {
		if v := b.Send(context.Background(), "loud", ChannelKey{Locale: InKernel, ID: "peer"}, Shared([]byte("test")), caps...); v.Kind != abi.VerdictAllow {
			t.Fatalf("send %d: %+v", i, v)
		}
		if _, _, err := b.Recv(context.Background(), ChannelKey{Locale: InKernel, ID: "peer"}, caps...); err != nil {
			t.Fatal(err)
		}
	}
	v := b.Send(context.Background(), "loud", ChannelKey{Locale: InKernel, ID: "peer"}, Shared([]byte("test")), caps...)
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonRateLimited || v.By != "a2achan/rate" {
		t.Fatalf("rate verdict = %+v", v)
	}
	if got := len(b.queues[ChannelKey{Locale: InKernel, ID: "peer"}]); got != 0 {
		t.Fatalf("queue depth = %d, want 0", got)
	}

	// Pair keying preserves an independent sender's access to the same target.
	if v := b.Send(context.Background(), "quiet", ChannelKey{Locale: InKernel, ID: "peer"}, Shared([]byte("test")), caps...); v.Kind != abi.VerdictAllow {
		t.Fatalf("independent pair: %+v", v)
	}
}

func TestQueueCapRefusalRefundsRateToken(t *testing.T) {
	b := NewBus()
	b.rateBurst, b.rateWindow, b.now = 1, time.Hour, func() time.Time { return time.Unix(100, 0) }
	b.SetQueueCap(1)
	caps := []abi.Capability{CapA2ASend, CapA2ARecv}
	// Pre-fill without charging the pair under test.
	if v := b.Send(context.Background(), "other", ChannelKey{Locale: InKernel, ID: "peer"}, Shared([]byte("test")), caps...); v.Kind != abi.VerdictAllow {
		t.Fatal(v)
	}
	if v := b.Send(context.Background(), "sender", ChannelKey{Locale: InKernel, ID: "peer"}, Shared([]byte("test")), caps...); v.By != "a2achan/cap" {
		t.Fatalf("cap verdict = %+v", v)
	}
	if _, _, err := b.Recv(context.Background(), ChannelKey{Locale: InKernel, ID: "peer"}, caps...); err != nil {
		t.Fatal(err)
	}
	if v := b.Send(context.Background(), "sender", ChannelKey{Locale: InKernel, ID: "peer"}, Shared([]byte("test")), caps...); v.Kind != abi.VerdictAllow {
		t.Fatalf("refunded token not available: %+v", v)
	}
}
