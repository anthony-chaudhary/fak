package a2achan

// queuecap_test.go — the length-bound witness for #3480: a process-global Bus
// queue keyed on an undrained/abandoned mailbox (a steer to a dead trace, a
// Publish copy to an inbox no one Recvs) must NOT grow without bound. Each test
// fails on the pre-cap code (an unbounded append) and passes with the cap.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestSendRefusesAtQueueCapAndFreesOnRecv: Send admits up to the cap, refuses the
// message that would push past it (deny-as-value, RATE_LIMITED — nothing enqueued),
// and admits again once a Recv drains a slot. This is the bound that keeps an
// undrained channel from accumulating forever.
func TestSendRefusesAtQueueCapAndFreesOnRecv(t *testing.T) {
	b := NewBus()
	b.SetQueueCap(3)
	to := ChannelKey{Locale: Session, ID: "dead-trace"}

	for i := 0; i < 3; i++ {
		if v := b.Send(bg(), "operator", to, Shared([]byte("steer")), CapA2ASend); v.Kind != abi.VerdictAllow {
			t.Fatalf("send %d: want Allow, got %v (%s)", i, v.Kind, abi.ReasonName(v.Reason))
		}
	}
	if got := b.Len(to); got != 3 {
		t.Fatalf("queue len = %d, want 3 (filled to cap)", got)
	}

	// The over-cap Send is refused as backpressure and enqueues nothing.
	v := b.Send(bg(), "operator", to, Shared([]byte("overflow")), CapA2ASend)
	if v.Kind != abi.VerdictDeny || abi.ReasonName(v.Reason) != "RATE_LIMITED" {
		t.Fatalf("over-cap send: want Deny/RATE_LIMITED, got %v (%s)", v.Kind, abi.ReasonName(v.Reason))
	}
	if got := b.Len(to); got != 3 {
		t.Fatalf("queue len after refused send = %d, want 3 (unchanged)", got)
	}
	if d := b.Stats().Denied; d != 1 {
		t.Fatalf("denied tally = %d, want 1 (the over-cap send)", d)
	}

	// Draining one message frees a slot, so the next Send is admitted again.
	if _, rv, ok := b.TryRecv(bg(), to, CapA2ARecv); !ok || rv.Kind != abi.VerdictAllow {
		t.Fatalf("drain: want a delivered message, ok=%v verdict=%v", ok, rv.Kind)
	}
	if v := b.Send(bg(), "operator", to, Shared([]byte("after-drain")), CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("post-drain send: want Allow (a slot freed), got %v (%s)", v.Kind, abi.ReasonName(v.Reason))
	}
	if got := b.Len(to); got != 3 {
		t.Fatalf("queue len after drain+send = %d, want 3", got)
	}
}

// TestPublishDropsCopyToFullSubscriberInbox: a Publish copy to a subscriber inbox
// already at the cap is dropped (skipped, not counted in the fan-out), so an
// abandoned subscriber inbox stays bounded and never blocks the live subscribers.
func TestPublishDropsCopyToFullSubscriberInbox(t *testing.T) {
	b := NewBus()
	b.SetQueueCap(2)
	topic := ChannelKey{Locale: InKernel, ID: "evt-topic"}
	inbox, cancel := b.Subscribe(topic)
	defer cancel()

	for i := 0; i < 2; i++ {
		if _, n := b.Publish(bg(), "pub", topic, Shared([]byte("evt")), CapA2ASend); n != 1 {
			t.Fatalf("publish %d: fan-out = %d, want 1", i, n)
		}
	}
	if got := b.Len(inbox); got != 2 {
		t.Fatalf("inbox len = %d, want 2 (at cap)", got)
	}

	// The full inbox is skipped: the copy is dropped and not counted.
	v, n := b.Publish(bg(), "pub", topic, Shared([]byte("evt-overflow")), CapA2ASend)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("over-cap publish: want Allow (adjudicated once), got %v (%s)", v.Kind, abi.ReasonName(v.Reason))
	}
	if n != 0 {
		t.Fatalf("over-cap publish fan-out = %d, want 0 (full inbox skipped)", n)
	}
	if got := b.Len(inbox); got != 2 {
		t.Fatalf("inbox len after over-cap publish = %d, want 2 (unchanged)", got)
	}
}

// TestNewBusDefaultsToDefaultQueueCap: the bound is ON by default, and SetQueueCap(0)
// is the explicit opt-out to the old unbounded behavior.
func TestNewBusDefaultsToDefaultQueueCap(t *testing.T) {
	if got := NewBus().queueCap; got != DefaultQueueCap {
		t.Fatalf("NewBus queueCap = %d, want DefaultQueueCap (%d)", got, DefaultQueueCap)
	}

	b := NewBus()
	b.SetQueueCap(0) // opt out of the bound
	to := ChannelKey{Locale: InKernel, ID: "unbounded"}
	for i := 0; i < DefaultQueueCap+5; i++ {
		if v := b.Send(bg(), "alpha", to, Shared([]byte("m")), CapA2ASend); v.Kind != abi.VerdictAllow {
			t.Fatalf("unbounded send %d refused: %v (%s)", i, v.Kind, abi.ReasonName(v.Reason))
		}
	}
	if got := b.Len(to); got != DefaultQueueCap+5 {
		t.Fatalf("unbounded queue len = %d, want %d (no bound applied)", got, DefaultQueueCap+5)
	}
}
