package a2achan

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func bg() context.Context { return context.Background() }

// TestDeterministicDelivery: a fixed send→recv yields a byte-identical Body
// (content-addressed digest) and the same delivery Seq across independent runs.
func TestDeterministicDelivery(t *testing.T) {
	run := func() (string, uint64) {
		b := NewBus()
		to := ChannelKey{Locale: InKernel, ID: "work"}
		if v := b.Send(bg(), "alpha", to, Shared([]byte("hello-peer")), CapA2ASend); v.Kind != abi.VerdictAllow {
			t.Fatalf("send: want Allow, got %v (%s)", v.Kind, abi.ReasonName(v.Reason))
		}
		m, rv, err := b.Recv(bg(), to, CapA2ARecv)
		if err != nil || rv.Kind != abi.VerdictAllow {
			t.Fatalf("recv: want Allow, got %v err=%v", rv.Kind, err)
		}
		return m.Body.Digest, m.Seq
	}
	d1, s1 := run()
	d2, s2 := run()
	if d1 == "" || d1 != d2 {
		t.Fatalf("non-deterministic digest: %q vs %q", d1, d2)
	}
	if s1 != s2 || s1 != 1 {
		t.Fatalf("non-deterministic seq: %d vs %d (want both 1)", s1, s2)
	}
}

// TestFailClosedDefault: no negotiated cap denies send AND recv; a ScopeAgent
// (Private) body addressed to another agent's channel is denied; nothing enqueues.
func TestFailClosedDefault(t *testing.T) {
	b := NewBus()
	to := ChannelKey{Locale: InKernel, ID: "x"}

	// No CapA2ASend advertised → deny (no affirmative send-right).
	if v := b.Send(bg(), "alpha", to, Shared([]byte("hi"))); v.Kind != abi.VerdictDeny {
		t.Fatalf("uncapped send: want Deny, got %v", v.Kind)
	} else if abi.ReasonName(v.Reason) != "DEFAULT_DENY" {
		t.Fatalf("uncapped send reason: want DEFAULT_DENY, got %s", abi.ReasonName(v.Reason))
	}
	// No CapA2ARecv advertised → deny.
	if _, v, _ := b.Recv(bg(), to); v.Kind != abi.VerdictDeny {
		t.Fatalf("uncapped recv: want Deny, got %v", v.Kind)
	}
	// A private (ScopeAgent) body to ANOTHER agent's channel → deny TRUST_VIOLATION.
	v := b.Send(bg(), "alpha", ChannelKey{Locale: InKernel, ID: "shared-q"}, Private([]byte("secret")), CapA2ASend)
	if v.Kind != abi.VerdictDeny || abi.ReasonName(v.Reason) != "TRUST_VIOLATION" {
		t.Fatalf("private cross-agent send: want Deny/TRUST_VIOLATION, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	if b.Len(to) != 0 || b.Len(ChannelKey{Locale: InKernel, ID: "shared-q"}) != 0 {
		t.Fatal("a denied send must not enqueue")
	}
	if b.Stats().Denied != 3 {
		t.Fatalf("want 3 denials, got %d", b.Stats().Denied)
	}
}

// TestScopeTaintEnforcement: a fleet-scoped body crosses agents and arrives with
// its Taint preserved; a private body to the sender's OWN channel (self-handoff)
// is allowed; a quarantined body is denied at send AND held by the ingress screen
// if it reaches the queue another way.
func TestScopeTaintEnforcement(t *testing.T) {
	b := NewBus()
	to := ChannelKey{Locale: InKernel, ID: "q"}

	// Shared (ScopeFleet) crosses agents; the delivered Ref stays Tainted.
	if v := b.Send(bg(), "alpha", to, Shared([]byte("data")), CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("shared cross-agent send: want Allow, got %v", v.Kind)
	}
	m, rv, _ := b.Recv(bg(), to, CapA2ARecv)
	if rv.Kind != abi.VerdictAllow {
		t.Fatalf("recv shared: want Allow, got %v", rv.Kind)
	}
	if m.Body.Taint != abi.TaintTainted {
		t.Fatalf("sharing a result shares its taint: want Tainted, got %v", m.Body.Taint)
	}

	// Private body to the sender's OWN channel (a window self-handoff) is allowed.
	self := ChannelKey{Locale: Window, ID: "alpha"}
	if v := b.Send(bg(), "alpha", self, Private([]byte("my-state")), CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("private self-handoff: want Allow, got %v (%s)", v.Kind, abi.ReasonName(v.Reason))
	}

	// A quarantined body is refused at send.
	q := Inline([]byte("poison"), abi.ScopeFleet, abi.TaintQuarantined)
	if v := b.Send(bg(), "alpha", to, q, CapA2ASend); v.Kind != abi.VerdictDeny || abi.ReasonName(v.Reason) != "TRUST_VIOLATION" {
		t.Fatalf("quarantined send: want Deny/TRUST_VIOLATION, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}

	// Defense in depth: a quarantined body that reaches the queue another way is
	// HELD by the ingress screen on recv (never admitted to the receiver context).
	b.mu.Lock()
	b.seq++
	b.queues[to] = append(b.queues[to], Message{From: "evil", To: to, Body: q, Seq: b.seq})
	b.mu.Unlock()
	_, rv2, _ := b.Recv(bg(), to, CapA2ARecv)
	if rv2.Kind != abi.VerdictQuarantine {
		t.Fatalf("ingress: want Quarantine (held), got %v", rv2.Kind)
	}
	if b.Stats().Held != 1 {
		t.Fatalf("want 1 held, got %d", b.Stats().Held)
	}
}

// TestAsyncRendezvous: a goroutine blocked in Recv on an empty channel unblocks
// and returns the message once another goroutine Sends (condvar rendezvous).
func TestAsyncRendezvous(t *testing.T) {
	b := NewBus()
	to := ChannelKey{Locale: InKernel, ID: "rv"}
	got := make(chan Message, 1)
	go func() {
		m, v, err := b.Recv(bg(), to, CapA2ARecv)
		if err == nil && v.Kind == abi.VerdictAllow {
			got <- m
		} else {
			close(got)
		}
	}()
	if v := b.Send(bg(), "alpha", to, Shared([]byte("ping")), CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("send: want Allow, got %v", v.Kind)
	}
	select {
	case m, ok := <-got:
		if !ok {
			t.Fatal("recv goroutine failed to admit the message")
		}
		if string(m.Body.Inline) != "ping" {
			t.Fatalf("rendezvous body: want ping, got %q", m.Body.Inline)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("recv did not rendezvous within 2s")
	}
}

// TestRecvTimeoutNoLoss: a Recv whose ctx expires on an EMPTY channel returns the
// ctx error and drops no message queued on a DIFFERENT channel.
func TestRecvTimeoutNoLoss(t *testing.T) {
	b := NewBus()
	x := ChannelKey{Locale: InKernel, ID: "x"}
	y := ChannelKey{Locale: InKernel, ID: "y"}
	if v := b.Send(bg(), "alpha", x, Shared([]byte("keep")), CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("seed send: want Allow, got %v", v.Kind)
	}
	ctx, cancel := context.WithTimeout(bg(), 50*time.Millisecond)
	defer cancel()
	if _, _, err := b.Recv(ctx, y, CapA2ARecv); err == nil {
		t.Fatal("recv on empty channel with expired ctx: want error, got nil")
	}
	if b.Len(x) != 1 {
		t.Fatalf("queued message lost: want 1 on x, got %d", b.Len(x))
	}
	m, v, _ := b.Recv(bg(), x, CapA2ARecv)
	if v.Kind != abi.VerdictAllow || string(m.Body.Inline) != "keep" {
		t.Fatalf("recover after timeout: want Allow/keep, got %v/%q", v.Kind, m.Body.Inline)
	}
}

// TestRegisteredFloorInKernelChain: the message floor is first-class IN the kernel
// — a2aGate is in the registered adjudicator chain the kernel walks for the a2a
// tool, and the registered adjudicator decides identically to the Bus's direct gate.
func TestRegisteredFloorInKernelChain(t *testing.T) {
	chain := abi.AdjudicatorsFor(&abi.ToolCall{Tool: ToolSend})
	found := false
	for _, a := range chain {
		if _, ok := a.(a2aGate); ok {
			found = true
		}
	}
	if !found {
		t.Fatal("a2aGate is not registered into the kernel's adjudicator chain for a2a.send")
	}
	if !abi.Supported(CapA2ASend) || !abi.Supported(CapA2ARecv) {
		t.Fatal("a2a capabilities are not registered/negotiable")
	}

	body := Shared([]byte("m"))
	to := ChannelKey{Locale: InKernel, ID: "q"}
	caps := []abi.Capability{CapA2ASend}
	direct := gateSend("alpha", to, body, caps)
	viaCall := a2aGate{}.Adjudicate(bg(), sendCall("alpha", to, body, caps))
	if direct.Kind != viaCall.Kind {
		t.Fatalf("registered gate (%v) disagrees with the Bus's direct gate (%v)", viaCall.Kind, direct.Kind)
	}
}

// TestPubSubFanout: the one-to-many option fans an adjudicated message out as an
// independent copy to every subscriber inbox, under the SAME capability floor as
// point-to-point Send (a private/uncapped publish is refused identically); cancel
// unsubscribes.
func TestPubSubFanout(t *testing.T) {
	b := NewBus()
	topic := ChannelKey{Locale: InKernel, ID: "events"}
	in1, cancel1 := b.Subscribe(topic)
	in2, _ := b.Subscribe(topic)
	if b.Subscribers(topic) != 2 {
		t.Fatalf("want 2 subscribers, got %d", b.Subscribers(topic))
	}

	// A shared publish reaches both private inboxes.
	v, n := b.Publish(bg(), "pub", topic, Shared([]byte("evt")), CapA2ASend)
	if v.Kind != abi.VerdictAllow || n != 2 {
		t.Fatalf("publish: want Allow/fanout=2, got %v/%d", v.Kind, n)
	}
	m1, rv1, _ := b.Recv(bg(), in1, CapA2ARecv)
	m2, rv2, _ := b.Recv(bg(), in2, CapA2ARecv)
	if rv1.Kind != abi.VerdictAllow || rv2.Kind != abi.VerdictAllow {
		t.Fatalf("subscriber recv: want Allow/Allow, got %v/%v", rv1.Kind, rv2.Kind)
	}
	if string(m1.Body.Inline) != "evt" || string(m2.Body.Inline) != "evt" {
		t.Fatalf("fanned-out body: want evt/evt, got %q/%q", m1.Body.Inline, m2.Body.Inline)
	}

	// Cancel one subscriber → the next publish reaches only the survivor.
	cancel1()
	if b.Subscribers(topic) != 1 {
		t.Fatalf("after cancel: want 1 subscriber, got %d", b.Subscribers(topic))
	}
	if _, n2 := b.Publish(bg(), "pub", topic, Shared([]byte("evt2")), CapA2ASend); n2 != 1 {
		t.Fatalf("post-cancel fanout: want 1, got %d", n2)
	}

	// Publishing a private (ScopeAgent) body is refused — publishing is sharing.
	if pv, pn := b.Publish(bg(), "pub", topic, Private([]byte("secret")), CapA2ASend); pv.Kind != abi.VerdictDeny || pn != 0 {
		t.Fatalf("private publish: want Deny/0, got %v/%d", pv.Kind, pn)
	}
	// An uncapped publish is refused (same floor as Send).
	if uv, _ := b.Publish(bg(), "pub", topic, Shared([]byte("x"))); uv.Kind != abi.VerdictDeny {
		t.Fatalf("uncapped publish: want Deny, got %v", uv.Kind)
	}
}
