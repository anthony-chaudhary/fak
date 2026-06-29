package a2achan

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Locale discriminates the three communication locales the one ChannelKey shape
// serves. The mechanism, adjudication, and ingress screen are identical across
// all three; only the key's Locale + ID differ (see the package doc).
type Locale uint8

const (
	// InKernel is a rendezvous within ONE process: two concurrent goroutine-agents
	// meet on a shared name (the wedge — fully real).
	InKernel Locale = iota
	// Session keys a mailbox by a peer's ToolCall.TraceID (cross-session handoff).
	// In-process today; durable cross-process backing is the named next rung.
	Session
	// Window keys a mailbox by a continuation id minted on context-window
	// compaction (the explicit, adjudicated handoff across a context window).
	Window
)

// String renders a Locale for audit/forensics (never panics on an unknown value).
func (l Locale) String() string {
	switch l {
	case InKernel:
		return "in-kernel"
	case Session:
		return "session"
	case Window:
		return "window"
	}
	return "locale?"
}

// ChannelKey names a mailbox. Two keys are equal iff both fields match, so a
// Session channel and an InKernel channel that happen to share an ID are distinct
// mailboxes (the Locale is part of the identity).
type ChannelKey struct {
	Locale Locale
	ID     string
}

// Message is one delivered unit: the Ref Body (its Taint+Scope ride unchanged),
// the From principal + To address, and a per-bus monotonic Seq that fixes a
// deterministic delivery order within a channel.
type Message struct {
	From string
	To   ChannelKey
	Body abi.Ref
	Seq  uint64
}

// Inline builds a small in-line Ref message body with an explicit Scope + Taint
// and a stable content digest (so a fixed payload has a byte-identical Ref across
// runs — the determinism the tests pin). The fail-closed default a caller should
// reach for is Private; Shared is the explicit "widen to cross an agent boundary."
func Inline(b []byte, scope abi.ShareScope, taint abi.TaintLabel) abi.Ref {
	sum := sha256.Sum256(b)
	cp := make([]byte, len(b))
	copy(cp, b)
	return abi.Ref{
		Kind:   abi.RefInline,
		Inline: cp,
		Digest: "sha256:" + hex.EncodeToString(sum[:]),
		Len:    int64(len(b)),
		Taint:  taint,
		Scope:  scope,
	}
}

// Private is the fail-closed default body: ScopeAgent + Tainted, deliverable only
// to the sender's OWN channel (a self-handoff). Sending it to another agent's
// channel is refused — the sender must widen Scope to share.
func Private(b []byte) abi.Ref { return Inline(b, abi.ScopeAgent, abi.TaintTainted) }

// Shared is the explicit cross-agent body: ScopeFleet + Tainted. It may cross to
// another agent's channel (the auditable widening act); it stays Tainted, so the
// receiver cannot re-share it past its Scope.
func Shared(b []byte) abi.Ref { return Inline(b, abi.ScopeFleet, abi.TaintTainted) }

// Bus is a process-global, mutex+condvar mailbox: one ordered queue per
// ChannelKey. Construct with NewBus; Default is the process-global instance the
// package-level Send/Recv use. Safe for concurrent use by many goroutine-agents.
type Bus struct {
	mu     sync.Mutex
	cond   *sync.Cond
	queues map[ChannelKey][]Message
	seq    uint64

	// subs is the pub/sub registry: per topic, the set of live subscriber inboxes
	// a Publish fans a copy out to (see topic.go). subSeq mints unique inbox ids.
	subs   map[ChannelKey]map[uint64]ChannelKey
	subSeq uint64

	sent   int64
	recvd  int64
	denied int64
	held   int64
}

// NewBus builds an empty mailbox bus.
func NewBus() *Bus {
	b := &Bus{
		queues: map[ChannelKey][]Message{},
		subs:   map[ChannelKey]map[uint64]ChannelKey{},
	}
	b.cond = sync.NewCond(&b.mu)
	return b
}

// Default is the process-global bus shared by every in-kernel agent in one
// process (the InKernel locale's rendezvous point).
var Default = NewBus()

// Send adjudicates and (on Allow) enqueues a Ref-backed message to a channel. It
// returns the admission Verdict as a value (deny-as-value): VerdictAllow when the
// message was enqueued, otherwise a deny carrying the closed-vocabulary reason
// (the message is NOT enqueued). caps are the capabilities the caller advertises
// (the "send right"); without CapA2ASend negotiated, Send fails closed. Send
// never blocks.
func (b *Bus) Send(ctx context.Context, from string, to ChannelKey, body abi.Ref, caps ...abi.Capability) abi.Verdict {
	v := gateSend(from, to, body, caps, true) // point-to-point: the self-channel exception applies
	if v.Kind != abi.VerdictAllow {
		atomic.AddInt64(&b.denied, 1)
		return v
	}
	b.mu.Lock()
	b.seq++
	msg := Message{From: from, To: to, Body: body, Seq: b.seq}
	b.queues[to] = append(b.queues[to], msg)
	b.cond.Broadcast()
	b.mu.Unlock()
	atomic.AddInt64(&b.sent, 1)
	return v
}

// Recv blocks (ctx-aware) until a message is available on the channel, then
// screens it on ingress and returns it. The returned Verdict is VerdictAllow for
// an admitted message, or VerdictQuarantine when the ingress screen HELD a
// poisoned message out of the receiver's context (the Message is still returned
// for forensics, but its Body must NOT be admitted). Without CapA2ARecv
// negotiated, Recv fails closed and returns immediately. A ctx that expires while
// the queue is empty returns ctx.Err() and drops no queued message.
func (b *Bus) Recv(ctx context.Context, to ChannelKey, caps ...abi.Capability) (Message, abi.Verdict, error) {
	if v := gateRecv(caps); v.Kind != abi.VerdictAllow {
		atomic.AddInt64(&b.denied, 1)
		return Message{}, v, nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	for len(b.queues[to]) == 0 {
		if err := ctx.Err(); err != nil {
			return Message{}, abi.Verdict{Kind: abi.VerdictDefer, By: "a2achan"}, err
		}
		// ctx-aware condvar wait: AfterFunc broadcasts when ctx is done so a
		// blocked Recv wakes and re-checks ctx; a Send broadcast wakes it with a
		// message. Either way the loop re-evaluates queue + ctx.
		stop := context.AfterFunc(ctx, func() {
			b.mu.Lock()
			b.cond.Broadcast()
			b.mu.Unlock()
		})
		b.cond.Wait()
		stop()
	}
	msg, v := b.dequeue(to)
	if v.Kind == abi.VerdictQuarantine {
		atomic.AddInt64(&b.held, 1)
		return msg, v, nil
	}
	atomic.AddInt64(&b.recvd, 1)
	return msg, v, nil
}

// TryRecv is the non-blocking peer of Recv: it returns the next message if one is
// queued (ok=true) or (zero, ok=false) immediately if the channel is empty. The
// cap floor + ingress screen apply identically.
func (b *Bus) TryRecv(ctx context.Context, to ChannelKey, caps ...abi.Capability) (Message, abi.Verdict, bool) {
	if v := gateRecv(caps); v.Kind != abi.VerdictAllow {
		atomic.AddInt64(&b.denied, 1)
		return Message{}, v, false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.queues[to]) == 0 {
		return Message{}, abi.Verdict{Kind: abi.VerdictDefer, By: "a2achan"}, false
	}
	msg, v := b.dequeue(to)
	if v.Kind == abi.VerdictQuarantine {
		atomic.AddInt64(&b.held, 1)
	} else {
		atomic.AddInt64(&b.recvd, 1)
	}
	return msg, v, true
}

// dequeue pops the head message for `to`, compacts (and prunes the now-empty)
// queue, and screens the body. The caller must hold b.mu. Shared by Recv and
// TryRecv, which differ only in their blocking and counter policy.
func (b *Bus) dequeue(to ChannelKey) (Message, abi.Verdict) {
	msg := b.queues[to][0]
	b.queues[to] = b.queues[to][1:]
	if len(b.queues[to]) == 0 {
		delete(b.queues, to)
	}
	return msg, screenIngress(msg.Body)
}

// Len reports how many messages are queued on a channel (forensics/tests).
func (b *Bus) Len(to ChannelKey) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.queues[to])
}

// Stats is a snapshot of the bus's audit tallies.
type Stats struct {
	Sent     int64 // messages enqueued (Send allowed)
	Received int64 // messages delivered + admitted (Recv allowed)
	Denied   int64 // sends/recvs refused by the cap/scope/taint floor
	Held     int64 // messages quarantine-held on ingress
}

// Stats returns the bus's audit tallies.
func (b *Bus) Stats() Stats {
	return Stats{
		Sent:     atomic.LoadInt64(&b.sent),
		Received: atomic.LoadInt64(&b.recvd),
		Denied:   atomic.LoadInt64(&b.denied),
		Held:     atomic.LoadInt64(&b.held),
	}
}

// --- package-level convenience over the process-global Default bus -------------

// Send sends on the process-global Default bus.
func Send(ctx context.Context, from string, to ChannelKey, body abi.Ref, caps ...abi.Capability) abi.Verdict {
	return Default.Send(ctx, from, to, body, caps...)
}

// Recv receives on the process-global Default bus.
func Recv(ctx context.Context, to ChannelKey, caps ...abi.Capability) (Message, abi.Verdict, error) {
	return Default.Recv(ctx, to, caps...)
}

// TryRecv non-blocking-receives on the process-global Default bus.
func TryRecv(ctx context.Context, to ChannelKey, caps ...abi.Capability) (Message, abi.Verdict, bool) {
	return Default.TryRecv(ctx, to, caps...)
}
