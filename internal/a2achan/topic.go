package a2achan

// topic.go — the one-to-many communication option. Send/Recv (a2achan.go) is
// point-to-point: one message, one receiver. Publish/Subscribe is its dual: one
// adjudicated message, fanned out as an independent copy to every current
// subscriber's PRIVATE inbox. Both options ride the SAME capability floor — a
// Publish is gated by the same gateSend a Send is (so publishing a ScopeAgent
// body, or a quarantined one, is refused identically), and each subscriber Recvs
// on its inbox through the SAME ingress screen. Pub/sub is therefore not a second
// security surface; it is a second delivery shape over one floor.

import (
	"context"
	"strconv"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Subscribe registers a fresh private inbox for a topic and returns it plus a
// cancel func that unsubscribes and drains the inbox. The subscriber Recvs on the
// returned inbox key exactly like any channel (same cap floor + ingress screen).
// Two Subscribe calls on the same topic get DISTINCT inboxes, so each subscriber
// receives its own copy of every published message.
func (b *Bus) Subscribe(topic ChannelKey) (inbox ChannelKey, cancel func()) {
	b.mu.Lock()
	b.subSeq++
	id := b.subSeq
	inbox = ChannelKey{Locale: topic.Locale, ID: topic.ID + "#sub" + strconv.FormatUint(id, 10)}
	if b.subs[topic] == nil {
		b.subs[topic] = map[uint64]ChannelKey{}
	}
	b.subs[topic][id] = inbox
	b.mu.Unlock()

	cancel = func() {
		b.mu.Lock()
		if m := b.subs[topic]; m != nil {
			delete(m, id)
			if len(m) == 0 {
				delete(b.subs, topic)
			}
		}
		delete(b.queues, inbox) // drop any undelivered copies
		b.mu.Unlock()
	}
	return inbox, cancel
}

// Publish adjudicates a message ONCE (the same gateSend a Send folds) and, on
// Allow, enqueues an independent copy to every current subscriber inbox of the
// topic. It returns the admission Verdict and the number of subscribers the
// message was fanned out to (0 when the topic has no subscribers — adjudicated but
// undelivered). Like Send, a denied Publish enqueues nothing.
func (b *Bus) Publish(ctx context.Context, from string, topic ChannelKey, body abi.Ref, caps ...abi.Capability) (abi.Verdict, int) {
	v := gateSend(from, topic, body, caps)
	if v.Kind != abi.VerdictAllow {
		atomic.AddInt64(&b.denied, 1)
		return v, 0
	}
	b.mu.Lock()
	n := 0
	for _, inbox := range b.subs[topic] {
		b.seq++
		b.queues[inbox] = append(b.queues[inbox], Message{From: from, To: inbox, Body: body, Seq: b.seq})
		n++
	}
	if n > 0 {
		b.cond.Broadcast()
	}
	b.mu.Unlock()
	if n > 0 {
		atomic.AddInt64(&b.sent, int64(n))
	}
	return v, n
}

// Subscribers reports how many live subscribers a topic has (forensics/tests).
func (b *Bus) Subscribers(topic ChannelKey) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.subs[topic])
}

// Publish on the process-global Default bus.
func Publish(ctx context.Context, from string, topic ChannelKey, body abi.Ref, caps ...abi.Capability) (abi.Verdict, int) {
	return Default.Publish(ctx, from, topic, body, caps...)
}

// Subscribe on the process-global Default bus.
func Subscribe(topic ChannelKey) (inbox ChannelKey, cancel func()) {
	return Default.Subscribe(topic)
}
