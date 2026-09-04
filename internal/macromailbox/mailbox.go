// Package macromailbox provides fail-closed authenticated message queuing and delivery.
// Invariant: mailbox routing is fail-closed and preserves exactly-once semantics.
package macromailbox

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sync"
)

// Schema identifies the macro-mailbox payload structure.
const Schema = "fak.macro-mailbox/1"

// Identity defines the credentials and address for an agent mailbox participant.
type Identity struct {
	AgentID string `json:"agent_id"`
	Address string `json:"address"`
	Secret  []byte `json:"-"`
}

// Message is an authenticated payload directed to a destination mailbox address.
type Message struct {
	ID   string `json:"id"`
	To   string `json:"to"`
	Body []byte `json:"body"`
	Auth string `json:"auth"`
}

// Receipt conveys the outcome of an enqueue or delivery action.
type Receipt struct {
	MessageID string `json:"message_id"`
	Action    string `json:"action"`
	Applied   bool   `json:"applied"`
}

// Mailbox provides fail-closed in-memory queued delivery for authenticated agent messages.
//
// Invariant: mailbox routing is fail-closed and preserves exactly-once semantics.
// Guard condition: unauthenticated, mistargeted, or retired attempts are refused immediately.
type Mailbox struct {
	mu        sync.Mutex
	identity  Identity
	queued    map[string]Message
	delivered map[string]bool
	retired   bool
}

// New constructs an active Mailbox initialized for the specified Identity.
func New(id Identity) *Mailbox {
	return &Mailbox{identity: id, queued: map[string]Message{}, delivered: map[string]bool{}}
}

// Sign computes the HMAC-SHA256 authentication signature over message identity and body.
func Sign(id Identity, m Message) string {
	h := hmac.New(sha256.New, id.Secret)
	h.Write([]byte(m.ID + "|" + m.To + "|" + string(m.Body)))
	return hex.EncodeToString(h.Sum(nil))
}

// Enqueue validates authentication and deduplicates messages before admission.
// Guard condition: refuses message if mailbox is retired or auth verification fails.
// Invariant: duplicate message IDs are recognized and deduplicated without failure.
func (m *Mailbox) Enqueue(msg Message) (Receipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.retired {
		return Receipt{}, errors.New("mailbox retired")
	}
	if msg.To != m.identity.Address || !hmac.Equal([]byte(msg.Auth), []byte(Sign(m.identity, msg))) {
		return Receipt{}, errors.New("authentication failed")
	}
	if m.delivered[msg.ID] || m.queued[msg.ID].ID != "" {
		return Receipt{msg.ID, "deduplicate", false}, nil
	}
	m.queued[msg.ID] = msg
	return Receipt{msg.ID, "enqueue", true}, nil
}

// Deliver retrieves and removes an enqueued message by ID, marking it delivered.
// Guard condition: refuses retrieval if message ID is not present in queue.
// Invariant: message delivery is exactly-once; delivered IDs cannot be re-delivered.
func (m *Mailbox) Deliver(id string) (Message, Receipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	msg, ok := m.queued[id]
	if !ok {
		return Message{}, Receipt{}, errors.New("not queued")
	}
	delete(m.queued, id)
	m.delivered[id] = true
	return msg, Receipt{id, "deliver", true}, nil
}

// Snapshot returns an unlinked slice of all currently queued messages.
func (m *Mailbox) Snapshot() []Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Message, 0, len(m.queued))
	for _, v := range m.queued {
		out = append(out, v)
	}
	return out
}

// Restore reconstructs a Mailbox state from an identity and a list of queued messages.
func Restore(id Identity, q []Message) *Mailbox {
	m := New(id)
	for _, v := range q {
		m.queued[v.ID] = v
	}
	return m
}

// Retire permanently disables the mailbox and flushes pending queued messages.
// Invariant: retirement is irreversible; all subsequent enqueues fail closed.
func (m *Mailbox) Retire() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.retired = true
	m.queued = map[string]Message{}
}
