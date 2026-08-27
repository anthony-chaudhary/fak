package macromailbox

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sync"
)

const Schema = "fak.macro-mailbox/1"

type Identity struct {
	AgentID string `json:"agent_id"`
	Address string `json:"address"`
	Secret  []byte `json:"-"`
}
type Message struct {
	ID   string `json:"id"`
	To   string `json:"to"`
	Body []byte `json:"body"`
	Auth string `json:"auth"`
}
type Receipt struct {
	MessageID string `json:"message_id"`
	Action    string `json:"action"`
	Applied   bool   `json:"applied"`
}
type Mailbox struct {
	mu        sync.Mutex
	identity  Identity
	queued    map[string]Message
	delivered map[string]bool
	retired   bool
}

func New(id Identity) *Mailbox {
	return &Mailbox{identity: id, queued: map[string]Message{}, delivered: map[string]bool{}}
}
func Sign(id Identity, m Message) string {
	h := hmac.New(sha256.New, id.Secret)
	h.Write([]byte(m.ID + "|" + m.To + "|" + string(m.Body)))
	return hex.EncodeToString(h.Sum(nil))
}
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
func (m *Mailbox) Snapshot() []Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Message, 0, len(m.queued))
	for _, v := range m.queued {
		out = append(out, v)
	}
	return out
}
func Restore(id Identity, q []Message) *Mailbox {
	m := New(id)
	for _, v := range q {
		m.queued[v.ID] = v
	}
	return m
}
func (m *Mailbox) Retire() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.retired = true
	m.queued = map[string]Message{}
}
