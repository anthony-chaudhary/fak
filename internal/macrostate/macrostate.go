package macrostate

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sort"
	"time"
)

const Schema = "fak.macro-state-event/1"

type Kind string

const (
	Promote Kind = "promote"
	Correct Kind = "correct"
	Delete  Kind = "delete"
	Retire  Kind = "retire"
)

type Event struct {
	Schema     string     `json:"schema"`
	ID         string     `json:"id"`
	At         time.Time  `json:"at"`
	Kind       Kind       `json:"kind"`
	Key        string     `json:"key,omitempty"`
	Value      string     `json:"value,omitempty"`
	Provenance string     `json:"provenance"`
	Replaces   string     `json:"replaces,omitempty"`
	ExpiresAt  *time.Time `json:"expires_at,omitempty"`
}
type Receipt struct {
	EventID string `json:"event_id"`
	Digest  string `json:"digest"`
	Applied bool   `json:"applied"`
	Reason  string `json:"reason,omitempty"`
}
type Store struct {
	events  []Event
	values  map[string]Event
	retired bool
}

func (s *Store) Apply(e Event) (Receipt, error) {
	if s.values == nil {
		s.values = map[string]Event{}
	}
	if e.Schema != Schema || e.ID == "" || e.Provenance == "" {
		return Receipt{}, errors.New("schema, id, and provenance are required")
	}
	if s.retired {
		return Receipt{}, errors.New("state is retired")
	}
	switch e.Kind {
	case Promote:
		s.values[e.Key] = e
	case Correct:
		if e.Replaces == "" {
			return Receipt{}, errors.New("correction requires replaced event")
		}
		s.values[e.Key] = e
	case Delete:
		delete(s.values, e.Key)
	case Retire:
		s.values = map[string]Event{}
		s.retired = true
	default:
		return Receipt{}, errors.New("unknown lifecycle event")
	}
	s.events = append(s.events, e)
	h := sha256.Sum256([]byte(e.ID + "|" + string(e.Kind) + "|" + e.Key + "|" + e.Provenance))
	return Receipt{e.ID, hex.EncodeToString(h[:]), true, ""}, nil
}
func (s *Store) Compact(now time.Time) map[string]string {
	out := map[string]string{}
	keys := make([]string, 0, len(s.values))
	for k := range s.values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		e := s.values[k]
		if e.ExpiresAt != nil && !e.ExpiresAt.After(now) {
			delete(s.values, k)
			continue
		}
		out[k] = e.Value
	}
	return out
}
func (s *Store) Events() []Event { return append([]Event(nil), s.events...) }
func Replay(events []Event) (*Store, error) {
	s := &Store{}
	for _, e := range events {
		if _, err := s.Apply(e); err != nil {
			return nil, err
		}
	}
	return s, nil
}
