// Package agentqueue plans deterministic bounded desired-state agent populations.
package agentqueue

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"time"
)

const Schema = "fak.agentqueue.snapshot.v1"

type PoolSpec struct {
	ID      string `json:"id"`
	Min     int    `json:"min"`
	Desired int    `json:"desired"`
	Max     int    `json:"max"`
}

func (s PoolSpec) Validate() error {
	if s.ID == "" {
		return errors.New("pool id is required")
	}
	if s.Min < 0 || s.Min > s.Desired || s.Desired > s.Max {
		return fmt.Errorf("bounds must satisfy 0 <= min <= desired <= max")
	}
	return nil
}

type IntentState string

const (
	IntentQueued    IntentState = "queued"
	IntentRunning   IntentState = "running"
	IntentCompleted IntentState = "completed"
	IntentFailed    IntentState = "failed"
	IntentHeld      IntentState = "held"
)

type Intent struct {
	ID            string      `json:"id"`
	State         IntentState `json:"state"`
	RetryEligible bool        `json:"retry_eligible,omitempty"`
	Launch        LaunchSpec  `json:"launch,omitempty"`
	PID           int         `json:"pid,omitempty"`
	ExpiresAt     time.Time   `json:"expires_at,omitempty"`
	LeaseExpires  time.Time   `json:"lease_expires,omitempty"`
}
type AttemptState string

const (
	AttemptReserved  AttemptState = "reserved"
	AttemptRunning   AttemptState = "running"
	AttemptSucceeded AttemptState = "succeeded"
	AttemptFailed    AttemptState = "failed"
)

type Attempt struct {
	ID           string       `json:"id"`
	IntentID     string       `json:"intent_id"`
	State        AttemptState `json:"state"`
	PID          int          `json:"pid,omitempty"`
	ExpiresAt    time.Time    `json:"expires_at,omitempty"`
	LeaseExpires time.Time    `json:"lease_expires,omitempty"`
}
type Snapshot struct {
	Schema     string    `json:"schema,omitempty"`
	Generation string    `json:"generation"`
	Pool       PoolSpec  `json:"pool"`
	Intents    []Intent  `json:"intents,omitempty"`
	Attempts   []Attempt `json:"attempts,omitempty"`
}
type StartAction struct {
	IntentID       string `json:"intent_id"`
	IdempotencyKey string `json:"idempotency_key"`
}
type Receipt struct {
	Schema     string        `json:"schema"`
	Generation string        `json:"generation"`
	PoolID     string        `json:"pool_id"`
	Min        int           `json:"min"`
	Desired    int           `json:"desired"`
	Max        int           `json:"max"`
	Observed   int           `json:"observed"`
	Start      []StartAction `json:"start"`
	Hold       []string      `json:"hold,omitempty"`
}

func Reconcile(s Snapshot) (Receipt, error) {
	if s.Schema != "" && s.Schema != Schema {
		return Receipt{}, fmt.Errorf("unsupported schema %q", s.Schema)
	}
	if s.Generation == "" {
		return Receipt{}, errors.New("generation is required")
	}
	if err := s.Pool.Validate(); err != nil {
		return Receipt{}, err
	}
	m := map[string]Intent{}
	for _, i := range s.Intents {
		if i.ID == "" {
			return Receipt{}, errors.New("intent id required")
		}
		if _, ok := m[i.ID]; ok {
			return Receipt{}, fmt.Errorf("duplicate intent %q", i.ID)
		}
		m[i.ID] = i
	}
	active := map[string]bool{}
	for _, a := range s.Attempts {
		if _, ok := m[a.IntentID]; !ok {
			return Receipt{}, fmt.Errorf("attempt references unknown intent %q", a.IntentID)
		}
		if a.State == AttemptReserved || a.State == AttemptRunning {
			active[a.IntentID] = true
		}
	}
	eligible := []string{}
	for _, i := range s.Intents {
		if !active[i.ID] && (i.State == IntentQueued || (i.State == IntentFailed && i.RetryEligible)) {
			eligible = append(eligible, i.ID)
		}
	}
	sort.Strings(eligible)
	n := s.Pool.Desired - len(active)
	if x := s.Pool.Max - len(active); n > x {
		n = x
	}
	if n < 0 {
		n = 0
	}
	if n > len(eligible) {
		n = len(eligible)
	}
	r := Receipt{Schema: "fak.agentqueue.reconcile.v1", Generation: s.Generation, PoolID: s.Pool.ID, Min: s.Pool.Min, Desired: s.Pool.Desired, Max: s.Pool.Max, Observed: len(active), Start: []StartAction{}}
	for _, id := range eligible[:n] {
		sum := sha256.Sum256([]byte(s.Generation + "\x00" + s.Pool.ID + "\x00" + id))
		r.Start = append(r.Start, StartAction{id, "start:" + hex.EncodeToString(sum[:16])})
	}
	if len(active) >= s.Pool.Desired || len(active) >= s.Pool.Max {
		r.Hold = []string{"AT_CAPACITY"}
	} else if len(eligible) == 0 {
		r.Hold = []string{"QUEUE_EMPTY"}
	}
	return r, nil
}
