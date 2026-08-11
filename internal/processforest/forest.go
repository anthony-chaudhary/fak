// Package processforest stores durable logical process ownership independent of PID ancestry.
package processforest

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

const Schema = "fak-process-forest/1"

type State string

const (
	StateActive    State = "active"
	StateCompleted State = "completed"
	StateFailed    State = "failed"
	StateLost      State = "lost"
	StateReaped    State = "reaped"
)

type ProcessObservation struct {
	HostID     string    `json:"host_id,omitempty"`
	PID        int       `json:"pid,omitempty"`
	StartedAt  time.Time `json:"started_at,omitempty"`
	ObservedAt time.Time `json:"observed_at,omitempty"`
}

type Member struct {
	Schema         string             `json:"schema"`
	ForestID       string             `json:"forest_id"`
	MemberID       string             `json:"member_id"`
	ParentMemberID string             `json:"parent_member_id,omitempty"`
	RootAuthority  string             `json:"root_authority"`
	AdapterKind    string             `json:"adapter_kind"`
	Generation     uint64             `json:"generation"`
	State          State              `json:"state"`
	Reason         string             `json:"reason,omitempty"`
	Observation    ProcessObservation `json:"observation,omitempty"`
	CreatedAt      time.Time          `json:"created_at"`
	UpdatedAt      time.Time          `json:"updated_at"`
	TerminalAt     time.Time          `json:"terminal_at,omitempty"`
}

type Snapshot struct {
	Schema        string   `json:"schema"`
	ForestID      string   `json:"forest_id"`
	RootAuthority string   `json:"root_authority"`
	Members       []Member `json:"members"`
}

type Registry struct {
	mu           sync.Mutex
	members      map[string]Member
	processOwner map[string]string
}

func NewRegistry() *Registry {
	return &Registry{members: map[string]Member{}, processOwner: map[string]string{}}
}

func (r *Registry) Register(m Member) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.put(m, false)
}

func (r *Registry) Update(m Member) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.put(m, true)
}

func (r *Registry) Reparent(forestID, memberID, parentID string, generation uint64, at time.Time) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.members[key(forestID, memberID)]
	if !ok {
		return errors.New("member not found")
	}
	if generation <= m.Generation {
		return fmt.Errorf("stale generation %d <= %d", generation, m.Generation)
	}
	m.ParentMemberID, m.Generation, m.UpdatedAt = strings.TrimSpace(parentID), generation, at.UTC()
	return r.put(m, true)
}

func (r *Registry) Adopt(forestID, memberID, newParent, authority string, generation uint64, at time.Time) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.members[key(forestID, memberID)]
	if !ok {
		return errors.New("member not found")
	}
	if strings.TrimSpace(authority) == "" || authority != m.RootAuthority {
		return errors.New("adoption authority does not match forest root")
	}
	if generation <= m.Generation {
		return fmt.Errorf("stale generation %d <= %d", generation, m.Generation)
	}
	m.ParentMemberID, m.Generation, m.UpdatedAt = strings.TrimSpace(newParent), generation, at.UTC()
	return r.put(m, true)
}

func (r *Registry) Terminalize(forestID, memberID string, generation uint64, state State, reason string, at time.Time) error {
	if state != StateCompleted && state != StateFailed && state != StateLost && state != StateReaped {
		return errors.New("terminal state required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.members[key(forestID, memberID)]
	if !ok {
		return errors.New("member not found")
	}
	if generation < m.Generation {
		return fmt.Errorf("stale generation %d < %d", generation, m.Generation)
	}
	if m.State == state && m.Generation == generation {
		return nil
	}
	m.State, m.Reason, m.Generation, m.UpdatedAt, m.TerminalAt = state, strings.TrimSpace(reason), generation, at.UTC(), at.UTC()
	return r.put(m, true)
}

func (r *Registry) Snapshot(forestID string) (Snapshot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := Snapshot{Schema: Schema, ForestID: forestID}
	for _, m := range r.members {
		if m.ForestID == forestID {
			out.Members = append(out.Members, m)
			if out.RootAuthority == "" {
				out.RootAuthority = m.RootAuthority
			}
		}
	}
	if len(out.Members) == 0 {
		return Snapshot{}, errors.New("forest not found")
	}
	sort.Slice(out.Members, func(i, j int) bool { return out.Members[i].MemberID < out.Members[j].MemberID })
	return out, nil
}

func (r *Registry) put(m Member, updating bool) error {
	m.ForestID, m.MemberID, m.ParentMemberID = strings.TrimSpace(m.ForestID), strings.TrimSpace(m.MemberID), strings.TrimSpace(m.ParentMemberID)
	m.RootAuthority, m.AdapterKind = strings.TrimSpace(m.RootAuthority), strings.TrimSpace(m.AdapterKind)
	if m.ForestID == "" || m.MemberID == "" || m.RootAuthority == "" || m.AdapterKind == "" {
		return errors.New("forest_id, member_id, root_authority, and adapter_kind are required")
	}
	if m.Schema == "" {
		m.Schema = Schema
	}
	if m.Schema != Schema {
		return errors.New("unsupported schema")
	}
	if m.State == "" {
		m.State = StateActive
	}
	k := key(m.ForestID, m.MemberID)
	old, exists := r.members[k]
	if exists {
		if !updating && sameMember(old, m) {
			return nil
		}
		if m.Generation < old.Generation {
			return fmt.Errorf("stale generation %d < %d", m.Generation, old.Generation)
		}
		if !updating {
			return errors.New("member already registered with different identity")
		}
		if old.RootAuthority != m.RootAuthority {
			return errors.New("root authority is immutable")
		}
	}
	if m.ParentMemberID == m.MemberID {
		return errors.New("member cannot parent itself")
	}
	if m.ParentMemberID != "" {
		p, ok := r.members[key(m.ForestID, m.ParentMemberID)]
		if !ok {
			return errors.New("parent member not found")
		}
		if p.RootAuthority != m.RootAuthority {
			return errors.New("parent authority mismatch")
		}
		if r.wouldCycle(m.ForestID, m.MemberID, m.ParentMemberID) {
			return errors.New("parent edge creates cycle")
		}
	}
	pk := processKey(m.Observation)
	if pk != "" {
		if owner := r.processOwner[pk]; owner != "" && owner != k && isLive(r.members[owner].State) {
			return errors.New("process identity already has live owner")
		}
	}
	if exists {
		oldKey := processKey(old.Observation)
		if oldKey != "" && oldKey != pk && r.processOwner[oldKey] == k {
			delete(r.processOwner, oldKey)
		}
	}
	if m.CreatedAt.IsZero() {
		if exists {
			m.CreatedAt = old.CreatedAt
		} else {
			m.CreatedAt = time.Now().UTC()
		}
	}
	if m.UpdatedAt.IsZero() {
		m.UpdatedAt = m.CreatedAt
	}
	r.members[k] = m
	if pk != "" && isLive(m.State) {
		r.processOwner[pk] = k
	}
	return nil
}
func (r *Registry) wouldCycle(forest, child, parent string) bool {
	for parent != "" {
		if parent == child {
			return true
		}
		p, ok := r.members[key(forest, parent)]
		if !ok {
			return false
		}
		parent = p.ParentMemberID
	}
	return false
}
func key(f, m string) string { return f + "\x00" + m }
func processKey(o ProcessObservation) string {
	if o.HostID == "" || o.PID <= 0 || o.StartedAt.IsZero() {
		return ""
	}
	return fmt.Sprintf("%s:%d@%s", o.HostID, o.PID, o.StartedAt.UTC().Format(time.RFC3339Nano))
}
func sameMember(a, b Member) bool {
	return a.ForestID == b.ForestID && a.MemberID == b.MemberID && a.ParentMemberID == b.ParentMemberID && a.RootAuthority == b.RootAuthority && a.AdapterKind == b.AdapterKind && a.Generation == b.Generation && a.Observation == b.Observation
}
func isLive(s State) bool { return s == StateActive || s == "" }
