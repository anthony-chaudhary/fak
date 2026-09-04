package policy

import (
	"sort"
	"sync"
	"time"
)

// TTLGrant represents a temporary, time-bounded capability widening granted to a holder.
type TTLGrant struct {
	Capability string    `json:"capability"`
	Target     string    `json:"target"`
	GrantedAt  time.Time `json:"granted_at"`
	ExpiresAt  time.Time `json:"expires_at"`
	HolderID   string    `json:"holder_id"`
}

type ttlGrantKey struct {
	holderID   string
	capability string
	target     string
}

// TTLGrantRegistry manages temporary capability leases and their lifecycle.
type TTLGrantRegistry struct {
	mu     sync.RWMutex
	grants map[ttlGrantKey]*TTLGrant
	now    func() time.Time
}

// NewTTLGrantRegistry initializes an empty temporary capability grant registry.
func NewTTLGrantRegistry() *TTLGrantRegistry {
	return &TTLGrantRegistry{
		grants: make(map[ttlGrantKey]*TTLGrant),
		now:    time.Now,
	}
}

func (r *TTLGrantRegistry) timeNow() time.Time {
	if r.now != nil {
		return r.now()
	}
	return time.Now()
}

// Grant creates or refreshes a temporary capability lease for the specified holder.
func (r *TTLGrantRegistry) Grant(holderID string, capability string, target string, ttl time.Duration) *TTLGrant {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := r.timeNow()
	expiresAt := now.Add(ttl)
	g := &TTLGrant{
		HolderID:   holderID,
		Capability: capability,
		Target:     target,
		GrantedAt:  now,
		ExpiresAt:  expiresAt,
	}

	k := ttlGrantKey{
		holderID:   holderID,
		capability: capability,
		target:     target,
	}
	r.grants[k] = g

	cp := *g
	return &cp
}

// IsGranted returns true iff the grant exists and time.Now().Before(grant.ExpiresAt).
// If expired, returns false.
func (r *TTLGrantRegistry) IsGranted(holderID string, capability string, target string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	k := ttlGrantKey{
		holderID:   holderID,
		capability: capability,
		target:     target,
	}
	g, ok := r.grants[k]
	if !ok {
		return false
	}
	return r.timeNow().Before(g.ExpiresAt)
}

// Revoke removes a grant for the specified holder, capability, and target.
// Returns true if a grant existed and was removed, or false otherwise.
func (r *TTLGrantRegistry) Revoke(holderID string, capability string, target string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	k := ttlGrantKey{
		holderID:   holderID,
		capability: capability,
		target:     target,
	}
	if _, ok := r.grants[k]; !ok {
		return false
	}
	delete(r.grants, k)
	return true
}

// PruneExpired removes all expired grants and returns the count of pruned items.
func (r *TTLGrantRegistry) PruneExpired() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := r.timeNow()
	pruned := 0
	for k, g := range r.grants {
		if !now.Before(g.ExpiresAt) {
			delete(r.grants, k)
			pruned++
		}
	}
	return pruned
}

// ActiveGrants returns all non-expired grants for the given holderID, sorted stably.
func (r *TTLGrantRegistry) ActiveGrants(holderID string) []TTLGrant {
	r.mu.RLock()
	defer r.mu.RUnlock()

	now := r.timeNow()
	res := make([]TTLGrant, 0)
	for k, g := range r.grants {
		if k.holderID == holderID && now.Before(g.ExpiresAt) {
			res = append(res, *g)
		}
	}

	sort.Slice(res, func(i, j int) bool {
		if res[i].Capability != res[j].Capability {
			return res[i].Capability < res[j].Capability
		}
		if res[i].Target != res[j].Target {
			return res[i].Target < res[j].Target
		}
		return res[i].ExpiresAt.Before(res[j].ExpiresAt)
	})

	return res
}
