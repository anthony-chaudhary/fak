package session

import (
	"sort"
	"sync"
	"time"
)

// descriptor.go — the DURABLE, addressable index of the live drive state (issue
// #1197, the Pillar-1 keystone of epic #1193). The Table holds the live PCB
// (RUNNING/THROTTLED/PAUSED/DRAINING/STOPPED, Decide, Snapshot) but it is IN-MEMORY
// ONLY: internal/sessionimage §5 fences it — "No persistence yet … a process
// restart re-attaches a session at its defaults." This file closes that fence for
// the live table with a small persisted INDEX of drive state + pointers, NOT the
// transcript (which stays in the provider's / Claude Code's own session store).
//
// THE THREE MOVES.
//   - register-on-start: a Descriptor is written when a session starts, keyed by a
//     stable id, idempotently (a relaunch UPDATES, never duplicates).
//   - update-on-transition: the Descriptor's pcb_state / budget / priority / rev /
//     generation / last-seen are re-stamped on every drive change, so the persisted
//     state tracks the live Table.
//   - TTL-GC: a stale Descriptor (no update within its TTL) is swept at read time —
//     never reaping a live lease (one whose last_seen is still within TTL).
//
// WHAT IT IS, WHAT IT IS NOT. The Descriptor is a PROJECTION of State (it reuses the
// existing RunState / Budget / Priority / Rev / Generation — it adds no drive field)
// plus the durable pointers a restart needs to re-attach (id, host, last_seen, ttl).
// It is the load-time complement of Table.Restore: Restore re-attaches ONE persisted
// State into the live table; the Registry is the persisted catalog that says WHICH
// sessions existed and at WHAT state, so a restart re-attaches each at its real state,
// not its default.
//
// THE PERSISTENCE SEAM. The Registry never imports a filesystem or sessionimage. It
// writes through a DescriptorStore interface the host wires — a process wires the
// sessionimage-backed store (which composes the session.json writer); the test wires
// the in-memory MemStore. The package stays a foundation leaf (stdlib + the existing
// in-package primitives), off the request path, registering nothing.
//
// THE CLOCK SEAM. Every time-taking method takes an explicit now time.Time, the same
// deterministic-clock posture scheduler.go takes (ReserveKnownComing/ExpireReservations
// are all caller-now) — so register / update / GC / restore are unit-testable to an
// exact sequence with an injected clock, no real time, no sleep.

// DefaultDescriptorTTL is the staleness window a Descriptor is GC'd after when none is
// configured: a descriptor whose LastSeen is older than this (relative to the sweep's
// now) is reaped at read time. It is generous — a live session re-stamps LastSeen on
// every drive change (each Decide, each control verb), so only a session whose process
// is genuinely gone for this long ages out. Per-descriptor TTL overrides it (TTL > 0).
const DefaultDescriptorTTL = 30 * time.Minute

// Descriptor is the small durable index record for one live session — the persisted
// projection of its drive State plus the pointers a restart needs to re-attach it. It
// is deliberately a PROJECTION (it reuses State's RunState/Budget/Priority/Rev/
// Generation, adding no drive field) so the live Table stays the single source of the
// drive and the Descriptor never drifts into a second, competing copy of policy.
//
// The TRANSCRIPT is NOT here (and never will be): the conversation lives in the
// provider's / Claude Code's own store and sessionimage deliberately excludes it for
// privacy. The Descriptor carries DRIVE STATE + POINTERS only.
type Descriptor struct {
	// ID is the stable, addressable key — the guard --session-id (defaulted to a
	// content/host-derived id when unset). Re-registering the same ID is idempotent.
	ID string `json:"id"`
	// Host names where the session runs, so an index spanning hosts stays addressable.
	Host string `json:"host,omitempty"`
	// Trace is the live Table key (State.TraceID) the descriptor mirrors. It MAY differ
	// from ID (a re-homed session keeps its ID but takes a new trace), which is why both
	// are carried — the restart re-attaches the persisted State under this Trace.
	Trace string `json:"trace"`
	// Run is the persisted PCB position. A restart re-attaches at THIS state, not the
	// Running default — a Stopped descriptor restores Stopped, never silently resurrected.
	Run RunState `json:"run"`
	// Budget / Priority / Pace / Generation mirror the live State fields so a restart
	// resumes at the real allotment / rank / throttle / re-continuation depth, not at
	// defaults.
	Budget     Budget `json:"budget"`
	Priority   int    `json:"priority"`
	Pace       Pace   `json:"pace"`
	Generation int    `json:"generation,omitempty"`
	// Reason is the closed token on a Throttled/Stopped descriptor ("" otherwise), carried
	// so a restart of a terminal session still reports WHY it stopped.
	Reason string `json:"reason,omitempty"`
	// Rev is the live State's monotonic revision at the last stamp — the optimistic-
	// concurrency cursor, preserved so an operator UI that held an If-Rev across the
	// restart still composes (the same Rev-preservation discipline as Table.Restore).
	Rev uint64 `json:"rev"`
	// CreatedAt is set once on register; UpdatedAt / LastSeen are re-stamped on every
	// drive change. LastSeen drives the TTL sweep — a descriptor older than its TTL is
	// stale and GC'd. TTL <= 0 means "use DefaultDescriptorTTL".
	CreatedAt time.Time     `json:"created_at"`
	UpdatedAt time.Time     `json:"updated_at"`
	LastSeen  time.Time     `json:"last_seen"`
	TTL       time.Duration `json:"ttl,omitempty"`
}

// effectiveTTL resolves the staleness window: the per-descriptor TTL when positive,
// else the package default. A descriptor never has a zero/negative live window — an
// unset TTL falls back to DefaultDescriptorTTL, so a sweep is always well-defined.
func (d Descriptor) effectiveTTL() time.Duration {
	if d.TTL > 0 {
		return d.TTL
	}
	return DefaultDescriptorTTL
}

// stale reports whether the descriptor has not been re-stamped within its TTL as of
// now — the GC predicate. A descriptor with LastSeen exactly at the boundary is NOT
// stale (strictly-after only), so a session re-stamping at the deadline survives.
func (d Descriptor) stale(now time.Time) bool {
	return now.Sub(d.LastSeen) > d.effectiveTTL()
}

// fromState projects the live State onto the descriptor's drive fields (Run / Budget /
// Priority / Generation / Reason / Rev). It is the single point where a State becomes a
// Descriptor's drive, so register-on-start and update-on-transition cannot diverge in
// which fields they carry. CreatedAt / Host / ID / TTL are owned by the registry, not
// the State, so they are left untouched here.
func descriptorFromState(st State) Descriptor {
	return Descriptor{
		Trace:      st.TraceID,
		Run:        st.Run,
		Budget:     st.Budget,
		Priority:   st.Priority,
		Pace:       st.Pace,
		Generation: st.Generation,
		Reason:     st.Reason,
		Rev:        st.Rev,
	}
}

// RestoredState rebuilds the drive State a restart re-attaches into the live Table from
// this descriptor — the load-time inverse of descriptorFromState. It carries the
// persisted Run/Budget/Priority/Generation/Reason/Rev under the descriptor's Trace, so
// Table.Restore(d.Trace, d.RestoredState()) re-attaches the session at its REAL state,
// not DefaultState's Running/unbounded default. The Rev is preserved (Restore does not
// bump it), so a Snapshot -> Descriptor -> RestoredState -> Restore round-trip is the
// identity on the drive fields.
func (d Descriptor) RestoredState() State {
	return State{
		TraceID:    d.Trace,
		Run:        d.Run,
		Budget:     d.Budget,
		Priority:   d.Priority,
		Pace:       d.Pace,
		Generation: d.Generation,
		Reason:     d.Reason,
		Rev:        d.Rev,
	}
}

// DescriptorStore is the pluggable persistence seam the Registry writes through — the
// only boundary between the in-memory index and durable storage. A production host
// wires a sessionimage-backed store (composing the session.json writer); a test wires
// MemStore. The Registry never imports a filesystem, so the package stays a foundation
// leaf and the persistence backend is swapped without touching the register / update /
// GC core.
//
// Put writes (or overwrites — idempotent by ID) one descriptor. Delete removes one by
// ID (the GC reap). List returns every persisted descriptor (unordered; the Registry
// sorts). All three may return an error the Registry surfaces to its caller; none is
// called under the Registry lock for an unbounded duration (the store owns its own I/O
// concurrency).
type DescriptorStore interface {
	Put(d Descriptor) error
	Delete(id string) error
	List() ([]Descriptor, error)
}

// MemStore is the in-memory DescriptorStore — the test backend and the byte-identical
// reference implementation the durable store must behave like. It is concurrency-safe
// (its own mutex) and keeps the latest descriptor per ID, so a re-Put of the same ID
// overwrites rather than duplicates (the idempotence the Registry relies on). The zero
// MemStore is not usable; construct with NewMemStore.
type MemStore struct {
	mu sync.Mutex
	m  map[string]Descriptor
}

// NewMemStore returns an empty in-memory DescriptorStore.
func NewMemStore() *MemStore {
	return &MemStore{m: map[string]Descriptor{}}
}

// Put writes one descriptor keyed by its ID, overwriting any prior record for that ID
// (idempotent). A blank ID is rejected so a malformed descriptor cannot occupy the
// empty key.
func (s *MemStore) Put(d Descriptor) error {
	if d.ID == "" {
		return errBlankDescriptorID
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.m == nil {
		s.m = map[string]Descriptor{}
	}
	s.m[d.ID] = d
	return nil
}

// Delete removes the descriptor for id. Deleting a missing id is a no-op (the GC reap
// is idempotent — a descriptor swept twice is not an error).
func (s *MemStore) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.m, id)
	return nil
}

// List returns a copy of every persisted descriptor (unordered). The slice is freshly
// allocated, so a caller may sort/mutate it without racing the store.
func (s *MemStore) List() ([]Descriptor, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Descriptor, 0, len(s.m))
	for _, d := range s.m {
		out = append(out, d)
	}
	return out, nil
}

// errBlankDescriptorID is returned when a register/put is attempted with no ID — the
// ID is the addressable key, so a blank one is a programming error, fail-closed.
var errBlankDescriptorID = registryError("descriptor id must be non-empty")

// registryError is the package's small error type, so callers can match on it without
// importing errors for a sentinel. Its value IS its message.
type registryError string

func (e registryError) Error() string { return string(e) }

// Registry is the in-session DURABLE index of live descriptors (issue #1197). It owns
// the three moves — Register (on start), Update (on transition), and GC (TTL sweep at
// read time) — over a pluggable DescriptorStore. It is a thin, pure coordinator: it
// holds NO drive state of its own (the live Table is the source) and never reaches a
// filesystem (the store does). Construct with NewRegistry; a nil receiver is a no-op-
// permissive shell so a host with no registry wired behaves byte-identically.
type Registry struct {
	mu    sync.Mutex
	store DescriptorStore
}

// NewRegistry builds a Registry over store. A nil store is replaced with a fresh
// MemStore so a registry is always usable (the caller does not need a nil check); a
// host that wants persistence wires the durable store explicitly.
func NewRegistry(store DescriptorStore) *Registry {
	if store == nil {
		store = NewMemStore()
	}
	return &Registry{store: store}
}

// Register records a descriptor for a session at start, keyed by id, mirroring the live
// drive st, as of now. It is IDEMPOTENT: re-registering the same id UPDATES the existing
// descriptor (re-stamping UpdatedAt / LastSeen and the drive projection) rather than
// duplicating it, and PRESERVES the original CreatedAt — a relaunch of the same session
// is the same row, aged from its first start. host is recorded once (kept if a relaunch
// passes a blank). ttl <= 0 uses DefaultDescriptorTTL. It returns the stored descriptor.
//
// A nil receiver returns the projected descriptor without persisting, so a loop with no
// registry wired behaves byte-identically to the pre-registry path.
func (r *Registry) Register(id, host string, st State, ttl time.Duration, now time.Time) (Descriptor, error) {
	d := descriptorFromState(st)
	d.ID = id
	d.Host = host
	d.TTL = ttl
	d.CreatedAt = now
	d.UpdatedAt = now
	d.LastSeen = now
	if r == nil {
		return d, nil
	}
	if id == "" {
		return d, errBlankDescriptorID
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	// Idempotent re-register: keep the original CreatedAt and a previously-recorded host
	// (a relaunch that passes a blank host must not erase the known one).
	if prev, ok := r.lookupLocked(id); ok {
		d.CreatedAt = prev.CreatedAt
		if d.Host == "" {
			d.Host = prev.Host
		}
		if d.TTL <= 0 {
			d.TTL = prev.TTL
		}
	}
	if err := r.store.Put(d); err != nil {
		return d, err
	}
	return d, nil
}

// Update re-stamps the descriptor for id from the live drive st as of now — the
// update-on-transition move. It re-projects the drive fields (Run/Budget/Priority/
// Generation/Reason/Rev) and bumps UpdatedAt / LastSeen, so the persisted pcb_state
// tracks the live Table on every control verb / Decide. The durable id / host /
// CreatedAt / TTL are preserved. An Update for an id that was never Registered is
// treated as a register (idempotent create), so a transition observed before an
// explicit register still persists. A nil receiver is a no-op.
func (r *Registry) Update(id string, st State, now time.Time) (Descriptor, error) {
	if r == nil {
		return Descriptor{}, nil
	}
	if id == "" {
		return Descriptor{}, errBlankDescriptorID
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	d := descriptorFromState(st)
	d.ID = id
	d.UpdatedAt = now
	d.LastSeen = now
	if prev, ok := r.lookupLocked(id); ok {
		d.Host = prev.Host
		d.CreatedAt = prev.CreatedAt
		d.TTL = prev.TTL
	} else {
		// Never registered: this Update creates the row, so CreatedAt is now.
		d.CreatedAt = now
	}
	if err := r.store.Put(d); err != nil {
		return d, err
	}
	return d, nil
}

// Get returns the persisted descriptor for id and whether it is present. It does NOT
// sweep (a pure read of one row); a stale row is still returned so a caller can inspect
// it. A nil receiver reports absent.
func (r *Registry) Get(id string) (Descriptor, bool, error) {
	if r == nil {
		return Descriptor{}, false, nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	d, ok := r.lookupLocked(id)
	return d, ok, nil
}

// List returns every NON-stale descriptor as of now, sorted by ID for determinism, AND
// sweeps the stale ones — the read-time TTL-GC. A descriptor whose LastSeen is older
// than its TTL is Deleted from the store and excluded from the result; a fresh one
// (re-stamped within its TTL) is never reaped, so the sweep never kills a live lease.
// The sweep is best-effort: a Delete error does not abort the listing (the row is
// simply omitted and retried on the next sweep). A nil receiver returns no descriptors.
func (r *Registry) List(now time.Time) ([]Descriptor, error) {
	if r == nil {
		return nil, nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	all, err := r.store.List()
	if err != nil {
		return nil, err
	}
	live := all[:0]
	for _, d := range all {
		if d.stale(now) {
			_ = r.store.Delete(d.ID) // best-effort reap; retried next sweep on error
			continue
		}
		live = append(live, d)
	}
	out := make([]Descriptor, len(live))
	copy(out, live)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// GC sweeps stale descriptors as of now and returns how many were reaped — the explicit
// form of the read-time sweep List performs, for a host that wants to age the index on
// a timer without listing. A nil receiver reaps nothing.
func (r *Registry) GC(now time.Time) (int, error) {
	if r == nil {
		return 0, nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	all, err := r.store.List()
	if err != nil {
		return 0, err
	}
	reaped := 0
	for _, d := range all {
		if d.stale(now) {
			if err := r.store.Delete(d.ID); err == nil {
				reaped++
			}
		}
	}
	return reaped, nil
}

// lookupLocked finds one descriptor by id via the store's List (the store is the source
// of truth — the registry holds no copy). Caller holds the lock. It returns the row and
// whether it was present.
func (r *Registry) lookupLocked(id string) (Descriptor, bool) {
	all, err := r.store.List()
	if err != nil {
		return Descriptor{}, false
	}
	for _, d := range all {
		if d.ID == id {
			return d, true
		}
	}
	return Descriptor{}, false
}
