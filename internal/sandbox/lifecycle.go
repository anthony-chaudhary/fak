// Adapted from opensandbox-group/OpenSandbox specs/sandbox-lifecycle.yml at
// f72baeae3a7feb11a49cc42f48dba3e93a94600d (Apache-2.0). This is a native
// reimplementation of the sandbox lifecycle state machine and its
// create/list/get/pause/resume/snapshot/renew/delete contract, not a copy.

package sandbox

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

// LifecycleState is the high-level lifecycle state of a managed sandbox.
type LifecycleState string

const (
	LifecyclePending    LifecycleState = "Pending"
	LifecycleRunning    LifecycleState = "Running"
	LifecyclePausing    LifecycleState = "Pausing"
	LifecyclePaused     LifecycleState = "Paused"
	LifecycleResuming   LifecycleState = "Resuming"
	LifecycleStopping   LifecycleState = "Stopping"
	LifecycleTerminated LifecycleState = "Terminated"
	LifecycleFailed     LifecycleState = "Failed"
)

// LifecycleStates lists every state in spec order.
var LifecycleStates = []LifecycleState{
	LifecyclePending, LifecycleRunning, LifecyclePausing, LifecyclePaused,
	LifecycleResuming, LifecycleStopping, LifecycleTerminated, LifecycleFailed,
}

var lifecycleEdges = map[LifecycleState][]LifecycleState{
	LifecyclePending:  {LifecycleRunning, LifecycleFailed},
	LifecycleRunning:  {LifecyclePausing, LifecycleStopping, LifecycleFailed},
	LifecyclePausing:  {LifecyclePaused},
	LifecyclePaused:   {LifecycleResuming, LifecycleStopping, LifecycleFailed},
	LifecycleResuming: {LifecycleRunning, LifecycleFailed},
	LifecycleStopping: {LifecycleTerminated},
}

// LifecycleCanTransition reports whether from -> to is a legal lifecycle edge.
func LifecycleCanTransition(from, to LifecycleState) bool {
	for _, next := range lifecycleEdges[from] {
		if next == to {
			return true
		}
	}
	return false
}

// Terminal reports whether the state has no outgoing edges.
func (s LifecycleState) Terminal() bool {
	return s == LifecycleTerminated || s == LifecycleFailed
}

// LifecycleOp names a lifecycle operation for error reporting.
type LifecycleOp string

const (
	LifecycleOpCreate   LifecycleOp = "create"
	LifecycleOpPause    LifecycleOp = "pause"
	LifecycleOpResume   LifecycleOp = "resume"
	LifecycleOpSnapshot LifecycleOp = "snapshot"
	LifecycleOpRenew    LifecycleOp = "renew"
	LifecycleOpDelete   LifecycleOp = "delete"
	LifecycleOpExpire   LifecycleOp = "expire"
)

var (
	ErrLifecycleIllegalTransition = errors.New("illegal sandbox lifecycle transition")
	ErrLifecycleNotFound          = errors.New("sandbox not found")
	ErrLifecycleInvalidRequest    = errors.New("invalid sandbox lifecycle request")
)

// LifecycleTransitionError reports an operation refused by the current state.
type LifecycleTransitionError struct {
	ID   string
	Op   LifecycleOp
	From LifecycleState
	To   LifecycleState
}

func (e *LifecycleTransitionError) Error() string {
	if e.To == "" {
		return fmt.Sprintf("sandbox %s: %s refused in state %s: %v", e.ID, e.Op, e.From, ErrLifecycleIllegalTransition)
	}
	return fmt.Sprintf("sandbox %s: %s %s -> %s: %v", e.ID, e.Op, e.From, e.To, ErrLifecycleIllegalTransition)
}

func (e *LifecycleTransitionError) Is(target error) bool {
	return target == ErrLifecycleIllegalTransition
}

// LifecycleRuntime performs the side effects behind each lifecycle operation.
type LifecycleRuntime interface {
	Create(ctx context.Context, id string, p Provider, spec Spec) error
	Pause(ctx context.Context, id string) error
	Resume(ctx context.Context, id string) error
	Snapshot(ctx context.Context, id, snapshotID string) error
	Terminate(ctx context.Context, id string) error
}

// LifecycleMinTimeout is the smallest non-zero create timeout accepted.
const LifecycleMinTimeout = 60 * time.Second

// LifecycleConfig configures a LifecycleManager.
type LifecycleConfig struct {
	Registry *Registry
	Runtime  LifecycleRuntime
	// SecureTier, when set, is the minimum isolation tier every sandbox is
	// resolved at; a request for a weaker tier is raised to it, never lowered.
	SecureTier Tier
	// MaxTimeout caps the create timeout; zero means uncapped. Renew is not capped.
	MaxTimeout time.Duration
	Now        func() time.Time
}

// LifecycleCreateRequest describes a sandbox to create.
type LifecycleCreateRequest struct {
	Spec Spec
	// Timeout is the time to live; zero disables expiry and requires explicit Delete.
	Timeout  time.Duration
	Metadata map[string]string
}

// LifecycleSandbox is a point-in-time view of a managed sandbox.
type LifecycleSandbox struct {
	ID               string
	Tier             Tier
	Provider         string
	State            LifecycleState
	Reason           string
	Metadata         map[string]string
	CreatedAt        time.Time
	LastTransitionAt time.Time
	// ExpiresAt is zero when the sandbox never expires.
	ExpiresAt time.Time
}

// LifecycleSnapshotState is the state of a recorded snapshot.
type LifecycleSnapshotState string

const (
	LifecycleSnapshotReady  LifecycleSnapshotState = "Ready"
	LifecycleSnapshotFailed LifecycleSnapshotState = "Failed"
)

// LifecycleSnapshot is a recorded snapshot descriptor; it outlives its source sandbox.
type LifecycleSnapshot struct {
	ID        string
	SandboxID string
	Name      string
	State     LifecycleSnapshotState
	CreatedAt time.Time
}

// LifecycleListFilter selects sandboxes: any listed state AND every metadata pair.
type LifecycleListFilter struct {
	States   []LifecycleState
	Metadata map[string]string
}

type lifecycleRecord struct {
	seq  uint64
	view LifecycleSandbox
	// snapshotting holds the record in Running so the capture sees a stable source.
	snapshotting bool
}

// LifecycleManager is an in-memory sandbox lifecycle state machine.
type LifecycleManager struct {
	cfg       LifecycleConfig
	mu        sync.Mutex
	nextSeq   uint64
	nextSnap  uint64
	records   map[string]*lifecycleRecord
	snapshots []LifecycleSnapshot
}

// NewLifecycleManager validates cfg and returns an empty manager.
func NewLifecycleManager(cfg LifecycleConfig) (*LifecycleManager, error) {
	if cfg.Registry == nil || cfg.Runtime == nil {
		return nil, fmt.Errorf("%w: registry and runtime are required", ErrLifecycleInvalidRequest)
	}
	if cfg.SecureTier != "" && !cfg.SecureTier.Valid() {
		return nil, fmt.Errorf("%w: secure tier %q", ErrLifecycleInvalidRequest, cfg.SecureTier)
	}
	if cfg.MaxTimeout < 0 {
		return nil, fmt.Errorf("%w: negative max timeout", ErrLifecycleInvalidRequest)
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &LifecycleManager{
		cfg:     cfg,
		records: make(map[string]*lifecycleRecord),
	}, nil
}

func (m *LifecycleManager) effectiveTier(requested Tier) Tier {
	secure := m.cfg.SecureTier
	if secure != "" && requested.IsolationLevel() < secure.IsolationLevel() {
		return secure
	}
	return requested
}

// Create resolves the sandbox tier through the registry and provisions it.
// On a runtime failure the sandbox is kept in Failed and the error returned.
func (m *LifecycleManager) Create(ctx context.Context, req LifecycleCreateRequest) (LifecycleSandbox, error) {
	if err := req.Spec.Validate(); err != nil {
		return LifecycleSandbox{}, err
	}
	if req.Timeout < 0 || (req.Timeout > 0 && req.Timeout < LifecycleMinTimeout) {
		return LifecycleSandbox{}, fmt.Errorf("%w: timeout %s must be zero or at least %s", ErrLifecycleInvalidRequest, req.Timeout, LifecycleMinTimeout)
	}
	if m.cfg.MaxTimeout > 0 && req.Timeout > m.cfg.MaxTimeout {
		return LifecycleSandbox{}, fmt.Errorf("%w: timeout %s exceeds max %s", ErrLifecycleInvalidRequest, req.Timeout, m.cfg.MaxTimeout)
	}
	spec := req.Spec
	spec.Tier = m.effectiveTier(spec.Tier)
	provider, err := m.cfg.Registry.ResolveTier(spec.Tier)
	if err != nil {
		return LifecycleSandbox{}, err
	}

	m.mu.Lock()
	m.nextSeq++
	now := m.cfg.Now()
	rec := &lifecycleRecord{seq: m.nextSeq, view: LifecycleSandbox{
		ID:               fmt.Sprintf("sbx-%d", m.nextSeq),
		Tier:             spec.Tier,
		Provider:         provider.Name(),
		State:            LifecyclePending,
		Metadata:         cloneLifecycleMetadata(req.Metadata),
		CreatedAt:        now,
		LastTransitionAt: now,
	}}
	if req.Timeout > 0 {
		rec.view.ExpiresAt = now.Add(req.Timeout)
	}
	m.records[rec.view.ID] = rec
	id := rec.view.ID
	m.mu.Unlock()

	return m.finish(id, LifecycleOpCreate, LifecyclePending, LifecycleRunning,
		m.cfg.Runtime.Create(ctx, id, provider, spec))
}

// Get returns the sandbox with id.
func (m *LifecycleManager) Get(id string) (LifecycleSandbox, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	rec, ok := m.records[id]
	if !ok {
		return LifecycleSandbox{}, lifecycleNotFound(id)
	}
	return rec.snapshotView(), nil
}

// List returns the sandboxes matching f in creation order.
func (m *LifecycleManager) List(f LifecycleListFilter) []LifecycleSandbox {
	m.mu.Lock()
	recs := make([]*lifecycleRecord, 0, len(m.records))
	for _, rec := range m.records {
		if f.matches(rec.view) {
			recs = append(recs, rec)
		}
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].seq < recs[j].seq })
	out := make([]LifecycleSandbox, len(recs))
	for i, rec := range recs {
		out[i] = rec.snapshotView()
	}
	m.mu.Unlock()
	return out
}

func (f LifecycleListFilter) matches(s LifecycleSandbox) bool {
	if len(f.States) > 0 {
		found := false
		for _, st := range f.States {
			if st == s.State {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	for k, v := range f.Metadata {
		if got, ok := s.Metadata[k]; !ok || got != v {
			return false
		}
	}
	return true
}

// Pause moves a Running sandbox through Pausing to Paused.
func (m *LifecycleManager) Pause(ctx context.Context, id string) (LifecycleSandbox, error) {
	return m.drive(ctx, id, LifecycleOpPause, LifecyclePausing, LifecyclePaused, m.cfg.Runtime.Pause)
}

// Resume moves a Paused sandbox through Resuming to Running.
func (m *LifecycleManager) Resume(ctx context.Context, id string) (LifecycleSandbox, error) {
	return m.drive(ctx, id, LifecycleOpResume, LifecycleResuming, LifecycleRunning, m.cfg.Runtime.Resume)
}

// Delete terminates a Running or Paused sandbox through Stopping and forgets it;
// a Terminated or Failed sandbox is forgotten without a runtime call.
func (m *LifecycleManager) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	rec, ok := m.records[id]
	if !ok {
		m.mu.Unlock()
		return lifecycleNotFound(id)
	}
	if rec.view.State.Terminal() {
		delete(m.records, id)
		m.mu.Unlock()
		return nil
	}
	m.mu.Unlock()
	if _, err := m.drive(ctx, id, LifecycleOpDelete, LifecycleStopping, LifecycleTerminated, m.cfg.Runtime.Terminate); err != nil {
		return err
	}
	m.mu.Lock()
	delete(m.records, id)
	m.mu.Unlock()
	return nil
}

// Snapshot records a snapshot of a Running sandbox; the sandbox keeps running and
// other operations on it are refused until the capture returns. A runtime failure records a Failed descriptor and leaves the sandbox unchanged.
func (m *LifecycleManager) Snapshot(ctx context.Context, id, name string) (LifecycleSnapshot, error) {
	m.mu.Lock()
	rec, ok := m.records[id]
	if !ok {
		m.mu.Unlock()
		return LifecycleSnapshot{}, lifecycleNotFound(id)
	}
	if rec.view.State != LifecycleRunning || rec.snapshotting {
		from := rec.view.State
		m.mu.Unlock()
		return LifecycleSnapshot{}, &LifecycleTransitionError{ID: id, Op: LifecycleOpSnapshot, From: from}
	}
	rec.snapshotting = true
	m.nextSnap++
	snap := LifecycleSnapshot{
		ID:        fmt.Sprintf("snap-%d", m.nextSnap),
		SandboxID: id,
		Name:      name,
		CreatedAt: m.cfg.Now(),
	}
	m.mu.Unlock()

	err := m.cfg.Runtime.Snapshot(ctx, id, snap.ID)
	snap.State = LifecycleSnapshotReady
	if err != nil {
		snap.State = LifecycleSnapshotFailed
		err = fmt.Errorf("sandbox %s: snapshot: %w", id, err)
	}
	m.mu.Lock()
	rec.snapshotting = false
	m.snapshots = append(m.snapshots, snap)
	m.mu.Unlock()
	return snap, err
}

// Snapshots returns the recorded snapshots of sandboxID in creation order.
func (m *LifecycleManager) Snapshots(sandboxID string) []LifecycleSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []LifecycleSnapshot
	for _, s := range m.snapshots {
		if s.SandboxID == sandboxID {
			out = append(out, s)
		}
	}
	return out
}

// Renew sets the absolute expiration, which must be after now; it may shorten
// or extend the current one and is refused once the sandbox is Stopping or terminal.
func (m *LifecycleManager) Renew(id string, expiresAt time.Time) (LifecycleSandbox, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	rec, ok := m.records[id]
	if !ok {
		return LifecycleSandbox{}, lifecycleNotFound(id)
	}
	if st := rec.view.State; st == LifecycleStopping || st.Terminal() {
		return LifecycleSandbox{}, &LifecycleTransitionError{ID: id, Op: LifecycleOpRenew, From: st}
	}
	if now := m.cfg.Now(); !expiresAt.After(now) {
		return LifecycleSandbox{}, fmt.Errorf("%w: expiresAt %s is not after now %s", ErrLifecycleInvalidRequest, expiresAt.Format(time.RFC3339Nano), now.Format(time.RFC3339Nano))
	}
	rec.view.ExpiresAt = expiresAt
	return rec.snapshotView(), nil
}

// Sweep terminates every Running or Paused sandbox whose expiry is at or before
// now. Expired sandboxes stay visible as Terminated until deleted.
func (m *LifecycleManager) Sweep(ctx context.Context, now time.Time) ([]string, error) {
	m.mu.Lock()
	var due []*lifecycleRecord
	for _, rec := range m.records {
		v := rec.view
		if v.ExpiresAt.IsZero() || now.Before(v.ExpiresAt) {
			continue
		}
		if LifecycleCanTransition(v.State, LifecycleStopping) {
			due = append(due, rec)
		}
	}
	sort.Slice(due, func(i, j int) bool { return due[i].seq < due[j].seq })
	ids := make([]string, len(due))
	for i, rec := range due {
		ids[i] = rec.view.ID
	}
	m.mu.Unlock()

	var expired []string
	var errs []error
	for _, id := range ids {
		if _, err := m.drive(ctx, id, LifecycleOpExpire, LifecycleStopping, LifecycleTerminated, m.cfg.Runtime.Terminate); err != nil {
			errs = append(errs, err)
			continue
		}
		expired = append(expired, id)
	}
	return expired, errors.Join(errs...)
}

// drive moves id through the in-flight state via the runtime call to done.
func (m *LifecycleManager) drive(ctx context.Context, id string, op LifecycleOp, inflight, done LifecycleState, call func(context.Context, string) error) (LifecycleSandbox, error) {
	m.mu.Lock()
	rec, ok := m.records[id]
	if !ok {
		m.mu.Unlock()
		return LifecycleSandbox{}, lifecycleNotFound(id)
	}
	from := rec.view.State
	if rec.snapshotting || !LifecycleCanTransition(from, inflight) {
		m.mu.Unlock()
		return LifecycleSandbox{}, &LifecycleTransitionError{ID: id, Op: op, From: from, To: inflight}
	}
	rec.setState(inflight, string(op)+"_requested", m.cfg.Now())
	m.mu.Unlock()

	err := call(ctx, id)
	if err != nil && !LifecycleCanTransition(inflight, LifecycleFailed) {
		m.mu.Lock()
		defer m.mu.Unlock()
		if rec, ok := m.records[id]; ok && rec.view.State == inflight {
			rec.setState(from, string(op)+"_failed", m.cfg.Now())
		}
		return LifecycleSandbox{}, fmt.Errorf("sandbox %s: %s: %w", id, op, err)
	}
	return m.finish(id, op, inflight, done, err)
}

// finish commits done, or Failed when runtimeErr is set. A runtime failure goes
// to Failed only where the spec has an in-flight -> Failed edge; otherwise drive
// restores the pre-operation state so the operation can be retried.
func (m *LifecycleManager) finish(id string, op LifecycleOp, inflight, done LifecycleState, runtimeErr error) (LifecycleSandbox, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	rec, ok := m.records[id]
	if !ok {
		return LifecycleSandbox{}, lifecycleNotFound(id)
	}
	target, reason := done, string(op)+"_completed"
	if runtimeErr != nil {
		target, reason = LifecycleFailed, string(op)+"_failed"
	}
	if rec.view.State != inflight || !LifecycleCanTransition(inflight, target) {
		return rec.snapshotView(), &LifecycleTransitionError{ID: id, Op: op, From: rec.view.State, To: target}
	}
	rec.setState(target, reason, m.cfg.Now())
	if runtimeErr != nil {
		return rec.snapshotView(), fmt.Errorf("sandbox %s: %s: %w", id, op, runtimeErr)
	}
	return rec.snapshotView(), nil
}

func (r *lifecycleRecord) setState(s LifecycleState, reason string, at time.Time) {
	r.view.State = s
	r.view.Reason = reason
	r.view.LastTransitionAt = at
}

func (r *lifecycleRecord) snapshotView() LifecycleSandbox {
	v := r.view
	v.Metadata = cloneLifecycleMetadata(v.Metadata)
	return v
}

func cloneLifecycleMetadata(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func lifecycleNotFound(id string) error {
	return fmt.Errorf("%w: %s", ErrLifecycleNotFound, id)
}
