// Package servicelease is the lease / generation / incarnation fencing layer
// over the fak.service.v1 contract (#4752, parent #4748, foundation #4749).
//
// An OS service manager can restart a local process, but it cannot decide
// whether a disconnected REMOTE node still owns a leased workload: a
// control-plane retry during a partition can create duplicate agents, bench
// jobs, tunnels, or model serves. This leaf gives the control plane the four
// durable facts that make that decision safe, and the fencing rules that keep
// two owners from ever being simultaneously valid:
//
//   - Generation — the durable desired-state generation of a workload. Every
//     intent change bumps it; a fencing token minted under an older generation
//     is refused everywhere (fence-on-generation-change).
//   - Incarnation — one boot of one node (node name + boot ID). Recording a
//     new boot ID supersedes the old incarnation forever: a stale incarnation
//     cannot renew a lease and cannot publish completion.
//   - Lease — grants exactly one incarnation ownership of one workload until
//     an explicit logical expiry, carrying the fencing token and the last
//     acknowledged checkpoint.
//   - FencingToken — (generation, lease sequence). Every effectful workload
//     claim must present one; the table refuses any token that is not the
//     newest granted.
//
// The safety split the leaf encodes: LOCAL restart under the SAME incarnation
// never needs the controller (offline recovery — LocalRestartAllowed is a pure
// function of the node's own lease copy), while REMOTE reassignment is refused
// until the lease expires or the holder's incarnation is known superseded
// (RemoteReassignAllowed). A partitioned old owner may keep its process
// running, but its token goes stale the instant a new owner is granted, so its
// renewals and completion publishes are refused — it is running, not valid.
//
// The leaf is pure and deterministic: all time is an explicit logical
// millisecond argument, the caller owns persistence (Table state is plain
// JSON-marshalable data) and I/O, and plans are dry-run JSON documents —
// nothing here mutates native service state. Restart pacing (bounded backoff,
// circuit-open) stays in servicespec.RestartPolicy; this leaf decides WHO may
// own, not how fast to retry.
package servicelease

import (
	"errors"
	"fmt"
)

// Generation is the durable desired-state generation of one workload. It only
// increases; a fencing token minted under an older generation is stale.
type Generation uint64

// Incarnation identifies one boot of one node. Two incarnations are the same
// only if BOTH fields match: the same node after a reboot is a different,
// superseding incarnation.
type Incarnation struct {
	Node   string `json:"node"`
	BootID string `json:"boot_id"`
}

// Zero reports whether the incarnation is unset.
func (i Incarnation) Zero() bool { return i.Node == "" && i.BootID == "" }

// FencingToken accompanies every effectful workload claim. It is stale — and
// refused — the moment the workload's generation is bumped or a newer lease
// sequence is granted.
type FencingToken struct {
	Generation Generation `json:"generation"`
	LeaseSeq   uint64     `json:"lease_seq"`
}

// Checkpoint is the last acknowledged progress marker for a workload. Seq is
// monotonic per workload; a publish that would move it backwards is refused.
type Checkpoint struct {
	Seq      uint64 `json:"seq"`
	ID       string `json:"id,omitempty"`
	AtUnixMS int64  `json:"at_unix_ms,omitempty"`
}

// Lease grants one incarnation ownership of one workload until ExpiresMS
// (logical milliseconds, same clock the caller passes as now). The embedded
// token is the ONLY currently-valid fencing token for the workload.
type Lease struct {
	Workload   string       `json:"workload"`
	Holder     Incarnation  `json:"holder"`
	Token      FencingToken `json:"token"`
	GrantedMS  int64        `json:"granted_ms"`
	ExpiresMS  int64        `json:"expires_ms"`
	Checkpoint Checkpoint   `json:"checkpoint"`
}

// Refusal vocabulary. Every fencing refusal is one of these sentinels so a
// caller (and a test) can bind on the class, not on message text.
var (
	// ErrStaleIncarnation — the claimant's boot ID is not the node's current
	// recorded incarnation. A stale incarnation can NEVER renew or publish.
	ErrStaleIncarnation = errors.New("servicelease: stale incarnation")
	// ErrFenced — the presented fencing token is not the newest granted
	// (older generation or older lease sequence).
	ErrFenced = errors.New("servicelease: fenced: token is not the newest granted")
	// ErrLeaseHeld — another incarnation holds a still-valid lease.
	ErrLeaseHeld = errors.New("servicelease: lease held by another valid owner")
	// ErrNotHolder — the claimant presents a token for a lease it does not hold.
	ErrNotHolder = errors.New("servicelease: claimant is not the lease holder")
	// ErrLeaseExpired — the lease lapsed; the holder must re-acquire.
	ErrLeaseExpired = errors.New("servicelease: lease expired; re-acquire")
	// ErrCheckpointRegression — a publish tried to move the checkpoint backwards.
	ErrCheckpointRegression = errors.New("servicelease: checkpoint sequence moved backwards")
)

// Table is the durable lease/fencing record: desired generation per workload,
// current incarnation per node, and the newest lease per workload. It is pure
// state — the caller owns persistence and locking; every method takes the
// logical clock explicitly, so replaying the same calls yields the same table.
type Table struct {
	Generations  map[string]Generation  `json:"generations"`
	Incarnations map[string]Incarnation `json:"incarnations"`
	Leases       map[string]*Lease      `json:"leases"`
	// LeaseTTLMS is how long a grant or renewal is valid. Zero uses DefaultLeaseTTLMS.
	LeaseTTLMS int64 `json:"lease_ttl_ms,omitempty"`
	// NextSeq is the next lease sequence to grant (monotonic across all workloads,
	// so any two leases are totally ordered).
	NextSeq uint64 `json:"next_seq"`
}

// DefaultLeaseTTLMS is the lease validity window when Table.LeaseTTLMS is zero.
const DefaultLeaseTTLMS int64 = 30000

// NewTable returns an empty table with the given lease TTL (0 = default).
func NewTable(leaseTTLMS int64) *Table {
	return &Table{
		Generations:  map[string]Generation{},
		Incarnations: map[string]Incarnation{},
		Leases:       map[string]*Lease{},
		LeaseTTLMS:   leaseTTLMS,
	}
}

func (t *Table) ttl() int64 {
	if t.LeaseTTLMS > 0 {
		return t.LeaseTTLMS
	}
	return DefaultLeaseTTLMS
}

// RecordIncarnation registers a node boot. Recording a NEW boot ID supersedes
// the previous incarnation permanently — every outstanding claim by the old
// incarnation becomes ErrStaleIncarnation. Recording the same boot ID again is
// a no-op. It reports whether the incarnation changed.
func (t *Table) RecordIncarnation(inc Incarnation) bool {
	cur, ok := t.Incarnations[inc.Node]
	if ok && cur == inc {
		return false
	}
	t.Incarnations[inc.Node] = inc
	return true
}

// currentIncarnation reports whether inc is the node's current recorded boot.
func (t *Table) currentIncarnation(inc Incarnation) bool {
	cur, ok := t.Incarnations[inc.Node]
	return ok && cur == inc
}

// BumpGeneration records a desired-state change for the workload and returns
// the new generation. Every token minted under the old generation is fenced
// from this call on (fence-on-generation-change).
func (t *Table) BumpGeneration(workload string) Generation {
	g := t.Generations[workload] + 1
	t.Generations[workload] = g
	return g
}

// Acquire grants the workload lease to a current incarnation. It refuses a
// stale incarnation, and refuses while ANOTHER incarnation holds a still-valid
// lease (ErrLeaseHeld) — remote takeover must wait for expiry or supersession
// (see RemoteReassignAllowed). The holder itself may re-acquire at any time
// (same owner, fresh token). The granted token carries the workload's current
// generation and a monotonic lease sequence; any previously granted token is
// stale from this call on.
func (t *Table) Acquire(workload string, by Incarnation, nowMS int64) (*Lease, error) {
	if !t.currentIncarnation(by) {
		return nil, fmt.Errorf("%w: %s@%s", ErrStaleIncarnation, by.Node, by.BootID)
	}
	if cur, ok := t.Leases[workload]; ok && cur.Holder != by && t.leaseValid(cur, nowMS) {
		return nil, fmt.Errorf("%w: %s held by %s@%s until %d", ErrLeaseHeld, workload, cur.Holder.Node, cur.Holder.BootID, cur.ExpiresMS)
	}
	t.NextSeq++
	l := &Lease{
		Workload:  workload,
		Holder:    by,
		Token:     FencingToken{Generation: t.Generations[workload], LeaseSeq: t.NextSeq},
		GrantedMS: nowMS,
		ExpiresMS: nowMS + t.ttl(),
	}
	if prev, ok := t.Leases[workload]; ok {
		l.Checkpoint = prev.Checkpoint // progress survives ownership changes
	}
	t.Leases[workload] = l
	return l, nil
}

// fence validates one effectful claim: the claimant must be the node's current
// incarnation, must hold the workload's newest lease, and must present exactly
// the newest token (current generation AND current lease sequence). This is
// the single chokepoint behind Renew and PublishCompletion.
func (t *Table) fence(workload string, by Incarnation, tok FencingToken) (*Lease, error) {
	if !t.currentIncarnation(by) {
		return nil, fmt.Errorf("%w: %s@%s", ErrStaleIncarnation, by.Node, by.BootID)
	}
	l, ok := t.Leases[workload]
	if !ok || l.Holder != by {
		return nil, fmt.Errorf("%w: %s", ErrNotHolder, workload)
	}
	if tok.Generation != t.Generations[workload] || tok != l.Token {
		return nil, fmt.Errorf("%w: presented gen=%d seq=%d, current gen=%d seq=%d",
			ErrFenced, tok.Generation, tok.LeaseSeq, t.Generations[workload], l.Token.LeaseSeq)
	}
	return l, nil
}

// Renew extends a still-valid lease held by a current incarnation presenting
// the newest token. An expired lease is not renewable (ErrLeaseExpired) — the
// holder must Acquire again, so a long-partitioned owner cannot resurrect its
// old grant after the table has moved on.
func (t *Table) Renew(workload string, by Incarnation, tok FencingToken, nowMS int64) (*Lease, error) {
	l, err := t.fence(workload, by, tok)
	if err != nil {
		return nil, err
	}
	if !t.leaseValid(l, nowMS) {
		return nil, fmt.Errorf("%w: %s at %d (expired %d)", ErrLeaseExpired, workload, nowMS, l.ExpiresMS)
	}
	l.ExpiresMS = nowMS + t.ttl()
	return l, nil
}

// PublishCompletion records an acknowledged checkpoint (completion is the
// final checkpoint) under full fencing: a stale incarnation or a fenced token
// cannot publish, and the checkpoint sequence may never move backwards.
// Expiry alone does NOT refuse a publish: if no newer lease was granted, the
// presented token is still the newest and the work provably happened under it.
func (t *Table) PublishCompletion(workload string, by Incarnation, tok FencingToken, cp Checkpoint) error {
	l, err := t.fence(workload, by, tok)
	if err != nil {
		return err
	}
	if cp.Seq < l.Checkpoint.Seq {
		return fmt.Errorf("%w: %d < %d", ErrCheckpointRegression, cp.Seq, l.Checkpoint.Seq)
	}
	l.Checkpoint = cp
	return nil
}

// leaseValid reports whether the lease is unexpired AND its holder is still
// the node's current incarnation AND its token generation is current. A lease
// held by a superseded incarnation or an outdated generation is dead even
// before its clock expiry — its holder could never renew it.
func (t *Table) leaseValid(l *Lease, nowMS int64) bool {
	return nowMS < l.ExpiresMS &&
		t.currentIncarnation(l.Holder) &&
		l.Token.Generation == t.Generations[l.Workload]
}

// ValidOwner returns the single currently-valid owner of the workload, or a
// zero Incarnation if there is none. By construction there is at most one:
// validity requires holding the ONE newest lease with the ONE newest token.
func (t *Table) ValidOwner(workload string, nowMS int64) Incarnation {
	if l, ok := t.Leases[workload]; ok && t.leaseValid(l, nowMS) {
		return l.Holder
	}
	return Incarnation{}
}

// WouldAccept reports whether an effectful claim (renew-shaped) by the given
// incarnation with the given token would be accepted right now. The partition
// simulation uses this as the "valid owner" witness: a node whose claims would
// be refused is running, not owning.
func (t *Table) WouldAccept(workload string, by Incarnation, tok FencingToken, nowMS int64) bool {
	l, err := t.fence(workload, by, tok)
	if err != nil {
		return false
	}
	return t.leaseValid(l, nowMS)
}

// LocalRestartAllowed reports whether a node may restart the workload's LOCAL
// process without any controller contact. It is a pure function of the node's
// own lease copy and its own incarnation — usable fully offline: restarting
// under the SAME incarnation that holds the lease changes nothing about
// ownership, so it is always safe. A rebooted node (new boot ID) fails this
// check and must re-acquire through the table.
func LocalRestartAllowed(l *Lease, self Incarnation) bool {
	return l != nil && !self.Zero() && l.Holder == self
}

// RemoteReassignAllowed reports whether the control plane may grant the
// workload to a DIFFERENT node/incarnation right now, and the reason class.
// Reassignment is allowed only when fencing already guarantees the old owner
// cannot act: no lease, holder incarnation superseded (reboot recorded),
// generation moved on, or clock expiry. While a valid lease stands, the answer
// is no — even if the holder is unreachable (a partitioned owner may still be
// running; wait out the lease instead of creating a second owner).
func RemoteReassignAllowed(t *Table, workload string, nowMS int64) (bool, string) {
	l, ok := t.Leases[workload]
	if !ok {
		return true, "no-lease"
	}
	if !t.currentIncarnation(l.Holder) {
		return true, "incarnation-superseded"
	}
	if l.Token.Generation != t.Generations[workload] {
		return true, "generation-bumped"
	}
	if nowMS >= l.ExpiresMS {
		return true, "lease-expired"
	}
	return false, "lease-valid"
}
