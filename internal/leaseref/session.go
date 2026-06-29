package leaseref

// session.go publishes a LIVE GUARD SESSION as a side ref under the SAME
// refs/fak/locks/* transport the lock leases ride — but in a DISTINCT basename
// namespace, refs/fak/locks/session-<id>, so the fleet can SEE every node's live
// guard sessions cross-machine after an ordinary `git fetch` (epic #1193 Pillar 1,
// issue #1198).
//
// WHAT THIS IS (and is NOT). This is a NEW REF KIND over the EXISTING transport, not
// a new transport: it reuses Store's writeBlob / update-ref / for-each-ref / cat-file
// plumbing and the one injectable Runner seam verbatim. The descriptor it publishes is
// the small {id, host, pcb_state, updated_at, ttl} PROJECTION of a guard session — a
// lightweight POINTER, not the heavy checkpoint (that stays the sessionimage bundle).
//
// THE SAME HONEST BOUNDARY as the lock leases (see the package doc): this is
// DISTRIBUTION / VISIBILITY, not arbitration. Converging the SET of session refs lets
// an operator SEE the fleet's live sessions; it never picks a cross-machine winner or
// influences another node's admission. Publishing is ADDITIVE and FAIL-OPEN at the
// call site — a publish failure (no git, detached, push rejected) must never block the
// local guard session, exactly as the lock-lease side is best-effort.
//
// THE NAMESPACE SPLIT (load-bearing). Both kinds live under refs/fak/locks/, but a
// session ref is refs/fak/locks/session-<id> and a lock lease is refs/fak/locks/<id>
// where <id> is NOT prefixed session-. The lock-lease List/Live/LiveLeases readers
// FILTER OUT session refs and the session readers FILTER OUT lock leases, so the two
// views stay distinct over the one shared `for-each-ref refs/fak/locks/` scan — a
// session is never mistaken for a tree-scoped lease, and vice versa.

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// sessionPrefix is the basename prefix that marks a session descriptor ref apart from a
// lock lease under the shared refs/fak/locks/ namespace. The full ref is
// refs/fak/locks/session-<id>; the stored id is <id> (without this prefix), so a caller
// addresses a session by its plain id and the prefix is an internal namespace marker.
const sessionPrefix = "session-"

// SessionDescriptor is the small projection of a LIVE GUARD SESSION published under
// refs/fak/locks/session-<id>. It is the lightweight POINTER the fleet reads — the
// heavy checkpoint stays the sessionimage bundle. Encoded as one JSON object (the blob
// the ref points at), so the bytes are diffable and git-integrity-checked, exactly like
// the lock-lease Record.
type SessionDescriptor struct {
	ID        string `json:"id"`          // the session id (the ref basename minus the session- prefix)
	Host      string `json:"host"`        // the node this session is live on (machine identity, free-form)
	PCBState  string `json:"pcb_state"`   // the session PCB run-state: RUNNING/THROTTLED/PAUSED/DRAINING/STOPPED
	UpdatedAt int64  `json:"updated_at"`  // unix seconds of the last publish (register or transition)
	TTLSecs   int64  `json:"ttl_seconds"` // lifetime in seconds; 0 means no expiry (an explicit Remove ends it)
}

// Expired reports whether the descriptor is past its TTL at time now, measured from its
// last UpdatedAt. A zero TTL never expires. A live session republishes on each PCB
// transition, refreshing UpdatedAt, so a still-running session keeps its ref fresh; a
// crashed node's descriptor lapses once TTL elapses past the last update and a reader
// drops it from the LIVE view — bounded staleness, not a permanent ghost.
func (d SessionDescriptor) Expired(now time.Time) bool {
	if d.TTLSecs <= 0 {
		return false
	}
	return now.Unix() >= d.UpdatedAt+d.TTLSecs
}

// Ref returns the full ref path this descriptor is stored at: refs/fak/locks/session-<id>.
func (d SessionDescriptor) Ref() string { return refPrefix + sessionPrefix + d.ID }

// isSessionRef reports whether a full ref under refs/fak/locks/ is a SESSION descriptor
// ref (basename starts with session-) rather than a lock lease. The one place the
// namespace split is decided, so both the session readers and the lock-lease readers
// agree on the partition.
func isSessionRef(ref string) bool {
	return strings.HasPrefix(ref, refPrefix+sessionPrefix)
}

// PublishSession writes d under refs/fak/locks/session-<d.ID>. It is the SINGLE operation
// behind both publish-on-register and update-on-transition: an unconditional set of the
// side ref to the current descriptor blob (a ref is not history, so re-pointing it is not
// a force-push). UpdatedAt defaults to now when unset so each republish refreshes the TTL
// clock. It reuses Store.writeBlob + update-ref verbatim — the same plumbing, the same
// "never touch a branch/HEAD, never force" safety as the lock-lease Acquire. Returns the
// written ref on success.
func (s *Store) PublishSession(ctx context.Context, d SessionDescriptor) (string, error) {
	if !validSessionID(d.ID) {
		return "", fmt.Errorf("leaseref: invalid session id %q (must be one safe ref segment)", d.ID)
	}
	if d.UpdatedAt == 0 {
		d.UpdatedAt = time.Now().Unix()
	}

	blob, err := json.Marshal(d)
	if err != nil {
		return "", fmt.Errorf("leaseref: marshal session descriptor: %w", err)
	}

	sha, err := s.writeBlob(ctx, blob)
	if err != nil {
		return "", err
	}

	ref := d.Ref()
	if _, code, err := s.run(ctx, s.dir, "update-ref", ref, sha); err != nil {
		return "", fmt.Errorf("leaseref: git not executable: %w", err)
	} else if code != 0 {
		return "", fmt.Errorf("leaseref: update-ref %s exited %d", ref, code)
	}
	return ref, nil
}

// RemoveSession deletes refs/fak/locks/session-<id> — the stop/expire side of the
// lifecycle. Idempotent: a missing ref is not an error (the session is already gone, the
// desired post-state holds), exactly like Release. It uses `git update-ref -d` on the
// named side ref only, never a branch/HEAD.
func (s *Store) RemoveSession(ctx context.Context, id string) error {
	if !validSessionID(id) {
		return fmt.Errorf("leaseref: invalid session id %q", id)
	}
	ref := refPrefix + sessionPrefix + id
	_, code, err := s.run(ctx, s.dir, "update-ref", "-d", ref)
	if err != nil {
		return fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		// A delete of a ref that does not exist is the already-removed state, not a
		// failure (same rule as Release): treat non-zero as success only after
		// confirming the ref is indeed absent.
		if exists, derr := s.has(ctx, ref); derr == nil && !exists {
			return nil
		}
		return fmt.Errorf("leaseref: update-ref -d %s exited %d", ref, code)
	}
	return nil
}

// GetSession reads back the single descriptor at refs/fak/locks/session-<id>, or
// (zero, false, nil) when no such ref exists — absence is a valid answer (no live
// session by that id on this clone).
func (s *Store) GetSession(ctx context.Context, id string) (SessionDescriptor, bool, error) {
	if !validSessionID(id) {
		return SessionDescriptor{}, false, fmt.Errorf("leaseref: invalid session id %q", id)
	}
	ref := refPrefix + sessionPrefix + id
	exists, err := s.has(ctx, ref)
	if err != nil {
		return SessionDescriptor{}, false, err
	}
	if !exists {
		return SessionDescriptor{}, false, nil
	}
	d, err := s.readSessionRef(ctx, ref)
	if err != nil {
		return SessionDescriptor{}, false, err
	}
	return d, true, nil
}

// ListSessions reads every descriptor under refs/fak/locks/session-*, sorted by id for a
// stable view. This is the source a fleet reader (C7, `fak guard ls --fleet`) folds: after
// an ordinary fetch, a peer's pushed session ref appears here. Lock-lease refs are
// EXCLUDED (the namespace split), and a descriptor whose blob does not parse is SKIPPED (a
// forward-incompatible or corrupt entry must not blind the whole view), not surfaced as an
// error — the same rules as the lock-lease List.
func (s *Store) ListSessions(ctx context.Context) ([]SessionDescriptor, error) {
	out, code, err := s.run(ctx, s.dir, "for-each-ref", "--format=%(refname)", refPrefix)
	if err != nil {
		return nil, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return nil, nil // absent/empty namespace is an empty list, not an error
	}
	var ds []SessionDescriptor
	for _, line := range strings.Split(out, "\n") {
		ref := strings.TrimSpace(line)
		if !isSessionRef(ref) {
			continue // skip lock leases and any non-session ref — keep the views distinct
		}
		d, rerr := s.readSessionRef(ctx, ref)
		if rerr != nil {
			continue // skip an unreadable/forward-incompatible descriptor, don't fail the view
		}
		ds = append(ds, d)
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i].ID < ds[j].ID })
	return ds, nil
}

// LiveSessions reads ListSessions and returns only the descriptors NOT expired at time
// now. This is the fleet-visibility projection: an expired (stale, likely crashed-node)
// descriptor is droppable and must not read as "alive". The expired ids are returned
// alongside so a caller can remove them (each via RemoveSession) — removal is itself a
// converging ref delete, the same reap shape as the lock-lease Live.
func (s *Store) LiveSessions(ctx context.Context, now time.Time) (live []SessionDescriptor, expired []string, err error) {
	all, err := s.ListSessions(ctx)
	if err != nil {
		return nil, nil, err
	}
	for _, d := range all {
		if d.Expired(now) {
			expired = append(expired, d.ID)
			continue
		}
		live = append(live, d)
	}
	return live, expired, nil
}

// validSessionID confines a session id to ONE safe ref segment, the same way validID does
// for a lock id. The session- namespace prefix is supplied by this package, not by the
// caller, so the caller's id must itself be a safe segment AND must not smuggle in a
// leading session- (which would double-prefix the ref); both are enforced here so the
// stored id and the addressed id always agree.
func validSessionID(id string) bool {
	if !validID(id) {
		return false
	}
	// Reject a caller-supplied id that already carries the namespace marker, so
	// PublishSession("session-x") can never collide with PublishSession("x").
	return !strings.HasPrefix(id, sessionPrefix)
}

// readSessionRef reads the blob a session ref points at (`git cat-file blob <ref>`) and
// unmarshals the descriptor. The ID is filled from the ref name (minus the session-
// prefix) so a descriptor always knows its own id even if the blob omitted it.
func (s *Store) readSessionRef(ctx context.Context, ref string) (SessionDescriptor, error) {
	out, code, err := s.run(ctx, s.dir, "cat-file", "blob", ref)
	if err != nil {
		return SessionDescriptor{}, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return SessionDescriptor{}, fmt.Errorf("leaseref: cat-file blob %s exited %d", ref, code)
	}
	var d SessionDescriptor
	if err := json.Unmarshal([]byte(out), &d); err != nil {
		return SessionDescriptor{}, fmt.Errorf("leaseref: unmarshal session descriptor at %s: %w", ref, err)
	}
	if d.ID == "" {
		d.ID = strings.TrimPrefix(strings.TrimPrefix(ref, refPrefix), sessionPrefix)
	}
	return d, nil
}
