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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
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
	// AgentUUID is the STABLE Claude Code session UUID (the transcript id, e.g. the value of
	// CLAUDE_CODE_SESSION_ID) this guard session runs under. The descriptor's own ID is the
	// VOLATILE agent-claude-<pid>-<hash> trace id — its pid rotates across restarts — but a
	// wip checkpoint (wipref) stamps this STABLE UUID instead. Carrying the UUID here lets a
	// liveness reader JOIN a checkpoint's stamped session to a live guard session that the
	// volatile trace id could never match (#5343, prerequisite for the #5340 collection cut).
	// OPTIONAL and additive: empty on a legacy blob published by a prior binary, or when no
	// Claude session env is set; omitempty keeps such a record byte-identical on the wire and
	// an older reader simply ignores the unknown key.
	AgentUUID string `json:"agent_uuid,omitempty"`
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
	return s.putBlobRef(ctx, d.Ref(), d)
}

// RemoveSession deletes refs/fak/locks/session-<id> — the stop/expire side of the
// lifecycle. Idempotent: a missing ref is not an error (the session is already gone, the
// desired post-state holds), exactly like Release. It uses `git update-ref -d` on the
// named side ref only, never a branch/HEAD.
func (s *Store) RemoveSession(ctx context.Context, id string) error {
	if !validSessionID(id) {
		return fmt.Errorf("leaseref: invalid session id %q", id)
	}
	return s.deleteRef(ctx, refPrefix+sessionPrefix+id)
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
	ds, err := s.readAllSessions(ctx)
	if err != nil {
		return nil, err
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i].ID < ds[j].ID })
	return ds, nil
}

// readAllSessions reads every session descriptor under refs/fak/locks/session-*, preferring
// ONE batched `git cat-file --batch` process (batchReadSessions) over the O(N)-spawn per-ref
// reader. This is the #5355 Half B hot-path cut: `fak leaseref audit` folds LiveSessions ->
// ListSessions over the ~14k-ref backlog, and the per-ref `cat-file blob` spawn (one git
// process PER ref, ~594s total) was the dominant cost. It degrades to the per-ref listRefs
// reader whenever no stdin seam is wired (a NewWithRunner store, the injected-Runner test
// path that carries no stdin) OR the batch invocation is unavailable/fails — exactly the way
// reap.go degrades its batched delete: the batch is an OPTIMIZATION, never a correctness
// dependency, and the fallback reads every ref rather than dropping any. New/NewInDir wire
// the real gitStdinRunner, so the production audit takes the batched path.
func (s *Store) readAllSessions(ctx context.Context) ([]SessionDescriptor, error) {
	if s.runStdin != nil {
		if ds, ok := s.batchReadSessions(ctx); ok {
			return ds, nil
		}
		// The batch invocation was unavailable or refused (git not executable, a broken
		// --batch stream) — degrade to the proven per-ref reader below, which reads every
		// ref and surfaces a genuine git-not-executable as a clean error (never a silent drop).
	}
	return listRefs(ctx, s, isSessionRef, s.readSessionRef)
}

// batchReadSessions reads EVERY session descriptor through a SINGLE `git cat-file --batch`
// process instead of one `cat-file blob` spawn per ref — the read-side mirror of reap.go's
// batched `git update-ref --stdin` delete (#5355 Half B). It lists the shared namespace once
// with for-each-ref, keeps only the session refs (the namespace split — lock leases and
// intent leases are EXCLUDED), feeds those ref names on stdin to one cat-file --batch, and
// unmarshals each streamed blob. It preserves readSessionRef's semantics EXACTLY: a blob that
// does not JSON-parse (forward-incompatible / corrupt), and a missing/absent object in the
// stream, are both SKIPPED — one bad or absent entry must never blind the whole view — and an
// id-less blob has its ID filled from the ref name. The bool result is the reap.go idiom: ok
// == true means the batch produced the authoritative view (an empty namespace is a valid
// empty view); ok == false signals the batch was unavailable/refused so readAllSessions must
// degrade to the per-ref reader instead of dropping refs.
func (s *Store) batchReadSessions(ctx context.Context) ([]SessionDescriptor, bool) {
	out, code, err := s.run(ctx, s.dir, "for-each-ref", "--format=%(refname)", refPrefix)
	if err != nil {
		return nil, false // git not executable -> let the per-ref path surface the clean error
	}
	if code != 0 {
		return nil, true // absent/empty namespace is a valid empty view, not an error (matches listRefs)
	}
	var refs []string
	for _, line := range strings.Split(out, "\n") {
		if ref := strings.TrimSpace(line); isSessionRef(ref) {
			refs = append(refs, ref)
		}
	}
	if len(refs) == 0 {
		return nil, true // no session refs -> empty view, and no need to spawn cat-file at all
	}
	var stdin strings.Builder
	for _, ref := range refs {
		// Every ref is a validated refs/fak/locks/session-* segment (validSessionID forbids
		// whitespace and ref-special bytes), so one ref per line is unambiguous — the same
		// safety batchDeleteRefs relies on for its update-ref --stdin payload.
		stdin.WriteString(ref)
		stdin.WriteByte('\n')
	}
	stream, code, err := s.runStdin(ctx, s.dir, stdin.String(), "cat-file", "--batch")
	if err != nil || code != 0 {
		return nil, false // git not executable or a refused/broken batch -> degrade to per-ref
	}
	return parseSessionBatch(stream, refs), true
}

// parseSessionBatch parses a `git cat-file --batch` stream into descriptors, zipping each
// record to the ref that produced it by POSITION — cat-file --batch emits exactly one record
// per input line, in the order fed. Each record is either a content header
// `<oid> <type> <size>\n` followed by <size> payload bytes and a trailing \n, or a
// `<object> missing\n` status line with no payload. The payload is read by its declared byte
// COUNT (not line-splitting) so a blob is decoded whole regardless of embedded newlines. A
// missing object, and a blob whose payload does not JSON-parse, are both SKIPPED — the same
// "absence and corruption never blind the whole view" rule the per-ref path's callers apply.
// An id-less blob has its ID filled from the ref name (the basename minus the session-
// prefix), identical to readSessionRef.
func parseSessionBatch(stream string, refs []string) []SessionDescriptor {
	data := []byte(stream)
	var ds []SessionDescriptor
	pos := 0
	for _, ref := range refs {
		nl := bytes.IndexByte(data[pos:], '\n')
		if nl < 0 {
			break // truncated stream: no header line left to read
		}
		header := string(data[pos : pos+nl])
		pos += nl + 1
		fields := strings.Fields(header)
		// A `<object> missing` (or `<object> ambiguous`) status line carries NO payload: it has
		// fewer than three fields (no <type> <size>), so skip it and advance to the next ref's
		// record. Only a three-field `<oid> <type> <size>` header is followed by payload bytes.
		size, ok := blobSize(fields)
		if !ok {
			continue
		}
		if pos+size > len(data) {
			break // header claims more bytes than the stream holds: truncated, stop safely
		}
		payload := data[pos : pos+size]
		pos += size
		if pos < len(data) && data[pos] == '\n' {
			pos++ // consume the trailing newline cat-file emits after each object's payload
		}
		var d SessionDescriptor
		if json.Unmarshal(payload, &d) != nil {
			continue // skip a forward-incompatible / corrupt blob, don't fail the whole view
		}
		if d.ID == "" {
			d.ID = strings.TrimPrefix(strings.TrimPrefix(ref, refPrefix), sessionPrefix)
		}
		ds = append(ds, d)
	}
	return ds
}

// blobSize reports the payload byte count of a cat-file --batch header when it is a
// content record (`<oid> <type> <size>`), returning ok=false for a status line like
// `<object> missing` that carries no payload. Only a three-field header whose last field
// parses as a non-negative int is a content record; anything else is a no-payload line.
func blobSize(fields []string) (int, bool) {
	if len(fields) != 3 {
		return 0, false
	}
	n, err := strconv.Atoi(fields[2])
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
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
	live, expired = liveExpire(all, now, func(d SessionDescriptor) string { return d.ID })
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
