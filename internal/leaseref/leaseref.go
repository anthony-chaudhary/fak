// Package leaseref is the CROSS-MACHINE VISIBILITY substrate for fak's leases:
// it persists a lease record under a dedicated refs/fak/locks/<id> ref namespace,
// so lease state rides ordinary `git fetch` / `git push` between clones — the same
// mechanism grite uses with refs/grite/locks.
//
// WHY (the named gap #3, cross-machine atomicity). Every layer that knows about
// leases today is LOCAL-ONLY: internal/safecommit guards a commit with an advisory
// flock (a single-host kernel primitive), and gitgate.CheckCollectiveCommit enforces
// pairwise-disjoint lease trees but only over the leases handed to it in one process
// on one tree. So a peer's live lease on ANOTHER machine is invisible at admission.
// A git ref, by contrast, is exactly the thing git already synchronizes across
// clones. Storing a lease as a ref under refs/fak/locks/* makes machine A's held
// lease readable on machine B after an ordinary fetch.
//
// THE HONEST BOUNDARY (load-bearing — keep it in the docs and the comments):
//   - This is DISTRIBUTION / VISIBILITY, *not* atomic acquisition. Two machines can
//     still both write a lease for overlapping trees in the same fetch window; git's
//     merge converges the SET of refs, it does not arbitrate a winner. The win is
//     that an arbiter can now SEE the conflict at all — final cross-machine race
//     arbitration is out of scope.
//   - fak remains the ENFORCED call-boundary admission layer; this ref store is the
//     cross-machine substrate UNDER it. grite's own leases are advisory ("a substrate
//     cannot enforce coordination on an uncooperative agent"); fak's contribution is
//     the refusal at the call. This package gives that refusal something cross-machine
//     to read.
//   - The local flock (internal/safecommit) stays the FAST same-host path; this ref
//     store is the slower, cross-host tier layered above it. Publishing here is
//     ADDITIVE and best-effort — a leaseref failure never blocks a same-host commit.
//   - A git ref is content-addressed, so a record's bytes are already integrity-checked
//     by git. A signature envelope proving WHICH holder wrote a record would be
//     additive — it is DEFERRED, not part of this package.
//   - (grite reports a 78%->0% conflict-reduction figure from a synthetic N=32 workload.
//     It is deliberately NOT cited as a benefit here — distribution is the claim; that
//     number does not characterize fak's residual collision rate.)
//
// APPEND/DELETE A SIDE REF ONLY (the safety contract): every write targets ONLY
// refs/fak/locks/<id> via `git update-ref`. It NEVER mutates main / HEAD / refs/heads,
// NEVER force-pushes, NEVER touches a commit object. The record blob is written with
// `git hash-object -w` and the ref is pointed at it — both ordinary plumbing. The
// package shells out through ONE injectable Runner seam (the same shape as
// internal/witness), so tests drive the whole algorithm with canned git evidence and
// assert the exact argv issued — no real git, no repo.
package leaseref

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
)

// refPrefix is the dedicated ref namespace every fak side ref lives under. Unlike
// notes (confined to refs/notes/*), an ordinary ref can be named directly, so the
// on-disk ref is EXACTLY refs/fak/locks/<id> — the namespace the issue asked for and
// the one git fetch/push converges across clones. TWO ref kinds share this prefix: a
// lock lease at refs/fak/locks/<id> (this file) and a live guard-session descriptor at
// refs/fak/locks/session-<id> (session.go); the readers on each side filter by the
// session- basename marker so the two views stay distinct.
const refPrefix = "refs/fak/locks/"

// Runner executes a git subcommand in dir and returns (stdout, exitCode, err). It is
// the SAME contract as witness.Runner / safecommit.Runner: err is non-nil ONLY when
// git could not be EXECUTED (git missing); a non-zero exit with git present is reported
// via code, not err. Injectable so tests drive the algorithm with canned evidence.
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// Record is one lease persisted under refs/fak/locks/<id>. It carries the same shape
// safecommit and gitgate.CheckCollectiveCommit already reason about — the leased tree
// globs, the holder identity, the acquisition time, and a TTL — just serialized to a
// ref instead of held in a flock. Encoded as one JSON object (the blob the ref points
// at), so the bytes are diffable and git-integrity-checked.
type Record struct {
	ID          string   `json:"id"`            // the lease id (the ref basename under refs/fak/locks/)
	TreeGlobs   []string `json:"tree_globs"`    // the repo-relative trees this lease covers
	Holder      string   `json:"holder"`        // who holds it (machine/session identity, free-form)
	AcquiredAt  int64    `json:"acquired_unix"` // unix seconds at acquisition (the current generation began)
	TTLSeconds  int64    `json:"ttl_seconds"`   // lifetime in seconds; 0 means no expiry
	Description string   `json:"description,omitempty"`
	// Generation is the monotonic FENCING TOKEN (#906 §3.3 / #1182): it is bumped on every
	// TRANSITION (a new holder reaping + reacquiring an expired lease) and NEVER on a
	// same-holder renew, so a write can be admitted only when the holder's presented
	// generation still matches the live lease's (see fence.go). 0 is the legacy/unfenced
	// value — a record written by the pre-fence blind Acquire carries no generation, and the
	// first AcquireFenced lifts it to 1. omitempty keeps a legacy record byte-identical.
	Generation int64 `json:"generation,omitempty"`
	// RenewedAt is the unix-seconds instant of the last same-holder RENEW (a liveness
	// heartbeat) — the K8s renewTime to AcquiredAt's leaseTransitions. A renew moves the
	// expiry window forward without a transition; 0 means never renewed since acquisition,
	// in which case the window is measured from AcquiredAt exactly as a pre-fence record was.
	RenewedAt int64 `json:"renewed_unix,omitempty"`
}

// effectiveActiveAt is the later of AcquiredAt and RenewedAt — the instant the lease's
// liveness window is measured from. A renew (heartbeat) moves the window forward without a
// transition; a record never renewed (RenewedAt == 0, every pre-fence record) measures from
// AcquiredAt exactly as before, so the renew-awareness is strictly backward-compatible.
func (r Record) effectiveActiveAt() int64 {
	if r.RenewedAt > r.AcquiredAt {
		return r.RenewedAt
	}
	return r.AcquiredAt
}

// Expired reports whether the lease is past its TTL at time now. A zero TTL never
// expires. An expired record is REAPABLE by a peer — a crashed holder's lease is
// bounded, not a permanent deadlock. (Reaping is itself a ref delete that converges
// across clones the same way acquisition does.) The window is measured from the
// renew-aware effectiveActiveAt, so a heartbeated lease stays live; a never-renewed
// record measures from AcquiredAt, identical to the pre-fence behavior.
func (r Record) Expired(now time.Time) bool {
	if r.TTLSeconds <= 0 {
		return false
	}
	return now.Unix() >= r.effectiveActiveAt()+r.TTLSeconds
}

// Ref returns the full ref path this record is stored at.
func (r Record) Ref() string { return refPrefix + r.ID }

// LastActive exposes the lease's dormancy clock (issue #1179, epic #1178): the durable
// LastActiveAt stamp derived from AcquiredAt (unix seconds), from which the lease's
// dormancy band (warm/cool/cold/frozen/ancient) is derivable without I/O via
// r.LastActive().HorizonAt(now). A holder that went dormant past its TTL and returns is
// the worst-case stale writer (#906 §3.3); this is the measured "how long has this lease
// been held without a refresh?" the Phase-2 lease-fence rung (#1182) keys halt-and-
// reacquire on. It reads the renew-aware effectiveActiveAt, so a heartbeated lease reads
// warm and only a genuinely un-refreshed one ages — exactly the "without a refresh"
// quantity the rung wants. Pure: reads only recorded times, adds no field, writes no ref.
// A zero effective time yields the zero (unknown) Stamp, which buckets to Ancient.
func (r Record) LastActive() dormancy.Stamp {
	return dormancy.FromUnix(r.effectiveActiveAt())
}

// Store reads and writes lease records under refs/fak/locks/* through the ONE Runner
// seam. Construct with New (real git) or NewWithRunner (injected evidence). dir is the
// repo to operate in ("" = git's own discovery from the process cwd).
type Store struct {
	run Runner
	dir string
}

// New is the real-git lease store (git discovers the repo from the process cwd).
func New() *Store { return &Store{run: gitRunner} }

// NewInDir is the real-git lease store operating in dir — the repo whose
// refs/fak/locks/* namespace it reads and writes. dir == "" is identical to New()
// (git discovers the repo from the process cwd). The CLI surface uses this to honor
// a --dir flag without exposing the unexported real-git runner.
func NewInDir(dir string) *Store { return &Store{run: gitRunner, dir: dir} }

// NewWithRunner injects a Runner + dir — the SAME seam as the production path, so a
// test exercises the whole acquire/read/reap algorithm with no real git.
func NewWithRunner(r Runner, dir string) *Store { return &Store{run: r, dir: dir} }

// validID rejects an id that cannot be a single ref-path segment under refs/fak/locks/.
// It must be non-empty, carry no path separator (so it stays ONE segment, never escaping
// the namespace), and contain no whitespace / ref-special bytes git's check-ref-format
// would reject. This is the namespace-confinement guard: an id is structurally unable to
// target a ref outside refs/fak/locks/.
func validID(id string) bool {
	if id == "" || len(id) > 200 {
		return false
	}
	if strings.HasPrefix(id, "-") || strings.HasPrefix(id, ".") {
		return false // a leading dash misparses as a flag; a leading dot is ref-illegal
	}
	for _, c := range []byte(id) {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '-' || c == '_' || c == '.':
		default:
			return false // no '/', no whitespace, no '~^:?*[\' — keep it one safe segment
		}
	}
	return true
}

// Acquire publishes rec under refs/fak/locks/<rec.ID>. It writes the JSON record as a
// blob via `git hash-object -w` (through a temp file, since the Runner seam carries no
// stdin) and points the ref at that blob via `git update-ref`. It is ADDITIVE to any
// local flock — a same-host caller still holds its flock; this just makes the lease
// visible cross-machine after a push. AcquiredAt defaults to now when unset.
//
// It NEVER force-updates an existing ref blindly: update-ref without an <oldvalue> still
// just sets the ref (a ref is not history, so this is not a force-push), but it does not
// touch any branch/HEAD. Returns the written ref on success.
func (s *Store) Acquire(ctx context.Context, rec Record) (string, error) {
	if !validID(rec.ID) {
		return "", fmt.Errorf("leaseref: invalid lease id %q (must be one safe ref segment)", rec.ID)
	}
	if rec.AcquiredAt == 0 {
		rec.AcquiredAt = time.Now().Unix()
	}
	return s.putBlobRef(ctx, rec.Ref(), rec)
}

// putBlobRef marshals v to a git blob and points ref at it with a plain update-ref
// (never a branch/HEAD, never a force). Returns the written ref on success. Shared
// by Acquire (lock leases) and PublishSession (session descriptors).
func (s *Store) putBlobRef(ctx context.Context, ref string, v any) (string, error) {
	blob, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("leaseref: marshal record: %w", err)
	}
	sha, err := s.writeBlob(ctx, blob)
	if err != nil {
		return "", err
	}
	if _, code, err := s.run(ctx, s.dir, "update-ref", ref, sha); err != nil {
		return "", fmt.Errorf("leaseref: git not executable: %w", err)
	} else if code != 0 {
		return "", fmt.Errorf("leaseref: update-ref %s exited %d", ref, code)
	}
	return ref, nil
}

// Release deletes refs/fak/locks/<id>. Idempotent at the package level: a missing ref is
// not an error (the lease is already gone — the desired post-state holds). It uses
// `git update-ref -d`, which targets ONLY the named ref and never a branch/HEAD.
func (s *Store) Release(ctx context.Context, id string) error {
	if !validID(id) {
		return fmt.Errorf("leaseref: invalid lease id %q", id)
	}
	return s.deleteRef(ctx, refPrefix+id)
}

// deleteRef removes the named ref with `git update-ref -d`, treating an
// already-absent ref as success (the idempotent release/remove post-state). It
// targets ONLY the named side ref, never a branch/HEAD. Shared by Release and
// RemoveSession.
func (s *Store) deleteRef(ctx context.Context, ref string) error {
	_, code, err := s.run(ctx, s.dir, "update-ref", "-d", ref)
	if err != nil {
		return fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		// A delete of a ref that does not exist is the already-released state, not a
		// failure. update-ref -d on a missing ref exits non-zero; treat it as success
		// only after confirming the ref is indeed absent.
		if exists, derr := s.has(ctx, ref); derr == nil && !exists {
			return nil
		}
		return fmt.Errorf("leaseref: update-ref -d %s exited %d", ref, code)
	}
	return nil
}

// Get reads back the single lease record at refs/fak/locks/<id>, or (zero, false, nil)
// when no such ref exists — absence is a valid, non-erroneous answer (no lease held).
func (s *Store) Get(ctx context.Context, id string) (Record, bool, error) {
	if !validID(id) {
		return Record{}, false, fmt.Errorf("leaseref: invalid lease id %q", id)
	}
	ref := refPrefix + id
	exists, err := s.has(ctx, ref)
	if err != nil {
		return Record{}, false, err
	}
	if !exists {
		return Record{}, false, nil
	}
	rec, err := s.readRef(ctx, ref)
	if err != nil {
		return Record{}, false, err
	}
	return rec, true, nil
}

// List reads every lease record under refs/fak/locks/*, sorted by id for a stable view.
// This is the source a cross-machine arbiter folds into its live_leases: after an
// ordinary fetch, a peer's pushed lease appears here. A record whose blob does not parse
// is SKIPPED (a forward-compatible or corrupt entry must not blind the whole view), not
// surfaced as an error.
func (s *Store) List(ctx context.Context) ([]Record, error) {
	out, code, err := s.run(ctx, s.dir, "for-each-ref", "--format=%(refname)", refPrefix)
	if err != nil {
		return nil, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		// No refs under the namespace (or the namespace is absent) is an empty list,
		// not an error — the same "absence is valid" rule as Get.
		return nil, nil
	}
	var recs []Record
	for _, line := range strings.Split(out, "\n") {
		ref := strings.TrimSpace(line)
		if !strings.HasPrefix(ref, refPrefix) {
			continue
		}
		if isSessionRef(ref) {
			continue // session descriptors (refs/fak/locks/session-*) are a DISTINCT kind, not lock leases
		}
		rec, rerr := s.readRef(ctx, ref)
		if rerr != nil {
			continue // skip an unreadable/forward-incompatible record, don't fail the view
		}
		recs = append(recs, rec)
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].ID < recs[j].ID })
	return recs, nil
}

// Live reads List and returns only the records that are NOT expired at time now. This is
// the admission-relevant projection: an expired record is reapable and must not block a
// peer, so the arbiter folds the LIVE set. The expired ids are returned alongside so a
// caller can reap them (each via Release) — reaping is itself a converging ref delete.
func (s *Store) Live(ctx context.Context, now time.Time) (live []Record, expired []string, err error) {
	all, err := s.List(ctx)
	if err != nil {
		return nil, nil, err
	}
	for _, r := range all {
		if r.Expired(now) {
			expired = append(expired, r.ID)
			continue
		}
		live = append(live, r)
	}
	return live, expired, nil
}

// ArbiterLease is the projection of a live lease record into the shape a
// dos_arbitrate-style admission kernel consumes as one of its live_leases:
// {lane, lane_kind, tree}. It is the READ-SIDE seam named gap #3 (cross-machine
// atomicity) asked for — the source that lets an arbiter on machine B fold a
// peer's lease (an ordinary `git fetch` put the ref into the local
// refs/fak/locks/* store) into its admission decision, instead of being blind to
// a lease machine A is holding right now.
//
// The mapping is deterministic and lossless for the arbiter's purpose: Lane is
// the lease id (the ref basename), Tree is the record's leased globs (the field
// the disjointness rung actually reasons over), and LaneKind is always "cluster"
// — a refs/fak/locks record is a TREE-SCOPED lease, which is exactly the
// arbiter's cluster kind (a keyword / global lane is not a tree the ref store
// carries). This stays VISIBILITY, not atomic acquisition: it lets the arbiter
// SEE a cross-machine conflict, it does not arbitrate a same-fetch-window race.
type ArbiterLease struct {
	Lane     string   `json:"lane"`
	LaneKind string   `json:"lane_kind"`
	Tree     []string `json:"tree"`
}

// arbiterLaneKind is the kind every refs/fak/locks lease projects to: a
// tree-scoped (cluster) lane. Named so the constant is greppable and the one
// honest mapping decision lives in one place.
const arbiterLaneKind = "cluster"

// LiveLeases reads the non-expired records under refs/fak/locks/* and projects
// each into the arbiter's live_leases shape. This is the read side of the
// cross-machine substrate: after an ordinary `git fetch`, a peer's pushed lease
// is in the local ref store, Live reads it, and this projection hands it to an
// admission kernel as a live lease it must respect. EXPIRED records are dropped
// (they are reapable, not blocking — see Live), so the arbiter never refuses on
// a crashed holder's lapsed lease. The slice is non-nil-and-empty when nothing
// is live, so a JSON encoder emits `[]`, the empty live_leases an arbiter reads
// as "nothing held".
func (s *Store) LiveLeases(ctx context.Context, now time.Time) ([]ArbiterLease, error) {
	live, _, err := s.Live(ctx, now)
	if err != nil {
		return nil, err
	}
	out := make([]ArbiterLease, 0, len(live))
	for _, r := range live {
		tree := r.TreeGlobs
		if tree == nil {
			tree = []string{} // a record with no globs still projects a concrete empty tree
		}
		out = append(out, ArbiterLease{Lane: r.ID, LaneKind: arbiterLaneKind, Tree: tree})
	}
	return out, nil
}

// has reports whether ref exists, via `git rev-parse --verify --quiet <ref>` (exit 0 =
// present, non-zero = absent). A non-executable git is the only hard error.
func (s *Store) has(ctx context.Context, ref string) (bool, error) {
	_, code, err := s.run(ctx, s.dir, "rev-parse", "--verify", "--quiet", ref)
	if err != nil {
		return false, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	return code == 0, nil
}

// readRef reads the blob a lease ref points at (`git cat-file blob <ref>`) and unmarshals
// the Record. The ID is filled from the ref name so a record always knows its own id even
// if the blob omitted it.
func (s *Store) readRef(ctx context.Context, ref string) (Record, error) {
	out, code, err := s.run(ctx, s.dir, "cat-file", "blob", ref)
	if err != nil {
		return Record{}, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return Record{}, fmt.Errorf("leaseref: cat-file blob %s exited %d", ref, code)
	}
	var rec Record
	if err := json.Unmarshal([]byte(out), &rec); err != nil {
		return Record{}, fmt.Errorf("leaseref: unmarshal record at %s: %w", ref, err)
	}
	if rec.ID == "" {
		rec.ID = strings.TrimPrefix(ref, refPrefix)
	}
	return rec, nil
}

// writeBlob writes blob into the object store via `git hash-object -w <file>` (a temp
// file, since the Runner seam carries no stdin) and returns the resulting object id.
func (s *Store) writeBlob(ctx context.Context, blob []byte) (string, error) {
	f, err := os.CreateTemp("", "fak-lease-*.json")
	if err != nil {
		return "", fmt.Errorf("leaseref: temp blob: %w", err)
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(blob); err != nil {
		f.Close()
		return "", fmt.Errorf("leaseref: write blob: %w", err)
	}
	if err := f.Close(); err != nil {
		return "", fmt.Errorf("leaseref: close blob: %w", err)
	}
	out, code, err := s.run(ctx, s.dir, "hash-object", "-w", tmp)
	if err != nil {
		return "", fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return "", fmt.Errorf("leaseref: hash-object exited %d", code)
	}
	sha := strings.TrimSpace(out)
	if sha == "" {
		return "", fmt.Errorf("leaseref: hash-object produced no object id")
	}
	return sha, nil
}
