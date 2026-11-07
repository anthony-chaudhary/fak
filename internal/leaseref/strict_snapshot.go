package leaseref

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

// StrictSnapshotResult is the typed snapshot result containing all parsed
// records alongside the partitioned live and expired sets.
type StrictSnapshotResult struct {
	Records []Record
	Live    []Record
	Expired []string
}

// readRefStrict reads the blob at ref and decodes it as a Record.
// In addition to JSON decode and git blob presence checks performed by readRef,
// readRefStrict validates that rec.ID is non-empty, matches the ref's basename,
// and satisfies validID.
func (s *Store) readRefStrict(ctx context.Context, ref string) (Record, error) {
	rec, err := s.readRef(ctx, ref)
	if err != nil {
		return Record{}, err
	}
	return validateStrictRecord(ref, rec)
}

func validateStrictRecord(ref string, rec Record) (Record, error) {
	expectedID := strings.TrimPrefix(ref, refPrefix)
	if !validID(rec.ID) {
		return Record{}, fmt.Errorf("leaseref: invalid lease id %q in ref %s", rec.ID, ref)
	}
	if rec.ID != expectedID {
		return Record{}, fmt.Errorf("leaseref: lease id mismatch: blob has %q, ref is %s", rec.ID, ref)
	}
	return rec, nil
}

// StrictSnapshot reads every lease record under refs/fak/locks/* strictly.
// Unlike List, which silently drops unparseable or corrupt blobs to keep the
// view alive, StrictSnapshot returns a CorruptRefError containing the offending
// ref name if any lease ref points to a missing, unreadable, or malformed blob.
// Session, intent, and contract refs are ignored (preserving namespace isolation).
// A successfully enumerated empty or absent namespace returns an empty slice and nil error.
// Failure to enumerate the namespace returns an error, never an empty successful view.
func (s *Store) StrictSnapshot(ctx context.Context) ([]Record, error) {
	if s == nil {
		return nil, fmt.Errorf("leaseref: nil store")
	}
	var recs []Record
	var err error
	if s.runStdin != nil {
		recs, err = s.strictSnapshotBatch(ctx)
	} else {
		recs, err = listRefsStrict(ctx, s, isLeaseRef, s.readRefStrict)
	}
	if err != nil {
		return nil, err
	}
	if recs == nil {
		recs = []Record{}
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].ID < recs[j].ID })
	return recs, nil
}

func (s *Store) strictSnapshotBatch(ctx context.Context) ([]Record, error) {
	out, code, err := s.run(ctx, s.dir, "for-each-ref", "--format=%(refname) %(objectname)", refPrefix)
	if err != nil {
		return nil, fmt.Errorf("leaseref: enumerate strict snapshot: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("leaseref: git for-each-ref %s exited %d", refPrefix, code)
	}
	type pinnedRef struct{ ref, oid string }
	var refs []pinnedRef
	var input strings.Builder
	seen := make(map[string]bool)
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 2 || !strings.HasPrefix(fields[0], refPrefix) || !validObjectID(fields[1]) || seen[fields[0]] {
			return nil, &CorruptRefError{Ref: fields[0], Err: fmt.Errorf("invalid strict ref enumeration")}
		}
		seen[fields[0]] = true
		if !isLeaseRef(fields[0]) {
			continue
		}
		refs = append(refs, pinnedRef{fields[0], fields[1]})
		input.WriteString(fields[1])
		input.WriteByte('\n')
	}
	if len(refs) == 0 {
		return nil, nil
	}
	// Read the observed objects, never mutable ref names or the tolerant parser.
	stream, code, err := s.runStdin(ctx, s.dir, input.String(), "cat-file", "--batch")
	if err != nil {
		return nil, fmt.Errorf("leaseref: read strict snapshot: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("leaseref: git cat-file --batch exited %d", code)
	}
	var recs []Record
	for _, ref := range refs {
		header, rest, ok := strings.Cut(stream, "\n")
		fields := strings.Fields(header)
		if !ok || len(fields) != 3 || fields[0] != ref.oid || fields[1] != "blob" {
			return nil, &CorruptRefError{Ref: ref.ref, Err: fmt.Errorf("missing or invalid strict object header")}
		}
		size, err := strconv.ParseUint(fields[2], 10, 64)
		if err != nil || size >= uint64(len(rest)) || rest[int(size)] != '\n' {
			return nil, &CorruptRefError{Ref: ref.ref, Err: fmt.Errorf("invalid or truncated strict object payload")}
		}
		var rec Record
		if err := json.Unmarshal([]byte(rest[:int(size)]), &rec); err != nil {
			return nil, &CorruptRefError{Ref: ref.ref, Err: err}
		}
		if rec.ID == "" {
			rec.ID = leaseIDFromRef(ref.ref)
		}
		rec, err = validateStrictRecord(ref.ref, rec)
		if err != nil {
			return nil, &CorruptRefError{Ref: ref.ref, Err: err}
		}
		recs = append(recs, rec)
		stream = rest[int(size)+1:]
	}
	if stream != "" {
		return nil, fmt.Errorf("leaseref: trailing strict object data")
	}
	return recs, nil
}

// StrictList reads every lease record under refs/fak/locks/* strictly, sorted by id.
// It is an alias for StrictSnapshot, providing API parity with List.
func (s *Store) StrictList(ctx context.Context) ([]Record, error) {
	return s.StrictSnapshot(ctx)
}

// StrictSnapshot is the package-level entry point for Store.StrictSnapshot.
func StrictSnapshot(ctx context.Context, s *Store) ([]Record, error) {
	if s == nil {
		return nil, fmt.Errorf("leaseref: nil store")
	}
	return s.StrictSnapshot(ctx)
}

// StrictList is the package-level entry point for Store.StrictList.
func StrictList(ctx context.Context, s *Store) ([]Record, error) {
	if s == nil {
		return nil, fmt.Errorf("leaseref: nil store")
	}
	return s.StrictList(ctx)
}

// StrictLiveSnapshot reads StrictSnapshot and partitions the records into
// live records and expired lease IDs at time now. If any lease ref is corrupt,
// it returns an error and does not return an incomplete or deceptive live view.
func (s *Store) StrictLiveSnapshot(ctx context.Context, now time.Time) (live []Record, expired []string, err error) {
	all, err := s.StrictSnapshot(ctx)
	if err != nil {
		return nil, nil, err
	}
	live, expired = liveExpire(all, now, func(r Record) string { return r.ID })
	return live, expired, nil
}

// StrictLive reads StrictSnapshot and partitions the records into
// live records and expired lease IDs at time now.
func (s *Store) StrictLive(ctx context.Context, now time.Time) (live []Record, expired []string, err error) {
	return s.StrictLiveSnapshot(ctx, now)
}

// StrictLive is the package-level entry point for Store.StrictLive.
func StrictLive(ctx context.Context, s *Store, now time.Time) (live []Record, expired []string, err error) {
	if s == nil {
		return nil, nil, fmt.Errorf("leaseref: nil store")
	}
	return s.StrictLive(ctx, now)
}

// StrictSnapshotWithResult evaluates a strict snapshot and partitions it into a StrictSnapshotResult.
func (s *Store) StrictSnapshotWithResult(ctx context.Context, now time.Time) (StrictSnapshotResult, error) {
	all, err := s.StrictSnapshot(ctx)
	if err != nil {
		return StrictSnapshotResult{}, err
	}
	live, expired := liveExpire(all, now, func(r Record) string { return r.ID })
	return StrictSnapshotResult{
		Records: all,
		Live:    live,
		Expired: expired,
	}, nil
}

// IsLeaseRef reports whether ref is a lock lease ref under refs/fak/locks/
// (and not a session descriptor, intent lease, or contract record).
func IsLeaseRef(ref string) bool {
	return isLeaseRef(ref)
}

// IsSessionRef reports whether ref is a session descriptor ref under refs/fak/locks/.
func IsSessionRef(ref string) bool {
	return isSessionRef(ref)
}

// IsIntentRef reports whether ref is an intent lease ref under refs/fak/locks/.
func IsIntentRef(ref string) bool {
	return isIntentRef(ref)
}
