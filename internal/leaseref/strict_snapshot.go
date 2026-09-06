package leaseref

import (
	"context"
	"fmt"
	"sort"
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
// An empty or absent namespace returns an empty slice and nil error.
func (s *Store) StrictSnapshot(ctx context.Context) ([]Record, error) {
	if s == nil {
		return nil, fmt.Errorf("leaseref: nil store")
	}
	recs, err := listRefsStrict(ctx, s, isLeaseRef, s.readRefStrict)
	if err != nil {
		return nil, err
	}
	if recs == nil {
		recs = []Record{}
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].ID < recs[j].ID })
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
