package leaseref

// refhelpers.go holds the small generic helpers shared by the parallel ref kinds
// (lock lease Record, intent lease IntentRecord, session descriptor SessionDescriptor,
// contract lease ContractRecord) that ride the SAME refs/fak/locks/* scan-and-partition shape:
// List/ListIntents/ListSessions/ListContracts all walk `for-each-ref`, keep one kind, and read each blob;
// Live/LiveIntents/LiveSessions/LiveContracts all then partition a List* result into live-vs-expired.
// Factoring the shape here keeps the readers behaviorally identical BY
// CONSTRUCTION instead of by hand-copied code that can drift.

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// listRefs scans every ref under the shared refs/fak/locks/ namespace and reads the
// ones keep accepts into a T via read, in for-each-ref's own order (the caller sorts
// to its own key). A non-executable git is the only hard error; an absent/empty
// namespace (code != 0) and an unreadable/forward-incompatible entry (read returns an
// error) are both silently skipped — the "absence and corruption never blind the whole
// view" rule List/ListIntents/ListSessions all share.
func listRefs[T any](ctx context.Context, s *Store, keep func(ref string) bool, read func(context.Context, string) (T, error)) ([]T, error) {
	out, code, err := s.run(ctx, s.dir, "for-each-ref", "--format=%(refname)", refPrefix)
	if err != nil {
		return nil, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return nil, nil // absent/empty namespace is an empty list, not an error
	}
	var recs []T
	for _, line := range strings.Split(out, "\n") {
		ref := strings.TrimSpace(line)
		if !keep(ref) {
			continue
		}
		rec, rerr := read(ctx, ref)
		if rerr != nil {
			continue // skip an unreadable/forward-incompatible record, don't fail the view
		}
		recs = append(recs, rec)
	}
	return recs, nil
}

// expirable is any TTL-bounded record with its own Expired(now) — Record, IntentRecord,
// and SessionDescriptor each implement it independently (their TTL field is even named
// differently), which is exactly what lets liveExpire stay generic over all three.
type expirable interface {
	Expired(now time.Time) bool
}

// liveExpire partitions all into live-vs-expired at time now, returning the ids of the
// expired ones (via idOf, since each kind's "id" is a different field) for a caller to
// reap — the shared body of Live, LiveIntents, and LiveSessions.
func liveExpire[T expirable](all []T, now time.Time, idOf func(T) string) (live []T, expired []string) {
	for _, r := range all {
		if r.Expired(now) {
			expired = append(expired, idOf(r))
			continue
		}
		live = append(live, r)
	}
	return live, expired
}

// expired is the shared TTL check behind Record.Expired and IntentRecord.Expired: a
// zero TTL never expires, otherwise the record lapses once now reaches
// effectiveActiveAt+ttlSeconds.
func expired(now time.Time, ttlSeconds, effectiveActiveAt int64) bool {
	if ttlSeconds <= 0 {
		return false
	}
	return now.Unix() >= effectiveActiveAt+ttlSeconds
}
