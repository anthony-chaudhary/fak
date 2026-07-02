package leaseref

// intent.go is the INTENT-LEVEL collision check (#2155): a third ref kind under the
// SAME refs/fak/locks/* transport — refs/fak/locks/intent-<key> — that leases a WORK
// TARGET (an issue number, a bug signature) instead of a file tree.
//
// THE GAP IT CLOSES. dos_arbitrate + the lock leases stop two agents editing the same
// FILES. They do not stop two agents fixing the same ISSUE in *different* files — both
// do the whole task and one dispatch is wasted (the fleet peer-fixes ready/bug issues
// within the hour, so this is a live hazard, not a theoretical one). An intent lease is
// claimed at TASK CLAIM time, before any tokens are spent: the second agent claiming
// the same target is refused INTENT_COLLISION *naming the incumbent* (holder, session,
// age), so it can pick different work instead of racing to a duplicate fix.
//
// WHAT AN INTENT LEASE IS (and is not):
//   - It COMPLEMENTS the file-tree lock lease; it never replaces one. Claim the intent
//     when you pick the task; take the tree lease when you start editing.
//   - It is ADVISORY and SHORT-LIVED by design (default TTL one hour — the observed
//     peer-fix window): a crashed claimant's intent lapses and is reapable, so a target
//     is never deadlocked. The refusal is a warning with evidence, not an enforcement
//     boundary — an operator who *wants* two agents on one target can force past it by
//     releasing or simply ignoring the verdict at the dispatch layer.
//   - THE SAME HONEST BOUNDARY as the lock leases (see the package doc): distribution /
//     visibility via ordinary git fetch/push; SAME-HOST atomicity via the update-ref
//     compare-and-swap; cross-machine same-fetch-window races stay out of scope.
//
// THE NAMESPACE SPLIT (load-bearing, same rule as session.go): an intent ref is
// refs/fak/locks/intent-<key>; the lock-lease readers (List/Live/LiveLeases) and the
// session readers each filter the other kinds out, so the three views stay distinct
// over the one shared for-each-ref scan.
//
// THE KEY (deterministic target normalization): two agents rarely type a target
// byte-identically, so the ref basename is a NORMALIZED key, not the raw string. A
// target that is an issue reference ("#2155", "issue 2155", "gh-2155", "2155")
// canonicalizes to issue-<n>; any other target (a free-form bug signature) collapses
// whitespace, lowercases, and hashes to sig-<16 hex of sha256>. The raw target rides in
// the record so a collision refusal can name the incumbent's own words.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// intentPrefix is the basename prefix that marks an intent lease apart from a lock
// lease and a session descriptor under the shared refs/fak/locks/ namespace. The full
// ref is refs/fak/locks/intent-<key>.
const intentPrefix = "intent-"

// ReasonIntentCollision is the structured refusal an agent gets when it claims a work
// target a LIVE peer already claimed: the same issue/bug is being fixed right now by
// someone else, so spending a turn on it duplicates work. The verdict names the
// incumbent (holder, session, raw target, age) so the refused agent can verify or pick
// different work. Deny-as-value, same discipline as the fence vocabulary above.
const ReasonIntentCollision = "INTENT_COLLISION"

// DefaultIntentTTLSeconds is the default lifetime of an intent lease: one hour — the
// observed fleet peer-fix window (#2155). Short-lived by design: an intent is a claim
// on a task-sized unit of work, and a crashed claimant must not deadlock the target.
const DefaultIntentTTLSeconds int64 = 3600

// IntentRecord is one claimed work target persisted under refs/fak/locks/intent-<key>.
// Same serialization discipline as Record/SessionDescriptor: one JSON blob the ref
// points at, diffable and git-integrity-checked.
type IntentRecord struct {
	Target     string `json:"target"`        // the raw claimed target, as the claimant stated it
	Key        string `json:"key,omitempty"` // the normalized ref-basename key (derived; stored for display)
	Holder     string `json:"holder"`        // who claimed it (machine/session identity, free-form)
	SessionID  string `json:"session_id,omitempty"`
	AcquiredAt int64  `json:"acquired_unix"`
	RenewedAt  int64  `json:"renewed_unix,omitempty"`
	TTLSeconds int64  `json:"ttl_seconds"`
}

// effectiveActiveAt mirrors Record: the later of AcquiredAt and RenewedAt, the instant
// the liveness window is measured from.
func (r IntentRecord) effectiveActiveAt() int64 {
	if r.RenewedAt > r.AcquiredAt {
		return r.RenewedAt
	}
	return r.AcquiredAt
}

// Expired reports whether the intent is past its TTL at time now. A zero TTL never
// expires (claim with the default instead — an immortal intent defeats the design).
func (r IntentRecord) Expired(now time.Time) bool {
	if r.TTLSeconds <= 0 {
		return false
	}
	return now.Unix() >= r.effectiveActiveAt()+r.TTLSeconds
}

// Ref returns the full ref path this intent is stored at: refs/fak/locks/intent-<key>.
func (r IntentRecord) Ref() string { return refPrefix + intentPrefix + IntentKey(r.Target) }

// isIntentRef reports whether a full ref under refs/fak/locks/ is an INTENT lease
// (basename starts with intent-). The one place this kind's namespace split is decided,
// mirrored by isSessionRef for sessions.
func isIntentRef(ref string) bool {
	return strings.HasPrefix(ref, refPrefix+intentPrefix)
}

// issueTargetRE recognizes a target that IS an issue reference and nothing else:
// an optional issue/gh/bug/# marker followed by digits. Deliberately anchored and
// conservative — "fix the 3 flaky tests" is a signature (hashed), not issue #3.
var issueTargetRE = regexp.MustCompile(`^(?:(?:issue|gh|bug)[\s#-]*|#\s*)?([0-9]+)$`)

// IntentKey canonicalizes a work target into the deterministic ref-basename key two
// independently-phrased claims of the SAME target both land on. An issue-shaped target
// ("#2155" / "issue 2155" / "gh-2155" / "2155") maps to issue-<n> (leading zeros
// stripped, so "#0012" and "#12" collide as they should); anything else lowercases,
// collapses runs of whitespace, and hashes to sig-<first 16 hex of sha256> — a fixed-
// width segment that is always ref-safe regardless of what bytes the signature held.
func IntentKey(target string) string {
	norm := strings.ToLower(strings.TrimSpace(target))
	if m := issueTargetRE.FindStringSubmatch(norm); m != nil {
		n := strings.TrimLeft(m[1], "0")
		if n == "" {
			n = "0"
		}
		return "issue-" + n
	}
	norm = strings.Join(strings.Fields(norm), " ")
	sum := sha256.Sum256([]byte(norm))
	return "sig-" + hex.EncodeToString(sum[:])[:16]
}

// IntentVerdict is the deny-as-value result of a ClaimIntent. OK is the only admit
// state; Reason names the closed refusal class (INTENT_COLLISION, or LEASE_CONTENDED
// for a lost same-host CAS). Peer is the live incumbent on a collision — the evidence
// the refused agent routes on (verify the peer really is on it, then pick other work).
type IntentVerdict struct {
	OK     bool          `json:"ok"`
	Reason string        `json:"reason,omitempty"`
	Key    string        `json:"key"`
	Peer   *IntentRecord `json:"peer,omitempty"`
	Detail string        `json:"detail,omitempty"`
}

// ClaimIntent claims a work target BEFORE any tokens are spent on it. Cases (now = the
// claim instant; rec carries Target/Holder/SessionID/TTLSeconds):
//
//   - no live claim on the key             -> CLAIM:   the target is yours
//   - current claim expired at now         -> TAKEOVER: reap the lapsed claim, take it
//   - current live, SAME holder            -> RENEW:   your own claim's window advances
//   - current live, DIFFERENT holder       -> REFUSE:  INTENT_COLLISION naming the peer
//
// A live claim with an EMPTY holder cannot be proven to be the caller's, so it refuses
// as a collision — the conservative posture, same as AcquireFenced. The write goes
// through the update-ref compare-and-swap, so a same-host racer loses cleanly: a lost
// CAS re-reads once and reports the winner as INTENT_COLLISION (that IS the collision
// the check exists to catch) or LEASE_CONTENDED when the re-read shows no live rival
// (retryable). TTLSeconds <= 0 takes DefaultIntentTTLSeconds — an intent is short-lived
// by contract.
func (s *Store) ClaimIntent(ctx context.Context, rec IntentRecord, now time.Time) (IntentRecord, IntentVerdict, error) {
	if strings.TrimSpace(rec.Target) == "" {
		return IntentRecord{}, IntentVerdict{}, fmt.Errorf("leaseref: empty intent target")
	}
	key := IntentKey(rec.Target)
	ref := refPrefix + intentPrefix + key
	if rec.TTLSeconds <= 0 {
		rec.TTLSeconds = DefaultIntentTTLSeconds
	}
	rec.Key = key

	oldOID, hasRef, err := s.currentOID(ctx, ref)
	if err != nil {
		return IntentRecord{}, IntentVerdict{}, err
	}
	var cur IntentRecord
	if hasRef {
		if cur, err = s.readIntentRef(ctx, ref); err != nil {
			return IntentRecord{}, IntentVerdict{}, err
		}
	}

	out := rec
	switch {
	case !hasRef, cur.Expired(now):
		// CLAIM or TAKEOVER: fresh window either way — a lapsed claimant's record is
		// reapable evidence, not a live rival.
		out.AcquiredAt = now.Unix()
		out.RenewedAt = 0
	case cur.Holder == rec.Holder && rec.Holder != "":
		// RENEW: the claimant's own live claim; the window moves, the claim identity
		// (target as originally stated, acquisition instant, session) stays.
		out = cur
		out.RenewedAt = now.Unix()
		if rec.TTLSeconds > 0 {
			out.TTLSeconds = rec.TTLSeconds
		}
		if out.SessionID == "" && rec.SessionID != "" {
			out.SessionID = rec.SessionID
		}
	default:
		return IntentRecord{}, collisionVerdict(key, cur, now), nil
	}

	written, err := s.casWriteJSON(ctx, ref, out, oldOID, hasRef)
	if err != nil {
		return IntentRecord{}, IntentVerdict{}, err
	}
	if !written {
		// A same-host peer raced this claim. Re-read once: a live different-holder
		// winner is exactly the collision this check exists to surface; anything else
		// (the racer released, or the record is unreadable) is a retryable contention.
		if rival, ok, rerr := s.getIntentByKey(ctx, key); rerr == nil && ok && !rival.Expired(now) &&
			(rival.Holder != rec.Holder || rec.Holder == "") {
			return IntentRecord{}, collisionVerdict(key, rival, now), nil
		}
		return IntentRecord{}, IntentVerdict{
			Reason: ReasonLeaseContended,
			Key:    key,
			Detail: fmt.Sprintf("intent %s changed under the claim (CAS lost); re-read and retry", key),
		}, nil
	}
	return out, IntentVerdict{OK: true, Key: key}, nil
}

// collisionVerdict builds the INTENT_COLLISION refusal naming the live incumbent — the
// #2155 witness: the second claimant learns WHO is on the target and since when, before
// spending a turn on it.
func collisionVerdict(key string, cur IntentRecord, now time.Time) IntentVerdict {
	peer := cur
	age := now.Unix() - cur.effectiveActiveAt()
	if age < 0 {
		age = 0
	}
	who := cur.Holder
	if who == "" {
		who = "(anonymous)"
	}
	detail := fmt.Sprintf("target %q is already claimed by %s (%ds ago", cur.Target, who, age)
	if cur.SessionID != "" {
		detail += ", session " + cur.SessionID
	}
	detail += ") — verify the peer's claim, then pick different work or wait out the TTL"
	return IntentVerdict{Reason: ReasonIntentCollision, Key: key, Peer: &peer, Detail: detail}
}

// ReleaseIntent deletes the intent lease for target — the done/abandoned side of the
// lifecycle. Idempotent: a missing ref is the already-released post-state, same as
// Release. Releasing on ship keeps the namespace clean ahead of the TTL.
func (s *Store) ReleaseIntent(ctx context.Context, target string) error {
	if strings.TrimSpace(target) == "" {
		return fmt.Errorf("leaseref: empty intent target")
	}
	return s.deleteRef(ctx, refPrefix+intentPrefix+IntentKey(target))
}

// GetIntent reads back the intent lease for target, or (zero, false, nil) when none is
// claimed — absence is a valid, non-erroneous answer.
func (s *Store) GetIntent(ctx context.Context, target string) (IntentRecord, bool, error) {
	if strings.TrimSpace(target) == "" {
		return IntentRecord{}, false, fmt.Errorf("leaseref: empty intent target")
	}
	return s.getIntentByKey(ctx, IntentKey(target))
}

func (s *Store) getIntentByKey(ctx context.Context, key string) (IntentRecord, bool, error) {
	ref := refPrefix + intentPrefix + key
	exists, err := s.has(ctx, ref)
	if err != nil || !exists {
		return IntentRecord{}, false, err
	}
	rec, err := s.readIntentRef(ctx, ref)
	if err != nil {
		return IntentRecord{}, false, err
	}
	return rec, true, nil
}

// ListIntents reads every intent lease under refs/fak/locks/intent-*, sorted by key for
// a stable view. Lock leases and session descriptors are EXCLUDED (the namespace
// split); an unparseable record is skipped, not surfaced — the same reader rules as
// List/ListSessions.
func (s *Store) ListIntents(ctx context.Context) ([]IntentRecord, error) {
	out, code, err := s.run(ctx, s.dir, "for-each-ref", "--format=%(refname)", refPrefix)
	if err != nil {
		return nil, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return nil, nil // absent/empty namespace is an empty list, not an error
	}
	var recs []IntentRecord
	for _, line := range strings.Split(out, "\n") {
		ref := strings.TrimSpace(line)
		if !isIntentRef(ref) {
			continue
		}
		rec, rerr := s.readIntentRef(ctx, ref)
		if rerr != nil {
			continue
		}
		recs = append(recs, rec)
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].Key < recs[j].Key })
	return recs, nil
}

// LiveIntents partitions ListIntents into live-vs-expired at time now, returning the
// expired KEYS alongside so a caller can reap them — the same shape as Live and
// LiveSessions.
func (s *Store) LiveIntents(ctx context.Context, now time.Time) (live []IntentRecord, expired []string, err error) {
	all, err := s.ListIntents(ctx)
	if err != nil {
		return nil, nil, err
	}
	for _, r := range all {
		if r.Expired(now) {
			expired = append(expired, r.Key)
			continue
		}
		live = append(live, r)
	}
	return live, expired, nil
}

// ReapIntents deletes every intent lease expired at time now and returns the keys
// reaped — a lapsed claimant's intent is bounded, never a permanent block on the
// target. Same best-effort, idempotent sweep contract as Reap/ReapSessions.
func (s *Store) ReapIntents(ctx context.Context, now time.Time) (reaped []string, err error) {
	_, expired, lerr := s.LiveIntents(ctx, now)
	if lerr != nil {
		return nil, lerr
	}
	var errs []string
	for _, key := range expired {
		if rerr := s.deleteRef(ctx, refPrefix+intentPrefix+key); rerr != nil {
			errs = append(errs, fmt.Sprintf("reap intent %s: %v", key, rerr))
			continue
		}
		reaped = append(reaped, key)
	}
	if len(errs) > 0 {
		return reaped, fmt.Errorf("leaseref: %s", strings.Join(errs, "; "))
	}
	return reaped, nil
}

// readIntentRef reads the blob an intent ref points at and unmarshals the record. The
// Key is (re)filled from the ref name so a record always knows its key even if the blob
// omitted it.
func (s *Store) readIntentRef(ctx context.Context, ref string) (IntentRecord, error) {
	out, code, err := s.run(ctx, s.dir, "cat-file", "blob", ref)
	if err != nil {
		return IntentRecord{}, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return IntentRecord{}, fmt.Errorf("leaseref: cat-file blob %s exited %d", ref, code)
	}
	var rec IntentRecord
	if err := json.Unmarshal([]byte(out), &rec); err != nil {
		return IntentRecord{}, fmt.Errorf("leaseref: unmarshal intent record at %s: %w", ref, err)
	}
	if rec.Key == "" {
		rec.Key = strings.TrimPrefix(strings.TrimPrefix(ref, refPrefix), intentPrefix)
	}
	return rec, nil
}
