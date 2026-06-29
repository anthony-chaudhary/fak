package leaseref

// fence.go is the FENCING-TOKEN half of the cross-machine lease substrate — the named
// highest-value gap of the agent-OS scheduling readout (#906 §3.3, child C1) and the
// mechanism the long-dormancy rehydration rung (#1182) composes.
//
// THE HAZARD IT CLOSES (the paused-then-resumed holder). leaseref.Record already bounds a
// crashed holder with a TTL so a peer can reap it — but reaping alone is not safe. The
// classic distributed-locking bug (Kleppmann; etcd/Chubby call it the same thing): holder A
// acquires a lease, then PAUSES — a GC pause, a dormancy gap past TTL, a slow host. A's
// lease expires, peer B reaps it and acquires, and B starts writing the leased tree. Then A
// resumes and, still "alive" by its own heartbeat, performs a write it believes its lease
// authorizes. Two holders now write the same tree. TTL + heartbeat cannot catch this: A IS
// alive; what is stale is its LEASE, not its process.
//
// THE FIX (a monotonic fencing token). Every TRANSITION (a new holder taking over an
// expired lease) bumps Record.Generation strictly. A write is admitted at the call boundary
// only if the holder's PRESENTED generation still equals the LIVE lease's generation
// (Fence). When B reaps and reacquires, the generation advances; A's later write presents
// the old generation, Fence sees the live lease is newer, and returns STALE_LEASE. A is
// refused and must halt-and-reacquire — never a silent resume. This is the etcd/Chubby
// monotonic-epoch rule applied at fak's enforced call boundary.
//
// THE HONEST BOUNDARY (kept in lockstep with the package doc). The cross-machine substrate
// remains DISTRIBUTION / VISIBILITY, not atomic acquisition: two clones can still both write
// a generation bump in the same fetch window, and git's merge converges the SET of refs
// without arbitrating a winner. What the fence DOES buy, beyond the blind Acquire, is real
// SAME-HOST atomicity: the generation bump and the renew go through an update-ref OLD-VALUE
// compare-and-swap, so a peer process on the same clone that raced the bump loses the CAS
// (LEASE_CONTENDED) instead of silently clobbering the token. The cross-host final-race
// arbitration stays out of scope, exactly as #825/#826 draw it.
//
// DENY-AS-VALUE. Every policy outcome — stale, held, contended, no-lease — is a FenceVerdict
// value, never a returned error (the same discipline as safecommit.Result and gitgate's
// Verdict). The returned error is reserved for INFRASTRUCTURE failure only: git not
// executable, or an unreadable record blob.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// The closed fence-verdict reason vocabulary. Local string constants in the same shape the
// rest of the lease/commit stack uses (safecommit's Reason* family); they are the --json
// contract a calling loop routes on.
const (
	// ReasonStaleLease is the headline refusal: a write or renew presented a lease GENERATION
	// behind the live lease — a newer holder was admitted while this one was paused/dormant,
	// so its lease instance is stale and its write must be refused (the etcd/Chubby
	// paused-then-resumed-holder hazard, #906 §3.3 / #1182). The corrective is to halt and
	// reacquire, never to resume.
	ReasonStaleLease = "STALE_LEASE"
	// ReasonLeaseHeld is the admission refusal: a DIFFERENT holder owns a LIVE (un-expired)
	// lease on this id, so this caller may not acquire it. Distinct from STALE_LEASE — that
	// is about generation order; this is that the incumbent is still alive and un-reapable.
	ReasonLeaseHeld = "LEASE_HELD"
	// ReasonLeaseContended is the same-host CAS loss: the ref advanced between this caller's
	// read and its compare-and-swap write (a peer raced the generation bump). Retryable —
	// re-read and decide again. This is the same-host atomicity the visibility-only boundary
	// does NOT promise cross-machine but CAN enforce on one host via update-ref's old-value.
	ReasonLeaseContended = "LEASE_CONTENDED"
	// ReasonNoLease is the fence verdict when NO live lease exists for the id — the caller
	// believes it holds a lease that has been released or reaped. Not a stale-write attempt,
	// but not a held lease either: the caller must reacquire before writing.
	ReasonNoLease = "NO_LEASE"
)

// FenceVerdict is the deny-as-value result of a fence check or a fenced acquire/renew. OK is
// the only admit state; a non-empty Reason names the closed refusal class. Presented is the
// generation the caller carried; Current is the live lease's generation (0 = none live);
// Holder is the live holder, surfaced so the refusal message names who actually owns it.
type FenceVerdict struct {
	OK        bool   `json:"ok"`
	Reason    string `json:"reason,omitempty"`
	Presented int64  `json:"presented_generation"`
	Current   int64  `json:"current_generation"`
	Holder    string `json:"current_holder,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

// Fence is the READ-SIDE fencing check enforced at the call boundary BEFORE a holder's
// write: given the lease the caller believes it holds (presented — its ID + Generation +
// Holder), it re-reads the LIVE lease at refs/fak/locks/<id> and decides whether the
// caller's generation still authorizes a write.
//
// Verdicts:
//   - no live lease (ref absent or expired at now)              -> NO_LEASE    (reacquire first)
//   - current.Generation  > presented.Generation                -> STALE_LEASE (a newer holder exists)
//   - current.Generation  < presented.Generation                -> STALE_LEASE (a token never issued; fail closed)
//   - current.Generation == presented.Generation, same holder   -> OK
//   - current.Generation == presented.Generation, other holder  -> STALE_LEASE (fail closed)
//
// Generation 0 on BOTH sides with a matching (or empty) holder is the legacy/unfenced lease:
// it carries no fence opinion and admits, so a pre-fence flow that never set a generation is
// not broken. The moment either side carries a real generation, the order rule applies.
func (s *Store) Fence(ctx context.Context, presented Record, now time.Time) (FenceVerdict, error) {
	if !validID(presented.ID) {
		return FenceVerdict{}, fmt.Errorf("leaseref: invalid lease id %q", presented.ID)
	}
	cur, ok, err := s.Get(ctx, presented.ID)
	if err != nil {
		return FenceVerdict{}, err
	}
	v := FenceVerdict{Presented: presented.Generation}
	if !ok || cur.Expired(now) {
		v.Reason = ReasonNoLease
		v.Detail = "no live lease for id " + presented.ID + "; reacquire before writing"
		return v, nil
	}
	v.Current = cur.Generation
	v.Holder = cur.Holder

	if cur.Generation == presented.Generation {
		// Same generation: an admit, UNLESS the holder identity disagrees (a different holder
		// at the same generation is an anomaly — fail closed rather than admit a possible
		// impostor). Empty holders on both sides are the anonymous legacy case and admit.
		if presented.Holder != "" && cur.Holder != "" && cur.Holder != presented.Holder {
			v.Reason = ReasonStaleLease
			v.Detail = fmt.Sprintf("generation %d is held by %q, not %q — halt and reacquire", cur.Generation, cur.Holder, presented.Holder)
			return v, nil
		}
		v.OK = true
		return v, nil
	}

	// Generations differ: stale either way (a newer holder advanced it, or the caller
	// presented a token that was never issued). Both fail closed.
	v.Reason = ReasonStaleLease
	if cur.Generation > presented.Generation {
		v.Detail = fmt.Sprintf("lease advanced to generation %d (held by %q); your generation %d is stale — halt and reacquire", cur.Generation, cur.Holder, presented.Generation)
	} else {
		v.Detail = fmt.Sprintf("presented generation %d exceeds the live generation %d — never issued; refused", presented.Generation, cur.Generation)
	}
	return v, nil
}

// AcquireFenced is the generation-aware, compare-and-swap WRITE side of the fencing token.
// Unlike Acquire — a blind update-ref that overwrites whatever is there, kept for the
// best-effort cross-machine PUBLISH path — AcquireFenced reads the current lease, decides
// per the four cases below, computes the next generation MONOTONICALLY, and writes under an
// update-ref OLD-VALUE compare-and-swap so a same-host racer loses (LEASE_CONTENDED) rather
// than clobbering the token.
//
// Cases (now = the acquisition instant; rec carries the requested ID/TreeGlobs/Holder/TTL):
//   - no current ref                  -> FRESH:      generation 1
//   - current expired at now          -> TRANSITION: generation = current+1 (reap + take over)
//   - current live, SAME holder       -> RENEW:      generation unchanged, RenewedAt = now
//   - current live, DIFFERENT holder  -> REFUSE:     LEASE_HELD (a live peer owns it)
//
// A live lease with an EMPTY holder cannot be proven to be the caller's, so re-acquiring one
// falls to the DIFFERENT-holder refuse: an anonymous live lease must expire or be released
// before a fenced acquire, the conservative posture against a stale anonymous writer.
//
// On admit it returns the WRITTEN record — so the caller learns its assigned Generation, the
// fencing token it must present on every later write — plus an OK verdict. On a refusal it
// returns the zero record and the deny verdict, and the ref is untouched.
func (s *Store) AcquireFenced(ctx context.Context, rec Record, now time.Time) (Record, FenceVerdict, error) {
	if !validID(rec.ID) {
		return Record{}, FenceVerdict{}, fmt.Errorf("leaseref: invalid lease id %q (must be one safe ref segment)", rec.ID)
	}
	ref := rec.Ref()
	oldOID, hasRef, err := s.currentOID(ctx, ref)
	if err != nil {
		return Record{}, FenceVerdict{}, err
	}
	var cur Record
	if hasRef {
		if cur, err = s.readRef(ctx, ref); err != nil {
			return Record{}, FenceVerdict{}, err
		}
	}

	out := rec
	switch {
	case !hasRef:
		out.Generation = 1
		out.AcquiredAt = now.Unix()
		out.RenewedAt = 0
	case cur.Expired(now):
		// TRANSITION: reap a dead holder's lease and take over. The strict bump is the
		// fence — A's later write, presenting the pre-bump generation, will read STALE.
		out.Generation = cur.Generation + 1
		out.AcquiredAt = now.Unix()
		out.RenewedAt = 0
	case cur.Holder == rec.Holder && rec.Holder != "":
		// RENEW: the same holder extends its own live lease. Generation is unchanged (a
		// renew is liveness, not a new admission); the tree stays fixed (no silent
		// expansion via "renew") and only the time window — and an explicit TTL bump — move.
		out = cur
		out.RenewedAt = now.Unix()
		if rec.TTLSeconds > 0 {
			out.TTLSeconds = rec.TTLSeconds
		}
	default:
		return Record{}, FenceVerdict{
			Reason:  ReasonLeaseHeld,
			Current: cur.Generation,
			Holder:  cur.Holder,
			Detail:  fmt.Sprintf("lease %s is held live by %q (generation %d); not acquired", rec.ID, cur.Holder, cur.Generation),
		}, nil
	}

	return s.commitFenced(ctx, ref, out, oldOID, hasRef, "acquire", rec.ID)
}

// commitFenced performs the CAS write that ends both AcquireFenced and Renew, then maps the
// outcome to a verdict: a lost CAS becomes LEASE_CONTENDED tagged with op (the verb that
// raced) and id, a clean write becomes an OK verdict carrying out's generation. hasRef is
// whether ref already existed (the casWrite precondition), op is "acquire"/"renew" for the
// contended message, and id names the lease in that message.
func (s *Store) commitFenced(ctx context.Context, ref string, out Record, oldOID string, hasRef bool, op, id string) (Record, FenceVerdict, error) {
	written, err := s.casWrite(ctx, ref, out, oldOID, hasRef)
	if err != nil {
		return Record{}, FenceVerdict{}, err
	}
	if !written {
		return Record{}, FenceVerdict{
			Reason: ReasonLeaseContended,
			Detail: fmt.Sprintf("lease %s changed under the %s (CAS lost); re-read and retry", id, op),
		}, nil
	}
	return out, FenceVerdict{OK: true, Presented: out.Generation, Current: out.Generation, Holder: out.Holder}, nil
}

// Renew extends a lease the caller already holds (a liveness heartbeat): it bumps RenewedAt
// to now and refreshes the TTL window WITHOUT bumping the generation — a renew is the SAME
// holder staying alive, not a new admission (the K8s renewTime vs leaseTransitions split).
// It refuses if the live lease is no longer this holder's: STALE_LEASE when a peer has taken
// over (a different holder), NO_LEASE when the lease lapsed or is gone (a lapsed lease must
// be reacquired, never revived by a renew, since a peer may already own it). On OK it returns
// the renewed record. ttlSeconds <= 0 keeps the lease's existing TTL.
func (s *Store) Renew(ctx context.Context, id, holder string, ttlSeconds int64, now time.Time) (Record, FenceVerdict, error) {
	if !validID(id) {
		return Record{}, FenceVerdict{}, fmt.Errorf("leaseref: invalid lease id %q", id)
	}
	ref := refPrefix + id
	oldOID, hasRef, err := s.currentOID(ctx, ref)
	if err != nil {
		return Record{}, FenceVerdict{}, err
	}
	if !hasRef {
		return Record{}, FenceVerdict{Reason: ReasonNoLease, Detail: "no lease for id " + id + "; reacquire before renewing"}, nil
	}
	cur, err := s.readRef(ctx, ref)
	if err != nil {
		return Record{}, FenceVerdict{}, err
	}
	if cur.Expired(now) {
		return Record{}, FenceVerdict{
			Reason:  ReasonNoLease,
			Current: cur.Generation,
			Holder:  cur.Holder,
			Detail:  fmt.Sprintf("lease %s expired; reacquire (a lapsed lease is not revived by a renew — a peer may already own it)", id),
		}, nil
	}
	if holder == "" || cur.Holder != holder {
		return Record{}, FenceVerdict{
			Reason:  ReasonStaleLease,
			Current: cur.Generation,
			Holder:  cur.Holder,
			Detail:  fmt.Sprintf("lease %s is now held by %q, not %q — halt and reacquire", id, cur.Holder, holder),
		}, nil
	}
	out := cur
	out.RenewedAt = now.Unix()
	if ttlSeconds > 0 {
		out.TTLSeconds = ttlSeconds
	}
	return s.commitFenced(ctx, ref, out, oldOID, true, "renew", id)
}

// currentOID returns the object id ref currently points at and whether ref exists, via
// `git rev-parse --verify --quiet <ref>` (the probe has() uses, but it KEEPS the object id
// for the compare-and-swap old-value). A non-executable git is the only hard error.
func (s *Store) currentOID(ctx context.Context, ref string) (string, bool, error) {
	out, code, err := s.run(ctx, s.dir, "rev-parse", "--verify", "--quiet", ref)
	if err != nil {
		return "", false, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return "", false, nil
	}
	return strings.TrimSpace(out), true, nil
}

// casWrite writes rec as a blob and points ref at it under an update-ref OLD-VALUE
// compare-and-swap, so a concurrent same-host writer that advanced ref since oldOID was read
// loses the race (update-ref exits non-zero) instead of clobbering the token. For a CREATE
// (hadRef == false) it uses git's zero-OID "must not exist" sentinel, sized to the repo's
// object format, so even the first acquire fails closed if a peer created the ref first.
// Returns written == false on a CAS loss (a value, not an error); a real git failure errors.
func (s *Store) casWrite(ctx context.Context, ref string, rec Record, oldOID string, hadRef bool) (bool, error) {
	blob, err := json.Marshal(rec)
	if err != nil {
		return false, fmt.Errorf("leaseref: marshal record: %w", err)
	}
	sha, err := s.writeBlob(ctx, blob)
	if err != nil {
		return false, err
	}
	old := oldOID
	if !hadRef {
		if old, err = s.zeroOID(ctx); err != nil {
			return false, err
		}
	}
	_, code, err := s.run(ctx, s.dir, "update-ref", ref, sha, old)
	if err != nil {
		return false, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return false, nil // CAS lost: ref advanced under us (or a create raced)
	}
	return true, nil
}

// zeroOID returns git's all-zeros object id for this repo's hash algorithm — the update-ref
// old-value sentinel meaning "the ref must not currently exist", used to make a CREATE atomic
// (it fails if a peer created the ref first). The width depends on the object format (40 hex
// for sha1, 64 for sha256); it is probed via `git rev-parse --show-object-format`, defaulting
// to sha1 when the probe is unavailable (an old git, or the injected test seam).
func (s *Store) zeroOID(ctx context.Context) (string, error) {
	out, code, err := s.run(ctx, s.dir, "rev-parse", "--show-object-format")
	if err != nil {
		return "", fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code == 0 && strings.TrimSpace(out) == "sha256" {
		return strings.Repeat("0", 64), nil
	}
	return strings.Repeat("0", 40), nil
}
