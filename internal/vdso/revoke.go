package vdso

// revoke.go — the INTEGRITY-DIRECTION eraser: revocation on refutation, the mutable
// TRUST EPOCH layered over the durable, content-addressed CAS.
//
// Why this exists (the load-bearing gap). scope.go's epoch vector is the CONSISTENCY
// eraser: a write-shaped completion bumps a monotone worldVer / node epoch, which
// strands the reads that write could have changed. That axis only ever advances on a
// real-world MUTATION ("the data changed"), and the CAS beneath it is frozen — the same
// content keeps the same digest forever. Neither can express the OTHER failure mode a
// shared pool has: a later witness REFUTES an already-pooled entry — it was poisoned, or
// the source it claimed never existed — even though nothing WROTE to it. worldVer cannot
// retire it (no write happened) and the durable CAS will happily re-serve the same bytes
// the consistency key still matches. Integrity therefore needs its OWN clock, dual to
// the consistency one:
//
//	worldVer    monotone CONSISTENCY epoch — bumped by writes      (scope.go)
//	trustEpoch  monotone INTEGRITY  epoch — bumped by refutations  (this file)
//
// Each tier-2 entry records the EXTERNAL world-state WITNESS it was admitted under — a
// git commit, blob hash, or lease epoch the orchestration substrate already holds,
// supplied as c.Meta["witness"]. ("" = no external witness => pure-consistency behaviour,
// byte-identical to v0.1.) Revoke(witness) is the refutation trigger:
//
//  1. it evicts EVERY entry admitted under that witness — the causal consumer-set
//     eviction (Part B's C4), targeted, not a blunt full flush: sibling witnesses stay
//     warm;
//  2. it marks the witness refuted so no later read RE-ADMITS under it (without this the
//     durable CAS would silently repopulate the evicted bytes on the next read);
//  3. it bumps trustEpoch and publishes a Revocation on the coherence bus, so every
//     OTHER agent / process sharing the pool (a private cache, a "what changed" feed) is
//     causally evicted too — the cross-agent propagation that is the MESI-invalidate
//     analogue in the integrity direction.
//
// Soundness ("a hit equals a fresh call", preserved). Revocation is sound BY
// CONSTRUCTION and trivially so: it only ever turns a would-be hit into a MISS (-> the
// engine, a fresh call) and only ever refuses a fill. It can never cause a stale serve
// because it never serves anything — it is the SAFE direction of the cache-coherence
// tradeoff. The witness binding is purely ADDITIVE to the consistency key (it is NOT a
// component of keyLocked), so an entry with no witness behaves exactly as v0.1, and an
// entry WITH a witness is gated by BOTH the consistency key AND the not-revoked check —
// two gates that can only ever remove serves, never add one.
//
// What this is NOT (honesty, per PRIOR-ART-fak-partb-residue): this builds C4 (causal
// refutation eviction across the recorded consumer set) and opens the C3-external seam
// (the witness is an external world-state token, not the internal worldVer counter), but
// it does not yet bind the witness into the tier-2 KEY, so two agents reading under
// different witnesses still share by (tool,args,worldVer). Witness-keying is the natural
// follow-on; the revocation axis is the load-bearing half and is what this file ships.

import (
	"container/list"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// defaultWitnessOf extracts the external world-state witness a call/result was admitted under.
// The READ's declared witness (c.Meta["witness"]) wins — it is the world-state the agent
// believes it is reading at — falling back to the result's. "" means no external witness
// was supplied: the entry is governed by the consistency eraser alone (v0.1 behaviour).
// A per-tool adapter (VDSO.RegisterWitness) can override this on the fill path.
func defaultWitnessOf(c *abi.ToolCall, r *abi.Result) string {
	if c != nil && c.Meta != nil {
		if w := c.Meta["witness"]; w != "" {
			return w
		}
	}
	if r != nil && r.Meta != nil {
		if w := r.Meta["witness"]; w != "" {
			return w
		}
	}
	return ""
}

// Revoke is the refutation trigger — the integrity-direction eraser. A later witness has
// shown that every entry admitted under `witness` is stale/poisoned; Revoke evicts them
// all NOW (independent of any world bump), refuses any future re-admission under that
// witness, advances the trust epoch, and publishes a Revocation on the coherence bus so
// cross-agent / cross-process consumers are causally evicted too. It returns the count of
// entries evicted from THIS pool. The empty witness is a no-op. A witness with no
// resident entries still marks-refuted, bumps the epoch, and publishes — a remote
// consumer may hold it even when this pool does not.
func (v *VDSO) Revoke(witness string) (evicted int) {
	if witness == "" {
		return 0
	}
	v.mu.Lock()
	v.ensureRevokedStateLocked()
	epoch := atomic.AddUint64(&v.trustEpoch, 1)
	v.rememberRevokedLocked(witness, epoch)
	var revokedJobs []emitJob
	for el := v.lru.Front(); el != nil; {
		next := el.Next() // capture before Remove unlinks el
		if e := el.Value.(*entry); e.witness == witness {
			v.lru.Remove(el)
			delete(v.cache, e.key)
			abi.UnpinResolved(e.ref) // refutation-evicted under v.mu -> release its CAS pin
			evicted++
			revokedJobs = append(revokedJobs, emitJob{key: e.key, ref: e.ref, witness: e.witness})
		}
		el = next
	}
	v.mutSeq++
	rv := Revocation{
		Witness:    witness,
		Evicted:    evicted,
		TrustEpoch: epoch,
		Seq:        v.mutSeq,
	}
	subs := v.revSubs.snapshot()
	v.mu.Unlock()

	atomic.AddInt64(&v.revocations, 1)
	for _, s := range subs { // subscribers run outside the lock; may re-enter the vDSO
		s.fn(rv)
	}
	for _, j := range revokedJobs { // cachemeta revoke events, also outside the lock
		v.emitCache(CacheRevoke, j.key, j.ref, j.witness)
	}
	return evicted
}

// Revoked reports whether a witness has been refuted (no entry admitted under it may be
// served or re-admitted). Observability / test hook.
func (v *VDSO) Revoked(witness string) bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.revokedLocked(witness)
}

// TrustEpoch reports the integrity clock — the count of refutations observed. It is the
// dual of WorldVersion (the consistency clock): a consumer that caches the pair
// (worldVer, trustEpoch) can detect BOTH a write (worldVer advanced) and a refutation
// (trustEpoch advanced) without re-reading the pool.
func (v *VDSO) TrustEpoch() uint64 { return atomic.LoadUint64(&v.trustEpoch) }

// Revocations reports how many refutations the vDSO has observed (the integrity-bus event
// count) — the dual of Mutations() on the consistency bus.
func (v *VDSO) Revocations() int64 { return atomic.LoadInt64(&v.revocations) }

// SetRevokedWitnessLimit bounds the exact revoked-witness ledger. If lowering the cap
// drops records, the vDSO enters overflow mode: unknown witness-bearing entries fail
// closed until process reset rather than risking stale re-admission.
func (v *VDSO) SetRevokedWitnessLimit(limit int) {
	if limit <= 0 {
		limit = DefaultRevokedWitnessLimit
	}
	v.mu.Lock()
	v.ensureRevokedStateLocked()
	v.revokedCap = limit
	v.trimRevokedLocked()
	v.mu.Unlock()
}

// RevokedWitnessLimit reports the configured exact revoked-witness cap.
func (v *VDSO) RevokedWitnessLimit() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.ensureRevokedStateLocked()
	return v.revokedCap
}

// RevokedWitnesses reports the retained exact revoked-witness count.
func (v *VDSO) RevokedWitnesses() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	return len(v.revoked)
}

// RevocationOverflow reports whether exact revoked-witness history has been trimmed.
// Once true, unknown witness-bearing entries are treated as revoked to preserve soundness.
func (v *VDSO) RevocationOverflow() bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.revokedOverflow
}

func (v *VDSO) ensureRevokedStateLocked() {
	if v.revokedCap <= 0 {
		v.revokedCap = DefaultRevokedWitnessLimit
	}
	if v.revoked == nil {
		v.revoked = map[string]uint64{}
	}
	if v.revokedLRU == nil {
		v.revokedLRU = list.New()
	}
	if v.revokedIndex == nil {
		v.revokedIndex = map[string]*list.Element{}
	}
}

func (v *VDSO) rememberRevokedLocked(witness string, epoch uint64) {
	if witness == "" {
		return
	}
	v.ensureRevokedStateLocked()
	v.revoked[witness] = epoch
	if el := v.revokedIndex[witness]; el != nil {
		v.revokedLRU.MoveToFront(el)
	} else {
		v.revokedIndex[witness] = v.revokedLRU.PushFront(witness)
	}
	v.trimRevokedLocked()
}

func (v *VDSO) trimRevokedLocked() {
	v.ensureRevokedStateLocked()
	for len(v.revoked) > v.revokedCap {
		el := v.revokedLRU.Back()
		if el == nil {
			return
		}
		witness := el.Value.(string)
		v.revokedLRU.Remove(el)
		delete(v.revokedIndex, witness)
		delete(v.revoked, witness)
		v.revokedOverflow = true
	}
}

func (v *VDSO) revokedLocked(witness string) bool {
	if witness == "" {
		return false
	}
	v.ensureRevokedStateLocked()
	if el := v.revokedIndex[witness]; el != nil {
		v.revokedLRU.MoveToFront(el)
		return true
	}
	// Once exact history has been trimmed, an unknown witness might be one of the
	// trimmed refutations. Denying witness-bearing cache/page-in is the sound side.
	return v.revokedOverflow
}

// Revocation is one refutation observed at the kernel's single integrity-notification
// point. Witness names the refuted external world-state witness; Evicted is how many
// pooled entries it stranded in this pool; TrustEpoch is the integrity clock after the
// bump; Seq is the per-VDSO monotone sequence SHARED with Mutation — one total order over
// the coherence bus, so a subscriber can interleave writes and refutations.
type Revocation struct {
	Witness    string // the refuted external world-state witness
	Evicted    int    // entries stranded in this pool (the local consumer-set size)
	TrustEpoch uint64 // integrity epoch after the bump — a monotone refutation clock
	Seq        uint64 // shared coherence-bus sequence (ordering without a wall clock)
}

// SubscribeRevocations registers an observer of refutations and returns a cancel func.
// It is the integrity-direction companion to Subscribe(func(Mutation)): the cache's own
// eviction is already done inside Revoke, so subscribers are ADDITIONAL observers (a
// cross-agent private-cache invalidator, the "what changed" feed, an audit log). Invoked
// synchronously AFTER the eviction and OUTSIDE v.mu, so a subscriber may re-enter the
// vDSO. Only refutations fire it — never the read hot path.
func (v *VDSO) SubscribeRevocations(fn func(Revocation)) (cancel func()) {
	return subscribe(v, &v.revSubs, fn)
}
