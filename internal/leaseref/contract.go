package leaseref

// contract.go manages ticket execution contract leases under refs/fak/locks/contract-<ticket_id>.
//
// Part of epic #11163 and issue #11165.
//
// leaseref.Record is specialized for spatial file-tree exclusion (TreeGlobs). Storing
// queue lifecycle states (token budgets, verification commands, worktree paths) on Record
// would destabilize tree arbitration and cause high-churn ref rewrites. This file provides
// a dedicated ref kind for ticket contracts to govern queue execution without polluting tree locks.
//
// THE NAMESPACE SPLIT: a contract ref is refs/fak/locks/contract-<ticket_id>; the lock-lease
// readers (List/Live/LiveLeases), session readers, and intent readers each filter out contract
// refs, so the views stay distinct over the one shared for-each-ref scan.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// contractPrefix is the basename prefix that marks a contract record apart from a lock
// lease, session descriptor, and intent lease under refs/fak/locks/.
const contractPrefix = "contract-"

// DefaultContractTTLSeconds is the default lifetime of a contract lease: one hour (3600s).
const DefaultContractTTLSeconds int = 3600

// ContractState represents the lifecycle state of a ticket execution contract.
type ContractState string

const (
	ContractStatePending   ContractState = "PENDING"
	ContractStateExecuting ContractState = "EXECUTING"
	ContractStateYieldedIO ContractState = "YIELDED_IO"
	ContractStateVerifying ContractState = "VERIFYING"
	ContractStateSucceeded ContractState = "SUCCEEDED"
	ContractStateFailed    ContractState = "FAILED"
)

// Matching short-name aliases for contract states.
const (
	PENDING    = ContractStatePending
	EXECUTING  = ContractStateExecuting
	YIELDED_IO = ContractStateYieldedIO
	VERIFYING  = ContractStateVerifying
	SUCCEEDED  = ContractStateSucceeded
	FAILED     = ContractStateFailed
)

// Pace tiers for execution contracts.
const (
	PaceTierFrontier  = "frontier"
	PaceTierCommodity = "commodity"
	PaceTierEvalOnly  = "eval_only"
)

// Reason vocabulary and sentinel errors for contract operations.
const (
	ReasonContractHeld      = "CONTRACT_HELD"
	ReasonContractContended = "CONTRACT_CONTENDED"
	ReasonContractNotFound  = "CONTRACT_NOT_FOUND"
	ReasonContractExpired   = "CONTRACT_EXPIRED"
)

var (
	ErrContractHeld     = errors.New("leaseref: contract already held")
	ErrContractNotFound = errors.New("leaseref: contract not found")
	ErrContractExpired  = errors.New("leaseref: contract expired")
	ErrCASContended     = errors.New("leaseref: contract lease contended (CAS lost)")
)

// ContractRecord is one ticket contract persisted under refs/fak/locks/contract-<ticket_id>.
type ContractRecord struct {
	TicketID    string        `json:"ticket_id"`
	Holder      string        `json:"holder"`
	SessionID   string        `json:"session_id,omitempty"`
	State       ContractState `json:"state"`
	PaceTier    string        `json:"pace_tier,omitempty"`
	TokenBudget int64         `json:"token_budget,omitempty"`
	TokensUsed  int64         `json:"tokens_used,omitempty"`
	TurnLimit   int           `json:"turn_limit,omitempty"`
	WorktreeDir string        `json:"worktree_dir,omitempty"`
	BaseSHA     string        `json:"base_sha,omitempty"`
	VerifyCmd   string        `json:"verify_cmd,omitempty"`
	Generation  int64         `json:"generation,omitempty"`
	AcquiredAt  int64         `json:"acquired_unix"`
	RenewedAt   int64         `json:"renewed_unix,omitempty"`
	TTLSeconds  int           `json:"ttl_seconds"`
}

// UnmarshalJSON supports both acquired_unix/renewed_unix and acquired_at/renewed_at.
func (r *ContractRecord) UnmarshalJSON(data []byte) error {
	type raw ContractRecord
	var aux struct {
		raw
		AcquiredAlt int64 `json:"acquired_at"`
		RenewedAlt  int64 `json:"renewed_at"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*r = ContractRecord(aux.raw)
	if r.AcquiredAt == 0 && aux.AcquiredAlt != 0 {
		r.AcquiredAt = aux.AcquiredAlt
	}
	if r.RenewedAt == 0 && aux.RenewedAlt != 0 {
		r.RenewedAt = aux.RenewedAlt
	}
	return nil
}

// effectiveActiveAt returns the later of AcquiredAt and RenewedAt.
func (r ContractRecord) effectiveActiveAt() int64 {
	if r.RenewedAt > r.AcquiredAt {
		return r.RenewedAt
	}
	return r.AcquiredAt
}

// Expired reports whether the contract lease is past its TTL at time now.
func (r ContractRecord) Expired(now time.Time) bool {
	return expired(now, int64(r.TTLSeconds), r.effectiveActiveAt())
}

// Ref returns the full ref path: refs/fak/locks/contract-<ticket_id>.
func (r ContractRecord) Ref() string {
	return contractRefName(r.TicketID)
}

// isContractRef reports whether a full ref under refs/fak/locks/ is a CONTRACT ref.
func isContractRef(ref string) bool {
	return strings.HasPrefix(ref, refPrefix+contractPrefix)
}

// contractRefName returns the full ref path for a ticket ID under refs/fak/locks/contract-*.
func contractRefName(ticketID string) string {
	id := strings.TrimPrefix(ticketID, refPrefix)
	id = strings.TrimPrefix(id, contractPrefix)
	return refPrefix + contractPrefix + id
}

// IsContractRef is the exported form of isContractRef.
func IsContractRef(ref string) bool {
	return isContractRef(ref)
}

// ContractRefName is the exported form of contractRefName.
func ContractRefName(ticketID string) string {
	return contractRefName(ticketID)
}

// AcquireContract acquires or transitions a contract lease for rec.TicketID.
//
// Rules:
//   - No existing ref: fresh acquisition (generation 1, AcquiredAt = now).
//   - Existing ref is expired or terminal (SUCCEEDED/FAILED): takeover/transition (generation = cur.Generation + 1).
//   - Existing ref is live by same holder: heartbeat/update (generation unchanged).
//   - Existing ref is live by different holder: refused with ErrContractHeld.
//
// All writes go through compare-and-swap (casWriteJSON) to guarantee same-host atomicity.
func (s *Store) AcquireContract(ctx context.Context, rec ContractRecord, now ...time.Time) (ContractRecord, error) {
	if strings.TrimSpace(rec.TicketID) == "" {
		return ContractRecord{}, fmt.Errorf("leaseref: empty ticket id")
	}
	cleanID := strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(rec.TicketID), refPrefix), contractPrefix)
	if !validID(cleanID) {
		return ContractRecord{}, fmt.Errorf("leaseref: invalid ticket id %q", rec.TicketID)
	}
	rec.TicketID = cleanID

	tNow := time.Now()
	if len(now) > 0 && !now[0].IsZero() {
		tNow = now[0]
	}

	if rec.TTLSeconds <= 0 {
		rec.TTLSeconds = DefaultContractTTLSeconds
	}
	if rec.State == "" {
		rec.State = ContractStatePending
	}

	ref := contractRefName(rec.TicketID)
	oldOID, hasRef, err := s.currentOID(ctx, ref)
	if err != nil {
		return ContractRecord{}, err
	}

	var cur ContractRecord
	if hasRef {
		if cur, err = s.readContractRef(ctx, ref); err != nil {
			return ContractRecord{}, err
		}
	}

	out := rec
	switch {
	case !hasRef:
		out.Generation = 1
		out.AcquiredAt = tNow.Unix()
		out.RenewedAt = 0
	case cur.Expired(tNow) || cur.State == ContractStateSucceeded || cur.State == ContractStateFailed:
		out.Generation = cur.Generation + 1
		out.AcquiredAt = tNow.Unix()
		out.RenewedAt = 0
	case cur.Holder == rec.Holder && rec.Holder != "":
		out = cur
		out.RenewedAt = tNow.Unix()
		if rec.TTLSeconds > 0 {
			out.TTLSeconds = rec.TTLSeconds
		}
		if rec.State != "" {
			out.State = rec.State
		}
		if rec.SessionID != "" {
			out.SessionID = rec.SessionID
		}
		if rec.PaceTier != "" {
			out.PaceTier = rec.PaceTier
		}
		if rec.TokenBudget > 0 {
			out.TokenBudget = rec.TokenBudget
		}
		if rec.TokensUsed > 0 {
			out.TokensUsed = rec.TokensUsed
		}
		if rec.TurnLimit > 0 {
			out.TurnLimit = rec.TurnLimit
		}
		if rec.WorktreeDir != "" {
			out.WorktreeDir = rec.WorktreeDir
		}
		if rec.BaseSHA != "" {
			out.BaseSHA = rec.BaseSHA
		}
		if rec.VerifyCmd != "" {
			out.VerifyCmd = rec.VerifyCmd
		}
		if rec.Generation > cur.Generation {
			out.Generation = rec.Generation
		}
	default:
		return ContractRecord{}, fmt.Errorf("leaseref: contract for ticket %q is held by %q (generation %d): %w",
			rec.TicketID, cur.Holder, cur.Generation, ErrContractHeld)
	}

	written, err := s.casWriteJSON(ctx, ref, out, oldOID, hasRef)
	if err != nil {
		return ContractRecord{}, err
	}
	if !written {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %s changed under acquire (CAS lost): %w", rec.TicketID, ErrCASContended)
	}
	return out, nil
}

// RenewContract extends the lifetime of a live contract lease held by holder.
// The generation remains strictly unchanged (liveness heartbeat).
func (s *Store) RenewContract(ctx context.Context, ticketID, holder string, ttlSeconds int, now ...time.Time) (ContractRecord, error) {
	if strings.TrimSpace(ticketID) == "" {
		return ContractRecord{}, fmt.Errorf("leaseref: empty ticket id")
	}
	cleanID := strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(ticketID), refPrefix), contractPrefix)
	if !validID(cleanID) {
		return ContractRecord{}, fmt.Errorf("leaseref: invalid ticket id %q", ticketID)
	}
	ref := contractRefName(cleanID)

	tNow := time.Now()
	if len(now) > 0 && !now[0].IsZero() {
		tNow = now[0]
	}

	oldOID, hasRef, err := s.currentOID(ctx, ref)
	if err != nil {
		return ContractRecord{}, err
	}
	if !hasRef {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %q not found: %w", cleanID, ErrContractNotFound)
	}

	cur, err := s.readContractRef(ctx, ref)
	if err != nil {
		return ContractRecord{}, err
	}
	if cur.Expired(tNow) {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %q expired: %w", cleanID, ErrContractExpired)
	}
	if holder != "" && cur.Holder != holder {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %q is held by %q, not %q: %w", cleanID, cur.Holder, holder, ErrContractHeld)
	}

	out := cur
	out.RenewedAt = tNow.Unix()
	if ttlSeconds > 0 {
		out.TTLSeconds = ttlSeconds
	}

	written, err := s.casWriteJSON(ctx, ref, out, oldOID, true)
	if err != nil {
		return ContractRecord{}, err
	}
	if !written {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %s changed under renew (CAS lost): %w", cleanID, ErrCASContended)
	}
	return out, nil
}

// UpdateContractState transitions the state of an existing, live contract lease.
func (s *Store) UpdateContractState(ctx context.Context, ticketID string, state ContractState, now ...time.Time) (ContractRecord, error) {
	if strings.TrimSpace(ticketID) == "" {
		return ContractRecord{}, fmt.Errorf("leaseref: empty ticket id")
	}
	cleanID := strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(ticketID), refPrefix), contractPrefix)
	if !validID(cleanID) {
		return ContractRecord{}, fmt.Errorf("leaseref: invalid ticket id %q", ticketID)
	}
	ref := contractRefName(cleanID)

	tNow := time.Now()
	if len(now) > 0 && !now[0].IsZero() {
		tNow = now[0]
	}

	oldOID, hasRef, err := s.currentOID(ctx, ref)
	if err != nil {
		return ContractRecord{}, err
	}
	if !hasRef {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %q not found: %w", cleanID, ErrContractNotFound)
	}

	cur, err := s.readContractRef(ctx, ref)
	if err != nil {
		return ContractRecord{}, err
	}
	if cur.Expired(tNow) {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %q expired: %w", cleanID, ErrContractExpired)
	}

	out := cur
	out.State = state
	out.RenewedAt = tNow.Unix()

	written, err := s.casWriteJSON(ctx, ref, out, oldOID, true)
	if err != nil {
		return ContractRecord{}, err
	}
	if !written {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %s changed under state update (CAS lost): %w", cleanID, ErrCASContended)
	}
	return out, nil
}

// UpdateContract updates mutable fields of an existing live contract under CAS.
func (s *Store) UpdateContract(ctx context.Context, rec ContractRecord, now ...time.Time) (ContractRecord, error) {
	if strings.TrimSpace(rec.TicketID) == "" {
		return ContractRecord{}, fmt.Errorf("leaseref: empty ticket id")
	}
	cleanID := strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(rec.TicketID), refPrefix), contractPrefix)
	if !validID(cleanID) {
		return ContractRecord{}, fmt.Errorf("leaseref: invalid ticket id %q", rec.TicketID)
	}
	ref := contractRefName(cleanID)

	tNow := time.Now()
	if len(now) > 0 && !now[0].IsZero() {
		tNow = now[0]
	}

	oldOID, hasRef, err := s.currentOID(ctx, ref)
	if err != nil {
		return ContractRecord{}, err
	}
	if !hasRef {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %q not found: %w", cleanID, ErrContractNotFound)
	}

	cur, err := s.readContractRef(ctx, ref)
	if err != nil {
		return ContractRecord{}, err
	}
	if cur.Expired(tNow) {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %q expired: %w", cleanID, ErrContractExpired)
	}
	if rec.Holder != "" && cur.Holder != rec.Holder {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %q held by %q, not %q: %w", cleanID, cur.Holder, rec.Holder, ErrContractHeld)
	}

	out := cur
	out.RenewedAt = tNow.Unix()
	if rec.State != "" {
		out.State = rec.State
	}
	if rec.SessionID != "" {
		out.SessionID = rec.SessionID
	}
	if rec.PaceTier != "" {
		out.PaceTier = rec.PaceTier
	}
	if rec.TokenBudget > 0 {
		out.TokenBudget = rec.TokenBudget
	}
	if rec.TokensUsed > 0 {
		out.TokensUsed = rec.TokensUsed
	}
	if rec.TurnLimit > 0 {
		out.TurnLimit = rec.TurnLimit
	}
	if rec.WorktreeDir != "" {
		out.WorktreeDir = rec.WorktreeDir
	}
	if rec.BaseSHA != "" {
		out.BaseSHA = rec.BaseSHA
	}
	if rec.VerifyCmd != "" {
		out.VerifyCmd = rec.VerifyCmd
	}
	if rec.TTLSeconds > 0 {
		out.TTLSeconds = rec.TTLSeconds
	}
	if rec.Generation > cur.Generation {
		out.Generation = rec.Generation
	}

	written, err := s.casWriteJSON(ctx, ref, out, oldOID, true)
	if err != nil {
		return ContractRecord{}, err
	}
	if !written {
		return ContractRecord{}, fmt.Errorf("leaseref: contract %s changed under update (CAS lost): %w", cleanID, ErrCASContended)
	}
	return out, nil
}

// GetContract reads back the single contract record for ticketID, or (zero, false, nil)
// when no such ref exists.
func (s *Store) GetContract(ctx context.Context, ticketID string) (ContractRecord, bool, error) {
	if strings.TrimSpace(ticketID) == "" {
		return ContractRecord{}, false, fmt.Errorf("leaseref: empty ticket id")
	}
	cleanID := strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(ticketID), refPrefix), contractPrefix)
	ref := contractRefName(cleanID)
	exists, err := s.has(ctx, ref)
	if err != nil || !exists {
		return ContractRecord{}, false, err
	}
	rec, err := s.readContractRef(ctx, ref)
	if err != nil {
		return ContractRecord{}, false, err
	}
	return rec, true, nil
}

// ListContracts reads every contract record under refs/fak/locks/contract-*, sorted by
// TicketID for a stable view.
func (s *Store) ListContracts(ctx context.Context) ([]ContractRecord, error) {
	recs, err := listRefs(ctx, s, isContractRef, s.readContractRef)
	if err != nil {
		return nil, err
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].TicketID < recs[j].TicketID })
	return recs, nil
}

// LiveContracts partitions ListContracts into live-vs-expired at time now, returning
// the expired ticket IDs alongside so a caller can reap them.
func (s *Store) LiveContracts(ctx context.Context, now ...time.Time) (live []ContractRecord, expired []string, err error) {
	tNow := time.Now()
	if len(now) > 0 && !now[0].IsZero() {
		tNow = now[0]
	}
	all, err := s.ListContracts(ctx)
	if err != nil {
		return nil, nil, err
	}
	live, expired = liveExpire(all, tNow, func(r ContractRecord) string { return r.TicketID })
	return live, expired, nil
}

// ReapContracts deletes every contract lease expired at time now and returns the ticket
// IDs reaped.
func (s *Store) ReapContracts(ctx context.Context, now ...time.Time) (reaped []string, err error) {
	tNow := time.Now()
	if len(now) > 0 && !now[0].IsZero() {
		tNow = now[0]
	}
	_, expired, lerr := s.LiveContracts(ctx, tNow)
	if lerr != nil {
		return nil, lerr
	}
	refOf := func(ticketID string) string { return contractRefName(ticketID) }
	return s.reapRefs(ctx, expired, "reap contract", refOf,
		func(ctx context.Context, ticketID string) error { return s.deleteRef(ctx, refOf(ticketID)) })
}

// ReleaseContract deletes the contract lease for ticketID.
func (s *Store) ReleaseContract(ctx context.Context, ticketID string) error {
	if strings.TrimSpace(ticketID) == "" {
		return fmt.Errorf("leaseref: empty ticket id")
	}
	cleanID := strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(ticketID), refPrefix), contractPrefix)
	return s.deleteRef(ctx, contractRefName(cleanID))
}

// readContractRef reads the blob a contract ref points at and unmarshals the record.
func (s *Store) readContractRef(ctx context.Context, ref string) (ContractRecord, error) {
	out, code, err := s.run(ctx, s.dir, "cat-file", "blob", ref)
	if err != nil {
		return ContractRecord{}, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return ContractRecord{}, fmt.Errorf("leaseref: cat-file blob %s exited %d", ref, code)
	}
	var rec ContractRecord
	if err := json.Unmarshal([]byte(out), &rec); err != nil {
		return ContractRecord{}, fmt.Errorf("leaseref: unmarshal contract record at %s: %w", ref, err)
	}
	if rec.TicketID == "" {
		rec.TicketID = strings.TrimPrefix(strings.TrimPrefix(ref, refPrefix), contractPrefix)
	}
	return rec, nil
}
