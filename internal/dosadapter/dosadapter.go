// Package dosadapter provides the adapter bridge between the fak agent kernel
// and the DOS trust substrate, handling arbitration requests, lease verification,
// claim witnessing, and structured refusal translation with fail-closed semantics.
//
// Invariant: Exclusive leases must be strictly tree-disjoint across concurrent holders.
// Guard: fail-closed arbitration defaults to refusal whenever evidence, witness, or communication fails.
package dosadapter

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"path"
	"strings"
	"sync"
	"time"
)

// Invariant: Exclusive leases must be strictly tree-disjoint across concurrent holders.
// Guard: fail-closed arbitration defaults to refusal whenever evidence, witness, or communication fails.

// LockMode identifies the concurrency lock acquisition mode for a lane lease.
type LockMode string

const (
	// LockModeExclusive requires strictly tree-disjoint file trees from any other lease.
	LockModeExclusive LockMode = "exclusive"

	// LockModeShared permits concurrent access with other shared leases over the same tree.
	LockModeShared LockMode = "shared"
)

// Arbitration outcome status constants.
const (
	// OutcomeAcquire indicates an arbitration request was successfully admitted.
	OutcomeAcquire = "acquire"

	// OutcomeRefuse indicates an arbitration request was refused.
	OutcomeRefuse = "refuse"
)

// Closed refusal category vocabulary matching the DOS taxonomy.
const (
	// CategoryTrueDrain signifies an exhaustion condition where no further work remains.
	CategoryTrueDrain = "TRUE_DRAIN"

	// CategoryOperatorGate indicates an operator decision or manual intervention is required.
	CategoryOperatorGate = "OPERATOR_GATE"

	// CategoryStaleClaim denotes that a claimed artifact or state is no longer fresh.
	CategoryStaleClaim = "STALE_CLAIM"

	// CategoryMisroute indicates an invalid routing, off-trunk target, or tree conflict.
	CategoryMisroute = "MISROUTE"

	// CategoryUnclassified is the fail-closed catch-all category for unknown refusal tokens.
	CategoryUnclassified = "UNCLASSIFIED"
)

// Standard closed refusal reason tokens from dos.toml.
const (
	// ReasonCollisionRisk is emitted when requested trees overlap an existing exclusive lease.
	ReasonCollisionRisk = "COLLISION_RISK"

	// ReasonLaneDrained is emitted when no further work items remain in the requested lane.
	ReasonLaneDrained = "LANE_DRAINED"

	// ReasonOperatorGate is emitted when an explicit operator gate halts unattended progress.
	ReasonOperatorGate = "OPERATOR_GATE"

	// ReasonStaleClaim is emitted when git/tree claims contradict ground truth.
	ReasonStaleClaim = "STALE_CLAIM"

	// ReasonOffTrunk is emitted when operations are attempted off the main trunk branch.
	ReasonOffTrunk = "OFF_TRUNK"

	// ReasonUnclassified is the fail-closed token representing unmapped refusal reasons.
	ReasonUnclassified = "UNCLASSIFIED"
)

var (
	// ErrArbitrationRefused is returned when an arbitration request cannot be granted.
	ErrArbitrationRefused = errors.New("dosadapter: arbitration refused")

	// ErrInvalidLease is returned when a lease request or descriptor fails validation.
	ErrInvalidLease = errors.New("dosadapter: invalid lease request")

	// ErrCorruptedEvidence is returned when an evidence or witness payload fails integrity checks.
	ErrCorruptedEvidence = errors.New("dosadapter: corrupted evidence or witness")

	// ErrDisjointnessViolation is returned when concurrent trees violate mutual exclusivity.
	ErrDisjointnessViolation = errors.New("dosadapter: tree disjointness violation")
)

// LeaseRequest specifies an arbitration request to hold a lane over a set of file trees.
type LeaseRequest struct {
	// ID uniquely identifies this lease request or granted lease.
	ID string `json:"id"`

	// Lane is the target workspace lane name (e.g., "gateway", "dosadapter").
	Lane string `json:"lane"`

	// LaneKind provides lane categorization metadata (e.g., "cluster", "leaf").
	LaneKind string `json:"lane_kind"`

	// LockMode designates "exclusive" or "shared" access (defaults to exclusive).
	LockMode LockMode `json:"lock_mode"`

	// Tree contains repository-relative glob patterns covered by this lease.
	Tree []string `json:"tree"`

	// WorkerID identifies the worker or subagent requesting the lease.
	WorkerID string `json:"worker_id"`

	// CreatedAt marks when the lease request was issued.
	CreatedAt time.Time `json:"created_at"`

	// TTL defines the maximum lease hold duration.
	TTL time.Duration `json:"ttl"`
}

// DecisionWitness encapsulates cryptographic proof and provenance for an arbitration verdict.
type DecisionWitness struct {
	// WitnessID uniquely identifies this witness record.
	WitnessID string `json:"witness_id"`

	// DecisionSHA is the SHA-256 hash over the normalized decision payload.
	DecisionSHA string `json:"decision_sha"`

	// Issuer is the authority or server that generated this witness.
	Issuer string `json:"issuer"`

	// IssuedAt is the timestamp when the witness was produced.
	IssuedAt time.Time `json:"issued_at"`

	// Signature is the tamper-evident signature or digest over the witness fields.
	Signature string `json:"signature"`

	// ValidUntil marks the expiration boundary for this witness.
	ValidUntil time.Time `json:"valid_until"`
}

// RefusalReason models a structured, typed refusal reason from the closed DOS vocabulary.
type RefusalReason struct {
	// Token is the canonical UPPER_SNAKE_CASE refusal token.
	Token string `json:"token"`

	// Category is the high-level classification of this refusal.
	Category string `json:"category"`

	// Description provides concise context explaining the refusal condition.
	Description string `json:"description"`

	// Refusal indicates whether this reason is blocking (true) or advisory (false).
	Refusal bool `json:"refusal"`

	// Remedy suggests the recommended corrective action for recovery.
	Remedy string `json:"remedy"`
}

// ArbitrationOutcome represents the adjudicated result of a lease arbitration request.
type ArbitrationOutcome struct {
	// Outcome is either OutcomeAcquire ("acquire") or OutcomeRefuse ("refuse").
	Outcome string `json:"outcome"`

	// Lane is the resolved lane name.
	Lane string `json:"lane"`

	// LaneKind is the lane kind descriptor.
	LaneKind string `json:"lane_kind"`

	// LockMode is the effective lock mode ("exclusive" or "shared").
	LockMode LockMode `json:"lock_mode"`

	// Tree is the list of repository-relative path globs granted or evaluated.
	Tree []string `json:"tree"`

	// Reason contains the structured refusal details when Outcome is OutcomeRefuse.
	Reason RefusalReason `json:"reason,omitempty"`

	// Witness contains the cryptographic decision witness verifying this verdict.
	Witness DecisionWitness `json:"witness"`

	// Interpretation is a human-readable one-line verdict summary (GO/STOP advice).
	Interpretation string `json:"interpretation"`

	// LeaseSource names which live-lease view produced this verdict (ArbitrateSet only).
	LeaseSource LeaseSource `json:"lease_source,omitempty"`

	// LeaseGeneration is the generation counter of the deciding live-lease set.
	LeaseGeneration int `json:"lease_generation,omitempty"`

	// LeaseObservedUnix is the observation time of the deciding live-lease set, in Unix seconds.
	LeaseObservedUnix int64 `json:"lease_observed_unix,omitempty"`
}

// LeaseSource identifies where a LiveLeaseSet's rows were read from, so a verdict
// can be audited against the view that produced it.
type LeaseSource string

const (
	// LeaseSourceSnapshot names a caller-supplied point-in-time snapshot.
	LeaseSourceSnapshot LeaseSource = "snapshot"

	// LeaseSourceLiveJournal names rows re-read from the live lane journal.
	LeaseSourceLiveJournal LeaseSource = "live-journal"

	// LeaseSourceCaller names rows handed in explicitly by the caller.
	LeaseSourceCaller LeaseSource = "caller"

	// LeaseSourceInMemory names the AdapterClient's own in-process lease table.
	LeaseSourceInMemory LeaseSource = "in-memory"

	// LeaseSourceUnreadable names a view that could not be read; fail closed.
	LeaseSourceUnreadable LeaseSource = "unreadable"
)

// LiveLeaseSet is an explicit, self-describing view of the live leases an
// arbitration decision is adjudicated against. It carries its own provenance
// (Source, Generation, ObservedUnix) so a stale snapshot cannot silently
// disagree with the live journal: the caller can observe staleness and re-derive.
type LiveLeaseSet struct {
	// Source names where these rows came from.
	Source LeaseSource

	// Generation increments whenever the underlying lease view changes.
	Generation int

	// ObservedUnix is when the rows were observed, in Unix seconds (0 == unknown).
	ObservedUnix int64

	// leases holds the observed holders; adjudication reads only this slice.
	leases []LeaseRequest
}

// NewLiveLeaseSet builds a LiveLeaseSet from an explicit set of lease rows.
func NewLiveLeaseSet(leases []LeaseRequest, observed time.Time, generation int, source LeaseSource) LiveLeaseSet {
	rows := make([]LeaseRequest, len(leases))
	copy(rows, leases)

	var observedUnix int64
	if !observed.IsZero() {
		observedUnix = observed.Unix()
	}
	return LiveLeaseSet{
		Source:       source,
		Generation:   generation,
		ObservedUnix: observedUnix,
		leases:       rows,
	}
}

// DeriveLiveLeaseSet re-derives a LiveLeaseSet from an observation. It is
// observably identical to NewLiveLeaseSet for identical inputs; the distinct
// name marks the caller's intent to RE-DERIVE the live set after a reap.
func DeriveLiveLeaseSet(leases []LeaseRequest, observed time.Time, generation int, source LeaseSource) LiveLeaseSet {
	return NewLiveLeaseSet(leases, observed, generation, source)
}

// Stale reports whether this view is too old to trust as fresh.
// Guard: fail-closed — an unreadable source, an unknown observation time, or an
// observation older than maxAge is stale. The boundary is half-open: an age
// exactly equal to maxAge is still fresh (age > maxAge is stale).
func (s LiveLeaseSet) Stale(now time.Time, maxAge time.Duration) bool {
	if s.Source == LeaseSourceUnreadable {
		return true
	}
	if s.ObservedUnix <= 0 {
		return true
	}
	observed := time.Unix(s.ObservedUnix, 0)
	return now.Sub(observed) > maxAge
}

// Describe returns a non-empty diagnostic naming the source token.
func (s LiveLeaseSet) Describe() string {
	return fmt.Sprintf("live-lease-set{source=%s generation=%d observed_unix=%d holders=%d}",
		s.Source, s.Generation, s.ObservedUnix, len(s.leases))
}

// AdapterClient coordinates lease arbitration, witness verification, and refusal handling.
type AdapterClient struct {
	mu           sync.RWMutex
	activeLeases map[string]LeaseRequest
	clock        func() time.Time
	issuer       string
	generation   int
}

// ClientOption configures an AdapterClient.
type ClientOption func(*AdapterClient)

// WithClock sets a custom time provider for determinism in testing.
func WithClock(clock func() time.Time) ClientOption {
	return func(c *AdapterClient) {
		if clock != nil {
			c.clock = clock
		}
	}
}

// WithIssuer sets the issuer identity string in decision witnesses.
func WithIssuer(issuer string) ClientOption {
	return func(c *AdapterClient) {
		if issuer != "" {
			c.issuer = issuer
		}
	}
}

// WithInitialLeases seeds the client with an initial snapshot of active leases.
func WithInitialLeases(leases []LeaseRequest) ClientOption {
	return func(c *AdapterClient) {
		for _, l := range leases {
			if l.ID != "" {
				c.activeLeases[l.ID] = l
			}
		}
	}
}

// NewAdapterClient constructs a new AdapterClient with safe defaults.
func NewAdapterClient(opts ...ClientOption) *AdapterClient {
	c := &AdapterClient{
		activeLeases: make(map[string]LeaseRequest),
		clock:        time.Now,
		issuer:       "fak/dosadapter",
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// ValidateLease verifies the structural validity and consistency of a LeaseRequest.
// Invariant: A valid lease request must specify a non-empty ID, Lane, and at least one tree pattern.
func ValidateLease(req LeaseRequest) error {
	if strings.TrimSpace(req.ID) == "" {
		return fmt.Errorf("%w: missing lease ID", ErrInvalidLease)
	}
	if strings.TrimSpace(req.Lane) == "" {
		return fmt.Errorf("%w: missing lane name", ErrInvalidLease)
	}
	if len(req.Tree) == 0 {
		return fmt.Errorf("%w: tree must contain at least one pattern", ErrInvalidLease)
	}
	for i, pattern := range req.Tree {
		if strings.TrimSpace(pattern) == "" {
			return fmt.Errorf("%w: empty pattern at index %d", ErrInvalidLease, i)
		}
	}
	mode := strings.ToLower(strings.TrimSpace(string(req.LockMode)))
	if mode != "" && mode != string(LockModeExclusive) && mode != string(LockModeShared) {
		return fmt.Errorf("%w: unsupported lock mode %q", ErrInvalidLease, req.LockMode)
	}
	return nil
}

// normalizeLockMode ensures empty lock mode defaults to exclusive.
func normalizeLockMode(mode LockMode) LockMode {
	cleaned := strings.ToLower(strings.TrimSpace(string(mode)))
	if cleaned == string(LockModeShared) {
		return LockModeShared
	}
	return LockModeExclusive
}

// normalizePath converts Windows-style backslashes and trims redundant slashes.
func normalizePath(p string) string {
	cleaned := strings.ReplaceAll(p, "\\", "/")
	cleaned = strings.TrimSpace(cleaned)
	for strings.Contains(cleaned, "//") {
		cleaned = strings.ReplaceAll(cleaned, "//", "/")
	}
	return strings.TrimSuffix(cleaned, "/")
}

// PathMatchesPattern evaluates whether a normalized path matches a glob pattern.
func PathMatchesPattern(pattern, targetPath string) bool {
	normPat := normalizePath(pattern)
	normPath := normalizePath(targetPath)

	if normPat == normPath || normPat == "**" || normPat == "**/*" {
		return true
	}

	if strings.HasPrefix(normPat, "**/") {
		suffix := strings.TrimPrefix(normPat, "**/")
		if matched, err := path.Match(suffix, path.Base(normPath)); err == nil && matched {
			return true
		}
	}

	if strings.HasSuffix(normPat, "/**") {
		prefix := strings.TrimSuffix(normPat, "/**")
		if prefix == "" || normPath == prefix || strings.HasPrefix(normPath, prefix+"/") {
			return true
		}
	}

	if strings.HasSuffix(normPat, "/*") {
		dir := strings.TrimSuffix(normPat, "/*")
		if normPath == dir || path.Dir(normPath) == dir {
			return true
		}
	}

	if matched, err := path.Match(normPat, normPath); err == nil && matched {
		return true
	}

	return false
}

// PatternsOverlap determines if two glob patterns can match overlapping file trees.
func PatternsOverlap(patA, patB string) bool {
	normA := normalizePath(patA)
	normB := normalizePath(patB)

	if normA == normB || normA == "**" || normA == "**/*" || normB == "**" || normB == "**/*" {
		return true
	}

	if strings.HasPrefix(normB, normA+"/") || strings.HasPrefix(normA, normB+"/") {
		return true
	}

	prefA := strings.TrimSuffix(normA, "/**")
	prefB := strings.TrimSuffix(normB, "/**")

	if strings.HasSuffix(normA, "/**") && strings.HasSuffix(normB, "/**") {
		if prefA == "" || prefB == "" || prefA == prefB {
			return true
		}
		if strings.HasPrefix(prefB, prefA+"/") || strings.HasPrefix(prefA, prefB+"/") {
			return true
		}
		return false
	}

	if strings.HasSuffix(normA, "/**") {
		if prefA == "" || normB == prefA || strings.HasPrefix(normB, prefA+"/") {
			return true
		}
	}

	if strings.HasSuffix(normB, "/**") {
		if prefB == "" || normA == prefB || strings.HasPrefix(normA, prefB+"/") {
			return true
		}
	}

	if PathMatchesPattern(normA, normB) || PathMatchesPattern(normB, normA) {
		return true
	}

	return false
}

// TreesOverlap checks whether two sets of file path globs intersect.
// Invariant: Two tree sets overlap if any single pattern pair intersects.
func TreesOverlap(treeA, treeB []string) bool {
	for _, a := range treeA {
		for _, b := range treeB {
			if PatternsOverlap(a, b) {
				return true
			}
		}
	}
	return false
}

// CheckDisjoint evaluates whether a LeaseRequest is disjoint from an active lease set.
// Invariant: Exclusive leases must be strictly tree-disjoint across concurrent holders.
// Guard: fail-closed arbitration defaults to refusal whenever evidence, witness, or communication fails.
func CheckDisjoint(req LeaseRequest, active []LeaseRequest) error {
	reqMode := normalizeLockMode(req.LockMode)
	for _, holder := range active {
		if holder.ID == req.ID {
			continue // ignore re-evaluation against own lease
		}
		holderMode := normalizeLockMode(holder.LockMode)

		// Rule: shared/shared may overlap; any exclusive participant must be tree-disjoint.
		if reqMode == LockModeShared && holderMode == LockModeShared {
			continue
		}

		if TreesOverlap(req.Tree, holder.Tree) {
			return fmt.Errorf("%w: requested lane %q overlaps active lane %q (holder ID %q)",
				ErrDisjointnessViolation, req.Lane, holder.Lane, holder.ID)
		}
	}
	return nil
}

// registeredRefusals holds the known closed refusal definitions.
var registeredRefusals = map[string]RefusalReason{
	ReasonCollisionRisk: {
		Token:       ReasonCollisionRisk,
		Category:    CategoryMisroute,
		Description: "file tree intersects an existing exclusive lease holder",
		Refusal:     true,
		Remedy:      "carve a disjoint lane or wait for colliding lease release",
	},
	ReasonLaneDrained: {
		Token:       ReasonLaneDrained,
		Category:    CategoryTrueDrain,
		Description: "no further work items remain on the requested lane",
		Refusal:     true,
		Remedy:      "replan or switch to another lane with pending work",
	},
	ReasonOperatorGate: {
		Token:       ReasonOperatorGate,
		Category:    CategoryOperatorGate,
		Description: "operator confirmation required before taking this lane",
		Refusal:     true,
		Remedy:      "obtain operator gate clearance or pass explicit override",
	},
	ReasonStaleClaim: {
		Token:       ReasonStaleClaim,
		Category:    CategoryStaleClaim,
		Description: "claim evidence does not match repository ground truth",
		Refusal:     true,
		Remedy:      "refresh workspace working tree and re-verify claim artifacts",
	},
	ReasonOffTrunk: {
		Token:       ReasonOffTrunk,
		Category:    CategoryMisroute,
		Description: "work attempted off the main trunk branch",
		Refusal:     true,
		Remedy:      "return to main trunk branch before dispatching worker",
	},
}

// ParseRefusal translates a raw refusal token into a structured RefusalReason.
// Guard: fail-closed mapping maps unknown tokens to CategoryUnclassified with Refusal=true.
func ParseRefusal(token string) RefusalReason {
	clean := strings.ToUpper(strings.TrimSpace(token))
	if clean == "" {
		clean = ReasonUnclassified
	}
	if reason, ok := registeredRefusals[clean]; ok {
		return reason
	}
	return RefusalReason{
		Token:       clean,
		Category:    CategoryUnclassified,
		Description: fmt.Sprintf("unrecognized refusal token %q", clean),
		Refusal:     true,
		Remedy:      "check reason validity using dos man wedge <TOKEN> --explain",
	}
}

// ComputeWitnessSignature generates a deterministic SHA-256 signature for evidence authentication.
func ComputeWitnessSignature(witnessID, decisionSHA, issuer string, issuedUnix int64) string {
	h := sha256.New()
	fmt.Fprintf(h, "dosadapter:v1:%s:%s:%s:%d", witnessID, decisionSHA, issuer, issuedUnix)
	return hex.EncodeToString(h.Sum(nil))
}

// MintWitness creates a tamper-evident DecisionWitness for an arbitration decision.
func MintWitness(witnessID, decisionSHA, issuer string, issuedAt time.Time, validFor time.Duration) DecisionWitness {
	if validFor <= 0 {
		validFor = 10 * time.Minute
	}
	sig := ComputeWitnessSignature(witnessID, decisionSHA, issuer, issuedAt.Unix())
	return DecisionWitness{
		WitnessID:   witnessID,
		DecisionSHA: decisionSHA,
		Issuer:      issuer,
		IssuedAt:    issuedAt,
		Signature:   sig,
		ValidUntil:  issuedAt.Add(validFor),
	}
}

// VerifyWitness validates the cryptographic integrity and freshness of a DecisionWitness.
// Guard: fail-closed verification rejects any witness with missing, malformed, or expired signatures.
func VerifyWitness(w DecisionWitness) error {
	if strings.TrimSpace(w.WitnessID) == "" {
		return fmt.Errorf("%w: missing witness ID", ErrCorruptedEvidence)
	}
	if len(w.DecisionSHA) != 64 {
		return fmt.Errorf("%w: decision SHA must be 64 hex characters", ErrCorruptedEvidence)
	}
	if _, err := hex.DecodeString(w.DecisionSHA); err != nil {
		return fmt.Errorf("%w: decision SHA is not valid hex: %v", ErrCorruptedEvidence, err)
	}
	if strings.TrimSpace(w.Issuer) == "" {
		return fmt.Errorf("%w: missing witness issuer", ErrCorruptedEvidence)
	}
	if w.IssuedAt.IsZero() {
		return fmt.Errorf("%w: zero witness issuance timestamp", ErrCorruptedEvidence)
	}
	if !w.ValidUntil.IsZero() && w.ValidUntil.Before(w.IssuedAt) {
		return fmt.Errorf("%w: valid_until precedes issued_at", ErrCorruptedEvidence)
	}

	expectedSig := ComputeWitnessSignature(w.WitnessID, w.DecisionSHA, w.Issuer, w.IssuedAt.Unix())
	if w.Signature != expectedSig {
		return fmt.Errorf("%w: witness signature mismatch (tampering detected)", ErrCorruptedEvidence)
	}
	return nil
}

// CheckFreshness validates that a DecisionWitness is valid at time at.
func CheckFreshness(w DecisionWitness, at time.Time) error {
	if err := VerifyWitness(w); err != nil {
		return err
	}
	if !w.ValidUntil.IsZero() && at.After(w.ValidUntil) {
		return fmt.Errorf("%w: witness expired at %v (checked at %v)", ErrCorruptedEvidence, w.ValidUntil, at)
	}
	return nil
}

// HandleFallback produces a conservative fail-closed refusal outcome when arbitration errors occur.
// Guard: fail-closed arbitration defaults to refusal whenever evidence, witness, or communication fails.
func HandleFallback(req LeaseRequest, cause error) ArbitrationOutcome {
	now := time.Now()
	decisionStr := fmt.Sprintf("fallback:refuse:%s:%s:%v", req.Lane, req.ID, cause)
	h := sha256.Sum256([]byte(decisionStr))
	decisionSHA := hex.EncodeToString(h[:])

	witness := MintWitness("witness-fallback-"+req.ID, decisionSHA, "fak/dosadapter/fallback", now, 5*time.Minute)
	reason := RefusalReason{
		Token:       ReasonOperatorGate,
		Category:    CategoryOperatorGate,
		Description: fmt.Sprintf("arbitration fallback triggered: %v", cause),
		Refusal:     true,
		Remedy:      "inspect workspace state and retry arbitration under manual supervision",
	}

	return ArbitrationOutcome{
		Outcome:        OutcomeRefuse,
		Lane:           req.Lane,
		LaneKind:       req.LaneKind,
		LockMode:       normalizeLockMode(req.LockMode),
		Tree:           req.Tree,
		Reason:         reason,
		Witness:        witness,
		Interpretation: fmt.Sprintf("STOP: fail-closed arbitration refused lease %q due to fallback", req.Lane),
	}
}

// Arbitrate evaluates a LeaseRequest against active leases with strict fail-closed guarantees.
// Invariant: Exclusive leases must be strictly tree-disjoint across concurrent holders.
// Guard: fail-closed arbitration defaults to refusal whenever evidence, witness, or communication fails.
func (c *AdapterClient) Arbitrate(req LeaseRequest) (ArbitrationOutcome, error) {
	if err := ValidateLease(req); err != nil {
		outcome := HandleFallback(req, err)
		return outcome, fmt.Errorf("%w: %v", ErrInvalidLease, err)
	}

	now := c.now()
	issuer := c.effectiveIssuer()

	c.mu.RLock()
	active := make([]LeaseRequest, 0, len(c.activeLeases))
	for _, l := range c.activeLeases {
		active = append(active, l)
	}
	generation := c.generation
	c.mu.RUnlock()

	outcome, err := c.adjudicate(req, active, now, issuer)
	outcome.LeaseSource = LeaseSourceInMemory
	outcome.LeaseGeneration = generation
	return outcome, err
}

// ArbitrateSet adjudicates req against an explicit LiveLeaseSet, never the
// client's own in-memory table, and stamps the deciding set's provenance on both
// outcomes. A zero now falls back to the client clock.
// Guard: fail-closed — a stale or unreadable set still decides against its rows.
func (c *AdapterClient) ArbitrateSet(req LeaseRequest, set LiveLeaseSet, now time.Time) (ArbitrationOutcome, error) {
	if err := ValidateLease(req); err != nil {
		outcome := HandleFallback(req, err)
		return outcome, fmt.Errorf("%w: %v", ErrInvalidLease, err)
	}

	if now.IsZero() {
		now = c.now()
	}
	issuer := c.effectiveIssuer()

	outcome, err := c.adjudicate(req, set.leases, now, issuer)
	outcome.LeaseSource = set.Source
	outcome.LeaseGeneration = set.Generation
	outcome.LeaseObservedUnix = set.ObservedUnix
	return outcome, err
}

// now resolves the client clock, defaulting to time.Now.
func (c *AdapterClient) now() time.Time {
	if c.clock != nil {
		return c.clock()
	}
	return time.Now()
}

// effectiveIssuer resolves the witness issuer, defaulting to "fak/dosadapter".
func (c *AdapterClient) effectiveIssuer() string {
	issuer := c.issuer
	if strings.TrimSpace(issuer) == "" {
		issuer = "fak/dosadapter"
	}
	return issuer
}

// adjudicate is the shared decision body for Arbitrate and ArbitrateSet: it
// validates+normalizes the request, checks it against the supplied holders, and
// mints the decision witness for both the refuse and acquire verdicts.
func (c *AdapterClient) adjudicate(req LeaseRequest, holders []LeaseRequest, now time.Time, issuer string) (ArbitrationOutcome, error) {
	mode := normalizeLockMode(req.LockMode)
	req.LockMode = mode

	if err := CheckDisjoint(req, holders); err != nil {
		decisionStr := fmt.Sprintf("decision:refuse:%s:%s:%s", req.Lane, req.ID, ReasonCollisionRisk)
		h := sha256.Sum256([]byte(decisionStr))
		decisionSHA := hex.EncodeToString(h[:])

		witness := MintWitness("witness-refuse-"+req.ID, decisionSHA, issuer, now, 15*time.Minute)
		outcome := ArbitrationOutcome{
			Outcome:        OutcomeRefuse,
			Lane:           req.Lane,
			LaneKind:       req.LaneKind,
			LockMode:       mode,
			Tree:           req.Tree,
			Reason:         ParseRefusal(ReasonCollisionRisk),
			Witness:        witness,
			Interpretation: fmt.Sprintf("STOP: lease %q refused due to conflicting active lease", req.Lane),
		}
		return outcome, fmt.Errorf("%w: %v", ErrArbitrationRefused, err)
	}

	decisionStr := fmt.Sprintf("decision:acquire:%s:%s:%s", req.Lane, req.ID, mode)
	h := sha256.Sum256([]byte(decisionStr))
	decisionSHA := hex.EncodeToString(h[:])

	witness := MintWitness("witness-acquire-"+req.ID, decisionSHA, issuer, now, req.TTL)
	outcome := ArbitrationOutcome{
		Outcome:        OutcomeAcquire,
		Lane:           req.Lane,
		LaneKind:       req.LaneKind,
		LockMode:       mode,
		Tree:           req.Tree,
		Witness:        witness,
		Interpretation: fmt.Sprintf("GO: lane %q acquired successfully under %s mode", req.Lane, mode),
	}
	return outcome, nil
}

// RegisterLease atomically registers an acquired lease into the active table if disjoint.
// Invariant: Exclusive leases must be strictly tree-disjoint across concurrent holders.
func (c *AdapterClient) RegisterLease(req LeaseRequest) error {
	if err := ValidateLease(req); err != nil {
		return err
	}
	req.LockMode = normalizeLockMode(req.LockMode)

	c.mu.Lock()
	defer c.mu.Unlock()

	var active []LeaseRequest
	for _, l := range c.activeLeases {
		active = append(active, l)
	}

	if err := CheckDisjoint(req, active); err != nil {
		return err
	}

	if c.activeLeases == nil {
		c.activeLeases = make(map[string]LeaseRequest)
	}
	c.activeLeases[req.ID] = req
	c.generation++
	return nil
}

// ReleaseLease removes a lease by its ID from the active table.
func (c *AdapterClient) ReleaseLease(leaseID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.activeLeases[leaseID]; exists {
		delete(c.activeLeases, leaseID)
		c.generation++
		return true
	}
	return false
}

// ActiveLeases returns a point-in-time snapshot of all currently registered leases.
func (c *AdapterClient) ActiveLeases() []LeaseRequest {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make([]LeaseRequest, 0, len(c.activeLeases))
	for _, l := range c.activeLeases {
		result = append(result, l)
	}
	return result
}

// Validate is an AdapterClient helper method that delegates to ValidateLease.
func (c *AdapterClient) Validate(req LeaseRequest) error {
	return ValidateLease(req)
}

// Verify is an AdapterClient helper method that delegates to VerifyWitness.
func (c *AdapterClient) Verify(w DecisionWitness) error {
	return VerifyWitness(w)
}

// Translate is an AdapterClient helper method that delegates to ParseRefusal.
func (c *AdapterClient) Translate(token string) RefusalReason {
	return ParseRefusal(token)
}
