package landingvoucher

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"time"
)

// VoucherSchemaV1 identifies the landing voucher format version 1.
const VoucherSchemaV1 = "fak-landing-voucher/v1"

// ReasonLandingInvariantRefused is the closed refusal token emitted when one
// or more pre-landing invariants are violated.
const ReasonLandingInvariantRefused = "LANDING_INVARIANT_REFUSED"

// ErrLandingInvariantRefused is the sentinel error wrapping pre-landing invariant refusals.
var ErrLandingInvariantRefused = errors.New(ReasonLandingInvariantRefused)

// InvariantChecklist records the pass/fail judgment of all 8 pre-landing invariant axes.
type InvariantChecklist struct {
	SpatialDisjointness bool `json:"spatial_disjointness"`
	BaseFreshness       bool `json:"base_freshness"`
	ABIStability        bool `json:"abi_stability"`
	CleanPathspec       bool `json:"clean_pathspec"`
	WitnessProof        bool `json:"witness_proof"`
	ZeroLockContention  bool `json:"zero_lock_contention"`
	WriterLeaseValidity bool `json:"writer_lease_validity"`
	BoundaryAdmission   bool `json:"boundary_admission"`
}

// AllPassed reports whether all 8 landing gap invariants were satisfied.
func (c InvariantChecklist) AllPassed() bool {
	return c.SpatialDisjointness &&
		c.BaseFreshness &&
		c.ABIStability &&
		c.CleanPathspec &&
		c.WitnessProof &&
		c.ZeroLockContention &&
		c.WriterLeaseValidity &&
		c.BoundaryAdmission
}

// LandingVoucher is the tamper-evident proof token attesting that all 8 landing
// invariants have been audited and passed prior to mutating git refs.
type LandingVoucher struct {
	Schema         string             `json:"schema"`
	Digest         string             `json:"digest"`
	Timestamp      time.Time          `json:"timestamp"`
	CandidateSHA   string             `json:"candidate_sha"`
	BaseSHA        string             `json:"base_sha"`
	Lane           string             `json:"lane"`
	LeasedGlobs    []string           `json:"leased_globs"`
	ChangedFiles   []string           `json:"changed_files"`
	WitnessReceipt string             `json:"witness_receipt"`
	Invariants     InvariantChecklist `json:"invariants"`
}

// ComputeDigest returns a deterministic SHA-256 hex digest over the voucher's fields.
// The Digest field is excluded from this computation to maintain idempotency.
func (v LandingVoucher) ComputeDigest() string {
	type canonicalVoucher struct {
		Schema         string             `json:"schema"`
		Timestamp      string             `json:"timestamp"`
		CandidateSHA   string             `json:"candidate_sha"`
		BaseSHA        string             `json:"base_sha"`
		Lane           string             `json:"lane"`
		LeasedGlobs    []string           `json:"leased_globs"`
		ChangedFiles   []string           `json:"changed_files"`
		WitnessReceipt string             `json:"witness_receipt"`
		Invariants     InvariantChecklist `json:"invariants"`
	}

	leased := v.LeasedGlobs
	if leased == nil {
		leased = []string{}
	}
	changed := v.ChangedFiles
	if changed == nil {
		changed = []string{}
	}

	payload := canonicalVoucher{
		Schema:         v.Schema,
		Timestamp:      v.Timestamp.UTC().Format(time.RFC3339Nano),
		CandidateSHA:   v.CandidateSHA,
		BaseSHA:        v.BaseSHA,
		Lane:           v.Lane,
		LeasedGlobs:    leased,
		ChangedFiles:   changed,
		WitnessReceipt: v.WitnessReceipt,
		Invariants:     v.Invariants,
	}

	b, err := json.Marshal(payload)
	if err != nil {
		return ""
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
