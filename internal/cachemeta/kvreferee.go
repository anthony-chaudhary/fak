package cachemeta

// KVRefereeReason is the closed local vocabulary for the shared K/V governance
// referee. It explains why a named K/V eviction was admitted or refused before a
// native or ridden KV tier mutates residency.
type KVRefereeReason string

const (
	KVRefereeAdmitted                   KVRefereeReason = "admitted"
	KVRefereeMissingTarget              KVRefereeReason = "missing_target"
	KVRefereeMissingAdmission           KVRefereeReason = "missing_admission"
	KVRefereeUnfinalAdmission           KVRefereeReason = "unfinal_admission"
	KVRefereeMissingAdmitter            KVRefereeReason = "missing_admitter"
	KVRefereeMissingLease               KVRefereeReason = "missing_lease"
	KVRefereeMissingDeletionCertificate KVRefereeReason = "missing_deletion_certificate"
)

// KVRefereeDecision is the control-path verdict from the shared K/V governance
// referee. A denied decision means the caller must not mutate the KV tier.
type KVRefereeDecision struct {
	Admitted bool
	Reason   KVRefereeReason
}

// KVGovernanceReferee is the field-only referee reused by Track A engine-cache
// lowering and Track B native paged-KV lowering. It validates the existing
// governance vocabulary; it does not invent a second policy model.
type KVGovernanceReferee struct{}

// DefaultKVGovernanceReferee is the single default referee used by the issue-#27
// lowering paths.
var DefaultKVGovernanceReferee KVGovernanceReferee

// AdmitEviction gates a named K/V eviction before any tier mutation. The
// eviction must name a target and carry a final admission descriptor, the lease
// that owns residency, and the deletion certificate receipt the eviction will
// propagate.
func (KVGovernanceReferee) AdmitEviction(target EntryID, gov KVGovernance) KVRefereeDecision {
	if !target.Valid() {
		return KVRefereeDecision{Reason: KVRefereeMissingTarget}
	}
	switch gov.Security.AdmissionVerdict {
	case "":
		return KVRefereeDecision{Reason: KVRefereeMissingAdmission}
	case AdmissionDeny, AdmissionDefer, AdmissionRequireWitness:
		return KVRefereeDecision{Reason: KVRefereeUnfinalAdmission}
	}
	if gov.Security.AdmittedBy == "" {
		return KVRefereeDecision{Reason: KVRefereeMissingAdmitter}
	}
	if gov.Lease == "" {
		return KVRefereeDecision{Reason: KVRefereeMissingLease}
	}
	if !gov.DeletionCertificate.Valid() {
		return KVRefereeDecision{Reason: KVRefereeMissingDeletionCertificate}
	}
	return KVRefereeDecision{Admitted: true, Reason: KVRefereeAdmitted}
}

// AdmitInvalidations gates the named K/V invalidation directives that will lower
// to either a remote engine reset/exact-span call or a native paged-KV mutation.
// Identity-less coarse directives are skipped here; exact-span callers still
// fail closed separately when they require a named span.
func (r KVGovernanceReferee) AdmitInvalidations(dirs []ExternalInvalidationDirective) KVRefereeDecision {
	for _, d := range dirs {
		if !d.Entry.Valid() {
			continue
		}
		if d.Kind != ExternalInvalidateKVSpan && d.Kind != ExternalInvalidateAttentionIndex {
			continue
		}
		if v := r.AdmitEviction(d.Entry, d.Governance); !v.Admitted {
			return v
		}
	}
	return KVRefereeDecision{Admitted: true, Reason: KVRefereeAdmitted}
}

// AttestEviction returns the common payload-free receipt after the referee has
// evaluated the supplied governance descriptor. Callers may surface denied
// attestations, but they must not mutate the KV tier unless RefereeAdmitted is true.
func (r KVGovernanceReferee) AttestEviction(kind ExternalInvalidationKind, target EntryID, scope KVEvictionScope, exactSpanSupported bool, degradeReason string, gov KVGovernance) KVEvictionAttestation {
	v := r.AdmitEviction(target, gov)
	return KVEvictionAttestation{
		Kind:               kind,
		Target:             target,
		Scope:              scope,
		ExactSpanSupported: exactSpanSupported,
		Degraded:           degradeReason != "",
		DegradeReason:      degradeReason,
		Governance:         gov,
		RefereeAdmitted:    v.Admitted,
		RefereeReason:      v.Reason,
	}
}
