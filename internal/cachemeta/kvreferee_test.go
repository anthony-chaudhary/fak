package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func governedKVEntry() Entry {
	return FromKVPrefix(
		KVPrefix{Tokens: []int{1, 2, 3}, ModelID: "m", Owner: "kvmmu"},
		WithResidency(TierRemote, "l3", "lease-27"),
		WithAdmission(AdmissionQuarantine, "l3-referee"),
		WithDeletionCertificate(DeletionCertificate{Schema: "fak.deletioncert/v1", Subject: "span-27", Digest: "cert-27"}),
		WithLabel("provider", "sglang"),
		WithLabel("engine", "sglang"),
	)
}

func TestKVGovernanceRefereeAdmitsGovernedEviction(t *testing.T) {
	e := governedKVEntry()
	gov := GovernanceFromEntry(e)
	v := DefaultKVGovernanceReferee.AdmitEviction(e.ID, gov)
	if !v.Admitted || v.Reason != KVRefereeAdmitted {
		t.Fatalf("governed eviction not admitted: %+v", v)
	}
	att := DefaultKVGovernanceReferee.AttestEviction(ExternalInvalidateKVSpan, e.ID, KVEvictionScopeExactSpan, true, "", gov)
	if !att.RefereeAdmitted || att.RefereeReason != KVRefereeAdmitted ||
		att.Governance.Security.Taint != abi.TaintTrusted ||
		att.Governance.DeletionCertificate.Digest != "cert-27" {
		t.Fatalf("attestation lost referee/governance fields: %+v", att)
	}
}

func TestKVGovernanceRefereeDeniesMissingDeletionCertificate(t *testing.T) {
	e := FromKVPrefix(
		KVPrefix{Tokens: []int{1, 2, 3}, ModelID: "m", Owner: "kvmmu"},
		WithResidency(TierRemote, "l3", "lease-27"),
		WithAdmission(AdmissionQuarantine, "l3-referee"),
	)
	v := DefaultKVGovernanceReferee.AdmitEviction(e.ID, GovernanceFromEntry(e))
	if v.Admitted || v.Reason != KVRefereeMissingDeletionCertificate {
		t.Fatalf("missing certificate should be refused, got %+v", v)
	}
}

func TestKVGovernanceRefereeDeniesAdvisoryAdmission(t *testing.T) {
	e := governedKVEntry()
	e.Security.AdmissionVerdict = AdmissionDefer
	v := DefaultKVGovernanceReferee.AdmitEviction(e.ID, GovernanceFromEntry(e))
	if v.Admitted || v.Reason != KVRefereeUnfinalAdmission {
		t.Fatalf("advisory admission should not mutate KV tier, got %+v", v)
	}
}
