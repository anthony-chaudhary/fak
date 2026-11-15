// apf_model.go — the APF (agent-policy-fabric) signed-policy artifact model.
//
// INSPIRE-ONLY port of the signed-policy admission concept from
// NVIDIA/agent-policy-fabric (Apache-2.0, pin
// NVIDIA/agent-policy-fabric@fb961f7cc78e3d5149f0505ca624d064be1866f8):
// a policy document travels as a SIGNED ARTIFACT whose content is bound to a
// policy version; a host-side ADMISSION POINT enforces the artifact at
// admission time, and every decision — including gate waivers — is bound to
// the admitted policy version.
//
// What is borrowed (concepts only, no Rust port):
//   - the signed-policy artifact shape: document + opaque signature, with the
//     document's canonical encoding digested so content and version travel
//     together;
//   - the admission-time enforcement point: the artifact is checked when it is
//     presented for admission, not when it is written;
//   - policy-version binding: the admission point is bound to one policy
//     version, and both admission and gate waivers are refused fail-closed
//     when the presented version does not match the bound version.
//
// What is deliberately NOT claimed: signature verification. The signature is
// carried opaquely and only its PRESENCE is a shape check — this package holds
// no native crypto witness, so it makes no claim that a signature is valid.
// Refusals cite codes from the closed abi refusal vocabulary, never free text.
package policy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// APFVersionSchema tags the artifact schema itself (distinct from the policy
// version carried inside a document).
const APFVersionSchema = "apf-policy/v1"

// APFDocument is the policy document carried inside a signed artifact. Its
// canonical encoding (json.Marshal of the struct, field order fixed by the
// declaration) is what the content digest binds.
type APFDocument struct {
	// Version is the policy version tag this document asserts (e.g. "1.4.0").
	// The admission point binds decisions to it.
	Version string `json:"version"`
	// Issuer names the authority that produced the document.
	Issuer string `json:"issuer"`
	// Rules are the closed-vocabulary rule names the document grants.
	Rules []string `json:"rules"`
}

// APFArtifact is the signed-policy artifact: a document plus an opaque
// signature. The signature is carried, never verified — see the package
// comment for the no-crypto-witness boundary.
type APFArtifact struct {
	Document  APFDocument `json:"document"`
	Signature []byte      `json:"signature"`
}

// APFArtifactDigest returns the hex-encoded SHA-256 of the document's canonical
// encoding. It is the content binding: two artifacts with the same document
// digest carry the same policy content regardless of signature bytes.
func APFArtifactDigest(doc APFDocument) (string, error) {
	b, err := json.Marshal(doc)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

// APFDecision is the outcome of an admission or waiver attempt. Refusals cite a
// closed-vocabulary reason; an admitted decision carries the content digest and
// the policy version the decision is bound to.
type APFDecision struct {
	Admitted         bool
	Reason           abi.ReasonCode
	ArtifactDigest   string
	PolicyVersion    string
}

// APFWaiver is a gate waiver bound to a policy version: it overrides one
// closed-vocabulary refusal, but only while the admission point has admitted a
// policy of exactly that version.
type APFWaiver struct {
	// ID names the waiver (non-empty).
	ID string
	// PolicyVersion is the version the waiver is bound to.
	PolicyVersion string
	// Reason is the closed-vocabulary refusal the waiver overrides. It must be
	// a resolvable core reason other than NONE.
	Reason abi.ReasonCode
	// GrantedBy names the authority that granted the waiver (non-empty).
	GrantedBy string
}

// APFAdmission is the admission-time enforcement point. It is bound to one
// policy version; it records the last admitted artifact and its content digest,
// and it answers admission and waiver attempts against that binding.
type APFAdmission struct {
	mu           sync.RWMutex
	boundVersion string
	admitted     *APFArtifact
	admittedDig  string
}

// NewAPFAdmission builds an admission point bound to boundVersion. An empty
// boundVersion is refused: an admission point with no version binding cannot
// answer version-bound decisions fail-closed.
func NewAPFAdmission(boundVersion string) (*APFAdmission, error) {
	if boundVersion == "" {
		return nil, &APFAdmissionError{Reason: abi.ReasonMalformed, Detail: "bound version is empty"}
	}
	return &APFAdmission{boundVersion: boundVersion}, nil
}

// APFAdmissionError is a structured refusal from the admission point.
type APFAdmissionError struct {
	Reason abi.ReasonCode
	Detail string
}

func (e *APFAdmissionError) Error() string {
	return "apf admission: " + abi.ReasonName(e.Reason) + ": " + e.Detail
}

// BoundVersion returns the policy version this admission point is bound to.
func (a *APFAdmission) BoundVersion() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.boundVersion
}

// AdmittedDigest returns the content digest of the last admitted artifact, or
// "" if nothing has been admitted yet.
func (a *APFAdmission) AdmittedDigest() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.admittedDig
}

// Admit presents a signed-policy artifact at the admission-time enforcement
// point. Checks run in order, fail-closed:
//  1. shape — issuer, version, rules, and signature presence (MALFORMED);
//  2. version binding — the document's version must equal the bound version
//     (POLICY_BLOCK);
//  3. content binding — the canonical digest is computed and recorded.
// A refused admission leaves the admitted state untouched.
func (a *APFAdmission) Admit(art APFArtifact) APFDecision {
	doc := art.Document
	if doc.Issuer == "" || doc.Version == "" || len(doc.Rules) == 0 || len(art.Signature) == 0 {
		return APFDecision{Reason: abi.ReasonMalformed}
	}
	if doc.Version != a.boundVersion {
		return APFDecision{Reason: abi.ReasonPolicyBlock, PolicyVersion: doc.Version}
	}
	dig, err := APFArtifactDigest(doc)
	if err != nil {
		return APFDecision{Reason: abi.ReasonMalformed}
	}
	a.mu.Lock()
	a.admitted = &art
	a.admittedDig = dig
	a.mu.Unlock()
	return APFDecision{Admitted: true, Reason: abi.ReasonNone, ArtifactDigest: dig, PolicyVersion: doc.Version}
}

// Waive presents a gate waiver. A waiver is honored only when it is well-formed
// (non-empty ID/version/granter, a resolvable core reason other than NONE) AND
// bound to the version the admission point has actually admitted — a waiver
// against a version that was never admitted, or against a different version, is
// refused fail-closed with POLICY_BLOCK.
func (a *APFAdmission) Waive(w APFWaiver) APFDecision {
	if w.ID == "" || w.PolicyVersion == "" || w.GrantedBy == "" {
		return APFDecision{Reason: abi.ReasonMalformed}
	}
	if w.Reason == abi.ReasonNone || w.Reason >= abi.ReasonCoreMax {
		return APFDecision{Reason: abi.ReasonMalformed}
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.admitted == nil || a.admitted.Document.Version != w.PolicyVersion {
		return APFDecision{Reason: abi.ReasonPolicyBlock, PolicyVersion: w.PolicyVersion}
	}
	return APFDecision{Admitted: true, Reason: w.Reason, ArtifactDigest: a.admittedDig, PolicyVersion: w.PolicyVersion}
}
