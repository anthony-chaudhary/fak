package modelsetlock

import (
	"encoding/json"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

var (
	stableToken     = regexp.MustCompile(`^[a-z0-9][a-z0-9._+:/-]*$`)
	stableID        = regexp.MustCompile(`^[a-z][a-z0-9._-]*$`)
	digestPattern   = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	revisionPattern = regexp.MustCompile(`^[0-9a-f]{40,64}$`)
)

type lockPayload struct {
	Schema          string       `json:"schema"`
	Inputs          InputDigests `json:"inputs"`
	Target          Target       `json:"target"`
	ResolverVersion string       `json:"resolver_version"`
	EvaluatedAt     string       `json:"evaluated_at"`
	Roles           []Role       `json:"roles"`
}

// CanonicalJSON verifies the lock's content digest and renders the one stable,
// newline-terminated wire representation.
func CanonicalJSON(lock Lock) ([]byte, error) {
	canonical := canonicalizeLock(lock)
	if err := validateLock(canonical); err != nil {
		return nil, err
	}
	raw, err := json.MarshalIndent(canonical, "", "  ")
	if err != nil {
		return nil, lockError(failure(CodeMalformed, "$", "lock cannot be encoded as JSON", "rebuild the lock from typed inputs"))
	}
	return append(raw, '\n'), nil
}

func payloadBytes(lock Lock) []byte {
	payload := lockPayload{
		Schema: lock.Schema, Inputs: lock.Inputs, Target: lock.Target,
		ResolverVersion: lock.ResolverVersion, EvaluatedAt: lock.EvaluatedAt, Roles: lock.Roles,
	}
	raw, _ := json.MarshalIndent(payload, "", "  ")
	return append(raw, '\n')
}

func targetBytes(target Target) []byte {
	raw, _ := json.Marshal(target)
	return raw
}

func canonicalizeLock(lock Lock) Lock {
	out := lock
	out.Schema = strings.TrimSpace(lock.Schema)
	out.ContentDigest = strings.TrimSpace(lock.ContentDigest)
	out.ResolverVersion = strings.TrimSpace(lock.ResolverVersion)
	out.EvaluatedAt = canonicalTime(lock.EvaluatedAt)
	out.Target = canonicalTarget(lock.Target)
	out.Roles = make([]Role, len(lock.Roles))
	for i, role := range lock.Roles {
		out.Roles[i] = Role{
			ID:             strings.TrimSpace(role.ID),
			Required:       role.Required,
			Status:         modelsetresolve.RoleStatus(strings.TrimSpace(string(role.Status))),
			AlternativeIDs: append([]string(nil), role.AlternativeIDs...),
			Rejections:     append([]modelsetresolve.Rejection(nil), role.Rejections...),
		}
		if role.Selected != nil {
			selected := *role.Selected
			selected.AlternativeID = strings.TrimSpace(selected.AlternativeID)
			selected.CandidateID = strings.TrimSpace(selected.CandidateID)
			selected.Identity = canonicalIdentity(selected.Identity)
			selected.Evidence = normalizeEvidence(selected.Evidence)
			out.Roles[i].Selected = &selected
		}
		sortRejections(out.Roles[i].Rejections)
	}
	sort.SliceStable(out.Roles, func(i, j int) bool { return out.Roles[i].ID < out.Roles[j].ID })
	return out
}

func canonicalTarget(target Target) Target {
	return Target{
		OS:           strings.ToLower(strings.TrimSpace(target.OS)),
		Architecture: strings.ToLower(strings.TrimSpace(target.Architecture)),
		Accelerator:  strings.ToLower(strings.TrimSpace(target.Accelerator)),
		Runtime:      strings.ToLower(strings.TrimSpace(target.Runtime)),
	}
}

func canonicalIdentity(identity modelinventory.Identity) modelinventory.Identity {
	identity.Source = modelinventory.SourceKind(strings.ToLower(strings.TrimSpace(string(identity.Source))))
	identity.Provider = strings.ToLower(strings.TrimSpace(identity.Provider))
	identity.Artifact = strings.TrimSpace(identity.Artifact)
	identity.Revision = strings.ToLower(strings.TrimSpace(identity.Revision))
	identity.Digest = strings.ToLower(strings.TrimSpace(identity.Digest))
	identity.Format = strings.ToLower(strings.TrimSpace(identity.Format))
	identity.Witnesses = normalizeEvidence(identity.Witnesses)
	return identity
}

func normalizeEvidence(witnesses []modelinventory.Witness) []modelinventory.Witness {
	out := make([]modelinventory.Witness, len(witnesses))
	for i, witness := range witnesses {
		out[i] = modelinventory.Witness{
			Kind:       modelinventory.WitnessKind(strings.ToLower(strings.TrimSpace(string(witness.Kind)))),
			Source:     strings.TrimSpace(witness.Source),
			ObservedAt: canonicalTime(witness.ObservedAt),
			ExpiresAt:  canonicalTime(witness.ExpiresAt),
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		if out[i].Source != out[j].Source {
			return out[i].Source < out[j].Source
		}
		if out[i].ObservedAt != out[j].ObservedAt {
			return out[i].ObservedAt < out[j].ObservedAt
		}
		return out[i].ExpiresAt < out[j].ExpiresAt
	})
	unique := out[:0]
	for _, witness := range out {
		if len(unique) == 0 || unique[len(unique)-1] != witness {
			unique = append(unique, witness)
		}
	}
	return unique
}

func sortRejections(rejections []modelsetresolve.Rejection) {
	sort.SliceStable(rejections, func(i, j int) bool {
		a, b := rejections[i], rejections[j]
		if a.RoleID != b.RoleID {
			return a.RoleID < b.RoleID
		}
		if a.AlternativeIndex != b.AlternativeIndex {
			return a.AlternativeIndex < b.AlternativeIndex
		}
		if a.CandidateID != b.CandidateID {
			return a.CandidateID < b.CandidateID
		}
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		if a.Constraint != b.Constraint {
			return a.Constraint < b.Constraint
		}
		if a.EvidenceSource != b.EvidenceSource {
			return a.EvidenceSource < b.EvidenceSource
		}
		if a.Expected != b.Expected {
			return a.Expected < b.Expected
		}
		if a.Actual != b.Actual {
			return a.Actual < b.Actual
		}
		return a.Remediation < b.Remediation
	})
}

func validateLock(lock Lock) error {
	var failures []Failure
	if lock.Schema != Schema {
		failures = append(failures, failure(CodeSchemaUnknown, "$.schema", "lock schema is unsupported", "regenerate the lock with schema "+Schema))
	}
	for field, digest := range map[string]string{
		"$.content_digest":   lock.ContentDigest,
		"$.inputs.intent":    lock.Inputs.Intent,
		"$.inputs.inventory": lock.Inputs.Inventory,
		"$.inputs.policy":    lock.Inputs.Policy,
		"$.inputs.target":    lock.Inputs.Target,
	} {
		if !digestPattern.MatchString(digest) {
			failures = append(failures, failure(CodeMalformed, field, "digest is not lowercase sha256", "regenerate the lock from typed inputs"))
		}
	}
	failures = append(failures, targetFailures(lock.Target)...)
	if lock.Inputs.Target != digestBytes(targetBytes(lock.Target)) {
		failures = append(failures, failure(CodeDigestMismatch, "$.inputs.target", "target digest does not bind the serialized target", "regenerate the lock for the current target"))
	}
	if lock.ResolverVersion == "" || strings.TrimSpace(lock.ResolverVersion) != lock.ResolverVersion {
		failures = append(failures, failure(CodeMalformed, "$.resolver_version", "resolver version is empty or not normalized", "regenerate the lock with an exact resolver version"))
	}
	if _, err := time.Parse(time.RFC3339, lock.EvaluatedAt); err != nil {
		failures = append(failures, failure(CodeMalformed, "$.evaluated_at", "evaluation time is not RFC3339", "regenerate the lock from a resolver result"))
	}
	if len(lock.Roles) == 0 {
		failures = append(failures, failure(CodeMalformed, "$.roles", "lock has no role outcomes", "resolve a validated non-empty intent"))
	}
	seenRoles := map[string]bool{}
	for i, role := range lock.Roles {
		path := "$.roles[" + itoa(i) + "]"
		if !stableID.MatchString(role.ID) || seenRoles[role.ID] {
			failures = append(failures, failure(CodeMalformed, path+".id", "role id is invalid or duplicated", "regenerate the lock from a validated intent"))
		}
		seenRoles[role.ID] = true
		if len(role.AlternativeIDs) == 0 {
			failures = append(failures, failure(CodeMalformed, path+".alternative_ids", "role has no ordered alternatives", "regenerate the lock from a validated intent"))
		}
		alternatives := map[string]bool{}
		for j, id := range role.AlternativeIDs {
			if !stableID.MatchString(id) || alternatives[id] {
				failures = append(failures, failure(CodeMalformed, path+".alternative_ids["+itoa(j)+"]", "alternative id is invalid or duplicated", "regenerate the lock from a validated intent"))
			}
			alternatives[id] = true
		}
		switch role.Status {
		case modelsetresolve.StatusSelected:
			if role.Selected == nil {
				failures = append(failures, failure(CodeMalformed, path+".selected", "selected status has no selected identity", "regenerate the lock from the resolver result"))
			}
		case modelsetresolve.StatusOptionalUnresolved:
			if role.Required || role.Selected != nil {
				failures = append(failures, failure(CodeMalformed, path+".status", "optional-unresolved status conflicts with role fields", "regenerate the lock from the resolver result"))
			}
		case modelsetresolve.StatusRequiredUnresolved:
			failures = append(failures, failure(CodeMalformed, path+".status", "a lock cannot contain an unresolved required role", "resolve every required role before writing a lock"))
		default:
			failures = append(failures, failure(CodeMalformed, path+".status", "role status is unknown", "regenerate the lock from the current resolver"))
		}
		if role.Selected != nil {
			if !alternatives[role.Selected.AlternativeID] || !stableID.MatchString(role.Selected.CandidateID) {
				failures = append(failures, failure(CodeMalformed, path+".selected", "selection does not bind a declared alternative and stable candidate", "regenerate the lock from the resolver result"))
			}
			failures = append(failures, identityFailures(role.Selected.Identity, path+".selected.identity")...)
			failures = append(failures, evidenceFailures(role.Selected.Evidence, path+".selected.evidence", true)...)
		}
		for j, rejection := range role.Rejections {
			orderedAlternative := rejection.AlternativeIndex >= 0 && rejection.AlternativeIndex < len(role.AlternativeIDs) &&
				role.AlternativeIDs[rejection.AlternativeIndex] == rejection.AlternativeID
			if rejection.RoleID != role.ID || !orderedAlternative || !stableID.MatchString(rejection.CandidateID) ||
				rejection.Code == "" || rejection.Constraint == "" || rejection.Remediation == "" {
				failures = append(failures, failure(CodeMalformed, path+".rejections["+itoa(j)+"]", "rejection is not bound to this role and a declared alternative", "regenerate the lock from the resolver result"))
			}
		}
	}
	if lock.ContentDigest != digestBytes(payloadBytes(lock)) {
		failures = append(failures, failure(CodeDigestMismatch, "$.content_digest", "content digest does not bind the canonical lock payload", "discard the modified lock and resolve again"))
	}
	return lockError(failures...)
}

func targetFailures(target Target) []Failure {
	var failures []Failure
	for field, value := range map[string]string{
		"target.os":           target.OS,
		"target.architecture": target.Architecture,
		"target.accelerator":  target.Accelerator,
		"target.runtime":      target.Runtime,
	} {
		if !stableToken.MatchString(value) || value != strings.ToLower(strings.TrimSpace(value)) {
			failures = append(failures, failure(CodeInputInvalid, field, "target value is empty or not a normalized token", "declare the exact lowercase platform target"))
		}
	}
	return failures
}

func identityFailures(identity modelinventory.Identity, path string) []Failure {
	var failures []Failure
	if identity.Source != modelinventory.SourceProvider && identity.Source != modelinventory.SourceLocal {
		failures = append(failures, failure(CodeMalformed, path+".source", "identity source is unknown", "regenerate the lock from a normalized inventory"))
	}
	if identity.Artifact == "" || !digestPattern.MatchString(identity.Digest) || !stableToken.MatchString(identity.Format) {
		failures = append(failures, failure(CodeMalformed, path, "immutable artifact identity is incomplete", "regenerate the lock from a normalized inventory"))
	}
	for _, value := range []string{identity.Provider, identity.Artifact, identity.Revision, identity.Digest, identity.Format} {
		if credentialLike(value) {
			failures = append(failures, failure(CodeMalformed, path, "immutable identity contains credential-shaped material", "store only public artifact identity fields"))
		}
	}
	if identity.Source == modelinventory.SourceProvider {
		if identity.Provider == "" || !revisionPattern.MatchString(identity.Revision) {
			failures = append(failures, failure(CodeMalformed, path, "provider identity lacks provider or immutable revision", "regenerate the lock from a normalized provider inventory"))
		}
	} else if identity.Provider != "" || identity.Revision != "" {
		failures = append(failures, failure(CodeMalformed, path, "local identity contains provider-only fields", "regenerate the lock from a normalized local inventory"))
	}
	failures = append(failures, evidenceFailures(identity.Witnesses, path+".witnesses", true)...)
	return failures
}

func evidenceFailures(witnesses []modelinventory.Witness, path string, required bool) []Failure {
	var failures []Failure
	if required && len(witnesses) == 0 {
		failures = append(failures, failure(CodeMalformed, path, "evidence references are empty", "regenerate the lock from a witnessed inventory"))
	}
	for i, witness := range witnesses {
		item := path + "[" + itoa(i) + "]"
		if witness.Kind != modelinventory.EvidenceProbe && witness.Kind != modelinventory.EvidenceOperatorAttestation && witness.Kind != modelinventory.EvidenceArtifactMetadata {
			failures = append(failures, failure(CodeMalformed, item+".kind", "evidence kind is unknown", "regenerate the lock from a normalized inventory"))
		}
		if strings.TrimSpace(witness.Source) == "" || credentialLike(witness.Source) {
			failures = append(failures, failure(CodeMalformed, item+".source", "evidence source is empty or credential-shaped", "store only credential-free evidence references"))
		}
		observed, observedErr := time.Parse(time.RFC3339, witness.ObservedAt)
		expires, expiresErr := time.Parse(time.RFC3339, witness.ExpiresAt)
		if observedErr != nil || expiresErr != nil || !expires.After(observed) {
			failures = append(failures, failure(CodeMalformed, item, "evidence interval is invalid", "regenerate the lock from bounded RFC3339 evidence"))
		}
	}
	return failures
}

func credentialLike(value string) bool {
	lower := strings.ToLower(value)
	for _, marker := range []string{"bearer ", "api_key=", "api-key=", "access_token=", "token=", "password=", "secret="} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func canonicalTime(value string) string {
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(value))
	if err != nil {
		return strings.TrimSpace(value)
	}
	return t.UTC().Format(time.RFC3339)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
