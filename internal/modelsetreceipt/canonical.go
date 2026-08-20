package modelsetreceipt

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

var (
	digestPattern  = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	stableID       = regexp.MustCompile(`^[a-z][a-z0-9._-]*$`)
	inventoryToken = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
	secretAssign   = regexp.MustCompile(`(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|authorization)\s*[:=]`)
)

func (e Expectation) Validate() error {
	var problems []string
	if e.Schema != ExpectationSchema {
		problems = append(problems, "$.schema must equal "+ExpectationSchema)
	}
	validateDigests(e.Digests, "$.digests", &problems)
	if len(e.Roles) == 0 {
		problems = append(problems, "$.roles must bind at least one selected role")
	}
	seen := map[string]bool{}
	for i, role := range e.Roles {
		path := fmt.Sprintf("$.roles[%d]", i)
		validateRoleBinding(role, path, &problems)
		if seen[role.RoleID] {
			problems = append(problems, path+".role_id is duplicated")
		}
		seen[role.RoleID] = true
	}
	return validationError(problems...)
}

func (e Expectation) CanonicalJSON() ([]byte, error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}
	canonical := canonicalExpectation(e)
	raw, err := json.MarshalIndent(canonical, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

func ParseExpectationJSON(raw []byte) (Expectation, error) {
	if err := checkStrictJSON(raw); err != nil {
		return Expectation{}, err
	}
	var expectation Expectation
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&expectation); err != nil {
		return Expectation{}, validationError("$: expectation is not valid strict JSON")
	}
	if err := expectation.Validate(); err != nil {
		return Expectation{}, err
	}
	return canonicalExpectation(expectation), nil
}

func (r Receipt) Validate() error {
	var problems []string
	if r.Schema != ReceiptSchema {
		problems = append(problems, "$.schema must equal "+ReceiptSchema)
	}
	if _, err := time.Parse(time.RFC3339, r.EvaluatedAt); err != nil {
		problems = append(problems, "$.evaluated_at must be RFC3339")
	}
	validateDigests(r.Expected, "$.expected", &problems)
	validateDigests(r.Observed, "$.observed", &problems)
	if r.Status != StatusCompatible && r.Status != StatusIncompatible {
		problems = append(problems, "$.status is unknown")
	}
	if r.Status == StatusCompatible && len(r.Failures) != 0 {
		problems = append(problems, "$.status cannot be compatible when failures are present")
	}
	if r.Status == StatusIncompatible && len(r.Failures) == 0 {
		problems = append(problems, "$.status cannot be incompatible without failures")
	}
	if len(r.Roles) == 0 {
		problems = append(problems, "$.roles must contain the expected role bindings")
	}
	seenRoles := map[string]bool{}
	globalFailure := hasGlobalFailure(r.Failures)
	for i, role := range r.Roles {
		path := fmt.Sprintf("$.roles[%d]", i)
		if !stableID.MatchString(role.RoleID) {
			problems = append(problems, path+".role_id is invalid")
		}
		if seenRoles[role.RoleID] {
			problems = append(problems, path+".role_id is duplicated")
		}
		seenRoles[role.RoleID] = true
		if role.Status != RoleCompatible && role.Status != RoleIncompatible {
			problems = append(problems, path+".status is unknown")
		}
		wantRoleStatus := RoleCompatible
		if globalFailure || hasRoleFailure(r.Failures, role.RoleID) {
			wantRoleStatus = RoleIncompatible
		}
		if role.Status != wantRoleStatus {
			problems = append(problems, path+".status does not match its failures")
		}
		if role.Expected != nil {
			validateRoleBinding(*role.Expected, path+".expected", &problems)
			if role.Expected.RoleID != role.RoleID {
				problems = append(problems, path+".expected.role_id does not match role_id")
			}
			if role.Expected.Required != role.Required {
				problems = append(problems, path+".expected.required does not match required")
			}
		} else {
			problems = append(problems, path+".expected is required")
		}
		if role.Observed != nil {
			validateRoleBinding(*role.Observed, path+".observed", &problems)
			if role.Observed.RoleID != role.RoleID {
				problems = append(problems, path+".observed.role_id does not match role_id")
			}
			if role.Observed.Required != role.Required {
				problems = append(problems, path+".observed.required does not match required")
			}
		}
		if role.Reevaluated != nil {
			if !stableID.MatchString(role.Reevaluated.AlternativeID) || !inventoryToken.MatchString(role.Reevaluated.CandidateID) ||
				credentialLike(role.Reevaluated.AlternativeID) || credentialLike(role.Reevaluated.CandidateID) {
				problems = append(problems, path+".reevaluated_selection is invalid")
			}
		}
		seenEvidence := map[string]bool{}
		for j, evidence := range role.FactBindings {
			evidencePath := fmt.Sprintf("%s.fact_bindings[%d]", path, j)
			if !inventoryToken.MatchString(evidence.Name) {
				problems = append(problems, evidencePath+".name is invalid")
			}
			if seenEvidence[evidence.Name] {
				problems = append(problems, evidencePath+".name is duplicated")
			}
			seenEvidence[evidence.Name] = true
			if !digestPattern.MatchString(evidence.ValueDigest) {
				problems = append(problems, evidencePath+".value_digest is invalid")
			}
			if len(evidence.Witnesses) == 0 {
				problems = append(problems, evidencePath+".witnesses is empty")
			}
			for k, witness := range evidence.Witnesses {
				witnessPath := fmt.Sprintf("%s.witnesses[%d]", evidencePath, k)
				if (witness.Kind != "probe" && witness.Kind != "operator-attestation") || witness.Source == "" {
					problems = append(problems, witnessPath+" is incomplete")
				}
				if credentialLike(witness.Source) {
					problems = append(problems, witnessPath+".source contains credential material")
				}
				if _, err := time.Parse(time.RFC3339, witness.ObservedAt); err != nil {
					problems = append(problems, witnessPath+".observed_at must be RFC3339")
				}
				if _, err := time.Parse(time.RFC3339, witness.ExpiresAt); err != nil {
					problems = append(problems, witnessPath+".expires_at must be RFC3339")
				}
			}
		}
	}
	for i, failure := range r.Failures {
		path := fmt.Sprintf("$.failures[%d]", i)
		if failure.Code == "" || failure.Field == "" || failure.Remediation == "" {
			problems = append(problems, path+" is incomplete")
		}
		if failure.RoleID != "" && !stableID.MatchString(failure.RoleID) {
			problems = append(problems, path+".role_id is invalid")
		}
		if failure.RoleID != "" && !seenRoles[failure.RoleID] {
			problems = append(problems, path+".role_id is absent from roles")
		}
		for _, value := range []string{string(failure.Code), failure.SourceCode, failure.CandidateID, failure.Field, failure.Expected, failure.Actual, failure.EvidenceSource, failure.Remediation} {
			if credentialLike(value) {
				problems = append(problems, path+" contains credential material")
				break
			}
		}
	}
	return validationError(problems...)
}

func (r Receipt) CanonicalJSON() ([]byte, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	canonical := canonicalReceipt(r)
	raw, err := json.MarshalIndent(canonical, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

// ParseJSON strictly decodes and canonicalizes one independently readable
// startup receipt. Unknown, duplicate, and trailing fields fail closed.
func ParseJSON(raw []byte) (Receipt, error) {
	if err := checkStrictJSON(raw); err != nil {
		return Receipt{}, err
	}
	var receipt Receipt
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&receipt); err != nil {
		return Receipt{}, validationError("$: receipt is not valid strict JSON")
	}
	if err := receipt.Validate(); err != nil {
		return Receipt{}, err
	}
	return canonicalReceipt(receipt), nil
}

func canonicalExpectation(in Expectation) Expectation {
	out := in
	out.Roles = append([]RoleBinding{}, in.Roles...)
	sort.Slice(out.Roles, func(i, j int) bool { return out.Roles[i].RoleID < out.Roles[j].RoleID })
	return out
}

func canonicalReceipt(in Receipt) Receipt {
	out := in
	out.EvaluatedAt = canonicalTime(in.EvaluatedAt)
	out.Roles = append([]RoleReceipt{}, in.Roles...)
	for i := range out.Roles {
		role := &out.Roles[i]
		role.FactBindings = append([]FactBinding{}, role.FactBindings...)
		for j := range role.FactBindings {
			evidence := &role.FactBindings[j]
			evidence.Witnesses = append([]EvidenceRef{}, evidence.Witnesses...)
			sort.Slice(evidence.Witnesses, func(a, b int) bool {
				left, right := evidence.Witnesses[a], evidence.Witnesses[b]
				return strings.Join([]string{left.Kind, left.Source, left.ObservedAt, left.ExpiresAt}, "\x00") <
					strings.Join([]string{right.Kind, right.Source, right.ObservedAt, right.ExpiresAt}, "\x00")
			})
		}
		sort.Slice(role.FactBindings, func(a, b int) bool {
			return role.FactBindings[a].Name < role.FactBindings[b].Name
		})
	}
	sort.Slice(out.Roles, func(i, j int) bool { return out.Roles[i].RoleID < out.Roles[j].RoleID })
	out.Failures = append([]Failure{}, in.Failures...)
	sortFailures(out.Failures)
	return out
}

func validateDigests(d Digests, path string, problems *[]string) {
	for name, value := range map[string]string{"requirements": d.Requirements, "resolution": d.Resolution, "inventory": d.Inventory} {
		if !digestPattern.MatchString(value) {
			*problems = append(*problems, path+"."+name+" is not a sha256 digest")
		}
	}
}

func validateRoleBinding(role RoleBinding, path string, problems *[]string) {
	if !stableID.MatchString(role.RoleID) {
		*problems = append(*problems, path+".role_id is invalid")
	}
	if !stableID.MatchString(role.AlternativeID) || !inventoryToken.MatchString(role.CandidateID) ||
		credentialLike(role.AlternativeID) || credentialLike(role.CandidateID) {
		*problems = append(*problems, path+" selection is invalid")
	}
	for name, value := range map[string]string{
		"identity_digest": role.IdentityDigest, "evidence_digest": role.EvidenceDigest,
		"fact_set_digest": role.FactSetDigest,
	} {
		if !digestPattern.MatchString(value) {
			*problems = append(*problems, path+"."+name+" is not a sha256 digest")
		}
	}
}

func canonicalResolutionJSON(in modelsetresolve.Resolution) ([]byte, error) {
	var problems []string
	if in.Schema != modelsetresolve.Schema {
		problems = append(problems, "resolution.schema is unsupported")
	}
	if _, err := time.Parse(time.RFC3339, in.EvaluatedAt); err != nil {
		problems = append(problems, "resolution.evaluated_at must be RFC3339")
	}
	if len(in.Roles) == 0 {
		problems = append(problems, "resolution.roles is empty")
	}
	seen := map[string]bool{}
	for i, role := range in.Roles {
		path := fmt.Sprintf("resolution.roles[%d]", i)
		if !stableID.MatchString(role.RoleID) {
			problems = append(problems, path+".role_id is invalid")
		}
		if seen[role.RoleID] {
			problems = append(problems, path+".role_id is duplicated")
		}
		seen[role.RoleID] = true
		switch role.Status {
		case modelsetresolve.StatusSelected:
			if role.Selection == nil || role.Selection.AlternativeID == "" || role.Selection.CandidateID == "" {
				problems = append(problems, path+" selected role has no complete selection")
			} else if !stableID.MatchString(role.Selection.AlternativeID) || !inventoryToken.MatchString(role.Selection.CandidateID) ||
				credentialLike(role.Selection.AlternativeID) || credentialLike(role.Selection.CandidateID) {
				problems = append(problems, path+" selected role has an invalid selection")
			}
		case modelsetresolve.StatusRequiredUnresolved:
			if !role.Required || role.Selection != nil {
				problems = append(problems, path+" required-unresolved status is inconsistent")
			}
		case modelsetresolve.StatusOptionalUnresolved:
			if role.Required || role.Selection != nil {
				problems = append(problems, path+" optional-unresolved status is inconsistent")
			}
		default:
			problems = append(problems, path+".status is unknown")
		}
		for _, rejection := range role.Rejections {
			for _, value := range []string{rejection.RoleID, rejection.AlternativeID, rejection.CandidateID, string(rejection.Code), rejection.Constraint, rejection.Expected, rejection.Actual, rejection.EvidenceSource, rejection.Remediation} {
				if credentialLike(value) {
					problems = append(problems, path+".rejections contains credential material")
					break
				}
			}
		}
	}
	if err := validationError(problems...); err != nil {
		return nil, err
	}
	canonical := in
	canonical.EvaluatedAt = canonicalTime(in.EvaluatedAt)
	canonical.Roles = append([]modelsetresolve.RoleResolution(nil), in.Roles...)
	for i := range canonical.Roles {
		role := &canonical.Roles[i]
		if role.Selection != nil {
			selection := *role.Selection
			role.Selection = &selection
		}
		role.Rejections = append([]modelsetresolve.Rejection(nil), role.Rejections...)
		sort.Slice(role.Rejections, func(a, b int) bool {
			return rejectionKey(role.Rejections[a]) < rejectionKey(role.Rejections[b])
		})
	}
	sort.Slice(canonical.Roles, func(i, j int) bool { return canonical.Roles[i].RoleID < canonical.Roles[j].RoleID })
	return json.Marshal(canonical)
}

func rejectionKey(r modelsetresolve.Rejection) string {
	return fmt.Sprintf("%s\x00%08d\x00%s\x00%s\x00%s\x00%s\x00%s\x00%s\x00%s",
		r.RoleID, r.AlternativeIndex, r.AlternativeID, r.CandidateID, r.Code, r.Constraint,
		r.Expected, r.Actual, r.EvidenceSource+"\x00"+r.Remediation)
}

func checkStrictJSON(raw []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := walkJSON(decoder, "$", map[string]bool{}); err != nil {
		return validationError(err.Error())
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return validationError("$: JSON contains trailing data")
	}
	return nil
}

func walkJSON(decoder *json.Decoder, path string, _ map[string]bool) error {
	token, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("%s: invalid JSON", path)
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return fmt.Errorf("%s: invalid object", path)
			}
			key, ok := keyToken.(string)
			if !ok {
				return fmt.Errorf("%s: invalid object key", path)
			}
			child := path + "." + key
			if seen[key] {
				return fmt.Errorf("%s: duplicate field", child)
			}
			seen[key] = true
			if err := walkJSON(decoder, child, nil); err != nil {
				return err
			}
		}
		_, err = decoder.Token()
		return err
	case '[':
		for i := 0; decoder.More(); i++ {
			if err := walkJSON(decoder, fmt.Sprintf("%s[%d]", path, i), nil); err != nil {
				return err
			}
		}
		_, err = decoder.Token()
		return err
	default:
		return fmt.Errorf("%s: invalid delimiter", path)
	}
}

func digest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return fmt.Sprintf("sha256:%x", sum)
}

func canonicalTime(raw string) string {
	parsed, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return raw
	}
	return parsed.UTC().Format(time.RFC3339)
}

func credentialLike(raw string) bool {
	lower := strings.ToLower(strings.TrimSpace(raw))
	for _, marker := range []string{"bearer ", "sk-ant-", "sk-proj-", "ghp_", "github_pat_", "xoxb-", "xoxp-"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	if secretAssign.MatchString(raw) {
		return true
	}
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Scheme == "" {
		return false
	}
	if parsed.User != nil {
		return true
	}
	for key := range parsed.Query() {
		switch strings.ToLower(strings.ReplaceAll(key, "-", "_")) {
		case "token", "api_key", "access_token", "refresh_token", "password", "secret", "authorization":
			return true
		}
	}
	return false
}

func sortFailures(failures []Failure) {
	sort.SliceStable(failures, func(i, j int) bool {
		a, b := failures[i], failures[j]
		return strings.Join([]string{a.RoleID, a.CandidateID, string(a.Code), a.SourceCode, a.Field, a.EvidenceSource, a.Expected, a.Actual, a.Remediation}, "\x00") <
			strings.Join([]string{b.RoleID, b.CandidateID, string(b.Code), b.SourceCode, b.Field, b.EvidenceSource, b.Expected, b.Actual, b.Remediation}, "\x00")
	})
}
