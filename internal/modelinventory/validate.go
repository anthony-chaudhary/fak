package modelinventory

import (
	"net/url"
	"path"
	"regexp"
	"strings"
	"time"
)

var (
	tokenPattern    = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
	providerPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]*$`)
	revisionPattern = regexp.MustCompile(`^[0-9a-f]{40,64}$`)
	digestPattern   = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	formatPattern   = regexp.MustCompile(`^[a-z0-9][a-z0-9._+-]*$`)
	drivePath       = regexp.MustCompile(`^[A-Za-z]:/`)
	secretAssign    = regexp.MustCompile(`(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|authorization)\s*[:=]`)
)

// ValidateAt re-evaluates freshness instead of trusting the inventory's authored
// as-of time. Unknown or expired evidence therefore never becomes a hard fact.
func (in Inventory) ValidateAt(asOf time.Time) Diagnostics {
	canonical := canonicalizeInventory(in)
	var diagnostics Diagnostics
	add := func(code Code, candidate, field, source, detail, remediation string) {
		diagnostics = append(diagnostics, newDiagnostic(code, candidate, field, source, detail, remediation))
	}
	if asOf.IsZero() {
		add(CodeAsOfRequired, "", "as_of", "", "validation time is required", "pass the explicit UTC evaluation time")
		return diagnostics.sorted()
	}
	asOf = asOf.UTC().Truncate(time.Second)
	if canonical.Schema != Schema {
		add(CodeSchemaMismatch, "", "schema", "", "inventory schema is unsupported", "regenerate the inventory with schema "+Schema)
	}
	createdAt, err := time.Parse(time.RFC3339, canonical.AsOf)
	if err != nil {
		add(CodeEvidenceMalformed, "", "as_of", "", "inventory as_of is not RFC3339", "write one explicit RFC3339 UTC timestamp")
	} else if createdAt.After(asOf) {
		add(CodeEvidenceFuture, "", "as_of", "", "inventory as_of postdates the evaluation time", "recreate the inventory from observations available at the evaluation time")
	}
	if len(canonical.Candidates) == 0 {
		add(CodeMissingIdentity, "", "candidates", "", "inventory has no candidates", "supply at least one provider or local candidate observation")
	}
	seenIDs := map[string]bool{}
	seenIdentity := map[string]string{}
	for _, candidate := range canonical.Candidates {
		validateEntry(candidate, asOf, add)
		if seenIDs[candidate.ID] {
			add(CodeDuplicateEntry, candidate.ID, "id", "", "candidate ID appears more than once", "assign one stable unique ID to each candidate")
		}
		seenIDs[candidate.ID] = true
		key := identityKey(candidate.Identity)
		if previous, ok := seenIdentity[key]; ok && previous != candidate.ID {
			add(CodeDuplicateEntry, candidate.ID, "identity", "", "immutable identity is already assigned to another candidate", "keep one candidate ID per immutable artifact identity")
		}
		seenIdentity[key] = candidate.ID
	}
	return diagnostics.sorted()
}

func validateEntry(candidate Candidate, asOf time.Time, add func(Code, string, string, string, string, string)) {
	id := candidate.ID
	if !tokenPattern.MatchString(id) || len(id) > 255 {
		add(CodeMissingIdentity, id, "id", "", "candidate ID is missing, oversized, or is not a stable lowercase token", "set id to a 1-255 byte [a-z0-9._-] token")
	}
	if credentialLike(id) {
		add(CodeCredentialMaterial, id, "id", "", "candidate ID contains credential-shaped material", "replace it with a stable non-secret identifier")
	}
	validateIdentity(id, candidate.Identity, asOf, add)
	validateFact(id, "evidence.availability", candidate.Evidence.Availability, asOf, true, add)
	if candidate.Evidence.Availability.Name != "available" {
		add(CodeMalformedFact, id, "evidence.availability.name", "", "availability fact must be named available", "use the closed availability fact name")
	}
	validateFactGroup(id, "evidence.serving", candidate.Evidence.Serving, asOf, []string{"runtime", "protocol"}, true, add)
	validateFactGroup(id, "evidence.platform", candidate.Evidence.Platform, asOf, []string{"os", "architecture", "accelerator"}, false, add)
	validateFactGroup(id, "evidence.policy", candidate.Evidence.Policy, asOf, []string{"locality", "license"}, false, add)
	validateFactGroup(id, "evidence.capabilities", candidate.Evidence.Capabilities, asOf, nil, true, add)
	if len(candidate.Evidence.Capabilities) == 0 {
		add(CodeMissingFact, id, "evidence.capabilities", "", "candidate has no witnessed capabilities", "record at least one capability from a probe or operator attestation")
	}
	validatePlatformConsistency(id, candidate.Evidence.Platform, add)
}

func validateIdentity(candidate string, identity Identity, asOf time.Time, add func(Code, string, string, string, string, string)) {
	if identity.Source != SourceProvider && identity.Source != SourceLocal {
		add(CodeInvalidSource, candidate, "identity.source", "", "source must be provider or local", "normalize through ProviderObservation or LocalObservation")
	}
	if identity.Artifact == "" || identity.Digest == "" || identity.Format == "" {
		add(CodeMissingIdentity, candidate, "identity", "", "artifact, digest, and format are required", "supply the immutable artifact fields and their provenance")
	}
	if !digestPattern.MatchString(identity.Digest) {
		add(CodeInvalidIdentity, candidate, "identity.digest", "", "digest must be a lowercase sha256 content digest", "record digest as sha256 followed by 64 lowercase hexadecimal characters")
	}
	if !formatPattern.MatchString(identity.Format) {
		add(CodeInvalidIdentity, candidate, "identity.format", "", "format is not a stable token", "record a lowercase artifact format token")
	}
	switch identity.Source {
	case SourceProvider:
		if !providerPattern.MatchString(identity.Provider) || identity.Artifact == "" || !revisionPattern.MatchString(identity.Revision) {
			add(CodeMissingIdentity, candidate, "identity.provider", "", "provider candidates require provider, repository, and immutable revision", "supply a provider token, repository identifier, and pinned 40-64 hex revision")
		}
		if strings.ContainsAny(identity.Artifact, " \t\r\n") || strings.Count(identity.Artifact, "/") < 1 {
			add(CodeInvalidIdentity, candidate, "identity.artifact", "", "provider artifact must be a repository identifier", "use a credential-free owner/repository identifier")
		}
	case SourceLocal:
		if identity.Provider != "" || identity.Revision != "" {
			add(CodeInvalidIdentity, candidate, "identity", "", "local identity cannot claim provider or remote revision fields", "keep local identity to logical artifact, digest, and format")
		}
		clean := path.Clean(strings.ReplaceAll(identity.Artifact, "\\", "/"))
		if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || strings.HasPrefix(clean, "/") || drivePath.MatchString(clean) {
			add(CodeInvalidIdentity, candidate, "identity.artifact", "", "local artifact must be a logical repository-relative path", "replace the host path with a stable relative artifact reference")
		}
	}
	for _, field := range []struct {
		name  string
		value string
	}{
		{"identity.provider", identity.Provider},
		{"identity.artifact", identity.Artifact},
		{"identity.revision", identity.Revision},
		{"identity.digest", identity.Digest},
		{"identity.format", identity.Format},
	} {
		if credentialLike(field.value) {
			add(CodeCredentialMaterial, candidate, field.name, "", "identity contains credential-shaped material", "store only public artifact identity and non-secret references")
		}
	}
	validateEvidence(candidate, "identity.witnesses", identity.Witnesses, asOf, false, add)
}

func validateFactGroup(candidate, field string, facts []Fact, asOf time.Time, required []string, forbidMetadata bool, add func(Code, string, string, string, string, string)) {
	seen := map[string]string{}
	for index, fact := range facts {
		factField := field + "[" + itoa(index) + "]"
		validateFact(candidate, factField, fact, asOf, forbidMetadata, add)
		key := valueKey(fact.Value)
		if previous, ok := seen[fact.Name]; ok && previous != key {
			code := CodeMalformedFact
			if field == "evidence.platform" {
				code = CodePlatformContradiction
			}
			add(code, candidate, field+"."+fact.Name, "", "the same fact has contradictory witnessed values", "remove the stale observation or split candidates whose envelopes differ")
		}
		seen[fact.Name] = key
	}
	for _, name := range required {
		if _, ok := seen[name]; !ok {
			add(CodeMissingFact, candidate, field+"."+name, "", "required evidence fact is unknown", "add a current witness for the required fact")
		}
	}
}

func validateFact(candidate, field string, fact Fact, asOf time.Time, forbidMetadata bool, add func(Code, string, string, string, string, string)) {
	if !tokenPattern.MatchString(fact.Name) {
		add(CodeMalformedFact, candidate, field+".name", "", "fact name is missing or is not a stable lowercase token", "use a non-empty [a-z0-9._-] fact name")
	}
	arms := 0
	if fact.Value.Bool != nil {
		arms++
	}
	if fact.Value.Integer != nil {
		arms++
	}
	if fact.Value.Text != nil {
		arms++
		if strings.TrimSpace(*fact.Value.Text) == "" {
			add(CodeMalformedFact, candidate, field+".value.text", "", "text fact value is empty", "record an explicit non-empty value")
		}
		if credentialLike(*fact.Value.Text) {
			add(CodeCredentialMaterial, candidate, field+".value.text", "", "fact value contains credential-shaped material", "replace it with a non-secret policy or capability value")
		}
	}
	if arms != 1 {
		add(CodeMalformedFact, candidate, field+".value", "", "fact must set exactly one scalar value arm", "set exactly one of bool, integer, or text")
	}
	if credentialLike(fact.Name) {
		add(CodeCredentialMaterial, candidate, field+".name", "", "fact name is credential-shaped", "remove credential material from the public inventory")
	}
	validateEvidence(candidate, field+".witnesses", fact.Witnesses, asOf, forbidMetadata, add)
}

func validateEvidence(candidate, field string, witnesses []Witness, asOf time.Time, forbidMetadata bool, add func(Code, string, string, string, string, string)) {
	if len(witnesses) == 0 {
		add(CodeEvidenceUnknown, candidate, field, "", "fact has no evidence source", "attach a current probe, operator attestation, or artifact metadata witness")
		return
	}
	for index, witness := range witnesses {
		evidenceField := field + "[" + itoa(index) + "]"
		source := witness.Source
		if credentialLike(source) {
			add(CodeCredentialMaterial, candidate, evidenceField+".source", "", "evidence source contains credential-shaped material", "record only a public artifact, probe, or attestation reference")
		}
		if source == "" {
			add(CodeEvidenceUnknown, candidate, evidenceField+".source", "", "evidence source is missing", "name the artifact, probe, or attestation that produced the fact")
		}
		switch witness.Kind {
		case EvidenceProbe, EvidenceOperatorAttestation, EvidenceArtifactMetadata:
		default:
			add(CodeEvidenceMalformed, candidate, evidenceField+".kind", source, "witness kind is unsupported", "use probe, operator-attestation, or artifact-metadata")
		}
		if forbidMetadata && witness.Kind == EvidenceArtifactMetadata {
			add(CodeEvidenceScopeMismatch, candidate, evidenceField+".kind", source, "artifact metadata cannot witness serving behavior or capabilities", "replace it with a FAK-owned probe or explicit operator attestation")
		}
		observed, observedErr := time.Parse(time.RFC3339, witness.ObservedAt)
		expires, expiresErr := time.Parse(time.RFC3339, witness.ExpiresAt)
		if observedErr != nil || expiresErr != nil {
			add(CodeEvidenceMalformed, candidate, evidenceField, source, "observed_at and expires_at must be RFC3339 timestamps", "record a bounded UTC evidence interval")
			continue
		}
		if observed.After(asOf) {
			add(CodeEvidenceFuture, candidate, evidenceField, source, "evidence observation postdates evaluation", "re-evaluate at or after the observation time")
		}
		if !expires.After(observed) {
			add(CodeEvidenceMalformed, candidate, evidenceField, source, "evidence expiry is not after observation", "set expires_at later than observed_at")
		} else if !expires.After(asOf) {
			add(CodeEvidenceStale, candidate, evidenceField, source, "evidence is expired at evaluation time", "refresh the named witness before normalization")
		}
	}
}

func validatePlatformConsistency(candidate string, facts []Fact, add func(Code, string, string, string, string, string)) {
	var accelerator string
	var memory int64
	for _, fact := range facts {
		switch fact.Name {
		case "accelerator":
			if fact.Value.Text != nil {
				accelerator = strings.ToLower(*fact.Value.Text)
			}
		case "accelerator_memory_bytes":
			if fact.Value.Integer != nil {
				memory = *fact.Value.Integer
			}
		}
	}
	if (accelerator == "cpu" || accelerator == "none") && memory > 0 {
		add(CodePlatformContradiction, candidate, "evidence.platform", "", "CPU-only platform declares positive accelerator memory", "set accelerator memory to zero or record the actual accelerator")
	}
}

func credentialLike(raw string) bool {
	text := strings.TrimSpace(raw)
	lower := strings.ToLower(text)
	if text == "" {
		return false
	}
	for _, marker := range []string{"bearer ", "sk-ant-", "sk-proj-", "ghp_", "github_pat_", "xoxb-", "xoxp-"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	if secretAssign.MatchString(text) {
		return true
	}
	parsed, err := url.Parse(text)
	if err == nil && parsed.Scheme != "" {
		if parsed.User != nil {
			return true
		}
		for key := range parsed.Query() {
			switch strings.ToLower(strings.ReplaceAll(key, "-", "_")) {
			case "token", "api_key", "access_token", "refresh_token", "password", "secret", "authorization":
				return true
			}
		}
	}
	return false
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
