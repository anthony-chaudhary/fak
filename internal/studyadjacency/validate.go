package studyadjacency

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

var (
	fullRevisionPattern = regexp.MustCompile(`^[0-9a-fA-F]{40}$`)
	sha256Pattern       = regexp.MustCompile(`^(?:sha256:)?[0-9a-fA-F]{64}$`)
	ownerPattern        = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$`)
	repoPattern         = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)
)

// Validate enforces the finite-set, pinning, receipt, and candidate-linkage
// invariants of a study adjacency manifest.
func Validate(manifest Manifest) error {
	var problems validationProblems
	if manifest.Schema != Schema {
		problems.add("schema must be %q", Schema)
	}
	requireText(&problems, "id", manifest.ID)
	requireText(&problems, "title", manifest.Title)
	validateScope(&problems, manifest.Scope)
	validateCorpusWitness(&problems, manifest.Anchor)

	declared := make(map[string]Repository, len(manifest.DeclaredRepositories))
	for i, repository := range manifest.DeclaredRepositories {
		path := fmt.Sprintf("declared_repositories[%d]", i)
		validateRepository(&problems, path, repository)
		key := repositoryKey(repository)
		if key == "/" {
			continue
		}
		if _, exists := declared[key]; exists {
			problems.add("%s duplicates declared repository %q", path, repository.String())
			continue
		}
		declared[key] = repository
	}
	if len(manifest.DeclaredRepositories) == 0 {
		problems.add("declared_repositories must not be empty")
	}
	for _, required := range RequiredCoreRepositories {
		if _, ok := declared[repositoryKey(required)]; !ok {
			problems.add("required core repository %q is not declared", required.String())
		}
	}

	processed := make(map[string]bool, len(manifest.Members))
	candidateOwners := make(map[string]string)
	for i, member := range manifest.Members {
		path := fmt.Sprintf("members[%d]", i)
		validateMember(&problems, path, member, declared, candidateOwners)
		key := repositoryKey(member.Repository)
		if key == "/" {
			continue
		}
		if processed[key] {
			problems.add("%s duplicates member repository %q", path, member.Repository.String())
		}
		processed[key] = true
		if _, ok := declared[key]; !ok {
			problems.add("%s repository %q is not declared", path, member.Repository.String())
		}
		if manifest.Anchor.Pin.Cutoff != "" && member.Pin.Cutoff != "" && member.Pin.Cutoff != manifest.Anchor.Pin.Cutoff {
			problems.add("%s pin.cutoff must match anchor cutoff %q", path, manifest.Anchor.Pin.Cutoff)
		}
	}
	if len(manifest.Members) == 0 {
		problems.add("members must not be empty")
	}

	declaredKeys := make([]string, 0, len(declared))
	for key := range declared {
		declaredKeys = append(declaredKeys, key)
	}
	sort.Strings(declaredKeys)
	for _, key := range declaredKeys {
		if !processed[key] {
			problems.add("declared repository %q was not processed", declared[key].String())
		}
	}
	return problems.err()
}

func validateScope(problems *validationProblems, scope Scope) {
	requireText(problems, "scope.bounded_meaning", scope.BoundedMeaning)
	validateTextList(problems, "scope.inclusion_criteria", scope.InclusionCriteria)
	validateTextList(problems, "scope.exclusion_criteria", scope.ExclusionCriteria)
}

func validateTextList(problems *validationProblems, path string, values []string) {
	if len(values) == 0 {
		problems.add("%s must not be empty", path)
		return
	}
	seen := make(map[string]bool, len(values))
	for i, value := range values {
		itemPath := fmt.Sprintf("%s[%d]", path, i)
		requireText(problems, itemPath, value)
		key := strings.ToLower(strings.TrimSpace(value))
		if key != "" && seen[key] {
			problems.add("%s is duplicated", itemPath)
		}
		seen[key] = true
	}
}

func validateCorpusWitness(problems *validationProblems, witness CorpusWitness) {
	validateRepository(problems, "anchor.repository", witness.Repository)
	validatePin(problems, "anchor.pin", witness.Pin)
	if witness.NormalizedRecords <= 0 {
		problems.add("anchor.normalized_records must be positive")
	}
	if !sha256Pattern.MatchString(strings.TrimSpace(witness.SHA256)) {
		problems.add("anchor.sha256 must be a SHA-256 digest")
	}
}

func validateMember(problems *validationProblems, path string, member Member, declared map[string]Repository, candidateOwners map[string]string) {
	requireText(problems, path+".name", member.Name)
	validateRepository(problems, path+".repository", member.Repository)
	validatePin(problems, path+".pin", member.Pin)
	if !member.Processed {
		problems.add("%s.processed must be true", path)
	}
	requireText(problems, path+".inclusion_rationale", member.InclusionRationale)
	requireText(problems, path+".decision_relation", member.DecisionRelation)
	requireText(problems, path+".freshness_notes", member.FreshnessNotes)
	requireText(problems, path+".partial_notes", member.PartialNotes)

	if len(member.SourceClassReceipts) == 0 {
		problems.add("%s.source_class_receipts must not be empty", path)
	}
	classes := make(map[string]bool, len(member.SourceClassReceipts))
	for i, receipt := range member.SourceClassReceipts {
		receiptPath := fmt.Sprintf("%s.source_class_receipts[%d]", path, i)
		validateSourceClassReceipt(problems, receiptPath, receipt)
		key := strings.ToLower(strings.TrimSpace(receipt.Class))
		if key != "" && classes[key] {
			problems.add("%s duplicates source class %q", receiptPath, receipt.Class)
		}
		classes[key] = true
	}

	ownerKey := repositoryKey(member.Repository)
	for i, candidate := range member.Candidates {
		candidatePath := fmt.Sprintf("%s.candidates[%d]", path, i)
		validateCandidate(problems, candidatePath, candidate, ownerKey, declared)
		idKey := strings.ToLower(strings.TrimSpace(candidate.ID))
		if idKey == "" {
			continue
		}
		if prior, exists := candidateOwners[idKey]; exists {
			problems.add("%s.id %q duplicates candidate from %s", candidatePath, candidate.ID, prior)
			continue
		}
		candidateOwners[idKey] = member.Repository.String()
	}
}

func validateSourceClassReceipt(problems *validationProblems, path string, receipt SourceClassReceipt) {
	requireText(problems, path+".class", receipt.Class)
	requireText(problems, path+".notes", receipt.Notes)
	switch receipt.Status {
	case ReceiptComplete:
		if strings.TrimSpace(receipt.TerminalReceipt) == "" {
			problems.add("%s.terminal_receipt is required when status is complete", path)
		}
	case ReceiptPartial, ReceiptMissing, ReceiptUnavailable:
	case "":
		problems.add("%s.status is required", path)
	default:
		problems.add("%s.status %q is unsupported", path, receipt.Status)
	}
}

func validateCandidate(problems *validationProblems, path string, candidate Candidate, ownerKey string, declared map[string]Repository) {
	requireText(problems, path+".id", candidate.ID)
	requireText(problems, path+".title", candidate.Title)
	requireText(problems, path+".rationale", candidate.Rationale)
	if strings.TrimSpace(candidate.VLLMMechanismLink) == "" && strings.TrimSpace(candidate.FrontierChangingContrast) == "" {
		problems.add("%s requires a vllm_mechanism_link or frontier_changing_contrast", path)
	}
	if len(candidate.RepositoryLinks) == 0 {
		problems.add("%s.repository_links must not be empty", path)
	}
	linkedOwner := false
	seen := make(map[string]bool, len(candidate.RepositoryLinks))
	for i, repository := range candidate.RepositoryLinks {
		linkPath := fmt.Sprintf("%s.repository_links[%d]", path, i)
		validateRepository(problems, linkPath, repository)
		key := repositoryKey(repository)
		if key == ownerKey {
			linkedOwner = true
		}
		if seen[key] {
			problems.add("%s duplicates repository link %q", linkPath, repository.String())
		}
		seen[key] = true
		if _, ok := declared[key]; !ok {
			problems.add("%s repository %q is not declared", linkPath, repository.String())
		}
	}
	if len(candidate.RepositoryLinks) > 0 && !linkedOwner {
		problems.add("%s.repository_links must include owning member %q", path, ownerKey)
	}
}

func validatePin(problems *validationProblems, path string, pin Pin) {
	if !fullRevisionPattern.MatchString(strings.TrimSpace(pin.Revision)) {
		problems.add("%s.revision must be a full 40-hex commit", path)
	}
	cutoff, cutoffOK := parseTimestamp(problems, path+".cutoff", pin.Cutoff)
	observed, observedOK := parseTimestamp(problems, path+".observed_at", pin.ObservedAt)
	if cutoffOK && observedOK && observed.Before(cutoff) {
		problems.add("%s.observed_at must not precede cutoff", path)
	}
}

func parseTimestamp(problems *validationProblems, path, value string) (time.Time, bool) {
	if strings.TrimSpace(value) == "" {
		problems.add("%s is required", path)
		return time.Time{}, false
	}
	parsed, err := time.Parse(time.RFC3339, value)
	if err != nil {
		problems.add("%s must be RFC3339", path)
		return time.Time{}, false
	}
	return parsed, true
}

func validateRepository(problems *validationProblems, path string, repository Repository) {
	owner := strings.TrimSpace(repository.Owner)
	repo := strings.TrimSpace(repository.Repo)
	if owner == "" {
		problems.add("%s.owner is required", path)
	} else if owner != repository.Owner || !ownerPattern.MatchString(owner) {
		problems.add("%s.owner must be a canonical GitHub owner", path)
	}
	if repo == "" {
		problems.add("%s.repo is required", path)
	} else if repo != repository.Repo || !repoPattern.MatchString(repo) || strings.Contains(repo, "/") {
		problems.add("%s.repo must be a canonical GitHub repository name", path)
	}
}

func repositoryKey(repository Repository) string {
	return strings.ToLower(strings.TrimSpace(repository.Owner)) + "/" + strings.ToLower(strings.TrimSpace(repository.Repo))
}

func requireText(problems *validationProblems, path, value string) {
	if strings.TrimSpace(value) == "" {
		problems.add("%s is required", path)
	}
}

type validationProblems []string

func (p *validationProblems) add(format string, args ...any) {
	*p = append(*p, fmt.Sprintf(format, args...))
}

func (p validationProblems) err() error {
	if len(p) == 0 {
		return nil
	}
	return fmt.Errorf("invalid study adjacency manifest: %s", strings.Join(p, "; "))
}
