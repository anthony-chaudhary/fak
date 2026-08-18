package disambiguation

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

const ReasonSourceSelfTestSchemaVersion = "fak-disambiguation-reason-source-self-test/1"

// VocabularyKind names the semantic role of a public uppercase token.
type VocabularyKind string

const (
	VocabularyReason    VocabularyKind = "reason"
	VocabularyVerdict   VocabularyKind = "verdict"
	VocabularyGateClass VocabularyKind = "gate-class"
	VocabularyDecision  VocabularyKind = "decision-kind"
)

var ErrVocabularyCollision = errors.New("incompatible vocabulary collision")

// VocabularyTerm is one public code declaration. CanonicalMeaning is the stable
// identity that allows intentional aliases across package boundaries.
type VocabularyTerm struct {
	Code             string         `json:"code"`
	Kind             VocabularyKind `json:"kind"`
	Package          string         `json:"package"`
	Symbol           string         `json:"symbol"`
	CanonicalMeaning string         `json:"canonical_meaning"`
	SourcePath       string         `json:"source_path"`
}

// ValidateVocabulary rejects an overloaded code unless every declaration names
// the same kind and canonical meaning. Package and symbol may differ: that is a
// declared cross-package alias, not a semantic collision.
func ValidateVocabulary(terms []VocabularyTerm) error {
	byCode := make(map[string][]VocabularyTerm)
	for n, term := range terms {
		if term.Code == "" || term.Kind == "" || term.Package == "" || term.Symbol == "" || term.CanonicalMeaning == "" || term.SourcePath == "" {
			return fmt.Errorf("vocabulary term %d is incomplete", n)
		}
		byCode[term.Code] = append(byCode[term.Code], term)
	}
	codes := make([]string, 0, len(byCode))
	for code := range byCode {
		codes = append(codes, code)
	}
	sort.Strings(codes)
	for _, code := range codes {
		declarations := byCode[code]
		first := declarations[0]
		for _, declaration := range declarations[1:] {
			if declaration.Kind != first.Kind || declaration.CanonicalMeaning != first.CanonicalMeaning {
				return fmt.Errorf("%w: %s means %s/%s in %s but %s/%s in %s", ErrVocabularyCollision, code, first.Kind, first.CanonicalMeaning, first.Package, declaration.Kind, declaration.CanonicalMeaning, declaration.Package)
			}
		}
	}
	return nil
}

// PublicReasonVocabulary returns representative public declarations spanning
// policy/ABI, hooks, DOS, and runtime dispatch surfaces.
func PublicReasonVocabulary() []VocabularyTerm {
	terms := []VocabularyTerm{
		{Code: "POLICY_BLOCK", Kind: VocabularyReason, Package: "internal/abi", Symbol: "ReasonPolicyBlock", CanonicalMeaning: "policy-refusal", SourcePath: "internal/abi/reasons.go"},
		{Code: "DENY", Kind: VocabularyVerdict, Package: "internal/policy", Symbol: "OrgVerdictDeny", CanonicalMeaning: "organization-amendment-denied", SourcePath: "internal/policy/orgprecedence.go"},
		{Code: "LANDS_TREE", Kind: VocabularyGateClass, Package: "internal/hooks", Symbol: "ClassLandsTree", CanonicalMeaning: "hook-mutates-index-or-worktree", SourcePath: "internal/hooks/gatescope.go"},
		{Code: "ARBITER_REFUSE", Kind: VocabularyDecision, Package: "internal/dosdecision", Symbol: "KindArbiterRefuse", CanonicalMeaning: "lane-arbiter-refusal", SourcePath: "internal/dosdecision/revalidate.go"},
		{Code: "COLLISION_RISK", Kind: VocabularyReason, Package: "internal/dispatchorder", Symbol: "ReasonCollisionRisk", CanonicalMeaning: "unsafe-tree-region-overlap", SourcePath: "internal/dispatchorder/dispatchorder.go"},
		{Code: "COLLISION_RISK", Kind: VocabularyReason, Package: "internal/dispatchtick", Symbol: "CollisionRisk", CanonicalMeaning: "unsafe-tree-region-overlap", SourcePath: "internal/dispatchtick/preflight_velocity.go"},
	}
	sort.Slice(terms, func(i, j int) bool {
		if terms[i].Code != terms[j].Code {
			return terms[i].Code < terms[j].Code
		}
		return strings.Compare(terms[i].Package, terms[j].Package) < 0
	})
	return terms
}

// ReasonSourceSelfTestReport captures both sides of the collision contract.
type ReasonSourceSelfTestReport struct {
	Schema                   string           `json:"schema"`
	IndexVersion             string           `json:"index_version"`
	Terms                    []VocabularyTerm `json:"terms"`
	IncompatibleRejected     bool             `json:"incompatible_duplicate_rejected"`
	CrossPackageAliasAllowed bool             `json:"cross_package_alias_allowed"`
}

// RunReasonSourceSelfTest proves the public inventory is valid, an incompatible
// duplicate fails, and the declared COLLISION_RISK alias remains accepted.
func RunReasonSourceSelfTest() (ReasonSourceSelfTestReport, error) {
	terms := PublicReasonVocabulary()
	if err := ValidateVocabulary(terms); err != nil {
		return ReasonSourceSelfTestReport{}, err
	}
	alias := filterVocabularyCode(terms, "COLLISION_RISK")
	if len(alias) != 2 || ValidateVocabulary(alias) != nil {
		return ReasonSourceSelfTestReport{}, errors.New("declared cross-package alias was not accepted")
	}
	incompatible := append([]VocabularyTerm(nil), alias...)
	incompatible[1].Kind = VocabularyVerdict
	incompatible[1].CanonicalMeaning = "request-denied"
	if !errors.Is(ValidateVocabulary(incompatible), ErrVocabularyCollision) {
		return ReasonSourceSelfTestReport{}, errors.New("incompatible duplicate was not rejected")
	}
	return ReasonSourceSelfTestReport{Schema: ReasonSourceSelfTestSchemaVersion, IndexVersion: PublicIndexVersion, Terms: terms, IncompatibleRejected: true, CrossPackageAliasAllowed: true}, nil
}

func filterVocabularyCode(terms []VocabularyTerm, code string) []VocabularyTerm {
	var out []VocabularyTerm
	for _, term := range terms {
		if term.Code == code {
			out = append(out, term)
		}
	}
	return out
}
