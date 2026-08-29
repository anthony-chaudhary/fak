package harnesskit

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const (
	CompatibilityReportSchema = "fak.harnesskit-compatibility-report/v1"
	ContractDiffSchema        = "fak.harnesskit-contract-diff/v1"
	UpgradePlanSchema         = "fak.harnesskit-upgrade-plan/v1"
)

// CapabilityStatus makes the maturity of an offered capability explicit.
// A version string is provenance; only negotiation against this status and a
// declared revision range is compatibility evidence.
type CapabilityStatus string

const (
	StatusStable       CapabilityStatus = "stable"
	StatusExperimental CapabilityStatus = "experimental"
	StatusDeprecated   CapabilityStatus = "deprecated"
)

// CapabilityRequirement is a builder's stable, host-independent requirement.
// Status is optional; when set, an offer must match it exactly.
type CapabilityRequirement struct {
	Name        Capability       `json:"name"`
	MinRevision int              `json:"min_revision"`
	MaxRevision int              `json:"max_revision"`
	Optional    bool             `json:"optional,omitempty"`
	Status      CapabilityStatus `json:"status,omitempty"`
}

// Deprecation describes the validated migration path for a deprecated offer.
// RemovalHorizon is an operator-readable release or date, not an inferred timer.
type Deprecation struct {
	Replacement    Capability `json:"replacement"`
	RemovalHorizon string     `json:"removal_horizon"`
}

// CapabilityOffer is one explicit revision implemented by a host runtime.
type CapabilityOffer struct {
	Name        Capability       `json:"name"`
	Revision    int              `json:"revision"`
	Status      CapabilityStatus `json:"status"`
	Semantics   string           `json:"semantics,omitempty"`
	Deprecation *Deprecation     `json:"deprecation,omitempty"`
}

// BuilderContract is the stable compatibility declaration owned by a product.
type BuilderContract struct {
	ContractVersion string                  `json:"contract_version"`
	Requirements    []CapabilityRequirement `json:"requirements,omitempty"`
}

// RuntimeContract is the explicit compatibility surface implemented by a host.
type RuntimeContract struct {
	ContractVersion string            `json:"contract_version"`
	Capabilities    []CapabilityOffer `json:"capabilities,omitempty"`
}

// CompatibilityReason is a stable machine reason. New reasons may be added;
// readers must treat unknown reasons as incompatible.
type CompatibilityReason string

const (
	ReasonCompatible           CompatibilityReason = "compatible"
	ReasonContractMismatch     CompatibilityReason = "contract_mismatch"
	ReasonCapabilityAbsent     CompatibilityReason = "capability_absent"
	ReasonRevisionBelowMin     CompatibilityReason = "revision_below_minimum"
	ReasonRevisionAboveMax     CompatibilityReason = "revision_above_maximum"
	ReasonStatusMismatch       CompatibilityReason = "status_mismatch"
	ReasonInvalidRequirement   CompatibilityReason = "invalid_requirement"
	ReasonInvalidOffer         CompatibilityReason = "invalid_offer"
	ReasonDuplicateRequirement CompatibilityReason = "duplicate_requirement"
	ReasonDuplicateOffer       CompatibilityReason = "duplicate_offer"
	ReasonInvalidDeprecation   CompatibilityReason = "invalid_deprecation"
)

// CapabilityOutcome records one required or optional negotiation result.
type CapabilityOutcome struct {
	Name            Capability          `json:"name"`
	Required        bool                `json:"required"`
	Compatible      bool                `json:"compatible"`
	Reason          CompatibilityReason `json:"reason"`
	MinRevision     int                 `json:"min_revision"`
	MaxRevision     int                 `json:"max_revision"`
	RequiredStatus  CapabilityStatus    `json:"required_status,omitempty"`
	OfferedRevision int                 `json:"offered_revision,omitempty"`
	OfferedStatus   CapabilityStatus    `json:"offered_status,omitempty"`
	Detail          string              `json:"detail"`
}

// CompatibilityReport is the canonical deterministic negotiation record.
type CompatibilityReport struct {
	SchemaVersion   string               `json:"schema_version"`
	Compatible      bool                 `json:"compatible"`
	ContractVersion string               `json:"contract_version"`
	HostContract    string               `json:"host_contract"`
	ContractReason  CompatibilityReason  `json:"contract_reason"`
	Outcomes        []CapabilityOutcome  `json:"outcomes"`
	Issues          []CompatibilityIssue `json:"issues,omitempty"`
}

// CompatibilityIssue records malformed canonical input independently of a
// builder's required/optional outcome.
type CompatibilityIssue struct {
	Scope  string              `json:"scope"`
	Name   Capability          `json:"name,omitempty"`
	Reason CompatibilityReason `json:"reason"`
	Detail string              `json:"detail"`
}

// JSON emits the stable machine form. Writers always emit the current schema;
// readers should ignore additive fields but refuse unknown schema versions.
func (r CompatibilityReport) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

// Error explains a refusal without requiring callers to parse diagnostic text.
func (r CompatibilityReport) Error() string {
	if r.Compatible {
		return "harnesskit compatibility: compatible"
	}
	parts := make([]string, 0, len(r.Outcomes)+1)
	if r.ContractReason != ReasonCompatible {
		parts = append(parts, fmt.Sprintf("contract %q vs host %q: %s", r.ContractVersion, r.HostContract, r.ContractReason))
	}
	for _, outcome := range r.Outcomes {
		if !outcome.Compatible && outcome.Required {
			parts = append(parts, fmt.Sprintf("%s: %s (%s)", outcome.Name, outcome.Reason, outcome.Detail))
		}
	}
	for _, issue := range r.Issues {
		parts = append(parts, fmt.Sprintf("%s %s: %s (%s)", issue.Scope, issue.Name, issue.Reason, issue.Detail))
	}
	if len(parts) == 0 {
		parts = append(parts, "unknown incompatibility")
	}
	return "harnesskit compatibility refused: " + strings.Join(parts, "; ")
}

// NegotiateCompatibility compares declarations without I/O or global state.
// Missing and malformed evidence is never upgraded to compatibility.
func NegotiateCompatibility(builder BuilderContract, host RuntimeContract) CompatibilityReport {
	report := CompatibilityReport{
		SchemaVersion:   CompatibilityReportSchema,
		Compatible:      true,
		ContractVersion: builder.ContractVersion,
		HostContract:    host.ContractVersion,
		ContractReason:  ReasonCompatible,
		Outcomes:        []CapabilityOutcome{},
	}
	if builder.ContractVersion == "" || host.ContractVersion == "" || builder.ContractVersion != host.ContractVersion {
		report.Compatible = false
		report.ContractReason = ReasonContractMismatch
	}

	offers, offerProblems := indexOffers(host.Capabilities)
	requirementCounts := make(map[Capability]int)
	for _, requirement := range builder.Requirements {
		requirementCounts[requirement.Name]++
	}
	for _, requirement := range builder.Requirements {
		outcome := CapabilityOutcome{
			Name: requirement.Name, Required: !requirement.Optional,
			MinRevision: requirement.MinRevision, MaxRevision: requirement.MaxRevision,
			RequiredStatus: requirement.Status,
		}
		switch {
		case requirement.Name == "" || requirement.MinRevision < 1 || requirement.MaxRevision < requirement.MinRevision || !validOptionalStatus(requirement.Status):
			outcome.Reason, outcome.Detail = ReasonInvalidRequirement, "name, positive ordered revisions, and a known status are required"
		case requirementCounts[requirement.Name] > 1:
			outcome.Reason, outcome.Detail = ReasonDuplicateRequirement, "capability is declared more than once"
		case offerProblems[requirement.Name] != "":
			outcome.Reason, outcome.Detail = offerProblems[requirement.Name], offerProblemDetail(offerProblems[requirement.Name])
		default:
			offer, ok := offers[requirement.Name]
			if !ok {
				outcome.Reason, outcome.Detail = ReasonCapabilityAbsent, "host did not explicitly offer this capability"
			} else {
				outcome.OfferedRevision, outcome.OfferedStatus = offer.Revision, offer.Status
				switch {
				case offer.Revision < requirement.MinRevision:
					outcome.Reason, outcome.Detail = ReasonRevisionBelowMin, fmt.Sprintf("host revision %d is below minimum %d", offer.Revision, requirement.MinRevision)
				case offer.Revision > requirement.MaxRevision:
					outcome.Reason, outcome.Detail = ReasonRevisionAboveMax, fmt.Sprintf("host revision %d exceeds maximum %d", offer.Revision, requirement.MaxRevision)
				case requirement.Status != "" && offer.Status != requirement.Status:
					outcome.Reason, outcome.Detail = ReasonStatusMismatch, fmt.Sprintf("host status %q does not match %q", offer.Status, requirement.Status)
				default:
					outcome.Compatible, outcome.Reason, outcome.Detail = true, ReasonCompatible, "explicit offer satisfies the declared range"
				}
			}
		}
		if !outcome.Compatible && (outcome.Required || outcome.Reason == ReasonInvalidRequirement || outcome.Reason == ReasonDuplicateRequirement) {
			report.Compatible = false
		}
		report.Outcomes = append(report.Outcomes, outcome)
	}
	for name, reason := range offerProblems {
		report.Compatible = false
		report.Issues = append(report.Issues, CompatibilityIssue{Scope: "host_offer", Name: name, Reason: reason, Detail: offerProblemDetail(reason)})
	}
	sort.Slice(report.Outcomes, func(i, j int) bool {
		a, b := report.Outcomes[i], report.Outcomes[j]
		if a.Name != b.Name {
			return a.Name < b.Name
		}
		if a.Reason != b.Reason {
			return a.Reason < b.Reason
		}
		if a.MinRevision != b.MinRevision {
			return a.MinRevision < b.MinRevision
		}
		if a.MaxRevision != b.MaxRevision {
			return a.MaxRevision < b.MaxRevision
		}
		if a.Required != b.Required {
			return a.Required
		}
		return a.RequiredStatus < b.RequiredStatus
	})
	sort.Slice(report.Issues, func(i, j int) bool {
		if report.Issues[i].Name != report.Issues[j].Name {
			return report.Issues[i].Name < report.Issues[j].Name
		}
		return report.Issues[i].Reason < report.Issues[j].Reason
	})
	return report
}

func indexOffers(input []CapabilityOffer) (map[Capability]CapabilityOffer, map[Capability]CompatibilityReason) {
	offers := make(map[Capability]CapabilityOffer, len(input))
	problems := make(map[Capability]CompatibilityReason)
	counts := make(map[Capability]int)
	for _, offer := range input {
		counts[offer.Name]++
		current, exists := offers[offer.Name]
		if !exists || offerKey(offer) < offerKey(current) {
			offers[offer.Name] = offer
		}
	}
	for name, offer := range offers {
		switch {
		case counts[name] > 1:
			problems[name] = ReasonDuplicateOffer
		case name == "" || offer.Revision < 1 || !validStatus(offer.Status):
			problems[name] = ReasonInvalidOffer
		case offer.Status == StatusDeprecated:
			if offer.Deprecation == nil || offer.Deprecation.Replacement == "" || offer.Deprecation.Replacement == name || strings.TrimSpace(offer.Deprecation.RemovalHorizon) == "" {
				problems[name] = ReasonInvalidDeprecation
				break
			}
			replacement, ok := offers[offer.Deprecation.Replacement]
			if !ok || counts[offer.Deprecation.Replacement] != 1 || replacement.Revision < 1 || !validStatus(replacement.Status) || replacement.Status == StatusDeprecated {
				problems[name] = ReasonInvalidDeprecation
			}
		case offer.Deprecation != nil:
			problems[name] = ReasonInvalidDeprecation
		}
	}
	return offers, problems
}

func offerKey(offer CapabilityOffer) string {
	return fmt.Sprintf("%012d\x00%s\x00%s\x00%s", offer.Revision, offer.Status, offer.Semantics, summarizeDeprecation(offer.Deprecation))
}

func offerProblemDetail(reason CompatibilityReason) string {
	switch reason {
	case ReasonDuplicateOffer:
		return "host offered the capability more than once"
	case ReasonInvalidDeprecation:
		return "deprecated capability requires a valid non-deprecated replacement and removal horizon"
	default:
		return "host offer has no positive revision or explicit known status"
	}
}

func validStatus(status CapabilityStatus) bool {
	return status == StatusStable || status == StatusExperimental || status == StatusDeprecated
}

func validOptionalStatus(status CapabilityStatus) bool { return status == "" || validStatus(status) }

// ChangeClass is the semantic impact of an explicit contract change.
type ChangeClass string

const (
	ChangeAdditive   ChangeClass = "additive"
	ChangeBehavioral ChangeClass = "behavioral"
	ChangeBreaking   ChangeClass = "breaking"
)

// ContractChange is one explicit, non-reflection-derived semantic delta.
type ContractChange struct {
	Class      ChangeClass `json:"class"`
	Capability Capability  `json:"capability,omitempty"`
	Field      string      `json:"field"`
	Before     string      `json:"before,omitempty"`
	After      string      `json:"after,omitempty"`
	Detail     string      `json:"detail"`
}

// ContractDiff classifies every semantic transition in deterministic order.
type ContractDiff struct {
	SchemaVersion string           `json:"schema_version"`
	Changes       []ContractChange `json:"changes"`
}

func (d ContractDiff) JSON() ([]byte, error) { return json.MarshalIndent(d, "", "  ") }

// DiffContracts performs an explicit semantic comparison; Go struct layout and
// reflection are deliberately not part of the public ABI.
func DiffContracts(before, after RuntimeContract) ContractDiff {
	diff := ContractDiff{SchemaVersion: ContractDiffSchema, Changes: []ContractChange{}}
	if before.ContractVersion != after.ContractVersion {
		diff.Changes = append(diff.Changes, ContractChange{Class: ChangeBreaking, Field: "contract_version", Before: before.ContractVersion, After: after.ContractVersion, Detail: "contract line changed"})
	}
	oldOffers, _ := indexOffers(before.Capabilities)
	newOffers, _ := indexOffers(after.Capabilities)
	names := make(map[Capability]bool, len(oldOffers)+len(newOffers))
	for name := range oldOffers {
		names[name] = true
	}
	for name := range newOffers {
		names[name] = true
	}
	for name := range names {
		oldOffer, hadOld := oldOffers[name]
		newOffer, hasNew := newOffers[name]
		switch {
		case !hadOld:
			diff.Changes = append(diff.Changes, ContractChange{Class: ChangeAdditive, Capability: name, Field: "capability", After: summarizeOffer(newOffer), Detail: "capability added"})
		case !hasNew:
			diff.Changes = append(diff.Changes, ContractChange{Class: ChangeBreaking, Capability: name, Field: "capability", Before: summarizeOffer(oldOffer), Detail: "capability removed"})
		default:
			if oldOffer.Revision != newOffer.Revision {
				class := ChangeBehavioral
				if newOffer.Revision < oldOffer.Revision {
					class = ChangeBreaking
				}
				diff.Changes = append(diff.Changes, ContractChange{Class: class, Capability: name, Field: "revision", Before: fmt.Sprint(oldOffer.Revision), After: fmt.Sprint(newOffer.Revision), Detail: "capability semantics revision changed"})
			}
			if oldOffer.Status != newOffer.Status {
				class := ChangeBehavioral
				if oldOffer.Status == StatusStable && newOffer.Status == StatusExperimental {
					class = ChangeBreaking
				}
				if oldOffer.Status == StatusExperimental && newOffer.Status == StatusStable {
					class = ChangeAdditive
				}
				diff.Changes = append(diff.Changes, ContractChange{Class: class, Capability: name, Field: "status", Before: string(oldOffer.Status), After: string(newOffer.Status), Detail: "capability maturity changed"})
			}
			if oldOffer.Semantics != newOffer.Semantics {
				diff.Changes = append(diff.Changes, ContractChange{Class: ChangeBehavioral, Capability: name, Field: "semantics", Before: oldOffer.Semantics, After: newOffer.Semantics, Detail: "documented behavior changed"})
			}
			if summarizeDeprecation(oldOffer.Deprecation) != summarizeDeprecation(newOffer.Deprecation) {
				diff.Changes = append(diff.Changes, ContractChange{Class: ChangeBehavioral, Capability: name, Field: "deprecation", Before: summarizeDeprecation(oldOffer.Deprecation), After: summarizeDeprecation(newOffer.Deprecation), Detail: "migration metadata changed"})
			}
		}
	}
	sort.Slice(diff.Changes, func(i, j int) bool {
		a, b := diff.Changes[i], diff.Changes[j]
		if a.Capability != b.Capability {
			return a.Capability < b.Capability
		}
		if a.Field != b.Field {
			return a.Field < b.Field
		}
		return a.Class < b.Class
	})
	return diff
}

func summarizeOffer(offer CapabilityOffer) string {
	return fmt.Sprintf("revision=%d,status=%s", offer.Revision, offer.Status)
}

func summarizeDeprecation(deprecation *Deprecation) string {
	if deprecation == nil {
		return ""
	}
	return fmt.Sprintf("replacement=%s,removal_horizon=%s", deprecation.Replacement, deprecation.RemovalHorizon)
}

// UpgradeStepCode is a stable action category in a read-only upgrade plan.
type UpgradeStepCode string

const (
	StepBlockBreaking  UpgradeStepCode = "block_breaking_change"
	StepBlockRequired  UpgradeStepCode = "block_required_incompatibility"
	StepReviewBehavior UpgradeStepCode = "review_behavioral_change"
	StepOptionalGap    UpgradeStepCode = "review_optional_gap"
	StepReplaceLegacy  UpgradeStepCode = "replace_deprecated_capability"
	StepReady          UpgradeStepCode = "ready"
)

type UpgradeStep struct {
	Code       UpgradeStepCode `json:"code"`
	Capability Capability      `json:"capability,omitempty"`
	Blocking   bool            `json:"blocking"`
	Detail     string          `json:"detail"`
}

// UpgradePlan is advisory and side-effect-free. It never edits product files.
type UpgradePlan struct {
	SchemaVersion string              `json:"schema_version"`
	Allowed       bool                `json:"allowed"`
	Current       CompatibilityReport `json:"current"`
	Target        CompatibilityReport `json:"target"`
	Diff          ContractDiff        `json:"diff"`
	Steps         []UpgradeStep       `json:"steps"`
}

func (p UpgradePlan) JSON() ([]byte, error) { return json.MarshalIndent(p, "", "  ") }

// PlanUpgrade computes an actionable plan and performs no writes or activation.
func PlanUpgrade(builder BuilderContract, current, target RuntimeContract) UpgradePlan {
	plan := UpgradePlan{
		SchemaVersion: UpgradePlanSchema,
		Allowed:       true,
		Current:       NegotiateCompatibility(builder, current),
		Target:        NegotiateCompatibility(builder, target),
		Diff:          DiffContracts(current, target),
		Steps:         []UpgradeStep{},
	}
	for _, change := range plan.Diff.Changes {
		switch change.Class {
		case ChangeBreaking:
			plan.Allowed = false
			plan.Steps = append(plan.Steps, UpgradeStep{Code: StepBlockBreaking, Capability: change.Capability, Blocking: true, Detail: change.Detail + " (" + change.Field + ")"})
		case ChangeBehavioral:
			plan.Steps = append(plan.Steps, UpgradeStep{Code: StepReviewBehavior, Capability: change.Capability, Detail: change.Detail + " (" + change.Field + ")"})
		}
	}
	for _, outcome := range plan.Target.Outcomes {
		if !outcome.Compatible {
			if outcome.Required {
				plan.Allowed = false
				plan.Steps = append(plan.Steps, UpgradeStep{Code: StepBlockRequired, Capability: outcome.Name, Blocking: true, Detail: string(outcome.Reason) + ": " + outcome.Detail})
			} else {
				plan.Steps = append(plan.Steps, UpgradeStep{Code: StepOptionalGap, Capability: outcome.Name, Detail: string(outcome.Reason) + ": " + outcome.Detail})
			}
		}
	}
	for _, issue := range plan.Target.Issues {
		plan.Allowed = false
		plan.Steps = append(plan.Steps, UpgradeStep{Code: StepBlockRequired, Capability: issue.Name, Blocking: true, Detail: string(issue.Reason) + ": " + issue.Detail})
	}
	for _, offer := range target.Capabilities {
		if offer.Status == StatusDeprecated && offer.Deprecation != nil {
			plan.Steps = append(plan.Steps, UpgradeStep{Code: StepReplaceLegacy, Capability: offer.Name, Detail: fmt.Sprintf("migrate to %s before %s", offer.Deprecation.Replacement, offer.Deprecation.RemovalHorizon)})
		}
	}
	if plan.Target.ContractReason != ReasonCompatible {
		plan.Allowed = false
		plan.Steps = append(plan.Steps, UpgradeStep{Code: StepBlockRequired, Blocking: true, Detail: "target contract: " + string(plan.Target.ContractReason)})
	}
	if len(plan.Steps) == 0 {
		plan.Steps = append(plan.Steps, UpgradeStep{Code: StepReady, Detail: "target satisfies the declared contract; activate with ActivateCompatible"})
	}
	sort.Slice(plan.Steps, func(i, j int) bool {
		a, b := plan.Steps[i], plan.Steps[j]
		if a.Blocking != b.Blocking {
			return a.Blocking
		}
		if a.Capability != b.Capability {
			return a.Capability < b.Capability
		}
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		return a.Detail < b.Detail
	})
	return plan
}
