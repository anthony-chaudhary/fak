package modelroute

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

const (
	AuditRouteRosterSchema  = "fak-crossaudit-route-roster/v1"
	AuditRouteHealthSchema  = "fak-crossaudit-route-health/v1"
	AuditIssuePlanSchema    = "fak-crossaudit-route-plan/v1"
	MaxAuditRouteCandidates = 16
	MaxAuditRoutePriority   = 1_000_000
)

type AuditProviderHealthStatus string

const (
	AuditProviderHealthy   AuditProviderHealthStatus = "HEALTHY"
	AuditProviderDegraded  AuditProviderHealthStatus = "DEGRADED"
	AuditProviderUnhealthy AuditProviderHealthStatus = "UNHEALTHY"
	AuditProviderUnknown   AuditProviderHealthStatus = "UNKNOWN"
)

func (status AuditProviderHealthStatus) Valid() bool {
	switch status {
	case AuditProviderHealthy, AuditProviderDegraded, AuditProviderUnhealthy, AuditProviderUnknown:
		return true
	default:
		return false
	}
}

type AuditCapacityStatus string

const (
	AuditCapacityAvailable AuditCapacityStatus = "AVAILABLE"
	AuditCapacitySaturated AuditCapacityStatus = "SATURATED"
	AuditCapacityUnknown   AuditCapacityStatus = "UNKNOWN"
)

func (status AuditCapacityStatus) Valid() bool {
	return status == AuditCapacityAvailable || status == AuditCapacitySaturated || status == AuditCapacityUnknown
}

type AuditCooldownStatus string

const (
	AuditCooldownReady   AuditCooldownStatus = "READY"
	AuditCooldownActive  AuditCooldownStatus = "ACTIVE"
	AuditCooldownUnknown AuditCooldownStatus = "UNKNOWN"
)

func (status AuditCooldownStatus) Valid() bool {
	return status == AuditCooldownReady || status == AuditCooldownActive || status == AuditCooldownUnknown
}

// AuditRouteHealth is an injected, credential-free readiness snapshot. Provider
// health applies to every alias on that provider; capacity is candidate-specific.
type AuditRouteHealth struct {
	Schema    string                               `json:"schema"`
	Providers map[string]AuditProviderHealthStatus `json:"providers"`
	Capacity  map[string]AuditCapacityStatus       `json:"capacity"`
	Cooldown  map[string]AuditCooldownStatus       `json:"cooldown"`
}

type AuditRoutePrice struct {
	InputMicrosPerMillionTokens  int64 `json:"input_micros_per_million_tokens"`
	OutputMicrosPerMillionTokens int64 `json:"output_micros_per_million_tokens"`
}

// AuditAuditorConfig is a configured candidate. Identity carries the actual
// effort posture that dispatch will use; Capability is the strongest work tier
// the operator's evidence says the candidate can serve.
type AuditAuditorConfig struct {
	ID               string          `json:"id"`
	Identity         AuditIdentity   `json:"identity"`
	Capability       WorkTier        `json:"capability"`
	CapabilitySource string          `json:"capability_source"`
	Price            AuditRoutePrice `json:"price"`
	Priority         int             `json:"priority,omitempty"`
}

// AuditRouteRoster contains model identities and public planning metadata only.
// There is deliberately no credential, token, base URL, or environment field.
type AuditRouteRoster struct {
	Schema                string               `json:"schema"`
	Aliases               []AuditIdentityAlias `json:"aliases"`
	Candidates            []AuditAuditorConfig `json:"candidates"`
	UnknownAuthorQuorum   int                  `json:"unknown_author_quorum,omitempty"`
	EstimatedInputTokens  int64                `json:"estimated_input_tokens"`
	EstimatedOutputTokens int64                `json:"estimated_output_tokens"`
}

type AuditRouteSkipReason string

const (
	AuditSkipNone                  AuditRouteSkipReason = ""
	AuditSkipIdentityUnresolved    AuditRouteSkipReason = "IDENTITY_UNRESOLVED"
	AuditSkipIdentityConflict      AuditRouteSkipReason = "IDENTITY_CONFLICT"
	AuditSkipDriverIdentityInvalid AuditRouteSkipReason = "DRIVER_IDENTITY_INVALID"
	AuditSkipReciprocalRule        AuditRouteSkipReason = "RECIPROCAL_RULE_MISMATCH"
	AuditSkipProviderUnhealthy     AuditRouteSkipReason = "PROVIDER_UNHEALTHY"
	AuditSkipProviderHealthUnknown AuditRouteSkipReason = "PROVIDER_HEALTH_UNKNOWN"
	AuditSkipCapacitySaturated     AuditRouteSkipReason = "CAPACITY_SATURATED"
	AuditSkipCapacityUnknown       AuditRouteSkipReason = "CAPACITY_UNKNOWN"
	AuditSkipCooldownActive        AuditRouteSkipReason = "COOLDOWN_ACTIVE"
	AuditSkipCooldownUnknown       AuditRouteSkipReason = "COOLDOWN_UNKNOWN"
	AuditSkipBelowTierFloor        AuditRouteSkipReason = "BELOW_TIER_FLOOR"
	AuditSkipBelowEffortFloor      AuditRouteSkipReason = "BELOW_EFFORT_FLOOR"
	AuditSkipIndependence          AuditRouteSkipReason = "INDEPENDENCE_NOT_ADMITTED"
	AuditSkipQuorumNotDiverse      AuditRouteSkipReason = "QUORUM_NOT_DIVERSE"
)

func (reason AuditRouteSkipReason) Valid() bool {
	switch reason {
	case AuditSkipNone, AuditSkipIdentityUnresolved, AuditSkipIdentityConflict, AuditSkipDriverIdentityInvalid, AuditSkipReciprocalRule,
		AuditSkipProviderUnhealthy, AuditSkipProviderHealthUnknown, AuditSkipCapacitySaturated,
		AuditSkipCapacityUnknown, AuditSkipCooldownActive, AuditSkipCooldownUnknown,
		AuditSkipBelowTierFloor, AuditSkipBelowEffortFloor, AuditSkipIndependence, AuditSkipQuorumNotDiverse:
		return true
	default:
		return false
	}
}

type AuditRouteNoRouteReason string

const (
	AuditNoRouteInvalidRisk       AuditRouteNoRouteReason = "NO_ROUTE_INVALID_RISK"
	AuditNoRouteInvalidRoster     AuditRouteNoRouteReason = "NO_ROUTE_INVALID_ROSTER"
	AuditNoRouteInvalidHealth     AuditRouteNoRouteReason = "NO_ROUTE_INVALID_HEALTH"
	AuditNoRouteAuthorConflict    AuditRouteNoRouteReason = "NO_ROUTE_AUTHOR_IDENTITY_CONFLICT"
	AuditNoRouteNoHealthyProvider AuditRouteNoRouteReason = "NO_ROUTE_NO_HEALTHY_PROVIDER"
	AuditNoRouteNoCapacity        AuditRouteNoRouteReason = "NO_ROUTE_NO_CAPACITY"
	AuditNoRouteCooldown          AuditRouteNoRouteReason = "NO_ROUTE_COOLDOWN"
	AuditNoRouteTierFloor         AuditRouteNoRouteReason = "NO_ROUTE_TIER_FLOOR"
	AuditNoRouteEffortFloor       AuditRouteNoRouteReason = "NO_ROUTE_EFFORT_FLOOR"
	AuditNoRouteNoIndependent     AuditRouteNoRouteReason = "NO_ROUTE_NO_INDEPENDENT_AUDITOR"
	AuditNoRouteDiversifiedQuorum AuditRouteNoRouteReason = "NO_ROUTE_DIVERSIFIED_QUORUM"
)

func (reason AuditRouteNoRouteReason) Valid() bool {
	switch reason {
	case AuditNoRouteInvalidRisk, AuditNoRouteInvalidRoster, AuditNoRouteInvalidHealth,
		AuditNoRouteAuthorConflict, AuditNoRouteNoHealthyProvider, AuditNoRouteNoCapacity,
		AuditNoRouteCooldown, AuditNoRouteTierFloor, AuditNoRouteEffortFloor, AuditNoRouteNoIndependent,
		AuditNoRouteDiversifiedQuorum:
		return true
	default:
		return false
	}
}

type AuditRouteCandidateDecision struct {
	CandidateID            string                    `json:"candidate_id"`
	Identity               AuditIdentity             `json:"identity"`
	Capability             WorkTier                  `json:"capability"`
	CapabilitySource       string                    `json:"capability_source"`
	ActualEffort           string                    `json:"actual_effort"`
	ProviderHealth         AuditProviderHealthStatus `json:"provider_health"`
	Capacity               AuditCapacityStatus       `json:"capacity"`
	Cooldown               AuditCooldownStatus       `json:"cooldown"`
	EstimatedCostMicrosUSD int64                     `json:"estimated_cost_micros_usd"`
	Independence           AuditIndependenceDecision `json:"independence"`
	Admitted               bool                      `json:"admitted"`
	SkipReason             AuditRouteSkipReason      `json:"skip_reason,omitempty"`
	Preference             int                       `json:"preference"`
	Priority               int                       `json:"priority,omitempty"`
}

type AuditPlannedAuditor struct {
	Rank                   int                       `json:"rank"`
	CandidateID            string                    `json:"candidate_id"`
	Identity               AuditIdentity             `json:"identity"`
	Capability             WorkTier                  `json:"capability"`
	CapabilitySource       string                    `json:"capability_source"`
	ActualEffort           string                    `json:"actual_effort"`
	ProviderHealth         AuditProviderHealthStatus `json:"provider_health"`
	Capacity               AuditCapacityStatus       `json:"capacity"`
	Cooldown               AuditCooldownStatus       `json:"cooldown"`
	EstimatedCostMicrosUSD int64                     `json:"estimated_cost_micros_usd"`
	CostCurrency           string                    `json:"cost_currency"`
	CostBasis              string                    `json:"cost_basis"`
	Independence           AuditIndependenceDecision `json:"independence"`
}

type AuditIssuePlan struct {
	Schema                string                        `json:"schema"`
	Risk                  AuditRisk                     `json:"risk"`
	Author                AuditIdentity                 `json:"author"`
	AuthorUnknown         bool                          `json:"author_unknown"`
	DiversifiedQuorum     bool                          `json:"diversified_quorum"`
	RequiredTier          WorkTier                      `json:"required_tier"`
	RequiredEffort        string                        `json:"required_effort"`
	RequiredQuorum        int                           `json:"required_quorum"`
	PolicyVersion         string                        `json:"policy_version"`
	PolicyDigest          string                        `json:"policy_digest"`
	EstimatedInputTokens  int64                         `json:"estimated_input_tokens"`
	EstimatedOutputTokens int64                         `json:"estimated_output_tokens"`
	QuorumCostMicrosUSD   int64                         `json:"quorum_cost_micros_usd"`
	QuorumGroups          []AuditQuorumGroup            `json:"quorum_groups,omitempty"`
	Candidates            []AuditPlannedAuditor         `json:"candidates"`
	Considered            []AuditRouteCandidateDecision `json:"considered"`
}

type AuditQuorumGroup struct {
	Rank                   int      `json:"rank"`
	CandidateIDs           []string `json:"candidate_ids"`
	EstimatedCostMicrosUSD int64    `json:"estimated_cost_micros_usd"`
}

type AuditNoRouteError struct {
	Reason AuditRouteNoRouteReason
	Detail string
	Plan   AuditIssuePlan
}

func (e *AuditNoRouteError) Error() string {
	if strings.TrimSpace(e.Detail) == "" {
		return "modelroute: " + string(e.Reason)
	}
	return fmt.Sprintf("modelroute: %s: %s", e.Reason, e.Detail)
}

func (e *AuditNoRouteError) Is(target error) bool {
	_, ok := target.(*AuditNoRouteError)
	return ok
}

func IsAuditNoRoute(err error) bool {
	var target *AuditNoRouteError
	return errors.As(err, &target)
}

type auditAuthorClass string

const (
	auditAuthorClaude  auditAuthorClass = "claude"
	auditAuthorGPT     auditAuthorClass = "gpt"
	auditAuthorLocal   auditAuthorClass = "local"
	auditAuthorUnknown auditAuthorClass = "unknown"
)

// PlanIssueAudit constructs a credential-free, ordered reciprocal audit route.
// It performs no I/O and never resolves credentials or calls a model.
func PlanIssueAudit(author AuditIdentity, risk AuditRisk, roster AuditRouteRoster, health AuditRouteHealth) (AuditIssuePlan, error) {
	plan := AuditIssuePlan{Schema: AuditIssuePlanSchema, Risk: risk}
	if risk != AuditRiskDefault && risk != AuditRiskHigh {
		return plan, &AuditNoRouteError{Reason: AuditNoRouteInvalidRisk, Detail: fmt.Sprintf("risk %q is not declared", risk), Plan: plan}
	}
	if err := validateAuditRouteRoster(roster); err != nil {
		return plan, &AuditNoRouteError{Reason: AuditNoRouteInvalidRoster, Detail: err.Error(), Plan: plan}
	}
	if err := validateAuditRouteHealth(roster, health); err != nil {
		return plan, &AuditNoRouteError{Reason: AuditNoRouteInvalidHealth, Detail: err.Error(), Plan: plan}
	}

	policy := DefaultAuditIndependencePolicy()
	if risk == AuditRiskHigh {
		policy = HighRiskAuditIndependencePolicy()
	}
	policy.Aliases = append([]AuditIdentityAlias(nil), roster.Aliases...)
	plan.PolicyVersion = policy.Version
	plan.PolicyDigest = policy.Digest()
	plan.EstimatedInputTokens = roster.EstimatedInputTokens
	plan.EstimatedOutputTokens = roster.EstimatedOutputTokens
	plan.RequiredTier, plan.RequiredEffort = auditRouteFloors(risk)

	canonicalAuthor, authorStatus := normalizeAuditIdentity(author, roster.Aliases)
	authorClass := classifyAuditAuthor(canonicalAuthor, authorStatus)
	if authorStatus == identityAliasConflict || authorStatus == identityAliasAmbiguous {
		plan.Author = canonicalAuthor
		return plan, &AuditNoRouteError{Reason: AuditNoRouteAuthorConflict, Detail: string(authorStatus), Plan: plan}
	}
	plan.Author = canonicalAuthor
	if err := validateAuditIdentityLocality(canonicalAuthor); err != nil {
		return plan, &AuditNoRouteError{Reason: AuditNoRouteAuthorConflict, Detail: err.Error(), Plan: plan}
	}
	authorResolved := authorStatus == identityRosterResolved && len(missingAuditAxes(canonicalAuthor, canonicalAuthor, DefaultAuditIndependencePolicy().RequiredAxes)) == 0
	plan.AuthorUnknown = !authorResolved
	plan.DiversifiedQuorum = authorClass == auditAuthorUnknown
	plan.RequiredQuorum = 1
	if plan.DiversifiedQuorum {
		plan.RequiredQuorum = roster.UnknownAuthorQuorum
		if plan.RequiredQuorum < 2 {
			plan.RequiredQuorum = 2
		}
	}

	providerHealth := normalizeAuditProviderHealth(health.Providers)
	decisions := make([]AuditRouteCandidateDecision, 0, len(roster.Candidates))
	for _, configured := range roster.Candidates {
		decision := evaluateAuditRouteCandidate(plan, authorClass, authorResolved, configured, policy, providerHealth, health.Capacity, health.Cooldown)
		decisions = append(decisions, decision)
	}
	sortAuditRouteDecisions(decisions)

	var orderedIndexes []int
	if plan.DiversifiedQuorum {
		groups := auditDiversifiedQuorumGroups(decisions, plan.RequiredQuorum)
		for groupIndex, indexes := range groups {
			group := AuditQuorumGroup{Rank: groupIndex + 1}
			for _, index := range indexes {
				group.CandidateIDs = append(group.CandidateIDs, decisions[index].CandidateID)
				if group.EstimatedCostMicrosUSD > math.MaxInt64-decisions[index].EstimatedCostMicrosUSD {
					return plan, &AuditNoRouteError{Reason: AuditNoRouteInvalidRoster, Detail: "quorum cost overflows int64", Plan: plan}
				}
				group.EstimatedCostMicrosUSD += decisions[index].EstimatedCostMicrosUSD
			}
			plan.QuorumGroups = append(plan.QuorumGroups, group)
		}
		selected := map[int]bool{}
		if len(groups) > 0 {
			for _, index := range groups[0] {
				orderedIndexes = append(orderedIndexes, index)
				selected[index] = true
			}
		}
		for i, decision := range decisions {
			if decision.Admitted && !selected[i] {
				orderedIndexes = append(orderedIndexes, i)
			}
		}
	} else {
		for i, decision := range decisions {
			if decision.Admitted {
				orderedIndexes = append(orderedIndexes, i)
			}
		}
	}
	for _, index := range orderedIndexes {
		decision := &decisions[index]
		plan.Candidates = append(plan.Candidates, AuditPlannedAuditor{
			Rank: len(plan.Candidates) + 1, CandidateID: decision.CandidateID, Identity: decision.Identity,
			Capability: decision.Capability, CapabilitySource: decision.CapabilitySource, ActualEffort: decision.ActualEffort,
			ProviderHealth: decision.ProviderHealth, Capacity: decision.Capacity, Cooldown: decision.Cooldown,
			EstimatedCostMicrosUSD: decision.EstimatedCostMicrosUSD, CostCurrency: "USD", CostBasis: "roster-estimate",
			Independence: decision.Independence,
		})
	}
	plan.Considered = decisions
	quorumReady := len(plan.Candidates) >= plan.RequiredQuorum
	if plan.DiversifiedQuorum {
		quorumReady = len(plan.QuorumGroups) > 0
	}
	if !quorumReady {
		reason := auditNoRouteReason(decisions, plan.DiversifiedQuorum)
		return plan, &AuditNoRouteError{
			Reason: reason, Detail: fmt.Sprintf("planned %d candidate(s), require %d", len(plan.Candidates), plan.RequiredQuorum), Plan: plan,
		}
	}
	if plan.DiversifiedQuorum {
		plan.QuorumCostMicrosUSD = plan.QuorumGroups[0].EstimatedCostMicrosUSD
	} else {
		for i := 0; i < plan.RequiredQuorum; i++ {
			if plan.QuorumCostMicrosUSD > math.MaxInt64-plan.Candidates[i].EstimatedCostMicrosUSD {
				return plan, &AuditNoRouteError{Reason: AuditNoRouteInvalidRoster, Detail: "quorum cost overflows int64", Plan: plan}
			}
			plan.QuorumCostMicrosUSD += plan.Candidates[i].EstimatedCostMicrosUSD
		}
	}
	return plan, nil
}

func validateAuditRouteRoster(roster AuditRouteRoster) error {
	if roster.Schema != AuditRouteRosterSchema {
		return fmt.Errorf("audit route roster schema %q, want %q", roster.Schema, AuditRouteRosterSchema)
	}
	if err := (AuditIdentityRoster{Schema: AuditIdentityRosterSchema, Aliases: roster.Aliases}).Validate(); err != nil {
		return err
	}
	if len(roster.Candidates) == 0 {
		return fmt.Errorf("audit route roster has no candidates")
	}
	if len(roster.Candidates) > MaxAuditRouteCandidates {
		return fmt.Errorf("audit route roster has %d candidates, maximum %d for deterministic quorum planning", len(roster.Candidates), MaxAuditRouteCandidates)
	}
	if roster.EstimatedInputTokens < 0 || roster.EstimatedOutputTokens < 0 {
		return fmt.Errorf("audit route token estimates cannot be negative")
	}
	if roster.UnknownAuthorQuorum < 0 || roster.UnknownAuthorQuorum == 1 {
		return fmt.Errorf("audit route unknown-author quorum must be 0 or at least 2")
	}
	seen := map[string]bool{}
	for i, candidate := range roster.Candidates {
		if candidate.ID == "" || candidate.ID != strings.TrimSpace(candidate.ID) || seen[candidate.ID] {
			return fmt.Errorf("audit route candidate %d has empty or duplicate id %q", i, candidate.ID)
		}
		seen[candidate.ID] = true
		if !candidate.Capability.Valid() {
			return fmt.Errorf("audit route candidate %q has invalid capability %s", candidate.ID, candidate.Capability)
		}
		if strings.TrimSpace(candidate.CapabilitySource) == "" {
			return fmt.Errorf("audit route candidate %q has no capability source", candidate.ID)
		}
		if candidate.Priority < 0 || candidate.Priority > MaxAuditRoutePriority {
			return fmt.Errorf("audit route candidate %q priority %d is outside 0..%d", candidate.ID, candidate.Priority, MaxAuditRoutePriority)
		}
		if candidate.Price.InputMicrosPerMillionTokens < 0 || candidate.Price.OutputMicrosPerMillionTokens < 0 {
			return fmt.Errorf("audit route candidate %q has negative price", candidate.ID)
		}
		if _, err := estimateAuditRouteCost(candidate.Price, roster.EstimatedInputTokens, roster.EstimatedOutputTokens); err != nil {
			return fmt.Errorf("audit route candidate %q price: %w", candidate.ID, err)
		}
	}
	return nil
}

func validateAuditRouteHealth(roster AuditRouteRoster, health AuditRouteHealth) error {
	if health.Schema != AuditRouteHealthSchema {
		return fmt.Errorf("audit route health schema %q, want %q", health.Schema, AuditRouteHealthSchema)
	}
	for provider, status := range health.Providers {
		if provider == "" || provider != strings.ToLower(strings.TrimSpace(provider)) || !status.Valid() {
			return fmt.Errorf("audit route provider health %q=%q is invalid", provider, status)
		}
	}
	for candidate, status := range health.Capacity {
		if candidate == "" || candidate != strings.TrimSpace(candidate) || !status.Valid() {
			return fmt.Errorf("audit route capacity %q=%q is invalid", candidate, status)
		}
	}
	for candidate, status := range health.Cooldown {
		if candidate == "" || candidate != strings.TrimSpace(candidate) || !status.Valid() {
			return fmt.Errorf("audit route cooldown %q=%q is invalid", candidate, status)
		}
	}
	wantCandidates := make(map[string]bool, len(roster.Candidates))
	for _, candidate := range roster.Candidates {
		wantCandidates[candidate.ID] = true
	}
	for candidate := range health.Capacity {
		if !wantCandidates[candidate] {
			return fmt.Errorf("audit route capacity names unknown candidate %q", candidate)
		}
	}
	for candidate := range health.Cooldown {
		if !wantCandidates[candidate] {
			return fmt.Errorf("audit route cooldown names unknown candidate %q", candidate)
		}
	}
	return nil
}

func normalizeAuditProviderHealth(in map[string]AuditProviderHealthStatus) map[string]AuditProviderHealthStatus {
	out := make(map[string]AuditProviderHealthStatus, len(in))
	for provider, status := range in {
		out[strings.ToLower(strings.TrimSpace(provider))] = status
	}
	return out
}

func auditRouteFloors(risk AuditRisk) (WorkTier, string) {
	if risk == AuditRiskHigh {
		return TierT0, "xhigh"
	}
	return TierT1, "high"
}

func classifyAuditAuthor(author AuditIdentity, status identityNormalizationStatus) auditAuthorClass {
	if status != identityRosterResolved || len(missingAuditAxes(author, author, DefaultAuditIndependencePolicy().RequiredAxes)) > 0 {
		return auditAuthorUnknown
	}
	if isLocalAuditIdentity(author) {
		return auditAuthorLocal
	}
	switch author.Family {
	case "claude":
		return auditAuthorClaude
	case "gpt", "openai-o":
		return auditAuthorGPT
	default:
		return auditAuthorUnknown
	}
}

func evaluateAuditRouteCandidate(
	plan AuditIssuePlan,
	authorClass auditAuthorClass,
	authorResolved bool,
	configured AuditAuditorConfig,
	policy AuditIndependencePolicy,
	providerHealth map[string]AuditProviderHealthStatus,
	capacity map[string]AuditCapacityStatus,
	cooldown map[string]AuditCooldownStatus,
) AuditRouteCandidateDecision {
	decision := AuditRouteCandidateDecision{
		CandidateID: configured.ID, Capability: configured.Capability, CapabilitySource: configured.CapabilitySource, Priority: configured.Priority,
		ProviderHealth: AuditProviderUnknown, Capacity: AuditCapacityUnknown, Cooldown: AuditCooldownUnknown,
	}
	identity, status := normalizeAuditIdentity(configured.Identity, policy.Aliases)
	decision.Identity = identity
	decision.ActualEffort = identity.ReasoningPosture
	decision.Preference = auditReciprocalPreference(authorClass, identity, configured.Capability)
	decision.EstimatedCostMicrosUSD, _ = estimateAuditRouteCost(configured.Price, plan.EstimatedInputTokens, plan.EstimatedOutputTokens)
	if status == identityAliasConflict || status == identityAliasAmbiguous {
		decision.SkipReason = AuditSkipIdentityConflict
		return decision
	}
	if err := validateAuditIdentityLocality(identity); err != nil {
		decision.SkipReason = AuditSkipIdentityConflict
		return decision
	}
	if status != identityRosterResolved || len(missingAuditAxes(identity, identity, DefaultAuditIndependencePolicy().RequiredAxes)) > 0 {
		decision.SkipReason = AuditSkipIdentityUnresolved
		return decision
	}
	validatedIdentity, err := ValidateAuditDriverIdentity(identity.Driver, identity, policy.Aliases)
	if err != nil {
		decision.SkipReason = AuditSkipDriverIdentityInvalid
		return decision
	}
	identity = validatedIdentity
	decision.Identity = identity
	if decision.Preference < 0 {
		decision.SkipReason = AuditSkipReciprocalRule
		return decision
	}
	decision.ProviderHealth = providerHealth[identity.Provider]
	switch decision.ProviderHealth {
	case AuditProviderHealthy, AuditProviderDegraded:
	case AuditProviderUnhealthy:
		decision.SkipReason = AuditSkipProviderUnhealthy
		return decision
	default:
		decision.ProviderHealth = AuditProviderUnknown
		decision.SkipReason = AuditSkipProviderHealthUnknown
		return decision
	}
	decision.Capacity = capacity[configured.ID]
	switch decision.Capacity {
	case AuditCapacityAvailable:
	case AuditCapacitySaturated:
		decision.SkipReason = AuditSkipCapacitySaturated
		return decision
	default:
		decision.Capacity = AuditCapacityUnknown
		decision.SkipReason = AuditSkipCapacityUnknown
		return decision
	}
	decision.Cooldown = cooldown[configured.ID]
	switch decision.Cooldown {
	case AuditCooldownReady:
	case AuditCooldownActive:
		decision.SkipReason = AuditSkipCooldownActive
		return decision
	default:
		decision.Cooldown = AuditCooldownUnknown
		decision.SkipReason = AuditSkipCooldownUnknown
		return decision
	}
	if !configured.Capability.MeetsRequirement(plan.RequiredTier) {
		decision.SkipReason = AuditSkipBelowTierFloor
		return decision
	}
	if !auditEffortMeets(identity.ReasoningPosture, plan.RequiredEffort) {
		decision.SkipReason = AuditSkipBelowEffortFloor
		return decision
	}
	decision.Independence = EvaluateAuditIndependence(plan.Author, identity, policy)
	if decision.Independence.Verdict == AuditIndependenceRefuse || (authorResolved && decision.Independence.Verdict != AuditIndependenceAdmit) {
		decision.SkipReason = AuditSkipIndependence
		return decision
	}
	decision.Admitted = true
	return decision
}

func auditReciprocalPreference(authorClass auditAuthorClass, auditor AuditIdentity, capability WorkTier) int {
	local := isLocalAuditIdentity(auditor)
	switch authorClass {
	case auditAuthorClaude:
		if auditor.Family == "gpt" || auditor.Family == "openai-o" {
			return 0
		}
		if local {
			return 1
		}
	case auditAuthorGPT:
		if auditor.Family == "claude" {
			return 0
		}
		if local {
			return 1
		}
	case auditAuthorLocal:
		if !local && capability == TierT0 {
			return 0
		}
	case auditAuthorUnknown:
		return 0
	}
	return -1
}

func isLocalAuditIdentity(identity AuditIdentity) bool {
	return identity.Provider == "local" || identity.EndpointClass == "local" || identity.EndpointClass == "local-http"
}

func validateAuditIdentityLocality(identity AuditIdentity) error {
	if identity.Provider == "" || identity.EndpointClass == "" {
		return nil
	}
	endpointLocal := identity.EndpointClass == "local" || identity.EndpointClass == "local-http"
	providerLocal := identity.Provider == "local"
	if endpointLocal != providerLocal {
		return fmt.Errorf("audit identity locality contradicts provider=%q endpoint_class=%q", identity.Provider, identity.EndpointClass)
	}
	return nil
}

func auditEffortMeets(actual, required string) bool {
	actualRank, actualOK := auditEffortRank(actual)
	requiredRank, requiredOK := auditEffortRank(required)
	return actualOK && requiredOK && actualRank >= requiredRank
}

func auditEffortRank(effort string) (int, bool) {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "none", "minimal":
		return 0, true
	case "low":
		return 1, true
	case "medium", "normal", "default":
		return 2, true
	case "high":
		return 3, true
	case "xhigh", "very-high", "ultra", "ultracode":
		return 4, true
	case "max", "maximum":
		return 5, true
	default:
		return 0, false
	}
}

func estimateAuditRouteCost(price AuditRoutePrice, inputTokens, outputTokens int64) (int64, error) {
	input, err := auditMicrosForTokens(inputTokens, price.InputMicrosPerMillionTokens)
	if err != nil {
		return 0, err
	}
	output, err := auditMicrosForTokens(outputTokens, price.OutputMicrosPerMillionTokens)
	if err != nil {
		return 0, err
	}
	if input > math.MaxInt64-output {
		return 0, fmt.Errorf("audit route cost overflows int64")
	}
	return input + output, nil
}

func auditMicrosForTokens(tokens, microsPerMillion int64) (int64, error) {
	if tokens < 0 || microsPerMillion < 0 {
		return 0, fmt.Errorf("audit route cost inputs cannot be negative")
	}
	const million int64 = 1_000_000
	tokenMillions, tokenRemainder := tokens/million, tokens%million
	priceMillions, priceRemainder := microsPerMillion/million, microsPerMillion%million

	// Divide before multiplying without losing the ceiling. Decomposing both
	// operands keeps every intermediate representable whenever the final cost is.
	if tokenMillions != 0 && microsPerMillion > math.MaxInt64/tokenMillions {
		return 0, fmt.Errorf("audit route cost overflows int64")
	}
	cost := tokenMillions * microsPerMillion
	remainderWhole := tokenRemainder * priceMillions
	if cost > math.MaxInt64-remainderWhole {
		return 0, fmt.Errorf("audit route cost overflows int64")
	}
	cost += remainderWhole
	remainderProduct := tokenRemainder * priceRemainder
	remainderCost := remainderProduct / million
	if remainderProduct%million != 0 {
		remainderCost++
	}
	if cost > math.MaxInt64-remainderCost {
		return 0, fmt.Errorf("audit route cost overflows int64")
	}
	return cost + remainderCost, nil
}

func sortAuditRouteDecisions(decisions []AuditRouteCandidateDecision) {
	sort.SliceStable(decisions, func(i, j int) bool {
		a, b := decisions[i], decisions[j]
		if a.Admitted != b.Admitted {
			return a.Admitted
		}
		if a.Preference != b.Preference {
			return a.Preference < b.Preference
		}
		aHealth, bHealth := auditHealthRank(a.ProviderHealth), auditHealthRank(b.ProviderHealth)
		if aHealth != bHealth {
			return aHealth < bHealth
		}
		if a.Priority != b.Priority {
			return a.Priority < b.Priority
		}
		if a.EstimatedCostMicrosUSD != b.EstimatedCostMicrosUSD {
			return a.EstimatedCostMicrosUSD < b.EstimatedCostMicrosUSD
		}
		if a.Capability != b.Capability {
			return a.Capability < b.Capability
		}
		return a.CandidateID < b.CandidateID
	})
}

// auditDiversifiedQuorumGroups enumerates every valid provider+family+weights
// diverse quorum, then orders groups by worst provider health, bounded summed
// priority, total microunit cost, and stable IDs. The roster is capped so
// exhaustive selection is bounded and cannot fall into a greedy local optimum.
func auditDiversifiedQuorumGroups(decisions []AuditRouteCandidateDecision, quorum int) [][]int {
	var chosen []int
	var groups [][]int
	providers, families, weights := map[string]bool{}, map[string]bool{}, map[string]bool{}
	var search func(int)
	search = func(start int) {
		if len(chosen) == quorum {
			groups = append(groups, append([]int(nil), chosen...))
			return
		}
		if len(decisions)-start < quorum-len(chosen) {
			return
		}
		for i := start; i < len(decisions); i++ {
			candidate := decisions[i]
			if !candidate.Admitted || providers[candidate.Identity.Provider] || families[candidate.Identity.Family] || weights[candidate.Identity.WeightsRevision] {
				continue
			}
			providers[candidate.Identity.Provider] = true
			families[candidate.Identity.Family] = true
			weights[candidate.Identity.WeightsRevision] = true
			chosen = append(chosen, i)
			search(i + 1)
			chosen = chosen[:len(chosen)-1]
			delete(providers, candidate.Identity.Provider)
			delete(families, candidate.Identity.Family)
			delete(weights, candidate.Identity.WeightsRevision)
		}
	}
	if quorum > 0 {
		search(0)
	}
	sort.Slice(groups, func(i, j int) bool {
		aHealth, bHealth := auditQuorumGroupWorstHealth(decisions, groups[i]), auditQuorumGroupWorstHealth(decisions, groups[j])
		if aHealth != bHealth {
			return aHealth < bHealth
		}
		aPriority, bPriority := auditQuorumGroupPriority(decisions, groups[i]), auditQuorumGroupPriority(decisions, groups[j])
		if aPriority != bPriority {
			return aPriority < bPriority
		}
		aCost, bCost := auditQuorumGroupCost(decisions, groups[i]), auditQuorumGroupCost(decisions, groups[j])
		if aCost != bCost {
			return aCost < bCost
		}
		return auditQuorumGroupKey(decisions, groups[i]) < auditQuorumGroupKey(decisions, groups[j])
	})
	return groups
}

func auditQuorumGroupWorstHealth(decisions []AuditRouteCandidateDecision, indexes []int) int {
	worst := 0
	for _, index := range indexes {
		if rank := auditHealthRank(decisions[index].ProviderHealth); rank > worst {
			worst = rank
		}
	}
	return worst
}

func auditQuorumGroupPriority(decisions []AuditRouteCandidateDecision, indexes []int) int64 {
	var total int64
	for _, index := range indexes {
		total += int64(decisions[index].Priority)
	}
	return total
}

func auditQuorumGroupCost(decisions []AuditRouteCandidateDecision, indexes []int) int64 {
	var total int64
	for _, index := range indexes {
		cost := decisions[index].EstimatedCostMicrosUSD
		if total > math.MaxInt64-cost {
			return math.MaxInt64
		}
		total += cost
	}
	return total
}

func auditQuorumGroupKey(decisions []AuditRouteCandidateDecision, indexes []int) string {
	ids := make([]string, 0, len(indexes))
	for _, index := range indexes {
		ids = append(ids, decisions[index].CandidateID)
	}
	return strings.Join(ids, "\x00")
}

func auditHealthRank(status AuditProviderHealthStatus) int {
	if status == AuditProviderHealthy {
		return 0
	}
	if status == AuditProviderDegraded {
		return 1
	}
	return 2
}

func auditNoRouteReason(decisions []AuditRouteCandidateDecision, diversifiedQuorum bool) AuditRouteNoRouteReason {
	if allAuditRouteSkippedFor(decisions, AuditSkipProviderUnhealthy, AuditSkipProviderHealthUnknown) {
		return AuditNoRouteNoHealthyProvider
	}
	if allAuditRouteSkippedFor(decisions, AuditSkipCapacitySaturated, AuditSkipCapacityUnknown) {
		return AuditNoRouteNoCapacity
	}
	if allAuditRouteSkippedFor(decisions, AuditSkipCooldownActive, AuditSkipCooldownUnknown) {
		return AuditNoRouteCooldown
	}
	if allAuditRouteSkippedFor(decisions, AuditSkipBelowTierFloor) {
		return AuditNoRouteTierFloor
	}
	if allAuditRouteSkippedFor(decisions, AuditSkipBelowEffortFloor) {
		return AuditNoRouteEffortFloor
	}
	if diversifiedQuorum {
		return AuditNoRouteDiversifiedQuorum
	}
	return AuditNoRouteNoIndependent
}

func allAuditRouteSkippedFor(decisions []AuditRouteCandidateDecision, reasons ...AuditRouteSkipReason) bool {
	if len(decisions) == 0 {
		return false
	}
	allowed := make(map[AuditRouteSkipReason]bool, len(reasons))
	for _, reason := range reasons {
		allowed[reason] = true
	}
	for _, decision := range decisions {
		if !allowed[decision.SkipReason] {
			return false
		}
	}
	return true
}
