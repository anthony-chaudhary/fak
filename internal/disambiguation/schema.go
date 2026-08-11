package disambiguation

// SchemaDescriptor is the public, machine-readable summary of the pinned wire
// contract. Required uses JSON paths so callers can display or audit the
// contract without duplicating the Go validator.
type SchemaDescriptor struct {
	Schema            string   `json:"schema"`
	Compatibility     string   `json:"compatibility"`
	UnknownFields     string   `json:"unknown_fields"`
	TrailingValues    string   `json:"trailing_values"`
	Aliases           string   `json:"aliases"`
	Required          []string `json:"required"`
	FreshnessVerdicts []string `json:"freshness_verdicts"`
	LifecycleClasses  []string `json:"lifecycle_classes"`
	Rollouts          []string `json:"rollouts"`
}

// Descriptor returns the deterministic v1 contract description.
func Descriptor() SchemaDescriptor {
	return SchemaDescriptor{
		Schema:         EntrySchemaVersion,
		Compatibility:  "exact-version",
		UnknownFields:  "reject",
		TrailingValues: "reject",
		Aliases:        "required-array-may-be-empty",
		Required: []string{
			"schema", "identity", "identity.canonical_term", "identity.aliases",
			"definition", "contrasts", "contrasts[].canonical_term", "contrasts[].explanation",
			"scope", "scope.kind", "scope.value", "owner", "owner.leaf", "owner.lane",
			"sources", "sources[].kind", "sources[].locator", "sources[].revision",
			"freshness", "freshness.verdict", "freshness.reason_code", "freshness.checked_at", "freshness.probe",
			"lifecycle", "lifecycle.class", "lifecycle.rollout",
		},
		FreshnessVerdicts: []string{string(FreshnessFresh), string(FreshnessStale), string(FreshnessUnknown), string(FreshnessInvalid)},
		LifecycleClasses:  []string{string(LifecycleCurrent), string(LifecycleVersioned), string(LifecycleResearch), string(LifecycleArchived)},
		Rollouts:          []string{string(RolloutOff), string(RolloutShadow), string(RolloutOn)},
	}
}
