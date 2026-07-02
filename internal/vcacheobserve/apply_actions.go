package vcacheobserve

import (
	"sort"
	"strings"
)

// LocalProviderManifestSchema is the local, fak-owned warm-routing manifest that
// apply-actions mutates. It is not a provider object and never proves a provider
// cache write happened.
const LocalProviderManifestSchema = "fak.vcache.local-provider-manifest.v1"

// ProviderActionApplySchema is the audit report emitted when a provider action
// plan is applied to the local manifest.
const ProviderActionApplySchema = "fak.vcache.provider-action-apply.v1"

const (
	LocalManifestModeWarm    = "warm"
	LocalManifestModeNoCache = "no_cache"
)

const (
	ApplyOutcomeApplied                  = "applied"
	ApplyOutcomeNoEffect                 = "no_effect"
	ApplyOutcomeRefusedGated             = "refused_gated"
	ApplyOutcomePendingProviderTransport = "pending_provider_transport"
	ApplyOutcomeRefusedUnknown           = "refused_unknown"
)

// LocalProviderManifest records the local routing posture for observed prefix
// families. It is the manifest side that can be changed without provider spend:
// evictions remove rows and no_cache rows force uncached routing.
type LocalProviderManifest struct {
	Schema  string                       `json:"schema"`
	Entries []LocalProviderManifestEntry `json:"entries"`
}

type LocalProviderManifestEntry struct {
	Family string `json:"family"`
	Mode   string `json:"mode"`
	Reason string `json:"reason,omitempty"`
	Action string `json:"action,omitempty"`
}

type ProviderActionApplyReport struct {
	Schema         string                    `json:"schema"`
	PlanSchema     string                    `json:"plan_schema"`
	ManifestSchema string                    `json:"manifest_schema"`
	BeforeFamilies int                       `json:"before_families"`
	AfterFamilies  int                       `json:"after_families"`
	DryRun         bool                      `json:"dry_run,omitempty"`
	Counts         ProviderActionApplyCounts `json:"counts"`
	Rows           []ProviderActionApplyRow  `json:"rows"`
	Manifest       LocalProviderManifest     `json:"manifest"`
	CorrectnessLaw string                    `json:"correctness_law"`
}

type ProviderActionApplyCounts struct {
	Applied   int `json:"applied"`
	NoEffect  int `json:"no_effect"`
	Pending   int `json:"pending"`
	Refused   int `json:"refused"`
	Processed int `json:"processed"`
}

type ProviderActionApplyRow struct {
	Family  string              `json:"family"`
	Action  string              `json:"action"`
	State   ProviderActionState `json:"state"`
	Outcome string              `json:"outcome"`
	Reason  string              `json:"reason"`
}

// ApplyProviderActionPlan applies only the local, no-provider-call consequences
// of a provider action plan. Spendful rows are left pending even when they are
// ready, because readiness is not execution; a host must supply a real provider
// executor before heartbeat_pin or explicit_cache can mutate provider state.
func ApplyProviderActionPlan(manifest LocalProviderManifest, plan ProviderActionPlan) (LocalProviderManifest, ProviderActionApplyReport) {
	manifest = NormalizeLocalProviderManifest(manifest)
	before := len(manifest.Entries)
	rows := make([]ProviderActionApplyRow, 0, len(plan.Actions))
	for _, action := range plan.Actions {
		row := applyProviderActionRow(&manifest, action)
		rows = append(rows, row)
	}
	manifest = NormalizeLocalProviderManifest(manifest)
	report := ProviderActionApplyReport{
		Schema:         ProviderActionApplySchema,
		PlanSchema:     plan.Schema,
		ManifestSchema: manifest.Schema,
		BeforeFamilies: before,
		AfterFamilies:  len(manifest.Entries),
		Rows:           rows,
		Manifest:       manifest,
		CorrectnessLaw: "full uncached prompt remains the correctness path; local manifest edits only affect cache posture",
	}
	for _, row := range rows {
		switch row.Outcome {
		case ApplyOutcomeApplied:
			report.Counts.Applied++
		case ApplyOutcomePendingProviderTransport:
			report.Counts.Pending++
		case ApplyOutcomeRefusedGated, ApplyOutcomeRefusedUnknown:
			report.Counts.Refused++
		default:
			report.Counts.NoEffect++
		}
	}
	report.Counts.Processed = len(rows)
	return manifest, report
}

func applyProviderActionRow(manifest *LocalProviderManifest, action ProviderAction) ProviderActionApplyRow {
	row := ProviderActionApplyRow{
		Family: strings.TrimSpace(action.Family),
		Action: strings.TrimSpace(action.Action),
		State:  action.State,
		Reason: action.Reason,
	}
	if row.Family == "" {
		row.Family = action.Family
	}
	if action.State == ActionGated {
		row.Outcome = ApplyOutcomeRefusedGated
		row.Reason = "action remains gated: " + action.Reason
		return row
	}
	switch action.Action {
	case "evict_manifest":
		if removeLocalProviderFamily(manifest, action.Family) {
			row.Outcome = ApplyOutcomeApplied
			row.Reason = "removed local warm-manifest row; no provider call issued"
			return row
		}
		row.Outcome = ApplyOutcomeNoEffect
		row.Reason = "local warm-manifest row was already absent"
		return row
	case "no_cache":
		if upsertLocalProviderFamily(manifest, LocalProviderManifestEntry{
			Family: action.Family,
			Mode:   LocalManifestModeNoCache,
			Reason: action.Reason,
			Action: action.Action,
		}) {
			row.Outcome = ApplyOutcomeApplied
			row.Reason = "local manifest now routes this family uncached; no provider call issued"
			return row
		}
		row.Outcome = ApplyOutcomeNoEffect
		row.Reason = "local manifest already routed this family uncached"
		return row
	case "ride_natural", "lazy_rebuild":
		row.Outcome = ApplyOutcomeNoEffect
		row.Reason = "planner selected a no-provider-call posture; local manifest unchanged"
		return row
	case "heartbeat_pin", "explicit_cache":
		if action.State == ActionReady {
			row.Outcome = ApplyOutcomePendingProviderTransport
			row.Reason = "ready row still needs a provider executor witness before fak can claim execution"
			return row
		}
		row.Outcome = ApplyOutcomeRefusedGated
		row.Reason = "spendful provider action is not ready"
		return row
	default:
		row.Outcome = ApplyOutcomeRefusedUnknown
		row.Reason = "unknown provider action"
		return row
	}
}

// NormalizeLocalProviderManifest canonicalizes schema, removes empty families,
// deduplicates by family, and sorts rows for deterministic diffs.
func NormalizeLocalProviderManifest(manifest LocalProviderManifest) LocalProviderManifest {
	if strings.TrimSpace(manifest.Schema) == "" {
		manifest.Schema = LocalProviderManifestSchema
	}
	byFamily := map[string]LocalProviderManifestEntry{}
	for _, entry := range manifest.Entries {
		entry.Family = strings.TrimSpace(entry.Family)
		entry.Mode = strings.TrimSpace(entry.Mode)
		entry.Reason = strings.TrimSpace(entry.Reason)
		entry.Action = strings.TrimSpace(entry.Action)
		if entry.Family == "" {
			continue
		}
		if entry.Mode == "" {
			entry.Mode = LocalManifestModeWarm
		}
		byFamily[entry.Family] = entry
	}
	manifest.Entries = manifest.Entries[:0]
	for _, entry := range byFamily {
		manifest.Entries = append(manifest.Entries, entry)
	}
	sort.SliceStable(manifest.Entries, func(i, j int) bool {
		return manifest.Entries[i].Family < manifest.Entries[j].Family
	})
	return manifest
}

func removeLocalProviderFamily(manifest *LocalProviderManifest, family string) bool {
	family = strings.TrimSpace(family)
	for i, entry := range manifest.Entries {
		if entry.Family == family {
			manifest.Entries = append(manifest.Entries[:i], manifest.Entries[i+1:]...)
			return true
		}
	}
	return false
}

func upsertLocalProviderFamily(manifest *LocalProviderManifest, next LocalProviderManifestEntry) bool {
	next.Family = strings.TrimSpace(next.Family)
	next.Mode = strings.TrimSpace(next.Mode)
	next.Reason = strings.TrimSpace(next.Reason)
	next.Action = strings.TrimSpace(next.Action)
	if next.Family == "" {
		return false
	}
	if next.Mode == "" {
		next.Mode = LocalManifestModeWarm
	}
	for i, entry := range manifest.Entries {
		if entry.Family != next.Family {
			continue
		}
		if entry == next {
			return false
		}
		manifest.Entries[i] = next
		return true
	}
	manifest.Entries = append(manifest.Entries, next)
	return true
}
