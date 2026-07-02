package vcacheobserve

import "testing"

func TestApplyProviderActionPlanMutatesOnlyLocalManifestRows(t *testing.T) {
	manifest := LocalProviderManifest{
		Entries: []LocalProviderManifestEntry{
			{Family: "hot", Mode: LocalManifestModeWarm},
			{Family: "old", Mode: LocalManifestModeWarm},
		},
	}
	plan := ProviderActionPlan{
		Schema: ProviderActionSchema,
		Actions: []ProviderAction{
			{Family: "old", Action: "evict_manifest", State: ActionReady, Reason: "cold"},
			{Family: "secret", Action: "no_cache", State: ActionReady, Reason: "not warmable"},
			{Family: "hot", Action: "ride_natural", State: ActionNoop, Reason: "natural traffic"},
		},
	}

	updated, report := ApplyProviderActionPlan(manifest, plan)
	if report.Schema != ProviderActionApplySchema || report.PlanSchema != ProviderActionSchema {
		t.Fatalf("report schema = %q plan=%q", report.Schema, report.PlanSchema)
	}
	if report.Counts.Applied != 2 || report.Counts.NoEffect != 1 || report.Counts.Refused != 0 || report.Counts.Pending != 0 {
		t.Fatalf("counts = %+v, want 2 applied / 1 no-effect", report.Counts)
	}
	if hasLocalProviderFamily(updated, "old") {
		t.Fatalf("old family still present after evict: %+v", updated.Entries)
	}
	secret, ok := localProviderFamily(updated, "secret")
	if !ok || secret.Mode != LocalManifestModeNoCache || secret.Action != "no_cache" {
		t.Fatalf("secret row = %+v/%v, want no_cache row", secret, ok)
	}
	hot, ok := localProviderFamily(updated, "hot")
	if !ok || hot.Mode != LocalManifestModeWarm {
		t.Fatalf("hot row = %+v/%v, want unchanged warm row", hot, ok)
	}
	if report.BeforeFamilies != 2 || report.AfterFamilies != 2 {
		t.Fatalf("family counts before/after = %d/%d, want 2/2", report.BeforeFamilies, report.AfterFamilies)
	}
}

func TestApplyProviderActionPlanDoesNotSelfClaimSpendfulExecution(t *testing.T) {
	manifest := LocalProviderManifest{Entries: []LocalProviderManifestEntry{{Family: "pin", Mode: LocalManifestModeWarm}}}
	plan := ProviderActionPlan{
		Schema: ProviderActionSchema,
		Actions: []ProviderAction{
			{Family: "pin", Action: "heartbeat_pin", State: ActionReady, Reason: "transport witnessed"},
			{Family: "regulated", Action: "explicit_cache", State: ActionGated, Reason: "missing deletion"},
		},
	}

	updated, report := ApplyProviderActionPlan(manifest, plan)
	if report.Counts.Pending != 1 || report.Counts.Refused != 1 || report.Counts.Applied != 0 {
		t.Fatalf("counts = %+v, want pending=1 refused=1 applied=0", report.Counts)
	}
	if report.Rows[0].Outcome != ApplyOutcomePendingProviderTransport {
		t.Fatalf("heartbeat outcome = %+v, want pending provider transport", report.Rows[0])
	}
	if report.Rows[1].Outcome != ApplyOutcomeRefusedGated {
		t.Fatalf("explicit-cache outcome = %+v, want refused gated", report.Rows[1])
	}
	if len(updated.Entries) != 1 || updated.Entries[0].Family != "pin" || updated.Entries[0].Mode != LocalManifestModeWarm {
		t.Fatalf("spendful rows mutated manifest: %+v", updated.Entries)
	}
}

func localProviderFamily(manifest LocalProviderManifest, family string) (LocalProviderManifestEntry, bool) {
	for _, entry := range manifest.Entries {
		if entry.Family == family {
			return entry, true
		}
	}
	return LocalProviderManifestEntry{}, false
}

func hasLocalProviderFamily(manifest LocalProviderManifest, family string) bool {
	_, ok := localProviderFamily(manifest, family)
	return ok
}
