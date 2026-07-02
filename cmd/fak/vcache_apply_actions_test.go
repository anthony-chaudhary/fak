package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func TestVCacheApplyActionsUpdatesManifestAndReportsLedger(t *testing.T) {
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "manifest.json")
	planPath := filepath.Join(dir, "plan.json")
	mustWriteJSON(t, manifestPath, vcacheobserve.LocalProviderManifest{
		Entries: []vcacheobserve.LocalProviderManifestEntry{
			{Family: "hot", Mode: vcacheobserve.LocalManifestModeWarm},
			{Family: "old", Mode: vcacheobserve.LocalManifestModeWarm},
		},
	})
	mustWriteJSON(t, planPath, vcacheobserve.ProviderActionPlan{
		Schema: vcacheobserve.ProviderActionSchema,
		Actions: []vcacheobserve.ProviderAction{
			{Family: "old", Action: "evict_manifest", State: vcacheobserve.ActionReady, Reason: "cold"},
			{Family: "secret", Action: "no_cache", State: vcacheobserve.ActionReady, Reason: "not warmable"},
			{Family: "pin", Action: "heartbeat_pin", State: vcacheobserve.ActionReady, Reason: "transport witnessed"},
		},
	})

	var stdout, stderr bytes.Buffer
	code := runVCache(&stdout, &stderr, []string{"apply-actions", "--manifest", manifestPath, "--plan", planPath, "--json"})
	if code != 0 {
		t.Fatalf("runVCache apply-actions exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var report vcacheobserve.ProviderActionApplyReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if report.Counts.Applied != 2 || report.Counts.Pending != 1 || report.Counts.Refused != 0 {
		t.Fatalf("report counts = %+v, want applied=2 pending=1", report.Counts)
	}
	var manifest vcacheobserve.LocalProviderManifest
	b, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &manifest); err != nil {
		t.Fatalf("decode manifest: %v\n%s", err, string(b))
	}
	if hasManifestEntry(manifest, "old") {
		t.Fatalf("old family still present after apply: %+v", manifest.Entries)
	}
	secret, ok := manifestEntry(manifest, "secret")
	if !ok || secret.Mode != vcacheobserve.LocalManifestModeNoCache {
		t.Fatalf("secret row = %+v/%v, want no_cache", secret, ok)
	}
	pin, ok := manifestEntry(manifest, "pin")
	if ok && pin.Mode != "" {
		t.Fatalf("spendful heartbeat row unexpectedly wrote manifest entry: %+v", pin)
	}
}

func mustWriteJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func manifestEntry(manifest vcacheobserve.LocalProviderManifest, family string) (vcacheobserve.LocalProviderManifestEntry, bool) {
	for _, entry := range manifest.Entries {
		if entry.Family == family {
			return entry, true
		}
	}
	return vcacheobserve.LocalProviderManifestEntry{}, false
}

func hasManifestEntry(manifest vcacheobserve.LocalProviderManifest, family string) bool {
	_, ok := manifestEntry(manifest, family)
	return ok
}
