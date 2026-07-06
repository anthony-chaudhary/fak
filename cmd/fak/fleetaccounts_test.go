package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func TestScrubFleetAccountWaveSecretsDropsOAuthToken(t *testing.T) {
	token := "sk-ant-oat01-fixture"
	wave := fleetaccounts.WaveResult{
		OK: true,
		Lanes: []fleetaccounts.WaveLane{{
			Resolved: fleetaccounts.Resolved{OK: true, OAuthToken: &token, Tag: "day26"},
			Pool:     "uuid:day26",
		}},
	}
	scrubbed := scrubFleetAccountWaveSecrets(wave)
	raw, err := json.Marshal(scrubbed)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "oauth_token") || strings.Contains(string(raw), token) {
		t.Fatalf("scrubbed wave leaked oauth token: %s", raw)
	}
}

func TestRunFleetAccountsStatusProviderTierFilter(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	cfg := filepath.Join(root, "cfg")
	reg := filepath.Join(root, "reg")
	groq := filepath.Join(cfg, "opencode-groq-kimi")
	for _, dir := range []string{home, groq, reg} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(groq, "opencode.json"),
		[]byte(`{"model":"groq/moonshotai/kimi-k2.6"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", cfg)
	t.Setenv("FLEET_REG_DIR", reg)
	t.Setenv("FLEET_POLICY_PATH", filepath.Join(root, "missing-policy.json"))

	var out, errb strings.Builder
	rc := runFleetAccounts(&out, &errb, []string{"status", "--provider", "groq", "--tier", "1"})
	if rc != 0 {
		t.Fatalf("status rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"fleet account status (provider=groq tier=t1)",
		"rollups by provider+tier",
		"provider=groq tier=t1",
		"opencode-groq-kimi",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("status output missing %q:\n%s", want, got)
		}
	}

	out.Reset()
	errb.Reset()
	rc = runFleetAccounts(&out, &errb, []string{"status", "--provider", "groq", "--json"})
	if rc != 0 {
		t.Fatalf("status --json rc=%d stderr=%s", rc, errb.String())
	}
	if j := out.String(); !strings.Contains(j, `"schema": "fleet-account-status/1"`) ||
		!strings.Contains(j, `"provider": "groq"`) ||
		!strings.Contains(j, `"node": "local"`) ||
		!strings.Contains(j, `"generated_at"`) {
		t.Fatalf("status --json missing schema/provider:\n%s", j)
	}
}

func TestRunFleetAccountsStatusGlobalSnapshots(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	cfg := filepath.Join(root, "cfg")
	reg := filepath.Join(root, "reg")
	for _, dir := range []string{home, cfg, reg} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", cfg)
	t.Setenv("FLEET_REG_DIR", reg)
	t.Setenv("FLEET_POLICY_PATH", filepath.Join(root, "missing-policy.json"))

	fresh := filepath.Join(root, "fresh.json")
	stale := filepath.Join(root, "stale.json")
	writeStatusSnapshotForCLITest(t, fresh, "node-a", time.Now().UTC().Add(-5*time.Minute),
		statusAccountForCLITest("node-a", "groq-a", 1, "ready", 1, 1, 0))
	writeStatusSnapshotForCLITest(t, stale, "node-b", time.Now().UTC().Add(-2*time.Hour),
		statusAccountForCLITest("node-b", "groq-b", 1, "ready", 2, 2, 0))

	var out, errb strings.Builder
	rc := runFleetAccounts(&out, &errb, []string{
		"status", "--snapshot", fresh, "--snapshot", stale,
		"--provider", "groq", "--tier", "1", "--json",
	})
	if rc != 0 {
		t.Fatalf("global status rc=%d stderr=%s", rc, errb.String())
	}
	var rep fleetaccounts.GlobalStatusReport
	if err := json.Unmarshal([]byte(out.String()), &rep); err != nil {
		t.Fatalf("decode global status: %v\n%s", err, out.String())
	}
	if rep.Schema != fleetaccounts.GlobalStatusReportSchema {
		t.Fatalf("schema = %q", rep.Schema)
	}
	if rep.Totals.FreeSlots != 1 || rep.StaleTotals.FreeSlots != 2 {
		t.Fatalf("free/stale slots = %d/%d, want 1/2; report=%+v",
			rep.Totals.FreeSlots, rep.StaleTotals.FreeSlots, rep)
	}
	if !strings.Contains(strings.Join(rep.Warnings, "\n"), "stale node node-b excluded") {
		t.Fatalf("warnings = %+v, want stale exclusion", rep.Warnings)
	}

	out.Reset()
	errb.Reset()
	rc = runFleetAccounts(&out, &errb, []string{
		"status", "--snapshot", fresh, "--snapshot", stale,
		"--provider", "groq", "--tier", "1", "--include-stale", "--json",
	})
	if rc != 0 {
		t.Fatalf("global status include-stale rc=%d stderr=%s", rc, errb.String())
	}
	rep = fleetaccounts.GlobalStatusReport{}
	if err := json.Unmarshal([]byte(out.String()), &rep); err != nil {
		t.Fatalf("decode include-stale global status: %v\n%s", err, out.String())
	}
	if rep.Totals.FreeSlots != 3 || !rep.IncludeStale {
		t.Fatalf("include-stale totals = %+v include=%v, want 3 free included", rep.Totals, rep.IncludeStale)
	}
}

func writeStatusSnapshotForCLITest(t *testing.T, path, node string, generatedAt time.Time, accounts ...fleetaccounts.StatusAccount) {
	t.Helper()
	rep := fleetaccounts.StatusReport{
		Schema:      fleetaccounts.StatusReportSchema,
		Node:        node,
		GeneratedAt: generatedAt.UTC().Format(time.RFC3339),
		Accounts:    accounts,
	}
	data, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func statusAccountForCLITest(node, tag string, tier int, state string, cap, free, blocked int) fleetaccounts.StatusAccount {
	return fleetaccounts.StatusAccount{
		Node:            node,
		Account:         "opencode-" + tag,
		Tag:             tag,
		Product:         "opencode",
		Provider:        "groq",
		ModelTier:       intPtrForCLITest(tier),
		Model:           "groq/moonshotai/kimi-k2.6",
		Kind:            string(fleetaccounts.KindWorker),
		State:           state,
		Pool:            "dir:opencode-" + tag,
		CapacityCounted: true,
		SessionCap:      cap,
		FreeSlots:       free,
		BlockedSlots:    blocked,
		Reason:          "fixture",
	}
}

func intPtrForCLITest(v int) *int { return &v }
