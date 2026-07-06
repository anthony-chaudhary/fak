package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
		!strings.Contains(j, `"provider": "groq"`) {
		t.Fatalf("status --json missing schema/provider:\n%s", j)
	}
}
