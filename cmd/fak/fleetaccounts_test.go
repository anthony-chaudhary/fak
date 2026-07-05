package main

import (
	"encoding/json"
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
