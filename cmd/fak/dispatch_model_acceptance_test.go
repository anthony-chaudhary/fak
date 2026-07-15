package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

func writeDispatchAcceptanceFixture(t *testing.T, mutate func(*modelaccept.Input)) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "examples", "model-acceptance-top3.json"))
	if err != nil {
		t.Fatal(err)
	}
	var in modelaccept.Input
	if err := json.Unmarshal(b, &in); err != nil {
		t.Fatal(err)
	}
	if mutate != nil {
		mutate(&in)
	}
	b, err = json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "acceptance.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestDispatchModelAcceptanceExactTierFold(t *testing.T) {
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	path := writeDispatchAcceptanceFixture(t, nil)
	for _, tc := range []struct {
		name, model, label string
		want               bool
	}{
		{"opus permits t0", "claude-opus-4-8", "tier/T0-required", true},
		{"sonnet refuses t0", "claude-sonnet-4-6", "tier/T0-required", false},
		{"sonnet permits t1", "claude-sonnet-4-6", "tier/T1-required", true},
		{"haiku refuses t1", "claude-haiku-4-5-20251001", "tier/T1-required", false},
		{"haiku permits t2", "claude-haiku-4-5-20251001", "tier/T2-optimal", true},
		{"alias fails closed", "sonnet", "tier/T1-required", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := evaluateDispatchModelAcceptance(path, tc.model, []string{tc.label}, now, "")
			if got.Allowed != tc.want {
				t.Fatalf("decision = %+v, want allowed=%v", got, tc.want)
			}
			if got.Model != tc.model || got.CorpusID == "" || got.RequiredTier < 0 {
				t.Fatalf("missing provenance: %+v", got)
			}
		})
	}
}

func TestDispatchModelAcceptanceFailsClosedAndAuditsOverride(t *testing.T) {
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	stale := writeDispatchAcceptanceFixture(t, nil)
	for _, tc := range []struct{ name, path, model string }{
		{"missing", filepath.Join(t.TempDir(), "missing.json"), "claude-opus-4-8"},
		{"malformed", func() string {
			p := filepath.Join(t.TempDir(), "bad.json")
			_ = os.WriteFile(p, []byte("{"), 0o600)
			return p
		}(), "claude-opus-4-8"},
		{"empty model", writeDispatchAcceptanceFixture(t, nil), ""},
		{"stale", stale, "claude-opus-4-8"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			checkNow := now
			if tc.name == "stale" {
				checkNow = time.Date(2027, 7, 15, 0, 0, 0, 0, time.UTC)
			}
			got := evaluateDispatchModelAcceptance(tc.path, tc.model, []string{"tier/T0-required"}, checkNow, "")
			if got.Allowed || got.Verdict != "HOLD" {
				t.Fatalf("want fail-closed HOLD, got %+v", got)
			}
		})
	}
	got := evaluateDispatchModelAcceptance(stale, "claude-opus-4-8", []string{"tier/T0-required"}, time.Date(2027, 7, 15, 0, 0, 0, 0, time.UTC), "incident 42")
	if !got.Allowed || !got.Override || got.Verdict != "HOLD" {
		t.Fatalf("override not audited: %+v", got)
	}
}

func TestApplyDispatchModelAcceptanceBlocksProviderSeam(t *testing.T) {
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	path := writeDispatchAcceptanceFixture(t, nil)
	payload := map[string]any{"ok": true}
	providerReached := false
	if applyDispatchModelAcceptance(path, "claude-haiku-4-5-20251001", []string{"tier/T0-required"}, now, "", payload) {
		providerReached = true // this branch is the provider-launch continuation in evaluateDispatchTick
	}
	if providerReached {
		t.Fatal("above-tier model reached provider continuation")
	}
	if payload["action"] != "model_acceptance_hold" || payload["verdict"] != "HOLD" || payload["ok"] != false {
		t.Fatalf("runtime refusal payload = %#v", payload)
	}
	d, ok := payload["model_acceptance"].(dispatchAcceptanceDecision)
	if !ok || d.Model == "" || d.CorpusID == "" || d.WitnessedTier == nil || d.RequiredTier != 0 || d.Reason == "" {
		t.Fatalf("runtime refusal lacks provenance: %#v", payload["model_acceptance"])
	}

	payload = map[string]any{"ok": true}
	providerReached = false
	if applyDispatchModelAcceptance(path, "claude-opus-4-8", []string{"tier/T0-required"}, now, "", payload) {
		providerReached = true
	}
	if !providerReached {
		t.Fatalf("in-tier exact model was refused: %#v", payload)
	}
}

func TestApplyDispatchModelAcceptanceUnconfiguredPreservesLegacyPayload(t *testing.T) {
	payload := map[string]any{"ok": true}
	if !applyDispatchModelAcceptance("", "", nil, time.Now(), "", payload) {
		t.Fatal("unconfigured acceptance gate refused legacy dispatch")
	}
	if _, changed := payload["model_acceptance"]; changed {
		t.Fatalf("unconfigured gate changed legacy payload: %#v", payload)
	}
}
