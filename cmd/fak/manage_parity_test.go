package main

import (
	"encoding/json"
	"testing"
)

func TestManageParityPacketCoversPopularHarnessesAndBoundaries(t *testing.T) {
	packet := buildComparisonReport("/tmp/fak-manage-parity")
	if packet.Verdict != "PASS" {
		t.Fatalf("parity verdict = %s: %+v", packet.Verdict, packet)
	}
	if len(packet.Cases) != 3 {
		t.Fatalf("cases = %d, want 3", len(packet.Cases))
	}
	seen := map[string]bool{}
	seenPlatform := map[string]bool{}
	seenSeparator := map[bool]bool{}
	seenAlias := false
	for _, row := range packet.Cases {
		seen[row.Manage.Harness] = true
		seenPlatform[row.Manage.Platform] = true
		seenSeparator[row.Manage.Separator] = true
		seenAlias = seenAlias || row.Manage.Invocation == "m"
		if row.Verdict != "PASS" || !sameLaunchContract(row.Manage, row.Legacy) {
			t.Fatalf("case %s diverged: %+v", row.Name, row)
		}
		if row.Manage.Provider == "" || row.Manage.BaseURL == "" || row.Manage.Policy == "" || len(row.Manage.ChildArgv) == 0 {
			t.Fatalf("case %s omitted launch contract: %+v", row.Name, row.Manage)
		}
	}
	for _, harness := range []string{"claude", "codex", "gemini"} {
		if !seen[harness] {
			t.Errorf("missing harness %s", harness)
		}
	}
	if !seenPlatform["windows"] || !seenPlatform["posix"] || !seenSeparator[true] || !seenSeparator[false] || !seenAlias {
		t.Fatalf("argv/alias coverage incomplete: platform=%v separator=%v alias=%v", seenPlatform, seenSeparator, seenAlias)
	}
	if !packet.Cases[0].Manage.Hooks.Settings {
		t.Fatal("Claude native hook posture was not captured")
	}
	if !packet.OperatorProbe.Routed || packet.OperatorProbe.ListenerMade || packet.OperatorProbe.Verdict != "PASS" {
		t.Fatalf("operator probe failed: %+v", packet.OperatorProbe)
	}
	if packet.ExternalModel {
		t.Fatal("dry-run parity packet claimed external model traffic")
	}
	if _, err := json.Marshal(packet); err != nil {
		t.Fatalf("packet is not machine readable: %v", err)
	}
}

func TestManageParityDetectsSemanticDrift(t *testing.T) {
	packet := buildComparisonReport("/tmp/fak-manage-parity")
	managed, legacy := packet.Cases[0].Manage, packet.Cases[0].Legacy
	managed.Provider = "openai"
	if sameLaunchContract(managed, legacy) {
		t.Fatal("provider drift was accepted")
	}
}
