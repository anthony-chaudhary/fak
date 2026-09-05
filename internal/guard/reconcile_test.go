package guard

import (
	"slices"
	"strings"
	"testing"
)

func TestReconcileCodexAlias(t *testing.T) {
	catalog := ToolCatalog{
		Version: "v1",
		Harness: "codex",
		Tools:   []string{"exec_command", "read_file"},
	}
	profile := CapabilityProfile{
		Name:         "standard",
		Version:      "v1",
		AllowedTools: []string{"bash", "read_file"},
		KnownAliases: map[string]string{
			"exec_command": "bash",
		},
	}

	res := ReconcileCatalog(catalog, profile)

	if !slices.Contains(res.Recognized, "exec_command") {
		t.Errorf("expected exec_command in Recognized, got %v", res.Recognized)
	}
	if !slices.Contains(res.Recognized, "read_file") {
		t.Errorf("expected read_file in Recognized, got %v", res.Recognized)
	}
	if got, want := res.DriftedAliases["exec_command"], "bash"; got != want {
		t.Errorf("DriftedAliases[exec_command] = %q, want %q", got, want)
	}
	if len(res.Unknown) != 0 {
		t.Errorf("expected 0 unknown tools, got %v", res.Unknown)
	}
	if len(res.Remedies) != 0 {
		t.Errorf("expected 0 remedies, got %v", res.Remedies)
	}
	if !profile.Allows("exec_command") {
		t.Errorf("expected profile.Allows(exec_command) to be true via alias")
	}
}

func TestReconcileDetectUnknownTool(t *testing.T) {
	catalog := ToolCatalog{
		Version: "v1",
		Harness: "claude",
		Tools:   []string{"bash", "deploy_preview"},
	}
	profile := CapabilityProfile{
		Name:         "standard",
		Version:      "v1",
		AllowedTools: []string{"bash"},
	}

	res := ReconcileCatalog(catalog, profile)

	if !slices.Contains(res.Recognized, "bash") {
		t.Errorf("expected bash in Recognized, got %v", res.Recognized)
	}
	if slices.Contains(res.Recognized, "deploy_preview") {
		t.Errorf("deploy_preview should not be in Recognized")
	}
	if len(res.Unknown) != 1 || res.Unknown[0] != "deploy_preview" {
		t.Fatalf("expected Unknown = [deploy_preview], got %v", res.Unknown)
	}

	wantRemedy := "--allow-tool deploy_preview"
	if len(res.Remedies) != 1 || res.Remedies[0] != wantRemedy {
		t.Fatalf("expected Remedies = [%q], got %v", wantRemedy, res.Remedies)
	}
}

func TestReconcileWarningOnDrift(t *testing.T) {
	t.Run("version drift", func(t *testing.T) {
		cat := ToolCatalog{
			Version: "v2.0",
			Tools:   []string{"bash"},
		}
		prof := CapabilityProfile{
			Version:      "v1.0",
			AllowedTools: []string{"bash"},
		}
		res := ReconcileCatalog(cat, prof)
		if res.Warning == "" {
			t.Fatal("expected non-empty warning on version drift, got empty")
		}
		if !strings.Contains(res.Warning, "drifts from profile version") {
			t.Errorf("warning does not mention version drift: %q", res.Warning)
		}
	})

	t.Run("alias name drift", func(t *testing.T) {
		cat := ToolCatalog{
			Version: "v1.0",
			Tools:   []string{"exec_command"},
		}
		prof := CapabilityProfile{
			Version:      "v1.0",
			AllowedTools: []string{"bash"},
			KnownAliases: map[string]string{"exec_command": "bash"},
		}
		res := ReconcileCatalog(cat, prof)
		if res.Warning == "" {
			t.Fatal("expected non-empty warning on alias name drift, got empty")
		}
		if !strings.Contains(res.Warning, "drift from standard profile") {
			t.Errorf("warning does not mention alias drift: %q", res.Warning)
		}
	})

	t.Run("unknown tool name drift", func(t *testing.T) {
		cat := ToolCatalog{
			Version: "v1.0",
			Tools:   []string{"deploy_preview"},
		}
		prof := CapabilityProfile{
			Version:      "v1.0",
			AllowedTools: []string{"bash"},
		}
		res := ReconcileCatalog(cat, prof)
		if res.Warning == "" {
			t.Fatal("expected non-empty warning on unknown tool drift, got empty")
		}
		if !strings.Contains(res.Warning, "unrecognized advertised tools") {
			t.Errorf("warning does not mention unrecognized tools: %q", res.Warning)
		}
	})

	t.Run("no drift", func(t *testing.T) {
		cat := ToolCatalog{
			Version: "v1.0",
			Tools:   []string{"bash"},
		}
		prof := CapabilityProfile{
			Version:      "v1.0",
			AllowedTools: []string{"bash"},
		}
		res := ReconcileCatalog(cat, prof)
		if res.Warning != "" {
			t.Errorf("expected empty warning when no drift, got %q", res.Warning)
		}
	})
}

func TestReconcileRemedyDoesNotAutoAuthorize(t *testing.T) {
	catalog := ToolCatalog{
		Version: "v1",
		Tools:   []string{"deploy_preview"},
	}
	profile := CapabilityProfile{
		Name:         "locked",
		Version:      "v1",
		AllowedTools: []string{"bash"},
	}

	res := ReconcileCatalog(catalog, profile)

	if len(res.Remedies) != 1 || res.Remedies[0] != "--allow-tool deploy_preview" {
		t.Fatalf("expected remedy for deploy_preview, got %v", res.Remedies)
	}

	// Security invariant: existence of an advisory remedy does NOT grant authority.
	if profile.Allows("deploy_preview") {
		t.Errorf("invariant violated: profile.Allows(deploy_preview) must be false")
	}
	if slices.Contains(res.Recognized, "deploy_preview") {
		t.Errorf("invariant violated: deploy_preview must not be in Recognized")
	}
	if !slices.Contains(res.Unknown, "deploy_preview") {
		t.Errorf("deploy_preview must remain classified as Unknown")
	}
	if len(profile.AllowedTools) != 1 || profile.AllowedTools[0] != "bash" {
		t.Errorf("profile.AllowedTools was mutated: %v", profile.AllowedTools)
	}
}
