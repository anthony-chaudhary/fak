package buildwitness

import (
	"strings"
	"testing"
)

func TestLoadVariantManifest(t *testing.T) {
	root := repoRoot(t)
	manifest, path, err := LoadVariantManifest(root)
	if err != nil {
		t.Fatalf("LoadVariantManifest failed: %v", err)
	}
	if path == "" {
		t.Fatal("manifest path is empty")
	}
	if len(manifest.ReleaseTargets) < 5 {
		t.Fatalf("expected at least 5 release targets, got %d", len(manifest.ReleaseTargets))
	}
	if len(manifest.Variants) < 7 {
		t.Fatalf("expected at least 7 variants, got %d", len(manifest.Variants))
	}

	requiredNames := []string{
		"default",
		"darwin-arm64",
		"linux-amd64",
		"windows-amd64",
		"wip_sessionfleet",
		"cuda",
		"cuda-nccl",
		"cuda-windows",
		"vulkan-windows",
		"metal",
	}
	for _, name := range requiredNames {
		v := manifest.FindVariant(name)
		if v == nil {
			t.Errorf("manifest missing required variant %q", name)
		}
	}
}

func TestVariantManifestRequiredFields(t *testing.T) {
	root := repoRoot(t)
	manifest, _, err := LoadVariantManifest(root)
	if err != nil {
		t.Fatalf("LoadVariantManifest: %v", err)
	}

	for _, v := range manifest.Variants {
		if strings.TrimSpace(v.Name) == "" {
			t.Error("variant with empty name")
		}
		if v.CGOEnabled != "0" && v.CGOEnabled != "1" {
			t.Errorf("variant %s has invalid cgo_enabled %q (expected 0 or 1)", v.Name, v.CGOEnabled)
		}
		if strings.TrimSpace(v.DefaultInclusion) == "" {
			t.Errorf("variant %s missing default_inclusion", v.Name)
		}
		if strings.TrimSpace(v.TargetOSArch) == "" {
			t.Errorf("variant %s missing target_os_arch", v.Name)
		}
		if strings.TrimSpace(v.ToolchainRequirements) == "" {
			t.Errorf("variant %s missing toolchain_requirements", v.Name)
		}
		if strings.TrimSpace(v.Gate) == "" {
			t.Errorf("variant %s missing gate", v.Name)
		}
	}
}

func TestPlanCompile(t *testing.T) {
	variant := Variant{
		Name:       "wip_sessionfleet",
		Tags:       "wip_sessionfleet",
		CGOEnabled: "0",
	}
	target := ReleaseTarget{GOOS: "linux", GOARCH: "amd64"}
	plan := PlanCompile("./cmd/fak", variant, target, "/dev/null")

	foundTags := false
	for i, arg := range plan.Command {
		if arg == "-tags" && i+1 < len(plan.Command) && plan.Command[i+1] == "wip_sessionfleet" {
			foundTags = true
		}
	}
	if !foundTags {
		t.Errorf("plan command missing -tags wip_sessionfleet: %v", plan.Command)
	}

	hasCgo0 := false
	hasLinux := false
	for _, env := range plan.Env {
		if env == "CGO_ENABLED=0" {
			hasCgo0 = true
		}
		if env == "GOOS=linux" {
			hasLinux = true
		}
	}
	if !hasCgo0 || !hasLinux {
		t.Errorf("plan env missing required settings: %v", plan.Env)
	}
}

func TestBuildGHAMatrix(t *testing.T) {
	root := repoRoot(t)
	manifest, _, err := LoadVariantManifest(root)
	if err != nil {
		t.Fatalf("LoadVariantManifest: %v", err)
	}

	matrix := BuildGHAMatrix(manifest, true)
	includes := matrix["include"]
	if len(includes) == 0 {
		t.Fatal("empty GHA matrix includes")
	}

	// Ensure default variant is present across release targets
	foundDefaultLinux := false
	foundDefaultWindows := false
	for _, entry := range includes {
		if entry.Variant == "default" && entry.Target == "linux/amd64" {
			foundDefaultLinux = true
			if entry.Advisory {
				t.Errorf("default variant should not be advisory")
			}
		}
		if entry.Variant == "default" && entry.Target == "windows/amd64" {
			foundDefaultWindows = true
		}
		if entry.Variant == "wip_sessionfleet" && !entry.Advisory {
			t.Errorf("wip_sessionfleet should be advisory")
		}
	}
	if !foundDefaultLinux {
		t.Error("matrix missing default for linux/amd64")
	}
	if !foundDefaultWindows {
		t.Error("matrix missing default for windows/amd64")
	}
}
