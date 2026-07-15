package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestGuardAllowOverlayPathsLayersUserThenRepo(t *testing.T) {
	home := t.TempDir()
	repo := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv(guardAllowOverlayEnv, "")
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })

	layers := guardAllowOverlayPaths()
	if len(layers) != 2 || layers[0].Name != "user" || layers[1].Name != "repo" {
		t.Fatalf("layers = %+v, want user then repo", layers)
	}
	if guardAllowOverlayPath() != layers[1].Path {
		t.Fatalf("default write path = %q, want repo %q", guardAllowOverlayPath(), layers[1].Path)
	}
	if guardAllowUserOverlayPath() != layers[0].Path {
		t.Fatalf("user write path = %q, want %q", guardAllowUserOverlayPath(), layers[0].Path)
	}
}

func TestGuardAllowWritePathTargetsRepoAndUser(t *testing.T) {
	home := t.TempDir()
	repo := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv(guardAllowOverlayEnv, "")
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })

	repoPath, err := guardAllowWritePath(false)
	if err != nil {
		t.Fatal(err)
	}
	userPath, err := guardAllowWritePath(true)
	if err != nil {
		t.Fatal(err)
	}
	layers := guardAllowOverlayPaths()
	if repoPath != layers[1].Path || userPath != layers[0].Path {
		t.Fatalf("write targets repo=%q user=%q layers=%+v", repoPath, userPath, layers)
	}
}

func TestGuardAllowOverlayEnvOverrideIsSoleLayer(t *testing.T) {
	override := filepath.Join(t.TempDir(), "allow.json")
	t.Setenv(guardAllowOverlayEnv, override)
	layers := guardAllowOverlayPaths()
	if len(layers) != 1 || layers[0].Name != "env" || layers[0].Path != override || guardAllowOverlayPath() != override {
		t.Fatalf("env layers = %+v write=%q", layers, guardAllowOverlayPath())
	}
}

func TestGuardAllowOverlayLayersAdmitUserAndRepoAtLaunchAndReload(t *testing.T) {
	home := t.TempDir()
	repo := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv(guardAllowOverlayEnv, "")
	t.Setenv(guardDenyOverlayEnv, filepath.Join(t.TempDir(), "deny.json"))
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })

	layers := guardAllowOverlayPaths()
	if err := saveGuardAllowOverlay(layers[0].Path, guardAllowOverlay{Allow: []string{"user_only_tool"}}); err != nil {
		t.Fatal(err)
	}
	if err := saveGuardAllowOverlay(layers[1].Path, guardAllowOverlay{Allow: []string{"repo_only_tool"}}); err != nil {
		t.Fatal(err)
	}

	rt, _, _, _ := loadGuardCapabilityFloor("")
	for _, tool := range []string{"user_only_tool", "repo_only_tool"} {
		if !rt.Adjudicator.Allow[tool] {
			t.Errorf("launch did not admit %s", tool)
		}
	}
	if _, err := guardPolicyReloader("")(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, tool := range []string{"user_only_tool", "repo_only_tool"} {
		v := adjudicator.Default.Adjudicate(context.Background(), guardToolCall(t, tool, map[string]any{}))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("reload %s = %v, want ALLOW", tool, v.Kind)
		}
	}
}

func TestGuardAllowLayerProvenanceRendering(t *testing.T) {
	layers := []guardAllowOverlayLayer{{Name: "user", Path: "home/allow.json"}, {Name: "repo", Path: "repo/allow.json"}}
	var out strings.Builder
	for _, layer := range layers {
		out.WriteString("[" + layer.Name + " layer]\n")
		printGuardAllowOverlay(&out, layer.Path, guardAllowOverlay{Allow: []string{layer.Name + "_tool"}})
	}
	got := out.String()
	for _, want := range []string{"[user layer]", "home/allow.json", "user_tool", "[repo layer]", "repo/allow.json", "repo_tool"} {
		if !strings.Contains(got, want) {
			t.Errorf("render missing %q: %s", want, got)
		}
	}
}
