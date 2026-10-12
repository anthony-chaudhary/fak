//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime integration est=1s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1 and FAK_VULKAN_SPIRV naming
// a bundle built from the current trusted registry; a skip is no witness. It
// checks that a registry-built bundle binds every V4.1 device capability the
// model's device-only admission requires, including the indexer score whose
// production gate is its dispatch prerequisites (Ref #13668).
func TestVulkanV41RegistryBundleBindsDeviceOps(t *testing.T) {
	v := vk(t)
	dir := os.Getenv("FAK_VULKAN_SPIRV")
	if dir == "" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required registry-bundle witness: FAK_VULKAN_SPIRV is unset")
		}
		t.Skip("FAK_VULKAN_SPIRV unset; bundle census unknown")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var census []string
	for _, entry := range entries {
		if name := entry.Name(); filepath.Ext(name) == ".spv" {
			census = append(census, strings.TrimSuffix(name, ".spv"))
		}
	}
	want := computebuild.CurrentVulkanShaderRegistryLatest()
	slices.Sort(census)
	slices.Sort(want)
	if !slices.Equal(census, want) {
		t.Fatalf("bundle census %d modules is not the current registry (%d)", len(census), len(want))
	}
	for _, c := range []struct {
		name string
		have bool
	}{
		{"V41RouterTopKAvailable", v.V41RouterTopKAvailable()},
		{"V41IndexSelectAvailable", v.V41IndexSelectAvailable()},
		{"SupportsV41MHC", v.SupportsV41MHC()},
		{"SupportsV41TailRoPE", v.SupportsV41TailRoPE()},
		{"V41TailRoPERowsAvailable", v.V41TailRoPERowsAvailable()},
		{"SupportsV41SharedAttention", v.SupportsV41SharedAttention()},
		{"V41EmbeddingGatherAvailable", v.V41EmbeddingGatherAvailable()},
		{"V41Q2KEmbeddingGatherAvailable", v.V41Q2KEmbeddingGatherAvailable()},
		{"SupportsV41ExactRMSNorm(512)", v.SupportsV41ExactRMSNorm(512, 1e-6)},
		{"SupportsV41ExactRMSNorm(128)", v.SupportsV41ExactRMSNorm(128, 1e-6)},
		{"V41RoutedSwiGLUAvailable", v.V41RoutedSwiGLUAvailable()},
		{"SupportsV41EngramGate", v.SupportsV41EngramGate()},
		{"SupportsV41CompressorPool", v.SupportsV41CompressorPool()},
		{"V41TailRoPEExactAvailable", v.V41TailRoPEExactAvailable()},
		{"V41ScaleExactAvailable", v.V41ScaleExactAvailable()},
	} {
		if !c.have {
			t.Errorf("registry-built bundle leaves %s false", c.name)
		}
	}
	if !v.SupportsV41IndexerScore() {
		t.Errorf("registry-built bundle leaves SupportsV41IndexerScore false: %s", v.V41IndexerScoreUnavailableReason())
	}
}
