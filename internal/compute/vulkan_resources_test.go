package compute

import (
	"os"
	"path/filepath"
	"testing"
)

func TestVulkanReleaseResourcesFollowRelocatedExecutable(t *testing.T) {
	root := t.TempDir()
	root, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	bundle := filepath.Join(root, "relocated release")
	resources := filepath.Join(bundle, "spirv")
	if err := os.MkdirAll(resources, 0755); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(bundle, "fak")
	if err := os.WriteFile(binary, nil, 0755); err != nil {
		t.Fatal(err)
	}
	if got := vulkanSPIRVDir("", binary); got != resources {
		t.Fatalf("relocated bundle: got %q, want %q", got, resources)
	}
	if got := vulkanSPIRVDir("explicit-missing-bundle", binary); got != "explicit-missing-bundle" {
		t.Fatalf("explicit resource override lost: %q", got)
	}
	if got := vulkanSPIRVDir("", filepath.Join(root, "missing", "fak")); got != "" {
		t.Fatalf("missing bundle must remain unavailable: %q", got)
	}
	link := filepath.Join(root, "fak-link")
	if err := os.Symlink(binary, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	if got := vulkanSPIRVDir("", link); got != resources {
		t.Fatalf("symlink bundle: got %q, want %q", got, resources)
	}
}
