package compute

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const qwen38KernelProvenance = "testdata/qwen38_kernel_provenance.json"

func TestQwen38KernelProvenanceManifest(t *testing.T) {
	manifest := loadQwen38KernelProvenance(t)
	if got := len(manifest.Upstreams); got != 7 {
		t.Fatalf("upstreams = %d, want 7", got)
	}
	if got := len(manifest.Kernels); got != 1 {
		t.Fatalf("kernels = %d, want one bounded Qwen3.8 seed", got)
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.ValidateTree(root); err != nil {
		t.Fatalf("ValidateTree: %v", err)
	}
}

func TestKernelProvenanceRejectsMissingFields(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*KernelProvenanceManifest)
		want   string
	}{
		{"sha", func(m *KernelProvenanceManifest) { m.Upstreams[0].SHA = "" }, ".sha"},
		{"license", func(m *KernelProvenanceManifest) { m.Upstreams[0].License = "" }, ".license"},
		{"source path", func(m *KernelProvenanceManifest) { m.Kernels[0].SourcePath = "" }, ".source_path"},
		{"destination", func(m *KernelProvenanceManifest) { m.Kernels[0].Destination = "" }, ".destination"},
		{"decision", func(m *KernelProvenanceManifest) { m.Kernels[0].Decision = "" }, ".decision"},
		{"notice", func(m *KernelProvenanceManifest) { m.Upstreams[0].NoticeRule = "" }, ".notice_rule"},
		{"parity witness", func(m *KernelProvenanceManifest) { m.Kernels[0].ParityWitness = "" }, ".parity_witness"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manifest := loadQwen38KernelProvenance(t)
			tt.mutate(&manifest)
			err := manifest.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestKernelProvenanceDistinguishesReuseDecisions(t *testing.T) {
	for _, decision := range []KernelReuseDecision{KernelCopyDirectly, KernelAdapt, KernelDelegate, KernelExclude} {
		t.Run(string(decision), func(t *testing.T) {
			manifest := loadQwen38KernelProvenance(t)
			manifest.Kernels[0].Decision = decision
			if err := manifest.Validate(); err != nil {
				t.Fatalf("Validate(%q): %v", decision, err)
			}
		})
	}
}

func TestKernelProvenanceRejectsRemovedAttribution(t *testing.T) {
	manifest := loadQwen38KernelProvenance(t)
	manifest.Kernels[0].Destination = "kernel.go"
	manifest.Kernels[0].ParityWitness = "kernel_test.go"
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "kernel.go"), []byte("package fixture\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "kernel_test.go"), []byte("package fixture\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := manifest.ValidateTree(root)
	if err == nil || !strings.Contains(err.Error(), "missing declared attribution") {
		t.Fatalf("ValidateTree() error = %v, want removed-attribution refusal", err)
	}
}

func loadQwen38KernelProvenance(t *testing.T) KernelProvenanceManifest {
	t.Helper()
	manifest, err := LoadKernelProvenance(qwen38KernelProvenance)
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}
