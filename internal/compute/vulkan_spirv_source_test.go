package compute

import (
	"os"
	"strings"
	"testing"
)

// Read the actual transitive source at the production include seam.
func vulkanAdmissionSource(t *testing.T) []byte {
	t.Helper()
	source, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	admission, err := os.ReadFile("vulkan_spirv_admission.h")
	if err != nil {
		t.Fatal(err)
	}
	const include = `#include "vulkan_spirv_admission.h"`
	if strings.Count(string(source), include) != 1 {
		t.Fatal("missing unique production admission include")
	}
	return []byte(strings.Replace(string(source), include, string(admission), 1))
}
