package vulkanregistry

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=1s lane=default
// fak-test:usefulness protects=vulkan-registry-current-export
func TestCurrentMatchesBuilderStampedRegistry(t *testing.T) {
	got, err := Current()
	if err != nil {
		t.Fatal(err)
	}
	stems := computebuild.CurrentVulkanShaderRegistryLatest()
	sort.Strings(stems)
	raw, err := json.Marshal(stems)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if got.ModuleCount != len(stems) || got.SHA256 != hex.EncodeToString(sum[:]) || got.ID == "" || got.ReceiptSchema == "" {
		t.Fatalf("Current() = %+v does not describe the %d-module builder registry", got, len(stems))
	}
	if !strings.HasSuffix(got.ID, fmt.Sprintf(".current%d", got.ModuleCount)) || !strings.HasPrefix(got.ReceiptSchema, "fak.vulkan-build-receipt.v") {
		t.Fatalf("Current() = %s/%s is not a stamped registry identity", got.ReceiptSchema, got.ID)
	}
}
