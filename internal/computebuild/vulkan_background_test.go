package computebuild

import (
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms
func TestVulkanBackgroundGitSpawns(t *testing.T) {
	src, err := os.ReadFile("vulkan.go")
	if err != nil {
		t.Fatal(err)
	}
	if got := windowgate.GoExecCandidates("internal/computebuild/vulkan.go", string(src)); len(got) != 0 {
		t.Fatalf("background spawn regression: %v", got)
	}
	unconfigured := strings.ReplaceAll(string(src), "sysproc.ConfigureBackground(cmd)", "")
	if got := windowgate.GoExecCandidates("internal/computebuild/vulkan.go", unconfigured); len(got) != 2 {
		t.Fatalf("scanner must detect both unconfigured Git spawns, got %v", got)
	}
}
