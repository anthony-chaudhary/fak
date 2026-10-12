package compute

import "testing"

// fak-test:runtime fast est=5ms lane=default
func TestVulkanOffloadPagesHostVisibleOnlyForSmallIntegratedCarveout(t *testing.T) {
	for _, tc := range []struct {
		name       string
		integrated bool
		carveout   int64
		known      bool
		want       bool
	}{
		{"strix3 512MiB carve-out", true, 512 << 20, true, true},
		{"4GiB carve-out bound", true, 4 << 30, true, true},
		{"strix1 64GiB carve-out keeps device-local pages", true, 64 << 30, true, false},
		{"unknown carve-out keeps device-local pages", true, 0, false, false},
		{"discrete GPU keeps device-local pages", false, 512 << 20, true, false},
	} {
		if got := vulkanOffloadPagesHostVisible(tc.integrated, tc.carveout, tc.known); got != tc.want {
			t.Errorf("%s: vulkanOffloadPagesHostVisible(%t, %d, %t) = %t, want %t", tc.name, tc.integrated, tc.carveout, tc.known, got, tc.want)
		}
	}
}
