//go:build !linux && !darwin

package researcharm

import (
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestDurableCoordinatorUnsupportedPlatformRefuses(t *testing.T) {
	c, err := NewDurableCoordinator(2, filepath.Join(t.TempDir(), "leases.json"))
	if err == nil {
		if c != nil {
			c.Close()
		}
		t.Fatal("unsupported platform admitted durable lease authority")
	}
	if c != nil {
		c.Close()
		t.Fatal("unsupported constructor returned a usable coordinator alongside refusal")
	}
}
