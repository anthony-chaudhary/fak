//go:build !linux

package treedoctor

import (
	"fmt"
	"os"
)

func samePlacementFilesystem(_, _ string) (bool, error) {
	return false, fmt.Errorf("Go cache filesystem identity is unsupported on this platform")
}

func openPlacementPlan(_ string) (*os.File, error) {
	return nil, fmt.Errorf("Go cache filesystem identity is unsupported on this platform")
}
