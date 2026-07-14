//go:build !windows

package main

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/hostfault"
	"time"
)

func gatherHostCrashEvents(time.Duration) ([]hostfault.ApplicationError1000, error) {
	return nil, fmt.Errorf("Windows Application Event Log unavailable; use --fixture for replay")
}
