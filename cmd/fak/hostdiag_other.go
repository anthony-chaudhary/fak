//go:build !windows

package main

import (
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostdiag"
)

func gatherHostdiagEvents(time.Duration) ([]hostdiag.ResourceEvent, error) {
	return nil, fmt.Errorf("Windows resource event logs unavailable; use --fixture for replay")
}
