//go:build !windows

package main

import (
	"fmt"
	"time"
)

func runSessionConPTY(string, time.Duration) ([]byte, error) {
	return nil, fmt.Errorf("ConPTY witness requires Windows")
}
