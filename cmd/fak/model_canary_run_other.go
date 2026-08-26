//go:build !darwin

package main

import (
	"fmt"
	"runtime"
)

func modelCanaryLiveDependencies() (modelCanaryRunDeps, error) {
	return modelCanaryRunDeps{}, fmt.Errorf("model canary-run live execution requires darwin/arm64; this host is %s/%s", runtime.GOOS, runtime.GOARCH)
}
