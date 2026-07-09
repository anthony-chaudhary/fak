//go:build !windows

package windowgate

import "fmt"

// ReadVersionInfo is unavailable off Windows: PE version resources are a Windows
// facility and no ConPTY exists to preflight. The pure resolution, comparison and
// verdict logic in conptyver.go still builds and tests here — callers inject a
// reader — so the regression witness runs on every platform in CI.
func ReadVersionInfo(path string) (ConPTYVersionInfo, error) {
	return ConPTYVersionInfo{}, fmt.Errorf("%w: %s", ErrFileVersionUnsupported, path)
}
