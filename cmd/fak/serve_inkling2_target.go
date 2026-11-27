package main

import (
	"fmt"
	"strings"
)

// serveInkling2MetalSelected applies the Metal selector to serve, which binds
// fak's device-resident forward: an explicit request for the host-driven native
// forward is refused rather than served under another path's name.
func serveInkling2MetalSelected(path, selector string) (inkling2MetalPath, error) {
	if strings.TrimSpace(selector) == "" {
		if !inkling2MetalResidentAvailable {
			return 0, &inkling2MetalServeRefusal{Path: path,
				Reason: "the device-resident Metal forward is not available in this build; build on darwin/arm64 with cgo enabled"}
		}
		return inkling2MetalResident, nil
	}
	kind, err := inkling2MetalSelect(selector, inkling2MetalResidentAvailable)
	if err != nil {
		return kind, err
	}
	if kind == inkling2MetalResident {
		return kind, nil
	}
	return kind, &inkling2MetalServeRefusal{Path: path,
		Reason: fmt.Sprintf("%s=%s selects the %s forward, which is not supported by serve yet; unset it or set it to resident",
			inkling2MetalTargetEnv, selector, inkling2MetalTargetNativeLabel)}
}
