package main

import "fmt"

// serveInkling2MetalSelected applies the run selector to serve, which binds the
// vendored target and fak's device-resident forward: an explicit request for the
// host-driven native forward is refused rather than served under another path's name.
func serveInkling2MetalSelected(path, selector string) (inkling2MetalPath, error) {
	kind, err := inkling2MetalSelect(selector, true, true, inkling2MetalResidentAvailable)
	if err != nil {
		return kind, err
	}
	switch kind {
	case inkling2MetalVendored, inkling2MetalResident:
		return kind, nil
	}
	return kind, &inkling2MetalServeRefusal{Path: path,
		Reason: fmt.Sprintf("%s=%s selects the %s forward, which is not supported by serve yet; unset it or set it to vendored or resident",
			inkling2MetalTargetEnv, selector, inkling2MetalTargetNativeLabel)}
}
