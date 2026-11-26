package main

import "fmt"

// serveInkling2MetalSelected applies the run selector to serve, which binds
// only the vendored target: an explicit request for one of fak's own forwards
// is refused rather than served under the other path's name.
func serveInkling2MetalSelected(path, selector string) error {
	kind, err := inkling2MetalSelect(selector, true, true, true)
	if err != nil {
		return err
	}
	label := inkling2MetalTargetNativeLabel
	switch kind {
	case inkling2MetalVendored:
		return nil
	case inkling2MetalResident:
		label = inkling2MetalTargetResidentLabel
	}
	return &inkling2MetalServeRefusal{Path: path,
		Reason: fmt.Sprintf("%s=%s selects the %s forward, which is not supported by serve yet; unset it or set it to vendored",
			inkling2MetalTargetEnv, selector, label)}
}
