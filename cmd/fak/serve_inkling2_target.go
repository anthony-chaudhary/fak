package main

import "fmt"

// serveInkling2MetalSelected applies the run selector to serve, which binds
// only the vendored target: an explicit request for fak's own forward is
// refused rather than served under the other path's name.
func serveInkling2MetalSelected(path, selector string) error {
	vendored, err := inkling2MetalUseVendored(selector, true, true)
	if err != nil {
		return err
	}
	if !vendored {
		return &inkling2MetalServeRefusal{Path: path,
			Reason: fmt.Sprintf("%s=native selects the %s forward, which is not supported by serve yet; unset it or set it to vendored",
				inkling2MetalTargetEnv, inkling2MetalTargetNativeLabel)}
	}
	return nil
}
