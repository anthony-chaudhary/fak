package main

// Test-only compatibility for legacy dispatch-structure witnesses. Production
// runtime help intentionally has no devindex dependency.

import "github.com/anthony-chaudhary/fak/internal/devindex"

func helpCatalog() *devindex.Catalog {
	root := devindex.FindRoot(".")
	cat, err := devindex.Load(root)
	if err != nil {
		return nil
	}
	return cat
}
