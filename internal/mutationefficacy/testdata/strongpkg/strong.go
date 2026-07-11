// Package strongpkg is the control fixture for the mutation-efficacy probe (#3845): the same
// pure function as weakpkg, but paired with a STRONG test that checks the boundary and a
// negative, so every operator mutant is killed. It lives under testdata/ so the module's
// `./...` wildcards skip it; the probe reaches it only by an explicit path.
package strongpkg

// IsPositive reports whether n is strictly greater than zero.
func IsPositive(n int) bool { return n > 0 }
