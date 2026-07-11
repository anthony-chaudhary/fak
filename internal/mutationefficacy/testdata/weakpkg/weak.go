// Package weakpkg is a fixture for the mutation-efficacy probe (#3845): a pure function paired
// with a DELIBERATELY WEAK test. It lives under testdata/ so the module's `./...` build, vet,
// and test wildcards skip it; the probe reaches it only by an explicit path. The weak test
// lets the `>` -> `>=` mutant survive, which is exactly the survivor the end-to-end probe test
// witnesses.
package weakpkg

// IsPositive reports whether n is strictly greater than zero.
func IsPositive(n int) bool { return n > 0 }
