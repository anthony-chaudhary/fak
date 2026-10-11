//go:build !darwin || !arm64 || !cgo

package metalgemm

// Q5KAvailable reports false when the Metal Q5_K pipelines cannot be built.
func Q5KAvailable() bool { return false }
