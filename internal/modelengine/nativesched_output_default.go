//go:build !fak_native_output_isolation_qualification

package modelengine

// Unqualified builds retain historical one-slot publication backpressure.
const nativeOutputIsolationEnabled = false
