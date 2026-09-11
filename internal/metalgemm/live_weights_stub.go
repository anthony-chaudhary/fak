//go:build !darwin || !arm64 || !cgo

package metalgemm

// LiveQ6KWeights returns the count of resident live q6_k weights.
// The darwin/arm64/cgo implementation reads the cgo live-count registry;
// on other platforms no Metal kernel is resident, so the count is 0.
func LiveQ6KWeights() int { return 0 }

// LiveQ8Weights returns the count of resident live q8 weights.
// The darwin/arm64/cgo implementation reads the cgo live-count registry;
// on other platforms no Metal kernel is resident, so the count is 0.
func LiveQ8Weights() int { return 0 }
