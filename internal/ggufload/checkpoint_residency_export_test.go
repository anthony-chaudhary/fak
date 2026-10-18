package ggufload

import "testing"

// CheckpointResidencyFixtureForTest exposes only the existing test fixture to
// the external HTTP integration tests; production exports remain unchanged.
func CheckpointResidencyFixtureForTest(t *testing.T) string {
	return writeDeepSeek41ExpertHALFile(t)
}
