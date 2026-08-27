//go:build !(darwin && arm64 && cgo)

package metalgemm

import "testing"

func TestProjectionGraphStubRefuses(t *testing.T) {
	if _, err := BeginProjectionGraph(nil, nil, nil, 1, 32); err == nil {
		t.Fatal("stub unexpectedly available")
	}
}
