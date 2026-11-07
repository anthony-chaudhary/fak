//go:build amd64

package model

import "testing"

func q8MoEScalarForTest(t *testing.T) {
	t.Helper()
	old := qtier
	qtier = tierScalar
	t.Cleanup(func() { qtier = old })
}
