//go:build arm64 && !(fakaccel && darwin && cgo)

package model

import "testing"

func q8MoEScalarForTest(t *testing.T) {
	t.Helper()
	oldTier, oldDot := qkernelTier, neonDot
	qkernelTier, neonDot = tierScalar, false
	t.Cleanup(func() { qkernelTier, neonDot = oldTier, oldDot })
}
