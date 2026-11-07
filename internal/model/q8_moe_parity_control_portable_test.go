//go:build (!amd64 && !arm64) || (fakaccel && darwin && arm64 && cgo)

package model

import "testing"

func q8MoEScalarForTest(t *testing.T) { t.Helper() }
