//go:build !darwin || !arm64 || !cgo

package model

// Portable sessions retain the selector for gating tests but never install a native executor.
func (s *Session) initMixedQKVNative() {}
