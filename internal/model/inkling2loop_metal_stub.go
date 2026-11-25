//go:build !darwin || !arm64 || !cgo

package model

// inkling2MetalLinked reports a build that links the native Metal target.
const inkling2MetalLinked = false
