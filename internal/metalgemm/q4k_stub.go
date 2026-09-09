//go:build !darwin || !arm64 || !cgo

package metalgemm

func UploadQ6KGoOwned(raw []byte, out, in int) *Q6KWeight { return nil }
