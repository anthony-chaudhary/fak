//go:build !(darwin && arm64 && cgo)

package model

func newQwen35ProjectionBatcher(Qwen35MetalBatchMode) qwen35ProjectionBatcher {
	return nil
}
