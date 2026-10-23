//go:build vulkan && (windows || linux) && cgo

package model

func init() { qwen35HybridQ4KBatchStep = stepBatchQwen35HybridQ4KVulkan }

// Keep the established Vulkan selection and test entrypoint on the shared dispatcher.
func stepBatchQwen35HybridQ4KVulkan(bs *BatchSession, ids []int) ([][]float32, bool) {
	return stepBatchQwen35HybridQ4KIndependent(bs, ids)
}
