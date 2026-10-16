package compute

// Only the original canonical backend may combine this publication and read.
// Promoted methods on wrappers cannot replace their ordinary operation or Read.
type v41AttentionBF16OwnedReader interface {
	v41AttentionBF16ReadOwned(Backend, Tensor, uint32) ([]float32, bool, error)
}

// TryV41AttentionBF16ReadOwned returns an independent host row only after the
// closed attention site3/4 verdict succeeds. A decline has no device effects;
// a selected error must never be replayed through the ordinary operation.
func TryV41AttentionBF16ReadOwned(backend Backend, input Tensor, site uint32) ([]float32, bool, error) {
	if reader, ok := backend.(v41AttentionBF16OwnedReader); ok {
		return reader.v41AttentionBF16ReadOwned(backend, input, site)
	}
	return nil, false, nil
}

// Raw result bytes stay private until the unchanged closed-site status decoder
// accepts them. One row owns its backing; no device lifetime escapes this call.
func v41AttentionBF16OwnedRow(image v41HostReadbackImage, count int, site uint32) ([]float32, error) {
	if statusErr := decodeV41AttentionBF16Status(image.region(0, v41BF16CheckedStatusBytes), count, site); statusErr != nil {
		return nil, &BackendError{Backend: "vulkan", Site: "V41AttentionBF16", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: statusErr.Error(), Recovered: statusErr}
	}
	return image.materializeRows([]int{v41BF16CheckedStatusBytes, 4 * count}, 1)[0], nil
}
