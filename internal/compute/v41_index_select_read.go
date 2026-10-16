package compute

import (
	"encoding/binary"
	"fmt"
)

// The private method remains bound to the original canonical receiver.
type v41IndexSelectOwnedReader interface {
	v41IndexSelectReadOwned(Backend, Tensor, int, int, int, int) ([]float32, bool, error)
}

// TryV41IndexSelectReadOwned combines status and selected indices into one
// checked transfer. Decline has no device effects; selected failures must not
// replay the ordinary operation or a host selector.
func TryV41IndexSelectReadOwned(backend Backend, scores Tensor, rows, topK, topKBlocks, blockSize int) ([]float32, bool, error) {
	if reader, ok := backend.(v41IndexSelectOwnedReader); ok {
		return reader.v41IndexSelectReadOwned(backend, scores, rows, topK, topKBlocks, blockSize)
	}
	return nil, false, nil
}

func v41IndexSelectOwnedRows(image v41HostReadbackImage, count int) ([]float32, error) {
	word := binary.LittleEndian.Uint32(image.region(0, 4))
	if word != 0 {
		return nil, &BackendError{Backend: "vulkan", Site: "V41IndexSelect", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: fmt.Sprintf("index select refused a NaN score (status %#x)", word)}
	}
	return image.materializeRows([]int{4, 4 * count}, 1)[0], nil
}
