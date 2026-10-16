package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// V41DeferredStatusKind names the checked V4.1 kernel a deferred status
// record belongs to.
type V41DeferredStatusKind uint8

const (
	V41DeferredRoutedSwiGLU V41DeferredStatusKind = iota + 1
	V41DeferredBF16Publish
)

// V41DeferredStatus is one checked kernel's 16-byte status record, left on the
// device so several kernels can be verified behind one host fence (#13820).
// Record is backend-owned storage; V41ReadDeferred consumes and frees it.
//
// Records > 1 marks a routed id or activation-chunk status: Record holds that
// many consecutive 16-byte records, one per pick, each of Count lanes.
type V41DeferredStatus struct {
	Kind    V41DeferredStatusKind
	Count   int
	Site    uint32
	Records int
	Record  Tensor
}

// V41DeferredStatusBackend records the routed SwiGLU and the ordinary BF16
// publication without their per-call status readbacks. Each deferred call
// issues the same kernel as its synchronous form and returns the same output
// tensor; the output is valid only once its status verdict is nil.
//
// V41ReadDeferred reads every status record and every row behind one host
// fence, frees every record, and returns one verdict per record in order (a
// status with Records n contributes n consecutive verdicts): nil,
// or exactly the error the synchronous operation returns after its own
// readback (*V41RoutedSwiGLUFault, or a *BackendError wrapping the checked
// BF16 arithmetic or protocol error). A non-nil err is a transport or
// submission failure and leaves every verdict unknown. No deferred kernel may
// be replayed on the host.
type V41DeferredStatusBackend interface {
	SupportsV41DeferredStatus(count int) bool
	V41RoutedSwiGLUDeferred(gate, up Tensor, limit, weight float32) (Tensor, V41DeferredStatus, error)
	V41BF16PublishDeferred(input Tensor, site uint32) (Tensor, V41DeferredStatus, error)
	V41ReadDeferred(statuses []V41DeferredStatus, rows []Tensor) ([]error, [][]float32, error)
}

// v41DeferredVerdict decodes one status record into the error the synchronous
// operation returns for the same record.
func v41DeferredVerdict(backend string, status V41DeferredStatus, record []byte) error {
	if len(record) != 16 {
		return v41DeferredProtocolError(backend, status.Kind, "status record is not 16 bytes")
	}
	switch status.Kind {
	case V41DeferredRoutedSwiGLU:
		var words [4]uint32
		for i := range words {
			words[i] = binary.LittleEndian.Uint32(record[4*i:])
		}
		decodeErr := decodeV41RoutedSwiGLUStatus(words, status.Count)
		var fault *V41RoutedSwiGLUFault
		if errors.As(decodeErr, &fault) {
			return fault
		}
		if decodeErr != nil {
			return &BackendError{Backend: backend, Site: "V41RoutedSwiGLU", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: decodeErr.Error(), Recovered: decodeErr}
		}
		return nil
	case V41DeferredBF16Publish:
		if statusErr := decodeV41BF16PublishStatus(record, status.Count, status.Site); statusErr != nil {
			return &BackendError{Backend: backend, Site: "V41BF16Publish", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: statusErr.Error(), Recovered: statusErr}
		}
		return nil
	}
	return v41DeferredProtocolError(backend, status.Kind, "unknown status kind")
}

func v41DeferredProtocolError(backend string, kind V41DeferredStatusKind, reason string) *BackendError {
	message := fmt.Sprintf("deferred V4.1 status kind %d: %s", kind, reason)
	return &BackendError{Backend: backend, Site: "V41ReadDeferred", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: message}
}

// v41DeferredLayout places each region at a 16-byte aligned offset of one
// host image, in order.
func v41DeferredLayout(sizes []int) (offsets []int, total int) {
	offsets = make([]int, len(sizes))
	for i, n := range sizes {
		total = (total + 15) &^ 15
		offsets[i] = total
		total += n
	}
	return offsets, total
}
