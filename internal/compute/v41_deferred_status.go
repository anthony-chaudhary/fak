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
	V41DeferredF32Finite
)

// V41DeferredStatus is one checked kernel's 16-byte status record, left on the
// device so several kernels can be verified behind one host fence (#13820).
// Record is backend-owned storage; V41ReadDeferred consumes and frees it.
// Raw F32 finite checks have no output tensor and contribute exactly one record.
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
// publication without their per-call status readbacks. Those two calls issue
// the same kernel as their synchronous forms and return an output tensor;
// the output is valid only once its status verdict is nil. The reader also
// accepts finite-only records, which have no output tensor.
//
// V41ReadDeferred reads every status record and every row behind one host
// fence, frees every record, and returns one verdict per record in order (a
// status with Records n contributes n consecutive verdicts): nil,
// or exactly the error the synchronous operation returns after its own
// readback (*V41RoutedSwiGLUFault, or a *BackendError wrapping the checked
// BF16 or raw-F32 arithmetic/protocol error). A non-nil err is a transport or
// submission failure and leaves every verdict unknown. No deferred kernel may
// be replayed on the host.
type V41DeferredStatusBackend interface {
	SupportsV41DeferredStatus(count int) bool
	V41RoutedSwiGLUDeferred(gate, up Tensor, limit, weight float32) (Tensor, V41DeferredStatus, error)
	V41BF16PublishDeferred(input Tensor, site uint32) (Tensor, V41DeferredStatus, error)
	V41ReadDeferred(statuses []V41DeferredStatus, rows []Tensor) ([]error, [][]float32, error)
}

// v41DeferredVerdict retains the Vulkan attribution used by existing callers.
// Other backends must attribute decodeV41DeferredStatus's causes themselves.
func v41DeferredVerdict(backend string, status V41DeferredStatus, record []byte) error {
	decodeErr := decodeV41DeferredStatus(status, record)
	if decodeErr == nil {
		return nil
	}
	var fault *V41RoutedSwiGLUFault
	if errors.As(decodeErr, &fault) {
		return fault
	}
	var protocol *v41DeferredStatusProtocolError
	if errors.As(decodeErr, &protocol) {
		return v41DeferredProtocolError(backend, protocol.kind, protocol.reason)
	}
	site := "V41RoutedSwiGLU"
	switch status.Kind {
	case V41DeferredF32Finite:
		site = "V41F32Finite"
	case V41DeferredBF16Publish:
		site = "V41BF16Publish"
	}
	return &BackendError{Backend: backend, Site: site, Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: decodeErr.Error(), Recovered: decodeErr}
}

// decodeV41DeferredStatus returns an issued record's arithmetic or protocol
// cause without backend attribution. The caller must establish completion and
// freshness before decoding; transport failures do not supply a known verdict.
func decodeV41DeferredStatus(status V41DeferredStatus, record []byte) error {
	if len(record) != 16 {
		return &v41DeferredStatusProtocolError{status.Kind, "status record is not 16 bytes"}
	}
	switch status.Kind {
	case V41DeferredRoutedSwiGLU:
		var words [4]uint32
		for i := range words {
			words[i] = binary.LittleEndian.Uint32(record[4*i:])
		}
		return decodeV41RoutedSwiGLUStatus(words, status.Count)
	case V41DeferredF32Finite:
		if status.Records < 0 || status.Records > 1 {
			return &v41DeferredStatusProtocolError{status.Kind, "raw F32 status must have one record"}
		}
		return decodeV41F32FiniteStatus(record, status.Count, status.Site)
	case V41DeferredBF16Publish:
		return decodeV41BF16PublishStatus(record, status.Count, status.Site)
	}
	return &v41DeferredStatusProtocolError{status.Kind, "unknown status kind"}
}

// Envelope failures have no operation-specific arithmetic cause. Keep them
// distinct so the compatibility wrapper retains its original error chain.
type v41DeferredStatusProtocolError struct {
	kind   V41DeferredStatusKind
	reason string
}

func (e *v41DeferredStatusProtocolError) Error() string {
	return fmt.Sprintf("deferred V4.1 status kind %d: %s", e.kind, e.reason)
}

func v41DeferredProtocolError(backend string, kind V41DeferredStatusKind, reason string) *BackendError {
	message := (&v41DeferredStatusProtocolError{kind, reason}).Error()
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
