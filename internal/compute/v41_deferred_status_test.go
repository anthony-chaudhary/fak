package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"testing"
)

func v41DeferredRecord(words ...uint32) []byte {
	out := make([]byte, 4*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(out[4*i:], w)
	}
	return out
}

// A deferred status record decodes to exactly the error the synchronous
// operation returns after its own readback (#13820).
// fak-test:runtime fast est=50ms lane=default
func TestV41DeferredVerdictMatchesSynchronousDecode(t *testing.T) {
	const none = math.MaxUint32
	swiglu := V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: 8}
	publish := V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: 8, Site: V41BF16PublishRoutedDown}
	for _, c := range []struct {
		name   string
		status V41DeferredStatus
		record []byte
		check  func(error) bool
	}{
		{"swiglu ok", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, none, none), func(err error) bool { return err == nil }},
		{"swiglu activation fault", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, 3, 1), func(err error) bool {
			var fault *V41RoutedSwiGLUFault
			return errors.As(err, &fault) && *fault == V41RoutedSwiGLUFault{Stage: V41RoutedSwiGLUStageActivation, Index: 3}
		}},
		{"swiglu down operand fault", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, none, 6), func(err error) bool {
			var fault *V41RoutedSwiGLUFault
			return errors.As(err, &fault) && *fault == V41RoutedSwiGLUFault{Stage: V41RoutedSwiGLUStageDownOperand, Index: 6}
		}},
		{"swiglu stale magic", swiglu, v41DeferredRecord(none, none, none, none), func(err error) bool {
			var backend *BackendError
			var fault *V41RoutedSwiGLUFault
			return errors.As(err, &backend) && backend.Site == "V41RoutedSwiGLU" && errors.Is(err, ErrVulkanExecutionFailed) && !errors.As(err, &fault)
		}},
		{"publish ok", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedDown, none, 8), func(err error) bool { return err == nil }},
		{"publish arithmetic", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedDown, 5, 8), func(err error) bool {
			var numeric *V41BF16CheckedArithmeticError
			var backend *BackendError
			return errors.As(err, &backend) && backend.Site == "V41BF16Publish" && errors.As(err, &numeric) &&
				*numeric == V41BF16CheckedArithmeticError{Site: V41BF16PublishRoutedDown, Index: 5}
		}},
		{"publish wrong site", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedGateUp, none, 8), func(err error) bool {
			var protocol *V41BF16CheckedProtocolError
			var numeric *V41BF16CheckedArithmeticError
			return errors.As(err, &protocol) && !errors.As(err, &numeric) && errors.Is(err, ErrVulkanExecutionFailed)
		}},
		{"short record", publish, v41DeferredRecord(v41BF16CheckedStatusMagic), func(err error) bool {
			var backend *BackendError
			return errors.As(err, &backend) && backend.Site == "V41ReadDeferred"
		}},
		{"unknown kind", V41DeferredStatus{Count: 8}, v41DeferredRecord(0, 0, 0, 0), func(err error) bool {
			var backend *BackendError
			return errors.As(err, &backend) && backend.Site == "V41ReadDeferred"
		}},
	} {
		if err := v41DeferredVerdict("vulkan", c.status, c.record); !c.check(err) {
			t.Errorf("%s: verdict %v", c.name, err)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41DeferredLayoutAlignsRegions(t *testing.T) {
	offsets, total := v41DeferredLayout([]int{16, 16, 10, 4, 1024})
	if want := []int{0, 16, 32, 48, 64}; !reflect.DeepEqual(offsets, want) || total != 1088 {
		t.Fatalf("offsets=%v total=%d", offsets, total)
	}
}
