//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// TestVulkanV41RoutedFusedGateUpShapeAdmission uses metadata-only tensors.
// Invalid ids keep even the matching-shape control before the first native call.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanV41RoutedFusedGateUpShapeAdmission(t *testing.T) {
	v := &vulkanBackend{}
	for _, tc := range []struct {
		name       string
		upRows     int
		upIn       int
		upDtype    Dtype
		wantReason string
	}{
		{"fewer_up_rows", 128, 512, Q2_K, "gate and up output rows differ"},
		{"more_up_rows", 512, 512, Q2_K, "gate and up output rows differ"},
		{"matching_rows_reach_ids", 256, 512, Q2_K, "routed ids must be"},
		{"input_width_refusal", 256, 256, Q2_K, "gate and up input widths differ"},
		{"input_width_refusal_precedes_rows", 128, 256, Q2_K, "gate and up input widths differ"},
		{"unsupported_up_dtype", 256, 512, Q3_K, "is not served by this device"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gates := []Tensor{{Dtype: Q2_K, Shape: []int{256, 512}}}
			ups := []Tensor{{Dtype: tc.upDtype, Shape: []int{tc.upRows, tc.upIn}}}
			out, status, err := v.V41RoutedGateUpSwiGLUDeferred(gates, ups, Tensor{}, Tensor{}, 1, 10)
			var backend *BackendError
			if !errors.As(err, &backend) || !errors.Is(err, ErrVulkanInvalidGeometry) {
				t.Fatalf("error = %v, want typed invalid geometry", err)
			}
			if backend.Site != "V41RoutedSwiGLU" || !strings.Contains(backend.Message, tc.wantReason) {
				t.Fatalf("error = %+v, want SwiGLU refusal containing %q", backend, tc.wantReason)
			}
			if !reflect.DeepEqual(out, Tensor{}) || !reflect.DeepEqual(status, V41DeferredStatus{}) {
				t.Fatalf("refused metadata returned output or status: %+v, %+v", out, status)
			}
		})
	}
}
