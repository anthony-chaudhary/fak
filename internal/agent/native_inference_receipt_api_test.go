package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// This compile-time witness keeps the receipt contract singular across the
// request option and completion result surfaces.
func TestNativeInferenceReceiptAPIIsUnique(t *testing.T) {
	var result Completion
	_ = WithNativeInferenceReceipt(true)
	result.NativeInference = &model.NativeInferenceReceipt{}
	if result.NativeInference == nil {
		t.Fatal("native inference receipt result wiring is missing")
	}
}
