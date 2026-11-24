package routing

import "testing"

func TestClassifyInKernelOOMIsNotBareUnavailable(t *testing.T) {
	body := `{"error":{"code":"in_kernel_oom","message":"kernel allocator exhausted device memory"}}`
	got := Classify(503, body)
	if got.Class != FailureClass("ENGINE_OOM") {
		t.Fatalf("Classify(503, in_kernel_oom).Class = %q, want %q", got.Class, "ENGINE_OOM")
	}
	if got.Class == FailureUnavailable {
		t.Fatalf("Classify(503, in_kernel_oom).Class collapsed to %q", FailureUnavailable)
	}
}
