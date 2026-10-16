//go:build vulkan && (windows || linux) && cgo

package compute

import "testing"

type v41HostInputBorrowWrapper struct{ *vulkanBackend }

// fak-test:runtime fast est=1ms lane=default
func TestV41HostInputBorrowExactBackend(t *testing.T) {
	v := &vulkanBackend{}
	var absent *vulkanBackend
	if !V41HostInputBorrowSafe(v) {
		t.Fatal("canonical identity declined")
	}
	for _, backend := range []Backend{nil, absent, Default(), &v41HostInputBorrowWrapper{v}} {
		if V41HostInputBorrowSafe(backend) {
			t.Fatalf("noncanonical backend %T admitted", backend)
		}
	}
}
