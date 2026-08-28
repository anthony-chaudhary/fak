//go:build !(darwin && arm64 && cgo)

package metalgemm

import "testing"

func TestProjectionGraphStubRefuses(t *testing.T) {
	g, err := BeginProjectionGraph(nil, nil, nil, 1, 32)
	if err == nil {
		t.Fatal("stub unexpectedly available")
	}
	if g != nil {
		t.Fatal("stub returned a graph on refusal")
	}
	if receipt := (GraphReceipt{}); receipt.HostUploadBytes != 0 || receipt.HostReadbackBytes != 0 {
		t.Fatalf("zero receipt transfer bytes = upload %d readback %d", receipt.HostUploadBytes, receipt.HostReadbackBytes)
	}
}
