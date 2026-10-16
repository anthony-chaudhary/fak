//go:build vulkan && (windows || linux) && cgo

package compute

import "testing"

type vulkanIndexerReadWrapper struct {
	*vulkanBackend
	values []float32
	calls  int
}

func (b *vulkanIndexerReadWrapper) Read(Tensor) []float32 { b.calls++; return b.values }

// fak-test:runtime fast est=5ms lane=default
func TestVulkanIndexerReadHostAndWrapperDefensive(t *testing.T) {
	v := &vulkanBackend{}
	host := []float32{1, 2}
	tensor := NewF32(Default(), []int{2}, host)
	got, ok := ReadV41IndexerScores(v, tensor, 2)
	if !ok || &got[0] == &host[0] {
		t.Fatal("host storage escaped")
	}
	wrapper := &vulkanIndexerReadWrapper{vulkanBackend: v, values: []float32{3, 4}}
	got, ok = ReadV41IndexerScores(wrapper, tensor, 2)
	if !ok || wrapper.calls != 1 || got[0] != 3 || &got[0] == &wrapper.values[0] {
		t.Fatal("wrapper override bypassed")
	}
}
