//go:build fak_v41_flashmla_component_witness && (!cuda || !cgo || !cudaarchspecific || !fak_cuda_flashmla_decode)

package model

import "testing"

// The required witness must never pass as "no tests to run" without its device
// profile. Select this test explicitly only in the required-device lane.
// fak-test:runtime integration est=1ms lane=optin
func TestV41FlashQualificationOwnedPhysical(t *testing.T) {
	t.Fatal("required FlashMLA component witness missing cuda+cgo+cudaarchspecific+fak_cuda_flashmla_decode profile; no host fallback")
}
