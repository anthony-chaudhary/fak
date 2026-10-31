//go:build !cuda || !cgo || !fak_cuda_q4k_qkv_qualification || !fak_cuda_v41_index_select

package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// Ordinary builds do not probe a candidate, stage weights, or change dispatch.
func (s *Session) qwen35QKVQualification(_ int, _ compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor, bool, error) {
	return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, false, nil
}
