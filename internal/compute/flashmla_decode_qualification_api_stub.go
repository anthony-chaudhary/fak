//go:build !cuda || !cgo || !cudaarchspecific || !fak_cuda_flashmla_decode

package compute

// QualifyFlashMLADecode fails closed when the explicitly selected qualification
// profile is absent. In particular, it never silently uses the F32 oracle.
func QualifyFlashMLADecode(b Backend, x FlashMLADecodeQualificationInput) (FlashMLADecodeQualificationOutput, error) {
	return FlashMLADecodeQualificationOutput{}, ErrFlashMLADecodeQualificationUnavailable
}
