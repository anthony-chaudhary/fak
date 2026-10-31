//go:build cuda && cgo && cudaarchspecific && fak_cuda_flashmla_decode

package compute

// QualifyFlashMLADecode invokes the actual checked owner. It neither advertises
// production support nor substitutes a host implementation after selection.
func QualifyFlashMLADecode(b Backend, x FlashMLADecodeQualificationInput) (FlashMLADecodeQualificationOutput, error) {
	c, ok := b.(*cudaBackend)
	if !ok || c == nil {
		return FlashMLADecodeQualificationOutput{}, ErrFlashMLADecodeQualificationUnavailable
	}
	return c.qualifyFlashMLADecode(x)
}
