//go:build !cuda || !cgo || !cudaarchspecific || !fak_cuda_flashmla_decode

package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestFlashMLADecodeQualificationMissingProfile(t *testing.T) {
	out, err := QualifyFlashMLADecode(nil, FlashMLADecodeQualificationInput{})
	if !errors.Is(err, ErrFlashMLADecodeQualificationUnavailable) || out.ValuesBF16 != nil || out.LSE != nil {
		t.Fatalf("selected unavailable profile returned output or lost refusal: %+v %v", out, err)
	}
}
