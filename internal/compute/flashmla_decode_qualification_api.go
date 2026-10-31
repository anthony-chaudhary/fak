package compute

import "errors"

// FlashMLADecodeQualificationInput is the explicit host-staged qualification
// operand, not a production attention capability or a device-resident cache.
type FlashMLADecodeQualificationInput = flashMLADecodeInput

// FlashMLADecodeQualificationOutput retains the donor's BF16 output and
// sink-excluded LSE (positive infinity for an all-invalid selection).
type FlashMLADecodeQualificationOutput = flashMLADecodeOutput

const FlashMLADecodeQualificationHostLimit = flashMLADecodeHostLimit

var ErrFlashMLADecodeQualificationUnavailable = errors.New("compute: FlashMLA decode qualification profile/backend unavailable")

// ValidateFlashMLADecodeQualificationInput applies the existing owner contract.
func ValidateFlashMLADecodeQualificationInput(x FlashMLADecodeQualificationInput) error {
	return validateFlashMLADecode(x)
}
