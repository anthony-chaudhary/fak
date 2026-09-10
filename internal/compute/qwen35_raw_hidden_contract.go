package compute

// Qwen35SequenceRawHiddenPath identifies the optional sequence extension which
// exposes the final residual rows before output normalization. The returned
// tensor stays device-resident and follows the request-resource lifetime.
const Qwen35SequenceRawHiddenPath = "qwen35-hybrid-sequence-raw-hidden-v1"

// Qwen35SequenceRawHiddenBackend prevents callers from requesting pre-output-
// norm residual rows from sequence implementations without the matching
// device-resident result and lifetime contract.
type Qwen35SequenceRawHiddenBackend interface {
	Qwen35SequenceRawHiddenPath() string
}
