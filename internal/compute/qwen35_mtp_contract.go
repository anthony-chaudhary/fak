package compute

import "fmt"

// Qwen35MTPDraftPath identifies the complete resident pre-layer fusion used by
// the retained one-layer Qwen3.8 draft head. A backend advertising this path
// must keep both RMSNorm results, their concatenation, and mtp.fc on-device.
const Qwen35MTPDraftPath = "qwen35-mtp-resident-draft-v1"

// Qwen35MTPFuseRequest contains the resident tensors for the checkpoint-defined
// MTP pre-layer operation. PriorHidden and CurrentEmbedding are the two explicit
// host/API boundary inputs after upload; every other tensor is immutable weight
// residency owned by the draft session.
type Qwen35MTPFuseRequest struct {
	PriorHidden      Tensor
	CurrentEmbedding Tensor
	HiddenNorm       Tensor
	EmbeddingNorm    Tensor
	FC               Tensor
	Epsilon          float32
}

// Qwen35MTPDraftBackend is the optional backend operation that prevents the
// 2H concatenation between the two pre-FC normalizations from returning to the
// host. The remaining retained decoder uses the ordinary resident HAL ops.
type Qwen35MTPDraftBackend interface {
	Qwen35MTPDraftPath() string
	Qwen35MTPFuse(Qwen35MTPFuseRequest) (Tensor, error)
}

// Qwen35MTPTransferCounter exposes cumulative physical host/device payload
// bytes. Callers take snapshots around setup, boundary, and resident phases;
// implementations must not synthesize counts from tensor shapes.
type Qwen35MTPTransferCounter interface {
	Qwen35MTPTransferBytes() (h2d, d2h uint64)
}

// Qwen35MTPQ6KBackend admits the packed Q6_K projections used by the exact
// Qwen3.8 Q4_K_M retained layer and shared head. The capability means MatMul
// consumes the packed resident tensor without expanding it on the host.
type Qwen35MTPQ6KBackend interface {
	SupportsQ6KMatMul() bool
}

// UnsupportedQwen35MTPDraftError is the fail-closed capability or tensor
// refusal for the resident MTP route. Callers must not retry the same position
// through the CPU draft head after receiving it.
type UnsupportedQwen35MTPDraftError struct {
	Backend string
	Path    string
	Stage   string
	Reason  string
}

func (e *UnsupportedQwen35MTPDraftError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.8 resident MTP refusal"
	}
	return fmt.Sprintf("compute: backend %q cannot execute Qwen3.8 MTP via %q at %s: %s; refusing CPU fallback", e.Backend, e.Path, e.Stage, e.Reason)
}
