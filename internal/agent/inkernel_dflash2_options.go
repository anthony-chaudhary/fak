package agent

import (
	"context"
	"errors"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// DFlash2DraftMode identifies the requested draft implementation.
type DFlash2DraftMode uint8

const (
	DFlash2HostDraft DFlash2DraftMode = iota
	DFlash2ResidentMLP
)

// ErrDFlash2ResidentAdmissionUnavailable is returned before acquiring an owner
// or opening any artifact. The resident primitive's complete physical target,
// snapshot/COW, panel, staging and lazy-copy peak recipe is not yet admitted.
var ErrDFlash2ResidentAdmissionUnavailable = errors.New("dflash2: resident MLP admission unavailable; complete derived physical memory recipe required")

// DFlash2TargetOptions is an immutable startup selection. Zero preserves host
// drafting and serial target verification. No environment variable enables it.
type DFlash2TargetOptions struct {
	Draft   DFlash2DraftMode
	Request model.Qwen38DFlash2RequestOptions
}

func (o DFlash2TargetOptions) validate(backend compute.Backend) error {
	switch o.Draft {
	case DFlash2HostDraft:
	case DFlash2ResidentMLP:
		return ErrDFlash2ResidentAdmissionUnavailable
	default:
		return errors.New("dflash2: unknown draft selection")
	}
	return o.Request.ValidateBackend(backend)
}

// NewDFlash2TargetWithOptions authenticates the same exact target as the default
// constructor. Selected panel absence/failure refuses without serial fallback.
// Resident selection is deliberately unavailable until its physical peak recipe
// is independently supported; this constructor does not allocate resident MLPs.
func NewDFlash2TargetWithOptions(ctx context.Context, spec DFlash2TargetSpec, backend compute.Backend, options DFlash2TargetOptions) (*DFlash2Target, error) {
	return newDFlash2TargetWithOptions(ctx, spec, backend, options, ggufload.OpenWeightsAuthenticated)
}
