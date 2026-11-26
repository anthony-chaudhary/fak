//go:build !(darwin && arm64 && cgo)

package ggufload

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// inkling2DFlash2Store is absent off Apple GPUs: every draft matrix uploads.
type inkling2DFlash2Store struct{}

func openInkling2DFlash2Store([]byte) (*inkling2DFlash2Store, error) {
	return nil, errors.ErrUnsupported
}

func (*inkling2DFlash2Store) view([]byte, int, int) (compute.Tensor, bool) {
	return compute.Tensor{}, false
}

func (*inkling2DFlash2Store) Close() {}
