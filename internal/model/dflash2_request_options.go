package model

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// DFlash2TargetPanel selects target verification only. Draft math and bootstrap
// remain host/serial. Width includes the mandatory target anchor.
type DFlash2TargetPanel uint8

const (
	DFlash2TargetSerial DFlash2TargetPanel = iota
	DFlash2TargetPanel4
	DFlash2TargetPanel8
)

// Qwen38DFlash2RequestOptions is copied at construction and never mutated.
// The zero value preserves the existing host-draft, serial-verification route.
type Qwen38DFlash2RequestOptions struct {
	TargetPanel DFlash2TargetPanel
}

func (o Qwen38DFlash2RequestOptions) panelWidth() (int, error) {
	switch o.TargetPanel {
	case DFlash2TargetSerial:
		return 0, nil
	case DFlash2TargetPanel4:
		return 4, nil
	case DFlash2TargetPanel8:
		return 8, nil
	default:
		return 0, errors.New("dflash2: unknown target panel selection")
	}
}

// ValidateBackend refuses unsupported selections before artifact acquisition.
// Exact scalar arithmetic, geometry and position are checked again on the
// fresh Session and on every selected transaction. There is no serial fallback.
func (o Qwen38DFlash2RequestOptions) ValidateBackend(be compute.Backend) error {
	width, err := o.panelWidth()
	if err != nil || width == 0 {
		return err
	}
	if !backendHasComparableIdentity(be) || backendExecutionOwner(be) == nil {
		return errors.New("dflash2: panel selection requires a stable owned backend")
	}
	if _, present, err := qwen35SequencePrefillBackend(be); err != nil || !present {
		return fmt.Errorf("dflash2: selected panel sequence operation unavailable: %v", err)
	}
	all, a := be.(compute.Qwen35SequenceAllLogitsBackend)
	replay, r := be.(compute.Qwen35SequencePrefixReplayBackend)
	feature, f := be.(compute.Qwen35SequenceFeatureRowsBackend)
	if !a || all.Qwen35SequenceAllLogitsPath() != compute.Qwen35SequenceAllLogitsPath ||
		!r || replay.Qwen35SequencePrefixReplayPath() != compute.Qwen35SequencePrefixReplayPath ||
		!f || feature.Qwen35SequenceFeatureRowsPath() != compute.Qwen35SequenceFeatureRowsPath {
		return errors.New("dflash2: selected panel requires exact logits, replay and feature capabilities")
	}
	return nil
}

// NewQwen38DFlash2RequestWithOptions transfers target ownership only on success.
// Invalid/unsupported selections refuse before opening the sidecar. Zero options
// delegate directly to the original constructor.
func NewQwen38DFlash2RequestWithOptions(ctx context.Context, target *Session, sidecarPath string, options Qwen38DFlash2RequestOptions) (*Qwen38DFlash2Request, error) {
	width, err := options.panelWidth()
	if err != nil {
		return nil, err
	}
	if width == 0 {
		return NewQwen38DFlash2Request(ctx, target, sidecarPath)
	}
	if ctx == nil {
		return nil, errors.New("dflash2: request context required")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if err = dflash2FreshTarget(target); err != nil {
		return nil, err
	}
	if err = admitQwen35TargetFeaturePanel(target, qwen38DFlash2Geometry().taps, width); err != nil {
		return nil, err
	}
	q, err := NewQwen38DFlash2Request(ctx, target, sidecarPath)
	if err != nil {
		return nil, err
	}
	// No callback or fallible work occurs between adoption and publication.
	q.options = options
	return q, nil
}

func (q *Qwen38DFlash2Request) beginVerification() (*qwen35TargetFeatureTransaction, error) {
	width, err := q.options.panelWidth()
	if err != nil {
		return nil, err
	}
	if width == 0 {
		return beginQwen35TargetFeatureTransaction(q.ref.target, q.logits, q.ref.g.taps)
	}
	return beginQwen35TargetFeaturePanelTransaction(q.ref.target, q.logits, q.ref.g.taps, width)
}

func (o Qwen38DFlash2RequestOptions) blockBudget(requested int) (int, error) {
	if requested < 1 || requested > 8 {
		return 0, errors.New("dflash2: one-to-eight token budget required")
	}
	width, err := o.panelWidth()
	if err != nil {
		return 0, err
	}
	if width != 0 {
		return min(requested, width), nil
	}
	return requested, nil
}
