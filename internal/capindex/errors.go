package capindex

import "errors"

var (
	ErrNotFound        = errors.New("capability not found")
	ErrKindMismatch    = errors.New("capability kind mismatch")
	ErrVersionNotFound = errors.New("capability version not found")
)
