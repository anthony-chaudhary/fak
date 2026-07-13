// Package refutil holds foundation-level helpers for materializing ABI refs.
package refutil

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Bytes returns inline bytes directly or materializes an external ref through
// the active resolver. Missing resolvers and resolution failures fail closed to nil.
func Bytes(ctx context.Context, ref abi.Ref) []byte {
	if ref.Kind == abi.RefInline {
		return ref.Inline
	}
	if resolver := abi.ActiveResolver(); resolver != nil {
		if body, err := resolver.Resolve(ctx, ref); err == nil {
			return body
		}
	}
	return nil
}
