// Package ctxmmu is the IMPORTABLE vendor surface of fak's context-MMU
// zero-copy observation admission seam.
//
// fak's context-MMU lives in internal/ctxmmu: the write-time gate on tool
// RESULTS, which decides at the moment a result would enter the conversation
// whether its bytes may be admitted as-is, must be quarantined, or paged out.
// The zero-copy admission path couples that gate to the kernel-owned
// RadixAttention prefix tree (internal/radixkv): an observation whose bytes
// already live in host/unified memory and start on a page boundary is folded to
// one token per page and bound into the tree with no copy.
//
// Go's internal/ rule seals internal/ctxmmu to this module, which is correct for
// the kernel's own consumers but blocks the ONE audience the seam exists to
// serve: an OUT-OF-TREE (private platform) caller that wants to construct the
// gate and pager and admit observations. This package is that surface, and ONLY
// that surface.
//
// Every name below is a Go TYPE ALIAS or a var re-export of a symbol in
// internal/ctxmmu (and internal/radixpager for the pager, which lives in its own
// package to avoid the radixkv -> ctxmmu import cycle). A value from
// pkg/ctxmmu is IDENTICAL (same underlying type) to its internal counterpart.
// Private platform code imports THIS package
// (github.com/anthony-chaudhary/fak/pkg/ctxmmu) and never internal/ctxmmu,
// satisfying the core import invariant: private code imports ONLY fak/pkg/*.
// This shim adds no behavior and holds no state; it is a stable, zero-cost
// re-export for Gate 3 of the architecture boundary.
package ctxmmu

import (
	internalctxmmu "github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/radixpager"
)

type (
	// ToolObservation is a raw tool observation whose Bytes live in host/unified memory.
	// It is named ToolObservation (not Observation) to avoid colliding with the
	// package's pre-existing #1598 disposition-minting Observation.
	ToolObservation = internalctxmmu.ToolObservation
	// AdmissionReceipt describes what the zero-copy admission path did.
	AdmissionReceipt = internalctxmmu.AdmissionReceipt
	// ZeroCopyAdmission is the write-time admission gate for tool observations.
	ZeroCopyAdmission = internalctxmmu.ZeroCopyAdmission
	// ObservationPager is the narrow paging seam the gate binds through.
	ObservationPager = internalctxmmu.ObservationPager
	// RadixKVPager pages pre-tokenized blocks into the Radix KV tree.
	RadixKVPager = radixpager.RadixKVPager
)

var (
	// NewZeroCopyAdmission builds an admission gate over an MMU and a pager.
	NewZeroCopyAdmission = internalctxmmu.NewZeroCopyAdmission
	// NewRadixKVPager builds a Radix KV pager (over an existing tree or a new one).
	NewRadixKVPager = radixpager.NewRadixKVPager
)

const (
	// FallbackReasonEmptyPayload is set when the observation carried no bytes.
	FallbackReasonEmptyPayload = internalctxmmu.FallbackReasonEmptyPayload
	// FallbackReasonNonPageAligned is set when the payload was not page aligned.
	FallbackReasonNonPageAligned = internalctxmmu.FallbackReasonNonPageAligned
	// FallbackReasonUnsupportedEncoding is set for a refused MIME encoding.
	FallbackReasonUnsupportedEncoding = internalctxmmu.FallbackReasonUnsupportedEncoding
)
