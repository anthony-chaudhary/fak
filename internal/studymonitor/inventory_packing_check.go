package studymonitor

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// PackingPathFacts names what a declared loader packing path produces for one
// kernel family's tensors at load time, on the same four layout axes as the
// kernel_inventory record it is checked against. Every field is optional: an
// empty string means the packing path does not declare that axis, which the
// check treats as unverifiable whenever the record obligates it.
type PackingPathFacts struct {
	ExpectedInputLayout string
	SwizzlePattern      string
	ScaleLayout         string
	PackingSource       string
}

var (
	// ErrLayoutMismatch is the closed refusal for a kernel_inventory record
	// whose filled layout obligation differs from the value the declared
	// loader packing path produces on the same axis. The witnessed class is
	// LML-07: ssm_out value-head-tiled GGUF columns fed through the Q8_0
	// deinterleave packing path.
	ErrLayoutMismatch = errors.New("studymonitor: layout obligation mismatch")
	// ErrLayoutUnknown is the closed refusal for a filled layout obligation
	// that cannot be verified: the record value falls outside the closed
	// vocabulary, or the packing path leaves the obligated axis undeclared.
	ErrLayoutUnknown = errors.New("studymonitor: layout obligation unverifiable")
)

// CheckLoaderPackingAgainstInventory compares the layout obligations a
// kernel_inventory record declares against a declared loader packing path and
// refuses typed instead of accepting:
//
//   - nil: every filled obligation matches the packing facts;
//   - ErrLayoutMismatch: both sides declare a value on an obligated axis and
//     they differ;
//   - ErrLayoutUnknown: a filled obligation is outside the closed vocabulary,
//     or the packing path does not declare that axis.
//
// Whitespace-only fields count as unfilled on both sides. When several axes
// fail, the first in field declaration order (expected_input_layout,
// swizzle_pattern, scale_layout, packing_source) is reported. Callers pin the
// refusal class with errors.Is; the wrapped message names the axis and both
// values for humans and must not be asserted on.
func CheckLoaderPackingAgainstInventory(rec InventorySourceEvidence, packing PackingPathFacts) error {
	axes := []struct {
		field string
		rec   string
		fact  string
		vocab []string
	}{
		{"expected_input_layout", strings.TrimSpace(rec.ExpectedInputLayout), strings.TrimSpace(packing.ExpectedInputLayout), KnownExpectedInputLayouts},
		{"swizzle_pattern", strings.TrimSpace(rec.SwizzlePattern), strings.TrimSpace(packing.SwizzlePattern), KnownSwizzlePatterns},
		{"scale_layout", strings.TrimSpace(rec.ScaleLayout), strings.TrimSpace(packing.ScaleLayout), KnownScaleLayouts},
		{"packing_source", strings.TrimSpace(rec.PackingSource), strings.TrimSpace(packing.PackingSource), KnownPackingSources},
	}
	for _, axis := range axes {
		if axis.rec == "" {
			continue
		}
		if !slices.Contains(axis.vocab, axis.rec) {
			return fmt.Errorf("%w: %s: record value %q is outside the closed vocabulary", ErrLayoutUnknown, axis.field, axis.rec)
		}
		if axis.fact == "" {
			return fmt.Errorf("%w: %s: packing path does not declare the obligated value %q", ErrLayoutUnknown, axis.field, axis.rec)
		}
		if axis.fact != axis.rec {
			return fmt.Errorf("%w: %s: record %q vs packing %q", ErrLayoutMismatch, axis.field, axis.rec, axis.fact)
		}
	}
	return nil
}
