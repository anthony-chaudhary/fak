package microagent

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// InputBasisVersion is the canonical schema version for an input snapshot basis.
const (
	InputBasisVersion   = "fak.microagent.input-basis/v1"
	InputBasisVersionV1 = "v1"

	maxInputBasisEntries     = 128
	maxInputBasisKeyBytes    = 256
	maxInputBasisDigestBytes = 128
)

var (
	// ErrMissingReceiptBasis is returned when the input basis is missing or empty.
	ErrMissingReceiptBasis = errors.New("microagent: missing or empty receipt input basis")
	// ErrReceiptBasisMismatch is returned when an input basis dependency is missing or mismatched in the snapshot.
	ErrReceiptBasisMismatch = errors.New("microagent: receipt input basis mismatch with snapshot")
	// ErrInvalidReceiptBasis is returned when the input basis is malformed or exceeds bounds.
	ErrInvalidReceiptBasis = errors.New("microagent: invalid receipt input basis")
)

// InputBasis represents the versioned set of input dependencies (key -> digest)
// that a child agent execution was grounded on.
type InputBasis struct {
	Version string            `json:"version"`
	Inputs  map[string]string `json:"inputs"`
}

// NewInputBasis constructs a valid InputBasis with the canonical version tag.
func NewInputBasis(inputs map[string]string) InputBasis {
	m := make(map[string]string, len(inputs))
	for k, v := range inputs {
		m[k] = v
	}
	return InputBasis{
		Version: InputBasisVersion,
		Inputs:  m,
	}
}

// ReceiptBasis pairs a CompletionReceipt with its required InputBasis.
type ReceiptBasis struct {
	Receipt CompletionReceipt `json:"receipt"`
	Basis   InputBasis        `json:"basis"`
}

// WithBasis binds an InputBasis to a CompletionReceipt.
func (r CompletionReceipt) WithBasis(basis InputBasis) ReceiptBasis {
	return ReceiptBasis{
		Receipt: r,
		Basis:   basis,
	}
}

// Snapshot is an authoritative immutable mapping of input keys to content digests.
type Snapshot map[string]string

// SnapshotResolver resolves authoritative content digests for input keys.
type SnapshotResolver interface {
	ResolveSnapshotDigest(key string) (digest string, ok bool)
}

// SnapshotResolverFunc adapts a function to SnapshotResolver.
type SnapshotResolverFunc func(key string) (string, bool)

// ResolveSnapshotDigest implements SnapshotResolver.
func (f SnapshotResolverFunc) ResolveSnapshotDigest(key string) (string, bool) {
	return f(key)
}

// ResolveSnapshotDigest implements SnapshotResolver for Snapshot.
func (s Snapshot) ResolveSnapshotDigest(key string) (string, bool) {
	if s == nil {
		return "", false
	}
	v, ok := s[key]
	return v, ok
}

// ValidateSnapshotBasis validates that basis is well-formed and that every input
// declared in basis matches the authoritative snapshot resolver.
func ValidateSnapshotBasis(basis InputBasis, resolver SnapshotResolver) error {
	if strings.TrimSpace(basis.Version) == "" {
		return fmt.Errorf("%w: version required", ErrMissingReceiptBasis)
	}
	if basis.Version != InputBasisVersion && basis.Version != InputBasisVersionV1 {
		return fmt.Errorf("%w: unsupported version %q", ErrInvalidReceiptBasis, basis.Version)
	}
	if len(basis.Inputs) == 0 {
		return fmt.Errorf("%w: inputs map is empty", ErrMissingReceiptBasis)
	}
	if len(basis.Inputs) > maxInputBasisEntries {
		return fmt.Errorf("%w: inputs count %d > %d", ErrInvalidReceiptBasis, len(basis.Inputs), maxInputBasisEntries)
	}

	keys := make([]string, 0, len(basis.Inputs))
	for k := range basis.Inputs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		v := basis.Inputs[k]
		if strings.TrimSpace(k) == "" {
			return fmt.Errorf("%w: input key cannot be empty", ErrInvalidReceiptBasis)
		}
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("%w: digest for input %q cannot be empty", ErrInvalidReceiptBasis, k)
		}
		if len(k) > maxInputBasisKeyBytes || len(v) > maxInputBasisDigestBytes {
			return fmt.Errorf("%w: input %q key or digest exceeds bounded size", ErrInvalidReceiptBasis, k)
		}
		if resolver == nil {
			return fmt.Errorf("%w: input %q missing in authoritative snapshot (nil snapshot)", ErrReceiptBasisMismatch, k)
		}
		gotDigest, ok := resolver.ResolveSnapshotDigest(k)
		if !ok {
			return fmt.Errorf("%w: input %q missing in authoritative snapshot", ErrReceiptBasisMismatch, k)
		}
		if gotDigest != v {
			return fmt.Errorf("%w: input %q digest mismatch (basis=%q, snapshot=%q)", ErrReceiptBasisMismatch, k, v, gotDigest)
		}
	}
	return nil
}

// ValidateBasis checks this InputBasis against an authoritative SnapshotResolver.
func (b InputBasis) ValidateBasis(resolver SnapshotResolver) error {
	return ValidateSnapshotBasis(b, resolver)
}

// ValidateBasis checks this Snapshot against an InputBasis.
func (s Snapshot) ValidateBasis(basis InputBasis) error {
	return ValidateSnapshotBasis(basis, s)
}

// FoldVerifiedReceiptWithSnapshot validates the input snapshot basis before admitting
// child completion receipts to the root context. If basis is empty, missing, or invalid,
// or if any input digest in basis does not match the authoritative snapshot (or is missing),
// it refuses with a typed error and leaves root unchanged.
func FoldVerifiedReceiptWithSnapshot(
	ctx context.Context,
	root *Context,
	receipt CompletionReceipt,
	basis InputBasis,
	snapshot map[string]string,
	verifier ReceiptVerifier,
	corrector ReceiptCorrector,
) error {
	return FoldVerifiedReceiptWithResolver(ctx, root, receipt, basis, Snapshot(snapshot), verifier, corrector)
}

// FoldVerifiedReceiptWithResolver folds a verified receipt after validating its input basis
// against a general SnapshotResolver.
func FoldVerifiedReceiptWithResolver(
	ctx context.Context,
	root *Context,
	receipt CompletionReceipt,
	basis InputBasis,
	resolver SnapshotResolver,
	verifier ReceiptVerifier,
	corrector ReceiptCorrector,
) error {
	if err := ValidateSnapshotBasis(basis, resolver); err != nil {
		return err
	}
	return FoldVerifiedReceipt(ctx, root, receipt, verifier, corrector)
}

// FoldVerifiedReceiptWithBasis folds a ReceiptBasis against an authoritative snapshot.
func FoldVerifiedReceiptWithBasis(
	ctx context.Context,
	root *Context,
	rb ReceiptBasis,
	snapshot map[string]string,
	verifier ReceiptVerifier,
	corrector ReceiptCorrector,
) error {
	return FoldVerifiedReceiptWithSnapshot(ctx, root, rb.Receipt, rb.Basis, snapshot, verifier, corrector)
}
