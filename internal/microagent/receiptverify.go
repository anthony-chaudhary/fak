package microagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	// CompletionReceiptSchema versions the only child payload admitted to root.
	CompletionReceiptSchema = "fak.microagent.completion-receipt/v1"
	// MaxCompletionReceiptBytes is the hard encoded root-context cost per child.
	MaxCompletionReceiptBytes     = 4096
	maxCompletionSummaryBytes     = 1024
	maxCompletionProvenanceRefs   = 16
	maxCompletionReceiptIDBytes   = 120
	truncatedReceiptSummaryMarker = "\n[child-summary-truncated]"
)

var (
	ErrNilReceiptVerifier     = errors.New("microagent: receipt verifier is required")
	ErrReceiptRejected        = errors.New("microagent: receipt rejected")
	ErrReceiptCorrection      = errors.New("microagent: receipt correction failed")
	ErrReceiptCorrectionLimit = errors.New("microagent: receipt correction limit reached")
	ErrReceiptProvenance      = errors.New("microagent: receipt requires stable provenance")
	ErrReceiptBounds          = errors.New("microagent: receipt exceeds bounded envelope")
)

// CompletionReceipt is the bounded child completion material presented to an
// independent verifier before its summary may enter the root context.
type CompletionReceipt struct {
	Schema           string        `json:"schema"`
	Child            string        `json:"child"`
	Summary          string        `json:"summary"`
	SummaryTruncated bool          `json:"summary_truncated,omitempty"`
	Provenance       []EvidenceRef `json:"provenance"`
	Allowed          int           `json:"allowed"`
	Denied           int           `json:"denied"`
	Errored          int           `json:"errored"`
}

// Receipt returns only the bounded result of an RPC child, never its context.
func (r RPCResult) Receipt(child string) (CompletionReceipt, error) {
	child = strings.TrimSpace(child)
	if child == "" || len(child) > maxCompletionReceiptIDBytes {
		return CompletionReceipt{}, fmt.Errorf("%w: child id must be within 1..%d bytes", ErrReceiptBounds, maxCompletionReceiptIDBytes)
	}
	provenance, err := normalizeReceiptProvenance(r.Provenance)
	if err != nil {
		return CompletionReceipt{}, err
	}
	if len(provenance) != r.Allowed+r.Denied {
		return CompletionReceipt{}, fmt.Errorf("%w: got %d durable refs for %d adjudicated steps", ErrReceiptProvenance, len(provenance), r.Allowed+r.Denied)
	}
	summary, truncated := compactReceiptSummary(r.Collapsed)
	receipt := CompletionReceipt{
		Schema: CompletionReceiptSchema, Child: child, Summary: summary,
		SummaryTruncated: truncated, Provenance: provenance,
		Allowed: r.Allowed, Denied: r.Denied, Errored: r.Errored,
	}
	if _, err := receipt.Encode(); err != nil {
		return CompletionReceipt{}, err
	}
	return receipt, nil
}

// Encode returns the canonical root-context payload and enforces the receipt's
// complete size and provenance boundary.
func (r CompletionReceipt) Encode() ([]byte, error) {
	if err := r.validate(); err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if len(encoded) > MaxCompletionReceiptBytes {
		return nil, fmt.Errorf("%w: encoded bytes %d > %d", ErrReceiptBounds, len(encoded), MaxCompletionReceiptBytes)
	}
	return encoded, nil
}

func (r CompletionReceipt) validate() error {
	if r.Schema != CompletionReceiptSchema {
		return fmt.Errorf("%w: unsupported schema %q", ErrReceiptBounds, r.Schema)
	}
	if strings.TrimSpace(r.Child) == "" || len(r.Child) > maxCompletionReceiptIDBytes {
		return fmt.Errorf("%w: invalid child id", ErrReceiptBounds)
	}
	if strings.TrimSpace(r.Summary) == "" || len(r.Summary) > maxCompletionSummaryBytes {
		return fmt.Errorf("%w: invalid summary size", ErrReceiptBounds)
	}
	normalized, err := normalizeReceiptProvenance(r.Provenance)
	if err != nil {
		return err
	}
	if len(normalized) != len(r.Provenance) {
		return fmt.Errorf("%w: provenance must be canonical and unique", ErrReceiptProvenance)
	}
	for i := range normalized {
		if normalized[i] != r.Provenance[i] {
			return fmt.Errorf("%w: provenance must be sorted", ErrReceiptProvenance)
		}
	}
	if r.Allowed < 0 || r.Denied < 0 || r.Errored < 0 || len(r.Provenance) != r.Allowed+r.Denied {
		return fmt.Errorf("%w: provenance does not cover adjudicated steps", ErrReceiptProvenance)
	}
	return nil
}

func normalizeReceiptProvenance(in []EvidenceRef) ([]EvidenceRef, error) {
	if len(in) == 0 || len(in) > maxCompletionProvenanceRefs {
		return nil, fmt.Errorf("%w: reference count must be within 1..%d", ErrReceiptProvenance, maxCompletionProvenanceRefs)
	}
	set := make(map[string]EvidenceRef, len(in))
	for _, ref := range in {
		ref.Kind = strings.TrimSpace(ref.Kind)
		ref.Ref = strings.TrimSpace(ref.Ref)
		if ref.Kind != "journal-row" || !validSHA256Ref(ref.Ref) {
			return nil, fmt.Errorf("%w: expected journal-row sha256 reference", ErrReceiptProvenance)
		}
		set[ref.Kind+"\x00"+ref.Ref] = ref
	}
	keys := make([]string, 0, len(set))
	for key := range set {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]EvidenceRef, 0, len(keys))
	for _, key := range keys {
		out = append(out, set[key])
	}
	return out, nil
}

func validSHA256Ref(ref string) bool {
	const prefix = "sha256:"
	if len(ref) != len(prefix)+64 || !strings.HasPrefix(ref, prefix) {
		return false
	}
	for _, char := range ref[len(prefix):] {
		if !(char >= '0' && char <= '9' || char >= 'a' && char <= 'f') {
			return false
		}
	}
	return true
}

func compactReceiptSummary(summary string) (string, bool) {
	summary = strings.TrimSpace(summary)
	if len(summary) <= maxCompletionSummaryBytes {
		return summary, false
	}
	limit := maxCompletionSummaryBytes - len(truncatedReceiptSummaryMarker)
	for limit > 0 && !utf8.ValidString(summary[:limit]) {
		limit--
	}
	return summary[:limit] + truncatedReceiptSummaryMarker, true
}

type ReceiptDisposition uint8

const (
	ReceiptAccept ReceiptDisposition = iota + 1
	ReceiptReject
	ReceiptCorrect
)

// ReceiptReview is an independent verifier's bounded routing result.
type ReceiptReview struct {
	Disposition ReceiptDisposition
	Reason      string
}

func AcceptReceipt() ReceiptReview { return ReceiptReview{Disposition: ReceiptAccept} }

func RejectReceipt(reason string) ReceiptReview {
	return ReceiptReview{Disposition: ReceiptReject, Reason: reason}
}

func CorrectReceipt(request string) ReceiptReview {
	return ReceiptReview{Disposition: ReceiptCorrect, Reason: request}
}

type ReceiptVerifier interface {
	VerifyReceipt(context.Context, CompletionReceipt) ReceiptReview
}

type ReceiptVerifierFunc func(context.Context, CompletionReceipt) ReceiptReview

func (f ReceiptVerifierFunc) VerifyReceipt(ctx context.Context, receipt CompletionReceipt) ReceiptReview {
	return f(ctx, receipt)
}

type ReceiptCorrector interface {
	CorrectReceipt(context.Context, CompletionReceipt, string) (CompletionReceipt, error)
}

type ReceiptCorrectorFunc func(context.Context, CompletionReceipt, string) (CompletionReceipt, error)

func (f ReceiptCorrectorFunc) CorrectReceipt(ctx context.Context, receipt CompletionReceipt, request string) (CompletionReceipt, error) {
	return f(ctx, receipt, request)
}

// FoldVerifiedReceipt admits a receipt to root only after independent
// verification. A verifier may request exactly one correction and then must
// accept or reject the corrected receipt.
func FoldVerifiedReceipt(ctx context.Context, root *Context, receipt CompletionReceipt, verifier ReceiptVerifier, corrector ReceiptCorrector) error {
	if verifier == nil {
		return ErrNilReceiptVerifier
	}
	for corrections := 0; ; {
		encoded, err := receipt.Encode()
		if err != nil {
			return err
		}
		decision := verifier.VerifyReceipt(ctx, receipt)
		switch decision.Disposition {
		case ReceiptAccept:
			root.Append("tool", string(encoded))
			return nil
		case ReceiptReject:
			return fmt.Errorf("%w: %s", ErrReceiptRejected, strings.TrimSpace(decision.Reason))
		case ReceiptCorrect:
			if corrections == 1 {
				return ErrReceiptCorrectionLimit
			}
			if corrector == nil {
				return fmt.Errorf("%w: no corrector", ErrReceiptCorrection)
			}
			corrected, err := corrector.CorrectReceipt(ctx, receipt, decision.Reason)
			if err != nil {
				return fmt.Errorf("%w: %v", ErrReceiptCorrection, err)
			}
			receipt = corrected
			corrections++
		default:
			return fmt.Errorf("%w: invalid verifier disposition %d", ErrReceiptRejected, decision.Disposition)
		}
	}
}
