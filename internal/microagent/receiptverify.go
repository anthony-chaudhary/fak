package microagent

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

var (
	ErrNilReceiptVerifier     = errors.New("microagent: receipt verifier is required")
	ErrReceiptRejected        = errors.New("microagent: receipt rejected")
	ErrReceiptCorrection      = errors.New("microagent: receipt correction failed")
	ErrReceiptCorrectionLimit = errors.New("microagent: receipt correction limit reached")
)

// CompletionReceipt is the bounded child completion material presented to an
// independent verifier before its summary may enter the root context.
type CompletionReceipt struct {
	Child   string
	Summary string
	Allowed int
	Denied  int
	Errored int
}

// Receipt returns only the bounded result of an RPC child, never its context.
func (r RPCResult) Receipt(child string) CompletionReceipt {
	return CompletionReceipt{
		Child: child, Summary: r.Collapsed, Allowed: r.Allowed,
		Denied: r.Denied, Errored: r.Errored,
	}
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
		decision := verifier.VerifyReceipt(ctx, receipt)
		switch decision.Disposition {
		case ReceiptAccept:
			root.Append("tool", receipt.Summary)
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
