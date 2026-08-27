package snapshot

import "github.com/anthony-chaudhary/fak/internal/metrics"

type QuestionReceipt = metrics.QuestionReceipt
type QuestionState = metrics.QuestionState
type ReceiptRefusal = metrics.ReceiptRefusal
type ReceiptCheck = metrics.ReceiptCheck

const (
	ReceiptAccepted  = metrics.ReceiptAccepted
	ReceiptMissing   = metrics.ReceiptMissing
	ReceiptMalformed = metrics.ReceiptMalformed
	ReceiptStale     = metrics.ReceiptStale
)

var NewQuestionReceipt = metrics.NewQuestionReceipt
var ParseQuestionReceipt = metrics.ParseQuestionReceipt
var VerifyQuestionReceipt = metrics.VerifyQuestionReceipt
