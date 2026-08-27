package metrics

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
)

// QuestionReceipt binds an operator answer to the governing state displayed
// with the question. Answer contents are deliberately absent from the receipt.
type QuestionReceipt struct {
	Version           int    `json:"version"`
	QuestionID        string `json:"question_id"`
	EvidenceDigest    string `json:"evidence_digest"`
	GoverningRevision string `json:"governing_revision"`
	AuthorityTenure   string `json:"authority_tenure"`
}

// QuestionState is the normalized governing input used both when displaying a
// question and immediately before applying its answer.
type QuestionState struct {
	Question          string
	RelevantEvidence  []byte
	GoverningRevision string
	AuthorityTenure   string
}

type ReceiptRefusal string

const (
	ReceiptAccepted  ReceiptRefusal = ""
	ReceiptMissing   ReceiptRefusal = "QUESTION_RECEIPT_MISSING"
	ReceiptMalformed ReceiptRefusal = "QUESTION_RECEIPT_MALFORMED"
	ReceiptStale     ReceiptRefusal = "QUESTION_RECEIPT_STALE"
)

type ReceiptCheck struct {
	Accepted bool           `json:"accepted"`
	Refusal  ReceiptRefusal `json:"refusal,omitempty"`
	Cause    string         `json:"cause,omitempty"`
}

func NewQuestionReceipt(s QuestionState) (QuestionReceipt, error) {
	q := normalizeQuestion(s.Question)
	policy := strings.TrimSpace(s.GoverningRevision)
	tenure := strings.TrimSpace(s.AuthorityTenure)
	if q == "" || policy == "" || tenure == "" {
		return QuestionReceipt{}, errors.New("question, policy revision, and authority tenure are required")
	}
	return QuestionReceipt{Version: 1, QuestionID: receiptDigest([]byte(q)), EvidenceDigest: receiptDigest(s.RelevantEvidence), GoverningRevision: policy, AuthorityTenure: tenure}, nil
}

func ParseQuestionReceipt(raw []byte) (QuestionReceipt, ReceiptCheck) {
	if len(strings.TrimSpace(string(raw))) == 0 {
		return QuestionReceipt{}, ReceiptCheck{Refusal: ReceiptMissing, Cause: "receipt omitted"}
	}
	var r QuestionReceipt
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil || r.Version != 1 || !validReceiptDigest(r.QuestionID) || !validReceiptDigest(r.EvidenceDigest) || strings.TrimSpace(r.GoverningRevision) == "" || strings.TrimSpace(r.AuthorityTenure) == "" {
		return QuestionReceipt{}, ReceiptCheck{Refusal: ReceiptMalformed, Cause: "receipt shape invalid"}
	}
	return r, ReceiptCheck{Accepted: true}
}

func VerifyQuestionReceipt(r QuestionReceipt, current QuestionState) ReceiptCheck {
	want, err := NewQuestionReceipt(current)
	if err != nil || r.Version != 1 || !validReceiptDigest(r.QuestionID) || !validReceiptDigest(r.EvidenceDigest) {
		return ReceiptCheck{Refusal: ReceiptMalformed, Cause: "receipt or current state invalid"}
	}
	switch {
	case r.QuestionID != want.QuestionID:
		return ReceiptCheck{Refusal: ReceiptStale, Cause: "question_changed"}
	case r.EvidenceDigest != want.EvidenceDigest:
		return ReceiptCheck{Refusal: ReceiptStale, Cause: "evidence_changed"}
	case r.GoverningRevision != want.GoverningRevision:
		return ReceiptCheck{Refusal: ReceiptStale, Cause: "governing_input_changed"}
	case r.AuthorityTenure != want.AuthorityTenure:
		return ReceiptCheck{Refusal: ReceiptStale, Cause: "authority_changed"}
	default:
		return ReceiptCheck{Accepted: true}
	}
}

func normalizeQuestion(s string) string { return strings.Join(strings.Fields(s), " ") }
func receiptDigest(b []byte) string     { sum := sha256.Sum256(b); return hex.EncodeToString(sum[:]) }
func validReceiptDigest(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size
}
